# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed storage for PLE embedding tables.

Without this, ``PleOffloadRunner`` holds every PLE table in anonymous host
memory for the lifetime of the server. On a Qwen3.8-Flash-Next NVFP4 checkpoint
that is ~49 GiB which the kernel can never reclaim, on top of whatever sleep
mode pins when the model is parked.

The tables are read-only lookup data, so they do not need to be anonymous. This
module writes each one to a cache file once and re-points the parameter at a
shared file mapping. The pages then live in the page cache: they still serve
lookups at memory speed while hot, they count as reclaimable rather than used,
and the kernel can evict them when something else needs the DRAM. Dropping them
explicitly after an idle period (``VLLM_PLE_TABLE_CACHE_TTL``) makes that
eviction prompt rather than waiting for pressure.

Building the cache is a one-time cost. On every later start the tables are
mapped straight from the cache and the checkpoint is not read at all.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

MANIFEST = "manifest.json"

#: Tables below this size are left in anonymous memory. Mapping a small tensor
#: costs a file, a VMA and a page-fault path for no measurable saving.
MIN_MMAP_BYTES = 64 * 1024 * 1024

#: ``MADV_DONTNEED`` from ``asm-generic/mman-common.h``. On a MAP_SHARED file
#: mapping it tears down the page-table entries without touching the file, so
#: the next access re-faults from disk.
MADV_DONTNEED = 4

_PAGE_SIZE = os.sysconf("SC_PAGESIZE")
_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.madvise.restype = ctypes.c_int


def _madvise_dontneed(addr: int, length: int) -> bool:
    """Drop a mapping's resident pages. Both bounds must be page-aligned."""
    start = (addr + _PAGE_SIZE - 1) // _PAGE_SIZE * _PAGE_SIZE
    end = (addr + length) // _PAGE_SIZE * _PAGE_SIZE
    if end <= start:
        return True
    if _libc.madvise(ctypes.c_void_p(start), end - start, MADV_DONTNEED) != 0:
        logger.warning(
            "PLE table cache: madvise(MADV_DONTNEED) failed: %s",
            os.strerror(ctypes.get_errno()),
        )
        return False
    return True


def _entry_name(layer_name: str, param_name: str) -> str:
    full = f"{layer_name}.{param_name}"
    # Parameter names contain dots and are long; hash for a flat, bounded
    # filename and keep the readable name in the manifest.
    return hashlib.sha256(full.encode()).hexdigest()[:32] + ".bin"


def _describe(layers: dict[str, nn.Module]) -> dict[str, dict]:
    """Describe every table worth caching: full name -> spec.

    The spec records where the parameter lives (``layer``/``param``) as well as
    its shape and dtype, so nothing has to re-derive the split between the two
    later. Layer names can prefix one another, which makes splitting a joined
    name ambiguous.
    """
    described: dict[str, dict] = {}
    for layer_name, layer in layers.items():
        for param_name, param in layer.named_parameters(recurse=True):
            tensor = param.data
            if tensor.device.type != "cpu" or tensor.is_meta:
                continue
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes < MIN_MMAP_BYTES:
                continue
            described[f"{layer_name}.{param_name}"] = {
                "layer": layer_name,
                "param": param_name,
                "file": _entry_name(layer_name, param_name),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": nbytes,
            }
    return described


def _resolve(layers: dict[str, nn.Module], spec: dict) -> tuple[nn.Module, str]:
    """Return the module owning the described parameter and its attribute name."""
    layer = layers[spec["layer"]]
    module_path, _, leaf = spec["param"].rpartition(".")
    module = layer.get_submodule(module_path) if module_path else layer
    return module, leaf


class PleTableCache:
    """One cache directory of mmap-backed PLE tables."""

    def __init__(self, root: str, key: str, ttl: float = 0.0):
        self.dir = os.path.join(root, hashlib.sha256(key.encode()).hexdigest()[:16])
        self.ttl = ttl
        # (path, mapping address, mapping length) per cached table. The address
        # is needed because dropping a *mapped* table takes madvise on the
        # mapping; posix_fadvise alone leaves pages that are mapped into this
        # process untouched.
        self._mappings: list[tuple[str, int, int]] = []
        self._bytes = 0

    # -- lookup ------------------------------------------------------------

    def _manifest_path(self) -> str:
        return os.path.join(self.dir, MANIFEST)

    def _read_manifest(self) -> dict | None:
        try:
            with open(self._manifest_path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def matches(self, layers: dict[str, nn.Module]) -> bool:
        """Whether a complete cache for exactly these tables already exists."""
        manifest = self._read_manifest()
        if manifest is None:
            return False
        wanted = _describe(layers)
        have = manifest.get("entries", {})
        if set(have) != set(wanted):
            return False
        for name, spec in wanted.items():
            if have[name] != spec:
                return False
            path = os.path.join(self.dir, spec["file"])
            try:
                if os.path.getsize(path) != spec["bytes"]:
                    return False
            except OSError:
                return False
        return True

    # -- attach ------------------------------------------------------------

    def attach(self, layers: dict[str, nn.Module]) -> None:
        """Re-point every cached parameter at its file mapping.

        The parameter's previous storage is dropped. On the build path that
        storage held the loaded weights and is freed here; on the fast path it
        was never faulted in, because the tensors were allocated and then
        replaced without ever being written.
        """
        total = 0
        for spec in _describe(layers).values():
            module, leaf = _resolve(layers, spec)
            param = getattr(module, leaf)
            path = os.path.join(self.dir, spec["file"])
            nbytes = param.data.numel() * param.data.element_size()
            # Mapping a byte view keeps this dtype-agnostic: fp8 and fp4 code
            # tensors have no numpy equivalent to route through.
            backing = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
            # Record the mapping base before any view: views may be offset, and
            # madvise needs the address the file was mapped at.
            self._mappings.append((path, backing.data_ptr(), nbytes))
            param.data = backing.view(param.data.dtype).view(param.data.shape)
            total += nbytes
        self._bytes = total
        logger.info(
            "PLE table cache: mapped %.2f GiB across %d table(s) from %s",
            total / 1024**3,
            len(self._mappings),
            self.dir,
        )

    def build(self, layers: dict[str, nn.Module]) -> None:
        """Write the loaded tables to the cache, then map them back."""
        os.makedirs(self.dir, exist_ok=True)
        entries = _describe(layers)
        for full_name, spec in entries.items():
            module, leaf = _resolve(layers, spec)
            tensor = getattr(module, leaf).data.contiguous()
            path = os.path.join(self.dir, spec["file"])
            tmp = path + ".partial"
            flat = tensor.view(torch.uint8).reshape(-1)
            with open(tmp, "wb") as f:
                f.write(memoryview(flat.numpy()))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            logger.info(
                "PLE table cache: wrote %s (%.2f GiB)",
                full_name,
                spec["bytes"] / 1024**3,
            )
        tmp = self._manifest_path() + ".partial"
        with open(tmp, "w") as f:
            json.dump({"entries": entries}, f, indent=2)
        os.replace(tmp, self._manifest_path())
        self.attach(layers)

    # -- reclaim -----------------------------------------------------------

    def drop(self) -> int:
        """Release the mapped tables from memory.

        Two steps, and both are needed. ``madvise(MADV_DONTNEED)`` tears down
        this process's page-table entries, which is what actually returns the
        RSS; ``posix_fadvise(POSIX_FADV_DONTNEED)`` then evicts the now-unmapped
        pages from the page cache. Skipping the madvise leaves the pages mapped
        and the fadvise does nothing at all.

        Returns the number of bytes covered. The pages are clean and
        file-backed, so the next lookup simply faults them back in from NVMe.
        """
        for path, addr, length in self._mappings:
            _madvise_dontneed(addr, length)
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)
        return self._bytes

    @property
    def mapped_bytes(self) -> int:
        return self._bytes

    @property
    def enabled(self) -> bool:
        return bool(self._mappings)
