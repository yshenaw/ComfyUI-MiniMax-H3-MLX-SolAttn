"""Direct, uncached safetensors reads for experimental block streaming."""

from __future__ import annotations

import fcntl
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


_NUMPY_DTYPES = {
    "BOOL": np.dtype("?"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I16": np.dtype("<i2"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "U64": np.dtype("<u8"),
    "U8": np.dtype("u1"),
}


@dataclass(frozen=True)
class RawTensor:
    dtype: str
    values: np.ndarray


@dataclass(frozen=True)
class RawParts:
    tensors: dict[str, RawTensor]
    bytes_read: int
    read_seconds: float
    nocache_files: int


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    parts = []
    remaining = int(size)
    while remaining:
        part = os.pread(fd, remaining, offset)
        if not part:
            raise EOFError(f"Short pread at offset {offset}: {remaining} bytes missing.")
        parts.append(part)
        offset += len(part)
        remaining -= len(part)
    return b"".join(parts)


def _read_file(path: Path, *, nocache: bool) -> tuple[dict[str, RawTensor], int, bool]:
    fd = os.open(path, os.O_RDONLY)
    uncached = False
    try:
        if nocache and hasattr(fcntl, "F_NOCACHE"):
            try:
                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                uncached = True
            except OSError:
                pass
        file_size = os.fstat(fd).st_size
        header_size = struct.unpack("<Q", _pread_exact(fd, 8, 0))[0]
        header_end = 8 + header_size
        if header_end > file_size:
            raise ValueError(f"Invalid safetensors header in {path}.")
        header = json.loads(_pread_exact(fd, header_size, 8))
        payload = _pread_exact(fd, file_size - header_end, header_end)
    finally:
        os.close(fd)

    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype_name = str(spec["dtype"])
        storage_dtype = np.dtype("<u2") if dtype_name == "BF16" else _NUMPY_DTYPES.get(dtype_name)
        if storage_dtype is None:
            raise TypeError(f"Unsupported safetensors dtype {dtype_name!r} in {path}.")
        start, end = (int(value) for value in spec["data_offsets"])
        shape = tuple(int(value) for value in spec["shape"])
        if not 0 <= start <= end <= len(payload):
            raise ValueError(f"Invalid tensor offsets for {name!r} in {path}.")
        expected = int(np.prod(shape, dtype=np.int64))
        values = np.frombuffer(payload, dtype=storage_dtype, count=expected, offset=start)
        if values.size != expected or values.nbytes != end - start:
            raise ValueError(
                f"Tensor size mismatch for {name!r} in {path}: "
                f"shape={shape}, bytes={end - start}."
            )
        tensors[name] = RawTensor(dtype_name, values.reshape(shape))
    return tensors, file_size, uncached


def read_safetensors_parts(
    model_dir: str | Path,
    names: list[str],
    *,
    nocache: bool = True,
) -> RawParts:
    """Read complete part files with ``pread`` without populating the macOS file cache."""
    started = time.perf_counter()
    tensors: dict[str, RawTensor] = {}
    bytes_read = 0
    nocache_files = 0
    model_dir = Path(model_dir)
    for name in names:
        loaded, file_bytes, uncached = _read_file(model_dir / name, nocache=nocache)
        duplicate = tensors.keys() & loaded.keys()
        if duplicate:
            raise KeyError(f"Duplicate tensors across streaming parts: {sorted(duplicate)[:4]}.")
        tensors.update(loaded)
        bytes_read += file_bytes
        nocache_files += int(uncached)
    return RawParts(tensors, bytes_read, time.perf_counter() - started, nocache_files)


def raw_parts_to_mlx(parts: RawParts) -> dict[str, mx.array]:
    """Copy raw part buffers into materialized MLX arrays with exact stored dtypes."""
    tensors = {}
    for name, tensor in parts.tensors.items():
        value = mx.array(tensor.values)
        if tensor.dtype == "BF16":
            value = value.view(mx.bfloat16)
        tensors[name] = value
    mx.eval(list(tensors.values()))
    return tensors


__all__ = ["RawParts", "RawTensor", "raw_parts_to_mlx", "read_safetensors_parts"]