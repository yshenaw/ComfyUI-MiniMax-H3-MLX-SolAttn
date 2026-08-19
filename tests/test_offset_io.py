from __future__ import annotations

import json
import struct

import mlx.core as mx
import pytest

from minimax_h3_mlx.offset_io import raw_parts_to_mlx, read_safetensors_parts


def test_offset_loader_round_trips_streaming_dtypes_exactly(tmp_path):
    expected = {
        "bf16": mx.array([1.0, -2.5, 0.125], dtype=mx.bfloat16),
        "f16": mx.array([[1.5, -4.0]], dtype=mx.float16),
        "f32": mx.array([3.25], dtype=mx.float32),
        "i32": mx.array([-7, 9], dtype=mx.int32),
        "u32": mx.array([0, 0x12345678, 0xFFFFFFFF], dtype=mx.uint32),
    }
    path = tmp_path / "part.safetensors"
    mx.save_safetensors(str(path), expected, metadata={"format": "mlx"})

    parts = read_safetensors_parts(tmp_path, [path.name])
    actual = raw_parts_to_mlx(parts)

    assert parts.bytes_read == path.stat().st_size
    assert parts.nocache_files in (0, 1)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        mx.eval(actual[name], value)
        if value.dtype == mx.bfloat16:
            assert mx.array_equal(actual[name].view(mx.uint16), value.view(mx.uint16)).item()
        else:
            assert mx.array_equal(actual[name], value).item()


def test_offset_loader_handles_scalar_and_empty_tensors(tmp_path):
    header = json.dumps(
        {
            "scalar": {"dtype": "I32", "shape": [], "data_offsets": [0, 4]},
            "empty": {"dtype": "F16", "shape": [0, 3], "data_offsets": [4, 4]},
        }
    ).encode()
    path = tmp_path / "part.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<i", 7))

    actual = raw_parts_to_mlx(read_safetensors_parts(tmp_path, [path.name]))

    assert actual["empty"].shape == (0, 3)
    assert actual["scalar"].shape == ()
    assert actual["scalar"].item() == 7


def test_offset_loader_rejects_duplicate_tensor_names(tmp_path):
    mx.save_safetensors(str(tmp_path / "first.safetensors"), {"same": mx.ones((1,))})
    mx.save_safetensors(str(tmp_path / "second.safetensors"), {"same": mx.zeros((1,))})

    with pytest.raises(KeyError, match="Duplicate tensors"):
        read_safetensors_parts(tmp_path, ["first.safetensors", "second.safetensors"])


def test_offset_loader_rejects_unknown_dtype(tmp_path):
    header = json.dumps(
        {"bad": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
    ).encode()
    path = tmp_path / "unknown.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")

    with pytest.raises(TypeError, match="Unsupported safetensors dtype"):
        read_safetensors_parts(tmp_path, [path.name])