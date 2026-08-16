import json

import pytest

from minimax_h3_mlx.streaming_layout import (
    MANIFEST_NAME,
    is_streaming_checkpoint,
    load_manifest,
    tensor_group,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("video_patch_proj.weight", ("fixed", None)),
        ("token_refiner.blocks.0.mlp.fc1.weight", ("fixed", None)),
        ("blocks.0.attn.qkv_proj.weight", ("core", 0)),
        ("blocks.4.norm2.weight", ("core", 0)),
        ("blocks.5.mlp.fc2.scales", ("core", 5)),
        ("blocks.49.adaln_proj.linear.weight", ("adaln", 45)),
    ],
)
def test_tensor_group_keeps_adaln_out_of_core_chunks(key, expected):
    assert tensor_group(key, chunk_size=5) == expected


def test_manifest_validation(tmp_path):
    assert not is_streaming_checkpoint(tmp_path)
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps({"format": "minimax-h3-mlx-block-stream-v1"}))
    assert is_streaming_checkpoint(tmp_path)
    assert load_manifest(tmp_path)["format"].endswith("v1")

    path.write_text(json.dumps({"format": "unknown"}))
    with pytest.raises(ValueError, match="Unsupported streaming checkpoint"):
        load_manifest(tmp_path)