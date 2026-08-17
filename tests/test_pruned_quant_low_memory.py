from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_fit32_quant import (
    full4_pruned_adaln8_quant_config,
    full8_pruned_adaln8_quant_config,
    pad_pruned_adaln_for_int8,
)
from build_pruned_quant_low_memory import build
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import quantize_dit
from test_dit_smoke import build_packed_layout, tiny_config


def _separate_qkv(value: mx.array, config) -> mx.array:
    rows = 3 * config.num_attention_heads * config.attention_head_dim
    if value.shape[0] != rows:
        return value
    tail = value.shape[1:]
    return value.reshape(
        config.num_attention_heads, 3, config.attention_head_dim, *tail
    ).transpose(1, 0, 2, *range(3, value.ndim + 2)).reshape(value.shape)


@pytest.mark.parametrize("bits", [4, 8])
def test_low_memory_pruned_quant_matches_full_model_recipe(tmp_path, bits):
    config = tiny_config()
    config.ffn_hidden_size = 64
    source = tmp_path / "source"
    output = tmp_path / f"output-{bits}"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))

    mx.random.seed(61)
    source_model = MiniMaxH3DiT(config, adaln_curve_grid=5, adaln_curve_dim=8)
    source_model.adaln_t_table = mx.random.normal((5, 8)).astype(mx.float32)
    source_weights = {
        key: _separate_qkv(value, config) if key.endswith("attn.qkv_proj.weight") else value
        for key, value in tree_flatten(source_model.parameters())
    }
    mx.save_safetensors(
        str(source / "minimax_h3_fl2va_pruned_bf16.safetensors"), source_weights
    )
    mx.save_safetensors(
        str(source / "h3_silu_temb_grid.safetensors"),
        {"silu_t_emb_grid": mx.random.normal((5, config.time_embed_dim)).astype(mx.bfloat16)},
    )

    expected = load_dit(source)
    paths = pad_pruned_adaln_for_int8(expected)
    quant_config = (
        full4_pruned_adaln8_quant_config(paths)
        if bits == 4
        else full8_pruned_adaln8_quant_config(paths)
    )
    quantize_dit(expected, quant_config)
    mx.eval(expected.parameters())

    metadata = build(source, output, bits)
    actual = load_dit(output)
    mx.eval(actual.parameters())

    expected_weights = dict(tree_flatten(expected.parameters()))
    actual_weights = dict(tree_flatten(actual.parameters()))
    assert metadata["qkv_layout"] == "interleaved"
    assert metadata["adaln_bytes"] > 0
    assert metadata["resident_bytes_after_adaln_drop"] < metadata["total_bytes"]
    assert expected_weights.keys() == actual_weights.keys()
    assert all(mx.array_equal(actual_weights[key], value).item() for key, value in expected_weights.items())

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, positions = build_packed_layout(
        n_text, n_video, n_audio
    )
    mx.random.seed(67)
    args = (
        mx.random.normal((1, n_video, config.video_patch_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_audio, config.audio_latents_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_text, config.text_dim)).astype(mx.bfloat16),
        mx.array([0.0, 0.7], dtype=mx.float32),
        ts_i, tags, positions, video_i, audio_i, text_i,
    )
    expected_video, expected_audio = expected(*args)
    actual_video, actual_audio = actual(*args)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)
    assert mx.array_equal(actual_video, expected_video).item()
    assert mx.array_equal(actual_audio, expected_audio).item()