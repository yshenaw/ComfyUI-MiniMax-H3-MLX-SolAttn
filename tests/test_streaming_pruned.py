from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.lora import apply_lora
from minimax_h3_mlx.streaming_dit import StreamingDiT
from test_dit_smoke import build_packed_layout, tiny_config


def _comfy_qkv_rows(value: mx.array, config) -> mx.array:
    rows = 3 * config.num_attention_heads * config.attention_head_dim
    if value.shape[0] != rows:
        return value
    tail = value.shape[1:]
    return value.reshape(
        config.num_attention_heads, 3, config.attention_head_dim, *tail
    ).transpose(1, 0, 2, *range(3, value.ndim + 2)).reshape(value.shape)


def test_pruned_stream2_matches_resident_with_full_width_lora(tmp_path):
    config = tiny_config()
    source = tmp_path / "source"
    streamed = tmp_path / "stream2"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))

    mx.random.seed(41)
    resident = MiniMaxH3DiT(config, adaln_curve_grid=5, adaln_curve_dim=8)
    resident.adaln_t_table = mx.random.normal((5, 8)).astype(mx.float32)
    full_curve = mx.random.normal((5, config.time_embed_dim)).astype(mx.bfloat16)
    resident._adaln_lora_t_table = full_curve
    mx.eval(resident.parameters(), full_curve)

    source_weights = {}
    for key, value in tree_flatten(resident.parameters()):
        source_weights[key] = (
            _comfy_qkv_rows(value, config)
            if key.endswith("attn.qkv_proj.weight")
            else value
        )
    mx.save_safetensors(
        str(source / "minimax_h3_fl2va_pruned_bf16.safetensors"), source_weights
    )
    mx.save_safetensors(
        str(source / "h3_silu_temb_grid.safetensors"),
        {"silu_t_emb_grid": full_curve},
    )

    rank = 2
    lora = {
        "final_layer.adaln_proj.linear.lora_A.weight": mx.random.normal(
            (rank, config.time_embed_dim)
        ).astype(mx.bfloat16),
        "final_layer.adaln_proj.linear.lora_B.weight": mx.random.normal(
            (config.final_adaln_out_features, rank)
        ).astype(mx.bfloat16),
    }
    for block_index in range(config.num_layers):
        lora[f"blocks.{block_index}.adaln_proj.linear.lora_A.weight"] = mx.random.normal(
            (rank, config.time_embed_dim)
        ).astype(mx.bfloat16)
        lora[f"blocks.{block_index}.adaln_proj.linear.lora_B.weight"] = mx.random.normal(
            (config.adaln_out_features, rank)
        ).astype(mx.bfloat16)
    lora_path = tmp_path / "lora.safetensors"
    mx.save_safetensors(str(lora_path), lora)
    apply_lora(resident, lora_path)

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_streaming_checkpoint.py"
    subprocess.run(
        [sys.executable, str(script), "--source", str(source), "--out", str(streamed), "--chunk-size", "2"],
        check=True,
        capture_output=True,
        text=True,
    )

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, positions = build_packed_layout(
        n_text, n_video, n_audio
    )
    mx.random.seed(43)
    args = (
        mx.random.normal((1, n_video, config.video_patch_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_audio, config.audio_latents_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_text, config.text_dim)).astype(mx.bfloat16),
        mx.array([0.0, 0.7], dtype=mx.float32),
        ts_i, tags, positions, video_i, audio_i, text_i,
    )
    resident_cache = ModulationCache.build(resident, args[3], dtype=mx.bfloat16)
    expected_video, expected_audio = resident(*args, modulation_cache=resident_cache)

    streaming = StreamingDiT(streamed, lora_path=lora_path, verbose=False)
    streamed_cache = streaming.build_modulation_cache(args[3])
    actual_video, actual_audio = streaming(*args, modulation_cache=streamed_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    assert streaming.summary()["adaln_curve_dim"] == 8
    assert mx.array_equal(actual_video, expected_video).item()
    assert mx.array_equal(actual_audio, expected_audio).item()