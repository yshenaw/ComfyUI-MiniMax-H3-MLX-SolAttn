from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import is_fp32_key
from minimax_h3_mlx.streaming_dit import StreamingDiT
from test_dit_smoke import build_packed_layout, tiny_config


def test_mixed_bf16_stream2_matches_resident(tmp_path):
    config = tiny_config()
    source = tmp_path / "source"
    streamed = tmp_path / "stream2"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))

    mx.random.seed(17)
    resident = MiniMaxH3DiT(config)
    weights = {
        key: value.astype(mx.float32 if is_fp32_key(key) else mx.bfloat16)
        for key, value in tree_flatten(resident.parameters())
    }
    resident.update(tree_unflatten(list(weights.items())))
    mx.eval(resident.parameters())
    mx.save_safetensors(str(source / "model.safetensors"), weights)

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
    mx.random.seed(23)
    args = (
        mx.random.normal((1, n_video, config.video_patch_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_audio, config.audio_latents_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_text, config.text_dim)).astype(mx.bfloat16),
        mx.array([0.0, 0.7], dtype=mx.float32),
        ts_i,
        tags,
        positions,
        video_i,
        audio_i,
        text_i,
    )
    resident_cache = ModulationCache.build(resident, args[3], dtype=mx.bfloat16)
    expected_video, expected_audio = resident(*args, modulation_cache=resident_cache)

    streaming = StreamingDiT(streamed, verbose=False)
    streamed_cache = streaming.build_modulation_cache(args[3])
    actual_video, actual_audio = streaming(*args, modulation_cache=streamed_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    assert streaming.quant_config is None
    assert streaming.summary()["quantized"] is False
    assert not (streamed / "quant_config.json").exists()
    assert mx.array_equal(actual_video, expected_video).item()
    assert mx.array_equal(actual_audio, expected_audio).item()