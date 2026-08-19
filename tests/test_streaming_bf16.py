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
from minimax_h3_mlx.first_block_cache import FirstBlockCache, FirstBlockCacheConfig
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
        [sys.executable, str(script), "--source", str(source), "--out", str(streamed), "--chunk-size", "1"],
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

    streaming = StreamingDiT(streamed, core_io="offset", verbose=False)
    streamed_cache = streaming.build_modulation_cache(args[3])
    actual_video, actual_audio = streaming(*args, modulation_cache=streamed_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    assert streaming.quant_config is None
    assert streaming.summary()["quantized"] is False
    assert streaming.summary()["core_io"] == "offset"
    assert streaming.summary()["core_file_gb_read"] > 0
    assert streaming.summary()["core_nocache_files"] > 0
    assert not (streamed / "quant_config.json").exists()
    assert mx.array_equal(actual_video, expected_video).item()
    assert mx.array_equal(actual_audio, expected_audio).item()

    cached_streaming = StreamingDiT(streamed, core_io="offset", verbose=False)
    cached_modulation = cached_streaming.build_modulation_cache(args[3])
    block_cache = FirstBlockCache(
        FirstBlockCacheConfig(
            threshold=0.08,
            start_percent=0.0,
            end_percent=1.0,
            max_consecutive_hits=2,
        )
    )
    block_cache.begin_step(0, 3)
    full_video, full_audio = cached_streaming(
        *args,
        modulation_cache=cached_modulation,
        first_block_cache=block_cache,
    )
    mx.eval(full_video, full_audio)
    full_loads = cached_streaming.chunk_loads
    block_cache.begin_step(1, 3)
    cached_video, cached_audio = cached_streaming(
        *args,
        modulation_cache=cached_modulation,
        first_block_cache=block_cache,
    )
    mx.eval(full_video, full_audio, cached_video, cached_audio)

    assert full_loads == config.num_layers
    assert cached_streaming.chunk_loads == full_loads + 1
    assert block_cache.summary()["cached_steps"] == 1
    assert max(block_cache.diff_values) < 1.0e-4
    assert mx.allclose(cached_video, full_video, atol=1.0e-4, rtol=1.0e-4).item()
    assert mx.allclose(cached_audio, full_audio, atol=1.0e-4, rtol=1.0e-4).item()