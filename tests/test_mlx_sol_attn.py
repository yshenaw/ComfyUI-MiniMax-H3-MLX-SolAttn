"""Focused tests for the BF16-native MLX Sol-Attn reference."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.mlx_sol_attn import MLXSolAttnReference, sol_attn_reference
from minimax_h3_mlx.attention_backends import create_attention_backend
from test_dit_smoke import build_packed_layout, tiny_config
from minimax_h3_mlx.dit import MiniMaxH3DiT


def _inputs(tokens=137, heads=2, head_dim=16):
    mx.random.seed(23)
    shape = (1, heads, tokens, head_dim)
    q = mx.random.normal(shape).astype(mx.bfloat16)
    return q, mx.random.normal(shape).astype(mx.bfloat16), mx.random.normal(shape).astype(mx.bfloat16)


def test_exact_routes_match_dense_with_tail_block():
    q, k, v = _inputs()
    blocks = (q.shape[2] + 63) // 64
    actual = sol_attn_reference(q, k, v, sink_q=(0, blocks))
    expected = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5
    )
    mx.eval(actual, expected)

    assert actual.dtype == mx.bfloat16
    assert mx.allclose(actual, expected, atol=0.02, rtol=0.02).item()


def test_sparse_reference_is_finite_and_preserves_contract():
    q, k, v = _inputs(tokens=257)
    actual = sol_attn_reference(q, k, v, tau=2.0)
    mx.eval(actual)

    assert actual.shape == q.shape
    assert actual.dtype == q.dtype
    assert mx.all(mx.isfinite(actual)).item()


def test_backend_falls_back_outside_reference_limit():
    q, k, v = _inputs(tokens=65)
    backend = MLXSolAttnReference(min_tokens=0, max_reference_tokens=64)
    actual = backend(q, k, v, scale=q.shape[-1] ** -0.5)
    expected = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5
    )
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()
    assert backend.summary()["dense_calls"] == 1


def test_reference_backend_runs_through_dit_in_bfloat16():
    config = tiny_config()
    mx.random.seed(31)
    dit = MiniMaxH3DiT(config)
    backend = MLXSolAttnReference(
        tau=2.0,
        start_percent=0.0,
        end_percent=1.0,
        min_tokens=0,
        max_reference_tokens=256,
    )
    dit.set_attention_backend(backend)

    n_text, n_video, n_audio = 5, 65, 3
    text_i, video_i, audio_i, tags, ts_i, pos = build_packed_layout(
        n_text, n_video, n_audio
    )
    video = mx.random.normal((1, n_video, config.video_patch_dim)).astype(mx.bfloat16)
    audio = mx.random.normal((1, n_audio, config.audio_latents_dim)).astype(mx.bfloat16)
    text = mx.random.normal((1, n_text, config.text_dim)).astype(mx.bfloat16)
    video_out, audio_out = dit(
        video,
        audio,
        text,
        mx.array([0.0, 0.7]),
        ts_i,
        tags,
        pos,
        video_i,
        audio_i,
        text_i,
    )
    mx.eval(video_out, audio_out)

    assert video_out.shape == (1, n_video, config.video_patch_dim)
    assert audio_out.shape == (1, n_audio, config.audio_latents_dim)
    assert mx.all(mx.isfinite(video_out)).item()
    assert mx.all(mx.isfinite(audio_out)).item()
    assert backend.summary()["sparse_calls"] == config.num_layers


def test_attention_backend_factory_keeps_optional_dependencies_lazy():
    assert create_attention_backend("none") is None
    backend = create_attention_backend(
        "mlx-reference",
        min_tokens=0,
        start_percent=0.0,
        end_percent=1.0,
    )
    assert isinstance(backend, MLXSolAttnReference)

