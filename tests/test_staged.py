"""Lightweight staged-runner schedule tests; no checkpoint download required."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.packing import TAG_TEXT, build_packed_sequence
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
from minimax_h3_mlx.staged import _attention_summary, _resolve_qwen_stages, _row_timestep_plan


def test_staged_timestep_plan_matches_pipeline():
    layout = build_packed_sequence(
        np.full((7,), TAG_TEXT, dtype=np.int64),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
    )
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(5)
    audio.set_timesteps(5)

    expected_table, expected_plan = MiniMaxH3Pipeline._row_timestep_plan(
        object(), layout, video.timesteps, audio.timesteps
    )
    actual_table, actual_plan = _row_timestep_plan(layout, video.timesteps, audio.timesteps)

    assert mx.array_equal(actual_table, expected_table).item()
    assert len(actual_plan) == len(expected_plan)
    assert all(
        mx.array_equal(actual, expected).item()
        for actual, expected in zip(actual_plan, expected_plan)
    )


def test_qwen8_auto_stays_single_but_explicit_split_is_honored():
    assert _resolve_qwen_stages(8, "auto", 32.0) == 1
    assert _resolve_qwen_stages(8, 2, 32.0) == 2
    assert _resolve_qwen_stages(None, "auto", 48.0) == 2
    assert _resolve_qwen_stages(None, "auto", 64.0) == 1
    assert _resolve_qwen_stages(None, 2, 512.0) == 2


def test_dense_attention_is_reported_as_mlx_fast_sdpa():
    assert _attention_summary(None) == {"backend": "mlx_fast_sdpa"}

    class Backend:
        def summary(self):
            return {"backend": "custom"}

    assert _attention_summary(Backend()) == {"backend": "custom"}