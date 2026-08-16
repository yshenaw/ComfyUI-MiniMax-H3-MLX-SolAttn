"""Numeric parity of the MLX DiT against the diffusers reference.

Runs at a tiny config with random weights, so no checkpoint download is needed. The MLX model is
the source of truth: its parameters are pushed through the *official* conversion path
(``reorder_interleaved_qkv`` + ``convert_transformer_key``) into ``MiniMaxH3Transformer3DModel``.
That way the test exercises the two raw-checkpoint layout quirks the port handles by reshape —
per-head-interleaved QKV and the fused ``[gate; value]`` SwiGLU — rather than assuming them.

Run with the project venv, which has the `minimax-h3` diffusers branch:

    ./.venv/bin/python tests/test_dit_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

from convert_minimax_h3_to_diffusers import convert_transformer_key, reorder_interleaved_qkv
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT

HIDDEN = 64


def tiny_config() -> DiTConfig:
    return DiTConfig(
        hidden_size=HIDDEN,
        num_layers=2,
        token_refiner_num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=HIDDEN,
        time_embed_dim=32,
        adaln_out_features=6 * 3 * HIDDEN,
        final_adaln_out_features=2 * HIDDEN,
        rope_inv_freq_len=2,
    )


def as_diffusers_config(cfg: DiTConfig) -> dict:
    return {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_refiner_layers": cfg.token_refiner_num_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "ffn_dim": cfg.ffn_hidden_size,
        "in_channels": cfg.latents_dim,
        "audio_in_channels": cfg.audio_latents_dim,
        "patch_size": tuple(cfg.patch_size),
        "text_dim": cfg.text_dim,
        "freq_dim": cfg.timestep_input_dim,
        "time_embed_hidden_dim": cfg.time_embed_hidden_size,
        "time_embed_dim": cfg.time_embed_dim,
        "rope_freq_dim": cfg.rope_inv_freq_len,
        "rope_theta": cfg.rope_theta,
        "norm_eps": cfg.norm_eps,
        "qk_norm_eps": cfg.qk_norm_eps,
        "final_norm_eps": cfg.final_norm_eps,
    }


def mlx_params_as_checkpoint(dit: MiniMaxH3DiT) -> dict[str, torch.Tensor]:
    """Flatten the MLX module tree into original-checkpoint key names."""
    state: dict[str, torch.Tensor] = {}
    for key, value in tree_flatten(dit.parameters()):
        # MLX names lists as `blocks.0.…`, which is already the checkpoint spelling.
        state[key] = torch.from_numpy(np.array(value.astype(mx.float32)))
    return state


def to_diffusers_state_dict(state: dict[str, torch.Tensor], cfg: DiTConfig) -> dict[str, torch.Tensor]:
    dcfg = as_diffusers_config(cfg)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.endswith(".attn.qkv_proj.weight"):
            tensor = reorder_interleaved_qkv(
                tensor, dcfg["num_attention_heads"], dcfg["attention_head_dim"]
            )
        for new_key, new_tensor in convert_transformer_key(key, tensor, dcfg):
            out[new_key] = new_tensor
    return out


def build_layout(n_text: int, n_video: int, n_audio: int):
    seq = n_text + n_video + n_audio
    text_i = np.arange(n_text)
    video_i = np.arange(n_text, n_text + n_video)
    audio_i = np.arange(n_text + n_video, seq)

    tags = np.concatenate(
        [
            np.full(n_text, TAG_TEXT),
            np.full(n_video, TAG_VIDEO),
            np.full(n_audio, TAG_AUDIO),
        ]
    ).astype(np.int64)
    ts_i = np.concatenate(
        [np.zeros(n_text), np.ones(n_video), np.full(n_audio, 2)]
    ).astype(np.int64)
    pos = np.stack(
        [np.arange(seq) % 3, np.arange(seq) % 5, np.arange(seq) % 7], axis=-1
    ).astype(np.float64)
    return text_i, video_i, audio_i, tags, ts_i, pos


def main() -> int:
    cfg = tiny_config()
    mx.random.seed(0)
    torch.manual_seed(0)

    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())

    state = mlx_params_as_checkpoint(dit)
    ref = MiniMaxH3Transformer3DModel(**as_diffusers_config(cfg)).eval()
    converted = to_diffusers_state_dict(state, cfg)
    missing, unexpected = ref.load_state_dict(converted, strict=False)
    missing = [k for k in missing if "rope.inv_freq" not in k]
    if missing or unexpected:
        print(f"FAIL: state dict mismatch\n  missing={missing[:8]}\n  unexpected={unexpected[:8]}")
        return 1
    print(f"state dict mapped: {len(converted)} tensors, no missing/unexpected keys")

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, pos = build_layout(n_text, n_video, n_audio)

    rng = np.random.default_rng(0)
    video = rng.standard_normal((1, n_video, cfg.video_patch_dim)).astype(np.float32)
    audio = rng.standard_normal((1, n_audio, cfg.audio_latents_dim)).astype(np.float32)
    text = rng.standard_normal((1, n_text, cfg.text_dim)).astype(np.float32)
    timestep = np.array([0.0, 0.35, 0.9], dtype=np.float32)

    with torch.no_grad():
        ref_v, ref_a = ref(
            hidden_states=torch.from_numpy(video),
            audio_hidden_states=torch.from_numpy(audio),
            encoder_hidden_states=torch.from_numpy(text),
            timestep=torch.from_numpy(timestep),
            timestep_indices=torch.from_numpy(ts_i),
            token_tags=torch.from_numpy(tags),
            position_ids=torch.from_numpy(pos),
            video_indices=torch.from_numpy(video_i),
            audio_indices=torch.from_numpy(audio_i),
            text_indices=torch.from_numpy(text_i),
            return_dict=False,
        )

    got_v, got_a = dit(
        mx.array(video),
        mx.array(audio),
        mx.array(text),
        mx.array(timestep),
        mx.array(ts_i.astype(np.int32)),
        mx.array(tags.astype(np.int32)),
        mx.array(pos.astype(np.float32)),
        mx.array(video_i.astype(np.int32)),
        mx.array(audio_i.astype(np.int32)),
        mx.array(text_i.astype(np.int32)),
    )
    mx.eval(got_v, got_a)

    dv = float(np.abs(np.array(got_v) - ref_v.numpy()).max())
    da = float(np.abs(np.array(got_a) - ref_a.numpy()).max())
    scale_v = float(np.abs(ref_v.numpy()).max())
    scale_a = float(np.abs(ref_a.numpy()).max())
    print(f"video: max|delta| = {dv:.3e}  (ref scale {scale_v:.3e})")
    print(f"audio: max|delta| = {da:.3e}  (ref scale {scale_a:.3e})")

    tol = 2e-4
    if dv > tol or da > tol:
        print(f"FAIL: exceeds tolerance {tol:.1e}")
        return 1
    print(f"\nPASS: MLX DiT matches the diffusers reference within {tol:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
