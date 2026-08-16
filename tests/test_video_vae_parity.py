"""Numeric parity of the MLX video VAE against the diffusers reference.

Uses the reference's own tiny CPU-parity config (`MINIMAX_H3_TEST_VIDEO_VAE_CONFIG`), which keeps
the checkpoint-tied dimensions — latent channels, the temporal geometry and the rotary ratio —
intact. The MLX model is the source of truth and its parameters are pushed through the official
`convert_video_vae_key` into the reference, so the per-head-interleaved `attn.to_qkv` and the fused
`[gate; up]` `ff.w1` are exercised rather than assumed.

    ./.venv/bin/python tests/test_video_vae_parity.py
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

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

from convert_minimax_h3_to_diffusers import (
    MINIMAX_H3_TEST_VIDEO_VAE_CONFIG,
    convert_video_vae_key,
)
from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def configs():
    raw = dict(MINIMAX_H3_TEST_VIDEO_VAE_CONFIG)
    mine = VideoVAEConfig(
        in_channels=raw["in_channels"],
        out_channels=raw["out_channels"],
        latent_channels=raw["latent_channels"],
        block_out_channels=tuple(raw["block_out_channels"]),
        layers_per_block=raw["layers_per_block"],
        spatial_downsample_factors=tuple(raw["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(raw["temporal_downsample_factors"]),
        norm_num_groups=raw["norm_num_groups"],
        norm_eps=raw["norm_eps"],
        decoder_num_layers=raw["decoder_num_layers"],
        decoder_num_attention_heads=raw["decoder_num_attention_heads"],
        decoder_attention_head_dim=raw["decoder_attention_head_dim"],
        decoder_num_register_tokens=raw["decoder_num_register_tokens"],
        decoder_ffn_mult=raw["decoder_ffn_mult"],
        decoder_rope_theta=raw["decoder_rope_theta"],
        decoder_rope_dim_ratio=raw["decoder_rope_dim_ratio"],
        decoder_norm_eps=raw["decoder_norm_eps"],
        clip_length=raw["clip_length"],
        token_drop=raw["token_drop"],
    )
    ref_kwargs = {k: v for k, v in raw.items()}
    return mine, ref_kwargs


def to_reference_state(model: VideoVAE, ref_config: dict) -> dict[str, torch.Tensor]:
    """MLX params -> original checkpoint layout -> diffusers keys."""
    out: dict[str, torch.Tensor] = {}
    for key, value in tree_flatten(model.parameters()):
        arr = np.array(value.astype(mx.float32))
        # Conv weights are channels-last in MLX: (C_out, kD, kH, kW, C_in) -> torch's
        # (C_out, C_in, kD, kH, kW).
        if arr.ndim == 5:
            arr = arr.transpose(0, 4, 1, 2, 3)
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        for new_key, new_tensor in convert_video_vae_key(key, tensor, ref_config):
            out[new_key] = new_tensor
    return out


def main() -> int:
    mine_cfg, ref_kwargs = configs()
    mx.random.seed(0)
    torch.manual_seed(0)

    model = VideoVAE(mine_cfg)
    mx.eval(model.parameters())

    ref = AutoencoderKLMiniMaxH3(**ref_kwargs).eval()
    state = to_reference_state(model, ref_kwargs)
    missing, unexpected = ref.load_state_dict(state, strict=False)
    missing = [k for k in missing if "mask_token" not in k]
    if missing or unexpected:
        print(f"FAIL: state dict mismatch\n  missing={missing[:10]}\n  unexpected={unexpected[:10]}")
        return 1
    check("state dict mapped", True, f"{len(state)} tensors")

    # A clip_length-aligned video, small enough that tiling is a no-op, plus a tiled case.
    rng = np.random.default_rng(0)
    for label, (frames, height, width), tile in [
        ("untiled 17f 64x64", (17, 64, 64), 256),
        ("tiled 17f 96x96", (17, 96, 96), 64),
    ]:
        model.tile_sample_min_height = model.tile_sample_min_width = tile
        model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = tile // 4
        ref.enable_tiling(tile, tile, tile // 4, tile // 4)

        pixels = rng.standard_normal((1, 3, frames, height, width)).astype(np.float32)

        got = np.array(model.encode(mx.array(pixels)))
        with torch.no_grad():
            want = ref._encode(torch.from_numpy(pixels)).numpy()
        ok = got.shape == want.shape
        check(f"{label} encode shape", ok, f"{got.shape} vs {want.shape}")
        if ok:
            delta = float(np.abs(got - want).max())
            check(f"{label} encode values", delta < 2e-4,
                  f"max|delta| {delta:.3e} (scale {np.abs(want).max():.3e})")

        ratio = mine_cfg.spatial_compression_ratio
        latent = rng.standard_normal(
            (1, mine_cfg.latent_channels, 5, height // ratio, width // ratio)
        ).astype(np.float32)
        got_d = np.array(model.decode(mx.array(latent)))
        with torch.no_grad():
            want_d = ref._decode(torch.from_numpy(latent)).numpy()
        ok = got_d.shape == want_d.shape
        check(f"{label} decode shape", ok, f"{got_d.shape} vs {want_d.shape}")
        if ok:
            delta = float(np.abs(got_d - want_d).max())
            check(f"{label} decode values", delta < 2e-4,
                  f"max|delta| {delta:.3e} (scale {np.abs(want_d).max():.3e})")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("video VAE parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
