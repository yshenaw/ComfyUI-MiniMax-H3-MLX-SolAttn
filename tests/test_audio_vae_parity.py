"""Numeric parity of the MLX audio VAE against the diffusers reference.

The MLX model is the source of truth. Its parameters are converted back into the reference's layout
— channels-first convolutions, torch's transposed-conv axis order, and the ``weight_g`` / ``weight_v``
weight-norm pair — and loaded into `AutoencoderKLMiniMaxH3Audio`. Reconstructing the weight-norm pair
as ``v = w``, ``g = ||w||`` reproduces the effective weight exactly, so folding weight norm at load in
the port is proven equivalent rather than assumed.

The recomputed Kaiser-sinc anti-aliasing filters are also checked against the ones the released
checkpoint actually ships, when it is available locally.

    ./.venv/bin/python tests/test_audio_vae_parity.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    AutoencoderKLMiniMaxH3Audio,
)

from minimax_h3_mlx.audio_vae import AudioVAE, AudioVAEConfig, kaiser_sinc_filter1d

CHECKPOINT = Path("/Volumes/models/MiniMax-H3/FL2VA/audio_vae/model.safetensors")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def tiny_config() -> AudioVAEConfig:
    """Same structure as the release, narrowed so the test runs on CPU in seconds.

    The hop length is kept a true product of both rate lists (4x4 = 16), since encode/decode
    consistency depends on it.
    """
    return AudioVAEConfig(
        encoder_dim=8,
        encoder_rates=(4, 4),
        latent_dim=64,
        latent_channels=8,
        num_attention_heads=2,
        decoder_dim=32,
        decoder_rates=(4, 4),
        decoder_kernel_sizes=(8, 8),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
    )


def to_reference_state(model: AudioVAE, ref_keys: set[str]) -> dict[str, torch.Tensor]:
    """MLX params -> the reference's state dict.

    * conv1d weights ``(C_out, kL, C_in)`` -> ``(C_out, C_in, kL)``
    * transposed-conv weights ``(C_out, kL, C_in)`` -> torch's ``(C_in, C_out, kL)``
    * Snake1d ``alpha`` ``(1, 1, C)`` -> ``(1, C, 1)``
    * a weight-normed conv's ``weight`` -> ``weight_v = w``, ``weight_g = ||w||`` over dims != 0
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in tree_flatten(model.parameters()):
        arr = np.array(value.astype(mx.float32))

        if key.endswith(".alpha") and arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        elif key.endswith(".weight") and arr.ndim == 3:
            is_transpose = ".ups." in key
            arr = arr.transpose(2, 0, 1) if is_transpose else arr.transpose(0, 2, 1)

        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if key.endswith(".weight") and f"{key}_g" in ref_keys:
            norm = tensor.flatten(1).norm(dim=1).reshape(-1, 1, 1)
            out[f"{key}_g"] = norm
            out[f"{key}_v"] = tensor
        else:
            out[key] = tensor
    return out


def test_filters_match_checkpoint() -> None:
    """The recomputed Kaiser-sinc filters must equal the ones the release ships."""
    if not CHECKPOINT.exists():
        print("skip  filters vs checkpoint — weights not present locally")
        return
    with open(CHECKPOINT, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data_start = 8 + n
        for key in ("decoder.activation_post.upsample.filter",
                    "decoder.activation_post.downsample.lowpass.filter"):
            meta = header[key]
            lo, hi = meta["data_offsets"]
            fh.seek(data_start + lo)
            shipped = np.frombuffer(fh.read(hi - lo), dtype=np.float32).reshape(meta["shape"])
            # BigVGAN's Activation1d defaults: ratio 2, kernel 12.
            mine = np.array(kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, 12)).reshape(shipped.shape)
            delta = float(np.abs(mine - shipped).max())
            check(f"filter {key.split('.')[-2]}", delta < 1e-7, f"max|delta| {delta:.3e}")


def main() -> int:
    cfg = tiny_config()
    mx.random.seed(0)
    torch.manual_seed(0)

    model = AudioVAE(cfg)
    mx.eval(model.parameters())

    ref = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=cfg.encoder_dim,
        encoder_rates=cfg.encoder_rates,
        latent_dim=cfg.latent_dim,
        latent_channels=cfg.latent_channels,
        num_attention_heads=cfg.num_attention_heads,
        decoder_dim=cfg.decoder_dim,
        decoder_rates=cfg.decoder_rates,
        decoder_kernel_sizes=cfg.decoder_kernel_sizes,
        resblock_kernel_sizes=cfg.resblock_kernel_sizes,
        resblock_dilation_sizes=cfg.resblock_dilation_sizes,
        sampling_rate=cfg.sampling_rate,
    ).eval()

    ref_keys = set(ref.state_dict().keys())
    state = to_reference_state(model, ref_keys)
    missing, unexpected = ref.load_state_dict(state, strict=False)
    # `filter` buffers are recomputed identically on both sides; `zero_k_bias` is a zero buffer.
    missing = [k for k in missing if not k.endswith(".filter")]
    if missing or unexpected:
        print(f"FAIL: state dict mismatch\n  missing={missing[:10]}\n  unexpected={unexpected[:10]}")
        return 1
    check("state dict mapped", True, f"{len(state)} tensors")

    rng = np.random.default_rng(0)
    hop = cfg.hop_length

    # encode: two batch items, as MiniMax-H3 passes stereo.
    wave = rng.standard_normal((2, 1, hop * 12)).astype(np.float32) * 0.1
    got_mean, got_logs = model.encode(mx.array(wave))
    mx.eval(got_mean, got_logs)
    with torch.no_grad():
        posterior = ref.encode(torch.from_numpy(wave), return_dict=False)[0]
    want_mean = posterior.mean.numpy()

    ok = tuple(got_mean.shape) == want_mean.shape
    check("encode shape", ok, f"{tuple(got_mean.shape)} vs {want_mean.shape}")
    if ok:
        delta = float(np.abs(np.array(got_mean) - want_mean).max())
        check("encode values", delta < 2e-4, f"max|delta| {delta:.3e} (scale {np.abs(want_mean).max():.3e})")

    # decode
    latents = rng.standard_normal((2, cfg.latent_channels, 12)).astype(np.float32)
    got = np.array(model.decode(mx.array(latents)))
    with torch.no_grad():
        want = ref.decode(torch.from_numpy(latents), return_dict=False)[0].numpy()
    ok = got.shape == want.shape
    check("decode shape", ok, f"{got.shape} vs {want.shape}")
    if ok:
        delta = float(np.abs(got - want).max())
        check("decode values", delta < 2e-4, f"max|delta| {delta:.3e} (scale {np.abs(want).max():.3e})")
        check("decode hop length", got.shape[-1] == 12 * hop, f"{got.shape[-1]} vs {12 * hop}")

    test_filters_match_checkpoint()

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("audio VAE parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
