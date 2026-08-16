"""Checkpoint loading for the MiniMax-H3 MLX port.

The MLX module tree reproduces the original checkpoint names exactly, so loading is a 1:1 key
match — the only tensor the checkpoint carries that the port does not hold is ``rope.inv_freq``,
which is recomputed bit-identically from the config.

MiniMax-H3 ships a **mixed-precision** transformer: the two input patch projections, the timestep
MLP and the two output heads are float32 while everything else (including the AdaLN projections)
is bfloat16. That split is preserved on load — it is not incidental. The timestep MLP feeds every
block's modulation, so rounding it biases all 50 blocks identically at every sampling step and the
error accumulates coherently along the denoising trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import MiniMaxH3DiT

# Substring matches, mirroring the reference's `_keep_in_fp32_modules`.
FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# Carried by the checkpoint but recomputed by the port.
SKIP_KEYS = ("rope.inv_freq",)


def is_fp32_key(key: str) -> bool:
    return key.startswith(FP32_PREFIXES)


def shard_paths(model_dir: str | Path) -> list[Path]:
    """Resolve the safetensors shards of a transformer directory, in index order."""
    model_dir = Path(model_dir)
    pruned = model_dir / "minimax_h3_fl2va_pruned_bf16.safetensors"
    if pruned.exists():
        return [pruned]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_dir / name for name in names]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}.")
    return shards


def _adaln_curve_shape(paths: list[Path]) -> tuple[int, int] | None:
    for path in paths:
        with safe_open(path, framework="numpy") as file:
            if "adaln_t_table" in file.keys():
                shape = file.get_slice("adaln_t_table").get_shape()
                return int(shape[0]), int(shape[1])
    return None


def load_dit(
    model_dir: str | Path,
    dtype: mx.Dtype | None = None,
    strict: bool = True,
    verbose: bool = False,
) -> MiniMaxH3DiT:
    """Load the 33B DiT from a released ``FL2VA/transformer`` (or ``Ref2VA/transformer``) directory.

    Args:
        model_dir: the transformer directory holding ``config.json`` and the shards.
        dtype: cast every tensor to this dtype. ``None`` (default) preserves the checkpoint's
            mixed float32/bfloat16 split, which is what the reference runs.
        strict: raise if the checkpoint and the module tree disagree on any key.
        verbose: print per-shard progress.

    Returns:
        A parameter-loaded :class:`MiniMaxH3DiT`.
    """
    model_dir = Path(model_dir)
    config = DiTConfig.from_json(model_dir / "config.json")
    paths = shard_paths(model_dir)
    curve_shape = _adaln_curve_shape(paths)
    model = MiniMaxH3DiT(
        config,
        adaln_curve_grid=None if curve_shape is None else curve_shape[0],
        adaln_curve_dim=8 if curve_shape is None else curve_shape[1],
    )

    # A quantized build carries `quant_config.json`. Quantized layers hold packed weights plus
    # scales and biases, so the module tree has to be quantized *before* loading or the keys will
    # not line up — the same recipe is replayed from the file rather than guessed.
    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        from .quantize import QuantConfig, apply_quantization_structure

        with open(quant_path) as fh:
            recipe = json.load(fh)
        apply_quantization_structure(
            model,
            QuantConfig(
                bits=recipe["bits"],
                group_size=recipe["group_size"],
                quantize_adaln=recipe.get("quantize_adaln", False),
                adaln_bits=recipe.get("adaln_bits") or 8,
                overrides={
                    str(path): None if bits is None else int(bits)
                    for path, bits in recipe.get("overrides", {}).items()
                },
                group_overrides={
                    str(path): int(group_size)
                    for path, group_size in recipe.get("group_overrides", {}).items()
                },
            ),
        )
        if verbose:
            print(f"  quantized structure: {recipe['bits']}-bit, group {recipe['group_size']}")

    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for shard in paths:
        started = time.perf_counter()
        loaded = mx.load(str(shard))
        for key, tensor in loaded.items():
            if key in SKIP_KEYS:
                continue
            if key not in expected:
                unexpected.append(key)
                continue
            if dtype is not None:
                # Bulk conversion of the whole 33B stack. Done on the CPU stream and materialized
                # per tensor: casting ~130 GB through Metal is enough submissions to trip the
                # command-buffer limits when anything else is using the device, and this path is
                # I/O-dominated anyway.
                with mx.stream(mx.cpu):
                    tensor = tensor.astype(dtype)
                    mx.eval(tensor)
            elif is_fp32_key(key) and tensor.dtype != mx.float32:
                # Only twelve small tensors; not worth a stream switch.
                tensor = tensor.astype(mx.float32)
            weights[key] = tensor
        if verbose:
            gb = sum(t.nbytes for t in loaded.values()) / 1e9
            print(f"  {shard.name}: {len(loaded)} tensors, {gb:.2f} GB, "
                  f"{time.perf_counter() - started:.1f}s")

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Checkpoint/module mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )

    model.update(tree_unflatten(list(weights.items())))
    if curve_shape is not None:
        full_curve_path = model_dir / "h3_silu_temb_grid.safetensors"
        if not full_curve_path.exists():
            raise FileNotFoundError(
                f"Curve checkpoint requires aligned full-width table {full_curve_path}."
            )
        full_curve = mx.load(str(full_curve_path))["silu_t_emb_grid"]
        if full_curve.shape[0] != curve_shape[0]:
            raise ValueError(
                f"AdaLN curve grids disagree: base={curve_shape[0]}, LoRA={full_curve.shape[0]}."
            )
        model._adaln_lora_t_table = full_curve
    mx.eval(model.parameters())
    if curve_shape is not None:
        mx.eval(model._adaln_lora_t_table)
    return model


def load_video_vae_config(model_dir: str | Path):
    """Read Video VAE metadata without constructing the model or loading weights."""
    from .video_vae import VideoVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)
    with open(model_dir / "source" / "config.json") as fh:
        source = json.load(fh)

    ch = source["ch"]
    return VideoVAEConfig(
        in_channels=source["in_channels"],
        out_channels=source["out_ch"],
        latent_channels=source["z_channels"],
        block_out_channels=tuple(ch * m for m in source["ch_mult"]),
        layers_per_block=source["num_res_blocks"],
        spatial_downsample_factors=tuple(source["space_down"]),
        temporal_downsample_factors=tuple(source["time_down"]),
        decoder_num_layers=source["vit_decoder_kwargs"]["num_layers"],
        decoder_num_attention_heads=source["vit_decoder_kwargs"]["heads"],
        decoder_attention_head_dim=source["vit_decoder_kwargs"]["dim_head"],
        decoder_rope_theta=source["vit_decoder_kwargs"]["rope_theta"],
        decoder_rope_dim_ratio=source["vit_decoder_kwargs"]["rope_dim_ratio"],
        clip_length=wrapper.get("vae_clip_length", 17),
        token_drop=wrapper.get("vae_token_drop", 3),
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def load_video_vae(model_dir: str | Path, strict: bool = True):
    """Load the video VAE from a released ``video_vae/`` directory.

    The weights live in ``source/model.safetensors`` under the original CompVis-style names, which
    the port reproduces. Only the convolution weights move: torch stores
    ``(C_out, C_in, kD, kH, kW)`` and MLX wants ``(C_out, kD, kH, kW, C_in)``.
    """
    from .video_vae import VideoVAE

    model_dir = Path(model_dir)
    config = load_video_vae_config(model_dir)
    model = VideoVAE(config)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    for key, tensor in mx.load(str(model_dir / "source" / "model.safetensors")).items():
        # An all-zero buffer of the masked-autoencoding objective; the decoder never reads it.
        if key == "decoder.mask_token":
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        if tensor.ndim == 5:
            # Channels-last conv weights, materialized one at a time **on the CPU stream**.
            #
            # Deferring 10 GB of transposes into a single graph overruns the Metal command-buffer
            # deadline outright. Doing them individually on the GPU is enough on an idle machine but
            # still fails when something else is competing for the device — which is exactly when a
            # user is most likely to be loading a model. The CPU stream has no such deadline. It
            # trades some load time for not failing — a one-time cost on a path that otherwise
            # aborts a multi-hour run at the last component.
            with mx.stream(mx.cpu):
                tensor = mx.contiguous(tensor.transpose(0, 2, 3, 4, 1))
                mx.eval(tensor)
        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Video VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def load_audio_vae_config(model_dir: str | Path):
    """Read Audio VAE metadata without constructing the model or loading weights."""
    from .audio_vae import AudioVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as fh:
        kwargs = json.load(fh)["metadata"]["kwargs"]
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)

    return AudioVAEConfig(
        encoder_dim=kwargs["encoder_dim"],
        encoder_rates=tuple(kwargs["encoder_rates"]),
        latent_dim=kwargs["latent_dim"],
        latent_channels=kwargs["vae_latent_channels"],
        decoder_dim=kwargs["decoder_dim"],
        decoder_rates=tuple(kwargs["decoder_rates"]),
        sampling_rate=kwargs["sample_rate"],
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def load_audio_vae(model_dir: str | Path, strict: bool = True):
    """Load the audio VAE from a released ``audio_vae/`` directory.

    Weight norm is **folded** here: the checkpoint stores ``weight_g`` / ``weight_v`` and the
    effective weight is ``g * v / ||v||`` with the norm taken over every axis but the first. Folding
    once at load is exactly equivalent to recomputing it on every forward, and it lets the port hold
    a plain weight (proven equivalent by the parity test, which reconstructs the pair).
    """
    from .audio_vae import AudioVAE

    model_dir = Path(model_dir)
    config = load_audio_vae_config(model_dir)
    model = AudioVAE(config)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    raw = dict(mx.load(str(model_dir / "model.safetensors")))
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for key, tensor in raw.items():
        if key.endswith(".filter"):
            continue  # recomputed by kaiser_sinc_filter1d
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            g = raw[f"{base}_g"]
            v = tensor
            norm = mx.sqrt(mx.sum(mx.square(v.reshape(v.shape[0], -1)), axis=1)).reshape(-1, 1, 1)
            tensor = g * v / norm
            key = base
        elif key.endswith(".weight_g"):
            continue

        if key not in expected:
            unexpected.append(key)
            continue

        if key.endswith(".weight") and tensor.ndim == 3:
            # Transposed convs are stored (C_in, C_out, kL); plain convs (C_out, C_in, kL).
            tensor = tensor.transpose(1, 2, 0) if ".ups." in key else tensor.transpose(0, 2, 1)
        elif key.endswith(".alpha") and tensor.ndim == 3:
            tensor = tensor.transpose(0, 2, 1)  # (1, C, 1) -> (1, 1, C)

        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Audio VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def parameter_summary(model: MiniMaxH3DiT) -> dict[str, object]:
    """Parameter counts and footprint, split by the AdaLN projections that can be dropped."""
    total = adaln = 0
    nbytes = adaln_bytes = 0
    for key, value in tree_flatten(model.parameters()):
        total += value.size
        nbytes += value.nbytes
        if ".adaln_proj." in key and key.startswith("blocks."):
            adaln += value.size
            adaln_bytes += value.nbytes
    return {
        "total_params": total,
        "adaln_params": adaln,
        "core_params": total - adaln,
        "total_gb": nbytes / 1e9,
        "adaln_gb": adaln_bytes / 1e9,
        "core_gb": (nbytes - adaln_bytes) / 1e9,
    }
