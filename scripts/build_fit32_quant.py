"""Build the Fit32 DiT: 8-bit attention, 4-bit MLP, pruned 16-bit AdaLN."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_quant import save_sharded
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit, resident_footprint


def fit32_quant_config(model, group_size: int = 64) -> QuantConfig:
    overrides = {
        path: 4
        for path, _module in model.named_modules()
        if path.endswith((".mlp.fc1", ".mlp.fc2"))
    }
    return QuantConfig(
        bits=8,
        group_size=group_size,
        quantize_adaln=False,
        overrides=overrides,
    )


def attention16_mlp4_quant_config(model, group_size: int = 64) -> QuantConfig:
    overrides = {
        path: None
        for path, _module in model.named_modules()
        if path.endswith((".attn.qkv_proj", ".attn.out_proj"))
    }
    return QuantConfig(
        bits=4,
        group_size=group_size,
        quantize_adaln=False,
        overrides=overrides,
    )


def attention8_mlp6_quant_config(model, group_size: int = 64) -> QuantConfig:
    overrides = {
        path: 6
        for path, _module in model.named_modules()
        if path.endswith((".mlp.fc1", ".mlp.fc2"))
    }
    return QuantConfig(
        bits=8,
        group_size=group_size,
        quantize_adaln=False,
        overrides=overrides,
    )


def full4_pruned_adaln16_quant_config(group_size: int = 64) -> QuantConfig:
    return QuantConfig(
        bits=4,
        group_size=group_size,
        quantize_adaln=False,
    )


def full8_pruned_adaln16_quant_config(group_size: int = 64) -> QuantConfig:
    return QuantConfig(
        bits=8,
        group_size=group_size,
        quantize_adaln=False,
    )


def pad_pruned_adaln_for_int8(model, padded_dim: int = 32) -> list[str]:
    if model.adaln_t_table.shape[-1] >= padded_dim:
        raise ValueError("AdaLN curve table is already padded or not rank-reduced.")
    pad = padded_dim - model.adaln_t_table.shape[-1]
    model.adaln_t_table = mx.pad(model.adaln_t_table, ((0, 0), (0, pad)))

    paths = []
    modules = [
        *[(f"blocks.{index}.adaln_proj", block.adaln_proj) for index, block in enumerate(model.blocks)],
        ("final_layer.adaln_proj", model.final_layer.adaln_proj),
    ]
    for path, projection in modules:
        source = projection.linear
        target = nn.Linear(padded_dim, source.weight.shape[0], bias=source.bias is not None)
        target.weight = mx.pad(source.weight, ((0, 0), (0, pad)))
        if source.bias is not None:
            target.bias = source.bias
        projection.linear = target
        paths.append(f"{path}.linear")
    mx.eval(model.adaln_t_table, model.parameters())
    return paths


def pruned_adaln8_quant_config(
    adaln_paths: list[str],
    bits: int,
    group_size: int = 64,
) -> QuantConfig:
    return QuantConfig(
        bits=bits,
        group_size=group_size,
        quantize_adaln=True,
        adaln_bits=8,
        group_overrides={path: 32 for path in adaln_paths},
    )


def full4_pruned_adaln8_quant_config(
    adaln_paths: list[str],
    group_size: int = 64,
) -> QuantConfig:
    return pruned_adaln8_quant_config(adaln_paths, 4, group_size)


def full8_pruned_adaln8_quant_config(
    adaln_paths: list[str],
    group_size: int = 64,
) -> QuantConfig:
    return pruned_adaln8_quant_config(adaln_paths, 8, group_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="pruned BF16 transformer directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--recipe",
        choices=[
            "fit32",
            "attention16-mlp4",
            "attention8-mlp6",
            "full4-pruned-adaln16",
            "full4-pruned-adaln8",
            "full8-pruned-adaln16",
            "full8-pruned-adaln8",
        ],
        default="fit32",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    if not (source / "h3_silu_temb_grid.safetensors").is_file():
        raise FileNotFoundError("Fit32 requires the pruned curve AdaLN checkpoint and grid.")

    started = time.perf_counter()
    print(f"loading pruned source {source}", flush=True)
    model = load_dit(source, verbose=True)
    if args.recipe == "fit32":
        config = fit32_quant_config(model, args.group_size)
        expected_counts = {4: 104, 8: 104}
        recipe_name = "qkv_out_8bit_mlp_4bit_pruned_adaln_16bit"
        description = "8-bit QKV/out projections and 4-bit MLP projections"
    elif args.recipe == "attention16-mlp4":
        config = attention16_mlp4_quant_config(model, args.group_size)
        expected_counts = {4: 104}
        recipe_name = "qkv_out_16bit_mlp_4bit_pruned_adaln_16bit"
        description = "16-bit QKV/out projections and 4-bit MLP projections"
    elif args.recipe == "attention8-mlp6":
        config = attention8_mlp6_quant_config(model, args.group_size)
        expected_counts = {6: 104, 8: 104}
        recipe_name = "qkv_out_8bit_mlp_6bit_pruned_adaln_16bit"
        description = "8-bit QKV/out projections and 6-bit MLP projections"
    elif args.recipe == "full4-pruned-adaln16":
        config = full4_pruned_adaln16_quant_config(args.group_size)
        expected_counts = {4: 208}
        recipe_name = "full_core_4bit_pruned_adaln_16bit"
        description = "4-bit QKV/out and MLP projections"
    elif args.recipe == "full8-pruned-adaln16":
        config = full8_pruned_adaln16_quant_config(args.group_size)
        expected_counts = {8: 208}
        recipe_name = "full_core_8bit_pruned_adaln_16bit"
        description = "8-bit QKV/out and MLP projections"
    elif args.recipe in ("full4-pruned-adaln8", "full8-pruned-adaln8"):
        adaln_paths = pad_pruned_adaln_for_int8(model)
        core_bits = 4 if args.recipe.startswith("full4") else 8
        config = pruned_adaln8_quant_config(adaln_paths, core_bits, args.group_size)
        expected_counts = {core_bits: 258} if core_bits == 8 else {4: 208, 8: 50}
        recipe_name = f"full_core_{core_bits}bit_pruned_rank8_adaln_8bit_group32"
        description = (
            f"{core_bits}-bit QKV/out and MLP projections with pruned 8-bit AdaLN"
        )
    expected_overrides = 0 if "full" in args.recipe else 104
    if len(config.overrides) != expected_overrides:
        raise RuntimeError(f"Expected 104 explicit projection overrides, found {len(config.overrides)}.")

    summary = quantize_dit(model, config, verbose=True)
    if summary["quantized_layers"] != expected_counts:
        raise RuntimeError(
            f"Unexpected Fit32 quantized layer counts: {summary['quantized_layers']}."
        )

    footprint = resident_footprint(model)
    metadata = {
        "bits": config.bits,
        "group_size": args.group_size,
        "quantize_adaln": config.quantize_adaln,
        "adaln_bits": config.adaln_bits if config.quantize_adaln else None,
        "overrides": config.overrides,
        "group_overrides": config.group_overrides,
        "quantized_layers": {
            str(bits): count for bits, count in sorted(summary["quantized_layers"].items())
        },
        "gb_on_disk": round(footprint["total_gb"], 2),
        "gb_resident_after_adaln_drop": round(footprint["resident_gb"], 2),
        "recipe": recipe_name,
        "qkv_layout": "interleaved",
    }

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "config.json", out / "config.json")
    shutil.copy(
        source / "h3_silu_temb_grid.safetensors",
        out / "h3_silu_temb_grid.safetensors",
    )
    with open(out / "quant_config.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    with open(out / "README.md", "w") as handle:
        handle.write(
            "# MiniMax-H3 MLX Fit32\n\n"
            "> Powered by MiniMax H3.\n\n"
            f"{description}, and the source "
            "pruned 16-bit curve AdaLN. Activations and the precomputed modulation "
            "cache remain BF16. See `quant_config.json` for the exact recipe.\n"
        )

    names = save_sharded(model, out, {"quantization": json.dumps(metadata)})
    print(
        f"Fit32 checkpoint: {len(names)} shards, {footprint['total_gb']:.2f} GB, "
        f"resident after AdaLN drop {footprint['resident_gb']:.2f} GB, "
        f"built in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())