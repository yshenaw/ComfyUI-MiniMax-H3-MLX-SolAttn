#!/usr/bin/env python3
"""Build a resident 4/8-bit pruned H3 checkpoint one block at a time."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT, TransformerBlock
from minimax_h3_mlx.load import SKIP_KEYS, _interleave_qkv_rows, shard_paths


CORE_PATHS = {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"}


def _source_index(source: Path) -> tuple[list[Path], dict[str, Path]]:
    shards = shard_paths(source)
    locations = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            locations.update({key: shard for key in handle.keys()})
    return shards, locations


def _read_weights(
    keys: list[str],
    locations: dict[str, Path],
    config: DiTConfig,
) -> dict[str, mx.array]:
    output = {}
    for path in sorted({locations[key] for key in keys}):
        with safe_open(path, framework="pt") as handle:
            for key in (item for item in keys if locations[item] == path):
                value = mx.from_dlpack(handle.get_tensor(key))
                mx.eval(value)
                output[key] = _interleave_qkv_rows(key, value, config)
    return output


def _update(module: nn.Module, weights: dict[str, mx.array], prefix: str = "") -> None:
    local = {
        key[len(prefix) :]: value
        for key, value in weights.items()
        if key.startswith(prefix)
    }
    expected = {key for key, _ in tree_flatten(module.parameters())}
    if expected != local.keys():
        missing = sorted(expected - local.keys())
        unexpected = sorted(local.keys() - expected)
        raise KeyError(f"Checkpoint mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
    module.update(tree_unflatten(list(local.items())))


def _pad_projection(projection, padded_dim: int = 32) -> None:
    source = projection.linear
    if source.weight.shape[-1] >= padded_dim:
        return
    pad = padded_dim - source.weight.shape[-1]
    target = nn.Linear(padded_dim, source.weight.shape[0], bias=source.bias is not None)
    target.weight = mx.pad(source.weight, ((0, 0), (0, pad)))
    if source.bias is not None:
        target.bias = source.bias
    projection.linear = target


def _quantize_block(block: TransformerBlock, bits: int) -> None:
    def predicate(path, module):
        if not isinstance(module, nn.Linear):
            return False
        if path == "adaln_proj.linear":
            return {"group_size": 32, "bits": 8}
        if path in CORE_PATHS:
            return {"group_size": 64, "bits": bits}
        return False

    nn.quantize(block, group_size=64, bits=bits, class_predicate=predicate)
    mx.eval(block.parameters())


def _quantize_fixed(model: MiniMaxH3DiT, bits: int) -> None:
    nn.quantize(
        model,
        group_size=64,
        bits=bits,
        class_predicate=lambda path, module: (
            {"group_size": 64, "bits": bits}
            if isinstance(module, nn.Linear) and any(path.endswith(item) for item in CORE_PATHS)
            else False
        ),
    )
    mx.eval(model.parameters())


def _save_part(
    output: Path,
    name: str,
    tensors: dict[str, mx.array],
    weight_map: dict[str, str],
) -> int:
    mx.save_safetensors(str(output / name), tensors, metadata={"format": "mlx"})
    weight_map.update({key: name for key in tensors})
    return sum(value.nbytes for value in tensors.values())


def build(source: Path, output: Path, bits: int) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    config = DiTConfig.from_json(source / "config.json")
    _shards, locations = _source_index(source)
    if "adaln_t_table" not in locations:
        raise ValueError("Source is not a pruned curve checkpoint.")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "config.json", output / "config.json")
    shutil.copy2(source / "h3_silu_temb_grid.safetensors", output)
    weight_map = {}
    total_bytes = 0
    adaln_bytes = 0

    fixed = MiniMaxH3DiT(config, adaln_curve_grid=1025, adaln_curve_dim=8)
    fixed.blocks = []
    gc.collect()
    fixed_keys = sorted(key for key in locations if not key.startswith("blocks.") and key not in SKIP_KEYS)
    fixed_weights = _read_weights(fixed_keys, locations, config)
    _update(fixed, fixed_weights)
    fixed.adaln_t_table = mx.pad(fixed.adaln_t_table, ((0, 0), (0, 24)))
    _pad_projection(fixed.final_layer.adaln_proj)
    _quantize_fixed(fixed, bits)
    fixed_tensors = dict(tree_flatten(fixed.parameters()))
    total_bytes += _save_part(output, "model-fixed.safetensors", fixed_tensors, weight_map)
    del fixed, fixed_tensors, fixed_weights
    gc.collect()
    mx.clear_cache()

    for block_index in range(config.num_layers):
        prefix = f"blocks.{block_index}."
        keys = sorted(key for key in locations if key.startswith(prefix))
        weights = _read_weights(keys, locations, config)
        block = TransformerBlock(config, curve_dim=8)
        _update(block, weights, prefix)
        _pad_projection(block.adaln_proj)
        _quantize_block(block, bits)
        tensors = {f"blocks.{block_index}.{key}": value for key, value in tree_flatten(block.parameters())}
        adaln_bytes += sum(
            value.nbytes for key, value in tensors.items() if ".adaln_proj." in key
        )
        total_bytes += _save_part(
            output,
            f"model-block-{block_index:02d}.safetensors",
            tensors,
            weight_map,
        )
        del block, tensors, weights
        gc.collect()
        mx.clear_cache()
        print(f"  block {block_index + 1}/{config.num_layers}", flush=True)

    core_count = 4 * (config.num_layers + config.token_refiner_num_layers)
    counts = {"8": core_count + config.num_layers} if bits == 8 else {
        "4": core_count,
        "8": config.num_layers,
    }
    metadata = {
        "bits": bits,
        "group_size": 64,
        "quantize_adaln": True,
        "adaln_bits": 8,
        "overrides": {},
        "group_overrides": {
            f"blocks.{index}.adaln_proj.linear": 32 for index in range(config.num_layers)
        },
        "quantized_layers": counts,
        "total_bytes": total_bytes,
        "adaln_bytes": adaln_bytes,
        "resident_bytes_after_adaln_drop": total_bytes - adaln_bytes,
        "gb_on_disk": round(total_bytes / 1e9, 2),
        "gb_resident_after_adaln_drop": round((total_bytes - adaln_bytes) / 1e9, 2),
        "recipe": f"full_core_{bits}bit_pruned_rank8_adaln_8bit_group32",
        "qkv_layout": "interleaved",
    }
    (output / "quant_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n"
    )
    (output / "README.md").write_text(
        f"# MiniMax-H3 MLX {bits}-bit pruned\n\n"
        "> Powered by MiniMax H3.\n\n"
        "Blockwise low-memory conversion from the official pruned BF16 checkpoint. "
        "The core is quantized, rank-8 AdaLN is padded and quantized to 8-bit group 32, "
        "and QKV rows are stored in MLX per-head-interleaved order.\n"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
    args = parser.parse_args()
    metadata = build(Path(args.source).expanduser().resolve(), Path(args.out).expanduser().resolve(), args.bits)
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())