#!/usr/bin/env python3
"""Repack an MLX DiT checkpoint into fixed, block-core, and AdaLN chunk parts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.load import (
    SKIP_KEYS,
    _adaln_curve_shape,
    _interleave_qkv_rows,
    shard_paths,
)
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.streaming_layout import MANIFEST_NAME, tensor_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    config = DiTConfig.from_json(source / "config.json")
    source_shards = shard_paths(source)
    curve_shape = _adaln_curve_shape(source_shards)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "config.json", out / "config.json")
    quant_config = source / "quant_config.json"
    source_qkv_layout = None
    if quant_config.is_file():
        source_qkv_layout = json.loads(quant_config.read_text()).get("qkv_layout")
        shutil.copy(quant_config, out / "quant_config.json")
    full_curve = source / "h3_silu_temb_grid.safetensors"
    if curve_shape is not None:
        if not full_curve.is_file():
            raise FileNotFoundError(f"Pruned curve checkpoint requires {full_curve}")
        shutil.copy(full_curve, out / full_curve.name)

    files: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = defaultdict(int)
    for part_index, shard in enumerate(source_shards):
        loaded = mx.load(str(shard))
        groups: dict[str, dict[str, mx.array]] = defaultdict(dict)
        for key, value in loaded.items():
            if key in SKIP_KEYS:
                continue
            if curve_shape is not None and source_qkv_layout != "interleaved":
                value = _interleave_qkv_rows(key, value, config)
            group, chunk_start = tensor_group(key, args.chunk_size)
            label = group if chunk_start is None else f"{group}-{chunk_start:03d}"
            groups[label][key] = value
            sizes[label] += value.nbytes

        for label, tensors in sorted(groups.items()):
            name = f"{label}.part-{part_index:03d}.safetensors"
            mx.save_safetensors(str(out / name), tensors, metadata={"format": "mlx"})
            files[label].append(name)
            print(f"  {name}: {sum(value.nbytes for value in tensors.values()) / 1e9:.2f} GB")
        del loaded, groups
        mx.clear_cache()

    chunks = []
    for start in range(0, config.num_layers, args.chunk_size):
        end = min(start + args.chunk_size, config.num_layers)
        core_label = f"core-{start:03d}"
        adaln_label = f"adaln-{start:03d}"
        chunks.append(
            {
                "start": start,
                "end": end,
                "core_files": files[core_label],
                "adaln_files": files[adaln_label],
                "core_bytes": sizes[core_label],
                "adaln_bytes": sizes[adaln_label],
            }
        )

    manifest = {
        "format": "minimax-h3-mlx-block-stream-v1",
        "quantized": quant_config.is_file(),
        "chunk_size": args.chunk_size,
        "num_blocks": config.num_layers,
        "adaln_curve_shape": list(curve_shape) if curve_shape is not None else None,
        "full_curve_file": full_curve.name if curve_shape is not None else None,
        "qkv_layout": "interleaved",
        "fixed_files": files["fixed"],
        "fixed_bytes": sizes["fixed"],
        "chunks": chunks,
    }
    with open(out / MANIFEST_NAME, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"Streaming checkpoint: {len(chunks)} chunks, "
        f"{sum(sizes.values()) / 1e9:.2f} GB -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())