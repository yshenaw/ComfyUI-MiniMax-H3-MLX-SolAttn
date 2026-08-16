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

from minimax_h3_mlx.load import shard_paths
from minimax_h3_mlx.streaming_layout import MANIFEST_NAME, tensor_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "config.json", out / "config.json")
    shutil.copy(source / "quant_config.json", out / "quant_config.json")

    files: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = defaultdict(int)
    for part_index, shard in enumerate(shard_paths(source)):
        loaded = mx.load(str(shard))
        groups: dict[str, dict[str, mx.array]] = defaultdict(dict)
        for key, value in loaded.items():
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
    for start in range(0, 50, args.chunk_size):
        end = min(start + args.chunk_size, 50)
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
        "chunk_size": args.chunk_size,
        "num_blocks": 50,
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