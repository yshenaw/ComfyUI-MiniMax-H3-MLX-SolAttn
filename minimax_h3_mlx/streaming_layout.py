"""Block-local checkpoint layout for exact DiT weight streaming."""

from __future__ import annotations

import json
import re
from pathlib import Path


MANIFEST_NAME = "streaming_manifest.json"
_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def tensor_group(key: str, chunk_size: int) -> tuple[str, int | None]:
    """Return ``(fixed|core|adaln, chunk_start)`` for a checkpoint tensor key."""
    match = _BLOCK_KEY.match(key)
    if match is None:
        return "fixed", None
    block_index = int(match.group(1))
    group = "adaln" if match.group(2).startswith("adaln_proj.") else "core"
    return group, (block_index // chunk_size) * chunk_size


def load_manifest(model_dir: str | Path) -> dict[str, object]:
    path = Path(model_dir) / MANIFEST_NAME
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "minimax-h3-mlx-block-stream-v1":
        raise ValueError(f"Unsupported streaming checkpoint manifest: {path}")
    return manifest


def is_streaming_checkpoint(model_dir: str | Path) -> bool:
    return (Path(model_dir) / MANIFEST_NAME).is_file()


__all__ = ["MANIFEST_NAME", "is_streaming_checkpoint", "load_manifest", "tensor_group"]