#!/usr/bin/env python3
"""Download the MiniMax H3 components needed by the ComfyUI node."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.comfy_models import (
    QWEN_FILES,
    QWEN_REPO,
    QWEN_REVISION,
    TRANSFORMER_REPOS,
    TURBO_FILENAME,
    TURBO_REPO,
    UPSTREAM_REPO,
    default_models_dir,
    model_paths,
)


UPSTREAM_ALLOW_PATTERNS = [
    "FL2VA/model_index.json",
    "FL2VA/video_vae/**",
    "FL2VA/audio_vae/**",
]


@dataclass(frozen=True)
class DownloadTask:
    repo_id: str
    local_dir: Path
    allow_patterns: tuple[str, ...] = ()
    filename: str | None = None
    revision: str | None = None


def download_plan(profile: str, models_dir: str | Path) -> list[DownloadTask]:
    profiles = list(TRANSFORMER_REPOS) if profile == "all" else [f"{profile}-bit"]
    first = model_paths(profiles[0], models_dir)
    tasks = [
        DownloadTask(
            repo_id=UPSTREAM_REPO,
            local_dir=first.root / "upstream",
            allow_patterns=tuple(UPSTREAM_ALLOW_PATTERNS),
        ),
        DownloadTask(
            repo_id=QWEN_REPO,
            local_dir=first.qwen,
            allow_patterns=QWEN_FILES,
            revision=QWEN_REVISION,
        ),
        DownloadTask(
            repo_id=TURBO_REPO,
            local_dir=first.root / "loras",
            filename=TURBO_FILENAME,
        ),
    ]
    tasks.extend(
        DownloadTask(
            repo_id=TRANSFORMER_REPOS[item],
            local_dir=model_paths(item, models_dir).transformer,
        )
        for item in profiles
    )
    return tasks


def _accept_license(non_interactive: bool) -> None:
    if non_interactive:
        return
    print(
        "MiniMax H3 weights use the MiniMax H3 Community License, not Apache-2.0.\n"
        "Review: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE\n"
        "The converted MLX transformer is a modified redistribution of those weights."
    )
    answer = input("Type ACCEPT to download the model files: ").strip()
    if answer != "ACCEPT":
        raise SystemExit("Model download cancelled; the license was not accepted.")


def _download(task: DownloadTask) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    task.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {task.repo_id} -> {task.local_dir}", flush=True)
    if task.filename:
        hf_hub_download(
            repo_id=task.repo_id,
            filename=task.filename,
            local_dir=task.local_dir,
            revision=task.revision,
        )
    else:
        snapshot_download(
            repo_id=task.repo_id,
            local_dir=task.local_dir,
            allow_patterns=list(task.allow_patterns) or None,
            revision=task.revision,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command MiniMax H3 model setup for ComfyUI on Apple Silicon."
    )
    parser.add_argument(
        "--profile",
        choices=["4", "8", "all"],
        default="4",
        help="4 is the recommended default; all also downloads the 8-bit quality profile.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="ComfyUI models directory; auto-detected when omitted.",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm the MiniMax H3 Community License non-interactively.",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve() if args.models_dir else default_models_dir()
    _accept_license(args.accept_license)
    for task in download_plan(args.profile, models_dir):
        _download(task)

    installed = "4-bit and 8-bit" if args.profile == "all" else f"{args.profile}-bit"
    print(
        f"\nInstalled MiniMax H3 {installed} under {models_dir / 'minimax_h3'}.\n"
        "Restart ComfyUI and load workflows/minimax_h3_mlx_turbo4_sol.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())