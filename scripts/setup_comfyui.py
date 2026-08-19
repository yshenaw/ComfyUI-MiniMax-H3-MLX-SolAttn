#!/usr/bin/env python3
"""Download the MiniMax H3 components needed by the ComfyUI node."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
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
PRUNED_FILENAME = "minimax_h3_fl2va_pruned_bf16.safetensors"
GRID_FILENAME = "h3_silu_temb_grid.safetensors"
GRID_SHA256 = "30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8"
GRID_URL = (
    "https://raw.githubusercontent.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/"
    "e7ad532857f2327feb56cf7729a9a76857a6799f/h3_silu_temb_grid.safetensors"
)

LOCAL_PROFILE_ALIASES = {
    "24": "4-bit-pruned",
    "4-pruned": "4-bit-pruned",
    "8-pruned": "8-bit-pruned",
    "attention16-mlp8": "attention16-mlp8-pruned",
}


@dataclass(frozen=True)
class DownloadTask:
    repo_id: str
    local_dir: Path
    allow_patterns: tuple[str, ...] = ()
    filename: str | None = None
    revision: str | None = None
    destination: Path | None = None


def download_plan(profile: str, models_dir: str | Path) -> list[DownloadTask]:
    profile = LOCAL_PROFILE_ALIASES.get(profile, profile)
    profiles = list(TRANSFORMER_REPOS) if profile == "all" else [
        profile if profile in TRANSFORMER_REPOS else f"{profile}-bit"
    ]
    download_profiles = []
    for item in profiles:
        source_item = (
            "bf16-pruned"
            if TRANSFORMER_REPOS[item].startswith("local:")
            else item
        )
        if source_item not in download_profiles:
            download_profiles.append(source_item)
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
        task
        for item in download_profiles
        for task in (
            (
                DownloadTask(
                    repo_id="Comfy-Org/MiniMax-H3",
                    filename=f"diffusion_models/{PRUNED_FILENAME}",
                    local_dir=model_paths(item, models_dir).transformer,
                    destination=model_paths(item, models_dir).transformer / PRUNED_FILENAME,
                ),
                DownloadTask(
                    repo_id="pipenetwork/MiniMax-H3-MLX-bf16",
                    filename="config.json",
                    local_dir=model_paths(item, models_dir).transformer,
                ),
            )
            if item == "bf16-pruned"
            else (
                DownloadTask(
                    repo_id=TRANSFORMER_REPOS[item],
                    local_dir=model_paths(item, models_dir).transformer,
                ),
            )
        )
    )
    return tasks


def _build_pruned_quant(profile: str, models_dir: Path) -> None:
    if profile not in LOCAL_PROFILE_ALIASES:
        return
    source = model_paths("bf16-pruned", models_dir).transformer
    output = model_paths(LOCAL_PROFILE_ALIASES[profile], models_dir).transformer
    import subprocess

    if not (output / "model.safetensors.index.json").is_file():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Incomplete pruned quant output exists: {output}")
        if profile == "attention16-mlp8":
            command = [
                sys.executable,
                str(Path(__file__).with_name("build_fit32_quant.py")),
                "--source", str(source),
                "--out", str(output),
                "--recipe", "attention16-mlp8-adaln16",
            ]
        else:
            command = [
                sys.executable,
                str(Path(__file__).with_name("build_pruned_quant_low_memory.py")),
                "--source", str(source),
                "--out", str(output),
                "--bits", "4" if profile == "24" else profile[0],
                "--adaln-bits", "16",
            ]
        subprocess.run(command, check=True)

    if profile == "24":
        streamed = model_paths("4-bit-pruned", models_dir).streaming_transformer_2
        if not (streamed / "streaming_manifest.json").is_file():
            if streamed.exists() and any(streamed.iterdir()):
                raise FileExistsError(f"Incomplete streaming output exists: {streamed}")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("build_streaming_checkpoint.py")),
                    "--source", str(output),
                    "--out", str(streamed),
                    "--chunk-size", "2",
                ],
                check=True,
            )


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
        downloaded = Path(hf_hub_download(
            repo_id=task.repo_id,
            filename=task.filename,
            local_dir=task.local_dir,
            revision=task.revision,
        ))
        if task.destination is not None and downloaded.resolve() != task.destination.resolve():
            task.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), task.destination)
    else:
        snapshot_download(
            repo_id=task.repo_id,
            local_dir=task.local_dir,
            allow_patterns=list(task.allow_patterns) or None,
            revision=task.revision,
        )


def _install_pruned_grid(destination: Path) -> None:
    target = destination / GRID_FILENAME
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == GRID_SHA256:
        return
    sibling = Path(__file__).resolve().parents[2] / "ComfyUI-MiniMax-H3-Turbo" / GRID_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if sibling.is_file():
        shutil.copy2(sibling, target)
    else:
        temporary = target.with_suffix(".incomplete")
        urllib.request.urlretrieve(GRID_URL, temporary)
        temporary.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != GRID_SHA256:
        target.unlink(missing_ok=True)
        raise RuntimeError("Downloaded H3 timestep grid failed SHA256 validation.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command MiniMax H3 model setup for ComfyUI on Apple Silicon."
    )
    parser.add_argument(
        "--profile",
        choices=[
            "24", "4", "8", "4-pruned", "8-pruned", "attention16-mlp8",
            "bf16", "bf16-pruned", "all",
        ],
        default="4-pruned",
        help="4-pruned is the 32 GB creator default; choose 24 for the experimental 24 GB bundle.",
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
    if args.profile in ("bf16-pruned", "all"):
        _install_pruned_grid(model_paths("bf16-pruned", models_dir).transformer)
    if args.profile in LOCAL_PROFILE_ALIASES:
        _install_pruned_grid(model_paths("bf16-pruned", models_dir).transformer)
        _build_pruned_quant(args.profile, models_dir)
    elif args.profile == "all":
        _build_pruned_quant("4-pruned", models_dir)
        _build_pruned_quant("8-pruned", models_dir)
        _build_pruned_quant("attention16-mlp8", models_dir)

    installed = (
        "all resident and BF16 profiles"
        if args.profile == "all"
        else (
            "24 GB Core4 stream2 bundle"
            if args.profile == "24"
            else "BF16 pruned"
            if args.profile == "bf16-pruned"
            else (
                LOCAL_PROFILE_ALIASES[args.profile]
                if args.profile in LOCAL_PROFILE_ALIASES
                else ("BF16" if args.profile == "bf16" else f"{args.profile}-bit")
            )
        )
    )
    workflow = (
        "workflows/minimax_h3_mlx_24gb_turbo4_sol.json"
        if args.profile == "24"
        else "workflows/minimax_h3_mlx_turbo4_sol.json"
    )
    print(
        f"\nInstalled MiniMax H3 {installed} under {models_dir / 'minimax_h3'}.\n"
        f"Restart ComfyUI and load {workflow}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())