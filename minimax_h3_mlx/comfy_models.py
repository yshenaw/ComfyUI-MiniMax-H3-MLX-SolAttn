"""ComfyUI model layout shared by the node and setup script."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_REPO = "MiniMaxAI/MiniMax-H3"
QWEN_REPO = "lmstudio-community/Qwen3-VL-32B-Instruct-MLX-8bit"
QWEN_REVISION = "2efd79148ce9a761b06051300ecf8beb486a68ad"
QWEN_FILES = (
    *(f"model-{index:05d}-of-00007.safetensors" for index in range(1, 7)),
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
)
TRANSFORMER_REPOS = {
    "4-bit": "pipenetwork/MiniMax-H3-MLX-4bit",
    "8-bit": "pipenetwork/MiniMax-H3-MLX-8bit",
    "bf16": "pipenetwork/MiniMax-H3-MLX-bf16",
    "bf16-pruned": "Comfy-Org/MiniMax-H3",
}
TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
TURBO_FILENAME = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"


@dataclass(frozen=True)
class ModelPaths:
    profile: str
    root: Path
    checkpoint: Path
    qwen: Path
    transformer: Path
    streaming_transformer_2: Path
    streaming_transformer: Path
    lora: Path


def default_models_dir() -> Path:
    override = os.environ.get("MINIMAX_H3_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    try:
        import folder_paths

        return Path(folder_paths.models_dir).resolve()
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        if repo_root.parent.name == "custom_nodes":
            return (repo_root.parent.parent / "models").resolve()
        for parent in repo_root.parents:
            if (parent / "folder_paths.py").is_file():
                return (parent / "models").resolve()
        return (repo_root / "models").resolve()


def model_paths(profile: str, models_dir: str | Path | None = None) -> ModelPaths:
    if profile not in TRANSFORMER_REPOS:
        choices = ", ".join(TRANSFORMER_REPOS)
        raise ValueError(f"Unknown MiniMax H3 profile {profile!r}; choose {choices}.")

    models = Path(models_dir).expanduser().resolve() if models_dir else default_models_dir()
    root = models / "minimax_h3"
    return ModelPaths(
        profile=profile,
        root=root,
        checkpoint=root / "upstream" / "FL2VA",
        qwen=root / "qwen" / "8-bit",
        transformer=root / "transformers" / profile,
        streaming_transformer_2=root / "transformers" / f"{profile}-stream2",
        streaming_transformer=root / "transformers" / f"{profile}-stream5",
        lora=root / "loras" / TURBO_FILENAME,
    )


def missing_model_files(paths: ModelPaths) -> list[Path]:
    required = [
        paths.checkpoint / "model_index.json",
        paths.checkpoint / "video_vae" / "config.json",
        paths.checkpoint / "audio_vae" / "config.json",
        *(paths.qwen / name for name in QWEN_FILES),
        paths.transformer / "config.json",
        paths.lora,
    ]
    if paths.profile in ("4-bit", "8-bit"):
        required.extend(
            (
                paths.transformer / "model.safetensors.index.json",
                paths.transformer / "quant_config.json",
            )
        )
    elif paths.profile == "bf16":
        required.append(paths.transformer / "model.safetensors.index.json")
    else:
        required.extend(
            (
                paths.transformer / "minimax_h3_fl2va_pruned_bf16.safetensors",
                paths.transformer / "h3_silu_temb_grid.safetensors",
            )
        )
    return [path for path in required if not path.is_file()]


def require_models(profile: str, models_dir: str | Path | None = None) -> ModelPaths:
    paths = model_paths(profile, models_dir)
    missing = missing_model_files(paths)
    if missing:
        setup = Path(__file__).resolve().parents[1] / "scripts" / "setup_comfyui.py"
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"MiniMax H3 {profile} model setup is incomplete:\n{details}\n"
            f"Run: python {setup} --profile {profile.replace('-bit', '')}"
        )
    return paths


__all__ = [
    "ModelPaths",
    "QWEN_FILES",
    "QWEN_REPO",
    "QWEN_REVISION",
    "TRANSFORMER_REPOS",
    "TURBO_FILENAME",
    "TURBO_REPO",
    "UPSTREAM_REPO",
    "default_models_dir",
    "missing_model_files",
    "model_paths",
    "require_models",
]