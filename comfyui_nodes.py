"""ComfyUI nodes for strict-staged MiniMax H3 MLX inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if __package__:
    from .minimax_h3_mlx.comfy_models import TRANSFORMER_REPOS, require_models
else:
    from minimax_h3_mlx.comfy_models import TRANSFORMER_REPOS, require_models


PACKAGE_ROOT = Path(__file__).resolve().parent

PRESETS = {
    "Turbo 4 Fast": {"steps": 5, "turbo": True, "fbc": "none"},
    "Turbo 6 Balanced": {"steps": 7, "turbo": True, "fbc": "none"},
    "Quality 20": {"steps": 21, "turbo": False, "fbc": "safe"},
}


def _comfy_outputs(result):
    import torch

    frames = np.ascontiguousarray(result.video)
    audio = np.ascontiguousarray(result.audio)
    images = torch.from_numpy(frames).to(torch.float32).div_(255.0)
    waveform = torch.from_numpy(audio).to(torch.float32).unsqueeze(0)
    metrics = {
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "frames": int(frames.shape[0]),
        "fps": int(result.fps),
        "seconds_per_step": round(float(result.seconds_per_step), 3),
        "total_seconds": round(float(result.total_seconds), 3),
        "stage_metrics": result.stage_metrics,
    }
    return images, {"waveform": waveform, "sample_rate": result.sample_rate}, json.dumps(
        metrics, indent=2
    )


class MiniMaxH3MLXGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model_profile": (
                    list(TRANSFORMER_REPOS),
                    {
                        "default": "4-bit",
                        "tooltip": "4/8-bit are fast resident tiers; BF16 stream2 is the 48/64 GB high-quality slow tier.",
                    },
                ),
                "generation_profile": (list(PRESETS), {"default": "Turbo 4 Fast"}),
                "memory_mode": (
                    ["auto", "resident", "stream2", "stream5"],
                    {"default": "auto"},
                ),
                "qwen_precision": (
                    ["prequantized 8-bit"],
                    {"default": "prequantized 8-bit"},
                ),
                "attention": (
                    ["sol_attn", "dense"],
                    {"default": "sol_attn"},
                ),
                "width": ("INT", {"default": 864, "min": 256, "max": 1280, "step": 32}),
                "height": ("INT", {"default": 480, "min": 256, "max": 720, "step": 32}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 18446744073709551615},
                ),
            },
            "optional": {
                "sol_tau": (
                    "FLOAT",
                    {"default": 1.3, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "stats")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3/MLX"
    DESCRIPTION = (
        "Generate synchronized video and audio with MiniMax H3 Turbo4, strict MLX "
        "component staging, and optional tiled Metal Sol-Attn. Powered by MiniMax H3."
    )

    def generate(
        self,
        prompt,
        model_profile,
        generation_profile,
        memory_mode,
        qwen_precision,
        attention,
        width,
        height,
        duration_seconds,
        seed,
        sol_tau=1.3,
    ):
        if width % 32 or height % 32:
            raise ValueError("MiniMax H3 width and height must be multiples of 32.")

        paths = require_models(model_profile)
        preset = PRESETS[generation_profile]
        if memory_mode == "auto":
            import os

            physical_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
            if model_profile == "bf16":
                memory_mode = "stream2" if physical_gb < 96.0 else "resident"
            else:
                memory_mode = "stream5" if physical_gb < 40.0 else "resident"
        else:
            physical_gb = 32.0 if memory_mode in ("stream2", "stream5") else 48.0
        qwen_stages = 2 if physical_gb < 40.0 else 1
        transformer = {
            "resident": paths.transformer,
            "stream2": paths.streaming_transformer_2,
            "stream5": paths.streaming_transformer,
        }[memory_mode]
        if memory_mode.startswith("stream") and not transformer.is_dir():
            raise FileNotFoundError(
                f"Streaming transformer not found: {transformer}. "
                "Build it with scripts/build_streaming_checkpoint.py."
            )

        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        if __package__:
            from .minimax_h3_mlx.attention_backends import create_attention_backend
            from .minimax_h3_mlx.staged import StrictStagedTextToVideo
        else:
            from minimax_h3_mlx.attention_backends import create_attention_backend
            from minimax_h3_mlx.staged import StrictStagedTextToVideo

        backend = create_attention_backend(
            "torch-mps" if attention == "sol_attn" else "none",
            tau=sol_tau,
            start_percent=0.2,
            end_percent=0.9,
            min_tokens=4096,
            sink_conditioning_rows=True,
            sol_attn_mps_dir=PACKAGE_ROOT / "sol_attn_mps",
        )
        runner = StrictStagedTextToVideo(
            paths.checkpoint,
            transformer,
            lora_path=paths.lora if preset["turbo"] else None,
            lora_strength=1.0,
            qwen_dir=paths.qwen,
            qwen_bits=None,
            qwen_stages=qwen_stages,
            first_block_cache=preset["fbc"],
            attention_backend=backend,
            verbose=True,
        )
        result = runner(
            prompt,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            num_inference_steps=preset["steps"],
            seed=seed,
            drop_adaln=True,
        )
        return _comfy_outputs(result)


MiniMaxH3MLXTurbo = MiniMaxH3MLXGenerate

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MLXGenerate": MiniMaxH3MLXGenerate,
    "MiniMaxH3MLXTurbo": MiniMaxH3MLXGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MLXGenerate": "MiniMax H3 MLX Generate (Sol-Attn)",
    "MiniMaxH3MLXTurbo": "MiniMax H3 MLX Turbo (Legacy)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]