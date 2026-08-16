"""Attention backend construction for staged MiniMax-H3 inference."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _load_torch_mps_kernel(source_dir: str | Path):
    source_dir = Path(source_dir).expanduser().resolve()
    if not (source_dir / "_metal_fwd.py").exists():
        raise FileNotFoundError(f"Sol-Attn MPS backend not found in {source_dir}.")

    package_name = "_minimax_h3_solattn_mps"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_dir)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}._metal_fwd").sol_attn_mps


def create_attention_backend(
    name: str,
    *,
    tau: float = 1.3,
    start_percent: float = 0.2,
    end_percent: float = 0.9,
    min_tokens: int = 4096,
    sink_conditioning_rows: bool = True,
    sol_attn_mps_dir: str | Path | None = None,
):
    if name == "none":
        return None
    common = {
        "tau": tau,
        "start_percent": start_percent,
        "end_percent": end_percent,
        "min_tokens": min_tokens,
        "sink_conditioning_rows": sink_conditioning_rows,
    }
    if name == "mlx-reference":
        from .mlx_sol_attn import MLXSolAttnReference

        return MLXSolAttnReference(**common)
    if name == "torch-mps":
        if sol_attn_mps_dir is None:
            raise ValueError("`sol_attn_mps_dir` is required for the torch-mps backend.")
        from .torch_sol_attn_bridge import TorchMPSSolAttnBridge

        return TorchMPSSolAttnBridge(
            _load_torch_mps_kernel(sol_attn_mps_dir),
            **common,
        )
    raise ValueError(f"Unknown attention backend {name!r}.")


__all__ = ["create_attention_backend"]