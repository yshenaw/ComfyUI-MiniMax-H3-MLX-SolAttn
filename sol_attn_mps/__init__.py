"""Apple Silicon Metal kernels used by the MLX-to-MPS Sol-Attn bridge."""

from ._metal_fwd import metal_supported, sol_attn_mps

__all__ = ["metal_supported", "sol_attn_mps"]
