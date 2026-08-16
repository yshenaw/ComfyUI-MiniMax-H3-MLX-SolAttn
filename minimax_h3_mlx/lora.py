"""Low-rank adapters for MiniMax-H3 MLX Linear and QuantizedLinear modules."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


def _linear_input_dim(layer: nn.Module) -> int:
    bits = getattr(layer, "bits", None)
    if bits is not None and layer.weight.dtype == mx.uint32:
        return int(layer.weight.shape[-1]) * 32 // int(bits)
    return int(layer.weight.shape[-1])


class LoRALinear(nn.Module):
    """Keep the base projection intact and add ``strength * B(A(x))``."""

    def __init__(self, base: nn.Module, lora_a: mx.array, lora_b: mx.array, strength: float):
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.strength = float(strength)

    @property
    def weight(self):
        return self.base.weight

    @property
    def scales(self):
        return getattr(self.base, "scales", None)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        adapter_input = x.astype(self.lora_a.dtype)
        update = (adapter_input @ self.lora_a.T) @ self.lora_b.T
        return out + (self.strength * update).astype(out.dtype)


class CurveLoRALinear(LoRALinear):
    """Use rank-reduced curve coordinates for the base and full coordinates for LoRA."""

    def __call__(self, inputs: tuple[mx.array, mx.array]) -> mx.array:
        base_input, adapter_input = inputs
        out = self.base(base_input)
        adapter_input = adapter_input.astype(self.lora_a.dtype)
        update = (adapter_input @ self.lora_a.T) @ self.lora_b.T
        return out + (self.strength * update).astype(out.dtype)


def apply_lora(dit, path: str | Path, strength: float = 1.0, verbose: bool = False) -> int:
    """Apply a MiniMax-H3 LoRA to every matching MLX projection as a bypass adapter."""
    weights = mx.load(str(Path(path)))
    target_names = sorted({key.rsplit(".lora_", 1)[0] for key in weights})
    modules = dict(dit.named_modules())
    replacements = []
    missing = []
    curve_adaln_lora = False

    for name in target_names:
        base = modules.get(name)
        a = weights.get(f"{name}.lora_A.weight")
        b = weights.get(f"{name}.lora_B.weight")
        if base is None or a is None or b is None:
            missing.append(name)
            continue
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid LoRA tensors for {name}: A={a.shape}, B={b.shape}.")
        if _linear_input_dim(base) != a.shape[-1]:
            if ".adaln_proj.linear" not in name:
                raise ValueError(
                    f"LoRA input mismatch for {name}: base input={_linear_input_dim(base)}, "
                    f"A={a.shape}."
                )
            if getattr(dit, "_adaln_lora_t_table", None) is None:
                raise ValueError(
                    f"Curve AdaLN LoRA for {name} requires the aligned full-width time table."
                )
            replacement = CurveLoRALinear(base, a, b, strength)
            curve_adaln_lora = True
        else:
            replacement = LoRALinear(base, a, b, strength)
        replacements.append((name, replacement))

    if missing or len(replacements) != len(target_names):
        raise KeyError(
            f"LoRA/module mismatch: applied {len(replacements)}/{len(target_names)}, "
            f"missing examples: {missing[:8]}."
        )

    dit.update_modules(tree_unflatten(replacements))
    if curve_adaln_lora:
        dit._adaln_curve_lora_enabled = True
    mx.eval(dit.parameters())
    if verbose:
        ranks = sorted({int(weights[f"{name}.lora_A.weight"].shape[0]) for name in target_names})
        print(f"  LoRA: {len(replacements)} adapters, ranks {ranks}, strength {strength}")
    return len(replacements)
