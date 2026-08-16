"""LoRA adapter compatibility with MLX quantized storage."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.lora import _linear_input_dim


class LinearHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 128, bias=False)


def test_quantized_linear_reports_logical_input_dimension():
    holder = LinearHolder()
    assert _linear_input_dim(holder.linear) == 64
    nn.quantize(holder, group_size=64, bits=8)
    mx.eval(holder.parameters())

    assert holder.linear.weight.dtype == mx.uint32
    assert holder.linear.weight.shape[-1] == 16
    assert _linear_input_dim(holder.linear) == 64