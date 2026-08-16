"""Fit32 precision recipe tests without full checkpoint weights."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_fit32_quant import (
    attention16_mlp4_quant_config,
    attention8_mlp6_quant_config,
    fit32_quant_config,
    full4_pruned_adaln8_quant_config,
    full4_pruned_adaln16_quant_config,
    full8_pruned_adaln8_quant_config,
    full8_pruned_adaln16_quant_config,
    pad_pruned_adaln_for_int8,
)
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.quantize import quantize_dit
from test_dit_smoke import tiny_config


def test_fit32_quantizes_attention_to_8bit_mlp_to_4bit_and_keeps_adaln_float():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    config = fit32_quant_config(model)
    summary = quantize_dit(model, config)

    assert summary["quantized_layers"] == {4: 8, 8: 8}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert isinstance(block.attn.qkv_proj, nn.QuantizedLinear)
        assert block.attn.qkv_proj.bits == 8
        assert isinstance(block.attn.out_proj, nn.QuantizedLinear)
        assert block.attn.out_proj.bits == 8
        assert isinstance(block.mlp.fc1, nn.QuantizedLinear)
        assert block.mlp.fc1.bits == 4
        assert isinstance(block.mlp.fc2, nn.QuantizedLinear)
        assert block.mlp.fc2.bits == 4
    assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)
    assert model.blocks[0].adaln_proj.linear.weight.dtype in (
        mx.float16,
        mx.bfloat16,
        mx.float32,
    )


def test_attention16_mlp4_keeps_attention_and_adaln_float():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    config = attention16_mlp4_quant_config(model)
    summary = quantize_dit(model, config)

    assert summary["quantized_layers"] == {4: 8}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert isinstance(block.attn.qkv_proj, nn.Linear)
        assert isinstance(block.attn.out_proj, nn.Linear)
        assert isinstance(block.mlp.fc1, nn.QuantizedLinear)
        assert block.mlp.fc1.bits == 4
        assert isinstance(block.mlp.fc2, nn.QuantizedLinear)
        assert block.mlp.fc2.bits == 4
    assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)


def test_attention8_mlp6_keeps_adaln_float():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    config = attention8_mlp6_quant_config(model)
    summary = quantize_dit(model, config)

    assert summary["quantized_layers"] == {6: 8, 8: 8}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert isinstance(block.attn.qkv_proj, nn.QuantizedLinear)
        assert block.attn.qkv_proj.bits == 8
        assert isinstance(block.attn.out_proj, nn.QuantizedLinear)
        assert block.attn.out_proj.bits == 8
        assert isinstance(block.mlp.fc1, nn.QuantizedLinear)
        assert block.mlp.fc1.bits == 6
        assert isinstance(block.mlp.fc2, nn.QuantizedLinear)
        assert block.mlp.fc2.bits == 6
    assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)


def test_full4_pruned_adaln16_quantizes_core_only():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    summary = quantize_dit(model, full4_pruned_adaln16_quant_config())

    assert summary["quantized_layers"] == {4: 16}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert isinstance(block.attn.qkv_proj, nn.QuantizedLinear)
        assert block.attn.qkv_proj.bits == 4
        assert isinstance(block.attn.out_proj, nn.QuantizedLinear)
        assert block.attn.out_proj.bits == 4
        assert isinstance(block.mlp.fc1, nn.QuantizedLinear)
        assert block.mlp.fc1.bits == 4
        assert isinstance(block.mlp.fc2, nn.QuantizedLinear)
        assert block.mlp.fc2.bits == 4
    assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)


def test_full4_pruned_adaln8_pads_curve_and_quantizes_blocks():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    paths = pad_pruned_adaln_for_int8(model)
    summary = quantize_dit(model, full4_pruned_adaln8_quant_config(paths))

    assert model.adaln_t_table.shape == (5, 32)
    assert summary["quantized_layers"] == {4: 16, 8: 2}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert block.attn.qkv_proj.bits == 4
        assert block.attn.out_proj.bits == 4
        assert block.mlp.fc1.bits == 4
        assert block.mlp.fc2.bits == 4
    for block in model.blocks:
        assert isinstance(block.adaln_proj.linear, nn.QuantizedLinear)
        assert block.adaln_proj.linear.bits == 8
        assert block.adaln_proj.linear.group_size == 32
    assert isinstance(model.final_layer.adaln_proj.linear, nn.Linear)
    assert model.final_layer.adaln_proj.linear.weight.shape[-1] == 32


def test_full8_pruned_adaln8_quantizes_core_and_blocks_to_8bit():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    paths = pad_pruned_adaln_for_int8(model)
    summary = quantize_dit(model, full8_pruned_adaln8_quant_config(paths))

    assert model.adaln_t_table.shape == (5, 32)
    assert summary["quantized_layers"] == {8: 18}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert block.attn.qkv_proj.bits == 8
        assert block.attn.out_proj.bits == 8
        assert block.mlp.fc1.bits == 8
        assert block.mlp.fc2.bits == 8
    for block in model.blocks:
        assert block.adaln_proj.linear.bits == 8
        assert block.adaln_proj.linear.group_size == 32


def test_full8_pruned_adaln16_quantizes_core_only():
    model_config = tiny_config()
    model_config.ffn_hidden_size = 64
    model = MiniMaxH3DiT(model_config, adaln_curve_grid=5, adaln_curve_dim=8)
    summary = quantize_dit(model, full8_pruned_adaln16_quant_config())

    assert summary["quantized_layers"] == {8: 16}
    for block in [*model.token_refiner.blocks, *model.blocks]:
        assert block.attn.qkv_proj.bits == 8
        assert block.attn.out_proj.bits == 8
        assert block.mlp.fc1.bits == 8
        assert block.mlp.fc2.bits == 8
    assert isinstance(model.blocks[0].adaln_proj.linear, nn.Linear)