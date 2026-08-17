import mlx.core as mx

from minimax_h3_mlx.load import _interleave_qkv_rows
from test_dit_smoke import tiny_config


def test_comfy_qkv_rows_are_interleaved_per_head():
    config = tiny_config()
    rows = 3 * config.num_attention_heads * config.attention_head_dim
    tensor = mx.arange(rows * 2).reshape(rows, 2)

    actual = _interleave_qkv_rows("blocks.0.attn.qkv_proj.weight", tensor, config)
    expected = tensor.reshape(
        3, config.num_attention_heads, config.attention_head_dim, 2
    ).transpose(1, 0, 2, 3).reshape(rows, 2)

    assert mx.array_equal(actual, expected).item()


def test_qkv_scales_and_biases_follow_weight_rows():
    config = tiny_config()
    rows = 3 * config.num_attention_heads * config.attention_head_dim
    values = mx.arange(rows).reshape(rows, 1)

    scales = _interleave_qkv_rows("blocks.0.attn.qkv_proj.scales", values, config)
    biases = _interleave_qkv_rows("blocks.0.attn.qkv_proj.biases", values, config)

    assert mx.array_equal(scales, biases).item()
    assert not mx.array_equal(scales, values).item()


def test_non_curve_parameter_is_unchanged():
    config = tiny_config()
    tensor = mx.arange(12).reshape(6, 2)

    assert _interleave_qkv_rows("blocks.0.mlp.fc1.weight", tensor, config) is tensor