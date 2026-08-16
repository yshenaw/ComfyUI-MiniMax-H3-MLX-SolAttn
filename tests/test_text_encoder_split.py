"""Two-stage Qwen layer-range parity without large checkpoint downloads."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from test_text_encoder_parity import READ_LAYER, text_config, write_release_layout


def test_two_stage_layer_ranges_match_single_stage():
    torch.manual_seed(0)
    config = text_config()
    reference = Qwen3VLTextModel(config).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("norm.weight"):
                parameter.add_(torch.randn_like(parameter) * 0.05)

    sequence_length = 12
    input_ids_np = torch.randint(
        0, config.vocab_size, (1, sequence_length)
    ).numpy().astype(np.int32)
    position_ids_np = (
        torch.arange(sequence_length)[None, None, :]
        .expand(3, 1, sequence_length)
        .numpy()
    )

    with tempfile.TemporaryDirectory() as temporary:
        model_dir = Path(temporary)
        write_release_layout(reference, config, model_dir)

        single = MiniMaxH3TextEncoder(
            model_dir,
            num_layers=READ_LAYER,
            dtype=mx.float32,
            load_vision=False,
        )
        expected = single._hidden_states(
            mx.array(input_ids_np), mx.array(position_ids_np)
        )
        mx.eval(expected)

        first = MiniMaxH3TextEncoder(
            model_dir,
            num_layers=2,
            layer_start=0,
            dtype=mx.float32,
            load_vision=False,
        )
        middle = first._hidden_states(
            mx.array(input_ids_np), mx.array(position_ids_np)
        )
        mx.eval(middle)
        middle_np = np.array(middle)
        del first, middle
        gc.collect()
        mx.clear_cache()

        second = MiniMaxH3TextEncoder(
            model_dir,
            num_layers=2,
            layer_start=2,
            dtype=mx.float32,
            load_vision=False,
        )
        actual = second._hidden_states(
            mx.array(input_ids_np),
            mx.array(position_ids_np),
            inputs_embeds=mx.array(middle_np),
        )
        mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


def test_prequantized_lmstudio_layout_round_trip():
    torch.manual_seed(7)
    config = text_config()
    reference = Qwen3VLTextModel(config).eval()
    sequence_length = 8
    input_ids = np.random.default_rng(7).integers(
        0, config.vocab_size, (1, sequence_length), dtype=np.int32
    )
    position_ids = np.broadcast_to(
        np.arange(sequence_length, dtype=np.int32),
        (3, 1, sequence_length),
    ).copy()

    with tempfile.TemporaryDirectory() as temporary:
        model_dir = Path(temporary)
        write_release_layout(reference, config, model_dir)
        encoder = MiniMaxH3TextEncoder(
            model_dir,
            num_layers=READ_LAYER,
            dtype=mx.bfloat16,
            load_vision=False,
        )
        nn.quantize(
            encoder.language,
            group_size=64,
            bits=8,
            mode="affine",
            class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
        )
        mx.eval(encoder.language.parameters())
        expected = encoder._hidden_states(mx.array(input_ids), mx.array(position_ids))
        mx.eval(expected)

        weights = {
            f"language_model.model.{key}": value
            for key, value in tree_flatten(encoder.language.parameters())
        }
        mx.save_safetensors(str(model_dir / "model.safetensors"), weights)
        raw_config = json.loads((model_dir / "config.json").read_text())
        raw_config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
        (model_dir / "config.json").write_text(json.dumps(raw_config))

        loaded = MiniMaxH3TextEncoder(
            model_dir,
            num_layers=READ_LAYER,
            dtype=mx.bfloat16,
            load_vision=False,
        )
        actual = loaded._hidden_states(mx.array(input_ids), mx.array(position_ids))
        mx.eval(actual)

    assert isinstance(loaded.language.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(loaded.language.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert loaded.language.embed_tokens.weight.dtype == mx.uint32
    assert mx.array_equal(actual, expected).item()
