"""Parity of the truncated Qwen3-VL conditioner against HF transformers.

What is novel in the port — and therefore what this checks — is not the transformer stack (that is
mlx-vlm's) but MiniMax-H3's *use* of it: stop after N decoder layers and return the hidden state
**before** the final norm, i.e. exactly `output_hidden_states=True -> hidden_states[N]`.

A tiny random-weight model is written out in the release's own on-disk shape (a `config.json` with a
`text_config` block plus `model.language_model.*` safetensors), so the loader's layer filtering is
exercised too, not just the forward.

    ./.venv/bin/python tests/test_text_encoder_parity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

FAILURES: list[str] = []

HIDDEN = 64
HEAD_DIM = 16
NUM_LAYERS = 6
READ_LAYER = 4  # stand-in for the release's 50-of-64


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000.0,
            # Must sum to head_dim / 2, as [24, 20, 20] does for the release's head_dim 128.
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )


def write_release_layout(model: Qwen3VLTextModel, cfg: Qwen3VLTextConfig, path: Path) -> None:
    """Write the model the way the release ships its text encoder."""
    # The release stores the older, flat rope shape (`rope_theta` + `rope_scaling`), not
    # transformers 5.x's nested `rope_parameters`. Mirror the file the checkpoint actually ships.
    rope = cfg.rope_parameters
    text_block = {
        "model_type": "qwen3_vl_text",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_position_embeddings": cfg.max_position_embeddings,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "rope_theta": rope["rope_theta"],
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": rope["mrope_section"],
            "mrope_interleaved": rope["mrope_interleaved"],
        },
    }
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
        "text_config": text_block,
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": HIDDEN,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
            "hidden_act": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
        },
    }
    (path / "config.json").write_text(json.dumps(config))

    state = {f"model.language_model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    # The release also carries a head the conditioner never evaluates; include it so the loader is
    # seen to skip it.
    state["lm_head.weight"] = torch.zeros(cfg.vocab_size, HIDDEN)
    save_file(state, str(path / "model.safetensors"))


def main() -> int:
    torch.manual_seed(0)
    cfg = text_config()
    ref = Qwen3VLTextModel(cfg).eval()
    # Random init makes RMSNorm weights all-ones, which hides transposition errors; perturb them.
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if name.endswith("norm.weight"):
                param.add_(torch.randn_like(param) * 0.05)

    seq_len = 12
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    # Text-only M-RoPE: the same ramp on all three axes.
    ramp = torch.arange(seq_len)[None, None, :].expand(3, 1, seq_len)

    with torch.no_grad():
        out = ref(input_ids=input_ids, position_ids=ramp, use_cache=False, output_hidden_states=True)
    want = out.hidden_states[READ_LAYER].numpy()
    want_final = out.last_hidden_state.numpy()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        write_release_layout(ref, cfg, path)

        enc = MiniMaxH3TextEncoder(path, num_layers=READ_LAYER, dtype=mx.float32, load_vision=False)
        check("loader kept only the read layers",
              len(enc.language.layers) == READ_LAYER,
              f"{len(enc.language.layers)} layers, skipped {enc.skipped_tensors} tensors")

        got = np.array(
            enc._hidden_states(mx.array(input_ids.numpy().astype(np.int32)), mx.array(ramp.numpy()))
        )

    ok = got.shape == want.shape
    check("hidden state shape", ok, f"{got.shape} vs {want.shape}")
    if ok:
        delta = float(np.abs(got - want).max())
        check(f"hidden_states[{READ_LAYER}] values", delta < 2e-4,
              f"max|delta| {delta:.3e} (scale {np.abs(want).max():.3e})")

        # The point of reading pre-norm: it must NOT equal the normed final state.
        final_delta = float(np.abs(got - want_final).max())
        check("read is pre-norm, not the final hidden state", final_delta > 1e-3,
              f"differs from last_hidden_state by {final_delta:.3e}")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("text encoder parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
