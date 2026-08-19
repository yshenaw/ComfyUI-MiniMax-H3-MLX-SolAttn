import numpy as np

from comfyui_nodes import (
    MiniMaxH3MLXTurbo,
    NODE_CLASS_MAPPINGS,
    PRESETS,
    _first_block_cache,
    _memory_mode,
    _stream_io,
)


def test_node_registers_upload_ready_defaults():
    required = MiniMaxH3MLXTurbo.INPUT_TYPES()["required"]
    optional = MiniMaxH3MLXTurbo.INPUT_TYPES()["optional"]

    assert NODE_CLASS_MAPPINGS["MiniMaxH3MLXTurbo"] is MiniMaxH3MLXTurbo
    assert required["model_profile"][1]["default"] == "4-bit-pruned"
    assert required["generation_profile"][1]["default"] == "Turbo 4 Fast"
    assert optional["full20_fbc"][1]["default"] is True
    assert optional["stream_io"][1]["default"] == "auto"
    assert list(optional) == ["sol_tau", "full20_fbc", "stream_io"]
    assert required["memory_mode"][1]["default"] == "auto"
    assert "32/48/64 GB" in required["model_profile"][1]["tooltip"]
    assert required["qwen_precision"][1]["default"] == "prequantized 8-bit"
    assert required["attention"][1]["default"] == "sol_attn"
    assert MiniMaxH3MLXTurbo.OUTPUT_NODE is True
    assert PRESETS == {
        "Turbo 4 Fast": {"steps": 5, "turbo": True},
        "Turbo 8 Balanced": {"steps": 9, "turbo": True},
        "Full 20 Quality": {"steps": 21, "turbo": False},
    }


def test_first_block_cache_is_optional_for_full20_only():
    assert _first_block_cache(PRESETS["Turbo 4 Fast"], True) == "none"
    assert _first_block_cache(PRESETS["Turbo 8 Balanced"], True) == "none"
    assert _first_block_cache(PRESETS["Full 20 Quality"], True) == "safe"
    assert _first_block_cache(PRESETS["Full 20 Quality"], False) == "none"


def test_auto_memory_mode_uses_valid_profile_tiers():
    assert _memory_mode("4-bit-pruned", "auto", 24.0) == "stream2"
    assert _memory_mode("4-bit-pruned", "auto", 32.0) == "resident"
    assert _memory_mode("8-bit-pruned", "auto", 48.0) == "resident"
    assert _memory_mode("attention16-mlp8-pruned", "auto", 64.0) == "resident"
    assert _memory_mode("bf16-pruned", "auto", 48.0) == "stream2"
    assert _memory_mode("bf16-pruned", "auto", 64.0) == "stream2"
    assert _memory_mode("bf16-pruned", "auto", 96.0) == "resident"
    assert _memory_mode("bf16", "auto", 64.0) == "stream2"
    assert _memory_mode("bf16", "auto", 96.0) == "resident"
    assert _memory_mode("bf16", "stream5", 128.0) == "stream5"


def test_auto_stream_io_uses_offset_only_for_low_memory_streaming():
    assert _stream_io("stream2", "auto", 24.0) == "offset"
    assert _stream_io("stream2", "auto", 32.0) == "mlx"
    assert _stream_io("stream2", "offset", 64.0) == "offset"
    assert _stream_io("resident", "auto", 24.0) is None


def test_comfy_outputs_match_image_and_audio_contract(monkeypatch):
    import torch
    from comfyui_nodes import _comfy_outputs

    class Result:
        video = np.zeros((3, 32, 64, 3), dtype=np.uint8)
        audio = np.zeros((2, 160), dtype=np.float32)
        fps = 24
        sample_rate = 16000
        seconds_per_step = 1.25
        total_seconds = 5.0
        stage_metrics = {"denoise": {"peak_gb": 12.0}}

    images, audio, stats = _comfy_outputs(Result())

    assert images.shape == (3, 32, 64, 3)
    assert images.dtype == torch.float32
    assert audio["waveform"].shape == (1, 2, 160)
    assert audio["sample_rate"] == 16000
    assert '"frames": 3' in stats