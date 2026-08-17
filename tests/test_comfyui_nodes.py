import numpy as np

from comfyui_nodes import MiniMaxH3MLXTurbo, NODE_CLASS_MAPPINGS, PRESETS


def test_node_registers_upload_ready_defaults():
    required = MiniMaxH3MLXTurbo.INPUT_TYPES()["required"]

    assert NODE_CLASS_MAPPINGS["MiniMaxH3MLXTurbo"] is MiniMaxH3MLXTurbo
    assert required["model_profile"][1]["default"] == "4-bit"
    assert required["generation_profile"][1]["default"] == "Turbo 4 Fast"
    assert required["memory_mode"][1]["default"] == "auto"
    assert "BF16 stream2" in required["model_profile"][1]["tooltip"]
    assert required["qwen_precision"][1]["default"] == "prequantized 8-bit"
    assert required["attention"][1]["default"] == "sol_attn"
    assert MiniMaxH3MLXTurbo.OUTPUT_NODE is True
    assert PRESETS["Turbo 4 Fast"] == {"steps": 5, "turbo": True, "fbc": "none"}
    assert PRESETS["Turbo 6 Balanced"] == {"steps": 7, "turbo": True, "fbc": "none"}
    assert PRESETS["Quality 20"] == {"steps": 21, "turbo": False, "fbc": "safe"}


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