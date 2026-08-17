import importlib.util
import sys
from pathlib import Path

from minimax_h3_mlx.comfy_models import TURBO_FILENAME


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_comfyui.py"
SPEC = importlib.util.spec_from_file_location("setup_comfyui", SCRIPT)
SETUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SETUP
SPEC.loader.exec_module(SETUP)


def test_default_plan_downloads_shared_assets_and_4bit_only(tmp_path):
    tasks = SETUP.download_plan("4", tmp_path)

    assert [task.repo_id for task in tasks] == [
        "MiniMaxAI/MiniMax-H3",
        "lmstudio-community/Qwen3-VL-32B-Instruct-MLX-8bit",
        "larryvrh/MiniMax-H3-Turbo-Lora",
        "pipenetwork/MiniMax-H3-MLX-4bit",
    ]
    assert tasks[0].local_dir == tmp_path / "minimax_h3" / "upstream"
    assert "FL2VA/transformer/**" not in tasks[0].allow_patterns
    assert "FL2VA/text_encoder/**" not in tasks[0].allow_patterns
    assert tasks[1].revision == "2efd79148ce9a761b06051300ecf8beb486a68ad"
    assert tasks[2].filename == TURBO_FILENAME


def test_all_plan_adds_8bit_without_duplicate_shared_assets(tmp_path):
    tasks = SETUP.download_plan("all", tmp_path)

    assert len(tasks) == 8
    assert tasks[-5].local_dir.name == "4-bit"
    assert tasks[-4].local_dir.name == "8-bit"
    assert tasks[-3].local_dir.name == "bf16"
    assert tasks[-2].destination.name == "minimax_h3_fl2va_pruned_bf16.safetensors"
    assert tasks[-1].filename == "config.json"
    assert sum(task.repo_id == "MiniMaxAI/MiniMax-H3" for task in tasks) == 1


def test_bf16_plan_selects_full_precision_transformer(tmp_path):
    tasks = SETUP.download_plan("bf16", tmp_path)

    assert tasks[-1].repo_id == "pipenetwork/MiniMax-H3-MLX-bf16"
    assert tasks[-1].local_dir.name == "bf16"


def test_pruned_bf16_plan_uses_comfy_weights_and_mlx_config(tmp_path):
    tasks = SETUP.download_plan("bf16-pruned", tmp_path)

    assert tasks[-2].repo_id == "Comfy-Org/MiniMax-H3"
    assert tasks[-2].filename == "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
    assert tasks[-2].destination.name == "minimax_h3_fl2va_pruned_bf16.safetensors"
    assert tasks[-1].repo_id == "pipenetwork/MiniMax-H3-MLX-bf16"
    assert tasks[-1].filename == "config.json"