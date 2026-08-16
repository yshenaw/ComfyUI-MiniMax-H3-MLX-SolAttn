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

    assert len(tasks) == 5
    assert tasks[-2].local_dir.name == "4-bit"
    assert tasks[-1].local_dir.name == "8-bit"
    assert sum(task.repo_id == "MiniMaxAI/MiniMax-H3" for task in tasks) == 1