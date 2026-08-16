from pathlib import Path

import pytest

from minimax_h3_mlx.comfy_models import (
    TURBO_FILENAME,
    missing_model_files,
    model_paths,
    require_models,
)


def test_profiles_share_components_but_select_distinct_transformers(tmp_path):
    int4 = model_paths("4-bit", tmp_path)
    int8 = model_paths("8-bit", tmp_path)

    assert int4.checkpoint == int8.checkpoint
    assert int4.qwen == int8.qwen
    assert int4.lora == int8.lora
    assert int4.transformer == tmp_path / "minimax_h3" / "transformers" / "4-bit"
    assert int8.transformer == tmp_path / "minimax_h3" / "transformers" / "8-bit"
    assert int4.lora.name == TURBO_FILENAME


def test_model_validation_reports_and_accepts_expected_layout(tmp_path):
    paths = model_paths("4-bit", tmp_path)
    missing = missing_model_files(paths)
    assert paths.lora in missing

    for path in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert missing_model_files(paths) == []
    assert require_models("4-bit", tmp_path) == paths


def test_unknown_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown MiniMax H3 profile"):
        model_paths("3-bit", tmp_path)