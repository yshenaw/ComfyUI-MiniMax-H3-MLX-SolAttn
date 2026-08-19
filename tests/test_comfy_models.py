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
    assert int4.streaming_transformer_2.name == "4-bit-stream2"
    assert int4.lora.name == TURBO_FILENAME


def test_bf16_profile_has_stream2_path(tmp_path):
    paths = model_paths("bf16", tmp_path)

    assert paths.transformer.name == "bf16"
    assert paths.streaming_transformer_2.name == "bf16-stream2"


@pytest.mark.parametrize("profile", ["4-bit-pruned", "8-bit-pruned"])
def test_pruned_quant_profiles_have_distinct_resident_paths(tmp_path, profile):
    paths = model_paths(profile, tmp_path)

    assert paths.transformer.name == profile
    assert paths.streaming_transformer_2.name == f"{profile}-stream2"


def test_attention16_mlp8_profile_has_a_distinct_resident_path(tmp_path):
    paths = model_paths("attention16-mlp8-pruned", tmp_path)

    assert paths.transformer.name == "attention16-mlp8-pruned"
    assert paths.streaming_transformer_2.name == "attention16-mlp8-pruned-stream2"


def test_attention16_mlp8_missing_files_suggest_valid_setup_alias(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"--profile attention16-mlp8$"):
        require_models("attention16-mlp8-pruned", tmp_path)


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