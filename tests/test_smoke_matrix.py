"""Validate the fixed smoke matrix without loading model weights."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_matrix.py"
SPEC = importlib.util.spec_from_file_location("smoke_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_matrix_has_sixteen_unique_cases_per_resolution_and_variant():
    cases = MODULE.build_cases()
    assert len(cases) == 32
    assert len({case.name for case in cases}) == 32
    assert sum(case.height == 480 for case in cases) == 16
    assert sum(case.height == 720 for case in cases) == 16
    assert sum(case.dit_variant == "bf16" for case in cases) == 16
    assert sum(case.dit_variant == "8bit" for case in cases) == 16


def test_turbo_and_quality_forward_counts_are_explicit():
    cases = MODULE.build_cases()
    assert {(case.sigma_points, case.forwards) for case in cases if "turbo4" in case.name} == {(5, 4)}
    assert {(case.sigma_points, case.forwards) for case in cases if "turbo8" in case.name} == {(9, 8)}
    assert {(case.sigma_points, case.forwards) for case in cases if "quality20" in case.name} == {(21, 20)}
    assert all(case.first_block_cache == "none" for case in cases if case.turbo)
    quality = [case for case in cases if not case.turbo]
    assert {(case.sol_attn, case.first_block_cache) for case in quality} == {
        (False, "none"),
        (True, "none"),
        (False, "safe"),
        (True, "safe"),
    }