"""Build a Markdown report from the 32-case smoke matrix results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _value(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _fmt(value, digits=2):
    return "-" if value is None else f"{float(value):.{digits}f}"


def build_report(summary: dict) -> str:
    results = summary.get("results", [])
    lines = [
        "# MiniMax-H3 MLX 5s Smoke Report",
        "",
        "## Configuration",
        "",
        "- Qwen: BF16, strict 25+25 stages",
        "- DiT variants: pruned BF16 and 8-bit weight-only",
        "- Resolutions: 864x480 and 1280x720",
        "- Duration: 5 seconds",
        "- Turbo: 4 and 6 real DiT forwards, with/without Sol-Attn",
        "- Quality: 20 real DiT forwards, all combinations of Sol-Attn and safe FBC",
        "",
        "## Results",
        "",
        "| Case | Status | Wall s | Denoise s | s/step | MLX peak GB | Footprint GB | Swap +GB | Sol sparse/dense | FBC cached/full | Output MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    failures = []
    for result in results:
        case = result.get("case", {})
        pipeline = result.get("pipeline", {})
        stages = pipeline.get("stage_metrics", {})
        denoise = stages.get("denoise", {})
        attention = denoise.get("attention_backend", {})
        fbc = denoise.get("first_block_cache", {})
        returncode = result.get("returncode")
        expected_forwards = case.get("forwards")
        actual_forwards = denoise.get("forward_count")
        forward_ok = expected_forwards is None or actual_forwards == expected_forwards
        status = "DRY" if returncode is None and "command" in result else ("PASS" if returncode == 0 and forward_ok else "FAIL")
        if status == "FAIL":
            failures.append(result)
        sol_counts = "-"
        if attention:
            sol_counts = f"{attention.get('sparse_calls', 0)}/{attention.get('dense_calls', 0)}"
        fbc_counts = "-"
        if fbc:
            fbc_counts = f"{fbc.get('cached_steps', 0)}/{fbc.get('full_steps', 0)}"
        lines.append(
            f"| {case.get('name', '?')} | {status} | {_fmt(result.get('wall_seconds'))} | "
            f"{_fmt(denoise.get('seconds'))} | {_fmt(pipeline.get('seconds_per_step'))} | "
            f"{_fmt(denoise.get('peak_gb'))} | {_fmt(result.get('peak_footprint_gb'))} | "
            f"{_fmt(result.get('swap_growth_gb'))} | {sol_counts} | {fbc_counts} | "
            f"{_fmt(result.get('output_bytes', 0) / 1e6)} |"
        )

    passed = [result for result in results if result.get("returncode") == 0]
    lines.extend(["", "## Completion", "", f"- Completed: {len(passed)}/32", f"- Failed or blocked: {len(failures)}/32"])

    lines.extend(["", "## Variant Summary", ""])
    for variant in ("bf16", "8bit"):
        selected = [result for result in passed if result.get("case", {}).get("dit_variant") == variant]
        denoise_times = [
            _value(result, "pipeline", "stage_metrics", "denoise", "seconds")
            for result in selected
        ]
        footprints = [result.get("peak_footprint_gb") for result in selected]
        denoise_times = [value for value in denoise_times if value is not None]
        footprints = [value for value in footprints if value is not None]
        lines.append(
            f"- {variant}: {len(selected)}/16 passed; median denoise "
            f"{_fmt(statistics.median(denoise_times) if denoise_times else None)}s; "
            f"median footprint {_fmt(statistics.median(footprints) if footprints else None)}GB"
        )

    by_name = {result.get("case", {}).get("name"): result for result in passed}
    lines.extend(["", "## Paired Comparisons", ""])
    for name, sol_result in sorted(by_name.items()):
        if not name or "_sol" not in name or "_sol_fbc" in name:
            continue
        dense_name = name.replace("_sol", "_dense")
        dense_result = by_name.get(dense_name)
        if dense_result is None:
            continue
        sol_time = _value(sol_result, "pipeline", "stage_metrics", "denoise", "seconds")
        dense_time = _value(dense_result, "pipeline", "stage_metrics", "denoise", "seconds")
        if sol_time and dense_time:
            lines.append(f"- `{name}` vs dense: {dense_time / sol_time:.3f}x denoise speedup")

    for name, int8_result in sorted(by_name.items()):
        if not name or not name.startswith("8bit_"):
            continue
        bf16_result = by_name.get("bf16_" + name[len("8bit_"):])
        if bf16_result is None:
            continue
        delta = bf16_result.get("peak_footprint_gb", 0) - int8_result.get("peak_footprint_gb", 0)
        lines.append(f"- `{name}` vs BF16: {_fmt(delta)}GB lower peak footprint")
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            error = (result.get("error_tail") or "").splitlines()
            root = error[-1] if error else f"see {result.get('log', '-')}"
            lines.append(f"- `{result.get('case', {}).get('name', '?')}`: {root}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.results.is_dir():
        results = []
        for path in sorted(args.results.glob("*.smoke.json")):
            with open(path) as file:
                results.append(json.load(file))
        summary = {"results": results}
        output = args.output or args.results / "SMOKE_REPORT.md"
    else:
        with open(args.results) as file:
            summary = json.load(file)
        output = args.output or args.results.with_name("SMOKE_REPORT.md")
    output.write_text(build_report(summary))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())