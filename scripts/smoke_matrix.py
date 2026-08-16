"""Run the fixed MiniMax-H3 MLX 480p/720p five-second smoke matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeCase:
    name: str
    dit_variant: str
    width: int
    height: int
    sigma_points: int
    forwards: int
    turbo: bool
    sol_attn: bool
    first_block_cache: str


def build_cases() -> list[SmokeCase]:
    cases = []
    for dit_variant in ("bf16", "8bit"):
        for resolution, width, height in (("480p", 864, 480), ("720p", 1280, 720)):
            for label, sigma_points in (("turbo4", 5), ("turbo6", 7)):
                for sol_attn in (True, False):
                    cases.append(
                        SmokeCase(
                            name=f"{dit_variant}_{resolution}_{label}_{'sol' if sol_attn else 'dense'}",
                            dit_variant=dit_variant,
                            width=width,
                            height=height,
                            sigma_points=sigma_points,
                            forwards=sigma_points - 1,
                            turbo=True,
                            sol_attn=sol_attn,
                            first_block_cache="none",
                        )
                    )
            for first_block_cache in ("none", "safe"):
                for sol_attn in (False, True):
                    suffix = "sol" if sol_attn else "dense"
                    if first_block_cache != "none":
                        suffix += "_fbc"
                    cases.append(
                        SmokeCase(
                            name=f"{dit_variant}_{resolution}_quality20_{suffix}",
                            dit_variant=dit_variant,
                            width=width,
                            height=height,
                            sigma_points=21,
                            forwards=20,
                            turbo=False,
                            sol_attn=sol_attn,
                            first_block_cache=first_block_cache,
                        )
                    )
    return cases


def _parse_memory(value: str) -> int:
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value.strip(), re.IGNORECASE)
    if match is None:
        return 0
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def _process_memory(pid: int) -> tuple[int, int]:
    rss_text = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    rss = int(rss_text or 0) * 1024
    top_text = subprocess.run(
        ["/usr/bin/top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem", "-n", "1"],
        capture_output=True,
        text=True,
    ).stdout
    footprint = 0
    for line in reversed(top_text.splitlines()):
        fields = line.split()
        if fields and fields[0] == str(pid) and len(fields) >= 2:
            footprint = _parse_memory(fields[1])
            break
    return rss, footprint


def _swap_used_bytes() -> int:
    output = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"used = ([0-9.]+)([MGT])", output)
    return _parse_memory("".join(match.groups())) if match else 0


def _monitor(process: subprocess.Popen, state: dict[str, int], interval: float) -> None:
    while process.poll() is None:
        rss, footprint = _process_memory(process.pid)
        state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss)
        state["peak_footprint_bytes"] = max(state["peak_footprint_bytes"], footprint)
        state["peak_swap_bytes"] = max(state["peak_swap_bytes"], _swap_used_bytes())
        threading.Event().wait(interval)


def _command(args, case: SmokeCase, output: Path) -> list[str]:
    transformer = args.bf16_transformer if case.dit_variant == "bf16" else args.int8_transformer
    command = [
        str(Path(args.python).expanduser()),
        str(Path(__file__).with_name("generate_staged.py")),
        args.prompt,
        "--checkpoint",
        str(Path(args.checkpoint).expanduser()),
        "--transformer",
        str(Path(transformer).expanduser()),
        "--qwen-stages",
        "2",
        "--first-block-cache",
        case.first_block_cache,
        "--attention-backend",
        args.sol_backend if case.sol_attn else "none",
        "--sol-attn-mps-dir",
        str(Path(args.sol_attn_mps_dir).expanduser()),
        "--sol-tau",
        str(args.sol_tau),
        "--duration",
        "5.0",
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(case.sigma_points),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    if args.qwen_bits is not None:
        command.extend(["--qwen-bits", str(args.qwen_bits)])
    if args.qwen_dir is not None:
        command.extend(["--qwen-dir", str(Path(args.qwen_dir).expanduser())])
    if case.turbo:
        command.extend(["--lora", str(Path(args.lora).expanduser()), "--lora-strength", "1.0"])
    return command


def run_case(args, case: SmokeCase, output_dir: Path) -> dict[str, object]:
    output = output_dir / f"{case.name}.mp4"
    result_path = output_dir / f"{case.name}.smoke.json"
    if args.resume and result_path.exists():
        with open(result_path) as file:
            previous = json.load(file)
        if previous.get("returncode") == 0:
            return previous

    command = _command(args, case, output)
    if args.dry_run:
        return {"case": asdict(case), "command": command}

    log_path = output_dir / f"{case.name}.log"
    swap_before = _swap_used_bytes()
    state = {
        "peak_rss_bytes": 0,
        "peak_footprint_bytes": 0,
        "peak_swap_bytes": swap_before,
    }
    started = time.perf_counter()
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        monitor = threading.Thread(target=_monitor, args=(process, state, args.sample_interval))
        monitor.start()
        returncode = process.wait()
        monitor.join()

    pipeline_metrics = {}
    pipeline_path = output.with_suffix(".json")
    if pipeline_path.exists():
        with open(pipeline_path) as file:
            pipeline_metrics = json.load(file)
    error_tail = None
    if returncode != 0:
        lines = log_path.read_text(errors="replace").splitlines()
        error_tail = "\n".join(lines[-20:])
    result = {
        "case": asdict(case),
        "command": command,
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_gb": state["peak_rss_bytes"] / 1e9,
        "peak_footprint_gb": state["peak_footprint_bytes"] / 1e9,
        "swap_before_gb": swap_before / 1e9,
        "peak_swap_gb": state["peak_swap_bytes"] / 1e9,
        "swap_growth_gb": max(0, state["peak_swap_bytes"] - swap_before) / 1e9,
        "pipeline": pipeline_metrics,
        "output_bytes": output.stat().st_size if output.exists() else 0,
        "error_tail": error_tail,
        "log": str(log_path),
        "output": str(output),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=str(root / "ComfyUI" / ".venv" / "bin" / "python3"))
    parser.add_argument("--checkpoint", default=str(root / "ckpt" / "MiniMax-H3" / "FL2VA"))
    parser.add_argument(
        "--bf16-transformer",
        default=str(root / "ckpt" / "MiniMax-H3" / "FL2VA" / "transformer"),
    )
    parser.add_argument(
        "--int8-transformer",
        default=str(root / "ckpt" / "MiniMax-H3-MLX-8bit"),
    )
    parser.add_argument(
        "--lora",
        default=str(root / "ckpt" / "MiniMax-H3-Turbo-Lora" / "minimax_h3_turbo_v4_step600_ema.safetensors"),
    )
    parser.add_argument("--sol-attn-mps-dir", default=str(repo_dir / "sol_attn_mps"))
    parser.add_argument("--output-dir", default=str(root / "smoke-results"))
    parser.add_argument("--prompt", default="Cinematic tracking shot of a red sports car driving through a rain-soaked neon city at night, synchronized engine audio.")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--qwen-bits", type=int, choices=[8], default=None)
    parser.add_argument("--qwen-dir", default=None)
    parser.add_argument("--sol-tau", type=float, default=1.3)
    parser.add_argument(
        "--sol-backend",
        choices=["torch-mps"],
        default="torch-mps",
    )
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = build_cases()
    if args.only:
        wanted = set(args.only)
        cases = [case for case in cases if case.name in wanted]
        missing = wanted - {case.name for case in cases}
        if missing:
            parser.error(f"unknown cases: {sorted(missing)}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        result = run_case(args, case, output_dir)
        results.append(result)
        if not args.dry_run and result.get("returncode") != 0:
            print(f"  failed; see {result['log']}", file=sys.stderr, flush=True)

    summary = {
        "configuration": {
            "duration_seconds": 5.0,
            "qwen": (
                "prequantized 8-bit 25+25 staged"
                if args.qwen_dir is not None
                else (
                    f"{args.qwen_bits}-bit 25+25 staged"
                    if args.qwen_bits is not None
                    else "BF16 25+25 staged"
                )
            ),
            "dit_variants": {
                "bf16": "pruned BF16 core with rank-8 curve AdaLN; BF16 activations/cache",
                "8bit": "8-bit weight-only core/AdaLN; BF16 activations/cache",
            },
            "sol_tau": args.sol_tau,
            "fbc_preset": "safe",
            "seed": args.seed,
            "prompt": args.prompt,
        },
        "results": results,
    }
    (output_dir / "smoke-matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if args.dry_run or all(result.get("returncode") == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())