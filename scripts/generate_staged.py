"""Generate MiniMax-H3 video and audio with strict component staging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.attention_backends import create_attention_backend
from minimax_h3_mlx.media import save_frames, save_mp4, save_wav
from minimax_h3_mlx.staged import StrictStagedTextToVideo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--qwen-dir", default=None)
    parser.add_argument("--qwen-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--qwen-stages", choices=["auto", "1", "2"], default="2")
    parser.add_argument(
        "--first-block-cache",
        choices=["none", "safe", "fast", "aggressive"],
        default="none",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["none", "mlx-reference", "torch-mps"],
        default="none",
    )
    parser.add_argument("--sol-attn-mps-dir", default=None)
    parser.add_argument("--sol-tau", type=float, default=1.3)
    parser.add_argument("--sol-start-percent", type=float, default=0.2)
    parser.add_argument("--sol-end-percent", type=float, default=0.9)
    parser.add_argument("--sol-min-tokens", type=int, default=4096)
    parser.add_argument("--no-sol-conditioning-sink", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-adaln", action="store_true")
    args = parser.parse_args()

    attention_backend = create_attention_backend(
        args.attention_backend,
        tau=args.sol_tau,
        start_percent=args.sol_start_percent,
        end_percent=args.sol_end_percent,
        min_tokens=args.sol_min_tokens,
        sink_conditioning_rows=not args.no_sol_conditioning_sink,
        sol_attn_mps_dir=args.sol_attn_mps_dir,
    )
    runner = StrictStagedTextToVideo(
        args.checkpoint,
        args.transformer,
        lora_path=args.lora,
        lora_strength=args.lora_strength,
        qwen_dir=args.qwen_dir,
        qwen_bits=args.qwen_bits,
        qwen_stages=args.qwen_stages if args.qwen_stages == "auto" else int(args.qwen_stages),
        first_block_cache=args.first_block_cache,
        attention_backend=attention_backend,
    )
    result = runner(
        args.prompt,
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
        drop_adaln=not args.keep_adaln,
    )

    output = Path(args.output)
    try:
        save_mp4(output, result.video, result.fps, result.audio, result.sample_rate)
    except RuntimeError:
        save_frames(output.with_suffix(""), result.video)
        save_wav(output.with_suffix(".wav"), result.audio, result.sample_rate)

    metrics = {
        "width": int(result.video.shape[2]),
        "height": int(result.video.shape[1]),
        "frames": int(result.video.shape[0]),
        "fps": result.fps,
        "seconds_per_step": result.seconds_per_step,
        "total_seconds": result.total_seconds,
        "stage_metrics": result.stage_metrics,
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())