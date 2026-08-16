"""Cross-step first-block residual caching for the MLX MiniMax-H3 DiT."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class FirstBlockCacheConfig:
    threshold: float = 0.10
    start_percent: float = 0.10
    end_percent: float = 0.95
    max_consecutive_hits: int = 2


class FirstBlockCache:
    """Reuse the residual produced by blocks 1-49 when block 0 changes little."""

    def __init__(self, config: FirstBlockCacheConfig | None = None):
        self.config = config or FirstBlockCacheConfig()
        self.reset()

    def reset(self) -> None:
        self.previous_first_residual = None
        self.remaining_blocks_residual = None
        self.pending_first_residual = None
        self.pending_first_output = None
        self.current_step = 0
        self.total_steps = 0
        self.consecutive_hits = 0
        self.full_steps = 0
        self.cached_steps = 0
        self.cached_step_numbers = []
        self.diff_values = []

    def begin_step(self, step: int, total_steps: int) -> None:
        self.current_step = int(step)
        self.total_steps = int(total_steps)
        self.pending_first_residual = None
        self.pending_first_output = None

    def _within_window(self) -> bool:
        if self.total_steps <= 1:
            return False
        percent = self.current_step / (self.total_steps - 1)
        return self.config.start_percent <= percent <= self.config.end_percent

    def after_first_block(self, block_input: mx.array, first_output: mx.array) -> mx.array | None:
        first_residual = first_output - block_input
        mx.eval(first_residual, first_output)
        previous = self.previous_first_residual
        tail = self.remaining_blocks_residual
        use_cache = False
        if (
            previous is not None
            and tail is not None
            and previous.shape == first_residual.shape
            and tail.shape == first_output.shape
            and self._within_window()
        ):
            numerator = mx.mean(mx.abs(first_residual - previous))
            denominator = mx.maximum(mx.mean(mx.abs(previous)), mx.array(1.0e-8))
            diff = float((numerator / denominator).item())
            self.diff_values.append(diff)
            use_cache = (
                math.isfinite(diff)
                and diff <= self.config.threshold
                and self.consecutive_hits < self.config.max_consecutive_hits
            )

        if use_cache:
            self.cached_steps += 1
            self.cached_step_numbers.append(self.current_step + 1)
            self.consecutive_hits += 1
            output = first_output + tail
            mx.eval(output)
            return output

        self.consecutive_hits = 0
        self.pending_first_residual = first_residual
        self.pending_first_output = first_output
        return None

    def finish_full_step(self, output: mx.array) -> None:
        if self.pending_first_residual is None or self.pending_first_output is None:
            raise RuntimeError("FirstBlockCache full-step state is incomplete")
        tail = output - self.pending_first_output
        mx.eval(tail, self.pending_first_residual)
        self.remaining_blocks_residual = tail
        self.previous_first_residual = self.pending_first_residual
        self.pending_first_residual = None
        self.pending_first_output = None
        self.full_steps += 1

    def summary(self) -> dict[str, object]:
        total = self.full_steps + self.cached_steps
        block_count = 50
        executed_blocks = self.full_steps * block_count + self.cached_steps
        return {
            "full_steps": self.full_steps,
            "cached_steps": self.cached_steps,
            "total_steps": total,
            "cached_step_numbers": list(self.cached_step_numbers),
            "estimated_block_speedup": (
                total * block_count / executed_blocks if executed_blocks else 1.0
            ),
            "diff_min": min(self.diff_values) if self.diff_values else None,
            "diff_max": max(self.diff_values) if self.diff_values else None,
        }


PRESETS = {
    "safe": FirstBlockCacheConfig(threshold=0.08),
    "fast": FirstBlockCacheConfig(threshold=0.10),
    "aggressive": FirstBlockCacheConfig(threshold=0.12),
}


__all__ = ["FirstBlockCache", "FirstBlockCacheConfig", "PRESETS"]