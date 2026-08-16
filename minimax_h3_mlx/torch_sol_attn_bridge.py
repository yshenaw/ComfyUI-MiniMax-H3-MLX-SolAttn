"""Zero-copy MLX-to-PyTorch MPS bridge for the existing SolAttn Metal kernel."""

from __future__ import annotations

import math

import mlx.core as mx


class TorchMPSSolAttnBridge:
    def __init__(
        self,
        kernel,
        *,
        tau: float = 1.3,
        start_percent: float = 0.2,
        end_percent: float = 0.9,
        min_tokens: int = 4096,
        sink_conditioning_rows: bool = True,
    ):
        self.kernel = kernel
        self.tau = float(tau)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.min_tokens = int(min_tokens)
        self.sink_conditioning_rows = bool(sink_conditioning_rows)
        self.sink_blocks = (0, 0)
        self.sink_q = (0, 0)
        self.current_percent = 0.0
        self.sparse_calls = 0
        self.dense_calls = 0

    def configure_layout(self, layout) -> None:
        video_start = int(layout.video_indices[0].item())
        sink_end = math.ceil(video_start / 64)
        self.sink_blocks = (0, sink_end)
        self.sink_q = (0, sink_end) if self.sink_conditioning_rows else (0, 0)

    def begin_step(self, step: int, total_steps: int) -> None:
        self.current_percent = step / max(total_steps - 1, 1)

    def __call__(self, q, k, v, *, scale: float, mask=None):
        if (
            mask is not None
            or q.shape[2] < self.min_tokens
            or not self.start_percent <= self.current_percent <= self.end_percent
        ):
            self.dense_calls += 1
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        import torch

        mx.eval(q, k, v)
        torch_q = torch.utils.dlpack.from_dlpack(q).transpose(1, 2)
        torch_k = torch.utils.dlpack.from_dlpack(k).transpose(1, 2)
        torch_v = torch.utils.dlpack.from_dlpack(v).transpose(1, 2)
        torch_output = self.kernel(
            torch_q,
            torch_k,
            torch_v,
            scale=scale,
            tau=self.tau,
            sink_blocks=self.sink_blocks,
            sink_q=self.sink_q,
        )
        torch.mps.synchronize()
        output = mx.from_dlpack(torch_output)
        self.sparse_calls += 1
        return output.transpose(0, 2, 1, 3)

    def summary(self) -> dict[str, object]:
        return {
            "backend": "torch_mps_dlpack_bridge",
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "tau": self.tau,
            "sink_blocks": list(self.sink_blocks),
            "sink_q": list(self.sink_q),
        }


__all__ = ["TorchMPSSolAttnBridge"]