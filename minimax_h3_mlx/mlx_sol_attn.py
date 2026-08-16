"""BF16-native MLX reference for Sol-Attn.

This implementation is intentionally small and explicit. It is the numerical
oracle for the fused Metal backend, not the production path for long H3
sequences. Inputs and outputs use MLX's native BHSD attention layout.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


BLOCK_SIZE = 64


def _block_summaries(x: mx.array) -> tuple[mx.array, mx.array]:
    batch, heads, tokens, head_dim = x.shape
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_tokens = blocks * BLOCK_SIZE
    if padded_tokens != tokens:
        x = mx.pad(x, ((0, 0), (0, 0), (0, padded_tokens - tokens), (0, 0)))
    blocked = x.reshape(batch, heads, blocks, BLOCK_SIZE, head_dim)
    lengths = mx.full((blocks,), BLOCK_SIZE, dtype=mx.int32)
    if tokens % BLOCK_SIZE:
        lengths[-1] = tokens % BLOCK_SIZE
    sums = mx.sum(blocked.astype(mx.float32), axis=3)
    return sums, lengths


def _routing_thresholds(
    q_centroids: mx.array,
    k_centroids: mx.array,
    scale: float,
    tau: float,
) -> mx.array:
    log2_scale = float(scale) * math.log2(math.e)
    k_mean = mx.mean(k_centroids, axis=2)
    k_variance = mx.mean(mx.square(k_centroids - k_mean[:, :, None]), axis=2)
    threshold_mean = mx.sum(q_centroids * k_mean[:, :, None], axis=-1) * log2_scale
    threshold_variance = (
        mx.sum(mx.square(q_centroids) * k_variance[:, :, None], axis=-1)
        * (log2_scale * log2_scale)
    )
    return threshold_mean + float(tau) * mx.sqrt(mx.maximum(threshold_variance, 0.0) + 1.0e-6)


def _route_blocks(
    q_centroids: mx.array,
    k_centroids: mx.array,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
) -> mx.array:
    blocks = q_centroids.shape[2]
    thresholds = _routing_thresholds(q_centroids, k_centroids, scale, tau)
    route_scores = mx.matmul(q_centroids, mx.swapaxes(k_centroids, -2, -1))
    route_scores = route_scores * (float(scale) * math.log2(math.e))
    routed = route_scores > thresholds[..., None]

    indices = mx.arange(blocks, dtype=mx.int32)
    neighbors = mx.abs(indices[:, None] - indices[None, :]) <= 1
    sink_start = max(0, min(int(sink_blocks[0]), blocks))
    sink_end = max(sink_start, min(int(sink_blocks[1]), blocks))
    sink_q_start = max(0, min(int(sink_q[0]), blocks))
    sink_q_end = max(sink_q_start, min(int(sink_q[1]), blocks))
    exact_keys = (indices >= sink_start) & (indices < sink_end)
    dense_queries = (indices >= sink_q_start) & (indices < sink_q_end)
    return (
        routed
        | neighbors[None, None]
        | exact_keys[None, None, None, :]
        | dense_queries[None, None, :, None]
    )


def sol_attn_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> mx.array:
    """Apply Sol-Attn to BHSD arrays using FP32 score and output accumulation."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same BHSD shape")
    if q.shape[2] == 0:
        return mx.zeros_like(q)

    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    q_sums, lengths = _block_summaries(q)
    k_sums, _ = _block_summaries(k)
    v_sums, _ = _block_summaries(v)
    lengths_float = lengths.astype(mx.float32)
    q_centroids = q_sums / lengths_float[None, None, :, None]
    k_centroids = k_sums / lengths_float[None, None, :, None]
    v_centroids = v_sums / lengths_float[None, None, :, None]
    routed = _route_blocks(
        q_centroids, k_centroids, scale, float(tau), sink_blocks, sink_q
    )
    mx.eval(k_centroids, v_centroids, routed)
    routed_host = np.asarray(routed)

    batch_outputs = []
    tokens = q.shape[2]
    blocks = lengths.shape[0]
    for batch_index in range(q.shape[0]):
        head_outputs = []
        for head_index in range(q.shape[1]):
            query_outputs = []
            for query_block in range(blocks):
                query_start = query_block * BLOCK_SIZE
                query_end = min(query_start + BLOCK_SIZE, tokens)
                candidate_keys = []
                candidate_values = []
                candidate_log_weights = []
                for key_block in range(blocks):
                    key_start = key_block * BLOCK_SIZE
                    key_end = min(key_start + BLOCK_SIZE, tokens)
                    if routed_host[batch_index, head_index, query_block, key_block]:
                        keys = k[batch_index, head_index, key_start:key_end].astype(mx.float32)
                        values = v[batch_index, head_index, key_start:key_end].astype(mx.float32)
                        log_weights = mx.zeros((key_end - key_start,), dtype=mx.float32)
                    else:
                        keys = k_centroids[batch_index, head_index, key_block : key_block + 1]
                        values = v_centroids[batch_index, head_index, key_block : key_block + 1]
                        log_weights = mx.full(
                            (1,), math.log(int(lengths[key_block].item())), dtype=mx.float32
                        )
                    candidate_keys.append(keys)
                    candidate_values.append(values)
                    candidate_log_weights.append(log_weights)

                keys = mx.concatenate(candidate_keys, axis=0)
                values = mx.concatenate(candidate_values, axis=0)
                log_weights = mx.concatenate(candidate_log_weights, axis=0)
                query = q[batch_index, head_index, query_start:query_end].astype(mx.float32)
                scores = mx.matmul(query, mx.swapaxes(keys, -2, -1)) * scale
                probabilities = mx.softmax(scores + log_weights[None], axis=-1)
                query_outputs.append(mx.matmul(probabilities, values).astype(q.dtype))
            head_outputs.append(mx.concatenate(query_outputs, axis=0))
        batch_outputs.append(mx.stack(head_outputs, axis=0))
    return mx.stack(batch_outputs, axis=0)


class MLXSolAttnReference:
    """DiT backend wrapper with dense fallbacks and runtime statistics."""

    def __init__(
        self,
        *,
        tau: float = 1.3,
        start_percent: float = 0.2,
        end_percent: float = 0.9,
        min_tokens: int = 4096,
        max_reference_tokens: int = 4096,
        sink_conditioning_rows: bool = True,
    ):
        self.tau = float(tau)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.min_tokens = int(min_tokens)
        self.max_reference_tokens = int(max_reference_tokens)
        self.sink_conditioning_rows = bool(sink_conditioning_rows)
        self.sink_blocks = (0, 0)
        self.sink_q = (0, 0)
        self.current_percent = 0.0
        self.sparse_calls = 0
        self.dense_calls = 0

    def configure_layout(self, layout) -> None:
        video_start = int(layout.video_indices[0].item())
        sink_end = math.ceil(video_start / BLOCK_SIZE)
        self.sink_blocks = (0, sink_end)
        self.sink_q = (0, sink_end) if self.sink_conditioning_rows else (0, 0)

    def begin_step(self, step: int, total_steps: int) -> None:
        self.current_percent = step / max(total_steps - 1, 1)

    def __call__(self, q, k, v, *, scale: float, mask=None):
        use_sparse = (
            mask is None
            and self.min_tokens <= q.shape[2] <= self.max_reference_tokens
            and self.start_percent <= self.current_percent <= self.end_percent
        )
        if not use_sparse:
            self.dense_calls += 1
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.sparse_calls += 1
        return sol_attn_reference(
            q,
            k,
            v,
            scale=scale,
            tau=self.tau,
            sink_blocks=self.sink_blocks,
            sink_q=self.sink_q,
        )

    def summary(self) -> dict[str, object]:
        return {
            "backend": "mlx_reference",
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "tau": self.tau,
            "sink_blocks": list(self.sink_blocks),
            "sink_q": list(self.sink_q),
        }


__all__ = ["MLXSolAttnReference", "sol_attn_reference"]