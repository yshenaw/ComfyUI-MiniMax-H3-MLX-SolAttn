"""Portable PyTorch implementation of the Sol-Attn forward pass.

This backend is intended for MPS. It keeps preprocessing, routing, and the
sparse gather on-device and mirrors the block-summary approximation used by
the Triton kernels.
"""

from __future__ import annotations

import math

import torch


BLOCK = 64


def _routing_thresholds(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    scale: float,
    tau: float,
) -> torch.Tensor:
    log2_scale = scale * math.log2(math.e)
    k_mean = k_centroids.mean(dim=2)
    k_variance = (k_centroids - k_mean.unsqueeze(2)).square().mean(dim=2)
    threshold_mean = (q_centroids * k_mean.unsqueeze(2)).sum(dim=-1) * log2_scale
    threshold_variance = (
        q_centroids.square() * k_variance.unsqueeze(2)
    ).sum(dim=-1) * (log2_scale * log2_scale)
    return threshold_mean + tau * torch.sqrt(
        torch.clamp_min(threshold_variance, 0.0) + 1.0e-6
    )


def _route_blocks(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    block_count = q_centroids.shape[2]
    log2_scale = scale * math.log2(math.e)
    route_scores = torch.matmul(
        q_centroids, k_centroids.transpose(-2, -1)
    ) * log2_scale
    if thresholds is None:
        thresholds = _routing_thresholds(q_centroids, k_centroids, scale, tau)
    routed = route_scores > thresholds.unsqueeze(-1)

    block_indices = torch.arange(block_count, device=q_centroids.device)
    neighbors = (
        block_indices[:, None] - block_indices[None, :]
    ).abs() <= 1
    routed |= neighbors[None, None]

    sink_start = max(0, min(int(sink_blocks[0]), block_count))
    sink_end = max(sink_start, min(int(sink_blocks[1]), block_count))
    if sink_start < sink_end:
        routed[..., sink_start:sink_end] = True
    sink_q_start = max(0, min(int(sink_q[0]), block_count))
    sink_q_end = max(sink_q_start, min(int(sink_q[1]), block_count))
    if sink_q_start < sink_q_end:
        routed[:, :, sink_q_start:sink_q_end, :] = True
    return routed


def _pad_blocks(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, head_dim = tensor.shape
    blocks = (tokens + BLOCK - 1) // BLOCK
    padded_tokens = blocks * BLOCK
    if padded_tokens != tokens:
        padding = tensor.new_zeros((batch, padded_tokens - tokens, heads, head_dim))
        tensor = torch.cat((tensor, padding), dim=1)
    lengths = torch.full((blocks,), BLOCK, device=tensor.device, dtype=torch.long)
    lengths[-1] = tokens - (blocks - 1) * BLOCK
    blocked = tensor.permute(0, 2, 1, 3).reshape(batch, heads, blocks, BLOCK, head_dim)
    return blocked, lengths


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
):
    q_blocks, lengths = _pad_blocks(q)
    k_blocks, _ = _pad_blocks(k)
    v_blocks, _ = _pad_blocks(v)
    block_count = lengths.numel()

    lengths_float = lengths.to(torch.float32)
    q_centroids = q_blocks.float().sum(dim=3) / lengths_float[None, None, :, None]
    k_centroids = k_blocks.float().sum(dim=3) / lengths_float[None, None, :, None]
    v_sums = v_blocks.float().sum(dim=3)

    routed = _route_blocks(
        q_centroids, k_centroids, scale, tau, sink_blocks, sink_q
    )

    return k_blocks, v_blocks, lengths, k_centroids, v_sums, routed


def _block_sums(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, head_dim = tensor.shape
    full_blocks, tail = divmod(tokens, BLOCK)
    summaries = []
    if full_blocks:
        blocked = tensor[:, :full_blocks * BLOCK].reshape(
            batch, full_blocks, BLOCK, heads, head_dim
        )
        summaries.append(blocked.sum(dim=2, dtype=torch.float32))
    if tail:
        summaries.append(
            tensor[:, full_blocks * BLOCK:].sum(dim=1, dtype=torch.float32).unsqueeze(1)
        )
    lengths = torch.full(
        (full_blocks + bool(tail),), BLOCK, device=tensor.device, dtype=torch.long
    )
    if tail:
        lengths[-1] = tail
    block_sums = summaries[0] if len(summaries) == 1 else torch.cat(summaries, dim=1)
    return block_sums.permute(0, 2, 1, 3), lengths


def _prepare_compact_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    tau: float,
    summary_dtype: torch.dtype | None = None,
):
    """Build linear-size Metal summaries and routing thresholds."""
    q_sums, lengths = _block_sums(q)
    k_sums, _ = _block_sums(k)
    v_sums, _ = _block_sums(v)
    lengths_float = lengths.to(torch.float32)
    q_centroids = q_sums / lengths_float[None, None, :, None]
    k_centroids = k_sums / lengths_float[None, None, :, None]
    if summary_dtype is not None:
        k_centroids = k_centroids.to(summary_dtype)
    thresholds = _routing_thresholds(
        q_centroids, k_centroids.float(), scale, tau
    )
    return lengths, q_centroids, k_centroids, v_sums, thresholds


def _prepare_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
):
    """Build Metal summaries and routes without full-size Q/K/V copies."""
    lengths, q_centroids, k_centroids, v_sums, thresholds = (
        _prepare_compact_statistics(q, k, v, scale, tau)
    )
    routed = _route_blocks(
        q_centroids, k_centroids, scale, tau, sink_blocks, sink_q,
        thresholds=thresholds,
    )

    return lengths, k_centroids, v_sums, routed


def sol_attn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
    **_ignored,
) -> torch.Tensor:
    """Run Sol-Attn on BTHD tensors using portable PyTorch operations."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same BTHD shape")
    if q.shape[1] == 0:
        return torch.empty_like(q)

    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    k_blocks, v_blocks, lengths, k_centroids, v_sums, routed = _prepare(
        q, k, v, scale, tau, sink_blocks, sink_q
    )
    block_count = lengths.numel()

    lengths_float = lengths.to(torch.float32)
    q_by_head = q.permute(0, 2, 1, 3)
    output_blocks = []
    token_offsets = torch.arange(BLOCK, device=q.device)
    approximate_values = v_sums / lengths_float[None, None, :, None]
    approximate_log_weights = lengths_float.log()
    summary_keys = k_centroids.to(k.dtype)
    summary_values = approximate_values.to(v.dtype)
    selected_counts = routed.sum(dim=-1).amax(dim=(0, 1)).to("cpu").tolist()

    for query_block in range(block_count):
        exact_blocks = routed[:, :, query_block]
        selected_count = int(selected_counts[query_block])
        selected = exact_blocks.float().topk(selected_count, dim=-1).indices
        selected_valid = exact_blocks.gather(-1, selected)

        gather_index = selected[..., None, None].expand(
            batch, heads, selected_count, BLOCK, head_dim
        )
        exact_keys = k_blocks.gather(2, gather_index).flatten(2, 3)
        exact_values = v_blocks.gather(2, gather_index).flatten(2, 3)
        selected_lengths = lengths[selected]
        exact_valid = (
            selected_valid[..., None] &
            (token_offsets < selected_lengths[..., None])
        ).flatten(2, 3)

        candidate_keys = torch.cat((exact_keys, summary_keys), dim=2)
        candidate_values = torch.cat((exact_values, summary_values), dim=2)
        approximate_valid = ~exact_blocks
        candidate_valid = torch.cat((exact_valid, approximate_valid), dim=-1)
        log_weights = torch.cat(
            (
                torch.zeros_like(exact_valid, dtype=torch.float32),
                approximate_log_weights[None, None].expand(batch, heads, -1),
            ),
            dim=-1,
        )

        query_start = query_block * BLOCK
        query_end = min(query_start + BLOCK, tokens)
        query = q_by_head[:, :, query_start:query_end]
        scores = torch.matmul(query, candidate_keys.transpose(-2, -1)).float() * scale
        scores = scores + log_weights.unsqueeze(-2)
        scores = scores.masked_fill(~candidate_valid.unsqueeze(-2), -torch.inf)
        probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
        output_blocks.append(torch.matmul(probabilities, candidate_values))

    return torch.cat(output_blocks, dim=2).permute(0, 2, 1, 3)


__all__ = ["sol_attn_torch"]