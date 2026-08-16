"""Fused Metal forward for the MPS Sol-Attn backend."""

from __future__ import annotations

import os

import torch

from ._metal_tiled_fwd import sol_attn_tiled_mps
from ._torch_fwd import _prepare_compact, sol_attn_torch


_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void sol_attn_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device const float* kc [[buffer(3)]],
    device const float* vc [[buffer(4)]],
    device const uchar* routes [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& blocks [[buffer(11)]],
    constant ulong& sq_b [[buffer(12)]],
    constant ulong& sq_t [[buffer(13)]],
    constant ulong& sq_h [[buffer(14)]],
    constant ulong& sk_b [[buffer(15)]],
    constant ulong& sk_t [[buffer(16)]],
    constant ulong& sk_h [[buffer(17)]],
    constant ulong& sv_b [[buffer(18)]],
    constant ulong& sv_t [[buffer(19)]],
    constant ulong& sv_h [[buffer(20)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float partials[128];

    const uint token = group % tokens;
    const uint batch_head = group / tokens;
    const uint head = batch_head % heads;
    const uint batch = batch_head / heads;
    const uint query_block = token / 64;
    const ulong q_base = batch * sq_b + token * sq_t + head * sq_h;
    const float query_value = lane < head_dim ? float(q[q_base + lane]) : 0.0f;

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator = 0.0f;

    for (uint key_block = 0; key_block < blocks; ++key_block) {
        const uint key_start = key_block * 64;
        const uint block_length = min(64u, tokens - key_start);
        const ulong route_offset =
            ((ulong(batch) * heads + head) * blocks + query_block) * blocks + key_block;

        if (routes[route_offset] != 0) {
            for (uint key_offset = 0; key_offset < block_length; ++key_offset) {
                const uint key_token = key_start + key_offset;
                const ulong k_base = batch * sk_b + key_token * sk_t + head * sk_h;
                partials[lane] = lane < head_dim
                    ? query_value * float(k[k_base + lane])
                    : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = lanes >> 1; stride > 0; stride >>= 1) {
                    if (lane < stride) partials[lane] += partials[lane + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                const float score = partials[0] * scale;
                const float next_maximum = max(maximum, score);
                const float alpha = exp(maximum - next_maximum);
                const float probability = exp(score - next_maximum);
                if (lane < head_dim) {
                    const ulong v_offset =
                        batch * sv_b + key_token * sv_t + head * sv_h + lane;
                    accumulator = accumulator * alpha + probability * float(v[v_offset]);
                }
                denominator = denominator * alpha + probability;
                maximum = next_maximum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            const ulong summary_offset =
                ((ulong(batch) * heads + head) * blocks + key_block) * head_dim + lane;
            partials[lane] = lane < head_dim ? query_value * kc[summary_offset] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = lanes >> 1; stride > 0; stride >>= 1) {
                if (lane < stride) partials[lane] += partials[lane + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float score = partials[0] * scale + log(float(block_length));
            const float next_maximum = max(maximum, score);
            const float alpha = exp(maximum - next_maximum);
            const float probability = exp(score - next_maximum);
            if (lane < head_dim) {
                accumulator = accumulator * alpha
                    + probability * (vc[summary_offset] / float(block_length));
            }
            denominator = denominator * alpha + probability;
            maximum = next_maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < head_dim) {
        const ulong output_offset =
            ((ulong(batch) * tokens + token) * heads + head) * head_dim + lane;
        output[output_offset] = half(accumulator / denominator);
    }
}
"""


_library = None
_MPS_QUERY_BLOCK_SIZE = int(os.environ.get("SOL_ATTN_MPS_BQ", "64"))
if _MPS_QUERY_BLOCK_SIZE not in (32, 64):
    raise ValueError("SOL_ATTN_MPS_BQ must be 32 or 64")


def _get_library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def metal_supported(q: torch.Tensor) -> bool:
    return hasattr(torch.mps, "compile_shader") and (
        (q.dtype == torch.float16 and q.shape[-1] <= 128)
        or (q.dtype == torch.bfloat16 and q.shape[-1] in (64, 128))
    )


def sol_attn_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
    **kwargs,
) -> torch.Tensor:
    """Run the fused Metal path when supported, otherwise use portable PyTorch."""
    if not metal_supported(q):
        return sol_attn_torch(
            q, k, v, scale=scale, tau=tau,
            sink_blocks=sink_blocks, sink_q=sink_q, **kwargs,
        )
    if q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] in (64, 128):
        return sol_attn_tiled_mps(
            q, k, v, scale=scale, tau=tau,
            sink_blocks=sink_blocks, sink_q=sink_q,
            query_block_size=_MPS_QUERY_BLOCK_SIZE,
        )
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same BTHD shape")
    if q.shape[1] == 0:
        return torch.empty_like(q)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, tokens, heads, head_dim = q.shape
    _, k_centroids, v_sums, routed = _prepare_compact(
        q, k, v, scale, float(tau), sink_blocks, sink_q
    )
    k_centroids = k_centroids.contiguous()
    v_sums = v_sums.contiguous()
    routes = routed.to(torch.uint8).contiguous()
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    group_size = 1 << (head_dim - 1).bit_length()
    groups = batch * heads * tokens

    _get_library().sol_attn_f16(
        q, k, v, k_centroids, v_sums, routes, output,
        scale, tokens, heads, head_dim, routed.shape[-1],
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        threads=groups * group_size,
        group_size=group_size,
        arg_casts={8: "int32", 9: "int32", 10: "int32", 11: "int32"},
    )
    return output


__all__ = ["metal_supported", "sol_attn_mps"]