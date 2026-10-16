"""float32 dequantization of compressed-tensors `pack-quantized` int4 linears, for the fp32 truth reference (ref.py `dtype=float32`).

`awq_dequant.dequant_pack_quantized` returns `bf16(W)` with `W = (q - zp) * scale` (compressed-tensors rounds the product to the
scale's dtype). The product of a 4-bit integer difference (|q - zp| <= 15) and a bf16 scale is exact in fp32, so here

    round_to_bf16=False   returns W itself (fp32, unrounded: the AWQ weight as the quantizer defined it)
    round_to_bf16=True    returns float32(bf16(W)): exactly the weight `dequant_pack_quantized` gives, widened

Same validation, packing and nibble order as awq_dequant.py (which stays the single source of the bf16 weight).
"""
from __future__ import annotations

import torch

from qwen4exp.awq_dequant import GROUP_SIZE, PACK_FACTOR, unpack_nibbles_dim1


def dequant_pack_quantized_f32(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_shape: torch.Tensor | tuple[int, int],
    group_size: int = GROUP_SIZE,
    round_to_bf16: bool = False,
) -> torch.Tensor:
    """fp32 [N, K] weight of one pack-quantized int4 asymmetric group linear."""
    n, k = (int(v) for v in weight_shape)
    if weight_scale.dtype != torch.bfloat16:
        raise ValueError(f"weight_scale must be bf16, got {weight_scale.dtype}")
    if k % group_size != 0:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    groups = k // group_size
    if tuple(weight_scale.shape) != (n, groups):
        raise ValueError(f"weight_scale shape {tuple(weight_scale.shape)} != {(n, groups)}")
    if tuple(weight_packed.shape) != (n, -(-k // PACK_FACTOR)):
        raise ValueError(f"weight_packed shape {tuple(weight_packed.shape)} != {(n, -(-k // PACK_FACTOR))}")
    if tuple(weight_zero_point.shape) != (-(-n // PACK_FACTOR), groups):
        raise ValueError(f"weight_zero_point shape {tuple(weight_zero_point.shape)} != {(-(-n // PACK_FACTOR), groups)}")
    q = unpack_nibbles_dim1(weight_packed, k)
    zp = unpack_nibbles_dim1(weight_zero_point.t().contiguous(), n).t()
    diff = q.view(n, groups, group_size) - zp.unsqueeze(-1)
    w = diff.to(torch.float32) * weight_scale.to(torch.float32).unsqueeze(-1)
    if round_to_bf16:
        w = w.to(torch.bfloat16).to(torch.float32)
    return w.view(n, k)
