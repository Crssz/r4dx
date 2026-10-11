"""Dequantize compressed-tensors `pack-quantized` W4A16 linears to BF16 (plan M0-4, Python half).

The checkpoint's quantization_config: int, num_bits 4, group_size 128, strategy group, asymmetric,
zp_dtype int8. Per quantized linear `<prefix>` a shard holds

    <prefix>.weight_packed      int32 [N, ceil(K/8)]   8 nibbles per int32 along K (dim 1)
    <prefix>.weight_scale       bf16  [N, K/128]
    <prefix>.weight_zero_point  int32 [ceil(N/8), K/128]  8 nibbles per int32 along N (dim 0)
    <prefix>.weight_shape       int64 [2] = (N, K)

Packing (compressed_tensors/compressors/pack_quantized/helpers.py, pack_to_int32): a signed int4 value
v in [-8, 7] is stored as the unsigned nibble u = v + 8; element j of each run of 8 goes to bits
[4j, 4j+4), i.e. q[n, k] is nibble (k % 8) of weight_packed[n, k // 8] and zp[n, g] is nibble (n % 8)
of weight_zero_point[n // 8, g]. Padding nibbles past N / K are dropped.

Dequantization (compressed_tensors/quantization/lifecycle/forward_helpers.py, _dequantize):
    W = (q.to(bf16) - zp.to(bf16)) * scale          (all in bf16, the scale's dtype)
The +8 offsets cancel, the subtraction is exact in bf16 (|q - zp| <= 15), and the product of a
4-bit integer with a bf16 scale is exact in fp32, so this equals one round-to-nearest-even of the fp32
product `(u_q - u_zp) * float(scale)` to bf16. That is what `dequant_pack_quantized` computes (the
C++ dequant of M1-2 MUST do the same).

CPU only; pure torch.
"""
from __future__ import annotations

import os
from typing import Iterator

import torch

NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS  # 8 nibbles per int32
GROUP_SIZE = 128
PACKED_SUFFIX = ".weight_packed"


def unpack_nibbles_dim1(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """int32 [R, C] -> unsigned nibbles int16 [R, cols]; nibble j of packed[r, c] is column 8c + j."""
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError(f"expected a 2D int32 tensor, got {packed.dtype} {tuple(packed.shape)}")
    if cols > packed.shape[1] * PACK_FACTOR:
        raise ValueError(f"{cols} columns do not fit in {packed.shape[1]} packed int32 words")
    shifts = torch.arange(0, 32, NUM_BITS, dtype=torch.int32)
    nib = (packed.unsqueeze(-1) >> shifts) & ((1 << NUM_BITS) - 1)  # [R, C, 8]
    return nib.reshape(packed.shape[0], -1)[:, :cols].to(torch.int16)


def dequant_pack_quantized(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    weight_shape: torch.Tensor | tuple[int, int],
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Return the bf16 [N, K] weight of one pack-quantized int4 asymmetric group linear."""
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
        raise ValueError(
            f"weight_zero_point shape {tuple(weight_zero_point.shape)} != {(-(-n // PACK_FACTOR), groups)}")

    q = unpack_nibbles_dim1(weight_packed, k)                       # [N, K]
    zp = unpack_nibbles_dim1(weight_zero_point.t().contiguous(), n).t()  # [N, G]
    diff = q.view(n, groups, group_size) - zp.unsqueeze(-1)         # exact, |diff| <= 15
    w = diff.to(torch.float32) * weight_scale.to(torch.float32).unsqueeze(-1)  # exact in fp32
    return w.to(torch.bfloat16).view(n, k)


def iter_dequant_shard(path: str | os.PathLike) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (`<prefix>.weight`, bf16 [N, K]) for every `<prefix>.weight_packed` in a safetensors shard.

    All four tensors of a linear must be in the same shard (true for compressed-tensors saves).
    """
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key in sorted(k for k in keys if k.endswith(PACKED_SUFFIX)):
            prefix = key[: -len(PACKED_SUFFIX)]
            parts = {s: f"{prefix}.{s}" for s in ("weight_scale", "weight_zero_point", "weight_shape")}
            missing = [name for name in parts.values() if name not in keys]
            if missing:
                raise KeyError(f"{path}: {key} lacks {missing}")
            yield f"{prefix}.weight", dequant_pack_quantized(
                f.get_tensor(key),
                f.get_tensor(parts["weight_scale"]),
                f.get_tensor(parts["weight_zero_point"]),
                f.get_tensor(parts["weight_shape"]).tolist(),
            )
