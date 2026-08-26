"""Fragment-order weight layout for r4d_gemm_mxfp4a8_nt_m64.

The kernel reads the weight pre-permuted so a wave's 32 lanes take 128 contiguous bytes per
(n-tile, k-step); in checkpoint order those lanes would straddle sixteen rows K/2 bytes apart.
This is the reference implementation of that reordering -- a serving stack does it once at
weight load, the same way the int4 path does.
"""
import torch


def permute_w(packed, N, K):
    """Checkpoint-order [N, K/2] uint8 -> fragment order the kernel reads.

    Slot l of tile (nt, ks) is the four bytes W[nt][l&15][ks][4*(l>>4) ..+4], i.e. lane l's eight
    elements: row 16*nt + (l&15), k starting at 16*ks + 8*(l>>4). One uint32 per lane, 32 lanes per
    (n-tile, k-step), so a wave reads 128 contiguous bytes.
    """
    nt, ks = N // 16, K // 16
    w = packed.reshape(nt, 16, ks, 8)
    out = torch.empty(nt, ks, 32, 4, dtype=torch.uint8)
    for l in range(32):
        r, h = l & 15, l >> 4
        out[:, :, l, :] = w[:, r, :, 4 * h:4 * h + 4]
    return out.reshape(-1).contiguous()
