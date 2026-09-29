"""tools/reference/trellis_golden.py -- CPU-only golden vectors for the trellis kernels.

The spec is docs/trellis-kernel.md: section 6 ("Golden vectors") says what is written, 2.1 defines
the pair grid, 4.8 the input transform and 4.5 the epilogue whose arithmetic the references follow.
It uses numpy and trellis_quant.py's CPU functions (codebook_np, decode_words, reconstruct,
load_manifest, tensor_file). decode_words is torch code and runs on torch's CPU device; torch.cuda is
never touched (checked at exit).

Output, tests/kernels/golden/trellis/ (read in C++ through tests/kernels/npy_fixture.hpp: '<u4' ring
words, '<f2' fp16 values, '<f4' everything else):

  rand_*        random ring words at KB = 4 and 5 (K = 1024, N = 512) in the pair grid and their
                decode_words Q; random activations, suh (2 parts) and svh; the fp32-emulated input
                transform at prescale 0 and 4; fp64 full-linear references for P = 1 and P = 2
  real_*        256 x 256 blocks cut from the matched-basis KB = 4 oracle (L03 attn.k, L10 mlp.down,
                L07 mlp.gate_up as 2 parts) with their suh/svh slices, and the same transform and
                references on random activations
  manifest.json sha256, dtype and shape of every file, the parameters, the source pins (oracle
                manifest and file sha256), and the conventions below

Conventions (every array C-order, row-major):

  Q [K][N]      the regularized-domain matrix decode_words returns (in, out). The stored weight is
                W_hat = diag(suh_p) P_K Q P_N diag(svh) [K][N]; the HF weight is W_hat^T. P = the
                natural-order Sylvester Hadamard / sqrt(128), blockwise.
  w             pair-grid uint32 words, 1-D [N*K*KB/32]: word w of tile (tn, tk) is at
                (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8*KB + w (docs/trellis-kernel.md 2.1)
  words         the oracle's own layout [K/16][N/16][8*KB] (real tiles only; the regrid's input)
  suh [P][K]    fp16; column n reads part p(n) = (n >= n_split); P = 1 means n_split = N
  a [P][M][K]   A_p = f16_rn(FWHT128_fp32(fp32(x) * fp32(suh_p)) * scale), scale = float32(2^s /
                sqrt(128)) computed on the host (correctly rounded, not a device rsqrt). The product
                is rounded to fp32 before the first butterfly (no FMA contraction into stage 0: the
                kernel needs __fmul_rn or `#pragma clang fp contract(off)`); stages lg = 0..6 in
                FwhtLds order (a + b, a - b); one rounding to f16 (not bf16 first).
  y [M][N]      fp64 x @ W_hat_p(n) from the exact x (the Frobenius reference), stored as fp32
  ya [M][N]     fp64 out_scale * svh[n] * FWHT128_n(sum_k A_p(n)[m][k] Q[k][n]) from the f16 A,
                out_scale = 2^-s / sqrt(128) (the per-element reference), stored as fp32

fp64 stored as fp32 costs 2^-24 relative, 2^16 below a bf16 ulp. x holds bf16-exact values.

Usage (the reference venv's python; CPU only, no $env:HIP_VISIBLE_DEVICES needed):

  & $py tools\\reference\\trellis_golden.py               # (re)generate; needs the K4m oracle
  & $py tools\\reference\\trellis_golden.py --no-real     # random cases only
  & $py tools\\reference\\trellis_golden.py --check       # the files against manifest.json
  & $py tools\\reference\\trellis_golden.py --selftest    # this script's own math; writes nothing

The pair-grid functions (to_pair_grid, from_pair_grid, pair_grid_rows, pair_grid_cols) are the
reference for the M3 converter's regrid and M4's tensor-parallel slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import trellis_quant as tq  # noqa: E402  (imports torch; used on the CPU only)
import torch  # noqa: E402

HAD = 128
DEFAULT_OUT = REPO / "tests" / "kernels" / "golden" / "trellis"
DEFAULT_ORACLE = Path(r"D:\models\r4dx\huihui\trellis-q\K4m")
FORMAT = "r4dx-trellis-golden"
VERSION = 1
SEED = 0x7E11  # every stream is default_rng([SEED, tag]) so adding a stream moves no other
M_ROWS = 64
RAND_K, RAND_N = 1024, 512
BLOCK = 256
PRESCALES = (0, 4)
# (case, layer, HF modules = parts, k0, n0). Offsets are 128-aligned and chosen off zero: down's k0
# is its TP = 2 rank K boundary, gate_up's n0 its TP = 2 rank row boundary (docs/trellis-kernel.md 2.4).
REAL_BLOCKS = (
    ("attn_k", 3, ("self_attn.k_proj",), 2560, 512),
    ("mlp_down", 10, ("mlp.down_proj",), 8704, 2560),
    ("mlp_gate_up", 7, ("mlp.gate_proj", "mlp.up_proj"), 1280, 8704),
)


# --------------------------------------------------------------------------------------------
# The pair grid (docs/trellis-kernel.md 2.1, 2.4)
# --------------------------------------------------------------------------------------------


def _as_u32(words: np.ndarray) -> np.ndarray:
    w = np.ascontiguousarray(words)
    if w.dtype == np.int32:  # the oracle's I32 tensors hold uint32 bit patterns
        return w.view(np.uint32)
    if w.dtype != np.uint32:
        raise TypeError(f"ring words must be int32 or uint32, got {w.dtype}")
    return w


def pair_grid_index(tn: int, tk: int, w: int, K: int, KB: int) -> int:
    """uint32 index of word w of tile (tn, tk) in the pair grid (docs/trellis-kernel.md 2.1)."""
    return (((tn >> 1) * (K // 16) + tk) * 2 + (tn & 1)) * 8 * KB + w


def to_pair_grid(parts) -> np.ndarray:
    """Oracle-layout words, one array per part ([K/16][n_p/16][8KB], int32 or uint32, equal K and
    KB), concatenated along N and permuted into the pair grid: a 1-D uint32 array [N*K*KB/32]. Whole
    tiles move; the bits inside a tile do not. A pair never straddles two parts (n_p % 32 == 0)."""
    arrs = [_as_u32(p) for p in (parts if isinstance(parts, (list, tuple)) else [parts])]
    tk, nw = arrs[0].shape[0], arrs[0].shape[2]
    for a in arrs:
        if a.ndim != 3 or a.shape[0] != tk or a.shape[2] != nw or a.shape[1] % 2:
            raise ValueError(f"part shape {a.shape}: need [{tk}][even][{nw}]")
    if nw % 8:
        raise ValueError(f"{nw} words per tile is not 8*KB")
    w = np.concatenate(arrs, axis=1)                      # [tk][tn][nw]
    tn = w.shape[1]
    g = w.transpose(1, 0, 2).reshape(tn // 2, 2, tk, nw)  # [pair][tn & 1][tk][nw]
    return np.ascontiguousarray(g.transpose(0, 2, 1, 3)).reshape(-1)  # [pair][tk][tn & 1][nw]


def from_pair_grid(flat: np.ndarray, K: int, N: int, KB: int) -> np.ndarray:
    """The inverse of to_pair_grid: [N*K*KB/32] uint32 -> [K/16][N/16][8KB] uint32."""
    tk, tn, nw = K // 16, N // 16, 8 * KB
    g = _as_u32(flat).reshape(tn // 2, tk, 2, nw)
    return np.ascontiguousarray(g.transpose(1, 0, 2, 3).reshape(tk, tn, nw))


def pair_grid_rows(flat: np.ndarray, K: int, KB: int, n0: int, cnt: int) -> np.ndarray:
    """Rows [n0, n0 + cnt) (32-aligned) of a pair grid: ONE contiguous run, pair rows
    [n0/32, (n0 + cnt)/32) -- a column-parallel rank's slice (docs/trellis-kernel.md 2.4)."""
    if n0 % 32 or cnt % 32:
        raise ValueError("row slices are whole tile pairs (32-aligned)")
    per_pair = (K // 16) * 2 * 8 * KB
    return _as_u32(flat)[n0 // 32 * per_pair:(n0 + cnt) // 32 * per_pair]


def pair_grid_cols(flat: np.ndarray, K: int, N: int, KB: int, k0: int, kc: int) -> np.ndarray:
    """Columns [k0, k0 + kc) (16-aligned) of a pair grid: per pair row pr, bytes
    [(pr*K/16 + k0/16)*64*KB, +kc/16*64*KB), N/32 runs -- a row-parallel rank's slice (2.4)."""
    if k0 % 16 or kc % 16:
        raise ValueError("column slices are whole k-tiles (16-aligned)")
    blk = 2 * 8 * KB  # uint32 words per (pair, k-tile) = 64*KB bytes
    f = _as_u32(flat)
    runs = [f[(pr * (K // 16) + k0 // 16) * blk:(pr * (K // 16) + (k0 + kc) // 16) * blk]
            for pr in range(N // 32)]
    return np.concatenate(runs)


# --------------------------------------------------------------------------------------------
# Transforms and references (docs/trellis-kernel.md 4.5, 4.8)
# --------------------------------------------------------------------------------------------


def fwht128(v: np.ndarray) -> np.ndarray:
    """Unnormalized natural-order FWHT over every contiguous 128-block of the last axis, in v's own
    dtype: stages lg = 0..6, pair (i, i + 2^lg) -> (a + b, a - b). For float32 these are FwhtLds's
    (hadamard_device.h:35-49) fp32 ops in its order, so the result is the kernel's bits."""
    shape = v.shape
    if shape[-1] % HAD:
        raise ValueError(f"last axis {shape[-1]} is not a multiple of {HAD}")
    out = np.array(v, copy=True).reshape(-1, HAD)
    for lg in range(7):
        h = 1 << lg
        w = out.reshape(-1, HAD // (2 * h), 2, h)
        a = w[:, :, 0, :].copy()
        b = w[:, :, 1, :].copy()
        w[:, :, 0, :] = a + b
        w[:, :, 1, :] = a - b
    return out.reshape(shape)


def transform_scale(prescale_log2: int) -> np.float32:
    """float32(2^s / sqrt(128)): the input transform's scale, correctly rounded on the host."""
    return np.float32(2.0 ** prescale_log2 / math.sqrt(HAD))


def transform_input(x: np.ndarray, suh: np.ndarray, prescale_log2: int) -> np.ndarray:
    """r4dx_trellis_input_bf16's arithmetic for one part: x [M][K] (bf16 values), suh [K] fp16 ->
    A [M][K] fp16 = f16_rn(FWHT128_fp32(fp32(x) * fp32(suh)) * float32(2^s / sqrt(128)))."""
    v = x.astype(np.float32) * suh.astype(np.float32)[None, :]
    v = fwht128(v)
    return (v * transform_scale(prescale_log2)).astype(np.float16)


def part_ranges(N: int, n_split: int, P: int) -> list[tuple[int, int]]:
    return [(0, N)] if P == 1 else [(0, n_split), (n_split, N)]


def linear_ref(x: np.ndarray, q: np.ndarray, suh: np.ndarray, svh: np.ndarray, n_split: int) -> np.ndarray:
    """fp64 x @ W_hat, W_hat = diag(suh_p) P_K Q P_N diag(svh), column n from part p(n)."""
    x64, q64, svh64 = x.astype(np.float64), q.astype(np.float64), svh.astype(np.float64)
    N = q.shape[1]
    y = np.empty((x.shape[0], N), dtype=np.float64)
    for p, (c0, c1) in enumerate(part_ranges(N, n_split, suh.shape[0])):
        t = fwht128(x64 * suh[p].astype(np.float64)[None, :]) / math.sqrt(HAD)
        y[:, c0:c1] = fwht128(t @ q64[:, c0:c1]) / math.sqrt(HAD) * svh64[None, c0:c1]
    return y


def linear_ref_from_a(a: np.ndarray, q: np.ndarray, svh: np.ndarray, n_split: int, prescale_log2: int) -> np.ndarray:
    """fp64 out_scale * svh[n] * FWHT128_n(A_p(n) @ Q) from the f16 A (docs/trellis-kernel.md 4.5)."""
    q64, svh64 = q.astype(np.float64), svh.astype(np.float64)
    N = q.shape[1]
    out_scale = 2.0 ** -prescale_log2 / math.sqrt(HAD)
    y = np.empty((a.shape[1], N), dtype=np.float64)
    for p, (c0, c1) in enumerate(part_ranges(N, n_split, a.shape[0])):
        y[:, c0:c1] = fwht128(a[p].astype(np.float64) @ q64[:, c0:c1]) * out_scale * svh64[None, c0:c1]
    return y


def reconstruct_f64(q: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """W_hat [K][N] = diag(suh) P_K Q P_N diag(svh) in fp64 (one part)."""
    t = fwht128(q.astype(np.float64).T).T / math.sqrt(HAD)
    return suh.astype(np.float64)[:, None] * fwht128(t) / math.sqrt(HAD) * svh.astype(np.float64)[None, :]


def decode_q(words: np.ndarray, KB: int) -> np.ndarray:
    """trellis_quant.decode_words (torch, CPU) of oracle-layout words -> Q [K][N] fp16."""
    w = torch.from_numpy(_as_u32(words).view(np.int32).copy())
    cb = torch.from_numpy(tq.codebook_np("mul1").copy())
    q = tq.decode_words(w, float(KB), cb).numpy()
    q16 = q.astype(np.float16)
    if not np.array_equal(q16.astype(np.float32), q):
        raise AssertionError("decode_words returned a value that is not an fp16 value")
    return q16


def bf16_rn(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 (round to nearest even), returned as float32 holding the bf16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


def bf16_ulp(v: np.ndarray) -> np.ndarray:
    """The bf16 ulp at |v| (8 significant bits)."""
    e = np.floor(np.log2(np.maximum(np.abs(v), 2.0 ** -126)))
    return 2.0 ** (e - 7)


def rng_for(tag: int) -> np.random.Generator:
    return np.random.default_rng([SEED, tag])


def random_words(tag: int, K: int, N: int, KB: int) -> np.ndarray:
    return rng_for(tag).integers(0, 1 << 32, size=(K // 16, N // 16, 8 * KB), dtype=np.uint64).astype(np.uint32)


def random_x(tag: int, M: int, K: int) -> np.ndarray:
    """bf16-exact activations: row scales log-uniform in [2^-8, 2^3] (the small rows reach the f16
    subnormal range at prescale 0) and two outliers of 20-60x per row."""
    rng = rng_for(tag)
    x = rng.standard_normal((M, K)) * np.exp2(rng.uniform(-8.0, 3.0, size=(M, 1)))
    for m in range(M):
        idx = rng.choice(K, size=2, replace=False)
        x[m, idx] *= rng.uniform(20.0, 60.0, size=2)
    return bf16_rn(x.astype(np.float32))


def random_scales(tag: int, n: int, lo: float, hi: float) -> np.ndarray:
    """fp16 +-scales, magnitudes log-uniform in [lo, hi] (the oracle's suh ~5e-3..4e-2, svh ~0.6..1.9)."""
    rng = rng_for(tag)
    mag = np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))
    return (np.where(rng.random(n) < 0.5, -1.0, 1.0) * mag).astype(np.float16)


# --------------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------------


def linear_case(prefix: str, x: np.ndarray, q: np.ndarray, suh: np.ndarray, svh: np.ndarray,
                n_split: int, files: dict) -> dict:
    """A (prescale 0), y and ya of one linear into `files`; returns the metadata."""
    a = np.stack([transform_input(x, suh[p], 0) for p in range(suh.shape[0])])
    files[f"{prefix}_a.npy"] = a
    files[f"{prefix}_ya.npy"] = linear_ref_from_a(a, q, svh, n_split, 0).astype(np.float32)
    files[f"{prefix}_y.npy"] = linear_ref(x, q, suh, svh, n_split).astype(np.float32)
    return {"P": int(suh.shape[0]), "n_split": int(n_split), "prescale": 0,
            "a": f"{prefix}_a.npy", "y": f"{prefix}_y.npy", "ya": f"{prefix}_ya.npy"}


def build_random(files: dict) -> dict:
    """rand_*: pair-grid words and Q at KB = 4 and 5; shared x/suh/svh/A; y and ya per KB and P."""
    K, N, M = RAND_K, RAND_N, M_ROWS
    x = random_x(10, M, K)
    suh = np.stack([random_scales(11, K, 5e-3, 4e-2), random_scales(12, K, 5e-3, 4e-2)])
    svh = random_scales(13, N, 0.6, 1.9)
    files["rand_x.npy"] = x
    files["rand_suh.npy"] = suh
    files["rand_svh.npy"] = svh
    for s in PRESCALES:
        files[f"rand_a_s{s}.npy"] = np.stack([transform_input(x, suh[p], s) for p in range(2)])
    meta = {"K": K, "N": N, "M": M, "prescales": list(PRESCALES), "n_split_p2": N // 2,
            "x": "rand_x.npy", "suh": "rand_suh.npy", "svh": "rand_svh.npy",
            "a": {f"s{s}": f"rand_a_s{s}.npy" for s in PRESCALES}, "kb": {}}
    for KB in (4, 5):
        words = random_words(KB, K, N, KB)
        q = decode_q(words, KB)
        pre = f"rand_k{KB}"
        files[f"{pre}_w.npy"] = to_pair_grid(words)
        files[f"{pre}_q.npy"] = q
        kb_meta = {"w": f"{pre}_w.npy", "q": f"{pre}_q.npy"}
        for P, n_split in ((1, N), (2, N // 2)):
            su = suh[:P]
            files[f"{pre}_y_p{P}.npy"] = linear_ref(x, q, su, svh, n_split).astype(np.float32)
            kb_meta[f"y_p{P}"] = f"{pre}_y_p{P}.npy"
            for s in (PRESCALES if P == 2 else (0,)):
                a = files[f"rand_a_s{s}.npy"][:P]
                name = f"{pre}_ya_p{P}_s{s}.npy"
                files[name] = linear_ref_from_a(a, q, svh, n_split, s).astype(np.float32)
                kb_meta[f"ya_p{P}_s{s}"] = name
        meta["kb"][str(KB)] = kb_meta
    return meta


def _hf_block(model_dir: Path, name: str, k0: int, n0: int) -> np.ndarray | None:
    """[BLOCK n][BLOCK k] of the bf16 checkpoint weight `name`, or None without the checkpoint."""
    from safetensors import safe_open

    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        return None
    shard = json.loads(idx.read_text(encoding="utf-8"))["weight_map"][name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
        return f.get_slice(name)[n0:n0 + BLOCK, k0:k0 + BLOCK].to(torch.float64).numpy()


def build_real(files: dict, oracle_dir: Path, sha_cache: dict) -> dict:
    """real_*: 256 x 256 blocks of the KB = 4 oracle, pinned by the oracle's own file sha256s."""
    from safetensors import safe_open

    man = tq.load_manifest(oracle_dir)
    man_sha = tq.sha256_path(Path(man["_path"]))
    if man.get("codebook") != "mul1" or not man.get("complete"):
        raise ValueError(f"{oracle_dir}: need a complete mul1 manifest")
    model_dir = Path(man["model_dir"])
    x = random_x(20, M_ROWS, BLOCK)
    files["real_x.npy"] = x
    meta = {"oracle_dir": str(oracle_dir), "manifest_sha256": man_sha,
            "hessian_basis": man.get("hessian_basis"), "config_sha256": man.get("config_sha256"),
            "block": BLOCK, "M": M_ROWS, "x": "real_x.npy", "cases": {}}
    for case, layer, modules, k0, n0 in REAL_BLOCKS:
        words_p, q_p, suh_p, svh_p, src = [], [], [], [], []
        for mod in modules:
            name = tq.hf_name(layer, mod)
            rec = man["tensors"][name]
            KB = rec["K"]
            if rec["encoding"] != tq.ENCODING_TRELLIS or float(KB) != 4.0 or rec["codebook"] != "mul1":
                raise ValueError(f"{name}: need a mul1 KB = 4 trellis tensor, got {rec['encoding']} K={KB}")
            if k0 % HAD or n0 % HAD or k0 + BLOCK > rec["k"] or n0 + BLOCK > rec["n"]:
                raise ValueError(f"{name}: block ({k0}, {n0}) outside [{rec['k']}, {rec['n']}] or unaligned")
            path = tq.tensor_file(man, rec)
            if str(path) not in sha_cache:
                sha_cache[str(path)] = tq.sha256_path(path)
            if sha_cache[str(path)] != rec["file_sha256"]:
                raise ValueError(f"{path}: sha256 {sha_cache[str(path)]} != manifest {rec['file_sha256']}")
            with safe_open(str(path), framework="np") as f:
                w = f.get_slice(name + ".trellis")[k0 // 16:(k0 + BLOCK) // 16, n0 // 16:(n0 + BLOCK) // 16, :]
                suh = f.get_slice(name + ".suh")[k0:k0 + BLOCK]
                svh = f.get_slice(name + ".svh")[n0:n0 + BLOCK]
            if w.dtype != np.int32 or suh.dtype != np.float16 or svh.dtype != np.float16:
                raise ValueError(f"{name}: dtypes {w.dtype}/{suh.dtype}/{svh.dtype}")
            words_p.append(_as_u32(w))
            suh_p.append(suh)
            svh_p.append(svh)
            # the cut's orientation and slices, against the bf16 checkpoint (a slip gives ~1.4)
            q_p.append(decode_q(words_p[-1], 4))
            w_hat = reconstruct_f64(q_p[-1], suh, svh)
            w_hf = _hf_block(model_dir, name, k0, n0)
            ent = {"hf_name": name, "file": path.name, "file_sha256": rec["file_sha256"],
                   "k": rec["k"], "n": rec["n"], "rel_weight_err_tensor": rec["rel_weight_err"]}
            if w_hf is not None:
                rel = float(np.linalg.norm(w_hat.T - w_hf) / np.linalg.norm(w_hf))
                cos = float(np.sum(w_hat.T * w_hf) / (np.linalg.norm(w_hat) * np.linalg.norm(w_hf)))
                if rel > 2.0 ** -(4 - 2) or cos < 0.9:
                    raise AssertionError(f"{name}: block rel {rel:.4f} cos {cos:.4f} vs the checkpoint")
                ent["rel_weight_err_block"] = rel
                ent["cos_block"] = cos
            src.append(ent)
        pre = f"real_{case}"
        # the oracle-layout cut (the regrid's input): real_attn_k_words, real_mlp_gate_words, ...
        word_files = [f"{pre}_words.npy"] if len(modules) == 1 else \
            [f"real_mlp_{m.split('.')[1].replace('_proj', '')}_words.npy" for m in modules]
        for fname, w in zip(word_files, words_p):
            files[fname] = w
        q = np.concatenate(q_p, axis=1)
        suh = np.stack(suh_p)
        svh = np.concatenate(svh_p)
        files[f"{pre}_w.npy"] = to_pair_grid(words_p)
        files[f"{pre}_q.npy"] = q
        files[f"{pre}_suh.npy"] = suh
        files[f"{pre}_svh.npy"] = svh
        lm = linear_case(pre, x, q, suh, svh, BLOCK if len(modules) > 1 else q.shape[1], files)
        meta["cases"][case] = {"layer": layer, "k0": k0, "n0": n0, "K": BLOCK, "N": int(q.shape[1]), "KB": 4,
                               "parts": src, "words": word_files, "w": f"{pre}_w.npy", "q": f"{pre}_q.npy",
                               "suh": f"{pre}_suh.npy", "svh": f"{pre}_svh.npy", **lm}
    return meta


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------


def tolerance_study(x, a, q, suh, svh, n_split, s, label: str) -> dict:
    """How the kernel's fp32 pipeline (A @ Q in fp32, fp32 FWHT, * svh * out_scale, one bf16
    rounding) sits against docs/trellis-kernel.md 6's tolerances: per element <= 4 bf16 ulp of ya,
    and ||C - y||/||y|| <= 2e-3. numpy's sgemm stands in for the WMMA fp32 accumulation."""
    N = q.shape[1]
    q32 = q.astype(np.float32)
    c = np.empty((a.shape[1], N), dtype=np.float32)
    scale = np.float32(2.0 ** -s / math.sqrt(HAD))
    for p, (c0, c1) in enumerate(part_ranges(N, n_split, a.shape[0])):
        g = fwht128(a[p].astype(np.float32) @ q32[:, c0:c1])
        c[:, c0:c1] = g * svh[c0:c1].astype(np.float32)[None, :] * scale
    c = bf16_rn(c).astype(np.float64)
    ya = linear_ref_from_a(a, q, svh, n_split, s)
    y = linear_ref(x, q, suh, svh, n_split)
    err = np.abs(c - ya)
    ulps = err / bf16_ulp(ya)
    rms = np.sqrt(np.mean(ya * ya, axis=1, keepdims=True))
    floor_bad = int(np.sum(err > 4 * bf16_ulp(ya) + 1e-4 * rms))
    out = {"max_ulp": float(ulps.max()), "over_4ulp": int(np.sum(ulps > 4)), "elements": int(ulps.size),
           "over_4ulp_with_1e-4_rms_floor": floor_bad,
           "rel_fro_c_vs_y": float(np.linalg.norm(c - y) / np.linalg.norm(y)),
           "rel_fro_ya_vs_y": float(np.linalg.norm(ya - y) / np.linalg.norm(y))}
    print(f"[golden]   {label}: fp32 pipeline vs ya max {out['max_ulp']:.2f} bf16 ulp, "
          f"{out['over_4ulp']}/{out['elements']} over 4 ulp ({floor_bad} with a 1e-4*rms floor); "
          f"||C-y||/||y|| {out['rel_fro_c_vs_y']:.2e} (A rounding alone {out['rel_fro_ya_vs_y']:.2e})")
    return out


def check_generated(files: dict, rand_meta: dict, real_meta: dict | None) -> dict:
    """Consistency of what is about to be written; returns the tolerance study for the manifest."""
    K, N = rand_meta["K"], rand_meta["N"]
    cb = torch.from_numpy(tq.codebook_np("mul1").copy())
    study = {}
    for KB in (4, 5):
        pre = f"rand_k{KB}"
        w = files[f"{pre}_w.npy"]
        words = from_pair_grid(w, K, N, KB)
        if not np.array_equal(to_pair_grid(words), w):
            raise AssertionError(f"{pre}: pair grid round trip")
        q = files[f"{pre}_q.npy"]
        if not np.array_equal(decode_q(words, KB).view(np.uint16), q.view(np.uint16)):
            raise AssertionError(f"{pre}: stored Q != decode_words(from_pair_grid(w))")
        # fp64 reference path vs trellis_quant.reconstruct (torch fp32)
        suh, svh = files["rand_suh.npy"], files["rand_svh.npy"]
        wt = tq.reconstruct(torch.from_numpy(words.view(np.int32).copy()), torch.from_numpy(suh[0].copy()),
                            torch.from_numpy(svh.copy()), float(KB), cb).numpy().astype(np.float64)
        w64 = reconstruct_f64(q, suh[0], svh)
        d = float(np.abs(wt - w64).max() / np.abs(w64).max())
        if d > 1e-5:
            raise AssertionError(f"{pre}: reconstruct_f64 vs trellis_quant.reconstruct {d:.2e}")
        y1 = files[f"{pre}_y_p1.npy"].astype(np.float64)
        yw = files["rand_x.npy"].astype(np.float64) @ w64
        d = float(np.linalg.norm(y1 - yw) / np.linalg.norm(yw))
        if d > 1e-6:
            raise AssertionError(f"{pre}: linear_ref vs x @ reconstruct_f64 {d:.2e}")
        print(f"[golden] {pre}: pair grid, decode, reconstruct (vs trellis_quant {d:.1e}) OK", flush=True)
        for P, n_split in ((1, N), (2, N // 2)):
            for s in (PRESCALES if P == 2 else (0,)):
                study[f"{pre}_p{P}_s{s}"] = tolerance_study(
                    files["rand_x.npy"], files[f"rand_a_s{s}.npy"][:P], q, suh[:P], svh, n_split, s,
                    f"{pre} P={P} s={s}")
    if real_meta is not None:
        for case, cm in real_meta["cases"].items():
            pre = f"real_{case}"
            words = from_pair_grid(files[f"{pre}_w.npy"], cm["K"], cm["N"], 4)
            if not np.array_equal(decode_q(words, 4).view(np.uint16), files[f"{pre}_q.npy"].view(np.uint16)):
                raise AssertionError(f"{pre}: stored Q != decode_words(from_pair_grid(w))")
            study[pre] = tolerance_study(files["real_x.npy"], files[f"{pre}_a.npy"], files[f"{pre}_q.npy"],
                                         files[f"{pre}_suh.npy"], files[f"{pre}_svh.npy"], cm["n_split"], 0, pre)
            for p in cm["parts"]:
                if "rel_weight_err_block" in p:
                    print(f"[golden]   {p['hf_name']}: block vs checkpoint rel {p['rel_weight_err_block']:.4f} "
                          f"cos {p['cos_block']:.4f} (whole tensor {p['rel_weight_err_tensor']:.4f})")
    return study


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npy_descr(arr: np.ndarray) -> str:
    return {np.dtype(np.uint32): "<u4", np.dtype(np.float16): "<f2", np.dtype(np.float32): "<f4"}[arr.dtype]


def cmd_generate(args) -> int:
    out_dir = Path(args.out_dir)
    files: dict[str, np.ndarray] = {}
    print("[golden] random cases (KB = 4, 5)", flush=True)
    rand_meta = build_random(files)
    real_meta = None
    if not args.no_real:
        if not Path(args.oracle_dir).exists():
            print(f"[golden] {args.oracle_dir} not found: pass --no-real to write the random cases only",
                  file=sys.stderr)
            return 2
        print(f"[golden] real tiles from {args.oracle_dir}", flush=True)
        real_meta = build_real(files, Path(args.oracle_dir), {})
    study = check_generated(files, rand_meta, real_meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in sorted(files):
        arr = np.ascontiguousarray(files[name])
        descr = npy_descr(arr)
        path = out_dir / name
        np.save(path, arr, allow_pickle=False)
        entries[name] = {"sha256": sha256_file(path), "dtype": descr, "shape": list(arr.shape),
                         "bytes": path.stat().st_size}
    total = sum(e["bytes"] for e in entries.values())
    stale = sorted(p.name for p in out_dir.glob("*.npy") if p.name not in entries)
    doc = {
        "format": FORMAT, "version": VERSION,
        "spec": "docs/trellis-kernel.md 2.1 (pair grid), 4.5 (epilogue), 4.8 (input transform), 6 (golden vectors)",
        "generator": {"script": "tools/reference/trellis_golden.py", "script_sha256": tq.sha256_path(Path(__file__)),
                      "trellis_quant_sha256": tq.sha256_path(Path(tq.__file__)), "seed": SEED,
                      "numpy": np.__version__, "torch": torch.__version__},
        "conventions": {
            "q": "[K][N] fp16, decode_words output (in, out); W_hat = diag(suh_p) P_K Q P_N diag(svh)",
            "w": "1-D uint32 pair grid: word w of tile (tn, tk) at (((tn>>1)*(K/16)+tk)*2+(tn&1))*8*KB+w",
            "words": "oracle layout [K/16][N/16][8*KB] uint32 (the regrid's input)",
            "parts": "column n reads part (n >= n_split); P = 1 means n_split = N",
            "a": "[P][M][K] fp16 = f16_rn(FWHT128_fp32(fp32(x)*fp32(suh_p)) * float32(2^s/sqrt(128))), "
                 "stages lg = 0..6 as FwhtLds, no FMA contraction, one rounding",
            "y": "[M][N] fp64 x @ W_hat from the exact x, stored fp32 (Frobenius reference)",
            "ya": "[M][N] fp64 2^-s/sqrt(128) * svh[n] * FWHT128_n(A_p(n) @ Q) from the f16 A, stored fp32",
            "x": "[M][K] bf16-exact values stored fp32",
        },
        "random": rand_meta, "real": real_meta, "tolerance_study": study,
        "files": entries, "total_bytes": total,
    }
    (out_dir / "manifest.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                                           newline="\n")
    print(f"[golden] wrote {len(entries)} files, {total / 2**20:.2f} MiB, to {out_dir}")
    if stale:
        print(f"[golden] WARNING: {len(stale)} .npy files not in the manifest (stale): {', '.join(stale)}")
    if torch.cuda.is_initialized():
        raise AssertionError("torch.cuda was initialized: this script must stay on the CPU")
    return 0


def cmd_check(args) -> int:
    """Every file against manifest.json: sha256, npy dtype and shape; no unlisted .npy files."""
    out_dir = Path(args.out_dir)
    doc = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        print(f"[golden] {out_dir}: not a {FORMAT} v{VERSION} manifest")
        return 1
    bad = 0
    for name, e in sorted(doc["files"].items()):
        path = out_dir / name
        if not path.exists():
            print(f"[golden] MISSING {name}")
            bad += 1
            continue
        arr = np.load(path, allow_pickle=False)
        ok = sha256_file(path) == e["sha256"] and arr.dtype.str == e["dtype"] and list(arr.shape) == e["shape"]
        if not ok:
            print(f"[golden] MISMATCH {name}")
            bad += 1
    extra = sorted(p.name for p in out_dir.glob("*.npy") if p.name not in doc["files"])
    for name in extra:
        print(f"[golden] UNLISTED {name}")
    bad += len(extra)
    print(f"[golden] check {out_dir}: {len(doc['files'])} files, {doc['total_bytes'] / 2**20:.2f} MiB: "
          + ("OK" if bad == 0 else f"FAILED ({bad})"))
    return 0 if bad == 0 else 1


def cmd_selftest(args) -> int:
    """This script's own math on the CPU: the pair grid (2.1 index, round trip, 2.4 slices), the
    fp32 FWHT against a literal FwhtLds loop, the fp64 FWHT against the Hadamard matrix, the input
    transform against an fp64 one, and the full-linear references against each other."""
    rng = np.random.default_rng(1)
    bad = 0

    def report(what: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        bad += not ok
        print(f"[selftest] {what}: {'OK' if ok else 'FAILED'}{(' (' + detail + ')') if detail else ''}", flush=True)

    for KB in (4, 5):
        K, parts_n = 256, (128, 384)
        parts = [rng.integers(0, 1 << 32, size=(K // 16, n // 16, 8 * KB), dtype=np.uint64).astype(np.uint32)
                 for n in parts_n]
        N = sum(parts_n)
        flat = to_pair_grid([p.view(np.int32) for p in parts])
        full = np.concatenate(parts, axis=1)
        idx_ok = all(flat[pair_grid_index(tn, tk, w, K, KB)] == full[tk, tn, w]
                     for tn in range(N // 16) for tk in range(K // 16) for w in (0, 8 * KB - 1))
        report(f"KB={KB} pair grid index formula (2.1)", idx_ok)
        report(f"KB={KB} from_pair_grid(to_pair_grid) round trip", np.array_equal(from_pair_grid(flat, K, N, KB), full))
        rows_ok = all(np.array_equal(pair_grid_rows(flat, K, KB, n0, cnt), to_pair_grid(full[:, n0 // 16:(n0 + cnt) // 16]))
                      for n0, cnt in ((0, 128), (128, 256), (256, 256), (96, 64)))
        report(f"KB={KB} row slices are one contiguous run (2.4 kRows)", rows_ok)
        cols_ok = all(np.array_equal(pair_grid_cols(flat, K, N, KB, k0, kc), to_pair_grid(full[k0 // 16:(k0 + kc) // 16]))
                      for k0, kc in ((0, 128), (128, 128), (64, 96)))
        report(f"KB={KB} column slices per pair row (2.4 kCols)", cols_ok)
        prefix_ok = np.array_equal(flat[:128 * K * KB // 32], to_pair_grid(parts[0]))
        report(f"KB={KB} part 0 is a prefix of the pair grid", prefix_ok)

    # fp32 FWHT == FwhtLds's loop, op for op (bitwise)
    v = (rng.standard_normal((3, 256)) * np.exp2(rng.uniform(-10, 10, size=(3, 1)))).astype(np.float32)
    ref = v.copy()
    for row in ref.reshape(-1, HAD):
        for lg in range(7):
            h = 1 << lg
            for p in range(HAD // 2):
                i = ((p >> lg) << (lg + 1)) | (p & (h - 1))
                a, b = row[i], row[i + h]
                row[i], row[i + h] = np.float32(a + b), np.float32(a - b)
    report("fwht128 fp32 == FwhtLds loop (bitwise)", np.array_equal(fwht128(v).view(np.uint32), ref.view(np.uint32)))
    i = np.arange(HAD)
    Hm = 1.0 - 2.0 * (np.vectorize(lambda t: bin(t).count("1") & 1)(i[:, None] & i[None, :]))
    v64 = rng.standard_normal((5, 384))
    d = float(np.abs(fwht128(v64) - (v64.reshape(5, 3, HAD) @ Hm).reshape(5, 384)).max())
    report("fwht128 fp64 == natural-order Hadamard matrix", d < 1e-12, f"max {d:.1e}")
    Pt = torch.from_numpy(tq.hadamard128("cpu").numpy().astype(np.float64))
    d = float(np.abs(Hm / math.sqrt(HAD) - Pt.numpy()).max())
    report("the same matrix as trellis_quant.hadamard128", d < 1e-7, f"max {d:.1e}")

    # input transform: fp32 emulation vs the exact value, f16 ulps
    x = random_x(99, 16, 512)
    suh = random_scales(98, 512, 5e-3, 4e-2)
    for s in PRESCALES:
        a = transform_input(x, suh, s).astype(np.float64)
        exact = fwht128(x.astype(np.float64) * suh.astype(np.float64)) / math.sqrt(HAD) * 2.0 ** s
        e16 = exact.astype(np.float16).astype(np.float64)
        ulp16 = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(exact), 2.0 ** -14))) - 10)
        du = np.abs(a - e16) / ulp16
        sub = int(np.sum(np.abs(a) < 2.0 ** -14))
        report(f"transform s={s} vs f16(exact fp64)", float(du.max()) <= 2.0,
               f"max {du.max():.0f} ulp, {int(np.sum(du > 0))}/{du.size} differ by rounding, {sub} subnormal")

    # references: ya with A = the exact transform (no f16 rounding) equals y
    q = decode_q(random_words(97, 256, 256, 4), 4)
    svh = random_scales(96, 256, 0.6, 1.9)
    suh2 = np.stack([random_scales(95, 256, 5e-3, 4e-2), random_scales(94, 256, 5e-3, 4e-2)])
    xx = random_x(93, 8, 256)
    for P, n_split in ((1, 256), (2, 128)):
        y = linear_ref(xx, q, suh2[:P], svh, n_split)
        a_exact = np.stack([fwht128(xx.astype(np.float64) * suh2[p].astype(np.float64)) / math.sqrt(HAD)
                            for p in range(P)])
        ya = linear_ref_from_a(a_exact, q, svh, n_split, 0)
        d = float(np.linalg.norm(ya - y) / np.linalg.norm(y))
        report(f"P={P} ya(exact A) == y", d < 1e-13, f"{d:.1e}")
        w = np.concatenate([reconstruct_f64(q[:, c0:c1], suh2[p], svh[c0:c1])
                            for p, (c0, c1) in enumerate(part_ranges(256, n_split, P))], axis=1)
        d = float(np.linalg.norm(xx.astype(np.float64) @ w - y) / np.linalg.norm(y))
        report(f"P={P} y == x @ reconstruct_f64", d < 1e-13, f"{d:.1e}")
    if torch.cuda.is_initialized():
        report("torch.cuda untouched", False)
    print("SELFTEST " + ("PASSED" if bad == 0 else f"FAILED ({bad})"))
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE,
                    help="a finished KB = 4 quantize-model directory (real tiles; read only)")
    ap.add_argument("--no-real", action="store_true", help="random cases only")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="verify the files against manifest.json")
    g.add_argument("--selftest", action="store_true", help="CPU checks of this script's math; writes nothing")
    args = ap.parse_args()
    if args.check:
        return cmd_check(args)
    if args.selftest:
        return cmd_selftest(args)
    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
