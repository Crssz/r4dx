"""tools/reference/trellis_quant.py -- the EXL3 trellis-quantization ORACLE for Qwen3.8-27B.

What it answers: the KL (on our tokens, weights only) of EXL3-style trellis quantization at 3.5 /
4.0 / 4.5 / 5.0 bits per weight, BEFORE any RDNA4 trellis kernel is written. It quantizes every
decoder linear the way exllamav3 does and writes the result in the exact stored form (packed
trellis bitstream + fp16 input/output scales), so the bits per weight are MEASURED, and
full_logits_golden.py --weights-override reconstructs the weights in the ORIGINAL basis for the
bf16 reference forward. No engine or converter change is involved.

The specification is docs/trellis.md (exllamav3 at commit 6b84a21, MIT, (c) turboderp; QTIP, Tseng,
Sun, Hou, De Sa, arXiv 2406.11235). Every function here names the spec section it implements. The
structure of the Viterbi recursion, the two-pass tail-biting, regularize(), the global-scale search,
block LDL, LDLQ and refit_scales() is adapted from exllamav3's
exllamav3/modules/quant/exl3_lib/quantize.py and exllamav3_ext/quant/*.cuh (MIT); the code is
re-written here from docs/trellis.md, not copied.

Pieces (see the spec section in brackets):

  codebook_np          mul1 / 3inst / mcg state -> value, exact fp16 semantics         [4]
  widths, frac_k       per-position bit widths (integer K, K + 0.5 = KA/KA+1 by 0xAAAA) [3.3, 5.2]
  tensor_core_perm     the order the trellis walks a 16x16 tile                         [3.2]
  viterbi_torch        pure-torch reference Viterbi, any state width / length / K      [6]
  TrellisEncoder       the fast encoders: native CPU (std::thread) and HIP (gfx1201),
                       both built from trellis_viterbi.hip, bit-identical to viterbi_torch
  pack_states / unpack_states / decode   the stored bitstream: 8K uint32 words per tile [5]
  regularize           signs, 128-point Hadamards both sides, suh/svh, global scale    [8.2, 8.3]
  finalize_hessian, block_ldl, ldlq      damping 0.025, sign-rotated H, block LDLQ      [7.2, 7.3]
  refit_scales         2 rounds of closed-form suh/svh refits in the H metric          [7.5]
  quantize_group       one shared-Hessian group of linears end to end                  [2]
  allocate             bpw -> per-tensor K (EXL3's allocator, EXL3's qgroups)           [9]
  int4_ldlq            our converter's w4a16 g64 LDLQ (quant_ldlq.hpp) in torch, the baseline

Recorded deviations from EXL3 (docs/trellis.md 13, RECIPE below): fp32 Viterbi path costs (EXL3:
fp16; spec 6.5 measures <= 0.04% MSE); smallest-edge tie rule on the final argmin (EXL3: a
bit-reversed rank); our own seeded sign draws (EXL3: torch.manual_seed(module index)); Hessians are
hessian-v2's `keys` (bf16 forward, non-sequential) instead of EXL3's sequential capture, and their
calibration text is domain-matched to the KL corpus with no uniform random-token rows (EXL3's default
set is generic text plus 15.6% random-token rows), which probably flatters the oracle on
tokens_canon; lm_head and embeddings stay bf16. The allocator matches EXL3 (qgroups q/k/v,
in_proj_qkv/in_proj_z and gate/up promoted as units; every q_priority 0). Q7 (the sign-only Hessian
basis of the LDL factor) is reproduced by default (--hessian-basis exl3, EXL3 fidelity);
--hessian-basis matched factors the Hessian of the weight actually quantized (magnitudes included),
a converter-only choice that is much better on attention q/k/v (docs/trellis.md 14).

Subcommands (run with the reference venv's python; GPU work only with $env:HIP_VISIBLE_DEVICES='1'):

  selftest        native encoders vs viterbi_torch on random tiles (run this first on a new box)
  gaussian        validation (a): iid N(0,1) MSE per K vs QTIP's tables and scalar Lloyd-Max
  linear          validation (b): the Hessian proxy loss of real linears, trellis vs int4 g64 LDLQ
  bench           validation (c): encoder tiles/s and the projected whole-model time
  quantize-model  quantize every decoder linear at one rate into an override directory
  mix             a bpw mix (e.g. 4.5 = 4/5) from finished integer-rate directories
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --------------------------------------------------------------------------------------------
# Constants (docs/trellis.md)
# --------------------------------------------------------------------------------------------

STATE_BITS = 16                 # L: 16-bit trellis state                               [3.3]
TILE = 256                      # 16 x 16 weights per tile = one tail-biting ring       [3.1]
ROLL = TILE // 2                # pass 1 of the tail-biting starts at position 128      [6.2]
MUL1_MULT = 0x83DCD12D          # codebook_mul1_mult                                    [4.1]
INST3_MULT, INST3_ADD = 89226354, 64248484  # QTIP Algorithm 2                          [4.2]
MCG_MULT = 0xCBAC1FED           # codebook_mcg_mult                                     [4.2]
CODEBOOK_SCALE = 1.24371088     # Q::codebook_scale (the 3INST std)                     [4.3]
SIGMA_REG = 0.025               # quant_args sigma_reg                                  [7.2]
HAD = 128                       # had_k = had_n = 128                                   [8.2]
BUF_K = 128                     # ldlq buf_size_k                                       [7.3]
FRAC_MASK = 0xAAAA              # the only half-rate mask EXL3 produces                 [3.3]
LDLQ_DRIFT = {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}  #                  [8.3]
REFIT_ROUNDS = 2                #                                                       [7.5]

LAYER_PREFIX = "model.language_model.layers."

#: HF module (under a decoder layer) -> (hessian-v2 container key suffix, the tap it reads, the
#: tensor class used in reports). Order = EXL3's module order inside a layer (the allocator's idx).
LINEARS = {
    "self_attn.q_proj": ("attn.qg", "in"),
    "self_attn.k_proj": ("attn.k", "in"),
    "self_attn.v_proj": ("attn.v", "in"),
    "self_attn.o_proj": ("attn.o", "out"),
    "linear_attn.in_proj_qkv": ("gdn.in_proj_qkv", "in"),
    "linear_attn.in_proj_z": ("gdn.in_proj_z", "in"),
    "linear_attn.out_proj": ("gdn.out_proj", "out"),
    "mlp.gate_proj": ("mlp.gate_up", "mlp_in"),
    "mlp.up_proj": ("mlp.gate_up", "mlp_in"),
    "mlp.down_proj": ("mlp.down", "mlp_mid"),
}
#: HF module -> EXL3's qgroup suffix (exllamav3 6b84a21 modules/attn.py `key + ".qkv"` / `".o"`,
#: modules/gated_delta_net.py `key + ".qkvz"` / `".o"`, modules/mlp.py GatedMLP `key + ".gu"` /
#: `".d"`): the allocator promotes a group only as a whole (docs/trellis.md 9).
QGROUPS = {
    "self_attn.q_proj": "self_attn.qkv",
    "self_attn.k_proj": "self_attn.qkv",
    "self_attn.v_proj": "self_attn.qkv",
    "self_attn.o_proj": "self_attn.o",
    "linear_attn.in_proj_qkv": "linear_attn.qkvz",
    "linear_attn.in_proj_z": "linear_attn.qkvz",
    "linear_attn.out_proj": "linear_attn.o",
    "mlp.gate_proj": "mlp.gu",
    "mlp.up_proj": "mlp.gu",
    "mlp.down_proj": "mlp.d",
}
ATTN_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
GDN_MODULES = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
MLP_MODULES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

OVERRIDE_FORMAT = "r4dx-weights-override"
OVERRIDE_VERSION = 1
MANIFEST_NAME = "weights_override.json"
ENCODING_TRELLIS = "trellis-exl3"
ENCODING_DENSE = "dense"


def module_list(layer_type: str) -> list[str]:
    return (ATTN_MODULES if layer_type == "full_attention" else GDN_MODULES) + MLP_MODULES


def hf_name(layer: int, module: str) -> str:
    return f"{LAYER_PREFIX}{layer}.{module}.weight"


def parse_hf_name(name: str) -> tuple[int, str] | None:
    """`model.language_model.layers.7.mlp.down_proj.weight` -> (7, 'mlp.down_proj')."""
    if not (name.startswith(LAYER_PREFIX) and name.endswith(".weight")):
        return None
    head, _, rest = name[len(LAYER_PREFIX):].partition(".")
    if not head.isdigit():
        return None
    return int(head), rest[: -len(".weight")]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    """A deterministic 31-bit seed from strings/ints (crc32; Python's hash() is salted)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


# --------------------------------------------------------------------------------------------
# Codebooks (docs/trellis.md 4)
# --------------------------------------------------------------------------------------------


def _fp16_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


MUL1_K_INV = _fp16_bits(0x1EEE)   # 0.00676727294921875
MUL1_K_BIAS = _fp16_bits(0xC931)  # -10.3828125


@functools.lru_cache(maxsize=None)
def codebook_np(name: str = "mul1") -> np.ndarray:
    """The value of every 16-bit state, float32 [65536] holding the exact fp16 values EXL3 decodes.

    mul1: x = s * 0x83DCD12D (mod 2^32); fp16(1024 + bytesum(x)) is exact; v = fp16((1024 + sum) *
    fp16(0x1EEE) + fp16(0xC931)) with ONE rounding (the float64 product and sum are exact, so a
    single float64 -> float16 round-to-nearest-even is __hfma's result). 3inst / mcg: y = (x &
    0x8FFF8FFF) ^ 0x3B603B60, v = fp16(half(lo) + half(hi)) (the float64 sum of two halves is exact).
    """
    s = np.arange(1 << STATE_BITS, dtype=np.uint64)
    m32 = np.uint64(0xFFFFFFFF)
    if name == "mul1":
        x = (s * np.uint64(MUL1_MULT)) & m32
        bsum = sum((x >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4))
        h = (np.uint64(1024) + bsum).astype(np.float64)
        v = h * MUL1_K_INV + MUL1_K_BIAS
        out = v.astype(np.float16)
    elif name in ("3inst", "mcg"):
        x = ((s * np.uint64(INST3_MULT) + np.uint64(INST3_ADD)) if name == "3inst"
             else (s * np.uint64(MCG_MULT))) & m32
        y = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
        lo = (y & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float64)
        hi = (y >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float64)
        out = (lo + hi).astype(np.float16)
    else:
        raise ValueError(f"unknown codebook {name!r} (mul1, 3inst, mcg)")
    out = out.astype(np.float32)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------------------------
# Rates, widths, tile order (docs/trellis.md 3)
# --------------------------------------------------------------------------------------------


def frac_k(K: float) -> tuple[int, int]:
    """Q::frac_k: integer K -> (K, 0); K + 0.5 -> (K, 0xAAAA)."""
    ka = int(math.floor(K + 1e-12))
    if abs(K - ka) < 1e-9:
        return ka, 0
    if abs(ka + 0.5 - K) >= 1e-9:
        raise ValueError(f"K={K}: only integers and integer + 0.5 are rates")
    return ka, FRAC_MASK


def widths(K: float, n: int = TILE) -> list[int]:
    """D(p) = KA + ((MASK >> (p & 15)) & 1) for p in 0..n-1."""
    ka, mask = frac_k(K)
    return [ka + ((mask >> (p & 15)) & 1) for p in range(n)]


def trellis_words(K: float) -> int:
    """uint32 words per 256-weight tile (the ring is 256 K bits)."""
    bits = sum(widths(K))
    assert bits % 32 == 0, (K, bits)
    return bits // 32


@functools.lru_cache(maxsize=None)
def tensor_core_perm() -> np.ndarray:
    """Q::tensor_core_perm: sequence position p = 8t + j holds row-major tile element perm[p]."""
    perm = []
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm += [r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                 r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1]
    out = np.array(perm, dtype=np.int64)
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=None)
def tensor_core_perm_inv() -> np.ndarray:
    out = np.argsort(tensor_core_perm())
    out.setflags(write=False)
    return out


@functools.lru_cache(maxsize=16)
def _perm_t(device: str, inverse: bool = False) -> torch.Tensor:
    src = tensor_core_perm_inv() if inverse else tensor_core_perm()
    return torch.tensor(src, dtype=torch.long, device=device)


# --------------------------------------------------------------------------------------------
# Viterbi: the pure-torch reference (docs/trellis.md 6)
# --------------------------------------------------------------------------------------------


def _viterbi_step(prev: torch.Tensor, cbv: torch.Tensor, w: torch.Tensor, a: int, b: int, L: int):
    """One step: prev [T, 2^(L-a)] in-edge costs -> ([T, 2^(L-b)] out-edge costs, uint8 choices).
    State s = (h << (L-b)) | (g << a) | o_lo, in-edge s >> a = (h << (L-a-b)) | g."""
    T = prev.shape[0]
    gb = L - a - b
    if gb < 0:
        raise ValueError(f"widths {a}+{b} exceed the {L}-bit state")
    d = cbv.view(1, 1 << b, 1 << gb, 1 << a) - w.view(T, 1, 1, 1)
    tot = prev.view(T, 1 << b, 1 << gb, 1) + d * d
    cur, arg = tot.min(dim=1)
    return cur.reshape(T, -1), arg.to(torch.uint8).reshape(T, -1)


def viterbi_torch(x: torch.Tensor, cb: torch.Tensor, D: list[int], L: int = STATE_BITS,
                  roll: int | None = None, pinned: torch.Tensor | None = None):
    """Tail-biting Viterbi over T rings at once (the reference both native encoders must match).

    x [T, n] float32 targets (as they will be compared: callers round to fp16 first), cb [2^L]
    float32 codebook, D the n per-position widths, L the state width. Arithmetic exactly as the
    native builds: d = cb - x, t = prev + d*d (two fp32 roundings), first minimum (smallest h /
    smallest edge) on ties. Two passes (docs/trellis.md 6.2) unless `pinned` [T] is given, in which
    case only pass 2 runs with that boundary edge (the brute-force tests use this: it is exact).
    Returns (states [T, n] int64, cost [T] float32)."""
    T, n = x.shape
    if len(D) != n:
        raise ValueError(f"{len(D)} widths for {n} positions")
    if cb.numel() != 1 << L:
        raise ValueError(f"codebook has {cb.numel()} entries, need 2^{L}")
    roll = n // 2 if roll is None else roll
    dev = x.device
    x = x.float()
    cb = cb.to(device=dev, dtype=torch.float32)
    ar = torch.arange(T, device=dev)
    inf = float("inf")
    bp: list[torch.Tensor | None] = [None] * n

    def forward(start: int, prev: torch.Tensor) -> torch.Tensor:
        for i in range(n):
            p = (i + start) % n
            prev, bp[p] = _viterbi_step(prev, cb, x[:, p], D[p], D[(p + 1) % n], L)
        return prev

    if pinned is None:
        prev = forward(roll, torch.zeros(T, 1 << (L - D[roll % n]), device=dev))
        e = prev.argmin(dim=1)
        for p in range(roll - 1, -1, -1):
            a, b = D[p], D[(p + 1) % n]
            h = bp[p][ar, e].long()
            e = ((h << (L - b)) | e) >> a
        end = e
    else:
        end = pinned.to(device=dev, dtype=torch.long)
    prev = torch.full((T, 1 << (L - D[0])), inf, device=dev)
    prev[ar, end] = 0.0
    prev = forward(0, prev)
    cost = prev[ar, end]
    states = torch.empty(T, n, dtype=torch.long, device=dev)
    e = end
    for p in range(n - 1, -1, -1):
        a, b = D[p], D[(p + 1) % n]
        h = bp[p][ar, e].long()
        s = (h << (L - b)) | e
        states[:, p] = s
        e = s >> a
    return states, cost


def ring_path_cost(states: torch.Tensor, x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """sum_p (cb[state_p] - x_p)^2 per ring, in float64 (for checks)."""
    v = cb.to(states.device)[states.long()].double()
    return ((v - x.double()) ** 2).sum(dim=1)


def ring_consistent(states: torch.Tensor, D: list[int], L: int = STATE_BITS) -> torch.Tensor:
    """True per ring when consecutive states overlap as a bitshift ring must:
    state(p) >> D(p) == state(p-1) & (2^(L - D(p)) - 1), including p = 0 against p = n-1."""
    n = states.shape[1]
    ok = torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
    for p in range(n):
        a = D[p]
        prev = states[:, (p - 1) % n]
        ok &= (states[:, p] >> a) == (prev & ((1 << (L - a)) - 1))
    return ok


# --------------------------------------------------------------------------------------------
# Native encoders (trellis_viterbi.hip), built on first use
# --------------------------------------------------------------------------------------------

NATIVE_SRC = HERE / "trellis_viterbi.hip"
BUILD_DIR = HERE / ".trellis_build"
NATIVE_ABI = 1
HIP_ARCH = os.environ.get("R4DX_TRELLIS_ARCH", "gfx1201")


def rocm_sdk_root() -> Path | None:
    """The ROCm SDK the reference venv's torch runs on (`_rocm_sdk_core`), which ships clang++ and
    the device libraries. None when absent."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _dll_suffix() -> str:
    return ".dll" if sys.platform == "win32" else ".so"


def native_build_command(kind: str, out: Path) -> list[str]:
    root = rocm_sdk_root()
    if root is None:
        raise RuntimeError("no _rocm_sdk_core package in this venv (it provides clang++)")
    exe = root / "lib" / "llvm" / "bin" / ("clang++.exe" if sys.platform == "win32" else "clang++")
    common = ["-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-o", str(out)]
    if sys.platform == "win32":
        common.append("-fuse-ld=lld")
    else:
        common.append("-fPIC")
    if kind == "cpu":
        return [str(exe), "-x", "c++", "-DTRELLIS_CPU", "-march=native", *common, str(NATIVE_SRC)]
    if kind == "hip":
        bc = root / "lib" / "llvm" / "amdgcn" / "bitcode"
        return [str(exe), "-x", "hip", f"--offload-arch={HIP_ARCH}", f"--rocm-path={root}",
                f"--rocm-device-lib-path={bc}", *common, str(NATIVE_SRC),
                f"-L{root / 'lib'}", "-lamdhip64"]
    raise ValueError(kind)


def native_lib_path(kind: str) -> Path:
    key = hashlib.sha256(NATIVE_SRC.read_bytes() + kind.encode() + HIP_ARCH.encode()).hexdigest()[:16]
    return BUILD_DIR / f"trellis_viterbi_{kind}_{key}{_dll_suffix()}"


_NATIVE: dict[str, object] = {}


def load_native(kind: str, verbose: bool = True):
    """ctypes handle of the `kind` ('cpu' or 'hip') encoder, building it if needed. Raises on any
    failure (callers fall back to the torch path and say so)."""
    if kind in _NATIVE:
        lib = _NATIVE[kind]
        if isinstance(lib, Exception):
            raise lib
        return lib
    try:
        path = native_lib_path(kind)
        if not path.exists():
            BUILD_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + f".tmp{os.getpid()}" + path.suffix)
            cmd = native_build_command(kind, tmp)
            if verbose:
                print(f"[trellis] building the {kind} encoder: {' '.join(cmd)}", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{kind} encoder build failed ({r.returncode}):\n{r.stderr[-4000:]}")
            os.replace(tmp, path)
            for extra in (".lib", ".exp"):
                side = tmp.with_suffix(extra)
                if side.exists():
                    side.unlink()
            if verbose:
                print(f"[trellis] built {path.name} in {time.perf_counter() - t0:.1f}s", flush=True)
        if kind == "hip":
            root = rocm_sdk_root()
            if sys.platform == "win32" and root is not None and (root / "bin").is_dir():
                os.add_dll_directory(str(root / "bin"))
            torch.cuda.init()  # the runtime torch loaded is the one the DLL binds to
        lib = ctypes.CDLL(str(path))
        lib.r4dx_trellis_abi.restype = ctypes.c_int
        if lib.r4dx_trellis_abi() != NATIVE_ABI:
            raise RuntimeError(f"{path.name}: ABI {lib.r4dx_trellis_abi()} != {NATIVE_ABI}")
        if kind == "cpu":
            f = lib.r4dx_trellis_viterbi_cpu
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_uint,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        else:
            f = lib.r4dx_trellis_viterbi_hip
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_uint, ctypes.c_int,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
            g = lib.r4dx_trellis_scratch_bytes
            g.restype = ctypes.c_int64
            g.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_int]
        _NATIVE[kind] = lib
        return lib
    except Exception as exc:  # noqa: BLE001 -- remembered, re-raised to every caller
        _NATIVE[kind] = exc
        raise


def hip_widths_ok(K: float) -> bool:
    return all(3 <= d <= 6 for d in widths(K, 16))


class TrellisEncoder:
    """Viterbi-encodes batches of 256-weight tiles (docs/trellis.md 6).

    backend: 'hip' (native, device tensors, gfx1201), 'cpu' (native, std::thread), 'torch'
    (viterbi_torch on `device`), or 'auto' (hip on a cuda device, else cpu, else torch). All three
    give the same states bit for bit (tests/reference/test_trellis_quant.py). Targets are rounded
    to fp16 first, as EXL3's kernel does (`__float2half_rn(input_tile[i])`)."""

    def __init__(self, backend: str = "auto", device: str | torch.device = "cpu",
                 codebook: str = "mul1", threads: int = 0, grid: int = 0,
                 tiles_per_launch: int = 0, verbose: bool = True):
        self.device = torch.device(device)
        self.codebook = codebook
        self.cb_np = codebook_np(codebook)
        self.cb = torch.from_numpy(self.cb_np.copy()).to(self.device)
        self.threads = threads or (os.cpu_count() or 1)
        self.verbose = verbose
        self.tiles = 0
        self.seconds = 0.0
        self._scratch = None
        self._fallback_warned: set = set()
        if backend == "auto":
            chosen = "torch"
            for cand in (["hip", "cpu"] if self.device.type == "cuda" else ["cpu"]):
                try:
                    load_native(cand, verbose)
                    chosen = cand
                    break
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"[trellis] {cand} encoder unavailable ({exc})", flush=True)
            if verbose and chosen != ("hip" if self.device.type == "cuda" else "cpu"):
                print(f"[trellis] --backend auto: using the {chosen} encoder", flush=True)
            backend = chosen
        elif backend in ("hip", "cpu"):
            load_native(backend, verbose)
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "hip" and self.device.type != "cuda":
            raise ValueError("the hip encoder needs --device cuda")
        self.backend = backend
        if backend == "hip":
            props = torch.cuda.get_device_properties(self.device)
            self.grid = grid or 4 * int(props.multi_processor_count)
            self.tiles_per_launch = tiles_per_launch or 4 * self.grid
        else:
            self.grid, self.tiles_per_launch = 0, 0

    def describe(self) -> dict:
        return {"backend": self.backend, "device": str(self.device), "codebook": self.codebook,
                "threads": self.threads if self.backend == "cpu" else None,
                "grid": self.grid or None, "tiles_per_launch": self.tiles_per_launch or None,
                "cost_dtype": "float32", "target_rounding": "fp16 (round to nearest even)",
                "selfcheck": getattr(self, "selfcheck", None),
                "tiles_encoded": self.tiles, "encoder_seconds": self.seconds}

    @staticmethod
    def round_targets(x: torch.Tensor) -> torch.Tensor:
        return x.float().clamp(-65504.0, 65504.0).half().float()

    def encode(self, x: torch.Tensor, K: float) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T, 256] (any float dtype, any device) -> (states [T, 256] int32, cost [T] float32),
        both on self.device."""
        if x.dim() != 2 or x.shape[1] != TILE:
            raise ValueError(f"encode wants [T, {TILE}], got {tuple(x.shape)}")
        t0 = time.perf_counter()
        T = x.shape[0]
        xr = self.round_targets(x.to(self.device))
        ka, mask = frac_k(K)
        backend = self.backend
        if backend == "hip" and not hip_widths_ok(K):
            backend = "torch"
            self._warn_fallback(K)
        if T == 0:
            st = torch.empty(0, TILE, dtype=torch.int32, device=self.device)
            cost = torch.empty(0, dtype=torch.float32, device=self.device)
        elif backend == "hip":
            st, cost = self._encode_hip(xr.contiguous(), ka, mask)
        elif backend == "cpu":
            st, cost = self._encode_cpu(xr, ka, mask)
        else:
            st, cost = self._encode_torch(xr, K)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.tiles += T
        self.seconds += time.perf_counter() - t0
        return st, cost

    def quantize(self, x: torch.Tensor, K: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(decoded values [T, 256] float32, states [T, 256] int32)."""
        st, _ = self.encode(x, K)
        return self.cb[st.long()], st

    def _warn_fallback(self, K) -> None:
        if K not in self._fallback_warned and self.verbose:
            print(f"[trellis] K={K} is outside the hip encoder's widths (3..6): torch path", flush=True)
        self._fallback_warned.add(K)

    def _encode_cpu(self, xr: torch.Tensor, ka: int, mask: int):
        lib = load_native("cpu")
        xc = np.ascontiguousarray(xr.cpu().numpy(), dtype=np.float32)
        T = xc.shape[0]
        st = np.empty((T, TILE), dtype=np.int32)
        cost = np.empty(T, dtype=np.float32)
        rc = lib.r4dx_trellis_viterbi_cpu(xc.ctypes.data, T, ka, mask, self.cb_np.ctypes.data,
                                          st.ctypes.data, cost.ctypes.data, int(self.threads))
        if rc != 0:
            raise RuntimeError(f"r4dx_trellis_viterbi_cpu returned {rc}")
        return torch.from_numpy(st).to(self.device), torch.from_numpy(cost).to(self.device)

    def _encode_hip(self, xr: torch.Tensor, ka: int, mask: int):
        lib = load_native("hip")
        T = xr.shape[0]
        st = torch.empty(T, TILE, dtype=torch.int32, device=self.device)
        cost = torch.empty(T, dtype=torch.float32, device=self.device)
        grid = min(self.grid, T)
        need = int(lib.r4dx_trellis_scratch_bytes(ka, mask, grid))
        if need <= 0:
            raise RuntimeError(f"hip encoder refuses widths ka={ka} mask={mask:#x}")
        if self._scratch is None or self._scratch.numel() < need:
            self._scratch = torch.empty(need, dtype=torch.uint8, device=self.device)
        cb_mode = 0 if self.codebook == "mul1" else 1
        stream = torch.cuda.current_stream(self.device).cuda_stream
        # Bounded launches: a few tiles per workgroup each, so no single kernel runs long.
        for t0 in range(0, T, self.tiles_per_launch):
            t1 = min(T, t0 + self.tiles_per_launch)
            rc = lib.r4dx_trellis_viterbi_hip(
                xr[t0:t1].data_ptr(), t1 - t0, ka, mask, cb_mode,
                self.cb.data_ptr() if cb_mode else None, st[t0:t1].data_ptr(),
                cost[t0:t1].data_ptr(), self._scratch.data_ptr(), need, min(grid, t1 - t0), stream)
            if rc != 0:
                raise RuntimeError(f"r4dx_trellis_viterbi_hip returned {rc}")
        return st, cost

    def check_against_reference(self, Ks, tiles: int = 32) -> dict:
        """Refuse to use a native encoder that disagrees with the reference on this machine: the
        hip encoder must match the native CPU one (itself tested bit for bit against viterbi_torch
        by tests/reference/test_trellis_quant.py), the CPU one must match viterbi_torch. States and
        costs must be identical for every K in `Ks`. Raises SystemExit otherwise."""
        if self.backend == "torch":
            return {"checked": False}
        ref_kind = "cpu" if self.backend == "hip" else "torch"
        ref = TrellisEncoder(ref_kind, "cpu", codebook=self.codebook, verbose=False)
        rng = np.random.default_rng(1234)
        tiles0, secs0 = self.tiles, self.seconds
        res = {}
        for K in sorted(set(float(k) for k in Ks)):
            x = torch.from_numpy(rng.standard_normal((tiles, TILE)).astype(np.float32) * 0.9)
            st, c = self.encode(x, K)
            st_r, c_r = ref.encode(x, K)
            same = torch.equal(st.cpu(), st_r.cpu()) and torch.equal(c.cpu(), c_r.cpu())
            res[str(K)] = same
            if not same:
                raise SystemExit(f"[trellis] the {self.backend} encoder disagrees with the {ref_kind} "
                                 f"reference at K={K} on {tiles} random tiles: do not trust it. Rerun "
                                 f"with --backend cpu (native CPU encoder, ~7k tiles/s) or --backend torch")
        self.tiles, self.seconds = tiles0, secs0
        if self.verbose:
            print(f"[trellis] {self.backend} encoder matches the {ref_kind} reference bit for bit "
                  f"at K in {sorted(res)} ({tiles} tiles each)", flush=True)
        self.selfcheck = {"checked": True, "reference": ref_kind, "tiles": tiles, "K": sorted(res)}
        return self.selfcheck

    def _encode_torch(self, xr: torch.Tensor, K: float):
        D = widths(K)
        emax = 1 << (STATE_BITS - min(D))
        # history TILE * emax bytes per tile plus ~3 x 65536 floats of temporaries per tile
        per_tile = TILE * emax + 3 * 4 * (1 << STATE_BITS)
        chunk = max(1, int((1 << 30) // per_tile))
        sts, costs = [], []
        for t0 in range(0, xr.shape[0], chunk):
            s, c = viterbi_torch(xr[t0:t0 + chunk], self.cb, D)
            sts.append(s.to(torch.int32))
            costs.append(c)
        return torch.cat(sts), torch.cat(costs)


# --------------------------------------------------------------------------------------------
# The stored bitstream (docs/trellis.md 5)
# --------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _ring_tables(K: float) -> dict:
    D = np.array(widths(K), dtype=np.int64)
    s_end = np.cumsum(D)                      # S(p): one past the last bit position p owns
    R = int(s_end[-1])                         # 256 K bits
    pos = np.repeat(np.arange(TILE), D)        # owner of stream bit q
    shift = s_end[pos] - 1 - np.arange(R)      # MSB-first inside each owner's field
    start = (s_end - STATE_BITS) % R           # window of state(p): 16 bits ending at S(p)
    nw = R // 32
    i0 = start // 32
    return {"D": D, "R": R, "nw": nw, "pos": pos, "shift": shift, "i0": i0, "i1": (i0 + 1) % nw,
            "off": start % 32}


def pack_states(states: torch.Tensor, K: float, chunk: int = 1 << 14) -> torch.Tensor:
    """[T, 256] states -> [T, 8K] int32 words (uint32 bit patterns): position p's own D(p) bits
    (the low bits of its state) MSB-first at stream bits [S(p) - D(p), S(p)); word w holds stream
    bits 32w .. 32w + 31 with the earliest at bit 31 (docs/trellis.md 5.1, canonical form).
    `chunk` tiles at a time (the bit temporaries are 8 bytes per stream bit)."""
    tb = _ring_tables(K)
    dev = states.device
    pos = torch.from_numpy(tb["pos"]).to(dev)
    shift = torch.from_numpy(tb["shift"]).to(dev)
    weights = (torch.ones(32, dtype=torch.long, device=dev) << torch.arange(31, -1, -1, device=dev))
    T = states.shape[0]
    out = torch.empty(T, tb["nw"], dtype=torch.int32, device=dev)
    for t0 in range(0, T, chunk):
        st = states[t0:t0 + chunk].long()
        bits = (st[:, pos] >> shift) & 1
        words = (bits.view(st.shape[0], tb["nw"], 32) * weights).sum(dim=2)
        out[t0:t0 + chunk] = torch.where(words >= (1 << 31), words - (1 << 32), words).to(torch.int32)
    return out


def roundtrip_ok(words: torch.Tensor, states: torch.Tensor, K: float, chunk: int = 1 << 16) -> bool:
    """unpack(words) == states over [T, .] tensors, `chunk` tiles at a time."""
    for t0 in range(0, states.shape[0], chunk):
        if not torch.equal(unpack_states(words[t0:t0 + chunk], K), states[t0:t0 + chunk].long()):
            return False
    return True


def unpack_states(words: torch.Tensor, K: float) -> torch.Tensor:
    """[..., 8K] int32 words -> [..., 256] int64 states: state(p) = the 16 stream bits ending at S(p)
    (mod the ring), read from two consecutive uint32 words (EXL3's funnel shift)."""
    tb = _ring_tables(K)
    dev = words.device
    lead = words.shape[:-1]
    w = words.reshape(-1, words.shape[-1]).long() & 0xFFFFFFFF
    i0 = torch.from_numpy(tb["i0"]).to(dev)
    i1 = torch.from_numpy(tb["i1"]).to(dev)
    off = torch.from_numpy(tb["off"]).to(dev)
    w0, w1 = w[:, i0], w[:, i1]
    st = (((w0 << off) & 0xFFFFFFFF) >> 16) | (w1 >> (48 - off))
    return (st & 0xFFFF).reshape(*lead, TILE)


def rows_to_tiles(rows: torch.Tensor) -> torch.Tensor:
    """[16, n] -> [n/16, 256] row-major tiles (Q::ldlq)."""
    n = rows.shape[1]
    return rows.reshape(16, n // 16, 16).permute(1, 0, 2).reshape(n // 16, TILE)


def tiles_to_matrix(tiles: torch.Tensor) -> torch.Tensor:
    """[tk, tn, 256] row-major tiles -> [16 tk, 16 tn]."""
    tk, tn = tiles.shape[:2]
    return tiles.reshape(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def matrix_to_tiles(W: torch.Tensor) -> torch.Tensor:
    k, n = W.shape
    return W.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n // 16, TILE)


def decode_words(words: torch.Tensor, K: float, cb: torch.Tensor, chunk: int = 1 << 16) -> torch.Tensor:
    """[tk, tn, 8K] words -> the regularized-domain matrix Q [16 tk, 16 tn] float32 (tiles decoded
    `chunk` at a time: the int64 unpacking temporaries are ~8 KiB per tile)."""
    tk, tn = words.shape[:2]
    flat = words.reshape(tk * tn, -1)
    cb = cb.to(words.device)
    inv = _perm_t(str(words.device), True)
    vals = torch.empty(tk * tn, TILE, dtype=torch.float32, device=words.device)
    for t0 in range(0, tk * tn, chunk):
        st = unpack_states(flat[t0:t0 + chunk], K)
        vals[t0:t0 + chunk] = cb[st][:, inv]
        del st
    return tiles_to_matrix(vals.reshape(tk, tn, TILE))


# --------------------------------------------------------------------------------------------
# Hadamards (docs/trellis.md 1.2, 8.2)
# --------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def hadamard128(device: str) -> torch.Tensor:
    """P_128: natural-order Sylvester Hadamard / sqrt(128), P[i][j] = (-1)^popcount(i & j) / sqrt(128)."""
    i = np.arange(HAD)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    P = (1.0 - 2.0 * par) / math.sqrt(HAD)
    return torch.tensor(P, dtype=torch.float32, device=device)


def had_l(W: torch.Tensor) -> torch.Tensor:
    """P_k W (blockwise along rows)."""
    k, n = W.shape
    P = hadamard128(str(W.device))
    return torch.matmul(P, W.reshape(k // HAD, HAD, n)).reshape(k, n)


def had_r(W: torch.Tensor) -> torch.Tensor:
    """W P_n (blockwise along columns)."""
    k, n = W.shape
    P = hadamard128(str(W.device))
    return torch.matmul(W.reshape(k, n // HAD, HAD), P).reshape(k, n)


def random_signs(n: int, seed: int) -> torch.Tensor:
    """(sign(randn) + 1e-5).sign(): +-1, never 0 (docs/trellis.md 8.1), from our own seed."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randn(n, generator=g).sign() + 1e-5).sign()


# --------------------------------------------------------------------------------------------
# Hessian (docs/trellis.md 7)
# --------------------------------------------------------------------------------------------

HESS_HEADER = struct.Struct("<8sIIQd32x")


def read_hess(path: Path, device: str | torch.device = "cpu") -> tuple[torch.Tensor, int]:
    """A hessian_capture.py `.hess` file -> (full symmetric H [K, K] float32 on `device`, rows).
    H is already X^T X / rows (hessian.json `semantics`)."""
    path = Path(path)
    with open(path, "rb") as f:
        magic, k, flags, rows, trace = HESS_HEADER.unpack(f.read(HESS_HEADER.size))
    if magic != b"R4DXHES1" or flags != 1:
        raise ValueError(f"{path}: not an R4DXHES1 packed-upper file")
    packed = np.fromfile(path, dtype="<f4", offset=HESS_HEADER.size)
    if packed.size != k * (k + 1) // 2:
        raise ValueError(f"{path}: {packed.size} values, K={k} needs {k * (k + 1) // 2}")
    up = np.zeros((k, k), dtype=np.float32)
    off = 0
    for i in range(k):
        m = k - i
        up[i, i:] = packed[off:off + m]
        off += m
    got = float(np.diagonal(up).astype(np.float64).sum())
    if not math.isclose(got, trace, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError(f"{path}: header trace {trace} but the diagonal sums to {got}")
    H = torch.from_numpy(up).to(device)
    H = H + torch.triu(H, 1).T  # the lower triangle of `up` is zero
    return H, int(rows)


def damp_hessian(H: torch.Tensor, sigma_reg: float = SIGMA_REG) -> torch.Tensor:
    """H + sigma_reg * mean(diag H) * I (Q::finalize_capture_H)."""
    dm = float(torch.diagonal(H).double().mean())
    if not math.isfinite(dm) or dm < 1e-20:
        raise ValueError(f"mean(diag H) = {dm}: no usable Hessian (EXL3 would take q_fallback)")
    Hd = H.clone()
    Hd.diagonal().add_(sigma_reg * dm)
    return Hd


def rotate_hessian(H: torch.Tensor, su: torch.Tensor) -> torch.Tensor:
    """P_k diag(su) H diag(su) P_k."""
    Hs = H * su[:, None] * su[None, :]
    return had_l(had_r(Hs))


def block_ldl(H: torch.Tensor, b: int = 16, sigma_reg: float = SIGMA_REG) -> tuple[torch.Tensor, int]:
    """Q::block_ldl: H = L D L^T with L unit block-lower (16 x 16 identity blocks), returned with
    its diagonal blocks ZEROED (the finalize step's `L[i, i] = 0`), i.e. the strictly block-lower
    feedback matrix of 7.3. Cholesky failures retry up to 10 times adding 2 sigma_reg mean(diag)
    to the diagonal each time. Returns (L, retries)."""
    k = H.shape[0]
    Hc = H
    retries = 0
    while True:
        L, info = torch.linalg.cholesky_ex(Hc)
        if int(info) == 0 and bool(torch.isfinite(L).all()):
            break
        retries += 1
        if retries > 10:
            raise RuntimeError(f"block_ldl: Cholesky of a {k} x {k} Hessian failed 11 times")
        dm = float(torch.diagonal(H).double().mean())
        Hc = Hc.clone()
        Hc.diagonal().add_(2.0 * sigma_reg * dm)
    m = k // b
    L3 = L.reshape(k, m, b)
    DL = torch.stack([L[i * b:(i + 1) * b, i * b:(i + 1) * b] for i in range(m)])  # [m, b, b]
    inv = torch.linalg.inv(DL)
    L3 = torch.einsum("kmj,mjl->kml", L3, inv)
    L = L3.reshape(k, k)
    for i in range(m):
        L[i * b:(i + 1) * b, i * b:(i + 1) * b] = 0.0
    return L, retries


# --------------------------------------------------------------------------------------------
# regularize + global scale (docs/trellis.md 8.2, 8.3)
# --------------------------------------------------------------------------------------------


def sample_scale_tiles(W_r: torch.Tensor, K: float, width: int = 3) -> torch.Tensor:
    """Q::sample_scale_tiles, in EXL3's exact order (stage 1 of the search scores every third sample,
    so the order decides which tiles it sees): the wrapped diagonal of width 3, i-major (tiles
    (i mod tk, (i + w) mod tn) for i = 0, 1, ..., w = 0, 1, 2 innermost, EXL3's repeat_interleave /
    repeat), then the num_x highest-mean-square tiles in descending order and the num_x lowest in
    ascending order (torch.topk), in tensor-core order, times the LDLQ drift factor."""
    k, n = W_r.shape
    tk, tn = k // 16, n // 16
    dev = W_r.device
    w4 = W_r.reshape(tk, 16, tn, 16)
    diag_len = max(tk, tn)
    ii = torch.arange(diag_len, device=dev).repeat_interleave(width)
    ww = torch.arange(width, device=dev).repeat(diag_len)
    kk, nn_ = ii % tk, (ii + ww) % tn
    tile_ms = w4.square().mean(dim=(1, 3)).flatten()
    num_x = min(max(8, (diag_len * width) // 16), (tile_ms.shape[0] + 1) // 2)
    hi = torch.topk(tile_ms, num_x).indices
    lo = torch.topk(tile_ms, num_x, largest=False).indices
    x = torch.cat((hi, lo))
    s = w4[torch.cat((kk, x // tn)), :, torch.cat((nn_, x % tn)), :].reshape(-1, TILE)
    return s[:, _perm_t(str(dev))].contiguous() * LDLQ_DRIFT.get(float(K), 1.0)


def g_scale_search(W_r: torch.Tensor, K: float, enc: TrellisEncoder) -> tuple[float, dict]:
    """Q::g_scale_gss: objective mean((viterbi(t s) / s - t)^2) over the sample tiles; stage 1 on
    s in {0.1, 0.3, ..., 1.9} over every third tile, stage 2 on c + 0.075 (i - 2), i = 0..4, over all
    of them, then a clamped parabolic step. Returns (g, record)."""
    t = sample_scale_tiles(W_r, K)

    def errs(scales, tiles):
        x = torch.cat([tiles * s for s in scales])
        q, _ = enc.quantize(x, K)
        q = q.reshape(len(scales), tiles.shape[0], TILE)
        return [float((((q[j] / s) - tiles) ** 2).mean()) for j, s in enumerate(scales)]

    s1 = [0.1 + 0.2 * i for i in range(10)]
    e1 = errs(s1, t[::3])
    c = s1[int(np.argmin(e1))]
    s2 = [c + 0.075 * (i - 2) for i in range(5)]
    e2 = errs(s2, t)
    bi = int(np.argmin(e2))
    off = 0.0
    if 0 < bi < 4:
        y0, y1, y2 = e2[bi - 1], e2[bi], e2[bi + 1]
        den = y0 - 2.0 * y1 + y2
        if den > 0:
            off = max(-0.5, min(0.5, 0.5 * (y0 - y2) / den)) * 0.075
    g = max(s2[bi] + off, 0.01)
    return g, {"g": g, "samples": int(t.shape[0]), "stage1": dict(zip([round(v, 3) for v in s1], e1)),
               "stage2": dict(zip([round(v, 4) for v in s2], e2)), "parabola_offset": off}


def regularize(W: torch.Tensor, su_sign: torch.Tensor, sv_sign: torch.Tensor, K: float,
               enc: TrellisEncoder, out_scales: bool = True):
    """Q::regularize, exact order (docs/trellis.md 8.2). W [k, n] (in, out) float32 on the device.
    Returns (W_r, su, sv, record)."""
    W = W.clone()
    c = torch.sqrt((W * W).mean(dim=0))
    mc = float(c.mean())
    if mc > 1e-30:
        c = c / mc
    zero = c < 1e-30
    if out_scales:
        c = torch.where(zero, torch.full_like(c, 0.1), c)
        sv = sv_sign * c + 1e-10
    else:
        sv = sv_sign.clone()
    W = W / sv[None, :]
    sv = torch.where(zero, torch.zeros_like(sv), sv)
    W = had_r(W)
    r = torch.sqrt((W * W).mean(dim=1))
    r = torch.where(r < 1e-30, torch.full_like(r, 0.1), r)
    su = su_sign * r / (-CODEBOOK_SCALE) + 1e-10
    W = W / su[:, None]
    W = had_l(W)
    g, grec = g_scale_search(W, K, enc)
    W = W * g
    su = su / g
    return W, su, sv, {"g_scale": grec, "dead_out_channels": int(zero.sum())}


# --------------------------------------------------------------------------------------------
# LDLQ (docs/trellis.md 7.3)
# --------------------------------------------------------------------------------------------


def ldlq(W_r: torch.Tensor, L: torch.Tensor, K: float, enc: TrellisEncoder,
         buf: int = BUF_K) -> tuple[torch.Tensor, torch.Tensor]:
    """Q::ldlq: 16-row blocks from the LAST to the first, lazy spans of `buf` rows; each block's
    n/16 tiles are Viterbi-encoded on the feedback-corrected values W_r[c] + sum_{r > c, outside the
    block} L[r, c]^T (W_r - Q)[r]. Returns (Q [k, n], states [k/16, n/16, 256] int32)."""
    k, n = W_r.shape
    if k % 16 or n % 16 or k % buf:
        raise ValueError(f"ldlq: shape {k} x {n} not a multiple of 16 / buf {buf}")
    dev = W_r.device
    perm = _perm_t(str(dev))
    perm_inv = _perm_t(str(dev), True)
    Q = torch.zeros_like(W_r)
    E = torch.zeros_like(W_r)
    prod = torch.zeros_like(W_r)
    states = torch.empty(k // 16, n // 16, TILE, dtype=torch.int32, device=dev)
    for j in range(k, 0, -buf):
        i = j - buf
        for bj in range(buf, 0, -16):
            bi = bj - 16
            r0, r1 = i + bi, i + bj
            comp = prod[r0:r1] + L[r1:j, r0:r1].T @ E[r1:j]
            rows = W_r[r0:r1] + comp
            tiles = rows_to_tiles(rows)[:, perm]
            vals, st = enc.quantize(tiles, K)
            q = vals[:, perm_inv].reshape(n // 16, 16, 16).permute(1, 0, 2).reshape(16, n)
            Q[r0:r1] = q
            E[r0:r1] = W_r[r0:r1] - q
            states[r0 // 16] = st
        if i > 0:
            prod[:i] += L[i:j, :i].T @ E[i:j]
    return Q, states


# --------------------------------------------------------------------------------------------
# Un-rotate, refit, proxy (docs/trellis.md 7.5, 8.4)
# --------------------------------------------------------------------------------------------


def unrotate(Q: torch.Tensor, su: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
    """W_hat = diag(su) P_k Q P_n diag(sv)."""
    return had_r(had_l(Q) * su[:, None]) * sv[None, :]


def dot64(a: torch.Tensor, b: torch.Tensor) -> float:
    """sum(a * b), fp32 products accumulated in fp64 (an fp32 sum over ~1e8 elements drifts by
    ~1e-4 relative, enough to push a cosine above 1)."""
    return float((a.float() * b.float()).sum(dtype=torch.float64))


def rel_cos(W: torch.Tensor, W_hat: torch.Tensor) -> tuple[float, float]:
    """(||W_hat - W|| / ||W||, cosine(W, W_hat)), fp64 accumulation."""
    ww, hh, wh = dot64(W, W), dot64(W_hat, W_hat), dot64(W, W_hat)
    E = W.float() - W_hat.float()
    rel = math.sqrt(dot64(E, E) / ww) if ww > 0 else 0.0
    cos = wh / math.sqrt(ww * hh) if ww > 0 and hh > 0 else 0.0
    return rel, cos


def hessian_energy(W: torch.Tensor, H: torch.Tensor) -> float:
    """tr(W^T H W): fp32 product, fp64 reduction."""
    return float(((H @ W) * W).sum(dtype=torch.float64))


def proxy_error(W: torch.Tensor, W_hat: torch.Tensor, H: torch.Tensor, den: float | None = None) -> float:
    """tr(E^T H E) / tr(W^T H W), E = W - W_hat, W [k, n] (in, out), H [k, k]. fp32 products
    (fp64 would be ~30x slower on the R9700), fp64 reductions; `den` may be passed precomputed."""
    E = (W - W_hat).float()
    num = float(((H @ E) * E).sum(dtype=torch.float64))
    den = hessian_energy(W.float(), H) if den is None else den
    return num / den if den > 0 else float("nan")


def refit_scales(W: torch.Tensor, Q: torch.Tensor, su: torch.Tensor, sv: torch.Tensor,
                 H: torch.Tensor, rounds: int = REFIT_ROUNDS):
    """Q::refit_scales in the original basis with the DAMPED H: per round, the closed-form column
    refit c_j = q_j^T H w_j / q_j^T H q_j, then the row refit r = solve((Q Q^T) o H + 1e-6 mean I,
    rowsum(Q o (H W))). The trellis never changes; only su / sv move. Returns (Q, su, sv, record)."""
    HW = H @ W
    rec = []
    for _ in range(rounds):
        HQ = H @ Q
        num = (Q * HW).sum(dim=0)
        den = (Q * HQ).sum(dim=0)
        c = torch.where(den > 1e-30, num / torch.where(den > 1e-30, den, torch.ones_like(den)),
                        torch.ones_like(den))
        Q = Q * c[None, :]
        sv = sv * c
        A = (Q @ Q.T) * H
        A.diagonal().add_(1e-6 * float(torch.diagonal(A).mean()))
        b = (Q * HW).sum(dim=1)
        r = torch.linalg.solve(A, b)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        Q = Q * r[:, None]
        su = su * r
        rec.append({"col_scale_range": [float(c.min()), float(c.max())],
                    "row_scale_range": [float(r.min()), float(r.max())]})
    return Q, su, sv, rec


def reconstruct(words: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: float,
                cb: torch.Tensor) -> torch.Tensor:
    """The stored form -> W_hat [k, n] (in, out) float32 in the ORIGINAL basis:
    diag(suh) P_k decode(words) P_n diag(svh) (docs/trellis.md 8.4)."""
    Q = decode_words(words, K, cb)
    return unrotate(Q, suh.float(), svh.float())


# --------------------------------------------------------------------------------------------
# One shared-Hessian group of linears (docs/trellis.md 2, 7.4)
# --------------------------------------------------------------------------------------------


def quantize_group(weights: dict[str, torch.Tensor], H: torch.Tensor, K: dict[str, float],
                   enc: TrellisEncoder, seed_group: int, seeds: dict[str, int],
                   hessian_basis: str = "exl3", verbose: bool = True) -> dict[str, dict]:
    """EXL3's pipeline for linears that read the same input (and so share H).

    weights: name -> HF weight [out, in] (any dtype/device); H: the undamped [in, in] Hessian on the
    compute device; K: name -> rate; seed_group: the input-sign seed (one per shared H, as EXL3 draws
    su in finalize_capture_H); seeds: name -> output-sign seed. hessian_basis 'exl3' factors the
    sign-only rotated H once for the group (exact EXL3); 'matched' factors P diag(su) H diag(su) P
    per tensor, su with its magnitudes and g -- the Hessian of the W_r LDLQ actually quantizes
    (docs/trellis.md 7.3 caveat, Q7; 7-73% lower proxy on attention q/k/v, 14 (b)).

    With 'exl3', tensors of equal K are regularized one by one and then run through ONE LDLQ pass
    concatenated along n (Q::quantize_exl3_batch): exactly separate passes, since LDLQ never mixes
    columns, but with n/16 of every tensor's tiles per encoder call.

    Returns name -> {"words" [k/16, n/16, 8K] int32, "suh" [k] fp16, "svh" [n] fp16, "record"}."""
    dev = H.device
    k = H.shape[0]
    t_group = time.perf_counter()
    if hessian_basis not in ("exl3", "matched"):
        raise ValueError(f"hessian_basis {hessian_basis!r}")
    Hd = damp_hessian(H)
    su_sign = random_signs(k, seed_group).to(dev)

    # 1. regularize (incoherence, scales, global scale search) every tensor
    prep: dict[str, dict] = {}
    for name, w_hf in weights.items():
        t0 = time.perf_counter()
        W = w_hf.to(device=dev, dtype=torch.float32).T.contiguous()  # [k, n]
        if W.shape[0] != k:
            raise ValueError(f"{name}: in_features {W.shape[0]} != Hessian K {k}")
        sv_sign = random_signs(W.shape[1], seeds[name]).to(dev)
        tiles_before = enc.tiles
        W_r, su, sv, rrec = regularize(W, su_sign, sv_sign, K[name], enc)
        prep[name] = {"W": W, "W_r": W_r, "su": su, "sv": sv, "rrec": rrec,
                      "tiles_scale": enc.tiles - tiles_before, "t_reg": time.perf_counter() - t0}

    # 2. LDLQ: one pass per K over the concatenated tensors (exl3), or per tensor (matched)
    batches: list[list[str]] = []
    if hessian_basis == "exl3":
        for Kv in sorted(set(K[n] for n in weights)):
            batches.append([n for n in weights if K[n] == Kv])
        L_shared, retries_shared = block_ldl(rotate_hessian(Hd, su_sign))
    else:
        batches = [[n] for n in weights]
        L_shared, retries_shared = None, 0
    for batch in batches:
        t1 = time.perf_counter()
        if L_shared is not None:
            L, retries = L_shared, retries_shared
        else:
            L, retries = block_ldl(rotate_hessian(Hd, prep[batch[0]]["su"]))
        W_cat = torch.cat([prep[n]["W_r"] for n in batch], dim=1) if len(batch) > 1 else prep[batch[0]]["W_r"]
        Qr, states = ldlq(W_cat, L, K[batch[0]], enc)
        del W_cat
        if L_shared is None:
            del L
        dt_ldlq = time.perf_counter() - t1
        c0 = 0
        for n in batch:
            nn_ = prep[n]["W_r"].shape[1]
            prep[n]["Qr"] = Qr[:, c0:c0 + nn_]
            prep[n]["states"] = states[:, c0 // 16:(c0 + nn_) // 16]
            prep[n]["ldl_retries"] = retries
            prep[n]["t_ldlq"] = dt_ldlq
            prep[n]["ldlq_batch"] = batch
            c0 += nn_
        del Qr, states
    del L_shared

    # 3. un-rotate, refit, pack, reconstruct from the stored bits, record
    out: dict[str, dict] = {}
    for name in weights:
        t2 = time.perf_counter()
        p = prep.pop(name)
        W, Kn = p["W"], K[name]
        n = W.shape[1]
        den_h, den_hd = hessian_energy(W, H), hessian_energy(W, Hd)
        W_hat0 = unrotate(p["Qr"], p["su"], p["sv"])
        proxy_pre = proxy_error(W, W_hat0, H, den_h)
        proxy_pre_d = proxy_error(W, W_hat0, Hd, den_hd)
        W_hat, su, sv, frec = refit_scales(W, W_hat0, p["su"], p["sv"], Hd)
        del W_hat0, W_hat
        suh, svh = su.half(), sv.half()
        tk, tn = k // 16, n // 16
        flat = p["states"].reshape(-1, TILE)
        words = pack_states(flat, Kn)
        if not roundtrip_ok(words, flat, Kn):
            raise RuntimeError(f"{name}: pack/unpack round trip failed (a ring did not close)")
        words = words.reshape(tk, tn, -1)
        W_st = reconstruct(words, suh, svh, Kn, enc.cb)  # from the stored bits only
        proxy = proxy_error(W, W_st, H, den_h)
        proxy_d = proxy_error(W, W_st, Hd, den_hd)
        rel, cos = rel_cos(W, W_st)
        bits_trellis = words.numel() * 32
        bits_scales = (suh.numel() + svh.numel()) * 16
        rrec = p["rrec"]
        rec = {
            "encoding": ENCODING_TRELLIS, "K": Kn, "codebook": enc.codebook, "k": k, "n": n,
            "shape_hf": [n, k], "words_shape": list(words.shape),
            "bits": {"trellis": bits_trellis, "scales": bits_scales, "marker": 32,
                     "total": bits_trellis + bits_scales + 32,
                     "bpw": (bits_trellis + bits_scales + 32) / (k * n),
                     "bpw_trellis": bits_trellis / (k * n)},
            "proxy": proxy, "proxy_damped_h": proxy_d, "proxy_before_refit": proxy_pre,
            "proxy_before_refit_damped_h": proxy_pre_d,
            "rel_weight_err": rel, "cos": cos,
            "hessian_basis": hessian_basis, "ldl_retries": p["ldl_retries"],
            "seeds": {"su": seed_group, "sv": seeds[name]},
            "regularize": rrec, "refit": frec,
            "tiles": {"ldlq": tk * tn, "scale_search": p["tiles_scale"]},
            "ldlq_batch": p["ldlq_batch"],
            "seconds": {"regularize": p["t_reg"], "ldlq_batch": p["t_ldlq"],
                        "finalize": time.perf_counter() - t2},
        }
        if verbose:
            print(f"[trellis]   {name}: K={Kn} {k}x{n} g={rrec['g_scale']['g']:.4f} "
                  f"proxy {proxy_pre:.5f} -> refit {proxy:.5f} rel {rel:.4f} "
                  f"bpw {rec['bits']['bpw']:.4f}", flush=True)
        out[name] = {"words": words.cpu(), "suh": suh.cpu(), "svh": svh.cpu(), "record": rec}
        del W, W_st, p
    if verbose:
        print(f"[trellis]   group done in {time.perf_counter() - t_group:.1f}s", flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Our converter's baseline: w4a16 int4 g64 LDLQ (src/convert/include/r4dx_convert/quant_ldlq.hpp)
# --------------------------------------------------------------------------------------------


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _int4_group_search(wg: torch.Tensor, wt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SearchInt4AsymGroup (quant_search.hpp) vectorized over rows, refit = false: the RTN grid
    first, then 21 scales x 3 zeros, strict improvement. wg [N, g], wt [g]. -> (scale [N], zero [N])."""
    wmin = wg.min(dim=1).values
    wmax = wg.max(dim=1).values
    rng = wmax - wmin
    s0 = torch.clamp(rng, min=1e-12) / 15.0

    def score(sc, z):
        q = torch.clamp(_round_half_away(wg / sc[:, None]) + z[:, None], 0, 15)
        d = wg - sc[:, None] * (q - z[:, None])
        return (d * d * wt[None, :]).sum(dim=1)

    best_s = s0.clone()
    best_z = torch.clamp(_round_half_away(-wmin / s0), 0, 15)
    best_e = score(best_s, best_z)
    for step in range(21):
        sc = s0 * (0.85 + 0.3 * (step / 20.0))
        zc = _round_half_away(-wmin / sc)
        for dz in (-1, 0, 1):
            z = torch.clamp(zc + dz, 0, 15)
            e = score(sc, z)
            better = e < best_e
            best_e = torch.where(better, e, best_e)
            best_s = torch.where(better, sc, best_s)
            best_z = torch.where(better, z, best_z)
    degenerate = rng <= 0
    if bool(degenerate.any()):
        zero_grp = degenerate & (wmax == 0)
        const = degenerate & (wmax != 0)
        best_s = torch.where(zero_grp, torch.zeros_like(best_s), best_s)
        best_z = torch.where(zero_grp, torch.zeros_like(best_z), best_z)
        best_s = torch.where(const, wmax.abs(), best_s)
        best_z = torch.where(const, torch.where(wmax >= 0, torch.zeros_like(best_z),
                                                torch.full_like(best_z, 15.0)), best_z)
    return best_s, best_z


def int4_ldlq(W_hf: torch.Tensor, H: torch.Tensor, group: int = 64, damp: float = 0.01,
              block: int = 128) -> torch.Tensor:
    """quant_ldlq.hpp's QuantizeInt4AsymmetricLdlq in torch (not byte-exact: the group search's
    error sums run in a different order). W_hf [N, K] (out, in), H [K, K] undamped. U = chol(H_d^-1)
    upper via the reversal trick; columns in 128-blocks; the group's (scale, zero) chosen at its first
    column from the CURRENT (feedback-updated) weights with importance diag(H) undamped; lazy block
    update. Returns the dequantized weight [N, K] float32."""
    dev = H.device
    N, K = W_hf.shape
    # The fp64 factorization runs on the CPU: hipBLAS's fp64 TRSM fails at K = 17408 on the R9700
    # (and fp64 is slow there); it is a one-off O(K^3), the column loop below stays on `dev`.
    Hd = H.to(device="cpu", dtype=torch.float64, copy=True)
    dm = float(torch.diagonal(Hd).mean())
    Hd.diagonal().add_(damp * dm)
    J = torch.arange(K - 1, -1, -1)
    Lr = torch.linalg.cholesky(Hd[J][:, J])
    del Hd
    Linv = torch.linalg.solve_triangular(Lr, torch.eye(K, dtype=torch.float64), upper=False)
    del Lr
    U = Linv[J][:, J].float().to(dev)  # upper, H_d^-1 = U^T U
    del Linv
    diag_h = torch.diagonal(H).float()
    W = W_hf.to(device=dev, dtype=torch.float32).clone()
    out = torch.empty_like(W)
    for b0 in range(0, K, block):
        b1 = b0 + block
        Wb = W[:, b0:b1].clone()
        E = torch.zeros(N, block, dtype=torch.float32, device=dev)
        sc = z = None
        for i in range(b0, b1):
            li = i - b0
            if li % group == 0:
                sc, z = _int4_group_search(Wb[:, li:li + group], diag_h[i:i + group])
            x = Wb[:, li]
            safe = torch.where(sc > 0, sc, torch.ones_like(sc))
            q = torch.clamp(_round_half_away(x / safe) + z, 0, 15)
            dq = torch.where(sc > 0, sc * (q - z), torch.zeros_like(x))
            out[:, i] = dq
            err = (x - dq) / U[i, i]
            E[:, li] = err
            if li + 1 < block:
                Wb[:, li + 1:] -= err[:, None] * U[i, i + 1:b1][None, :]
        if b1 < K:
            W[:, b1:] -= E @ U[b0:b1, b1:]
    return out


# --------------------------------------------------------------------------------------------
# Allocation (docs/trellis.md 9)
# --------------------------------------------------------------------------------------------


def rate_floor(bpw: float) -> float:
    r = math.floor(2 * bpw) / 2 if bpw < 4 else math.floor(bpw)
    return float(min(max(r, 1), 8))


def rate_next(r: float) -> float | None:
    nr = r + (0.5 if r < 4 else 1.0)
    return None if nr > 8 else nr


def allocate(tensors: list[dict], bpw: float) -> dict[str, float]:
    """AL::create_q_strategy (docs/trellis.md 9). Tensors are grouped by (layer, qgroup) -- EXL3's
    qgroups: q/k/v one group, GDN in_proj_qkv + in_proj_z one, gate + up one, o_proj / out_proj /
    down_proj each alone (QGROUPS; a tensor without a "qgroup" is its own group) -- and every group
    has priority 0 (no qwen3_5 module sets q_priority). All start at rate_floor(bpw); then,
    repeatedly, groups sorted by (min(layer, last - layer), layer, first idx) are each raised one
    rate step when the WHOLE group's promotion fits int(bpw * numel)."""
    last = max(t["layer"] for t in tensors)
    budget = int(bpw * sum(t["numel"] for t in tensors))
    rate = {t["name"]: rate_floor(bpw) for t in tensors}
    used = sum(t["numel"] * rate[t["name"]] for t in tensors)
    groups: dict[tuple, list[dict]] = {}
    for t in sorted(tensors, key=lambda t: (t["layer"], t["idx"])):  # EXL3's traversal order
        groups.setdefault((t["layer"], t.get("qgroup") or t["name"]), []).append(t)
    order = sorted(groups.values(), key=lambda g: (min(g[0]["layer"], last - g[0]["layer"]),
                                                   g[0]["layer"], g[0]["idx"]))
    changed = True
    while changed:
        changed = False
        for g in order:
            extra = 0.0
            for t in g:
                nr = rate_next(rate[t["name"]])
                extra += 0.0 if nr is None else t["numel"] * (nr - rate[t["name"]])
            if extra > 0 and used + extra <= budget:
                for t in g:
                    nr = rate_next(rate[t["name"]])
                    if nr is not None:
                        rate[t["name"]] = nr
                used += extra
                changed = True
    return rate


def model_linears(model_dir: Path) -> list[dict]:
    """Every quantized decoder linear: name, layer, idx (module order), module, qgroup, numel, k, n."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
    from safetensors import safe_open

    shapes: dict[str, list[int]] = {}
    by_shard: dict[str, list[str]] = {}
    out = []
    for layer, lt in enumerate(tc["layer_types"]):
        for idx, mod in enumerate(module_list(lt)):
            name = hf_name(layer, mod)
            by_shard.setdefault(wm[name], []).append(name)
            out.append({"name": name, "layer": layer, "idx": idx, "module": mod, "layer_type": lt,
                        "qgroup": QGROUPS[mod]})
    for shard, names in by_shard.items():
        with safe_open(str(Path(model_dir) / shard), framework="pt", device="cpu") as f:
            for nm in names:
                shapes[nm] = list(f.get_slice(nm).get_shape())
    for t in out:
        n, k = shapes[t["name"]]
        t.update({"n": n, "k": k, "numel": n * k})
    return out


# --------------------------------------------------------------------------------------------
# The override directory (read by full_logits_golden.py --weights-override)
# --------------------------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != OVERRIDE_FORMAT or doc.get("version") != OVERRIDE_VERSION:
        raise ValueError(f"{path}: not a {OVERRIDE_FORMAT} v{OVERRIDE_VERSION} manifest")
    doc["_path"] = str(path.resolve())
    doc["_dir"] = str(path.resolve().parent)
    return doc


def tensor_file(manifest: dict, rec: dict) -> Path:
    p = Path(rec["file"])
    return p if p.is_absolute() else Path(manifest["_dir"]) / p


def load_override_tensor(manifest: dict, name: str, device, cb_cache: dict | None = None) -> torch.Tensor:
    """The HF-layout [out, in] float32 weight of `name` from an override directory, on `device`."""
    from safetensors import safe_open

    rec = manifest["tensors"][name]
    path = tensor_file(manifest, rec)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if rec["encoding"] == ENCODING_DENSE:
            return f.get_tensor(name).to(device=device, dtype=torch.float32)
        if rec["encoding"] != ENCODING_TRELLIS:
            raise ValueError(f"{name}: unknown encoding {rec['encoding']!r}")
        words = f.get_tensor(name + ".trellis").to(device)
        suh = f.get_tensor(name + ".suh").to(device)
        svh = f.get_tensor(name + ".svh").to(device)
    key = (rec["codebook"], str(device))
    cb_cache = {} if cb_cache is None else cb_cache
    if key not in cb_cache:
        cb_cache[key] = torch.from_numpy(codebook_np(rec["codebook"]).copy()).to(device)
    W_hat = reconstruct(words, suh, svh, rec["K"], cb_cache[key])
    return W_hat.T.contiguous()


def summarize_tensors(tensors: dict[str, dict]) -> dict:
    by_class: dict[str, dict] = {}
    tot_bits = tot_numel = tot_trellis = 0
    for name, rec in tensors.items():
        parsed = parse_hf_name(name)
        cls = parsed[1] if parsed else name
        numel = rec["k"] * rec["n"]
        c = by_class.setdefault(cls, {"n": 0, "numel": 0, "bits": 0, "K": {}})
        c["n"] += 1
        c["numel"] += numel
        c["bits"] += rec["bits"]["total"]
        kk = str(rec["K"])
        c["K"][kk] = c["K"].get(kk, 0) + 1
        tot_bits += rec["bits"]["total"]
        tot_trellis += rec["bits"]["trellis"]
        tot_numel += numel
    for c in by_class.values():
        c["bpw"] = c["bits"] / c["numel"]
    proxies = [r["proxy"] for r in tensors.values() if "proxy" in r]
    return {"n_tensors": len(tensors), "numel": tot_numel, "bits": tot_bits,
            "bpw": tot_bits / tot_numel if tot_numel else None,
            "bpw_trellis_only": tot_trellis / tot_numel if tot_numel else None,
            "decode_gib": tot_bits / 8 / 2**30,
            "proxy_mean": float(np.mean(proxies)) if proxies else None,
            "proxy_max": float(np.max(proxies)) if proxies else None,
            "by_class": dict(sorted(by_class.items()))}


def write_json_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, default=str)
    os.replace(tmp, path)


def git_rev() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def code_sha256() -> dict:
    """The quantizer's and the encoders' source hashes (part of quantize-model's resume key)."""
    return {"trellis_quant": sha256_path(Path(__file__)), "trellis_viterbi": sha256_path(NATIVE_SRC)}


def provenance(enc: TrellisEncoder | None = None) -> dict:
    return {"tool": "tools/reference/trellis_quant.py", "git_head": git_rev(),
            "trellis_quant_sha256": sha256_path(Path(__file__)),
            "trellis_viterbi_sha256": sha256_path(NATIVE_SRC),
            "torch_version": torch.__version__, "encoder": enc.describe() if enc else None,
            "spec": "docs/trellis.md (exllamav3 6b84a21, QTIP arXiv 2406.11235)"}


RECIPE = {
    "tile": "16x16, tensor-core order", "state_bits": STATE_BITS, "tail_biting": "EXL3 two-pass",
    "codebook": "mul1", "incoherence": "random signs + 128-point Sylvester Hadamard, both sides",
    "out_scales": True, "g_scale_search": "2-stage + parabola, LDLQ drift", "sigma_reg": SIGMA_REG,
    "ldlq": "block LDLQ, 16-row blocks last-to-first, lazy spans of 128",
    "refit_rounds": REFIT_ROUNDS,
    "hessians": "hessian-v2 'keys' (post-norm inputs of the bf16 forward; not sequential)",
    "allocation": ("EXL3's (docs/trellis.md 9): qgroups q/k/v, in_proj_qkv + in_proj_z and gate + up "
                   "promoted as units, o/out/down alone; every q_priority 0, as in EXL3's qwen3_5"),
    "deviations": [
        "fp32 Viterbi path costs (EXL3 fp16)", "smallest-edge final argmin tie rule",
        "own seeded sign draws", "bf16-forward non-sequential Hessians",
        # docs/trellis.md 7.1 / 13: why the oracle is probably optimistic vs a stock EXL3 conversion
        "calibration domain-matched to the KL corpus: hessian-v2 is 834,820 tokens (WikiText-2 262k, "
        "this repo's code 98k, text self-generated by the quantized model: chat 168k, english_prose "
        "115k, thai_prose 84k, code 70k, multilingual 31k; calib.txt + kv_calib_corpus 7k), text-disjoint from "
        "kl_corpus but covering its domains (this repo's C++/Python, English, Thai), with no uniform "
        "random-token rows; EXL3's default is 250 x 2048 = 512,000 tokens of generic text (c4, code, "
        "multilingual, technical, wiki, tiny) of which 39 rows (15.6%) are uniform random tokens. "
        "Likely biases the oracle's KL on tokens_canon low relative to a stock EXL3 conversion (not "
        "relative to quant2, which uses the same Hessians)",
        "lm_head and embeddings bf16",
    ],
}


# --------------------------------------------------------------------------------------------
# CLI: quantize-model
# --------------------------------------------------------------------------------------------


def _layer_groups(layer: int, layer_type: str) -> list[tuple[str, list[str]]]:
    """(tap, modules) per shared-H group of a layer."""
    groups: dict[str, list[str]] = {}
    for mod in module_list(layer_type):
        groups.setdefault(LINEARS[mod][1], []).append(mod)
    return list(groups.items())


def cmd_quantize_model(args) -> int:
    from common import ShardIndex, resolve_device, sha256_file  # noqa: E402
    from full_logits_golden import get_tensors_grouped  # noqa: E402

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    hdir = Path(args.hessian_dir)
    hman = json.loads((hdir / "hessian.json").read_text(encoding="utf-8"))
    keys = hman["keys"]
    linears = model_linears(model_dir)
    if args.bpw is not None:
        rates = allocate(linears, args.bpw)
    else:
        rates = {t["name"]: float(args.K) for t in linears}
    layers = sorted({t["layer"] for t in linears})
    if args.layers:
        want = set()
        for part in args.layers.split(","):
            a, _, b = part.partition("-")
            want.update(range(int(a), int(b or a) + 1))
        layers = [i for i in layers if i in want]
    enc = TrellisEncoder(args.backend, device, threads=args.threads, grid=args.grid)
    print(f"[trellis] encoder {enc.describe()}", flush=True)
    enc.check_against_reference(set(rates.values()))
    config_sha = sha256_file(model_dir / "config.json")
    hman_sha = sha256_file(hdir / "hessian.json")
    index = ShardIndex.load(model_dir)
    # The resume key: a layer written by other code (quantizer or encoder source) is redone, not
    # kept. The encoder backend is not part of it (hip, cpu and torch give identical states).
    job = {"model_dir": str(model_dir), "config_sha256": config_sha, "hessian_dir": str(hdir),
           "hessian_manifest_sha256": hman_sha, "hessian_basis": args.hessian_basis,
           "codebook": "mul1", "recipe": RECIPE, "code_sha256": code_sha256()}
    t_all = time.perf_counter()
    for layer in layers:
        lt = next(t["layer_type"] for t in linears if t["layer"] == layer)
        rec_path = out_dir / f"L{layer:02d}.json"
        st_path = out_dir / f"L{layer:02d}.safetensors"
        want_k = {hf_name(layer, m): rates[hf_name(layer, m)] for m in module_list(lt)}
        if rec_path.exists() and st_path.exists():
            old = json.loads(rec_path.read_text(encoding="utf-8"))
            if (layer_record_usable(old, job, rates)
                    and {n: float(r["K"]) for n, r in old["tensors"].items()} == want_k):
                print(f"[trellis] L{layer:02d}: done already, skipping", flush=True)
                continue
            print(f"[trellis] L{layer:02d}: existing output is for another job, code or rate; redoing",
                  flush=True)
        t_layer = time.perf_counter()
        names = [hf_name(layer, m) for m in module_list(lt)]
        raw = get_tensors_grouped(index, names)
        tensors_out: dict[str, torch.Tensor] = {}
        records: dict[str, dict] = {}
        for tap, mods in _layer_groups(layer, lt):
            ckey = f"text.layers.{layer}.{LINEARS[mods[0]][0]}"
            hfile = hdir / keys[ckey]
            t0 = time.perf_counter()
            H, rows = read_hess(hfile, device)
            print(f"[trellis] L{layer:02d} {tap}: {', '.join(mods)} <- {hfile.name} "
                  f"(K={H.shape[0]}, read {time.perf_counter() - t0:.1f}s)", flush=True)
            names_g = [hf_name(layer, m) for m in mods]
            res = quantize_group({n: raw[n] for n in names_g}, H, {n: rates[n] for n in names_g}, enc,
                                 seed_group=stable_seed("su", layer, tap),
                                 seeds={n: stable_seed("sv", n) for n in names_g},
                                 hessian_basis=args.hessian_basis)
            for n, r in res.items():
                tensors_out[n + ".trellis"] = r["words"].contiguous()
                tensors_out[n + ".suh"] = r["suh"].contiguous()
                tensors_out[n + ".svh"] = r["svh"].contiguous()
                r["record"]["hessian"] = {"file": hfile.name, "rows": rows, "key": ckey}
                r["record"]["file"] = st_path.name
                records[n] = r["record"]
            del H
            if device.type == "cuda":
                torch.cuda.empty_cache()
        from safetensors.torch import save_file

        tmp = st_path.with_name(st_path.name + ".tmp")
        save_file(tensors_out, str(tmp), metadata={"format": OVERRIDE_FORMAT, "layer": str(layer)})
        os.replace(tmp, st_path)
        file_sha = sha256_path(st_path)  # binds the manifest (and its sha256) to these bits
        for r in records.values():
            r["file_sha256"] = file_sha
        write_json_atomic(rec_path, {"layer": layer, "job": job, "tensors": records,
                                     "seconds": time.perf_counter() - t_layer,
                                     "provenance": provenance(enc)})
        print(f"[trellis] L{layer:02d} done in {time.perf_counter() - t_layer:.0f}s "
              f"(total {time.perf_counter() - t_all:.0f}s, encoder {enc.tiles} tiles "
              f"{enc.tiles / max(enc.seconds, 1e-9):.0f} tiles/s)", flush=True)
        write_manifest(out_dir, job, rates, args, enc)
    write_manifest(out_dir, job, rates, args, enc)
    return 0


def layer_record_usable(doc: dict, job: dict, rates: dict) -> bool:
    """An L<ii>.json belongs to this run: same job (checkpoint, Hessians, basis, recipe, code) and
    every tensor it holds is one this run quantizes, at exactly the K this run gives it."""
    if doc.get("job") != job:
        return False
    tens = doc.get("tensors") or {}
    return bool(tens) and all(n in rates and float(r.get("K")) == float(rates[n]) for n, r in tens.items())


def write_manifest(out_dir: Path, job: dict, rates: dict, args, enc=None) -> dict:
    """weights_override.json from every finished L*.json of this job and these rates in `out_dir`
    (a layer of another job, other code or another K is listed under `stale_layers`, not used)."""
    tensors: dict[str, dict] = {}
    layers_done, stale = [], []
    for p in sorted(out_dir.glob("L*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not layer_record_usable(doc, job, rates):
            stale.append(p.name)
            continue
        layers_done.append(doc["layer"])
        tensors.update(doc["tensors"])
    missing = sorted(n for n in rates if n not in tensors)
    man = {"format": OVERRIDE_FORMAT, "version": OVERRIDE_VERSION, "encoding": ENCODING_TRELLIS,
           "complete": not missing, "missing": missing[:32], "missing_count": len(missing),
           "bpw_target": getattr(args, "bpw", None), "K_uniform": getattr(args, "K", None),
           "layers_done": sorted(layers_done), "stale_layers": stale, **job,
           "summary": summarize_tensors(tensors), "tensors": tensors,
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "provenance": provenance(enc)}
    write_json_atomic(out_dir / MANIFEST_NAME, man)
    return man


# --------------------------------------------------------------------------------------------
# CLI: mix
# --------------------------------------------------------------------------------------------


def cmd_mix(args) -> int:
    """A bpw target from finished integer-rate directories: EXL3's allocation (section 9) decides
    each tensor's K; its record/file is taken from the directory quantized at that K (the trellis
    of a tensor depends only on the tensor, its H, K and seeds, so no requantization)."""
    srcs: dict[float, dict] = {}
    for s in args.src:
        man = load_manifest(Path(s))
        if not man.get("complete"):
            raise SystemExit(f"[trellis] {s}: manifest is not complete")
        ks = {float(r["K"]) for r in man["tensors"].values()}
        if len(ks) != 1:
            raise SystemExit(f"[trellis] {s}: not a uniform-rate directory ({sorted(ks)})")
        srcs[ks.pop()] = man
    base = next(iter(srcs.values()))
    job_keys = ("model_dir", "config_sha256", "hessian_manifest_sha256", "hessian_basis", "codebook")
    for man in srcs.values():
        for key in job_keys:
            if man.get(key) != base.get(key):
                raise SystemExit(f"[trellis] source directories differ in {key}")
    code = {str(K): m.get("code_sha256") for K, m in srcs.items()}
    if len({json.dumps(v, sort_keys=True) for v in code.values()}) > 1:
        print(f"[trellis] WARNING: the source directories were written by different trellis_quant.py / "
              f"trellis_viterbi.hip versions (recorded in the manifest): {code}", flush=True)
    linears = model_linears(Path(base["model_dir"]))
    rates = allocate(linears, args.bpw)
    need = sorted(set(rates.values()))
    lacking = [r for r in need if r not in srcs]
    if lacking:
        raise SystemExit(f"[trellis] --bpw {args.bpw} needs K in {need}; no source for {lacking}")
    tensors = {}
    for name, K in rates.items():
        man = srcs[K]
        rec = dict(man["tensors"][name])
        rec["file"] = str(tensor_file(man, rec))
        tensors[name] = rec
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for K in rates.values():
        counts[str(K)] = counts.get(str(K), 0) + 1
    man = {"format": OVERRIDE_FORMAT, "version": OVERRIDE_VERSION, "encoding": ENCODING_TRELLIS,
           "complete": True, "missing": [], "missing_count": 0, "bpw_target": args.bpw,
           "K_uniform": None, **{k: base.get(k) for k in job_keys}, "recipe": base.get("recipe"),
           "hessian_dir": base.get("hessian_dir"),
           "allocation": {"rule": "docs/trellis.md 9 (AL::create_q_strategy) with EXL3's qgroups (q/k/v, "
                                  "in_proj_qkv + in_proj_z, gate + up promoted as units; o/out/down "
                                  "alone), every priority 0", "tensors_per_K": counts,
                          "sources": {str(K): m["_path"] for K, m in srcs.items()},
                          "source_code_sha256": code},
           "summary": summarize_tensors(tensors), "tensors": tensors,
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": provenance()}
    write_json_atomic(out_dir / MANIFEST_NAME, man)
    s = man["summary"]
    print(f"[trellis] mix {args.bpw} bpw: {counts} -> {s['bpw']:.4f} bpw measured, "
          f"{s['decode_gib']:.3f} GiB -> {out_dir / MANIFEST_NAME}")
    return 0


# --------------------------------------------------------------------------------------------
# CLI: validation (a) gaussian, (b) linear, (c) bench; selftest
# --------------------------------------------------------------------------------------------


def lloyd_max_mse(bits: int, iters: int = 2000) -> float:
    """MSE of the optimal scalar (Lloyd-Max) quantizer of N(0, 1) at `bits` bits."""
    n = 1 << bits
    pdf = lambda v: math.exp(-0.5 * v * v) / math.sqrt(2 * math.pi)  # noqa: E731
    cdf = lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2)))           # noqa: E731
    c = [(-2.0 + 4.0 * (i + 0.5) / n) for i in range(n)]
    for _ in range(iters):
        t = [-math.inf] + [(c[i] + c[i + 1]) / 2 for i in range(n - 1)] + [math.inf]
        newc = []
        for i in range(n):
            a, b = t[i], t[i + 1]
            pa = 0.0 if a == -math.inf else pdf(a)
            pb = 0.0 if b == math.inf else pdf(b)
            mass = cdf(b) - cdf(a)
            newc.append((pa - pb) / mass)
        c = newc
    t = [-math.inf] + [(c[i] + c[i + 1]) / 2 for i in range(n - 1)] + [math.inf]
    ec2 = sum((cdf(t[i + 1]) - cdf(t[i])) * c[i] * c[i] for i in range(n))
    return 1.0 - ec2


QTIP_PUBLISHED = {  # docs/trellis.md 6.5 / 14: QTIP Table 1 (L = 16, 2 bit) and Table 2 (L = 12)
    1: {"table2_L12": 0.2803},
    2: {"table1_L16_1mad_3inst": 0.069, "table2_L12": 0.0733},
    3: {"table2_L12": 0.0198},
    4: {"table2_L12": 0.0055},
}


def cmd_gaussian(args) -> int:
    from common import resolve_device

    device = resolve_device(args.device)
    enc = TrellisEncoder(args.backend, device, codebook=args.codebook, threads=args.threads)
    rng = np.random.default_rng(args.seed)
    rows = []
    for K in [float(k) for k in args.K.split(",")]:
        x = torch.from_numpy(rng.standard_normal((args.tiles, TILE)).astype(np.float32)).to(device)
        std = float(enc.cb.std())
        scales = [std * (args.scale_lo + i * args.scale_step)
                  for i in range(int(round((args.scale_hi - args.scale_lo) / args.scale_step)) + 1)]
        t0 = time.perf_counter()
        xx = torch.cat([x * s for s in scales])
        q, st = enc.quantize(xx, K)
        q = q.reshape(len(scales), args.tiles, TILE)
        mses = [float(((q[j] / s - x) ** 2).mean()) for j, s in enumerate(scales)]
        j = int(np.argmin(mses))
        st_best = st.reshape(len(scales), args.tiles, TILE)[j]
        D = widths(K)
        closed = bool(ring_consistent(st_best.long().cpu(), D).all())
        words = pack_states(st_best.cpu(), K)
        rt = torch.equal(unpack_states(words, K), st_best.long().cpu())
        bpw = words.numel() * 32 / (args.tiles * TILE)
        lm = lloyd_max_mse(int(K)) if float(K).is_integer() else None
        row = {"K": K, "mse": mses[j], "best_scale_over_std": scales[j] / std,
               "bpw_measured": bpw, "rings_closed": closed, "pack_roundtrip": rt,
               "lloyd_max_mse": lm, "trellis_over_lloyd_max": (mses[j] / lm) if lm else None,
               "gaussian_D_R": 2.0 ** (-2 * K), "qtip": QTIP_PUBLISHED.get(int(K)) if float(K).is_integer() else None,
               "tiles": args.tiles, "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(f"K={K}: mse {mses[j]:.5f} (scale {scales[j] / std:.3f} std) bpw {bpw:.3f} "
              f"Lloyd-Max {lm if lm is None else round(lm, 5)} D(R) {2.0 ** (-2 * K):.5f} "
              f"QTIP {row['qtip']} closed={closed} roundtrip={rt} ({row['seconds']:.1f}s)", flush=True)
    doc = {"validation": "gaussian", "codebook": args.codebook, "seed": args.seed, "rows": rows,
           "encoder": enc.describe(), "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.out:
        write_json_atomic(Path(args.out), doc)
    return 0


def cmd_linear(args) -> int:
    """Validation (b): proxy tr(E H E^T)/tr(W H W^T) of real linears: trellis (EXL3 pipeline, the
    stored bits) vs our converter's w4a16 g64 LDLQ, with and without a 128-block random Hadamard on
    the input side (the q2ab-style rotation of down/o/out_proj)."""
    from common import ShardIndex, resolve_device
    from full_logits_golden import get_tensors_grouped

    device = resolve_device(args.device)
    model_dir = Path(args.model_dir)
    hdir = Path(args.hessian_dir)
    keys = json.loads((hdir / "hessian.json").read_text(encoding="utf-8"))["keys"]
    index = ShardIndex.load(model_dir)
    enc = TrellisEncoder(args.backend, device, threads=args.threads, grid=args.grid)
    enc.check_against_reference([float(k) for k in args.K.split(",")])
    results = []
    for spec in args.tensor:
        layer_s, mod = spec.split(":", 1)
        layer = int(layer_s)
        name = hf_name(layer, mod)
        ckey = f"text.layers.{layer}.{LINEARS[mod][0]}"
        H, rows = read_hess(hdir / keys[ckey], device)
        w = get_tensors_grouped(index, [name])[name]
        W = w.to(device=device, dtype=torch.float32).T.contiguous()  # [k, n]
        res = {"name": name, "hessian": keys[ckey], "k": W.shape[0], "n": W.shape[1], "variants": {}}
        for K in [float(k) for k in args.K.split(",")]:
            for basis in args.basis.split(","):
                t0 = time.perf_counter()
                # the seeds quantize-model uses for this tensor, so the numbers carry over
                out = quantize_group({name: w}, H, {name: K}, enc,
                                     seed_group=stable_seed("su", layer, LINEARS[mod][1]),
                                     seeds={name: stable_seed("sv", name)}, hessian_basis=basis,
                                     verbose=False)[name]
                r = out["record"]
                res["variants"][f"trellis K={K} {basis}"] = {
                    "proxy": r["proxy"], "proxy_damped_h": r["proxy_damped_h"],
                    "proxy_before_refit": r["proxy_before_refit"], "bpw": r["bits"]["bpw"],
                    "rel_weight_err": r["rel_weight_err"], "g": r["regularize"]["g_scale"]["g"],
                    "seconds": time.perf_counter() - t0}
                print(f"{name} trellis K={K} {basis}: proxy {r['proxy']:.6f} (before refit "
                      f"{r['proxy_before_refit']:.6f}) bpw {r['bits']['bpw']:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if not args.skip_int4:
            for rot in (False, True):
                t0 = time.perf_counter()
                Wh = w.to(device=device, dtype=torch.float32)  # [n, k]
                Hq = H
                if rot:
                    s = random_signs(W.shape[0], stable_seed("int4rht", name)).to(device)
                    Wh = had_r(Wh * s[None, :])          # W R, R = diag(s) P
                    Hq = rotate_hessian(H, s)            # R^T H R
                dq = int4_ldlq(Wh, Hq, group=64, damp=0.01)
                if rot:
                    dq = had_r(dq) * s[None, :]          # back: W_hat = W'_hat R^T
                p = proxy_error(W, dq.T, H)
                tag = "int4 g64 LDLQ" + (" + input RHT128" if rot else "")
                res["variants"][tag] = {"proxy": p, "bpw": 4.5, "seconds": time.perf_counter() - t0}
                print(f"{name} {tag}: proxy {p:.6f} bpw 4.5 ({time.perf_counter() - t0:.0f}s)", flush=True)
        results.append(res)
        del H, W
    doc = {"validation": "linear-proxy", "results": results, "encoder": enc.describe(),
           "provenance": provenance(enc), "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.out:
        write_json_atomic(Path(args.out), doc)
    return 0


#: tiles of the 400 quantized decoder linears (24,326,963,200 weights / 256; model_linears())
MODEL_TILES = 95_027_200


def cmd_bench(args) -> int:
    from common import resolve_device

    device = resolve_device(args.device)
    enc = TrellisEncoder(args.backend, device, threads=args.threads, grid=args.grid)
    rng = np.random.default_rng(0)
    rows = []
    for K in [float(k) for k in args.K.split(",")]:
        x = torch.from_numpy(rng.standard_normal((args.tiles, TILE)).astype(np.float32) * 0.8).to(device)
        enc.encode(x[: min(64, args.tiles)], K)  # warm-up / build
        t0 = time.perf_counter()
        enc.encode(x, K)
        dtm = time.perf_counter() - t0
        tps = args.tiles / dtm
        # LDLQ encodes every tile once; the scale search adds ~9% (spec 13)
        proj_h = MODEL_TILES * 1.09 / tps / 3600
        rows.append({"K": K, "tiles": args.tiles, "seconds": dtm, "tiles_per_s": tps,
                     "projected_model_hours": proj_h})
        print(f"K={K}: {args.tiles} tiles in {dtm:.2f}s = {tps:.0f} tiles/s -> whole 27B "
              f"(95.0 M tiles + 9% scale search) ~ {proj_h:.2f} h of encoder time", flush=True)
    doc = {"validation": "throughput", "rows": rows, "encoder": enc.describe(),
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.out:
        write_json_atomic(Path(args.out), doc)
    return 0


def cmd_selftest(args) -> int:
    """The encoders against each other on random tiles, every rate the oracle uses: native CPU vs
    viterbi_torch on the CPU and hip vs native CPU must give identical states and costs (bit for
    bit); viterbi_torch on the GPU must give identical costs (its argmin tie order is torch's GPU
    reduction's, so on an exact tie it may pick another, equally good path: reported only)."""
    from common import resolve_device

    device = resolve_device(args.device)
    rng = np.random.default_rng(1)
    backends = [b for b in args.backends.split(",") if b]
    bad = 0
    tref = TrellisEncoder("torch", "cpu", verbose=False)
    cpu = TrellisEncoder("cpu", "cpu", verbose=False)
    for K in [float(k) for k in args.K.split(",")]:
        x = torch.from_numpy(rng.standard_normal((args.tiles, TILE)).astype(np.float32))
        st_c, c_c = cpu.encode(x, K)
        pairs = [("cpu vs torch(cpu)", tref, "cpu", True)]
        if "hip" in backends and device.type == "cuda" and hip_widths_ok(K):
            pairs.append(("hip vs cpu", TrellisEncoder("hip", device, verbose=False), device, True))
        if device.type == "cuda" and "torch" in backends:
            pairs.append(("torch(cuda) vs cpu", TrellisEncoder("torch", device, verbose=False), device, False))
        for what, enc, dev, strict_states in pairs:
            st, c = enc.encode(x.to(dev), K)
            same = torch.equal(st.cpu(), st_c.cpu())
            dc = float((c.cpu() - c_c.cpu()).abs().max())
            ok = dc == 0.0 and (same or not strict_states)
            bad += not ok
            print(f"K={K} {what}: states identical={same} max|cost diff|={dc:.3g} "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
    print("SELFTEST " + ("PASSED" if bad == 0 else f"FAILED ({bad})"))
    return 0 if bad == 0 else 1


def main() -> int:
    from common import DEFAULT_MODEL_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def enc_args(p, device_default="cuda"):
        p.add_argument("--device", default=device_default, choices=["cuda", "cpu"])
        p.add_argument("--backend", default="auto", choices=["auto", "hip", "cpu", "torch"])
        p.add_argument("--threads", type=int, default=0, help="cpu encoder threads (0 = all)")
        p.add_argument("--grid", type=int, default=0, help="hip workgroups per launch (0 = 4 x CUs)")

    p = sub.add_parser("quantize-model", help="quantize every decoder linear into an override dir")
    enc_args(p)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--hessian-dir", type=Path, default=Path(r"C:\AI\r4dx-hessian\hessian-v2"))
    p.add_argument("--out-dir", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--K", type=float, help="one rate for every linear (3.5, 4, 5, ...)")
    g.add_argument("--bpw", type=float, help="a bpw target through EXL3's allocator")
    p.add_argument("--layers", default="", help="subset, e.g. 0-3,10 (default all)")
    p.add_argument("--hessian-basis", default="exl3", choices=["exl3", "matched"],
                   help="exl3 (default): EXL3's LDL factor of the sign-only rotated H (fidelity); "
                        "matched: the factor of the H the quantized weight actually sees (magnitudes "
                        "included), much better on attention q/k/v (docs/trellis.md 14)")
    p.set_defaults(fn=cmd_quantize_model)

    p = sub.add_parser("mix", help="a bpw mix from finished uniform-rate directories")
    p.add_argument("--bpw", type=float, required=True)
    p.add_argument("--src", action="append", required=True, help="a finished --K directory (repeat)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(fn=cmd_mix)

    p = sub.add_parser("gaussian", help="validation (a): iid N(0,1) MSE per K")
    enc_args(p, "cpu")
    p.add_argument("--K", default="2,3,4")
    p.add_argument("--tiles", type=int, default=256)
    p.add_argument("--codebook", default="mul1", choices=["mul1", "3inst", "mcg"])
    p.add_argument("--scale-lo", type=float, default=0.70)
    p.add_argument("--scale-hi", type=float, default=1.30)
    p.add_argument("--scale-step", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(fn=cmd_gaussian)

    p = sub.add_parser("linear", help="validation (b): proxy loss of real linears")
    enc_args(p)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--hessian-dir", type=Path, default=Path(r"C:\AI\r4dx-hessian\hessian-v2"))
    p.add_argument("--tensor", action="append", required=True, help="LAYER:module, e.g. 10:mlp.down_proj")
    p.add_argument("--K", default="4")
    p.add_argument("--basis", default="exl3,matched", help="exl3, matched or exl3,matched")
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(fn=cmd_linear)

    p = sub.add_parser("bench", help="validation (c): encoder throughput")
    enc_args(p)
    p.add_argument("--K", default="4")
    p.add_argument("--tiles", type=int, default=4096)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("selftest", help="native encoders vs the torch reference")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--backends", default="hip,torch",
                   help="besides cpu vs torch(cpu), which to check on the device: hip, torch")
    p.add_argument("--K", default="3,3.5,4,4.5,5,6")
    p.add_argument("--tiles", type=int, default=96)
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
