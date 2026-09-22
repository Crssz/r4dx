"""tools/reference/kv_fakequant_golden.py

How much of the measured rung-4 KL is the **fp8 paged KV cache**, and nothing else?

This is `full_logits_golden.py`'s streaming bf16 forward with exactly one thing changed: at every
`full_attention` layer, the two tensors the engine's paged cache actually stores -- the **full
post-rope K vector** and the **full V vector** -- are replaced, in place in the forward, by

    x' = dequant_e4m3(quant_e4m3(x * (1 / descale[head]))) * descale[head]

with the per-kv-head `descale` the converter itself would write (`amax / 448`, from a
`kv_calibrate*.py` JSON). Everything else is untouched bf16: the query is **not** quantized (it is
never cached), the GDN / `linear_attention` layers are **not** touched (they have no KV cache at
all), and no weight anywhere is quantized. So a KL against `tools/reference/kl_out/ref` -- the
unquantized run of the very same code -- is the fp8 KV cache's own contribution, isolated from the
4-bit body weights, the 4-bit `lm_head` and the engine's kernels.

**Matching the kernel's numerics exactly.** The engine's write path is
`KvWritePagedFp8Kernel` in `src/kernels/src/r4dx_kernels.hip`:

    const float kd_inv = 1.0f / k_descale[h];
    kv_cache[base + d] = FloatToFp8E4M3(__bfloat162float(kk[d]) * kd_inv);

and `FloatToFp8E4M3` (`src/core/include/r4dx/core/dtype.hpp`) is OCP **e4m3fn** whose
out-of-range behaviour is spelled out in the kernel rather than inherited from a library, which is
why this file re-derives the grid instead of calling `.to(torch.float8_e4m3fn)`:

  * it **saturates**: `if (af > 448.0f) af = 448.0f;`, so any finite magnitude above `448 * descale`
    lands on `+-448 * descale`, never on NaN. PyTorch's reference `fp8e4m3fn_from_fp32_value`
    emits NaN for anything that rounds past the top of the grid (`>= 480`, and by RNE anything
    `> 464`), which would poison a whole attention row rather than clip one element; whether a
    given build does that is a property of that build, not something a measurement should inherit.
    `--self-test` check 3 prints what the installed torch actually does and asserts the *kernel's*
    rule regardless.
  * only a genuine fp32 inf/NaN input becomes NaN (`abs_bits >= 0x7f800000 -> 0x7f | sign`).

Rounding is `rintf` under the default rounding mode, i.e. round-half-to-**even**, on both the
normal mantissa (`rintf(frac * 8)`, with the `mi == 8` carry into the next exponent) and the
subnormal grid (`rintf(af / 2^-6 * 8)`, i.e. steps of `2^-9`). `torch.round` is the same
round-half-to-even, so `fake_quant_e4m3` below reproduces the kernel bit for bit; `--self-test`
proves it against a hand-computed table, against an exhaustive sweep of all 256 e4m3 codes and
their midpoints, and against `torch.float8_e4m3fn` on the range where the two agree.

The read side needs no modelling beyond the multiply: the attention kernels never materialize a
dequantized cache, they fold `k_descale[h]` into the query once
(`r4d_attn_prefill_h256_gqa6.hip`: "FOLDQ scale * k_descale * log2(e) folded into the query ONCE at
load") and multiply the softmax-weighted sum by `v_descale[h]`, both of which are algebraically the
per-head scalar multiply this file applies. The one deviation from the engine that remains is that
the dequantized value is rounded back to **bf16** here, because the reference's attention math is
bf16 throughout: that is a <=2^-9 relative perturbation on top of e4m3's own 2^-4, i.e. ~1.5% of
the effect being measured, and it is what keeps the *only* difference from `--mode none` the 8-bit
grid rather than a change of arithmetic precision.

`--mode none` installs the same taps with the quantizer replaced by the identity, so it must
reproduce `full_logits_golden.py`'s output exactly; that is this file's plumbing gate.

Usage (reference venv only -- read-only against the venv and the checkpoint):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\kv_fakequant_golden.py `
        --kv-calib D:\\models\\r4dx\\qwen38-27b.kvcalib-full.json --mode both `
        --tokens tools\\reference\\kl_corpus\\tokens.json `
        --out-dir tools\\reference\\kl_out\\kvfq-full-both

Then pair it with `kl_report.py` against `tools/reference/kl_out/ref`. See
tools/reference/README.md ("kv_fakequant_golden.py") for the measured table and the interpretation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    resolve_device,
    sha256_file,
)
import full_logits_golden as flg  # noqa: E402
from full_logits_golden import StreamingReference  # noqa: E402

#: OCP e4m3fn finite max magnitude. `descale = amax / 448` (src/convert/.../kv_calib.hpp), and the
#: kernel's own saturation bound.
FP8_E4M3_MAX = 448.0

#: e4m3fn's smallest normal is 2^-6; below it the grid is uniform with step 2^-6 / 8 = 2^-9.
FP8_E4M3_MIN_NORMAL_EXP = -6
FP8_E4M3_MANTISSA_BITS = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KV_CALIB = Path(r"D:\models\r4dx\qwen38-27b.kvcalib-full.json")


def repo_relative(path: Path) -> str:
    """Path as written into a committed artefact: repo-relative with forward slashes when the file
    lives in the repo (never an absolute path through a user profile -- see CLAUDE.md)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# The e4m3fn grid, exactly as src/core/include/r4dx/core/dtype.hpp implements it
# --------------------------------------------------------------------------------------------


def fake_quant_e4m3(x: torch.Tensor) -> torch.Tensor:
    """`Fp8E4M3ToFloat(FloatToFp8E4M3(x))`, elementwise, without ever building a byte.

    `x` is fp32 (or anything that widens to it); the result is fp32 on the same device. Every
    value is snapped to the e4m3fn grid with round-half-to-even, saturating at +-448, with the
    subnormal range (|x| < 2^-6) on its uniform 2^-9 grid. Real infinities and NaNs propagate as
    NaN, which is what the kernel's `abs_bits >= 0x7f800000 -> 0x7f | sign` produces.
    """
    xf = x.float()
    a = xf.abs()
    bad = ~torch.isfinite(xf)  # inf/NaN -> NaN, per the kernel's early-out
    a = a.clamp(max=FP8_E4M3_MAX)  # the kernel's `if (af > kMax) af = kMax` -- saturate, never NaN

    # Unbiased exponent e with a in [2^e, 2^(e+1)); torch.frexp gives a = m * 2^exp2, m in [0.5, 1),
    # exactly as the kernel's frexpf does. Clamping e at -6 is the kernel's normal/subnormal split:
    # below 2^-6 the step stops shrinking and stays at 2^(-6-3) = 2^-9.
    _mant, exp2 = torch.frexp(a)
    e = (exp2 - 1).clamp(min=FP8_E4M3_MIN_NORMAL_EXP)
    step = torch.ldexp(torch.ones_like(a), e - FP8_E4M3_MANTISSA_BITS)

    # round-half-to-even on the grid. This is `rintf(frac * 8)` plus the `mi == 8` carry in one
    # expression: rounding a/step up to 2^(e+1)/step = 16 lands on a value that IS representable at
    # exponent e+1, which is exactly what the kernel's `mi = 0; e += 1` produces.
    q = torch.round(a / step) * step
    q = q.clamp(max=FP8_E4M3_MAX)  # the kernel's `e > 8 -> e = 8, mi = 6` top-of-grid clamp

    out = torch.copysign(q, xf)  # sign comes from the input's sign bit, so -0.0 stays -0.0
    if bool(bad.any()):
        out = torch.where(bad, torch.full_like(out, float("nan")), out)
    return out


def fake_quant_kv(x: torch.Tensor, descale: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    """One cache write + read back, in the reference's own dtype.

    `x` is the bf16 tensor the engine would hand `KvWritePagedFp8Kernel`; `descale` broadcasts to
    it over the kv-head axis. Returns `(x', clipped_elements, max_abs_over_448)` where `x'` has
    `x`'s dtype -- see the module docstring on the bf16 round trip.
    """
    inv = 1.0 / descale.float()  # `const float kd_inv = 1.0f / k_descale[h];` -- multiply, not divide
    y = x.float() * inv
    ay = y.abs()
    clipped = int((ay > FP8_E4M3_MAX).sum().item())
    headroom = float(ay.max().item()) / FP8_E4M3_MAX if ay.numel() else 0.0
    q = fake_quant_e4m3(y)
    return (q * descale.float()).to(x.dtype), clipped, headroom


# --------------------------------------------------------------------------------------------
# Descales from a kv_calibrate*.py JSON
# --------------------------------------------------------------------------------------------


def load_descales(path: Path, layer_types: list[str], kv_heads: int, device,
                  mult: float = 1.0) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    """`ResolveKvDescale` (src/convert/include/r4dx_convert/kv_calib.hpp) in Python, for every
    `full_attention` layer: `descale[h] = k_amax[h] / 448`, computed in fp32 exactly as the
    converter does, for both K and V. A layer the calibration does not cover falls back to
    `descale = 1.0` with a loud warning -- the converter's own behaviour, and a real (bad) thing to
    be able to measure.

    `mult` scales every descale (the `--descale-mult` sensitivity knob): >1 widens the representable
    range and coarsens the step, <1 tightens the step and clips more.
    """
    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    out: dict[int, dict[str, torch.Tensor]] = {}
    warnings: list[str] = []
    provenance = {"path": str(path), "sha256": sha256_file(path), "descale_mult": mult,
                  "layers_in_file": sorted(int(k) for k in calib.keys() if k.lstrip("-").isdigit()),
                  "full_attention_layers": full_layers}
    for i in full_layers:
        entry = calib.get(str(i))
        row = {}
        for kind in ("k", "v"):
            amax = None
            if entry is None:
                warnings.append(f"kv-calib has no entry for layer {i} -- falling back to descale=1.0")
            elif f"{kind}_amax" not in entry:
                warnings.append(f"kv-calib layer {i} is missing {kind}_amax -- descale=1.0")
            elif len(entry[f"{kind}_amax"]) != kv_heads:
                warnings.append(f"kv-calib layer {i} {kind}_amax has {len(entry[f'{kind}_amax'])} "
                                f"entries, expected {kv_heads} -- descale=1.0")
            else:
                amax = torch.tensor(entry[f"{kind}_amax"], dtype=torch.float32)
            if amax is None:
                d = torch.ones(kv_heads, dtype=torch.float32)
            else:
                d = amax / torch.tensor(FP8_E4M3_MAX, dtype=torch.float32)
            row[kind] = (d * float(mult)).to(device=device, dtype=torch.float32)
        out[i] = row
    provenance["warnings"] = sorted(set(warnings))
    provenance["descale_summary"] = {
        str(i): {kind: {"min": float(out[i][kind].min()), "max": float(out[i][kind].max())}
                 for kind in ("k", "v")}
        for i in full_layers
    }
    return out, provenance


# --------------------------------------------------------------------------------------------
# The taps
# --------------------------------------------------------------------------------------------


class KvFakeQuant:
    """Replaces the cached K/V inside `StreamingReference`'s streaming forward.

    Same two taps `kv_calibrate_full.py` *observes* with, used here to *substitute*:

      * **K, post-rope** -- `apply_rotary_pos_emb` is monkeypatched in the modeling module and its
        `k_embed` return value is replaced. This is precisely the tensor the engine writes: the
        whole `head_dim` vector, rotary dims and pass-through dims alike, after rope.
      * **V** -- a `v_proj` forward hook whose return value replaces the projection's output. No
        rope is applied to V, and the engine caches all `head_dim` of it.

    The query is deliberately left alone: `q_embed` is returned untouched, so nothing that is not
    in the cache is quantized.

    Layer identity: `LazyDecoderLayers` builds layer `i` immediately before the forward loop runs
    it and drops it immediately after, strictly in order, so wrapping the builder is enough to know
    which layer the global `apply_rotary_pos_emb` patch is currently serving (the same argument
    `kv_calibrate_full.KvCapture` relies on).
    """

    def __init__(self, ref: StreamingReference, descales: dict[int, dict[str, torch.Tensor]],
                 quant_k: bool, quant_v: bool):
        self.ref = ref
        self.descales = descales
        self.quant_k = quant_k
        self.quant_v = quant_v
        self.kv_heads = ref.text_config.num_key_value_heads
        self.head_dim = ref.text_config.head_dim
        self.layer_types = ref.layer_types
        self.stats = {"k_calls": 0, "v_calls": 0, "k_elements": 0, "v_elements": 0,
                      "k_clipped": 0, "v_clipped": 0, "k_max_headroom": 0.0, "v_max_headroom": 0.0,
                      "layers_touched": set()}
        self._current: int | None = None
        self._orig_builder = None
        self._orig_rope = None
        self._handles: list = []

    # -- context manager ----------------------------------------------------------------------

    def __enter__(self) -> "KvFakeQuant":
        mod = self.ref.m
        self._orig_rope = mod.apply_rotary_pos_emb
        orig_rope = self._orig_rope

        def patched_rope(*args, **kwargs):
            q_embed, k_embed = orig_rope(*args, **kwargs)
            layer_idx = self._current
            if layer_idx is not None and layer_idx in self.descales:
                k_embed = self._on_k(layer_idx, k_embed)
            return q_embed, k_embed  # q_embed untouched: the query is never cached

        mod.apply_rotary_pos_emb = patched_rope
        self._orig_builder = self.ref.lazy._builder

        def builder(i: int):
            layer = self._orig_builder(i)
            self._current = i
            if i in self.descales:
                assert self.layer_types[i] == "full_attention", (
                    f"layer {i} is {self.layer_types[i]!r}, not full_attention")
                self._handles.append(layer.self_attn.v_proj.register_forward_hook(self._make_v_hook(i)))
            return layer

        self.ref.lazy._builder = builder
        return self

    def __exit__(self, *exc) -> None:
        self.ref.m.apply_rotary_pos_emb = self._orig_rope
        self.ref.lazy._builder = self._orig_builder
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._current = None

    # -- taps ---------------------------------------------------------------------------------

    def _on_k(self, layer_idx: int, k_embed: torch.Tensor) -> torch.Tensor:
        # [batch, kv_heads, T, head_dim]; batch is always 1 (one sequence per forward).
        assert k_embed.dim() == 4 and k_embed.shape[0] == 1 and k_embed.shape[1] == self.kv_heads \
            and k_embed.shape[3] == self.head_dim, (
                f"layer {layer_idx}: post-rope K has shape {tuple(k_embed.shape)}, expected "
                f"[1, {self.kv_heads}, T, {self.head_dim}]")
        self.stats["k_calls"] += 1
        self.stats["layers_touched"].add(layer_idx)
        if not self.quant_k:
            return k_embed
        d = self.descales[layer_idx]["k"].view(1, self.kv_heads, 1, 1)
        out, clipped, headroom = fake_quant_kv(k_embed, d)
        self.stats["k_elements"] += k_embed.numel()
        self.stats["k_clipped"] += clipped
        self.stats["k_max_headroom"] = max(self.stats["k_max_headroom"], headroom)
        return out

    def _make_v_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            # v_proj output: [batch, T, kv_heads * head_dim]; no rope is ever applied to V.
            assert output.dim() == 3 and output.shape[0] == 1 and \
                output.shape[2] == self.kv_heads * self.head_dim, (
                    f"layer {layer_idx}: v_proj output has shape {tuple(output.shape)}")
            self.stats["v_calls"] += 1
            self.stats["layers_touched"].add(layer_idx)
            if not self.quant_v:
                return output
            t = output.shape[1]
            d = self.descales[layer_idx]["v"].view(1, 1, self.kv_heads, 1)
            view = output.view(1, t, self.kv_heads, self.head_dim)
            out, clipped, headroom = fake_quant_kv(view, d)
            self.stats["v_elements"] += view.numel()
            self.stats["v_clipped"] += clipped
            self.stats["v_max_headroom"] = max(self.stats["v_max_headroom"], headroom)
            return out.reshape(output.shape)

        return hook

    def summary(self) -> dict:
        s = dict(self.stats)
        s["layers_touched"] = sorted(s["layers_touched"])
        for kind in ("k", "v"):
            n = s[f"{kind}_elements"]
            s[f"{kind}_clipped_fraction"] = (s[f"{kind}_clipped"] / n) if n else 0.0
        return s


# --------------------------------------------------------------------------------------------
# Self-test: the e4m3 grid against hand-computed expectations
# --------------------------------------------------------------------------------------------


def e4m3_all_codes() -> list[float]:
    """Every finite value the 256 e4m3fn codes decode to, derived from the format definition here
    rather than from any library: exp_field 0 is `(mant/8) * 2^-6`, else `(1 + mant/8) * 2^(e-7)`,
    and `S.1111.111` is the NaN that has no value."""
    vals = []
    for sign in (1.0, -1.0):
        for exp_field in range(16):
            for mant in range(8):
                if exp_field == 15 and mant == 7:
                    continue  # the only NaN in e4m3fn
                if exp_field == 0:
                    v = (mant / 8.0) * 2.0 ** -6
                else:
                    v = (1.0 + mant / 8.0) * 2.0 ** (exp_field - 7)
                vals.append(sign * v)
    return vals


def self_test() -> int:
    """No GPU, no checkpoint, no files. Four checks, all on hand-derivable numbers."""
    print("[kv_fq self-test] 1. hand-computed table (descale = 1)")
    # (input, expected). Grid step at exponent e is 2^(e-3); ties go to the EVEN mantissa.
    table = [
        (0.0, 0.0),                       # zero
        (-0.0, -0.0),
        (1.0, 1.0),                       # exactly representable (e=0, mant 0)
        (1.125, 1.125),                   # e=0, mant 1 -- one step up
        (1.0625, 1.0),                    # midpoint 8.5/8 -> ties to even mantissa 8 -> 1.0
        (1.1875, 1.25),                   # midpoint 9.5/8 -> ties to even mantissa 10 -> 1.25
        (1.9999, 2.0),                    # rounds up across the exponent (the `mi == 8` carry)
        (260.0, 256.0),                   # e=8, step 32: 8.125 -> 8
        (448.0, 448.0),                   # the largest finite magnitude
        (449.0, 448.0),                   # saturate, NOT NaN (this is where torch's cast differs)
        (500.0, 448.0),
        (1e30, 448.0),
        (-500.0, -448.0),
        (2.0 ** -6, 2.0 ** -6),           # smallest normal
        (2.0 ** -9, 2.0 ** -9),           # smallest subnormal
        (0.017, 9 * 2.0 ** -9),           # subnormal grid: 0.017 / 2^-9 = 8.704 -> 9
        (3 * 2.0 ** -10, 2.0 ** -8),      # subnormal midpoint 1.5 -> ties to even 2
        (2.0 ** -10, 0.0),                # subnormal midpoint 0.5 -> ties to even 0
        (2.0 ** -11, 0.0),                # below half a step -> 0
    ]
    x = torch.tensor([t[0] for t in table], dtype=torch.float32)
    want = torch.tensor([t[1] for t in table], dtype=torch.float32)
    got = fake_quant_e4m3(x)
    bad = [(table[i][0], float(want[i]), float(got[i])) for i in range(len(table))
           if not (float(got[i]) == float(want[i])
                   and torch.signbit(got[i]) == torch.signbit(want[i]))]
    for xi, w, g in bad:
        print(f"    MISMATCH x={xi!r} want={w!r} got={g!r}")
    if bad:
        raise SystemExit("[kv_fq self-test] FAILED: hand-computed table")
    print(f"    {len(table)}/{len(table)} values match, including both tie directions, the "
          f"exponent carry, the subnormal grid and saturation at 448")

    print("[kv_fq self-test] 2. exhaustive sweep of every finite e4m3 value + every midpoint")
    codes = sorted(set(e4m3_all_codes()))
    cv = torch.tensor(codes, dtype=torch.float32)
    rt = fake_quant_e4m3(cv)
    if not torch.equal(rt, cv):
        raise SystemExit("[kv_fq self-test] FAILED: a representable value did not survive the round trip")
    # Every interior midpoint must land on whichever neighbour has the even mantissa. Derive the
    # expectation from the code's own low bit, not from the quantizer.
    pos = [c for c in codes if c > 0]
    mids, expect = [], []
    for lo, hi in zip(pos[:-1], pos[1:]):
        m = 0.5 * (lo + hi)
        if m in (lo, hi):
            continue  # fp32 could not represent the midpoint distinctly
        # "even mantissa" == the neighbour whose 3-bit mantissa field is even. For a uniform step,
        # lo and hi differ by one in the mantissa field, so exactly one of them is even; the one
        # that is even is the one whose value divided by the step is an even integer.
        step = hi - lo
        even_lo = (round(lo / step) % 2) == 0
        mids.append(m)
        expect.append(lo if even_lo else hi)
    mt = torch.tensor(mids, dtype=torch.float32)
    me = torch.tensor(expect, dtype=torch.float32)
    mg = fake_quant_e4m3(mt)
    nbad = int((mg != me).sum())
    if nbad:
        i = int((mg != me).nonzero()[0])
        raise SystemExit(f"[kv_fq self-test] FAILED: {nbad} midpoints did not round half to even, "
                         f"e.g. {mids[i]} -> {float(mg[i])}, expected {expect[i]}")
    print(f"    {len(codes)} representable values round-trip exactly; {len(mids)} midpoints all "
          f"round half to even")

    print("[kv_fq self-test] 3. agreement with torch.float8_e4m3fn where the two are defined alike")
    probe = torch.tensor(
        [c * f for c in codes for f in (1.0, 0.9993, 1.0007)] + [0.0, 1e-8, -1e-8, 3.3, -77.7],
        dtype=torch.float32)
    probe = probe[probe.abs() <= 464.0]  # above this, torch emits NaN where the kernel saturates
    ours = fake_quant_e4m3(probe)
    theirs = probe.to(torch.float8_e4m3fn).float()
    if not torch.equal(ours, theirs):
        i = int((ours != theirs).nonzero()[0])
        raise SystemExit(f"[kv_fq self-test] FAILED: disagrees with torch.float8_e4m3fn at "
                         f"{float(probe[i])}: ours {float(ours[i])}, torch {float(theirs[i])}")
    over = torch.tensor([465.0, 479.0, 500.0, 1e4], dtype=torch.float32)
    t_over = over.to(torch.float8_e4m3fn).float()
    o_over = fake_quant_e4m3(over)
    print(f"    {probe.numel()} probes in |x| <= 464 agree exactly with torch.float8_e4m3fn")
    print(f"    above it the kernel's own rule is asserted: {over.tolist()} -> ours "
          f"{o_over.tolist()}; this torch build gives {t_over.tolist()}")
    if not bool(torch.isnan(t_over).all()):
        print("    NOTE: this torch build SATURATES above the grid too, so the two happen to agree "
              "everywhere; the assertion below is on the kernel's rule, not on torch's")
    if not bool((o_over == FP8_E4M3_MAX).all()):
        raise SystemExit("[kv_fq self-test] FAILED: did not saturate above the grid")

    print("[kv_fq self-test] 4. the scaled write/read round trip (fake_quant_kv)")
    descale = torch.tensor([2.0, 0.25], dtype=torch.float32)  # amax 896 and 112
    # head 0: saturates at 448*2 = 896; head 1 at 448*0.25 = 112.
    xs = torch.tensor([[900.0, 896.0, 2.0, 2.125], [1000.0, 112.0, 0.25, 0.265625]],
                      dtype=torch.bfloat16)
    want = torch.tensor([[896.0, 896.0, 2.0, 2.0], [112.0, 112.0, 0.25, 0.25]], dtype=torch.bfloat16)
    # head 0: 900/2 = 450 -> sat 448 -> 896. 2/2 = 1 -> 1 -> 2. 2.125/2 = 1.0625 -> 1.0 (tie, even)
    #         -> 2.0.  head 1: 1000/0.25 = 4000 -> sat 448 -> 112. 0.265625/0.25 = 1.0625 -> 1 -> .25
    got, clipped, headroom = fake_quant_kv(xs, descale.view(2, 1))
    if not torch.equal(got.float(), want.float()):
        raise SystemExit(f"[kv_fq self-test] FAILED: fake_quant_kv gave {got.tolist()}, "
                         f"expected {want.tolist()}")
    if clipped != 2:
        raise SystemExit(f"[kv_fq self-test] FAILED: clipped count {clipped}, expected 2")
    print(f"    per-head saturation at 448*descale (896 / 112) and the tie-to-even both hold; "
          f"clipped={clipped}, max|x/descale|/448={headroom:.4f}")
    # identity gate in miniature: descale=1 and a value already on the grid must be a no-op
    grid = torch.tensor(codes, dtype=torch.float32).clamp(-448, 448)
    idg, _, _ = fake_quant_kv(grid, torch.ones(1))
    if not torch.equal(idg, grid):
        raise SystemExit("[kv_fq self-test] FAILED: descale=1 on-grid values were perturbed")

    print("[kv_fq self-test] PASS")
    return 0


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="check the e4m3 fake-quant against hand-computed values and exit (no GPU)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--kv-calib", type=Path, default=DEFAULT_KV_CALIB,
                    help="kv_calibrate*.py JSON whose k_amax/v_amax give descale = amax/448")
    ap.add_argument("--mode", default="both", choices=["both", "k-only", "v-only", "none"],
                    help="which cached tensors to fake-quantize ('none' = the plumbing gate: must "
                         "reproduce full_logits_golden.py exactly)")
    ap.add_argument("--descale-mult", type=float, default=1.0,
                    help="multiply every descale by this (sensitivity: 0.5 clips harder, 2.0 "
                         "coarsens the step)")
    ap.add_argument("--tokens", type=Path, default=Path(__file__).parent / "kl_corpus" / "tokens.json")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "kl_out" / "kvfq")
    ap.add_argument("--segment", action="append", default=None,
                    help="segment name to run (repeatable, or comma-separated); default: all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--impl", default="model", choices=["model", "manual"])
    ap.add_argument("--lm-head-chunk", type=int, default=32768)
    ap.add_argument("--row-block", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    device = resolve_device(args.device)  # enforces $env:HIP_VISIBLE_DEVICES == '1' for cuda
    doc = flg.load_tokens_file(args.tokens)
    wanted = None
    if args.segment:
        wanted = [s for arg in args.segment for s in arg.split(",") if s]
    segments = [s for s in doc["segments"] if wanted is None or s["name"] in wanted]
    if not segments:
        raise SystemExit(f"no segments selected (available: {[s['name'] for s in doc['segments']]})")
    if args.max_tokens:
        segments = [{"name": s["name"], "token_ids": s["token_ids"][: args.max_tokens]} for s in segments]

    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    except Exception as exc:
        print(f"[kv_fq] tokenizer unavailable ({type(exc).__name__}: {exc}); ids only")

    print(f"[kv_fq] building the streaming skeleton from {args.model_dir} on {device} ...", flush=True)
    t0 = time.perf_counter()
    ref = StreamingReference(args.model_dir, device)
    print(f"[kv_fq] skeleton ready in {time.perf_counter() - t0:.1f}s: {ref.n_layers} layers, "
          f"hidden={ref.text_config.hidden_size}, V={ref.vocab_size}", flush=True)

    descales, calib_prov = load_descales(args.kv_calib, ref.layer_types,
                                         ref.text_config.num_key_value_heads, device,
                                         mult=args.descale_mult)
    for w in calib_prov["warnings"]:
        print(f"[kv_fq] WARNING {w}")
    quant_k = args.mode in ("both", "k-only")
    quant_v = args.mode in ("both", "v-only")
    print(f"[kv_fq] mode={args.mode} (quant K={quant_k}, quant V={quant_v}) "
          f"descale_mult={args.descale_mult} over {len(descales)} full_attention layers "
          f"{sorted(descales)} from {calib_prov['path']}", flush=True)

    run = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool": "kv_fakequant_golden.py",
        "model_dir": str(args.model_dir),
        "config_sha256": sha256_file(args.model_dir / "config.json"),
        "tokens_file": repo_relative(args.tokens),
        "device": str(device),
        "torch_dtype": "bfloat16",
        "impl": args.impl,
        "mode": args.mode,
        "quantized": {"k_post_rope": quant_k, "v": quant_v, "query": False,
                      "linear_attention_layers": False},
        "descale_mult": args.descale_mult,
        "kv_calib": calib_prov,
        "n_layers": ref.n_layers,
        "vocab_size": ref.vocab_size,
        "torch_version": torch.__version__,
        "segments": {},
    }

    with KvFakeQuant(ref, descales, quant_k, quant_v) as fq:
        for seg in segments:
            meta = flg.run_segment(ref, seg["name"], seg["token_ids"], args.out_dir, args, tok)
            # Same shared format `kl_report.py` pairs on, plus what this run actually did.
            meta["source"] = "reference-kvfakequant"
            meta["kv_fakequant"] = {"mode": args.mode, "descale_mult": args.descale_mult,
                                    "kv_calib_sha256": calib_prov["sha256"]}
            with open(args.out_dir / f"{seg['name']}.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            run["segments"][seg["name"]] = meta
    run["kv_fakequant_stats"] = fq.summary()
    run["total_seconds"] = sum(s["seconds_stack"] + s["seconds_lm_head"] for s in run["segments"].values())

    s = run["kv_fakequant_stats"]
    expect_calls = len(descales) * len(segments)
    if s["k_calls"] != expect_calls or s["v_calls"] != expect_calls:
        raise SystemExit(f"[kv_fq] tap count wrong: k_calls={s['k_calls']} v_calls={s['v_calls']}, "
                         f"expected {expect_calls} each ({len(descales)} layers x {len(segments)} segments)")
    print(f"[kv_fq] taps fired on layers {s['layers_touched']}: K {s['k_calls']} calls "
          f"({s['k_clipped']}/{s['k_elements']} elements clipped = {s['k_clipped_fraction']:.3e}, "
          f"max |x/descale|/448 = {s['k_max_headroom']:.3f}); V {s['v_calls']} calls "
          f"({s['v_clipped']}/{s['v_elements']} = {s['v_clipped_fraction']:.3e}, "
          f"max {s['v_max_headroom']:.3f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "kv_fakequant_run.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(f"[kv_fq] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
