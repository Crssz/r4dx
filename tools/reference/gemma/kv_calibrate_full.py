"""tools/reference/gemma/kv_calibrate_full.py -- static fp8 KV-cache calibration for Gemma 4 (docs/gemma4-plan.md M1-23).

The Gemma counterpart of tools/reference/kv_calibrate_full.py, producing the JSON `r4dx-convert --kv-calib` eats
(src/convert/include/r4dx_convert/kv_calib.hpp: `{"<layer_idx>": {"k_amax": [kv_heads], "v_amax": [kv_heads],
...}}`, `descale = amax / 448.0`, container tensors `text.layers.{i}.attn.k_descale|v_descale`, fp32 [kv_heads of
THAT layer]).

What is measured, per layer and per kv head, exactly the tensors the paged fp8 cache stores:

  K  post-`k_norm`, post-rope: the output of `apply_rotary_pos_emb` that follows `k_norm` (the call on the
     query is not tapped). Full layers rotate only the first `partial_rotary_factor * global_head_dim` dims
     (proportional rope), the rest are NoPE; the tap sees the whole 512-dim head either way.
  V  "as cached": the output of `v_norm` (RMSNorm WITHOUT weight, applied on EVERY layer). Sliding layers:
     V = v_norm(v_proj(x)). Full layers (`attention_k_eq_v`, no v_proj): V = v_norm(RAW k_proj(x)), i.e. the
     k_proj output BEFORE `k_norm` and rope, while K = rope(k_norm(k_proj(x))) is a separate tensor. A V
     descale taken from raw k_proj (or from K) would be wrong; `selfcheck` below recomputes v_norm(source)
     independently on the first sequence of every layer and records the difference.

Per-layer-type head counts come from `GemmaArch` (Huihui 12B: 40 sliding layers x 8 kv heads x 256, 8 full layers
x 1 kv head x 512). Layers are visited layer-major (one layer's weights on the device at a time, every
sequence's hidden state carried from layer to layer, as hessian_capture.py does), so the 24 GB checkpoint is read
once. Calibration sequences are the model's own token ids (gen_samples.py's `token_ids`, via
gemma.capture.build_tokens_corpus; rejected samples and any with KL-corpus overlap are excluded, so the gate
corpus never calibrates itself), each run from position 0 with the real per-layer-type masks, up to
`--max-seq-len` tokens (default 4096: the sliding window is 1024, so positions past it are covered).

Numerics caveat (docs/gemma4-plan.md 9.9): the stack here is the HF bf16 reference (bf16 residual). K and V are
post-RMSNorm tensors, which are far less sensitive to the residual rounding than logits are, but an amax is a
max: the recorded p99.99 and the `fp16_range` block (does every |K|, |V| fit f16? the engine may stage K/V through
f16 in places) are there to make a tail visible. This is not per-token dynamic scaling.

Outputs: `--out` (the converter's JSON, all calibrated layers) and `<out>.summary.json` (per-layer-type statistics:
descale range, amax/p99.99 tail ratio, f16 range, selfcheck results). Refuses to write a calibration that does not
cover every layer unless `--allow-partial` (a missing layer silently becomes descale 1.0 in the converter).

Usage (reference venv; GPU runs need the user's go-ahead, device 1 by default):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\kv_calibrate_full.py `
        --gen-file D:\\models\\r4dx\\huihui-gemma\\corpus\\samples.jsonl --out D:\\models\\r4dx\\huihui-gemma\\kvcalib.json

CPU dry runs (no GPU; `--device cpu` hides every GPU before torch is imported):

    python tools\\reference\\gemma\\kv_calibrate_full.py --tiny --out <tmp>\\kvcalib.json
    python tools\\reference\\gemma\\kv_calibrate_full.py --device cpu --gen-file ...samples.shard1.jsonl --limit-samples 2 `
        --max-seq-len 256 --max-layers 6 --allow-partial --out <tmp>\\kvcalib.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if "--tiny" in sys.argv or "cpu" in [a for i, a in enumerate(sys.argv) if i and sys.argv[i - 1] == "--device"]:
    # Hidden before torch is imported (PowerShell '' would not hide a GPU from every runtime).
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402

from gemma.arch import FULL, GemmaArch  # noqa: E402
from gemma.common_gemma import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    resolve_device,
    sha256_file,
)
from gemma.ref import add_tiny_args, build_tiny_checkpoint  # noqa: E402

#: OCP e4m3fn finite max magnitude; `descale = amax / FP8_E4M3_MAX` (src/convert/.../kv_calib.hpp).
FP8_E4M3_MAX = 448.0
F16_MAX = 65504.0
TAIL_PERCENTILE = 99.99
#: Largest |values| kept per (layer, head, tensor) so the 99.99th percentile is exact: it needs 0.01% of the
#: N = tokens x head_dim elements of a head (full layers: 4096 x 512 x ~100 sequences = 2e8 -> 2e4 values).
TOP_BUFFER = 65536
DEFAULT_GEN = Path(r"D:\models\r4dx\huihui-gemma\corpus\samples.jsonl")
DEFAULT_OUT = Path(r"D:\models\r4dx\huihui-gemma\kvcalib.json")


class HeadStats:
    """Running per-head amax and exact-tail percentile over a stream of [H, T, D] blocks (same statistic
    as tools/reference/kv_calibrate_full.py's HeadStats, kept here so this script needs no Qwen module)."""

    def __init__(self, num_heads: int, top_buffer: int = TOP_BUFFER):
        self.num_heads = num_heads
        self.top_buffer = top_buffer
        self.amax = torch.zeros(num_heads, dtype=torch.float32)
        self.top = torch.zeros(num_heads, 0, dtype=torch.float32)
        self.count = 0  # elements seen PER HEAD

    def update(self, x: torch.Tensor) -> None:
        assert x.dim() == 3 and x.shape[0] == self.num_heads, f"unexpected shape {tuple(x.shape)}"
        a = x.detach().float().abs()
        self.amax = torch.maximum(self.amax, a.amax(dim=(1, 2)).cpu())
        flat = a.reshape(self.num_heads, -1)
        k = min(self.top_buffer, flat.shape[1])
        merged = torch.cat([self.top, flat.topk(k, dim=1).values.cpu()], dim=1)
        self.top = merged.topk(min(self.top_buffer, merged.shape[1]), dim=1).values
        self.count += flat.shape[1]

    def percentile(self, p: float):
        """(values per head, exact) of the nearest-rank percentile of |value|; (None, False) if the kept
        buffer is too small for it."""
        if self.count == 0:
            return None, False
        rank_from_top = self.count - math.ceil((p / 100.0) * self.count)
        if rank_from_top >= self.top.shape[1]:
            return None, False
        return self.top[:, rank_from_top].tolist(), True


class LayerStats:
    def __init__(self, kv_heads: int):
        self.k = HeadStats(kv_heads)
        self.v = HeadStats(kv_heads)
        self.selfcheck: dict = {}


def rms_noscale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """v_norm: x * (mean(x^2) + eps)^-0.5 in fp32, one cast back (docs/gemma4-semantics.md item 12)."""
    xf = x.float()
    return (xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).to(x.dtype)


class KvTaps:
    """K (post-norm, post-rope) and V (post-v_norm) taps for the layer currently being run.

    The rope tap relies on the call order inside Gemma4UnifiedTextAttention.forward: `k_norm` runs
    immediately before the key's `apply_rotary_pos_emb`, the query's rope call comes earlier and after
    q_norm. A forward hook on `k_norm` arms the patched rope function for exactly the next call."""

    def __init__(self, modeling, arch: GemmaArch):
        self.m = modeling
        self.arch = arch
        self.stats: dict[int, LayerStats] = {}
        self.geom: dict[int, tuple[int, int]] = {}   # layer -> (kv_heads, head_dim), read off the module
        self.layer = None          # layer index being run, None between layers
        self.record_check = False  # first sequence only: keep the raw tensors for `selfcheck`
        self._armed = False
        self._raw: dict = {}
        self._orig = None
        self._handles: list = []

    def __enter__(self):
        self._orig = self.m.apply_rotary_pos_emb
        orig = self._orig

        def patched(x, cos, sin, unsqueeze_dim=1):
            out = orig(x, cos, sin, unsqueeze_dim=unsqueeze_dim)
            if self._armed:
                self._armed = False
                self._on_k(out, x, cos)
            return out

        self.m.apply_rotary_pos_emb = patched
        return self

    def __exit__(self, *exc):
        self.m.apply_rotary_pos_emb = self._orig
        self.detach()

    def attach(self, layer, i: int) -> None:
        """Install the hooks on a freshly built layer `i` and start its statistics."""
        a = self.arch
        attn = layer.self_attn
        # Geometry from the module itself (the tiny checkpoint's saved config drops global_head_dim, the
        # real one carries it; the caller cross-checks this against the arch table for a real model).
        nkv = attn.k_proj.out_features // attn.head_dim
        self.geom[i] = (nkv, attn.head_dim)
        self.stats.setdefault(i, LayerStats(nkv))
        assert attn.k_norm is not None and attn.v_norm is not None, "kv-shared layers are not supported"
        assert (attn.v_proj is None) == (not a.has_v_proj(i)), f"layer {i}: v_proj presence != arch table"
        self.layer = i
        self._handles = [
            attn.k_norm.register_forward_hook(self._k_norm_hook),
            attn.v_norm.register_forward_hook(self._v_hook),
            (attn.v_proj if attn.v_proj is not None else attn.k_proj).register_forward_hook(self._vsrc_hook),
        ]

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.layer = None
        self._armed = False

    # -- taps ---------------------------------------------------------------------------------

    def _k_norm_hook(self, _m, _inp, out):
        self._armed = True
        if self.record_check:
            self._raw["k_normed"] = out.detach().clone()

    def _vsrc_hook(self, _m, _inp, out):
        if self.record_check:
            self._raw["v_src"] = out.detach().clone()   # [1,T,nkv*hd]: v_proj out, or RAW k_proj out (full)

    def _on_k(self, k_rope, k_pre, cos) -> None:
        i = self.layer
        nkv, hd = self.geom[i]
        assert k_rope.dim() == 4 and k_rope.shape[0] == 1 and k_rope.shape[2] == nkv and k_rope.shape[3] == hd, (
            f"layer {i}: post-rope K has shape {tuple(k_rope.shape)}, expected [1, T, {nkv}, {hd}]")
        self.stats[i].k.update(k_rope[0].transpose(0, 1))
        if self.record_check:
            self._raw["k_rope"] = k_rope.detach().clone()
            self._raw["rot_dim"] = int(cos.shape[-1])

    def _v_hook(self, _m, _inp, out):
        i = self.layer
        nkv, hd = self.geom[i]
        assert out.dim() == 4 and out.shape[0] == 1 and out.shape[2] == nkv and out.shape[3] == hd, (
            f"layer {i}: v_norm output has shape {tuple(out.shape)}, expected [1, T, {nkv}, {hd}]")
        self.stats[i].v.update(out[0].transpose(0, 1))
        if self.record_check:
            self._raw["v_norm"] = out.detach().clone()

    def selfcheck(self) -> dict:
        """Independent recomputation on the sequence that was recorded: V == v_norm(source) from the module's
        own source tensor (full layers: the raw k_proj output), and K really is the rotated k_norm output."""
        i = self.layer
        a = self.arch
        nkv, hd = self.geom[i]
        r = self._raw
        T = r["v_src"].shape[1]
        want_v = rms_noscale(r["v_src"].view(1, T, nkv, hd), a.eps)
        v_diff = float((want_v.float() - r["v_norm"].float()).abs().max())
        pre, post = r["k_normed"].float(), r["k_rope"].float()
        rd = r["rot_dim"]
        na, nb = pre[..., :rd].norm(dim=-1), post[..., :rd].norm(dim=-1)
        rel = float(((nb - na).abs() / na.clamp_min(1e-6)).max())
        out = {"v_source": "raw k_proj output (before k_norm / rope)" if not a.has_v_proj(i) else "v_proj output",
               "v_equals_vnorm_of_source_max_abs_diff": v_diff,
               "k_differs_from_k_norm_output": bool((post != pre).any()),
               "k_rotary_dims": rd, "k_head_dim": hd,
               "k_passthrough_dims_bit_identical": bool(torch.equal(post[..., rd:], pre[..., rd:])),
               "k_rotary_subspace_norm_max_rel_diff": rel,
               "k_vs_v_max_abs_diff": float((post - r["v_norm"].float()).abs().max()) if not a.has_v_proj(i)
               else None}
        self._raw = {}
        return out


# --------------------------------------------------------------------------------------------


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--gen-file", type=Path, default=DEFAULT_GEN, metavar="SAMPLES_JSONL",
                    help="gen_samples.py token-id JSONL (the merged samples.jsonl for the real run)")
    ap.add_argument("--max-seq-len", type=int, default=4096, help="first N ids of every sample")
    ap.add_argument("--max-seqs", type=int, default=None, help="at most N samples per category")
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="dry runs: use only N samples, spread evenly over the sorted list (all categories)")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="SMOKE: run only the first N layers (needs --allow-partial to write)")
    ap.add_argument("--allow-partial", action="store_true", help="write an --out that does not cover every layer")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    add_tiny_args(ap)
    return ap.parse_args(argv)


def run(args) -> dict:
    """Returns the converter JSON (also written to --out)."""
    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.device = "cpu"
    if args.device == "cpu":
        assert not torch.cuda.is_available(), "a GPU is visible to a --device cpu run"
    device = resolve_device(args.device)
    arch = GemmaArch.from_model_dir(args.model_dir)

    if args.tiny:
        gen = torch.Generator().manual_seed(args.tiny_seed + 5)
        from hessian_capture import Seq

        seqs = [Seq(f"tiny/{k}", "tokens", [arch.bos_id] + torch.randint(4, arch.vocab, (n - 1,), generator=gen).tolist())
                for k, n in enumerate((37, 21, 50))]
        corpus_meta = {"source": "tiny random ids"}
    else:
        from gemma.capture import build_tokens_corpus

        seqs, sources = build_tokens_corpus(args.gen_file, args.max_seq_len, args.max_seqs)
        if args.limit_samples and len(seqs) > args.limit_samples:
            step = len(seqs) / args.limit_samples
            seqs = [seqs[int(k * step)] for k in range(args.limit_samples)]
        corpus_meta = {"gen_file": str(args.gen_file), "gen_file_sha256": sha256_file(args.gen_file),
                       "sources": sources, "sample_ids": [s.name for s in seqs]}
    total = sum(len(s.token_ids) for s in seqs)
    print(f"[kv_calib] {len(seqs)} sequences, {total:,} tokens (<= {args.max_seq_len} each), device {device}",
          flush=True)

    from gemma.capture import GemmaCaptureRef

    n_run = arch.n_layers if args.max_layers is None else min(args.max_layers, arch.n_layers)
    cap = GemmaCaptureRef(args.model_dir, device, dtype=torch.bfloat16, verbose=False)
    t0 = time.perf_counter()
    taps = KvTaps(cap.ref.m, arch)
    with torch.no_grad(), taps:
        hidden = [cap.embed(s.token_ids) for s in seqs]
        for i in range(n_run):
            t1 = time.perf_counter()
            layer = cap.build_layer(i)
            taps.attach(layer, i)
            if not args.tiny and taps.geom[i] != (arch.kv_heads_of(i), arch.head_dim_of(i)):
                raise SystemExit(f"[kv_calib] layer {i}: module geometry {taps.geom[i]} != arch table "
                                 f"{(arch.kv_heads_of(i), arch.head_dim_of(i))}")
            for k in range(len(seqs)):
                taps.record_check = (k == 0)
                hidden[k] = cap.capture_layer_forward(layer, i, hidden[k])
                if k == 0:
                    taps.stats[i].selfcheck = taps.selfcheck()
            taps.detach()
            del layer
            st = taps.stats[i]
            print(f"[kv_calib] L{i:02d} {arch.layer_types[i][:4]:<4} kv_heads={taps.geom[i][0]} hd={taps.geom[i][1]} "
                  f"k_amax max {float(st.k.amax.max()):9.3f}  v_amax max {float(st.v.amax.max()):9.3f}  "
                  f"vcheck {st.selfcheck['v_equals_vnorm_of_source_max_abs_diff']:.2e}  "
                  f"{time.perf_counter() - t1:6.1f}s", flush=True)
    wall = time.perf_counter() - t0

    full_cover = n_run == arch.n_layers
    if not full_cover and not args.allow_partial:
        raise SystemExit(f"[kv_calib] only {n_run} of {arch.n_layers} layers were calibrated; the converter would "
                         "give every other layer descale 1.0. Pass --allow-partial for a smoke run.")
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    out: dict = {}
    summary_layers = {}
    for i in range(n_run):
        st = taps.stats[i]
        for nm, hs in (("k", st.k), ("v", st.v)):
            amax = hs.amax.tolist()
            if hs.count == 0 or not all(math.isfinite(a) and a > 0 for a in amax):
                raise SystemExit(f"[kv_calib] layer {i} {nm}_amax is empty, zero or non-finite: {amax}")
        k_tail, k_ex = st.k.percentile(TAIL_PERCENTILE)
        v_tail, v_ex = st.v.percentile(TAIL_PERCENTILE)
        k_amax, v_amax = st.k.amax.tolist(), st.v.amax.tolist()
        entry = {
            "layer_idx": i, "layer_type": arch.layer_types[i], "kv_heads": taps.geom[i][0],
            "head_dim": taps.geom[i][1], "num_calibration_tokens": total,
            "k_amax": k_amax, "v_amax": v_amax,
            # informational (the converter reads k_amax / v_amax only; descale = amax / 448):
            "k_descale": [a / FP8_E4M3_MAX for a in k_amax], "v_descale": [a / FP8_E4M3_MAX for a in v_amax],
            # v_norm output has rms ~1 (eps aside), so |V| <= sqrt(head_dim) exactly: a data-free upper bound
            "v_amax_analytic_bound": math.sqrt(taps.geom[i][1]),
            "k_p9999": k_tail, "v_p9999": v_tail, "tail_percentile": TAIL_PERCENTILE, "tail_exact": bool(k_ex and v_ex),
            "tail_elements_per_head": st.k.count,
            "fp16_range": {"max_abs": max(k_amax + v_amax), "fits_f16": max(k_amax + v_amax) < F16_MAX},
            "k_tap": "post k_norm, post rope", "v_tap": "post v_norm (weightless RMSNorm); "
            + ("source = RAW k_proj output (attention_k_eq_v)" if not arch.has_v_proj(i) else "source = v_proj output"),
            "selfcheck": st.selfcheck, "method": "full-forward, layer-major, position 0, per-layer-type masks",
        }
        out[str(i)] = entry
        summary_layers[str(i)] = {"type": arch.layer_types[i], "k_amax_max": max(k_amax), "v_amax_max": max(v_amax),
                                  "k_tail_ratio": None if not k_tail else max(a / max(t, 1e-9) for a, t in zip(k_amax, k_tail)),
                                  "v_tail_ratio": None if not v_tail else max(a / max(t, 1e-9) for a, t in zip(v_amax, v_tail))}

    by_type = {}
    for lt in (FULL, "sliding_attention"):
        ids = [i for i in range(n_run) if arch.layer_types[i] == lt]
        if not ids:
            continue
        ka = [a for i in ids for a in out[str(i)]["k_amax"]]
        va = [a for i in ids for a in out[str(i)]["v_amax"]]
        by_type[lt] = {"layers": ids, "kv_heads_per_layer": taps.geom[ids[0]][0], "head_dim": taps.geom[ids[0]][1],
                       "k_amax": {"min": min(ka), "max": max(ka)}, "v_amax": {"min": min(va), "max": max(va)},
                       "k_descale": {"min": min(ka) / FP8_E4M3_MAX, "max": max(ka) / FP8_E4M3_MAX},
                       "v_descale": {"min": min(va) / FP8_E4M3_MAX, "max": max(va) / FP8_E4M3_MAX},
                       "fits_f16": max(ka + va) < F16_MAX,
                       "max_v_selfcheck_diff": max(out[str(i)]["selfcheck"]["v_equals_vnorm_of_source_max_abs_diff"] for i in ids)}
    summary = {"generated_at": generated_at, "model_dir": str(args.model_dir), "layers_calibrated": n_run,
               "layers_total": arch.n_layers, "complete": full_cover, "tokens": total, "wall_s": round(wall, 1),
               "corpus": corpus_meta, "by_layer_type": by_type, "layers": summary_layers,
               "caveat": "static per-kv-head scale from the HF bf16 stack; amax is a max, see k_p9999 / v_p9999",
               "torch": torch.__version__, "device": str(device)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    sp = args.out.with_name(args.out.name + ".summary.json")
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[kv_calib] wrote {args.out} ({len(out)} layers) and {sp}; {wall:.1f}s")
    for lt, d in by_type.items():
        print(f"[kv_calib] {lt}: layers {len(d['layers'])} x {d['kv_heads_per_layer']} heads x {d['head_dim']}  "
              f"K descale {d['k_descale']['min']:.4f}..{d['k_descale']['max']:.4f}  "
              f"V descale {d['v_descale']['min']:.4f}..{d['v_descale']['max']:.4f}  f16 ok {d['fits_f16']}")
    if tmp is not None:
        tmp.cleanup()
    return out


def main(argv=None) -> int:
    run(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
