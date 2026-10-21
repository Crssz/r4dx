"""tests/reference/test_no_evidence.py -- experts without calibration evidence (hessian_q35.py) and null groups (oracle_q35.py).
CPU only, ~1 min:

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tests\\reference\\test_no_evidence.py

Ornith-1.5-122B has pruned experts whose gate / up / down weights are all 1e-26 .. 1e-35 but which the router still feeds with tens
of thousands of rows (layer 3: experts 72, 166, 180, 252): silu(gate) * up is exactly 0, so their down_proj Hessian is exactly 0.

  * expert_hessian_usable: NaN or inf raises (naming the expert); an all-zero Hessian is "no evidence" (False); a Hessian with some
    dead channels and some live ones is usable (True);
  * null_group_result: zero words, su = 1, sv = 0, reconstructs to exactly 0, the record carries what r4dx-convert and the allocator
    need (encoding, K, k, n, shape_hf, words_shape, rel_weight_err 1.0, bits, null_group);
  * oracle_q35.quantize_group_record on stub objects: negligible weights -> null group (hessian.no_evidence / fallback recorded, proxy_abs
    0), normal weights -> the real quantizer on the same Hessian file;
  * alloc_q35's report counts both.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "reference"))
sys.path.insert(0, str(REPO / "tools" / "quant2" / "qwen35moe"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hessian_capture as hc  # noqa: E402
import oracle_q35 as O  # noqa: E402
import trellis_quant as tq  # noqa: E402
from qwen35moe import alloc_q35 as A  # noqa: E402
from qwen35moe import hessian_q35 as HQ  # noqa: E402

CHECKS, FAILS = 0, []


def check(cond, what):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(what)
        print("FAIL:", what)


def test_policy():
    good = torch.diag(torch.rand(64) + 0.5)
    check(HQ.expert_hessian_usable(3, 7, "down", good, 5000) is True, "a live Hessian is usable")
    part = good.clone()
    part[10:60, 10:60] = 0.0
    check(HQ.expert_hessian_usable(3, 7, "down", part, 5000) is True, "dead channels next to live ones: still usable")
    check(HQ.expert_hessian_usable(3, 72, "down", torch.zeros(64, 64), 63208) is False, "an all-zero Hessian is NO EVIDENCE")
    check(HQ.expert_hessian_usable(3, 72, "down", torch.eye(64) * 1e-30, 63208) is False, "mean(diag) < 1e-20 (damp_hessian would refuse it) is NO EVIDENCE too")
    for bad, what in ((float("nan"), "NaN"), (float("inf"), "inf")):
        h = good.clone()
        h[3, 4] = bad
        try:
            HQ.expert_hessian_usable(3, 72, "gu", h, 100)
            check(False, f"{what} must raise")
        except RuntimeError as e:
            check("expert 72" in str(e) and "not finite" in str(e), f"{what} raises, naming the expert")
    h = torch.zeros(64, 64)
    h[0, 0] = float("nan")
    try:
        HQ.expert_hessian_usable(3, 5, "down", h, 100)
        check(False, "NaN in an otherwise zero Hessian must raise, not count as no-evidence")
    except RuntimeError:
        check(True, "NaN in an otherwise zero Hessian raises (not absorbed as no evidence)")


def test_null_group():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    k, n = 256, 128
    w = torch.full((n, k), 3e-30)
    res = O.null_group_result({"w": w.to(torch.bfloat16)}, torch.eye(k), 4.0, enc, 3e-30)["w"]
    check(res["words"].shape == (k // 16, n // 16, 32) and res["words"].dtype == torch.int32 and not bool(res["words"].any()),
          "null group: zero int32 words [k/16, n/16, 8K]")
    check(bool((res["suh"] == 1).all()) and bool((res["svh"] == 0).all()) and res["suh"].dtype == torch.float16, "su = 1, sv = 0 (fp16)")
    what = tq.reconstruct(res["words"], res["suh"], res["svh"], 4.0, enc.cb)
    check(what.shape == (k, n) and float(what.abs().max()) == 0.0, "reconstructs to EXACTLY zero")
    r = res["record"]
    json.dumps(r)
    check(r["encoding"] == "trellis-exl3" and r["K"] == 4.0 and r["codebook"] == enc.codebook and r["shape_hf"] == [n, k]
          and r["words_shape"] == [k // 16, n // 16, 32], "record: the fields r4dx-convert requires")
    check(r["rel_weight_err"] == 1.0 and r["bits"]["trellis"] == k * n * 4 and r["null_group"]["max_abs_weight"] == 3e-30, "rel 1.0, bits, null_group")
    r0 = O.null_group_result({"w": torch.zeros(n, k)}, torch.eye(k), 3.0, enc, 0.0)["w"]["record"]
    check(r0["rel_weight_err"] == 0.0 and r0["words_shape"] == [k // 16, n // 16, 24], "all-zero weights: rel 0.0, K=3 words")


class StubArch:
    @staticmethod
    def layer_prefix(i):
        return f"model.language_model.layers.{i}."

    def expert_base(self, i, e, proj):
        return f"{self.layer_prefix(i)}mlp.experts.{e}.{proj}"


class StubReader:
    def __init__(self, w):
        self.w = w

    def linear_weight(self, base, device):
        return self.w.clone()


def test_oracle_record():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    arch = StubArch()
    k, n = 256, 128
    base = arch.expert_base(3, 5, "down_proj")
    with tempfile.TemporaryDirectory() as td:
        hdir = Path(td)
        # the layer-wide fallback diagonal hessian_q35 substitutes for a no-evidence expert
        hc.write_hess_file(hdir / "L03.e0005.down.hess", np.diag(np.linspace(0.5, 1.5, k)).astype(np.float32), 8_000_000)
        hman = {"keys": {base: "L03.e0005.down.hess"}}
        routing = {"tokens": 1_000_000, "fallback": {"gate_up": [], "down": [5]}, "no_evidence": {"gate_up": [], "down": [5]},
                   "tokens_routed": [0, 0, 0, 0, 0, 63208]}
        g = {"tap": "e5.down", "bases": [base], "cls": "down", "expert": 5}
        args = SimpleNamespace(verbose=False)
        O.PROBE = None
        for scale, null in ((2e-30, True), (2e-2, False)):
            reader = StubReader(torch.randn(n, k, generator=torch.Generator().manual_seed(4)) * scale)
            tensors, recs = O.quantize_group_record(args, arch, reader, enc, 3, g, routing, hman, hdir, torch.device("cpu"), "f.safetensors", 4.0)
            r = recs[base + ".weight"]
            check(bool(r.get("null_group")) == null, f"weights {scale:g}: null group = {null}")
            check(r["hessian"]["fallback"] is True and r["hessian"]["no_evidence"] is True and r["hessian"]["routed"] == 63208,
                  f"weights {scale:g}: hessian provenance records fallback + no_evidence + routed rows")
            check(base + ".weight.trellis" in tensors and tensors[base + ".weight.trellis"].shape == (k // 16, n // 16, 32), "tensors keyed as the converter expects")
            if null:
                check(r["proxy_abs"] == 0.0 and r["rel_weight_err"] == 1.0 and r["class"] == "down" and r["tap"] == "e5.down", "null record: proxy_abs 0, rel 1.0")
            else:
                check(0.0 < r["proxy"] < 0.5 and "null_group" not in r, f"a normal group is quantized against the same Hessian (proxy {r['proxy']:.4f})")

class _Acc:
    def __init__(self, h, rows):
        self.h, self.rows = h, rows

    def finalize_fp32(self):
        return self.h.clone()


def _taps(E=16, gu_k=128, down_k=64, nan_expert=None):
    """A LayerTaps stand-in for finalize_layer: layer 3 (full attention) of the tiny arch, small K, hand-made accumulators."""
    from qwen35moe.arch import FULL

    g = torch.Generator().manual_seed(3)
    prefix = "model.language_model.layers.3."
    live = lambda k: (lambda x: x.T @ x / x.shape[0])(torch.randn(8 * k, k, generator=g))   # noqa: E731
    gu = torch.stack([live(gu_k) * 200 for _ in range(E)])
    dn = torch.stack([live(down_k) * 200 for _ in range(E)])
    n = torch.full((E,), 400, dtype=torch.int64)
    dn[2] = 0.0                      # expert 2: routed 400 rows, down_proj input exactly zero (a pruned expert)
    n[3] = 10                        # expert 3: too few rows (the Q16 fallback)
    if nan_expert is not None:
        gu[nan_expert, 5, 7] = float("nan")
    dense = {prefix + "self_attn.q_proj": _Acc(live(gu_k), 5000), prefix + "self_attn.o_proj": _Acc(torch.zeros(gu_k, gu_k), 5000)}
    return SimpleNamespace(
        lt=FULL, prefix=prefix, E=E, top_k=8, gu_k=gu_k, down_k=down_k, gu=gu * 400, dn=dn * 400, n_gu=n.numpy(), n_dn=n.numpy(),
        routed=n.numpy(), sq_in=torch.rand(gu_k, dtype=torch.float64) * 1000, t_in=1000, sq_h=torch.rand(down_k, dtype=torch.float64) * 100,
        pairs=8000, shared={"checked": 3, "equal": True}, dense=dense, probe=torch.zeros(E, HQ.PROBE_ROWS, gu_k, dtype=torch.bfloat16),
        rep_tap={prefix + "self_attn.q_proj": "in", prefix + "self_attn.o_proj": "out"})


def test_finalize():
    from qwen35moe.arch import Qwen35MoeArch

    tiny_hf = Path(r"E:\models\r4dx\tiny-q35moe\hf")
    if not (tiny_hf / "config.json").exists():
        print("SKIP finalize check: no tiny model")
        return
    arch = Qwen35MoeArch.from_config(tiny_hf / "config.json")
    ref = SimpleNamespace(arch=arch, model_dir=tiny_hf)
    corpus = {"path": "x", "sha256": "0" * 64, "segments": 1, "tokens": 1000, "tokenizer_mode": "none"}
    with tempfile.TemporaryDirectory() as td:
        cap = HQ.finalize_layer(Path(td), 3, _taps(), ref, corpus, {})
        d = Path(td) / "hessian" / "L03"
        rc = json.loads((d / "routing_counts.json").read_text())
        check(cap["no_evidence"] == {"gate_up": [], "down": [2], "dense": ["L03.out.hess"]}, f"no_evidence recorded: {cap['no_evidence']}")
        check(rc["no_evidence"]["down"] == [2] and rc["fallback"]["down"] == [2, 3], "routing_counts: no_evidence [2], fallback [2, 3] (too few rows + zero)")
        check((d / "DONE.json").exists() and cap["nonpositive_diag_nonfallback"] == {}, "layer finalized, DONE.json written")
        h2, _, _ = hc.read_hess_file(d / "L03.e0002.down.hess")
        check(np.array_equal(h2, np.diag(np.diag(h2))) and float(np.diag(h2).min()) > 0, "the no-evidence expert got the layer-wide (positive, diagonal) Hessian")
        ho, _, _ = hc.read_hess_file(d / "L03.out.hess")
        check(np.array_equal(ho, np.eye(128, dtype=np.float32)), "the all-zero dense tap got the data-free identity Hessian")
        man = json.loads((d / "hessian.json").read_text())
        check(arch.expert_base(3, 2, "down_proj") in man["keys"], "its key is still in hessian.json")
    with tempfile.TemporaryDirectory() as td:
        try:
            HQ.finalize_layer(Path(td), 3, _taps(nan_expert=6), ref, corpus, {})
            check(False, "a NaN expert Hessian must fail the layer")
        except RuntimeError as e:
            check("expert 6" in str(e) and "not finite" in str(e), "a NaN expert Hessian still fails loudly, naming the expert")
        check(not (Path(td) / "hessian" / "L03" / "DONE.json").exists(), "and no DONE.json is written")
def tiny_channel_weights(seed_scale=2e-3, k=256, n=256):
    """HF [n, k] weights with 60 output rows and 30 input columns of log-uniform scale 1e-14.5 .. 1e-10 (Ornith's pruned channels):
    the band in which trellis_quant.regularize's `+ 1e-10` can cancel a channel's divisor."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(n, k, generator=g) * seed_scale
    w[:60] *= (10 ** torch.empty(60).uniform_(-14.5, -10.0, generator=g))[:, None] / seed_scale
    w[:, 200:230] *= (10 ** torch.empty(30).uniform_(-14.5, -10.0, generator=g))[None, :] / seed_scale
    return w.to(torch.bfloat16)


def test_clean_channels():
    w = tiny_channel_weights()
    same, ro, co = O.clean_negligible_channels(w, 1e-8)
    check(ro == 60 and co == 30 and same is not w, f"clean: {ro} rows and {co} columns below 1e-8 zeroed")
    check(bool((same[:60] == 0).all()) and bool((same[:, 200:230] == 0).all()) and torch.equal(same[60:, :200], w[60:, :200]),
          "clean: only those channels changed")
    good = torch.randn(64, 64) * 2e-3
    out, ro, co = O.clean_negligible_channels(good, 1e-8)
    check(out is good and ro == 0 and co == 0, "clean: a tensor without negligible channels is returned as is (same object)")
    z = good.clone()
    z[3] = 0
    out, ro, co = O.clean_negligible_channels(z, 1e-8)
    check(out is z and ro == 0, "clean: exactly zero channels are not 'changed' (regularize already handles them)")
    check(O.clean_negligible_channels(w, 0.0)[0] is w, "clean: below = 0 is off")
    # the failure it removes: with the raw weights some seeds quantize to noise, cleaned never
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    k = w.shape[1]
    live_r, live_c = (w.float().abs().amax(1) >= 1e-8), (w.float().abs().amax(0) >= 1e-8)
    wl = w.float().T[live_c][:, live_r]
    worst_raw, worst_clean = 0.0, 0.0
    for seed in range(6):
        for tag, ww in (("raw", w.float()), ("clean", O.clean_negligible_channels(w.float(), 1e-8)[0])):
            res = tq.quantize_group({"w": ww}, torch.eye(k), {"w": 3.0}, enc, 100 + seed, {"w": 200 + seed}, hessian_basis="matched", verbose=False)["w"]
            what = tq.reconstruct(res["words"], res["suh"], res["svh"], 3.0, enc.cb)
            rel = float((wl - what[live_c][:, live_r]).norm() / wl.norm())
            if tag == "raw":
                worst_raw = max(worst_raw, rel)
            else:
                worst_clean = max(worst_clean, rel)
    check(worst_clean < 0.25, f"cleaned weights quantize well for every seed (worst live-channel rel err {worst_clean:.3f}; K3 normal ~0.15)")
    print(f"  (raw weights, same seeds: worst live-channel rel err {worst_raw:.3f}: >0.5 = the cancellation failure; informational)")
    # make_job: the key is present when cleaning is on, absent when off
    saved = O.CLEAN_BELOW
    try:
        import tempfile as _t

        md = Path(_t.mkdtemp())
        (md / "config.json").write_text("{}")
        O.CLEAN_BELOW = 1e-8
        j1 = O.make_job(md, 3.0)
        O.CLEAN_BELOW = 0.0
        j0 = O.make_job(md, 3.0)
        check(j1.get("clean_channels_below") == 1e-8 and "clean_channels_below" not in j0 and not O.jobs_match(j0, j1),
              "job key: clean_channels_below present with cleaning (old chunks go stale), absent with --clean-below 0")
    finally:
        O.CLEAN_BELOW = saved
    # through quantize_group_record: the record says what was cleaned
    arch = StubArch()
    base = arch.expert_base(3, 9, "gate_proj")
    with tempfile.TemporaryDirectory() as td:
        hdir = Path(td)
        hc.write_hess_file(hdir / "L03.e0009.gu.hess", np.diag(np.linspace(0.5, 1.5, k)).astype(np.float32), 90000)
        hman = {"keys": {base: "L03.e0009.gu.hess"}}
        routing = {"tokens": 100000, "fallback": {"gate_up": [], "down": []}, "tokens_routed": [0] * 9 + [90000]}
        g = {"tap": "e9.gu", "bases": [base], "cls": "gu", "expert": 9}
        O.PROBE = None
        _, recs = O.quantize_group_record(SimpleNamespace(verbose=False), arch, StubReader(w), enc, 3, g, routing, hman, hdir,
                                          torch.device("cpu"), "f.safetensors", 3.0)
        r = recs[base + ".weight"]
        check(r.get("cleaned_channels") == {"out": 60, "in": 30, "below": 1e-8} and not r.get("null_group") and r["proxy"] < 0.3,
              f"record carries cleaned_channels, group quantized normally (proxy {r['proxy']:.4f}: dead rows reconstruct as noise, which an identity Hessian counts; garbage is > 0.5)")

def test_alloc_report():
    from qwen35moe.arch import Qwen35MoeArch

    real = Path(r"E:\models\r4dx\tiny-q35moe\hf\config.json")
    if not real.exists():
        print("SKIP alloc report check: no tiny config")
        return
    arch = Qwen35MoeArch.from_config(real)
    src = {}
    for K in (3.0, 4.0):
        tensors = {}
        for i in range(arch.n_layers):
            for base in arch.quantized_linears(i):
                kd = A.parse_name(base + ".weight")[1]
                tensors[base + ".weight"] = {"K": K, "class": kd, "tap": "out" if base.endswith(("out_proj", "o_proj")) else "x", "numel": 100, "energy": 1.0, "proxy": 0.01 / K, "proxy_abs": 0.01 / K,
                                             "rel_weight_err": 0.1, "bits": {"total": int(100 * K)}, "hessian": {"tokens": 100, "routed": 10},
                                             "probe": {"rows": 1, "out_err": 0.01 / K, "out_energy": 1.0}}
        d = arch.expert_base(1, 3, "down_proj") + ".weight"
        tensors[d]["hessian"]["no_evidence"] = True
        tensors[arch.expert_base(1, 3, "gate_proj") + ".weight"]["null_group"] = {"x": 1}
        src[K] = {"tensors": tensors}
    stats = {K: A.class_stats(s) for K, s in src.items()}
    ks = {key: 4.0 for key in stats[4.0]}
    m = A.Model(arch)
    doc, text = A.build_report(m, stats, ks, int(10 * A.GIB), [], {K: {"_path": "x"} for K in src}, A.parse_floor(None))
    check(doc["no_evidence"]["down"]["no_evidence"] == 1 and doc["no_evidence"]["gu"]["null"] == 1 and doc["null_tensors_by_layer"] == {"1": 1},
          "allocation report counts no-evidence and null tensors")
    check("WITHOUT calibration evidence" in text and "null groups" in text and "L1:1" in text, "report text names them")


def main() -> int:
    test_policy()
    test_null_group()
    test_oracle_record()
    test_finalize()
    test_clean_channels()
    test_alloc_report()
    if FAILS:
        print(f"{len(FAILS)} of {CHECKS} checks FAILED")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
