"""tests/reference/test_q4exp_quant.py

Tests of the qwen4_exp quantization tooling (docs/qwen4exp-plan.md M1-10 .. M1-14): tools/quant2/qwen4exp/{corpus_q4,oracle_q4}.py,
tools/reference/qwen4exp/{hessian_q4,alloc_q4}.py, pipeline.ps1.

    <venv>\\Scripts\\python.exe tests\\reference\\test_q4exp_quant.py [--e2e] [--e2e-device cpu|cuda]

CPU only by default (the reference venv D:\\venvs\\r4dx-qwen4-ref or any with torch + numpy):

    (a) alloc_q4: the byte model reproduces the loader test's documented numbers (uniform KB 3 gate/up + KB 4 down = 51.24 GiB,
        ChooseSplit k = 24, 25.46 / 25.78 GiB), `choose_split` equals ChooseSplit on the C++ test's synthetic cases, the greedy
        allocation respects the cap on both cards, is maximal, ranks by loss per GiB and is deterministic;
    (b) corpus_q4: packing (<= 4096, EOS-separated, order kept), slicing, random corpora;
    (c) hessian_q4: synthetic routed rows through LayerTaps + finalize_layer: expert Hessians equal fp64 X^T X / n to 1e-5, files
        read back through trellis_quant.read_hess / hessian_capture.read_hess_file, symmetric, PSD, the fallback rule on a
        deliberately rare expert (class-specific: 100 rows -> gate_up AND down fallback, 700 rows -> gate_up only), the fallback
        formula, routing_counts.json, hessian.json keys, a routed-count mismatch is refused;
    (d) residual boundaries: crc32 per segment, a corrupted segment is refused.

`--e2e` also runs the tiny-model chain (tools/quant2/qwen4exp/tiny_e2e.py: Hessians of all 4 tiny layers, oracles K3/K4/K5,
allocation under a scaled cap, converter) -- minutes on the CPU.

Plain script, no pytest: exits 0 and prints "OK (<n> checks)", else 1 and every failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "quant2" / "qwen4exp"))

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"SKIP: {e} -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

from qwen4exp import alloc_q4 as A  # noqa: E402
from qwen4exp import hessian_q4 as HQ  # noqa: E402
from qwen4exp.arch import Qwen4ExpArch  # noqa: E402

import corpus_q4 as C  # noqa: E402
import hessian_capture as hc  # noqa: E402
import trellis_quant as tq  # noqa: E402

CHECKS = 0
FAILS: list[str] = []
REAL_CONFIG = Path(r"E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\config.json")
TINY_HF = Path(r"E:\models\r4dx\tiny-q4exp\hf")


def check(cond, what: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(what)
        print(f"FAIL: {what}")


def raises(fn, exc=Exception, text: str = "") -> bool:
    try:
        fn()
    except exc as e:
        return text in str(e)
    return False


# --------------------------------------------------------------------------------------------
# (a) allocation
# --------------------------------------------------------------------------------------------


def test_choose_split() -> None:
    cs = A.choose_split
    check(cs([100] * 10, 0, 0, 1000) == (5, 500, 500), "uniform 10x100 -> k 5, 500 / 500")
    check(cs([100] * 10, 0, 200, 2000) == (6, 600, 600), "tail 200 -> k 6, 600 / 600")
    check(cs([100] * 10, 300, 0, 2000) == (3, 600, 700), "head 300 -> k 3 (tie with 4 -> lower k)")
    check(cs([100] * 7, 0, 0, 1000) == (3, 300, 400), "7x100 -> k 3 (tie -> lower k)")
    check(cs([500, 50, 50, 50, 50, 50], 0, 0, 1000) == (1, 500, 250), "{500, 50 x 5} -> k 1")
    check(cs([100] * 10, 0, 0, 600)[0] == 5, "cap 600 keeps k 5")
    v = [300] + [100] * 7
    check(cs(v, 0, 0, 1000) == (3, 500, 500), "uneven -> k 3, 500 / 500")
    check(cs(v, 0, 0, 520)[0] == 3, "tight cap -> the one feasible k")
    check(raises(lambda: cs(v, 0, 0, 480), RuntimeError, "no layer boundary keeps both cards"), "cap below half -> refused")
    check(raises(lambda: cs([100], 0, 0, 1000), ValueError, "at least two layers"), "one layer refused")
    check(raises(lambda: cs([100, 100], -1, 0, 1000), ValueError, "must be >= 0"), "negative head refused")
    check(raises(lambda: cs([100, -5], 0, 0, 1000), ValueError, "negative bytes"), "negative layer refused")
    check(raises(lambda: cs([100, 100], 0, 0, 0), ValueError, "cap_bytes > 0"), "cap 0 refused")


def test_documented_numbers() -> None:
    if not REAL_CONFIG.exists():
        print(f"SKIP: documented numbers (no {REAL_CONFIG})")
        return
    arch = Qwen4ExpArch.from_config(REAL_CONFIG)
    m = A.Model(arch)
    u = A.uniform_assignment(48, 4, 3, 4)   # projections KB 4, gate/up KB 3, down KB 4
    total = m.total(u) / A.GIB
    k, d0, d1 = m.split(u, 26 * A.GIB)
    check(abs(total - 51.24) < 0.005, f"uniform KB3 gate_up + KB4 down = {total:.3f} GiB (documented 51.24)")
    check(k == 24, f"ChooseSplit k = {k} (documented 24)")
    check(abs(d0 / A.GIB - 25.46) < 0.005 and abs(d1 / A.GIB - 25.78) < 0.005,
          f"cards {d0 / A.GIB:.3f} / {d1 / A.GIB:.3f} GiB (documented 25.46 / 25.78)")
    check(m.total(A.uniform_assignment(48, 4, 4, 4)) / A.GIB > 52, "all-KB4 does not fit two 26 GiB cards")
    check(abs(A.class_bytes(arch, 5, "gu", 3) - 512 * 2 * A.trellis_tensor_bytes(2560, 640, 3)) == 0, "gu class bytes = 1024 tensors")
    # one linear, spelled out: 640 x 2560 at KB 3 = 614,400 words bytes + 4 * 2560 + 4 * 640 + 4 * 5
    check(A.trellis_tensor_bytes(2560, 640, 3) == 640 * 2560 * 3 // 8 + 4 * 2560 + 4 * 640 + 20, "tensor bytes: words + suh + svh + tickets")


def synthetic_stats(arch: Qwen4ExpArch, Ks, seed: int = 7):
    """loss(K) per class: a log-normal base at the lowest K, divided by ~3.6 per extra KB (fp noise), every class its own."""
    rng = np.random.default_rng(seed)
    stats: dict = {K: {} for K in Ks}
    for i in range(arch.n_layers):
        for kd in A.KINDS:
            base = float(np.exp(rng.normal(0.0, 1.0))) * (3.0 if kd == "down" else 1.0)
            for j, K in enumerate(sorted(Ks)):
                stats[K][(i, kd)] = {"loss": base / (3.6 ** j) * float(1 + 0.05 * rng.standard_normal()),
                                     "names": [], "experts": 0, "fallback": 0}
    return stats


def test_allocation() -> None:
    if not REAL_CONFIG.exists():
        print(f"SKIP: allocation (no {REAL_CONFIG})")
        return
    arch = Qwen4ExpArch.from_config(REAL_CONFIG)
    m = A.Model(arch)
    stats = synthetic_stats(arch, (3.0, 4.0))
    cap = 26 * A.GIB
    floor = A.DEFAULT_FLOOR
    ks, log = A.allocate(m, stats, cap, floor)
    k, d0, d1 = m.split(ks, cap)
    check(d0 <= cap and d1 <= cap, f"both cards within the cap ({d0 / A.GIB:.3f}, {d1 / A.GIB:.3f} GiB)")
    check(all(ks[(i, "p")] >= 4.0 for i in range(48)), "projections never below their floor KB 4")
    # maximal: every remaining promotion would break the cap
    maximal = True
    for (i, kd), K in ks.items():
        nxt = [x for x in (3.0, 4.0) if x > K]
        if nxt:
            t = dict(ks)
            t[(i, kd)] = nxt[0]
            try:
                m.split(t, cap)
                maximal = False
            except RuntimeError:
                pass
    check(maximal, "no further promotion fits (the allocation is maximal)")
    check(len(log) > 0 and all(p["gain"] > 0 for p in log), "promotions happened and each reduces the loss")
    floor_ks = {key: max(floor[key[1]], 3.0) for key in ks}
    check(A.total_loss(stats, ks) < A.total_loss(stats, floor_ks), "expected loss below the floor assignment's")
    # ranking: the first promotion is the best loss-per-GiB one among all classes (the cap does not bind yet)
    best = max(((stats[floor_ks[key]][key]["loss"] - stats[4.0][key]["loss"]) /
                (m.cbytes(key[0], key[1], 4.0) - m.cbytes(key[0], key[1], floor_ks[key])), key)
               for key in floor_ks if floor_ks[key] == 3.0)
    check((log[0]["layer"], log[0]["class"]) == best[1], f"first promotion is the best ratio class {best[1]}")
    ks2, log2 = A.allocate(m, stats, cap, floor)
    check(ks2 == ks and log2 == log, "deterministic")
    # generous cap: everything at the highest K; floor total over the cap: refused
    ks3, _ = A.allocate(m, stats, 40 * A.GIB, floor)
    check(all(K == 4.0 for K in ks3.values()), "a 40 GiB cap per card promotes every class to KB 4")
    check(raises(lambda: A.allocate(m, stats, 20 * A.GIB, floor), RuntimeError, "no layer boundary"), "a cap below the floor total is refused")
    # a third K only for projections (KB 5) is used when it fits
    stats5 = synthetic_stats(arch, (3.0, 4.0, 5.0))
    ks5, _ = A.allocate(m, stats5, cap, floor)
    check(all(m.split(ks5, cap)[1:]) and max(ks5.values()) <= 5.0, "KB 5 sources are accepted")


def test_report_and_fallback_parse() -> None:
    if not REAL_CONFIG.exists():
        return
    check(A.parse_name("model.language_model.layers.7.mlp.experts.12.gate_proj.weight") == (7, "gu", 12), "parse gate_proj")
    check(A.parse_name("model.language_model.layers.7.mlp.experts.12.down_proj.weight") == (7, "down", 12), "parse down_proj")
    check(A.parse_name("model.language_model.layers.3.self_attn.q_proj.weight") == (3, "p", None), "parse q_proj")
    check(A.parse_floor("p=5,gu=4") == {"p": 5.0, "gu": 4.0, "down": 3.0}, "parse floor")
    check(raises(lambda: A.parse_floor("x=1"), ValueError, "unknown class"), "bad floor class refused")


# --------------------------------------------------------------------------------------------
# (b) corpus
# --------------------------------------------------------------------------------------------


def test_corpus() -> None:
    docs = [(f"d{i}", [i + 1] * n) for i, n in enumerate([4096, 100, 4000, 90, 3000, 1000, 10])]
    segs = C.pack(docs, eos=99, seg_max=4096)
    check(all(len(s[1]) <= 4096 for s in segs), "packed segments are <= 4096 tokens")
    check(sum(len(s[1]) for s in segs) == sum(len(d[1]) for d in docs) + sum(len(s[2]) - 1 for s in segs),
          "tokens kept: documents + one EOS per join")
    for name, ids, names in segs:
        parts, cur = [], []
        for t in ids:
            if t == 99:
                parts.append(cur)
                cur = []
            else:
                cur.append(t)
        parts.append(cur)
        check(len(parts) == len(names) and all(set(p) == {int(n[1:]) + 1} for p, n in zip(parts, names)),
              f"{name}: EOS-separated documents in order")
    check(segs[0][2] == ["d0"], "a full-length document gets its own segment")
    try:
        C.pack([("x", [1] * 4097)], eos=99)
        check(False, "a document > 4096 is refused")
    except ValueError:
        check(True, "a document > 4096 is refused")
    seg_list = [{"name": f"s{i}", "token_ids": [0] * 100} for i in range(50)]
    sl = C.slice_segments(seg_list, 1000)
    check(len(sl) == 10 and sl[0]["name"] == "s0" and sl[-1]["name"] == "s49", "slice: 10 evenly spread segments, first and last kept")
    check(len(C.slice_segments(seg_list, 10**9)) == 50, "slice above the total keeps everything")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r.json"
        import argparse

        C.cmd_random(argparse.Namespace(tokens=2500, vocab=4095, eos=4095, seg_len=1000, doc_len=300, seed=1, out=out))
        doc = json.loads(out.read_text())
        ids = [t for s in doc["segments"] for t in s["token_ids"]]
        check(len(ids) == 2500 and len(doc["segments"]) == 3 and 4095 in ids and max(ids) <= 4095, "random corpus shape")
    real = Path(r"E:\models\r4dx\q4exp\tokens_calib.meta.json")
    if real.exists():
        meta = json.loads(real.read_text())
        check(meta["hv2_tokens"] == 834820, "the hessian-v2 part is the documented 834,820 canonical tokens")
        check(all(s["tokens"] <= 4096 for s in meta["segment_list"]), "every real segment <= 4096 tokens")
        check(abs(meta["tokens"] - 1_000_000) < 10_000, f"real corpus ~1M tokens ({meta['tokens']})")


# --------------------------------------------------------------------------------------------
# (c) Hessian capture: synthetic rows through LayerTaps / finalize_layer
# --------------------------------------------------------------------------------------------


class StubRef:
    def __init__(self, arch, model_dir):
        self.arch, self.model_dir, self.tie_rule = arch, model_dir, "recent"


def ar1_rows(rng, n: int, k: int, scale: np.ndarray) -> torch.Tensor:
    """Correlated rows with a heavy-tailed channel scale, rounded to bf16 as real activations are."""
    z = rng.standard_normal((n, k)).astype(np.float32)
    x = z.copy()
    for j in range(1, k):
        x[:, j] = 0.7 * x[:, j - 1] + 0.714 * z[:, j]
    return torch.from_numpy(x * scale[None, :]).to(torch.bfloat16)


def test_hessian_capture() -> None:
    if not (TINY_HF / "config.json").exists():
        print(f"SKIP: hessian capture (no {TINY_HF})")
        return
    arch = Qwen4ExpArch.from_config(TINY_HF / "config.json")
    layer = 1                                       # a GDN layer
    E = arch.n_experts
    rng = np.random.default_rng(11)
    dev = torch.device("cpu")
    taps = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    pfx = arch.layer_prefix(layer)
    sc_in = np.exp(0.5 * rng.standard_normal(2560)).astype(np.float32)
    sc_h = np.exp(0.5 * rng.standard_normal(640)).astype(np.float32)
    counts = np.full(E, 2900, dtype=np.int64)
    counts[3], counts[9], counts[7] = 100, 700, 2700   # rare / gate_up-fallback only / still above 2560
    rows_in, rows_h = {}, {}
    # dense taps: two chunks each (as two segments would), companions read the same rows
    T = 3000
    x_in = ar1_rows(rng, T, 2560, sc_in)
    x_out = ar1_rows(rng, T, 6144, np.ones(6144, dtype=np.float32))
    for part in (slice(0, 1500), slice(1500, 3000)):
        taps.linear_input(layer, pfx + "linear_attn.in_proj_qkv", x_in[part])
        taps.linear_input(layer, pfx + "linear_attn.in_proj_z", x_in[part])
        taps.linear_input(layer, pfx + "linear_attn.out_proj", x_out[part])
        taps.end_segment()
    # the router sees every token; each expert its own rows (gate and up the same rows, then down's input)
    taps.linear_input(layer, pfx + "mlp.gate", x_in)
    for e in range(E):
        n = int(counts[e])
        xe = ar1_rows(rng, n, 2560, sc_in)
        he = ar1_rows(rng, n, 640, sc_h)
        rows_in[e], rows_h[e] = xe, he
        taps.linear_input(layer, arch.expert_base(layer, e, "gate_proj"), xe)
        taps.linear_input(layer, arch.expert_base(layer, e, "up_proj"), xe)
        taps.linear_input(layer, arch.expert_base(layer, e, "down_proj"), he)
    taps.moe_routing(layer, None, torch.from_numpy(counts))
    check(taps.shared["checked"] == 2 and taps.shared["equal"], "shared-input gate saw in_proj_z read the same rows (2 segments)")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cap = HQ.finalize_layer(root, layer, taps, StubRef(arch, TINY_HF), {"path": "synthetic", "sha256": "0" * 64, "segments": 2,
                                                                          "tokens": T, "tokenizer_mode": None}, {})
        out = HQ.layer_dir(root, layer)
        man = json.loads((out / "hessian.json").read_text())
        routing = json.loads((out / "routing_counts.json").read_text())
        check(routing["tokens_routed"] == counts.tolist(), "routing_counts.json: tokens_routed = the fed counts")
        from safetensors.torch import load_file

        pf = load_file(str(out / "L01.probe.safetensors"))
        check(tuple(pf["x"].shape) == (E, HQ.PROBE_ROWS, 2560) and pf["x"].dtype == torch.bfloat16
              and pf["n"].tolist() == [min(int(c), HQ.PROBE_ROWS) for c in counts] == routing["probe_rows"],
              "probe file: [E, 256, 2560] bf16 + the valid row count per expert")
        check(torch.equal(pf["x"][3, :100], rows_in[3][:100]), "probe rows are routed rows (expert 3, 100 rows: all of them, in order)")
        check(routing["fallback"]["gate_up"] == [3, 9] and routing["fallback"]["down"] == [3],
              f"fallback lists per class: gate_up {routing['fallback']['gate_up']}, down {routing['fallback']['down']}")
        check(abs(routing["fallback_share"]["gate_up"] - 2 / E) < 1e-12 and abs(routing["fallback_share"]["down"] - 1 / E) < 1e-12,
              "fallback shares")
        check(routing["tokens"] == T and routing["pairs"] == int(counts.sum()), "tokens / pairs recorded")
        check(man["keys"][arch.expert_base(layer, 4, "gate_proj")] == man["keys"][arch.expert_base(layer, 4, "up_proj")]
              == HQ.expert_file(layer, 4, "gu"), "gate and up share one Hessian file")
        check(man["keys"][pfx + "linear_attn.in_proj_qkv"] == man["keys"][pfx + "linear_attn.in_proj_z"] == "L01.in.hess"
              and man["keys"][pfx + "linear_attn.out_proj"] == "L01.out.hess", "GDN taps serve qkv + z, and out_proj")
        check(len(man["keys"]) == 3 + 3 * E, f"{len(man['keys'])} keys = 3 GDN linears + 3 per expert")
        # accuracy of the stored Hessians against an fp64 reference on the same bf16 rows
        worst = 0.0
        for e in (0, 5, 7, 12):
            f = out / HQ.expert_file(layer, e, "gu")
            H, rows = tq.read_hess(f)
            Hb, rows_b, _ = hc.read_hess_file(f)
            ref = rows_in[e].double().T @ rows_in[e].double() / rows_in[e].shape[0]
            err = float((H.double() - ref).abs().max() / ref.diagonal().max())
            worst = max(worst, err)
            check(rows == rows_b == counts[e], f"e{e} gu rows = routed count")
            check(np.array_equal(Hb, Hb.T) and torch.equal(H, H.T), f"e{e} gu: symmetric after expansion")
            if e == 7:
                check(float(torch.linalg.eigvalsh(H.double())[0]) > -1e-6 * float(H.diagonal().max()), "e7 gu: PSD (rows 2700 >= 2560)")
        check(worst < 1e-5, f"fp32 expert accumulation within 1e-5 of fp64 (worst {worst:.2e})")
        f = out / HQ.expert_file(layer, 4, "down")
        H, rows = tq.read_hess(f)
        ref = rows_h[4].double().T @ rows_h[4].double() / rows_h[4].shape[0]
        check(rows == 2900 and float((H.double() - ref).abs().max() / ref.diagonal().max()) < 1e-5, "down Hessian (640) correct")
        # dense taps: fp64 accumulation
        H, rows = tq.read_hess(out / "L01.in.hess")
        ref = x_in.double().T @ x_in.double() / T
        check(rows == T and float((H.double() - ref).abs().max() / ref.diagonal().max()) < 1e-6, "in tap (fp64 accumulate)")
        H6, _ = tq.read_hess(out / "L01.out.hess")
        check(H6.shape == (6144, 6144), "out tap K = 6144")
        # the fallback formula
        m_in = x_in.double().pow(2).mean(0)
        want = m_in + HQ.FALLBACK_DAMP * m_in.mean()
        Hf, rows_f = tq.read_hess(out / HQ.expert_file(layer, 3, "gu"))
        check(rows_f == T and torch.equal(Hf, torch.diag(torch.diagonal(Hf))), "fallback gate_up is diagonal, header rows = tokens")
        check(float(((torch.diagonal(Hf).double() - want).abs() / want).max()) < 1e-5, "fallback gate_up diagonal = m + 0.01 mean(m)")
        allh = torch.cat([rows_h[e] for e in range(E)]).double()
        want_d = allh.pow(2).mean(0)
        want_d = want_d + HQ.FALLBACK_DAMP * want_d.mean()
        Hd, rows_d = tq.read_hess(out / HQ.expert_file(layer, 3, "down"))
        check(rows_d == int(counts.sum()) and float(((torch.diagonal(Hd).double() - want_d).abs() / want_d).max()) < 1e-5,
              "fallback down diagonal = pooled h^2 moment (+ damp), header rows = pairs")
        Hf9, _ = tq.read_hess(out / HQ.expert_file(layer, 9, "gu"))
        Hd9, rows9 = tq.read_hess(out / HQ.expert_file(layer, 9, "down"))
        check(torch.equal(Hf9, Hf), "every gate_up fallback expert gets the same layer-wide diagonal")
        check(rows9 == 700 and not torch.equal(Hd9, torch.diag(torch.diagonal(Hd9))), "expert 9 (700 rows >= 640): down keeps its real Hessian")
        check(cap["fallback_count"] == {"gate_up": 2, "down": 1}, "capture.json fallback counts")
        done = json.loads((out / "DONE.json").read_text())
        check(set(done["files"]) >= {"hessian.json", "routing_counts.json", "L01.in.hess"}, "DONE.json hashes the files")
        # the layer check subcommand's gates pass on it
        import argparse

        rc = HQ.cmd_check(argparse.Namespace(root=str(root), layer=layer, eig_sample=2))
        check(rc == 0, "hessian_q4 check: symmetric / finite / positive diagonal / rows >= K / PSD sample")
    # a routed-count mismatch is refused
    taps2 = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    taps2.linear_input(layer, pfx + "linear_attn.in_proj_qkv", x_in[:50])
    taps2.linear_input(layer, pfx + "linear_attn.out_proj", x_out[:50])
    taps2.linear_input(layer, pfx + "mlp.gate", x_in[:50])
    for e in range(E):
        taps2.linear_input(layer, arch.expert_base(layer, e, "gate_proj"), x_in[:5])
        taps2.linear_input(layer, arch.expert_base(layer, e, "down_proj"), x_in[:5, :640])
    taps2.moe_routing(layer, None, torch.full((E,), 6, dtype=torch.int64))
    with tempfile.TemporaryDirectory() as td:
        check(raises(lambda: HQ.finalize_layer(Path(td), layer, taps2, StubRef(arch, TINY_HF),
                                               {"path": "x", "sha256": "0", "segments": 1, "tokens": 50, "tokenizer_mode": None}, {}),
                     RuntimeError, "routed counts disagree"), "routed-count mismatch refused")


# --------------------------------------------------------------------------------------------
# (d) boundaries
# --------------------------------------------------------------------------------------------


def test_boundary() -> None:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        b = HQ.Boundary(work, 2, [5, 3], 16)
        b.open(create=True)
        x0 = torch.randn(1, 5, 16).to(torch.bfloat16)
        x1 = torch.randn(1, 3, 16).to(torch.bfloat16)
        b.write(0, x0)
        b.write(1, x1)
        b.save_meta(True)
        r = HQ.Boundary(work, 2, [5, 3], 16)
        r.load_meta()
        r.open(create=False)
        check(torch.equal(r.read(0, torch.device("cpu")), x0) and torch.equal(r.read(1, torch.device("cpu")), x1), "boundary round trip")
        r.mm[2, 3] ^= 0x10
        check(raises(lambda: r.read(0, torch.device("cpu")), RuntimeError, "fails its crc32"), "a corrupted segment is refused")
        check(torch.equal(r.read(1, torch.device("cpu")), x1), "the other segment still reads")
        check(raises(lambda: b.write(1, x0), RuntimeError, "rows"), "wrong row count refused")
        b.close()
        r.close()


def test_probe_reservoir() -> None:
    if not (TINY_HF / "config.json").exists():
        print(f"SKIP: probe reservoir (no {TINY_HF})")
        return
    arch = Qwen4ExpArch.from_config(TINY_HF / "config.json")
    layer, P = 1, HQ.PROBE_ROWS
    dev = torch.device("cpu")

    def feed(taps, e, ids):  # rows whose element 0 is the (exactly representable) row id, so the sample can be audited
        x = torch.zeros(len(ids), 2560, dtype=torch.bfloat16)
        t = torch.tensor(ids)
        x[:, 0] = (t // 64).to(torch.bfloat16)   # bf16 holds integers exactly only up to 256: the id in base 64
        x[:, 1] = (t % 64).to(torch.bfloat16)
        taps.linear_input(layer, arch.expert_base(layer, e, "gate_proj"), x)

    calls = [list(range(0, 100)), list(range(100, 400)), list(range(400, 3000)), list(range(3000, 3030))]
    ta = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    for e in (0, 1, 2):
        for ids in calls:
            feed(ta, e, ids)
    def ids_of(probe, e, n=None):
        a = probe[e, :n].float().long()
        return (a[:, 0] * 64 + a[:, 1]).numpy()

    first = ids_of(ta.probe, 0)
    check(len(set(first.tolist())) == P and first.min() >= 0 and first.max() < 3030, "reservoir: P distinct rows of the stream")
    check(abs(first.mean() - 1515) < 300, f"reservoir: uniform over the stream (mean row id {first.mean():.0f}, expected ~1515)")
    check(not np.array_equal(ids_of(ta.probe, 0), ids_of(ta.probe, 1)), "reservoir: experts draw different samples")
    tb = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    for e in (0, 1, 2):
        for ids in calls:
            feed(tb, e, ids)
    check(torch.equal(ta.probe, tb.probe), "reservoir: deterministic (seeded per layer)")
    # an interrupted run (state saved, taps rebuilt, state loaded) draws the same sample as an uninterrupted one
    tc = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    for e in (0, 1, 2):
        for ids in calls[:2]:
            feed(tc, e, ids)
    td = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    td.load_small(tc.small_state())
    for e in (0, 1, 2):
        for ids in calls[2:]:
            feed(td, e, ids)
    # tc / td ran experts in a different interleaving than ta/tb (rng draws are consumed in call order), so compare to a
    # reference with the same interleaving
    te = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    for e in (0, 1, 2):
        for ids in calls[:2]:
            feed(te, e, ids)
    for e in (0, 1, 2):
        for ids in calls[2:]:
            feed(te, e, ids)
    check(torch.equal(td.probe, te.probe), "reservoir: resuming from a saved state continues the same stream")
    # fewer rows than P: all of them, in order
    tf = HQ.LayerTaps(StubRef(arch, TINY_HF), layer, dev)
    feed(tf, 5, list(range(40)))
    feed(tf, 5, list(range(40, 90)))
    check(ids_of(tf.probe, 5, 90).tolist() == list(range(90)), "reservoir: an expert with < P rows keeps them all")


def test_class_stats() -> None:
    def rec(layer, tap, cls, proxy, energy, proxy_abs, routed=None, tokens=1000, probe=None, fb=False, k=3.0):
        r = {"K": k, "proxy": proxy, "energy": energy, "proxy_abs": proxy_abs, "numel": 100, "rel_weight_err": 0.1,
             "bits": {"total": 400}, "tap": tap, "class": cls, "layer": layer, "hessian": {"tokens": tokens}}
        if routed is not None:
            r["hessian"].update({"routed": routed, "fallback": fb})
        if probe is not None:
            r["probe"] = probe
        return r

    pfx = "model.language_model.layers.0."
    man = {"tensors": {
        pfx + "linear_attn.in_proj_qkv.weight": rec(0, "in", "p", 0.01, 5.0, 0.05),
        pfx + "linear_attn.in_proj_z.weight": rec(0, "in", "p", 0.03, 4.0, 0.12),
        pfx + "linear_attn.out_proj.weight": rec(0, "out", "p", 0.02, 9.0, 0.18),
        pfx + "mlp.experts.0.gate_proj.weight": rec(0, "e0.gu", "gu", 0.1, 2.0, 0.2, 100, probe={"rows": 8, "out_err": 0.5, "out_energy": 3.0}),
        pfx + "mlp.experts.0.up_proj.weight": rec(0, "e0.gu", "gu", 0.1, 2.0, 0.3, 100, probe={"rows": 8, "out_err": 0.5, "out_energy": 3.0}),
        pfx + "mlp.experts.1.gate_proj.weight": rec(0, "e1.gu", "gu", 0.1, 2.0, 0.2, 300, probe={"rows": 8, "out_err": 0.1, "out_energy": 3.0}, fb=True),
        pfx + "mlp.experts.1.up_proj.weight": rec(0, "e1.gu", "gu", 0.1, 2.0, 0.2, 300, probe={"rows": 8, "out_err": 0.1, "out_energy": 3.0}, fb=True),
        pfx + "mlp.experts.0.down_proj.weight": rec(0, "e0.down", "down", 0.1, 2.0, 0.4, 100, probe={"rows": 8, "out_err": 0.02, "out_energy": 3.0}),
    }}
    s = A.class_stats(man, "out")
    check(abs(s[(0, "p")]["loss"] - 9.0 * (0.01 + 0.03 + 0.02)) < 1e-12, "p loss = E_mixer (out_proj energy) x sum of the relative proxies")
    check(abs(s[(0, "gu")]["loss"] - (0.1 * 0.5 + 0.3 * 0.1)) < 1e-12, "gu loss = sum_e routed/tokens x probe out_err, once per expert (gate + up)")
    check(abs(s[(0, "down")]["loss"] - 0.1 * 0.02) < 1e-12, "down loss from its probe error")
    check(s[(0, "gu")]["experts"] == 2 and s[(0, "gu")]["fallback"] == 1, "unique experts / fallback experts counted once")
    sl = A.class_stats(man, "local")
    check(abs(sl[(0, "gu")]["loss"] - (0.1 * 0.2 + 0.1 * 0.3 + 0.3 * 0.2 + 0.3 * 0.2)) < 1e-12, "local loss = routing-weighted sum of proxy_abs")
    check(abs(sl[(0, "p")]["loss"] - (0.05 + 0.12 + 0.18)) < 1e-12, "p local loss = sum of proxy_abs")
    del man["tensors"][pfx + "mlp.experts.1.up_proj.weight"]["probe"]
    del man["tensors"][pfx + "mlp.experts.1.gate_proj.weight"]["probe"]
    check(raises(lambda: A.class_stats(man, "out"), ValueError, "no probe error"), "an expert without probe error is refused (out mode)")
    check(A.class_stats(man, "local")[(0, "gu")]["loss"] > 0, "... and fine in local mode")


def main() -> int:
    run_e2e = "--e2e" in sys.argv
    tests = [test_choose_split, test_documented_numbers, test_allocation, test_report_and_fallback_parse, test_corpus,
             test_hessian_capture, test_probe_reservoir, test_class_stats, test_boundary]
    for t in tests:
        try:
            t()
        except Exception:  # noqa: BLE001
            FAILS.append(f"{t.__name__} raised")
            traceback.print_exc()
    if run_e2e:
        try:
            import tiny_e2e

            dev = sys.argv[sys.argv.index("--e2e-device") + 1] if "--e2e-device" in sys.argv else "cpu"
            res = tiny_e2e.run(device=dev, check=check)
            print(json.dumps(res, indent=1, default=str)[:3000])
        except Exception:  # noqa: BLE001
            FAILS.append("tiny_e2e raised")
            traceback.print_exc()
    if FAILS:
        print(f"{len(FAILS)} FAILED of {CHECKS} checks")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
