"""CPU tests of tools/reference/qwen4exp/gate_eval.py (the two qwen4_exp quality gates, plan 4.10 / Q10).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe -m pytest tests\\reference\\test_q4exp_gate.py -q

Synthetic log-probabilities in a tiny vocabulary (numpy only; no GPU, no model): gate A / gate B arithmetic, the pass and fail sides of
both thresholds, the end-to-end `evaluate` (fake reference root + noise cache) with its JSON / markdown report, and a missing dump.
The real-data self-test (`gate_eval.py --self-test`, REAL_REF.md section 1 (a): canon KL 0.01386 / 95.19%) runs when
E:\\models\\r4dx\\q4exp\\ref is present, else it skips.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

REF = Path(__file__).resolve().parents[2] / "tools" / "reference"
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF / "qwen4exp"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import gate_eval  # noqa: E402
import kl_report  # noqa: E402

V = 24
SEGS = {"english_prose": 1, "thai_prose": 2, "python_source": 3, "cpp_source": 4}


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def write_dump(d: Path, name: str, logp: np.ndarray, ids: list[int]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.clip(logp, kl_report.LOGPROB_CLAMP, None).astype("<f2").tofile(d / f"{name}.logprobs.f16")
    (d / f"{name}.meta.json").write_text(json.dumps({
        "T": len(ids), "V": V, "dtype": "float16", "rows": len(ids) - 1, "n_layers": 4,
        "sha256_of_token_ids_json": kl_report.token_ids_sha256(ids)}), encoding="utf-8")


class World:
    """Four canon-named segments: peaked truth logits, a bf16 'reference' and an 'r4dx' dump at chosen noise levels."""

    def __init__(self, tmp: Path, T: int = 400):
        self.tmp = tmp
        rng = np.random.default_rng(7)
        self.ids = {n: [2] + [int(x) for x in rng.integers(4, V, T - 1)] for n in SEGS}
        self.truth = {n: rng.normal(0, 1, (T - 1, V)) * 2.5 for n in SEGS}
        self.root = tmp / "ref"
        self.corpus = tmp / "corpus"
        self.corpus.mkdir()
        (self.corpus / "tokens_canon.json").write_text(json.dumps(
            {"tokenizer": "synthetic", "segments": [{"name": n, "token_ids": self.ids[n]} for n in SEGS]}), encoding="utf-8")
        for n in SEGS:
            write_dump(self.root / "truth-fp32", n, lsm(self.truth[n]), self.ids[n])
        self.dump("canon", 0.35, 11)        # the bf16 reference: sits at the "noise" distance from the truth
        self.noise = self.pool_kl("truth-fp32", "canon")

    def dump(self, where: str, sigma: float, seed: int) -> Path:
        rng = np.random.default_rng(seed)
        d = (self.root / where) if where == "canon" else (self.tmp / where)
        for n in SEGS:
            write_dump(d, n, lsm(self.truth[n] + rng.normal(0, sigma, self.truth[n].shape)), self.ids[n])
        return d

    def pool_kl(self, ref: str, test: str) -> dict:
        res = [kl_report.compare_segment(n, self.ids[n], self.root / ref, self.root / test, 32, True) for n in SEGS]
        o = kl_report.overall(res)
        return {"mean_kl": o["mean_kl"], "top1_pct": o["top1_agreement_pct"]}

    def write_noise_cache(self, per_group: dict) -> None:
        d = self.root / "noise_floor"
        d.mkdir(parents=True, exist_ok=True)
        rows = {g: {"noise_mean_kl": n["mean_kl"], "noise_top1_pct": n["top1_pct"], **kl_report.gate_thresholds(n)} for g, n in per_group.items()}
        (d / "truth_gate_exact.json").write_text(json.dumps({"scopes": {"canon": {"gate_proposal": rows}}}), encoding="utf-8")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    monkeypatch.setattr(gate_eval, "KL_CORPUS_DIR", w.corpus)
    # the cache holds the noise term the synthetic bf16 reference defines, for every group (same value: one noise level)
    w.write_noise_cache({g: w.noise for g in gate_eval.GROUP_ORDER})
    return w


def results_vs(world: World, ref: Path, test: Path) -> dict:
    return {n: kl_report.compare_segment(n, world.ids[n], ref, test, 32, True) for n in SEGS}


def test_gate_a_thresholds_are_the_original_ones():
    assert gate_eval.GATE_A_KL_MAX == 0.02 and gate_eval.GATE_A_TOP1_MIN == 93.0


def test_gate_a_pass_and_fail(world):
    near = world.dump("near", 0.02, 21)
    far = world.dump("far", 1.5, 22)
    ref = world.root / "truth-fp32"   # gate A only sees "a reference dump"; the truth keeps the sigma <-> KL mapping clean
    good = gate_eval.gate_a(results_vs(world, ref, near))
    bad = gate_eval.gate_a(results_vs(world, ref, far))
    assert good["pass"] and good["pooled"]["mean_kl"] < 0.02 and good["pooled"]["top1_pct"] >= 93.0
    assert not bad["pass"] and bad["pooled"]["mean_kl"] > 0.02
    assert set(good["segments"]) == set(SEGS) and good["pooled"]["rows"] == 4 * 399


def test_gate_a_top1_side_alone_fails(world):
    # KL stays tiny but the top-1 of every row flips: swap the two largest logits' ranks via a fake test dump
    d = world.tmp / "flip"
    for n in SEGS:
        lg = world.truth[n].copy()
        o = np.argsort(lg, axis=1)
        r = np.arange(lg.shape[0])
        a, b = lg[r, o[:, -1]].copy(), lg[r, o[:, -2]].copy()
        lg[r, o[:, -1]], lg[r, o[:, -2]] = b - 1e-4, a  # top-2 becomes the (barely) larger one for the rows where they swap
        write_dump(d, n, lsm(lg), world.ids[n])
    res = results_vs(world, world.root / "truth-fp32", d)
    g = gate_eval.gate_a(res)
    assert g["pooled"]["top1_pct"] < 93.0 and not g["pass"]


def test_gate_b_groups_and_thresholds(world):
    thr = {g: {"noise_mean_kl": world.noise["mean_kl"], "noise_top1_pct": world.noise["top1_pct"],
               **kl_report.gate_thresholds(world.noise)} for g in gate_eval.GROUP_ORDER}
    res = results_vs(world, world.root / "truth-fp32", world.root / "canon")  # at the noise distance itself
    b = gate_eval.gate_b(res, thr)
    assert set(b["groups"]) == set(gate_eval.GROUP_ORDER)
    assert b["groups"]["ALL"]["rows"] == 4 * 399 and b["groups"]["code"]["rows"] == 2 * 399   # code = python + cpp
    assert b["groups"]["english"]["rows"] == 399 and b["groups"]["thai"]["rows"] == 399
    assert b["groups"]["ALL"]["kl_max"] == pytest.approx(1.5 * world.noise["mean_kl"] + 0.005)
    assert b["groups"]["ALL"]["top1_min_pct"] == pytest.approx(world.noise["top1_pct"] - 1.0)
    assert b["pass"]
    far = results_vs(world, world.root / "truth-fp32", world.dump("far", 1.5, 22))
    assert not gate_eval.gate_b(far, thr)["pass"]
    # a single failing group fails the gate
    tight = {**thr, "thai": {**thr["thai"], "kl_max": 1e-6}}
    r = gate_eval.gate_b(res, tight)
    assert not r["pass"] and not r["groups"]["thai"]["pass"] and r["groups"]["english"]["pass"]


def test_evaluate_end_to_end_and_report(world):
    near = world.dump("near", world.noise["mean_kl"] ** 0.5 * 0.5, 31)   # about as far from the truth as the reference is
    rep = gate_eval.evaluate(near, ["canon"], world.root, "truth-fp32", 32, False)
    sc = rep["scopes"]["canon"]
    assert sc["gate_a"]["counts"] and sc["gate_b"]["counts"]
    assert sc["gate_b"]["noise_source"].startswith("cached")
    assert rep["pass_a"] == sc["gate_a"]["pass"] and rep["pass_b"] == sc["gate_b"]["pass"]
    md = gate_eval.render_markdown(rep)
    assert "| A original | canon |" in md and "| B noise-relative | canon / ALL |" in md and "english_prose" in md
    json.dumps(rep)  # serialisable

    far = world.dump("far", 1.5, 22)
    rep2 = gate_eval.evaluate(far, ["canon"], world.root, "truth-fp32", 32, False)
    assert rep2["pass"] is False and rep2["pass_a"] is False and rep2["pass_b"] is False


def test_recomputed_noise_matches_the_dumps(world):
    # no cache: the thresholds come from truth_gate's functions (needs the +1-token dump, provided here as canon_p1)
    (world.root / "noise_floor" / "truth_gate_exact.json").unlink()
    p1 = world.root / "canon_p1"
    ids1 = {n: world.ids[n] + [5] for n in SEGS}
    rng = np.random.default_rng(99)
    for n in SEGS:   # one more token, a different noise draw: prefix_check compares its first rows with the canon dump
        lg = np.concatenate([world.truth[n], rng.normal(0, 1, (1, V)) * 2.5])
        write_dump(p1, n + "_p1", lsm(lg + rng.normal(0, 0.35, lg.shape)), ids1[n])
    (p1 / "tokens_canon_p1.json").write_text(json.dumps(
        {"segments": [{"name": n + "_p1", "token_ids": ids1[n]} for n in SEGS]}), encoding="utf-8")
    thr, src = gate_eval.noise_thresholds(world.root, "truth-fp32", "canon", 32, False)
    assert src.startswith("recomputed") and set(thr) == set(gate_eval.GROUP_ORDER)
    assert all(t["kl_max"] == pytest.approx(1.5 * t["noise_mean_kl"] + 0.005) for t in thr.values())


def test_missing_dump_is_an_error(world, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        gate_eval.evaluate(empty, ["canon"], world.root, "truth-fp32", 32, False)
    assert gate_eval.main(["--test-dir", str(empty), "--root", str(world.root)]) == 2


def test_custom_scope_is_gate_a_only(world):
    near = world.dump("near", 0.02, 21)
    rep = gate_eval.evaluate(near, [], world.root, "truth-fp32", 32, False,
                             custom={"tokens": world.corpus / "tokens_canon.json", "ref_dir": world.root / "truth-fp32"})
    assert list(rep["scopes"]) == ["custom"] and rep["scopes"]["custom"]["gate_b"] is None
    assert rep["pass_b"] is None and rep["pass_a"] is True and rep["pass"] is True


@pytest.mark.skipif(not (gate_eval.DEFAULT_ROOT / "truth-fp32" / "cpp_source.logprobs.f16").exists(), reason="real reference dumps absent")
def test_real_self_test_reproduces_real_ref_numbers():
    assert gate_eval.self_test(gate_eval.DEFAULT_ROOT, "truth-fp32", 32, False) == 0
