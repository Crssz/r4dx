"""CPU unit test (numpy only, synthetic log-probs) of kl_report.py's score-mask mode and
`--gate gemma-fp32` (docs/gemma4-plan.md 9.9).

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\test_kl_report_gate.py      (or pytest)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kl_report as K  # noqa: E402

V = 24


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def write_dump(d: Path, name: str, logp: np.ndarray, ids: list[int]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.clip(logp, K.LOGPROB_CLAMP, None).astype("<f2").tofile(d / f"{name}.logprobs.f16")
    (d / f"{name}.meta.json").write_text(json.dumps({
        "T": len(ids), "V": V, "dtype": "float16", "rows": len(ids) - 1, "n_layers": 48,
        "sha256_of_token_ids_json": K.token_ids_sha256(ids)}), encoding="utf-8")


def fp64_kl(p, q):
    return (np.exp(p) * (p - q)).sum(-1)


class World:
    """Truth logits (peaked, so top-1 is well defined) and perturbed copies."""

    def __init__(self, tmp: Path, seed=0):
        self.rng = np.random.default_rng(seed)
        self.tmp = tmp
        self.segs = {}   # name -> (ids, mask, truth_logits)
        spec = [("e1", "english", 260, 12), ("t1", "thai", 240, 10), ("p1", "python", 300, 14), ("c1", "cpp", 220, 8)]
        for name, group, T, prompt in spec:
            ids = [2] + [int(x) for x in self.rng.integers(4, V, T - 1)]
            mask = [0] * prompt + [1] * (T - prompt)
            logits = self.rng.normal(0, 1, (T - 1, V)) * 2.5
            self.segs[name] = {"group": group, "ids": ids, "mask": mask, "truth": logits}
        self.raw = {"r1": {"ids": [2] + [int(x) for x in self.rng.integers(4, V, 33)],
                           "truth": self.rng.normal(0, 1, (33, V)) * 2.5}}
        for name, s in self.segs.items():
            write_dump(tmp / "truth", name, lsm(s["truth"]), s["ids"])
        for name, s in self.raw.items():
            write_dump(tmp / "truth", name, lsm(s["truth"]), s["ids"])

    def dump(self, dirname: str, sigma: float, seed: int, flip_scored: bool = False):
        rng = np.random.default_rng(seed)
        for name, s in list(self.segs.items()) + list(self.raw.items()):
            lg = s["truth"] + rng.normal(0, sigma, s["truth"].shape)
            if flip_scored and "mask" in s:
                # swap the top-2 logits on every scored row: top-1 collapses, KL grows
                rows = np.flatnonzero(np.asarray(s["mask"][1:], dtype=bool))
                o = np.argsort(lg[rows], axis=1)
                lg[rows, o[:, -1]], lg[rows, o[:, -2]] = lg[rows, o[:, -2]].copy(), lg[rows, o[:, -1]].copy()
            write_dump(self.tmp / dirname, name, lsm(lg), s["ids"])

    def tokens(self) -> tuple[Path, Path]:
        chat = {"tokenizer_arch": "gemma4", "segments": [
            {"name": n, "group": s["group"], "token_ids": s["ids"], "score_mask": s["mask"]}
            for n, s in self.segs.items()]}
        raw = {"tokenizer_arch": "gemma4", "segments": [{"name": n, "token_ids": s["ids"]} for n, s in self.raw.items()]}
        pc, pr = self.tmp / "chat.json", self.tmp / "raw.json"
        pc.write_text(json.dumps(chat)), pr.write_text(json.dumps(raw))
        return pc, pr


def test_row_mask_and_masked_kl():
    with tempfile.TemporaryDirectory() as t:
        w = World(Path(t))
        w.dump("test", 0.3, 1)
        s = w.segs["e1"]
        seg = K.compare_segment("e1", s["ids"], Path(t) / "truth", Path(t) / "test", 7, True,
                                row_mask=K.row_mask_from_score_mask(s["mask"]))
        # rows are scored iff the TOKEN they predict (i+1) is a model-turn token
        want_rows = np.flatnonzero(np.asarray(s["mask"][1:], dtype=bool))
        assert seg["rows"] == want_rows.size and seg["rows_total"] == len(s["ids"]) - 1
        assert (seg["_pos"] == want_rows).all()
        p = lsm(s["truth"]).astype("<f2").astype(np.float64)
        rng = np.random.default_rng(1)
        q = lsm(s["truth"] + rng.normal(0, 0.3, s["truth"].shape)).astype("<f2").astype(np.float64)
        want = fp64_kl(p, q)[want_rows]
        assert np.allclose(seg["_kl"], want, atol=1e-9), np.abs(seg["_kl"] - want).max()
        full = K.compare_segment("e1", s["ids"], Path(t) / "truth", Path(t) / "test", 7, True)
        assert full["rows"] == len(s["ids"]) - 1 and abs(full["mean_kl"] - seg["mean_kl"]) > 1e-6
        assert seg["max_kl_position"] in want_rows


def test_gate_pass_fail_and_groups():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        w = World(tmp)
        w.dump("noise", 0.1, 11)        # stock HF bf16 sdpa stand-in
        w.dump("good", 0.1, 12)         # same noise level: must pass
        w.dump("bad", 0.9, 13)           # far noisier: KL fails
        w.dump("flip", 0.02, 14, flip_scored=True)
        pc, pr = w.tokens()
        args = dict(truth_dir=tmp / "truth", noise_dir=tmp / "noise", chat_tokens=pc, raw_tokens=pr, raw_max_tokens=0)
        good = K.gate_gemma_fp32(test_dir=tmp / "good", **args)
        names = [g["name"] for g in good["groups"]]
        assert names == ["chat-code", "chat-english", "chat-thai", "chat-ALL", "raw:r1", "raw-ALL"], names
        assert good["pass"], K.gate_gemma_fp32_markdown(good)
        g_all = next(g for g in good["groups"] if g["name"] == "chat-ALL")
        assert abs(g_all["thresholds"]["kl_max"] - (1.5 * g_all["noise"]["mean_kl"] + 0.005)) < 1e-12
        assert abs(g_all["thresholds"]["top1_min_pct"] - (g_all["noise"]["top1_pct"] - 1.0)) < 1e-12
        code = next(g for g in good["groups"] if g["name"] == "chat-code")
        assert code["segments"] == ["p1", "c1"]
        raw_ = next(g for g in good["groups"] if g["name"] == "raw:r1")
        assert raw_["pass"] is None and not raw_["gated"]       # secondary: reported only
        assert "median_kl" in g_all["test"] and "p99_kl" in g_all["test"] and g_all["worst_rows"]

        bad = K.gate_gemma_fp32(test_dir=tmp / "bad", **args)
        assert not bad["pass"] and not next(g for g in bad["groups"] if g["name"] == "chat-ALL")["kl_ok"]
        # raw groups never decide the verdict: a gate over chat only passes/fails the same way
        flip = K.gate_gemma_fp32(test_dir=tmp / "flip", **args)
        fg = next(g for g in flip["groups"] if g["name"] == "chat-ALL")
        assert not fg["top1_ok"] and not flip["pass"]
        assert K.gate_gemma_fp32_markdown(flip).count("FAIL") >= 1


def test_quantized_increment_mode():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        w = World(tmp)
        w.dump("noise", 0.1, 21)
        w.dump("base", 0.08, 22)          # r4dx bf16 container
        w.dump("q_ok", 0.085, 23)         # tiny increment
        w.dump("q_bad", 0.7, 24)         # large increment
        pc, pr = w.tokens()
        kw = dict(truth_dir=tmp / "truth", noise_dir=tmp / "noise", chat_tokens=pc, base_dir=tmp / "base")
        ok = K.gate_gemma_fp32(test_dir=tmp / "q_ok", **kw)
        bad = K.gate_gemma_fp32(test_dir=tmp / "q_bad", **kw)
        assert ok["mode"] == "quantized-increment"
        a = next(g for g in ok["groups"] if g["name"] == "chat-ALL")
        assert abs(a["increment_kl"] - (a["test"]["mean_kl"] - a["base"]["mean_kl"])) < 1e-12
        assert a["increment_kl"] < K.GATE_QUANT_INCREMENT and ok["pass"]
        b = next(g for g in bad["groups"] if g["name"] == "chat-ALL")
        assert b["increment_kl"] > K.GATE_QUANT_INCREMENT and not bad["pass"]


def test_existing_behaviour_unchanged_without_mask():
    with tempfile.TemporaryDirectory() as t:
        w = World(Path(t))
        w.dump("test", 0.3, 5)
        s = w.segs["t1"]
        a = K.compare_segment("t1", s["ids"], Path(t) / "truth", Path(t) / "test", 5, True)
        assert a["rows"] == a["rows_total"] and (a["_pos"] == np.arange(a["rows"])).all()


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("PASS")

