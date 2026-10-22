"""CPU test of `gate_q35.py eval` on tiny-q35moe dumps (`tools/reference/qwen35moe/tiny.py build` first).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\test_gate_q35.py

HF bf16 / fp32 dumps of three tiny segments (a chat one with a score_mask, two raw ones) come from ref_run.py; the "test" dumps are the
bf16 dump itself (KL 0, passes both gates) and a copy with Gaussian noise on the logits (an independent numpy KL must equal the
report's pooled value; the position buckets must partition the scored rows; the chat segment only counts its masked rows).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")

from qwen35moe import tiny  # noqa: E402

TINY_HF = tiny.DEFAULT_OUT / "hf"
RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, what: str) -> None:
    RESULTS.append((bool(cond), what))
    print(("ok   " if cond else "FAIL ") + what, flush=True)


def run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)


def dump(tokens: Path, out: Path, work: Path, dtype: str) -> None:
    while True:
        p = run(str(HERE / "ref_run.py"), "logprobs", "--tokens", str(tokens), "--out-dir", str(out), "--work", str(work),
                "--dtype", dtype, "--model-dir", str(TINY_HF), "--device", "cpu")
        if p.returncode != 75:
            break
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]


def read(d: Path, name: str, rows: int, V: int) -> np.ndarray:
    return np.fromfile(d / f"{name}.logprobs.f16", dtype="<f2").reshape(rows, V).astype(np.float64)


def main() -> int:
    rng = np.random.default_rng(5)
    vocab = json.loads((TINY_HF / "config.json").read_text())["text_config"]["vocab_size"]
    mask = [0] * 12 + [1] * 40
    segs = [
        {"name": "chat_a", "kind": "chat", "group": "english", "token_ids": [int(x) for x in rng.integers(0, vocab, 52)], "score_mask": mask},
        {"name": "raw_b", "kind": "raw", "group": "code", "token_ids": [int(x) for x in rng.integers(0, vocab, 70)]},
        {"name": "raw_c", "kind": "raw", "group": "thai", "token_ids": [int(x) for x in rng.integers(0, vocab, 33)]},
    ]
    tmp = Path(tempfile.mkdtemp(prefix="gate_q35_test_"))
    tok = tmp / "tokens.json"
    tok.write_text(json.dumps({"tokenizer": "tiny", "segments": segs}), encoding="utf-8")
    dump(tok, tmp / "bf16", tmp / "w1", "bf16")
    dump(tok, tmp / "fp32", tmp / "w2", "fp32")

    same, noisy = tmp / "same", tmp / "noisy"
    shutil.copytree(tmp / "bf16", same)
    shutil.copytree(tmp / "bf16", noisy)
    V = vocab
    for s in segs:
        rows = len(s["token_ids"]) - 1
        lp = read(tmp / "bf16", s["name"], rows, V)
        z = lp + rng.normal(0, 0.3, lp.shape)
        z -= np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) + z.max(1, keepdims=True)
        z.astype("<f2").tofile(noisy / f"{s['name']}.logprobs.f16")

    rc = run(str(HERE / "gate_q35.py"), "eval", "--tokens", str(tok), "--stock", str(tmp / "bf16"), "--truth", str(tmp / "fp32"),
             "--test", f"same={same}", "--test", f"noisy={noisy}", "--out", str(tmp / "report"), "--row-chunk", "16")
    print(rc.stdout[-600:], rc.stderr[-800:])
    rep = json.loads((tmp / "report.json").read_text())
    s_, n_ = rep["tests"]["same"], rep["tests"]["noisy"]
    check(s_["gate_a"]["pooled"]["mean_kl"] == 0.0 and s_["gate_a"]["pooled"]["top1_pct"] == 100.0 and s_["gate_a"]["pass"],
          "a dump identical to the stock bf16 one has KL 0, top-1 100 %, passes gate A")
    check(s_["gate_b"]["groups"]["ALL"]["mean_kl"] == rep["noise"]["ALL"]["mean_kl"], "identical dump: KL vs truth equals the noise term")
    check(s_["gate_b"]["pass"], "identical dump passes gate B (KL <= 1.5 N + 0.005, top-1 >= N - 1pt)")

    # independent numpy KL of the noisy dump vs bf16, scored rows only
    num, den = 0.0, 0
    for s in segs:
        rows = len(s["token_ids"]) - 1
        a, b = read(tmp / "bf16", s["name"], rows, V), read(noisy, s["name"], rows, V)
        m = np.asarray(s["score_mask"][1:], bool) if "score_mask" in s else np.ones(rows, bool)
        kl = (np.exp(a[m]) * (a[m] - b[m])).sum(1)
        num += kl.sum()
        den += int(m.sum())
    got = n_["vs_bf16"]["ALL"]
    check(got["rows"] == den == 40 + 69 + 32, f"pooled rows are the scored rows only ({got['rows']})")
    check(abs(got["mean_kl"] - num / den) < 1e-9, f"pooled KL equals an independent numpy computation ({got['mean_kl']:.6f} vs {num / den:.6f})")
    check(sum(b["rows"] for b in n_["vs_bf16_by_position"]) == den, "position buckets partition the scored rows")
    check(abs(sum(b["mean_kl"] * b["rows"] for b in n_["vs_bf16_by_position"]) / den - got["mean_kl"]) < 1e-9, "bucket means recombine to the pooled mean")
    check(set(n_["gate_b"]["groups"]) >= {"english", "thai", "code", "ALL"}, "gate B reports english / thai / code / ALL")
    check((tmp / "report.md").read_text().count("## ") >= 3, "markdown report written")

    rc = run(str(HERE / "gate_q35.py"), "attrib", "--tokens", str(tok), "--stock", str(tmp / "bf16"), "--test", f"noisy={noisy}",
             "--test", f"same={same}", "--out", str(tmp / "attrib"), "--row-chunk", "16")
    att = json.loads((tmp / "attrib.json").read_text())["tests"]
    check(rc.returncode == 0 and abs(att[0]["groups"]["ALL"]["mean_kl"] - got["mean_kl"]) < 1e-12 and att[1]["groups"]["ALL"]["mean_kl"] == 0.0,
          "attrib: same pooled KL as eval for the noisy dump, 0 for the identical one")

    bad = [w for ok, w in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
