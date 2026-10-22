"""CPU tests of `ref_run.py` on tiny-q35moe (`tools/reference/qwen35moe/tiny.py build` first).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\test_ref_run.py

* logprobs: the layer-major multi-segment run, resumed after every layer (exit 75), writes the same log-probs as
  `Qwen35MoeStreamingReference.forward_logits` (bf16 and fp32), in the Rung 4 file format (f16 [T-1, V] + meta.json).
* kvcalib: the recorded per-kv-head amax of K (post-rope) and V equal an independent measurement (apply_rotary_pos_emb and v_proj
  wrapped in a plain streaming run), the JSON has exactly the full-attention layers with kv_heads entries, and the result is the
  same whether the run was resumed or not and whether it came from `kvcalib` or `logprobs --record-kv`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qwen35moe import tiny  # noqa: E402
from qwen35moe.ref import Qwen35MoeStreamingReference  # noqa: E402

TINY_HF = tiny.DEFAULT_OUT / "hf"
RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, what: str) -> None:
    RESULTS.append((bool(cond), what))
    print(("ok   " if cond else "FAIL ") + what, flush=True)


def run_tool(*args: str) -> tuple[int, int]:
    """Runs ref_run.py until it stops asking for another slice (exit 75); returns (final exit code, number of runs)."""
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    n = 0
    while True:
        p = subprocess.run([sys.executable, str(HERE / "ref_run.py"), *args, "--model-dir", str(TINY_HF), "--device", "cpu"], env=env,
                           capture_output=True, text=True)
        n += 1
        if p.returncode == 75 and n < 100:
            continue
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-2000:])
        return p.returncode, n


def main() -> int:
    rng = np.random.default_rng(3)
    vocab = json.loads((TINY_HF / "config.json").read_text())["text_config"]["vocab_size"]
    segs = [{"name": f"s{k}", "token_ids": [int(x) for x in rng.integers(0, vocab, size=n)]} for k, n in enumerate((37, 64, 21))]
    tmp = Path(tempfile.mkdtemp(prefix="ref_run_test_"))
    tok = tmp / "tokens.json"
    tok.write_text(json.dumps({"segments": segs}), encoding="utf-8")

    for dtype, mode in (("bf16", "hf-bf16"), ("fp32", "fp32")):
        out = tmp / f"lp_{dtype}"
        rc, runs_n = run_tool("logprobs", "--tokens", str(tok), "--out-dir", str(out), "--work", str(tmp / f"w_{dtype}"), "--dtype", dtype,
                              "--budget-seconds", "0.0001", "--record-kv", str(tmp / f"kv_{dtype}.json"))
        check(rc == 0 and runs_n > 1, f"logprobs {dtype}: finished after {runs_n} slices (one per layer), rc {rc}")
        ref = Qwen35MoeStreamingReference(TINY_HF, "cpu", dtype=torch.float32 if dtype == "fp32" else torch.bfloat16)
        worst = 0.0
        shapes_ok = True
        for s in segs:
            ids = s["token_ids"]
            lp_ref = F.log_softmax(ref.forward_logits(ids, mode), dim=-1)[:-1].clamp_min(-1e4).to(torch.float16).float().numpy()
            meta = json.loads((out / f"{s['name']}.meta.json").read_text())
            raw = np.fromfile(out / f"{s['name']}.logprobs.f16", dtype="<f2")
            shapes_ok &= meta["T"] == len(ids) and meta["rows"] == len(ids) - 1 and raw.size == (len(ids) - 1) * meta["V"]
            worst = max(worst, float(np.abs(raw.reshape(-1, meta["V"]).astype(np.float32) - lp_ref).max()))
        check(shapes_ok, f"logprobs {dtype}: file sizes / meta.json match T, rows, V")
        check(worst == 0.0, f"logprobs {dtype}: equal to forward_logits (max |diff| {worst:.3g})")

    # independent K / V measurement: wrap rotary application and v_proj in a plain streaming run
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    ref = Qwen35MoeStreamingReference(TINY_HF, "cpu")
    full = [i for i, t in enumerate(ref.arch.layer_types) if t == "full_attention"]
    seen_k: dict[int, np.ndarray] = {}
    seen_v: dict[int, np.ndarray] = {}
    cur = {"layer": -1}
    orig_rot = M.apply_rotary_pos_emb

    def rot(q, k, cos, sin, *a, **kw):
        q2, k2 = orig_rot(q, k, cos, sin, *a, **kw)
        m = k2.abs().amax(dim=(0, 2, 3)).float().numpy()
        seen_k[cur["layer"]] = np.maximum(seen_k.get(cur["layer"], m), m)
        return q2, k2

    M.apply_rotary_pos_emb = rot
    orig_build = ref._build_layer

    def build(i):
        layer = orig_build(i)
        cur["layer"] = i
        if layer.block_type == "full_attention":
            def vhook(mod, args, out, i=i):
                v = out.view(*out.shape[:-1], -1, ref.arch.head_dim).abs().amax(dim=(0, 1, 3)).float().numpy()
                seen_v[i] = np.maximum(seen_v.get(i, v), v)
            layer.self_attn.v_proj.register_forward_hook(vhook)
        return layer

    ref._build_layer = build
    try:
        for s in segs:
            ref.run_layers(s["token_ids"])
    finally:
        M.apply_rotary_pos_emb = orig_rot

    runs = {}
    for label, args, path in (
        ("kvcalib (resumed)", ("kvcalib", "--tokens", str(tok), "--out", str(tmp / "kv_a.json"), "--work", str(tmp / "w_a"), "--budget-seconds", "0.0001"), tmp / "kv_a.json"),
        ("kvcalib (one go)", ("kvcalib", "--tokens", str(tok), "--out", str(tmp / "kv_b.json"), "--work", str(tmp / "w_b"), "--budget-seconds", "0"), tmp / "kv_b.json"),
    ):
        rc, runs_n = run_tool(*args)
        check(rc == 0 and (runs_n > 1) == ("resumed" in label), f"{label}: finished in {runs_n} run(s), rc {rc}")
        runs[label] = json.loads(path.read_text())
    a, b = runs["kvcalib (resumed)"], runs["kvcalib (one go)"]
    check(sorted(map(int, a)) == full, f"kvcalib JSON has exactly the full-attention layers {full}")
    check(all(len(v["k_amax"]) == ref.arch.kv_heads and len(v["v_amax"]) == ref.arch.kv_heads for v in a.values()),
          f"every layer has kv_heads={ref.arch.kv_heads} k_amax and v_amax entries")
    check(a == b, "resumed and uninterrupted runs give identical JSON")
    errk = max(float(np.abs(np.array(a[str(i)]["k_amax"]) - seen_k[i]).max()) for i in full)
    errv = max(float(np.abs(np.array(a[str(i)]["v_amax"]) - seen_v[i]).max()) for i in full)
    check(errk == 0.0 and errv == 0.0, f"K / V amax equal the independent measurement (max |diff| K {errk:.3g}, V {errv:.3g})")
    kvb = json.loads((tmp / "kv_bf16.json").read_text())
    check(all(kvb[str(i)]["k_amax"] == a[str(i)]["k_amax"] and kvb[str(i)]["v_amax"] == a[str(i)]["v_amax"] for i in full),
          "logprobs --record-kv (bf16) records the same amax as kvcalib")
    check(all(all(x > 0 for x in a[str(i)]["k_amax"] + a[str(i)]["v_amax"]) for i in full), "all amax are positive")

    # simulated w8a16 (--rtn-group) equals the converter's python reference (tools/convert_ref/w8_ref.py), bf16-rounded
    from ref_run import rtn_int8

    rngw = np.random.default_rng(11)
    w = (rngw.standard_normal((48, 256)) * 0.05).astype(np.float32)
    w[3, :] = 0.0
    w[5, 7] = 3.0
    np.save(tmp / "w.npy", w)
    snippet = ("import numpy as np, sys; sys.path.insert(0, sys.argv[1]); import w8_ref as R; w = np.load(sys.argv[2]); out = {}\n"
               "for g in (64, 128):\n"
               "    q, s16, _ = R.quantize_int8_symmetric(w, g); N, K = w.shape\n"
               "    s = s16.view(np.float16).astype(np.float32) * np.float32(2.0 ** -12)\n"
               "    out[str(g)] = ((q.astype(np.int32) - 128).astype(np.float32).reshape(N, K // g, g) * s[:, :, None]).reshape(N, K)\n"
               "np.savez(sys.argv[3], **out)\n")
    ok = subprocess.run([sys.executable, "-c", snippet, str(HERE.parents[1] / "convert_ref"), str(tmp / "w.npy"), str(tmp / "w8.npz")],
                        capture_output=True, text=True, cwd=str(tmp))
    check(ok.returncode == 0, "w8_ref.py ran" + ("" if ok.returncode == 0 else ": " + ok.stderr[-300:]))
    if ok.returncode == 0:
        ref8 = np.load(tmp / "w8.npz")
        for g in (64, 128):
            mine = rtn_int8(torch.from_numpy(w), g, torch.device("cpu")).float().numpy()
            want = torch.from_numpy(ref8[str(g)]).to(torch.bfloat16).float().numpy()
            check(np.array_equal(mine, want), f"rtn_int8 group {g} equals w8_ref.py dequant (bf16-rounded), rel err vs fp32 weights "
                  f"{np.linalg.norm(mine - w) / np.linalg.norm(w):.4f}")
    bad = [w for ok, w in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
