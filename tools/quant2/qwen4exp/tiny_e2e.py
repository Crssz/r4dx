"""tools/quant2/qwen4exp/tiny_e2e.py -- the quantization chain end to end on the random-weight tiny-q4exp (plan 4.11 item 2, M1-10..14).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\quant2\\qwen4exp\\tiny_e2e.py [--device cpu|cuda] [--root E:\\models\\r4dx\\q4exp-tiny]
        [--ks 3,4,5] [--tokens 20000] [--skip-existing]

(also driven by `tests\\reference\\test_q4exp_quant.py --e2e`). The steps, each a separate process like the real pipeline's steps:

  1. a random corpus (corpus_q4.py random, 20,000 tokens, EOS-separated documents);
  2. Hessians of all 4 layers (hessian_q4.py capture, reference venv) + `check` (symmetric / finite / PSD sample / rows >= K);
  3. the oracle per layer at every K of --ks (oracle_q4.py quantize, the rocm10 venv) -> oracle-k3 / oracle-k4 / oracle-k5, manifests;
  4. `verify` (sha256, shapes, two tensors per chunk reconstructed from the stored bits); the Hessians are NOT deleted here;
  5. the per-tensor proxy ordering K3 > K4 > K5 (and the relative weight error);
  6. a SHORT corpus (~4,100 tokens: ~2,560 routed rows for the median expert, the gate_up threshold): layers 0-1 Hessians, the fallback
     rule must trigger on some experts and spare others, their files diagonal; oracle K4 on layer 1 records the flags;
  7. the allocation (alloc_q4.py) under a scaled-down cap: both cards within the cap, one KB per (layer, expert class), complete mix manifest;
  8. r4dx-convert --trellis-from <mix> --ngram-sidecar ... when a converter that knows qwen4_exp is built.

Returns a dict of numbers (times per step, bytes, fallback counts, proxy ratios, converter result).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "reference"))

MODELS = Path(r"E:\models")
TINY = MODELS / "r4dx" / "tiny-q4exp"
REF_PY = Path(r"D:\venvs\r4dx-qwen4-ref\Scripts\python.exe")
ORACLE_PY = MODELS / "r4dx" / "huihui" / "venv-rocm10" / "Scripts" / "python.exe"
LOCK = REPO / "tools" / "with_lock.ps1"
HESS = REPO / "tools" / "reference" / "qwen4exp" / "hessian_q4.py"
ORACLE = REPO / "tools" / "quant2" / "qwen4exp" / "oracle_q4.py"
ALLOC = REPO / "tools" / "reference" / "qwen4exp" / "alloc_q4.py"
CORPUS = REPO / "tools" / "quant2" / "qwen4exp" / "corpus_q4.py"


def sh(args: list, gpu: bool = False, device: str = "cpu", log: Path | None = None, ok=(0,)) -> tuple[int, float]:
    """Run one step; GPU steps go through with_lock.ps1 (HIP_VISIBLE_DEVICES=1 set inside the lock)."""
    args = [str(a) for a in args]
    env_extra = {"HIP_VISIBLE_DEVICES": "-1"} if not (gpu and device == "cuda") else {}
    t0 = time.perf_counter()
    if gpu and device == "cuda":
        inner = "set HIP_VISIBLE_DEVICES=1&& " + subprocess.list2cmdline([a.replace("\\", "/") for a in args])
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LOCK), "-Name", "gpu", "-Run", inner]
    else:
        cmd = args
    import os

    env = dict(os.environ, PYTHONIOENCODING="utf-8", **env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\nexit {r.returncode}\n")
    if r.returncode not in ok:
        raise RuntimeError(f"step failed (exit {r.returncode}): {' '.join(args[:6])} ...\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.returncode, dt


def run_loop(args: list, gpu: bool, device: str, log: Path) -> float:
    """A time-sliced step: exit 75 = run again."""
    total = 0.0
    while True:
        rc, dt = sh(args, gpu, device, log, ok=(0, 75))
        total += dt
        if rc == 0:
            return total


def run(device: str = "cpu", check=None, root: Path | None = None, ks=(3, 4, 5), n_tokens: int = 20000,
        short_tokens: int | None = None, chunk_experts: int = 16, skip_existing: bool = True) -> dict:
    import numpy as np

    if check is None:
        def check(cond, what):  # noqa: ANN001
            print(("ok   " if cond else "FAIL ") + what)
            if not cond:
                raise SystemExit(1)

    from qwen4exp import alloc_q4 as A
    from qwen4exp import hessian_q4 as HQ

    root = Path(root) if root else MODELS / "r4dx" / "q4exp-tiny"
    logs = root / "logs"
    res: dict = {"device": device, "seconds": {}}
    tokens = root / "tokens_tiny.json"
    hf, sidecar = TINY / "hf", TINY / "sidecar.r4dx-ngram"
    if not (hf / "config.json").exists():
        raise SystemExit(f"no tiny model at {hf}: tools/reference/qwen4exp/tiny.py build")
    from qwen4exp.arch import Qwen4ExpArch

    arch = Qwen4ExpArch.from_config(hf / "config.json")

    # 1. corpus
    _, res["seconds"]["corpus"] = sh([REF_PY, CORPUS, "random", "--tokens", n_tokens, "--out", tokens], log=logs / "e2e.log")

    # 2. Hessians
    t = 0.0
    for L in range(arch.n_layers):
        if skip_existing and (HQ.layer_dir(root, L) / "DONE.json").exists():
            continue
        t += run_loop([REF_PY, HESS, "capture", "--layer", L, "--root", root, "--tokens", tokens, "--model-dir", hf,
                       "--ngram-sidecar", sidecar, "--device", device, "--budget-seconds", 0, *([] if skip_existing else ["--force"])],
                      True, device, logs / "hessian.log")
    res["seconds"]["hessian"] = t
    fb_total = 0
    for L in range(arch.n_layers):
        d = HQ.layer_dir(root, L)
        sh([REF_PY, HESS, "check", "--layer", L, "--root", root, "--eig-sample", 3], log=logs / "check.log")
        cap = json.loads((d / "capture.json").read_text())
        routing = json.loads((d / "routing_counts.json").read_text())
        fb_total += sum(len(v) for v in routing["fallback"].values())
        check(cap["shared_input_gate"]["equal"] and cap["shared_input_gate"]["checked"] > 0,
              f"L{L}: companion linears read their representative's input ({cap['shared_input_gate']['checked']} comparisons)")
        check(routing["min_routed"] >= 2560, f"L{L}: 20k-token corpus: every expert has >= 2560 routed rows (min {routing['min_routed']})")
        check(len(routing["tokens_routed"]) == arch.n_experts and sum(routing["tokens_routed"]) == routing["pairs"],
              f"L{L}: routing_counts.json sums to T * top_k")
        res.setdefault("layers", {})[L] = {"tokens_routed_min": routing["min_routed"], "median": routing["median_routed"],
                                           "asymmetry": cap["asymmetry"], "peak_rss_gib": cap["peak_rss_gib"]}

    # 3. oracles
    dirs = {k: root / f"oracle-k{k}" for k in ks}
    t = {k: 0.0 for k in ks}
    for k in ks:
        for L in range(arch.n_layers):
            t[k] += run_loop([ORACLE_PY, ORACLE, "quantize", "--layer", L, "--K", k, "--out-dir", dirs[k], "--hessian-root", root,
                              "--model-dir", hf, "--device", device, "--chunk-experts", chunk_experts, "--budget-seconds", 0],
                             True, device, logs / f"oracle-k{k}.log")
        sh([ORACLE_PY, ORACLE, "manifest", "--out-dir", dirs[k], "--K", k, "--model-dir", hf, "--chunk-experts", chunk_experts],
           log=logs / "manifest.log")
    res["seconds"]["oracle"] = t

    # 4. verify (keep the Hessians)
    for L in range(arch.n_layers):
        sh([ORACLE_PY, ORACLE, "verify", "--layer", L, "--model-dir", hf, "--chunk-experts", chunk_experts,
            *sum([["--out-dir", f"{dirs[k]}@{k}"] for k in ks], [])], log=logs / "verify.log")
    check(True, "oracle verify: sha256, shapes and two reconstructed tensors per chunk, 4 layers x every K")

    # 5. manifests and the proxy ordering
    mans = {k: json.loads((dirs[k] / "weights_override.json").read_text()) for k in ks}
    n_expected = sum(len(arch.quantized_linears(i)) for i in range(arch.n_layers))
    for k in ks:
        m = mans[k]
        check(m["complete"] and m["missing_count"] == 0 and len(m["tensors"]) == n_expected,
              f"K{k} manifest complete: {len(m['tensors'])} / {n_expected} tensors, {len(m['layers_done'])} layers")
        check(m["encoding"] == "trellis-exl3" and m["format"] == "r4dx-weights-override" and m["hessian_basis"] == "matched"
              and m["codebook"] == "mul1" and len(m["config_sha256"]) == 64, f"K{k} manifest header fields")
        bpw = [r["bits"]["bpw_trellis"] for r in m["tensors"].values()]
        check(all(abs(b - k) < 1e-9 for b in bpw), f"K{k}: trellis bits per weight exactly {k}")
    order_ok = {"proxy": 0, "proxy_abs": 0, "rel_weight_err": 0}
    names = list(mans[ks[0]]["tensors"])
    ratios = []
    for n in names:
        for key in order_ok:
            v = [mans[k]["tensors"][n][key] for k in ks]
            if all(v[i] > v[i + 1] for i in range(len(v) - 1)):
                order_ok[key] += 1
        ratios.append(mans[ks[0]]["tensors"][n]["proxy"] / mans[ks[1]]["tensors"][n]["proxy"])
    for key, c in order_ok.items():
        check(c == len(names), f"{key}: K{' > K'.join(str(k) for k in ks)} strictly ordered on {c} / {len(names)} tensors")
    res["proxy_ratio_k3_over_k4"] = {"min": float(np.min(ratios)), "median": float(np.median(ratios)), "max": float(np.max(ratios))}
    res["proxy_ratio_k3_over_k4_by_class"] = {}
    for cls in ("p", "gu", "down"):
        rr = [mans[ks[0]]["tensors"][n]["proxy"] / mans[ks[1]]["tensors"][n]["proxy"] for n in names
              if mans[ks[0]]["tensors"][n]["class"] == cls]
        res["proxy_ratio_k3_over_k4_by_class"][cls] = float(np.median(rr))
    res["fallback_experts_20k"] = fb_total
    # the output-referred probe errors: present on every expert record, ordered K3 > K4 > K5 per expert group, and the
    # per-class numbers the allocator ranks by (loss) next to the local proxy's
    groups = {}
    for n, r in mans[ks[0]]["tensors"].items():
        if r["class"] != "p":
            groups[(r["layer"], r["tap"])] = n
    probe_ok = all("probe" in mans[k]["tensors"][n] and mans[k]["tensors"][n]["probe"]["rows"] > 0 for k in ks for n in groups.values())
    check(probe_ok, f"every expert group of every K carries probe errors on >= 1 row ({len(groups)} groups)")
    pe = {k: np.array([mans[k]["tensors"][n]["probe"]["out_err"] for n in groups.values()]) for k in ks}
    check(all(np.all(pe[ks[i]] > pe[ks[i + 1]]) for i in range(len(ks) - 1)),
          f"probe out_err strictly decreasing K{' > K'.join(str(k) for k in ks)} on every expert group")
    rel = {k: np.array([mans[k]["tensors"][n]["probe"]["out_err"] / mans[k]["tensors"][n]["probe"]["out_energy"] for n in groups.values()])
           for k in ks}
    res["probe_rel_out_err_median"] = {f"K{k}": float(np.median(rel[k])) for k in ks}
    lc = {k: A.class_stats(mans[k], "out") for k in ks}
    ll = {k: A.class_stats(mans[k], "local") for k in ks}
    res["class_loss_k3_over_k4"] = {kd: {"out": float(np.median([lc[ks[0]][key]["loss"] / lc[ks[1]][key]["loss"] for key in lc[ks[0]] if key[1] == kd])),
                                        "local": float(np.median([ll[ks[0]][key]["loss"] / ll[ks[1]][key]["loss"] for key in ll[ks[0]] if key[1] == kd]))}
                                   for kd in A.KINDS}
    res["class_loss_at_k4"] = {kd: {"out": float(np.sum([lc[ks[1]][key]["loss"] for key in lc[ks[1]] if key[1] == kd])),
                                   "local": float(np.sum([ll[ks[1]][key]["loss"] for key in ll[ks[1]] if key[1] == kd]))} for kd in A.KINDS}

    # 6. the fallback rule on a short corpus (natural counts around the threshold)
    if short_tokens is None:  # the corpus length at which the MEDIAN expert has ~2,560 routed rows: some experts fall back, some do not
        rc1 = json.loads((HQ.layer_dir(root, 1) / "routing_counts.json").read_text())
        short_tokens = int(2560 / (rc1["median_routed"] / rc1["tokens"]))
    short_root = Path(str(root) + "-short")
    short_tokens_f = short_root / "tokens_short.json"
    sh([REF_PY, CORPUS, "random", "--tokens", short_tokens, "--seed", 5, "--out", short_tokens_f], log=logs / "e2e.log")
    for L in (0, 1):
        if not (HQ.layer_dir(short_root, L) / "DONE.json").exists():
            run_loop([REF_PY, HESS, "capture", "--layer", L, "--root", short_root, "--tokens", short_tokens_f, "--model-dir", hf,
                      "--ngram-sidecar", sidecar, "--device", device, "--budget-seconds", 0], True, device, logs / "hessian_short.log")
        sh([REF_PY, HESS, "check", "--layer", L, "--root", short_root, "--eig-sample", 2], log=logs / "check.log")
    rc = json.loads((HQ.layer_dir(short_root, 1) / "routing_counts.json").read_text())
    n_fb = len(rc["fallback"]["gate_up"])
    check(0 < n_fb < arch.n_experts,
          f"short corpus: the fallback rule triggers on {n_fb} of {arch.n_experts} experts (gate_up) and spares the others "
          f"(routed {rc['min_routed']}..{rc['max_routed']}, threshold 2560)")
    check(len(rc["fallback"]["down"]) == 0, "short corpus: no expert is below the down threshold 640")
    res["short_corpus"] = {"tokens": short_tokens, "routed_min": rc["min_routed"], "routed_max": rc["max_routed"],
                           "fallback_gate_up": n_fb, "fallback_share": rc["fallback_share"]}
    t = run_loop([ORACLE_PY, ORACLE, "quantize", "--layer", 1, "--K", 4, "--out-dir", short_root / "oracle-k4", "--hessian-root", short_root,
                  "--model-dir", hf, "--device", device, "--chunk-experts", chunk_experts, "--budget-seconds", 0],
                 True, device, logs / "oracle_short.log")
    res["seconds"]["oracle_short_L1_k4"] = t
    sm = json.loads((short_root / "oracle-k4" / "weights_override.json").read_text())
    flags = {n: r["hessian"].get("fallback") for n, r in sm["tensors"].items() if r["class"] == "gu" and n.startswith(arch.layer_prefix(1))}
    check(sum(1 for f in flags.values() if f) == 2 * n_fb and all(np.isfinite(r["proxy"]) for r in sm["tensors"].values()),
          "short-corpus oracle: gate/up records of fallback experts are flagged, every proxy finite")
    fb_proxy = [r["proxy"] for n, r in sm["tensors"].items() if r["class"] == "gu" and r["hessian"].get("fallback")]
    ok_proxy = [r["proxy"] for n, r in sm["tensors"].items() if r["class"] == "gu" and not r["hessian"].get("fallback")]
    res["short_corpus"]["proxy_gu_fallback_median"] = float(np.median(fb_proxy)) if fb_proxy else None
    res["short_corpus"]["proxy_gu_real_median"] = float(np.median(ok_proxy)) if ok_proxy else None

    # 7. allocation under a scaled cap
    m = A.Model(arch)
    floor = dict(A.DEFAULT_FLOOR)
    lo = A.uniform_assignment(arch.n_layers, floor["p"], floor["gu"], floor["down"])
    hi = A.uniform_assignment(arch.n_layers, 4, 4, 4)
    tot_lo, tot_hi = m.total(lo), m.total(hi)
    cap = int(0.5 * (tot_lo + tot_hi) / 2 * 1.12)           # half way between the floor and all-KB4 (per card, 12% split slack)
    mix = root / "mix"
    sh([REF_PY, ALLOC, "allocate", "--out-dir", mix, "--cap-bytes", cap, *sum([["--src", str(dirs[k])] for k in ks], [])], log=logs / "alloc.log")
    am = json.loads((mix / "weights_override.json").read_text())
    ad = json.loads((mix / "allocation.json").read_text())
    check(am["complete"] and am["missing_count"] == 0 and "layers_done" not in am and "allocation" in am, "mix manifest: complete, mix form")
    check(ad["device0_gib"] * A.GIB <= cap and ad["device1_gib"] * A.GIB <= cap,
          f"allocation: both cards within the scaled cap ({ad['device0_gib']:.4f} / {ad['device1_gib']:.4f} of {cap / A.GIB:.4f} GiB), k={ad['k']}")
    per_class: dict = {}
    for n, r in am["tensors"].items():
        layer, kd, _ = A.parse_name(n)
        per_class.setdefault((layer, kd), set()).add(r["K"])
    check(all(len(v) == 1 for v in per_class.values()), "mix: ONE KB per (layer, class): every expert of a class shares it")
    check(len({K for v in per_class.values() for K in v}) > 1, f"mix mixes rates: {am['allocation']['tensors_per_K']}")
    check(all(r["file"].endswith(".safetensors") and Path(r["file"]).is_absolute() for r in am["tensors"].values()), "mix: absolute file paths")
    res["allocation"] = {"cap_gib": cap / A.GIB, "total_gib": ad["total_gib"], "k": ad["k"], "tensors_per_K": am["allocation"]["tensors_per_K"],
                         "promotions": ad["promotions"], "expected_loss": ad["expected_loss"], "floor_gib": tot_lo / A.GIB,
                         "uniform_k4_gib": tot_hi / A.GIB}

    # 8. converter
    exe = None
    for c in ("build/win-hip/src/convert/r4dx-convert.exe", "build/win-hip-dry/src/convert/r4dx-convert.exe"):
        if (REPO / c).exists():
            exe = REPO / c
            break
    res["converter"] = {"exe": str(exe) if exe else None}
    if exe is not None:
        import hashlib

        sha = hashlib.sha256((mix / "weights_override.json").read_bytes()).hexdigest()
        out = root / "tiny-mix.r4dx"
        try:
            _, dt = sh([exe, "--input", hf, "--output", out, "--trellis-from", mix, "--trellis-manifest-sha256", sha,
                        "--trellis-verify", "full", "--ngram-sidecar", sidecar, "--layouts", "w4a16", "--no-bf16",
                        "--lm-head", "w4a16", "--w4a16-group-rule", "^lm_head$=32", "--threads", 8], log=logs / "convert.log")
            res["converter"].update({"ok": True, "seconds": dt, "bytes": out.stat().st_size})
            check(True, f"r4dx-convert --trellis-from <mix> accepts the manifest ({dt:.0f}s, {out.stat().st_size / 2**20:.1f} MiB)")
        except RuntimeError as e:
            res["converter"].update({"ok": False, "error": str(e)[-1200:]})
            check(False, f"r4dx-convert --trellis-from <mix>: {str(e)[-300:]}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--chunk-experts", type=int, default=16)
    ap.add_argument("--fresh", action="store_true", help="redo Hessians even when DONE")
    a = ap.parse_args()
    res = run(a.device, None, a.root, tuple(int(k) for k in a.ks.split(",")), a.tokens, chunk_experts=a.chunk_experts,
              skip_existing=not a.fresh)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
