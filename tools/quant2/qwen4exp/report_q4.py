"""tools/quant2/qwen4exp/report_q4.py -- what a (partial) run measured, and what the full 48-layer / 1M-token run would cost.

    python tools\\quant2\\qwen4exp\\report_q4.py --root E:\\models\\r4dx\\q4exp-m100k --layers 0,3 --corpus-tokens 1003972 [--json out.json]

Reads, per layer: hessian\\L{ii}\\capture.json + routing_counts.json (Hessian pass: seconds, peak RAM / VRAM, bytes, fallback),
oracle-k*\\L{ii}.*.json chunk records (seconds per class, peak memory, bytes of the .safetensors, per-class proxy / relative
error). The full-run extrapolation: Hessian forward seconds scale with tokens (x corpus_tokens / measured tokens) plus the
token-independent part (state I/O, finalize); oracle seconds do not depend on the corpus; the fallback share at the full corpus
is predicted from the measured routing counts scaled by the token ratio (an expert's share of the routing is what it is).
Pure python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "reference"))

GIB = 2**30


def layer_hessian(root: Path, L: int) -> dict | None:
    d = root / "hessian" / f"L{L:02d}"
    if not (d / "capture.json").exists():
        return None
    cap = json.loads((d / "capture.json").read_text())
    rc = json.loads((d / "routing_counts.json").read_text())
    return {"cap": cap, "routing": rc}


def layer_oracle(root: Path, K: int, L: int) -> dict | None:
    d = root / f"oracle-k{K}"
    chunks = sorted(d.glob(f"L{L:02d}.*.json"))
    if not chunks:
        return None
    secs, by_cls, peak_rss, peak_vram, nbytes = 0.0, {"p": 0.0, "gu": 0.0, "down": 0.0}, 0.0, 0.0, 0
    cls_stat: dict = {}
    tiles = 0
    for c in chunks:
        doc = json.loads(c.read_text())
        secs += doc["seconds"]
        for k, v in doc["seconds_by_class"].items():
            by_cls[k] += v
        peak_rss = max(peak_rss, doc.get("peak_rss_gib") or 0)
        peak_vram = max(peak_vram, doc.get("peak_vram_gib") or 0)
        nbytes += c.with_suffix(".safetensors").stat().st_size
        for r in doc["tensors"].values():
            key = r["class"] + (":fallback" if r.get("hessian", {}).get("fallback") else "")
            s = cls_stat.setdefault(key, {"n": 0, "numel": 0, "proxy": 0.0, "rel2": 0.0, "proxy_abs": 0.0, "fallback": 0,
                                                 "routed_w": 0.0, "bits": 0})
            s["n"] += 1
            s["numel"] += r["numel"]
            s["proxy"] += r["proxy"] * r["numel"]
            s["rel2"] += r["rel_weight_err"] ** 2 * r["numel"]
            s["proxy_abs"] += r["proxy_abs"]
            s["bits"] += r["bits"]["total"]
            h = r.get("hessian", {})
            if h.get("fallback"):
                s["fallback"] += 1
            if r["class"] != "p":
                s["routed_w"] += r["proxy_abs"] * (h.get("routed", 0) / h["tokens"]) if h.get("tokens") else 0.0
    return {"seconds": secs, "seconds_by_class": by_cls, "peak_rss_gib": peak_rss, "peak_vram_gib": peak_vram, "bytes": nbytes,
            "chunks": len(chunks), "classes": cls_stat}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--ks", default="3,4")
    ap.add_argument("--corpus-tokens", type=int, default=1_003_972)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    Ks = [int(k) for k in a.ks.split(",")]
    out: dict = {"layers": {}}
    for L in layers:
        h = layer_hessian(a.root, L)
        row: dict = {}
        if h:
            cap, rc = h["cap"], h["routing"]
            t = cap["timing"]
            scale = a.corpus_tokens / rc["tokens"]
            n = np.array(rc["tokens_routed"], dtype=np.float64)
            row["hessian"] = {
                "tokens": rc["tokens"], "forward_s": t["seconds_forward_total"], "state_io_s": t["seconds_state_io"],
                "finalize_s": t["finalize_s"], "slices": t["slices"], "files": cap["files"], "hessian_gib": cap["bytes"] / GIB,
                "peak_rss_gib": cap["peak_rss_gib"], "peak_vram_gib": cap["peak_vram_gib"],
                "fallback_gate_up": len(rc["fallback"]["gate_up"]), "fallback_down": len(rc["fallback"]["down"]),
                "routed_min": rc["min_routed"], "routed_median": rc["median_routed"], "routed_max": rc["max_routed"],
                "never_routed": len(rc["experts_never_routed"]),
                "predicted_fallback_at_full": {"gate_up": float((n * scale < 2560).mean()), "down": float((n * scale < 640).mean())},
                "predicted_fallback_at_full_count": {"gate_up": int((n * scale < 2560).sum()), "down": int((n * scale < 640).sum())},
                "fixed_s": t["seconds_state_io"] + t["finalize_s"],
                "full_run_s": t["seconds_forward_total"] * scale + t["seconds_state_io"] + t["finalize_s"]}
        for K in Ks:
            o = layer_oracle(a.root, K, L)
            if o:
                row[f"oracle_k{K}"] = o
        out["layers"][L] = row
    # table
    print(f"corpus tokens for the extrapolation: {a.corpus_tokens}")
    for L, row in out["layers"].items():
        h = row.get("hessian")
        if h:
            print(f"\nL{L:02d} Hessian pass: {h['tokens']} tokens, forward {h['forward_s']:.0f}s + state io {h['state_io_s']:.0f}s + finalize "
                  f"{h['finalize_s']:.0f}s ({h['slices']} slice(s)); {h['files']} files {h['hessian_gib']:.2f} GiB; peak RSS {h['peak_rss_gib']:.1f} GiB, "
                  f"VRAM {h['peak_vram_gib']}")
            print(f"     routed per expert: min {h['routed_min']} median {h['routed_median']:.0f} max {h['routed_max']} (never routed {h['never_routed']}); "
                  f"fallback at this corpus: gate_up {h['fallback_gate_up']}/512, down {h['fallback_down']}/512; predicted at "
                  f"{a.corpus_tokens} tokens: gate_up {h['predicted_fallback_at_full_count']['gate_up']}/512 "
                  f"({h['predicted_fallback_at_full']['gate_up']:.1%}), down {h['predicted_fallback_at_full_count']['down']}/512 "
                  f"({h['predicted_fallback_at_full']['down']:.1%})")
        for K in Ks:
            o = row.get(f"oracle_k{K}")
            if o:
                bc = o["seconds_by_class"]
                print(f"L{L:02d} oracle K{K}: {o['seconds']:.0f}s (p {bc['p']:.0f}s, gu {bc['gu']:.0f}s, down {bc['down']:.0f}s), "
                      f"{o['bytes'] / GIB:.3f} GiB written, peak RSS {o['peak_rss_gib']:.1f} GiB, VRAM {o['peak_vram_gib']:.1f} GiB")
        if all(f"oracle_k{K}" in row for K in Ks[:2]) and len(Ks) >= 2:
            k3, k4 = row[f"oracle_k{Ks[0]}"], row[f"oracle_k{Ks[1]}"]
            for cls in ("p", "gu", "gu:fallback", "down", "down:fallback"):
                if cls in k3["classes"] and cls in k4["classes"]:
                    a3, a4 = k3["classes"][cls], k4["classes"][cls]
                    print(f"     {cls:>13}: {a3['n']:>4} tensors, proxy K{Ks[0]} {a3['proxy'] / a3['numel']:.5f} -> K{Ks[1]} {a4['proxy'] / a4['numel']:.5f} "
                          f"(x{a3['proxy'] / a4['proxy']:.2f});  rel weight err {np.sqrt(a3['rel2'] / a3['numel']):.5f} -> "
                          f"{np.sqrt(a4['rel2'] / a4['numel']):.5f};  proxy_abs {a3['proxy_abs']:.4g} -> {a4['proxy_abs']:.4g}; "
                          f"routed-weighted {a3['routed_w']:.4g} -> {a4['routed_w']:.4g}; fallback tensors {a3['fallback']}/{a3['n']}")
    out["extrapolation_note"] = ("48 layers = 36 GDN + 12 QSA; see run_summary in the final report")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
