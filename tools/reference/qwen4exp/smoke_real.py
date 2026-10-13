"""Smoke of the streaming reference on the REAL Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ checkpoint (plan M0-5).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\smoke_real.py [--layers 4] [--tokens 16] [--device cpu]

Loads the first N layers (one at a time, AWQ int4 dequantized per linear / per routed expert), runs a short prompt
(the first tokens of the canon english_prose segment) and checks that every layer's residual is finite, then applies the
final mixer and lm_head to prove their weights load too. Prints per-layer statistics, MoE routing coverage, wall time and
the process' peak working set. N < 48 is NOT a model forward: nothing here is a golden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from qwen4exp.common_qwen4exp import DEFAULT_MODEL_DIR, DEFAULT_SIDECAR, KL_CORPUS_DIR  # noqa: E402
from qwen4exp.ref import Qwen4ExpStreamingReference, ReferenceHooks  # noqa: E402


def peak_working_set_gib() -> float | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 2**30
    except ImportError:
        return None


class Stats(ReferenceHooks):
    def __init__(self):
        self.rows = []
        self.routing = {}

    def layer_output(self, layer, residual):
        x = residual.float()
        self.rows.append((layer, bool(torch.isfinite(x).all()), float(x.std()), float(x.abs().max())))

    def moe_routing(self, layer, topk_ids, counts):
        self.routing[layer] = (int(counts.sum()), int((counts > 0).sum()), int(counts.max()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ngram-sidecar", type=Path, default=DEFAULT_SIDECAR)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--tie-rule", default="recent")
    args = ap.parse_args(argv)

    doc = json.loads((KL_CORPUS_DIR / "tokens_canon.json").read_text(encoding="utf-8"))
    ids = next(s for s in doc["segments"] if s["name"] == "english_prose")["token_ids"][: args.tokens]
    t0 = time.perf_counter()
    ref = Qwen4ExpStreamingReference(args.model_dir, args.device, tie_rule=args.tie_rule, ngram_sidecar=args.ngram_sidecar,
                                     max_layers=args.layers, verbose=True)
    stats = Stats()
    residual = ref.run_layers(ids, stats)
    t_layers = time.perf_counter() - t0
    print(f"{args.layers} layers x {len(ids)} tokens in {t_layers:.1f}s (includes setup)")
    ok = True
    for layer, finite, std, amax in stats.rows:
        total, hit, top = stats.routing[layer]
        print(f"layer {layer:2d} {ref.arch.layer_types[layer]:16s} finite={finite} residual std {std:.4g} absmax {amax:.4g}; "
              f"routed {total} (= {len(ids)} x {ref.arch.top_k}: {total == len(ids) * ref.arch.top_k}), "
              f"{hit} experts hit, busiest {top}")
        ok &= finite and total == len(ids) * ref.arch.top_k and std > 0
    t1 = time.perf_counter()
    mixed = ref.mix(residual)
    logits = ref.logits_rows(mixed[0, -1:], "fp32")[0]
    ok &= bool(torch.isfinite(mixed.float()).all()) and bool(torch.isfinite(logits).all())
    print(f"mixer + lm_head on the last position: finite={bool(torch.isfinite(logits).all())}, logits std {float(logits.std()):.4g}, "
          f"top token {int(logits.argmax())} ({time.perf_counter() - t1:.1f}s)")
    print(f"total {time.perf_counter() - t0:.1f}s, peak working set {peak_working_set_gib():.2f} GiB, device {ref.device}")
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
