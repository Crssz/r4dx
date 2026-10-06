"""Build a whole-layer K4/K5 mix manifest from the sens_sweep results (docs/gemma4-plan.md item 12, re-rank).

  python tools/gemma/sens_mix.py --mode all|worst --bpw 4.5 --out <models root>\\r4dx\\huihui-gemma\\trellis\\q\\mix4.5m-sensA

all:   promote layers in order of their measured chat-ALL KL gain.
worst: greedy -- each step promotes the layer that most lowers the WORST group's predicted KL (additive model from the
       per-layer gains), until the bit budget is used.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_sweep import HG, Q  # noqa: E402

GROUPS = ("chat-code", "chat-english", "chat-thai")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("all", "worst"), required=True)
    ap.add_argument("--bpw", type=float, default=4.5)
    ap.add_argument("--sens", default=str(HG / "sens"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = [json.loads(l) for f in glob.glob(str(Path(a.sens) / "results_dev*.jsonl")) for l in open(f) if l.strip()]
    base = next(r for r in rows if r["layer"] == -1)["groups"]
    per = {r["layer"]: r for r in rows if r["layer"] >= 0}
    k4 = json.loads((Q / "K4m" / "weights_override.json").read_text(encoding="utf-8"))
    k5 = json.loads((Q / "K5m" / "weights_override.json").read_text(encoding="utf-8"))
    numel = sum(e["k"] * e["n"] for e in k4["tensors"].values())
    bits4 = sum(e["bits"]["total"] for e in k4["tensors"].values())
    budget = a.bpw * numel - bits4
    chosen, used = [], 0.0
    if a.mode == "all":
        order = sorted(per, key=lambda L: base["chat-ALL"] - per[L]["groups"]["chat-ALL"], reverse=True)
        for L in order:
            if used + per[L]["bits_extra"] <= budget:
                chosen.append(L)
                used += per[L]["bits_extra"]
    else:
        pred = {g: base[g] for g in GROUPS}
        left = set(per)
        while True:
            best, best_val = None, None
            for L in left:
                if used + per[L]["bits_extra"] > budget:
                    continue
                val = max(pred[g] - (base[g] - per[L]["groups"][g]) for g in GROUPS)
                if best_val is None or val < best_val:
                    best, best_val = L, val
            if best is None:
                break
            for g in GROUPS:
                pred[g] -= base[g] - per[best]["groups"][g]
            chosen.append(best)
            used += per[best]["bits_extra"]
            left.remove(best)
    chosen.sort()
    mixbase = json.loads((Q / "mix4.5m" / "weights_override.json").read_text(encoding="utf-8"))
    n5 = 0
    for name in mixbase["tensors"]:
        L = int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else -1
        src, kd = (k5, "K5m") if L in chosen else (k4, "K4m")
        e = dict(src["tensors"][name])
        if not Path(e["file"]).is_absolute():
            e["file"] = str(Q / kd / e["file"])
        mixbase["tensors"][name] = e
        n5 += L in chosen
    total = sum(e["bits"]["total"] for e in mixbase["tensors"].values())
    mixbase["bpw_target"] = a.bpw
    mixbase["allocation"]["rule"] = f"tools/gemma/sens_mix.py --mode {a.mode}: whole layers at K5 by measured KL gain"
    mixbase["allocation"]["ranking"] = f"sens-{a.mode}"
    mixbase["allocation"]["tensors_per_K"] = {"4.0": len(mixbase["tensors"]) - n5, "5.0": n5}
    mixbase["allocation"]["k5_layers"] = chosen
    mixbase["summary"]["bits"] = total
    mixbase["summary"]["bpw"] = total / numel
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "weights_override.json").write_text(json.dumps(mixbase, indent=1), encoding="utf-8")
    print(f"[sens_mix] {a.mode}: {len(chosen)} layers at K5 {chosen} -> {total / numel:.4f} bpw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
