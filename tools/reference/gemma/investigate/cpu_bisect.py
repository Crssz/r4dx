"""CPU numerical bisection of the Gemma-4 reference noise floor (eager/sdpa x fp32/bf16).

NO GPU: HIP/CUDA visibility is blanked before torch is imported and the script aborts if cuda is visible.

Runs the decoder stack layer by layer in lockstep for 4 configs (fp32-eager = truth, fp32-sdpa, bf16-eager,
bf16-sdpa), each layer streamed once from safetensors (so RAM stays ~ a few GB + the embedding table).
Reports per-layer relative error / max-abs / max|h|, final-logit KL + top-1, and per-position breakdowns.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\cpu_bisect.py \
        --segment english_prose --tokens 256 --layers 48 --out D:\\models\\r4dx\\huihui-gemma\\kl\\bisect_256.json
"""
from __future__ import annotations

import os

# NB: on this ROCm-Windows torch build an EMPTY value is ignored (device_count() == 2); "-1" hides all GPUs.
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import sys
import time
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("ABORT: torch.cuda.is_available() is True; this script must run CPU-only")

from gemma.common_gemma import DEFAULT_MODEL_DIR, KL_CORPUS_DIR  # noqa: E402
from gemma.ref import FULL, SLIDING, GemmaReference, additive_mask  # noqa: E402

CONFIGS = [("fp32-eager", torch.float32, "eager"), ("fp32-sdpa", torch.float32, "sdpa"),
           ("bf16-eager", torch.bfloat16, "eager"), ("bf16-sdpa", torch.bfloat16, "sdpa")]


def rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def kl_top1(lt, lc):
    """KL(truth||cfg) per row (mean) and top-1 agreement, from fp32 softcapped logits [R, V]."""
    lpt = torch.log_softmax(lt, -1)
    lpc = torch.log_softmax(lc, -1)
    kl = (lpt.exp() * (lpt - lpc)).sum(-1)
    top1 = (lt.argmax(-1) == lc.argmax(-1)).float().mean().item()
    return kl.mean().item(), kl.max().item(), top1, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens-file", type=Path, default=KL_CORPUS_DIR / "tokens_gemma.json")
    ap.add_argument("--segment", default="english_prose")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--no-logits", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    doc = json.load(open(a.tokens_file, encoding="utf-8"))
    seg = next(s for s in doc["segments"] if s["name"] == a.segment)
    ids = seg["token_ids"][:a.tokens]
    T = len(ids)
    dev = torch.device("cpu")
    print(f"segment={a.segment} T={T} layers={a.layers} threads={a.threads}", flush=True)

    refs = {}
    for name, dt, attn in CONFIGS:
        r = GemmaReference(a.model_dir, dev, dtype=dt, max_layers=a.layers, resident=False, attn=attn,
                           load_table=False)
        refs[name] = r
    # embedding table: load once (bf16), convert for fp32 on demand per config (rows only for embed, full for logits)
    from gemma.ref import EMBED_NAME
    table_bf16 = refs["fp32-eager"].index.get_tensor(EMBED_NAME).to(torch.bfloat16)
    for name, dt, attn in CONFIGS:
        refs[name].table = table_bf16  # values identical (checkpoint is bf16); logits path upcasts to fp32
        # same embed scale everywhere: HF uses bf16(sqrt(H)) = 62.0, truth must not differ by that
        refs[name].embed_scale = torch.tensor(62.0).to(dt) if dt == torch.bfloat16 else torch.tensor(62.0)

    # embeddings, rope, masks per config
    state = {}
    for name, dt, attn in CONFIGS:
        r = refs[name]
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        h = (table_bf16[ids_t].to(dt) * r.embed_scale).unsqueeze(0)
        pos = torch.arange(T).unsqueeze(0)
        pe = {lt: r.rotary(h, pos, lt) for lt in sorted(set(r.layer_types))}
        masks = {FULL: additive_mask(T, T, 0, None, dev, dt), SLIDING: additive_mask(T, T, 0, r.arch.window, dev, dt)}
        state[name] = dict(h=h, pos=pos, pe=pe, masks=masks, shared=UserDict())

    layer_types = refs["fp32-eager"].layer_types
    per_layer = []
    hs = {n: [] for n, _, _ in CONFIGS}  # fp32 copies [T,H] of per-layer outputs, kept for pos analysis (small)
    t0 = time.time()
    for i in range(a.layers):
        lt = layer_types[i]
        out = {}
        for name, dt, attn in CONFIGS:
            r = refs[name]
            s = state[name]
            layer = r.layer(i)
            with torch.no_grad():
                s["h"] = layer(s["h"], shared_kv_states=s["shared"], position_embeddings=s["pe"][lt],
                               attention_mask=s["masks"][lt], position_ids=s["pos"], past_key_values=None)
            out[name] = s["h"][0].float()
            del layer
        truth = out["fp32-eager"]
        rec = {"layer": i, "type": lt, "max_abs_h_truth": truth.abs().max().item(),
               "max_abs_h_truth_excl_pos0": truth[1:].abs().max().item(),
               "rms_h_truth": truth.pow(2).mean().sqrt().item(),
               "bf16_roundtrip_rel": rel(truth.to(torch.bfloat16).float(), truth)}
        for name in ("fp32-sdpa", "bf16-eager", "bf16-sdpa"):
            d = out[name] - truth
            per_pos = d.norm(dim=-1) / truth.norm(dim=-1)
            rec[name] = {"rel": rel(out[name], truth), "max_abs": d.abs().max().item(),
                         "rel_excl_pos0": rel(out[name][1:], truth[1:]),
                         "worst_pos": int(per_pos.argmax()), "worst_pos_rel": per_pos.max().item(),
                         "median_pos_rel": per_pos.median().item()}
        d = out["bf16-sdpa"] - out["bf16-eager"]
        rec["bf16-sdpa_vs_bf16-eager"] = {"rel": (d.norm() / out["bf16-eager"].norm()).item(),
                                         "max_abs": d.abs().max().item()}
        per_layer.append(rec)
        for n in hs:
            hs[n].append(out[n])
        print(f"L{i:02d} {lt[:4]} max|h|={rec['max_abs_h_truth']:.1f} (ex0 {rec['max_abs_h_truth_excl_pos0']:.1f}) "
              f"f32sdpa={rec['fp32-sdpa']['rel']:.2e} b16e={rec['bf16-eager']['rel']:.2e} "
              f"b16s={rec['bf16-sdpa']['rel']:.2e} b16e/b16s={rec['bf16-sdpa_vs_bf16-eager']['rel']:.2e} "
              f"t={time.time() - t0:.0f}s", flush=True)

    result = {"segment": a.segment, "T": T, "layers": a.layers, "per_layer": per_layer}

    if not a.no_logits:
        # final logits: final norm + fp32 lm_head + softcap (reference logits_mode fp32), rows 0..T-2
        logits = {}
        for name, dt, attn in CONFIGS:
            r = refs[name]
            hfin = r.final_norm(state[name]["h"][0])
            logits[name] = r.logits_rows(hfin[:T - 1], "fp32", 32768)
            print(f"logits {name} done t={time.time() - t0:.0f}s", flush=True)
        tl = logits["fp32-eager"]
        fin = {}
        for name in ("fp32-sdpa", "bf16-eager", "bf16-sdpa"):
            mk, xk, t1, klrow = kl_top1(tl, logits[name])
            worst = torch.topk(klrow, 5)
            fin[name] = {"mean_kl": mk, "max_row_kl": xk, "top1": t1,
                         "worst_rows": [(int(i), float(v)) for v, i in zip(worst.values, worst.indices)],
                         "kl_row0": float(klrow[0]), "mean_kl_excl_row0": float(klrow[1:].mean()),
                         "max_abs_logit_diff": (logits[name] - tl).abs().max().item()}
        mk, xk, t1, klrow = kl_top1(logits["bf16-eager"], logits["bf16-sdpa"])
        fin["KL(bf16-eager||bf16-sdpa)"] = {"mean_kl": mk, "max_row_kl": xk, "top1": t1,
                                            "mean_kl_excl_row0": float(klrow[1:].mean())}
        # entropy / tied-ness of truth rows for context
        lp = torch.log_softmax(tl, -1)
        ent = -(lp.exp() * lp).sum(-1)
        fin["truth_mean_entropy_nats"] = ent.mean().item()
        top2 = torch.topk(tl, 2, dim=-1).values
        fin["truth_mean_top1_top2_logit_gap"] = (top2[:, 0] - top2[:, 1]).mean().item()
        fin["truth_frac_gap_lt_0.1"] = ((top2[:, 0] - top2[:, 1]) < 0.1).float().mean().item()
        result["final"] = fin

    args_out = a.out
    args_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args_out, "w"), indent=1)
    print("DONE", args_out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
