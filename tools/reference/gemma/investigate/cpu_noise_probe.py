"""CPU-only probe of the Gemma bf16 reference self-disagreement (eager vs sdpa).

Streams the 48 decoder layers one at a time from safetensors (no GPU, ~2 GB RAM for the layer + 1.9 GB
embedding table) over a few short segments and runs, in lockstep, for every segment
    A  bf16 weights/activations, attn=eager   (the primary pass of full_logits_gemma.py)
    B  bf16 weights/activations, attn=sdpa    (the noise-floor pass)
    C  fp32 weights/activations (exact upcast of the bf16 weights), attn=eager  (the "truth")
It records per-layer relative hidden-state error of A and B against C, the max |attention score| seen in
C (scaling is 1.0, so scores are unbounded by 1/sqrt(d)), and finally next-token NLL and KL/top-1 between
the three on the first N tokens.  Aborts if a GPU is visible.

    $env:HIP_VISIBLE_DEVICES=''; $env:CUDA_VISIBLE_DEVICES=''
    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\investigate\\cpu_noise_probe.py --n 160
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import UserDict
from pathlib import Path

# NB: an empty value is ignored by the ROCm runtime on Windows (GPU stays visible); "-1" hides every device.
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is True: refusing to run (CPU only)")

import transformers.models.gemma4_unified.modeling_gemma4_unified as modeling  # noqa: E402
from gemma.arch import FULL, SLIDING  # noqa: E402
from gemma.common_gemma import DEFAULT_MODEL_DIR, KL_CORPUS_DIR  # noqa: E402
from gemma.ref import GemmaReference, additive_mask  # noqa: E402

score_max: dict[int, float] = {}
_orig_eager = modeling.eager_attention_forward


def _spy_eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
    if query.dtype == torch.float32:
        k = modeling.repeat_kv(key, module.num_key_value_groups)
        s = torch.matmul(query, k.transpose(2, 3)) * (1.0 if scaling is None else scaling)
        if attention_mask is not None:
            s = s + attention_mask
        v = s.masked_fill(attention_mask.expand_as(s) < -1e30, 0.0).abs().max().item() if attention_mask is not None else s.abs().max().item()
        score_max[module.layer_idx] = max(score_max.get(module.layer_idx, 0.0), v)
    return _orig_eager(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                       softcap=softcap, **kw)


modeling.eager_attention_forward = _spy_eager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--segments", default="english_prose,thai_prose,python_source")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    doc = json.load(open(KL_CORPUS_DIR / "tokens_gemma.json", encoding="utf-8"))
    segs = {s["name"]: s["token_ids"][: a.n] for s in doc["segments"] if s["name"] in a.segments.split(",")}
    dev = torch.device("cpu")
    ref = GemmaReference(DEFAULT_MODEL_DIR, dev, resident=False, attn="eager", gemm_guard=None)
    cfg = ref.text_config
    variants = ["A_eager_bf16", "B_sdpa_bf16", "C_eager_fp32"]
    st = {}
    pe = {}
    masks = {}
    with torch.no_grad():
        for name, ids in segs.items():
            T = len(ids)
            h0 = ref.embed(ids).unsqueeze(0)
            pos = torch.arange(T).unsqueeze(0)
            for v in variants:
                dt = torch.float32 if v.startswith("C") else torch.bfloat16
                st[(name, v)] = h0.to(dt) if dt == torch.bfloat16 else h0.float()
                pe[(name, v)] = {lt: ref.rotary(st[(name, v)], pos, lt) for lt in (FULL, SLIDING)}
                masks[(name, v)] = {FULL: additive_mask(T, T, 0, None, dev, dt),
                                    SLIDING: additive_mask(T, T, 0, ref.arch.window, dev, dt)}
        rel = {(n, v): [] for n in segs for v in variants[:2]}
        hmax = {n: [] for n in segs}
        shared = {}
        for i in range(ref.n_layers):
            t0 = time.perf_counter()
            layer = ref.build_layer(i)
            lt = ref.layer_types[i]
            for v in variants[:2]:
                cfg._attn_implementation = "eager" if v.startswith("A") else "sdpa"
                for name in segs:
                    sk = UserDict()
                    st[(name, v)] = layer(st[(name, v)], shared_kv_states=sk, position_embeddings=pe[(name, v)][lt],
                                          attention_mask=masks[(name, v)][lt], position_ids=None, past_key_values=None)
            layer = layer.float()
            cfg._attn_implementation = "eager"
            for name in segs:
                sk = UserDict()
                st[(name, "C_eager_fp32")] = layer(st[(name, "C_eager_fp32")], shared_kv_states=sk,
                                                   position_embeddings=pe[(name, "C_eager_fp32")][lt],
                                                   attention_mask=masks[(name, "C_eager_fp32")][lt],
                                                   position_ids=None, past_key_values=None)
                c = st[(name, "C_eager_fp32")]
                hmax[name].append(float(c.abs().max()))
                for v in variants[:2]:
                    d = st[(name, v)].float() - c
                    rel[(name, v)].append(float(d.norm() / c.norm()))
            print(f"L{i:02d} {lt[:4]} {time.perf_counter() - t0:.1f}s max|score|={score_max.get(i, 0):.1f} "
                  + " ".join(f"{n[:4]}:A={rel[(n, variants[0])][-1]:.4f},B={rel[(n, variants[1])][-1]:.4f}"
                             for n in segs), flush=True)
            del layer

        # logits (fp32, softcapped) for rows 0..T-2
        cap = ref.arch.softcap
        res = {}
        for name, ids in segs.items():
            T = len(ids)
            lps = {}
            for v in variants:
                h = None
                if not v.startswith("C"):
                    h = ref.final_norm(st[(name, v)][0])
                else:
                    x = st[(name, v)][0]
                    x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + ref.arch.eps, -0.5)
                    h = x * ref.final_norm_w.float()
                lp = []
                for s in range(0, T - 1, 64):
                    e = min(s + 64, T - 1)
                    lg = torch.cat([torch.tanh((h[s:e].float() @ ref.table[c0:c0 + 32768].float().T) / cap) * cap
                                    for c0 in range(0, ref.vocab_size, 32768)], dim=-1)
                    lp.append(torch.log_softmax(lg, -1))
                lps[v] = torch.cat(lp)
            tgt = torch.tensor(ids[1:])
            r = {"T": T}
            for v in variants:
                r[f"ppl_{v}"] = float(torch.exp(-lps[v][torch.arange(T - 1), tgt].mean()))

            def kl(p, q):
                return float((p.exp() * (p - q)).sum(-1).mean())

            def top1(p, q):
                return float((p.argmax(-1) == q.argmax(-1)).float().mean())
            for x, y in (("A_eager_bf16", "B_sdpa_bf16"), ("C_eager_fp32", "A_eager_bf16"),
                         ("C_eager_fp32", "B_sdpa_bf16")):
                r[f"KL({x[0]}||{y[0]})"] = kl(lps[x], lps[y])
                r[f"top1({x[0]},{y[0]})"] = top1(lps[x], lps[y])
            res[name] = r
            print(name, json.dumps(r), flush=True)
        out = {"segments": res, "layer_rel_err": {f"{k[0]}/{k[1]}": v for k, v in rel.items()},
               "max_abs_hidden_fp32": hmax, "max_abs_score_fp32": score_max}
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
