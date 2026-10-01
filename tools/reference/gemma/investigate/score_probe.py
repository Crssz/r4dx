"""CPU single-layer probe: where does bf16-eager lose precision vs bf16-sdpa? (companion of cpu_bisect.py)

Runs the fp32-eager truth stack up to --upto, and at the probe layers feeds the bf16-rounded truth input to
one decoder layer under several attention variants, comparing with the fp32 layer output (layer-local error).
Variants: eager (stock bf16), sdpa (bf16), eager_s32 (scores in fp32), eager_s32_p32 (scores+probs+PV fp32),
eager_pv32 (scores bf16, probs/PV fp32). Also prints attention-score magnitude stats (scaling = 1.0 with qk-norm).

NO GPU (CUDA/HIP hidden, aborts if visible).
"""
from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import sys
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("ABORT: torch.cuda.is_available() is True")

import transformers.models.gemma4_unified.modeling_gemma4_unified as M  # noqa: E402
from transformers import AttentionInterface  # noqa: E402

from gemma.common_gemma import DEFAULT_MODEL_DIR, KL_CORPUS_DIR  # noqa: E402
from gemma.ref import EMBED_NAME, FULL, SLIDING, GemmaReference, additive_mask  # noqa: E402

STATS = {}


def make_variant(name, scores32, probs32):
    def fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
        k = M.repeat_kv(key, module.num_key_value_groups)
        v = M.repeat_kv(value, module.num_key_value_groups)
        if scores32:
            s = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
        else:
            s = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            s = s + attention_mask
        p = torch.softmax(s, dim=-1, dtype=torch.float32)
        if probs32:
            o = torch.matmul(p, v.float()).to(query.dtype)
        else:
            o = torch.matmul(p.to(query.dtype), v)
        return o.transpose(1, 2).contiguous(), None
    AttentionInterface.register(name, fn)


def stat_eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
    k = M.repeat_kv(key, module.num_key_value_groups)
    s = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        s = s + attention_mask.float()
    fin = s[s > -1e30]
    STATS[module.layer_idx] = {"max_abs_score": fin.abs().max().item(), "mean_abs_score": fin.abs().mean().item(),
                               "frac_abs_gt_32": (fin.abs() > 32).float().mean().item(),
                               "frac_abs_gt_128": (fin.abs() > 128).float().mean().item(),
                               "bf16_ulp_at_max": float(torch.finfo(torch.bfloat16).eps * fin.abs().max().item())}
    p = torch.softmax(s, -1)
    v = M.repeat_kv(value, module.num_key_value_groups)
    return torch.matmul(p, v.float()).to(query.dtype).transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--probe", default="10,17,29")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(16)
    probes = [int(x) for x in a.probe.split(",")]
    upto = max(probes)
    doc = json.load(open(KL_CORPUS_DIR / "tokens_gemma.json", encoding="utf-8"))
    ids = next(s for s in doc["segments"] if s["name"] == "english_prose")["token_ids"][:a.tokens]
    T = len(ids)
    for nm, s32, p32 in (("eager_s32", True, False), ("eager_s32_p32", True, True), ("eager_pv32", False, True)):
        make_variant(nm, s32, p32)
    AttentionInterface.register("stat", stat_eager)

    ref = GemmaReference(DEFAULT_MODEL_DIR, torch.device("cpu"), dtype=torch.float32, resident=False, attn="eager",
                         load_table=False)
    tab = ref.index.get_tensor(EMBED_NAME).to(torch.bfloat16)
    ids_t = torch.as_tensor(ids)
    h = (tab[ids_t].float() * torch.tensor(62.0)).unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    pe32 = {lt: ref.rotary(h, pos, lt) for lt in sorted(set(ref.layer_types))}
    masks32 = {FULL: additive_mask(T, T, 0, None, "cpu", torch.float32),
               SLIDING: additive_mask(T, T, 0, ref.arch.window, "cpu", torch.float32)}
    res = {}
    shared = UserDict()
    for i in range(upto + 1):
        lt = ref.layer_types[i]
        layer = ref.layer(i)
        if i in probes:
            hin = h.clone()
            ref.text_config._attn_implementation = "stat"
            with torch.no_grad():
                layer(hin, shared_kv_states=UserDict(shared), position_embeddings=pe32[lt], attention_mask=masks32[lt],
                      position_ids=pos, past_key_values=None)
            ref.text_config._attn_implementation = "eager"
        with torch.no_grad():
            hout = layer(h, shared_kv_states=shared, position_embeddings=pe32[lt], attention_mask=masks32[lt],
                         position_ids=pos, past_key_values=None)
        if i in probes:
            # bf16 copy of this layer, bf16 inputs
            lay16 = layer.to(torch.bfloat16)
            hin16 = h.to(torch.bfloat16)
            pe16 = tuple(t.to(torch.bfloat16) for t in pe32[lt])
            m16 = masks32[lt].to(torch.bfloat16)
            row = {"type": lt, "scores": STATS.get(i)}
            for impl in ("eager", "sdpa", "eager_s32", "eager_s32_p32", "eager_pv32"):
                ref.text_config._attn_implementation = impl
                with torch.no_grad():
                    o = lay16(hin16, shared_kv_states=UserDict(shared), position_embeddings=pe16, attention_mask=m16,
                              position_ids=pos, past_key_values=None)
                o = o.float()
                row[impl] = {"rel": ((o - hout).norm() / hout.norm()).item(), "max_abs": (o - hout).abs().max().item()}
            ref.text_config._attn_implementation = "eager"
            row["bf16_roundtrip_of_output"] = ((hout.to(torch.bfloat16).float() - hout).norm() / hout.norm()).item()
            res[i] = row
            print(i, json.dumps(row), flush=True)
        h = hout
        # shared kv states of the stack are only needed for kv-shared layers (num_kv_shared_layers=0 here)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
