"""CPU probe of the bf16-eager error jump at layer 10: is it the layer-10 implementation or amplification of
an already-perturbed input? Runs fp32-eager / bf16-eager / bf16-sdpa stacks to layer 9, then layer 10 with
every (input, impl, dtype) combination and reports per-position error of the layer-10 output vs fp32 truth.
NO GPU (hidden, aborts if visible)."""
from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import sys
from collections import UserDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

if torch.cuda.is_available():
    raise SystemExit("ABORT: cuda visible")
from gemma.common_gemma import DEFAULT_MODEL_DIR, KL_CORPUS_DIR  # noqa: E402
from gemma.ref import EMBED_NAME, FULL, SLIDING, GemmaReference, additive_mask  # noqa: E402

torch.set_num_threads(16)
T, L = 256, 10
doc = json.load(open(KL_CORPUS_DIR / "tokens_gemma.json", encoding="utf-8"))
ids = torch.as_tensor(next(s for s in doc["segments"] if s["name"] == "english_prose")["token_ids"][:T])
ref = GemmaReference(DEFAULT_MODEL_DIR, torch.device("cpu"), dtype=torch.float32, resident=False, load_table=False)
tab = ref.index.get_tensor(EMBED_NAME).to(torch.bfloat16)
layers = {i: ref.layer(i) for i in range(L + 1)}  # fp32, ~10 GB
pos = torch.arange(T).unsqueeze(0)
h0 = (tab[ids].float() * 62.0).unsqueeze(0)
pe32 = {lt: ref.rotary(h0, pos, lt) for lt in set(ref.layer_types)}
pe16 = {k: tuple(t.bfloat16() for t in v) for k, v in pe32.items()}
m32 = {FULL: additive_mask(T, T, 0, None, "cpu", torch.float32), SLIDING: additive_mask(T, T, 0, ref.arch.window, "cpu", torch.float32)}
m16 = {k: v.bfloat16() for k, v in m32.items()}


def run(layer, h, i, dt, impl):
    ref.text_config._attn_implementation = impl
    lt = ref.layer_types[i]
    pe = pe32[lt] if dt == torch.float32 else pe16[lt]
    m = m32[lt] if dt == torch.float32 else m16[lt]
    with torch.no_grad():
        return layer(h.to(dt), shared_kv_states=UserDict(), position_embeddings=pe, attention_mask=m, position_ids=pos,
                     past_key_values=None)


h = {"fp32": h0, "b16e": h0.bfloat16(), "b16s": h0.bfloat16()}
cfg = {"fp32": (torch.float32, "eager"), "b16e": (torch.bfloat16, "eager"), "b16s": (torch.bfloat16, "sdpa")}
for i in range(L):
    for k, (dt, impl) in cfg.items():
        lay = layers[i] if dt == torch.float32 else layers[i].to(torch.bfloat16)
        h[k] = run(lay, h[k], i, dt, impl)
        if dt != torch.float32:
            layers[i].to(torch.float32)  # (bf16 cast is lossy; reload exact below)
    print("layer", i, "done", flush=True)
# reload layer 10 exactly in fp32 from checkpoint (the .to(bf16)/.to(fp32) round trip above is lossless for
# bf16-origin weights, but rebuild anyway for safety)
l10 = ref.build_layer(L)
truth_in = h["fp32"]
truth_out = run(l10, truth_in, L, torch.float32, "eager").float()


def err(o):
    d = (o.float() - truth_out)[0]
    per = d.norm(dim=-1) / truth_out[0].norm(dim=-1)
    return {"rel": (d.norm() / truth_out.norm()).item(), "pos1_rel": per[1].item(), "pos0_rel": per[0].item(),
            "median_pos": per.median().item(), "worst": (int(per.argmax()), per.max().item())}


res = {}
l10_16 = ref.build_layer(L).to(torch.bfloat16)
for inp in ("fp32", "b16e", "b16s"):
    res[f"in_err_{inp}"] = ((h[inp].float() - truth_in).norm() / truth_in.norm()).item()
    res[f"in_{inp}_pos1_rel"] = ((h[inp].float() - truth_in)[0, 1].norm() / truth_in[0, 1].norm()).item()
    for impl in ("eager", "sdpa"):
        res[f"in={inp} bf16layer {impl}"] = err(run(l10_16, h[inp], L, torch.bfloat16, impl))
    res[f"in={inp} fp32layer eager"] = err(run(l10, h[inp], L, torch.float32, "eager"))
print(json.dumps(res, indent=1))
json.dump(res, open(r"D:\models\r4dx\huihui-gemma\kl\layer10_amp.json", "w"), indent=1)
