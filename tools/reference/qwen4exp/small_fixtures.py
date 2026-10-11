"""Fixtures for the small qwen4_exp GPU kernels (src/model/qwen4exp: hc, moe, ple, gdn_sigmoid).

Instantiates the HF modules of transformers' qwen4_exp (the source of truth, plan 4.11) with random
weights at the REAL dims (hidden 2560, hc 4 x 2560, 640-wide experts, PLE 10240 channels), runs them in
bfloat16 on the CPU, and writes the inputs, weights and expected outputs that the C++ tests
(tests/model/qwen4exp/test_gpu_{hc,moe,ple,gdn_sigmoid}*.cpp) compare against.

Run (no GPU needed):
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools/reference/qwen4exp/small_fixtures.py

Output dir (override with --out): E:\\models\\r4dx\\q4exp-fixtures\\SmallKernels
    index.txt            one line per tensor: `<key> <bf16|f32|i32> <rank> <d0> .. <dn-1>`; the data is the
                         raw little-endian array in `<key>.bin`.
    params.txt           `<key> <integer or float>` lines (shapes, eps, tie statistics, ...).

Router ties: torch.topk's order among equal probabilities is implementation defined; r4dx's rule is
"larger value first, ties to the lower expert id" (stable). The expected ids of the crafted tie cases
use that rule, and `route.*.torch_topk_ties_differ` records how many rows torch.topk itself disagreed on.
"""

import argparse
import copy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from transformers import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as m
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

BF = torch.bfloat16
HIDDEN = 2560
HC = 4
RMS_EPS = 1e-6


def make_config():
    return Qwen4ExpTextConfig(
        hidden_size=HIDDEN, num_hidden_layers=4, vocab_size=4096, eos_token_id=4095, num_experts=16,
        num_experts_per_tok=10, moe_intermediate_size=640, shared_expert_intermediate_size=640,
        hc_count=HC, hc_lowrank=320, ple_layer_ids=[2], ple_embed_dim=HIDDEN, ngram_vocab_size_base=2003,
        split_ngram_parts=4, linear_num_value_heads=48, linear_num_key_heads=16, output_gate_type="sigmoid",
        num_attention_heads=24, num_key_value_heads=2, head_dim=256, rms_norm_eps=RMS_EPS, hidden_act="silu")


class Writer:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.index = []
        self.params = []

    def tensor(self, key, t):
        if t.dtype == torch.bfloat16:
            kind, arr = "bf16", t.contiguous().view(torch.int16).numpy()
        elif t.dtype == torch.float32:
            kind, arr = "f32", t.contiguous().numpy()
        elif t.dtype in (torch.int32, torch.int64):
            kind, arr = "i32", t.contiguous().to(torch.int32).numpy()
        else:
            raise ValueError(f"{key}: unsupported dtype {t.dtype}")
        arr.tofile(os.path.join(self.out, key + ".bin"))
        self.index.append(f"{key} {kind} {len(t.shape)} " + " ".join(str(d) for d in t.shape))

    def param(self, key, value):
        self.params.append(f"{key} {value}")

    def finish(self):
        with open(os.path.join(self.out, "index.txt"), "w") as f:
            f.write("\n".join(self.index) + "\n")
        with open(os.path.join(self.out, "params.txt"), "w") as f:
            f.write("\n".join(self.params) + "\n")


def rnd(gen, *shape, std=1.0, mean=0.0):
    return (torch.randn(*shape, generator=gen) * std + mean).to(BF)


def set_weight(param, value):
    with torch.no_grad():
        param.copy_(value)


def residual_input(gen, rows):
    h = torch.randn(rows, HC * HIDDEN, generator=gen) * 2.0
    # a few outlier channels, as real residual streams have
    idx = torch.randperm(HC * HIDDEN, generator=gen)[:8]
    h[:, idx] *= 30.0
    return h.to(BF)


# ---- gated residual + embedding repeat --------------------------------------------------------------
def gen_hc(w, cfg):
    gen = torch.Generator().manual_seed(11)
    rows = 7
    w.param("hc.rows", rows)
    w.param("hc.eps", RMS_EPS)
    for name, use_combine in (("mix", True), ("final", False)):
        gr = m.Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine).to(BF).eval()
        set_weight(gr.hc_norm.weight, rnd(gen, HC * HIDDEN, std=0.1))
        set_weight(gr.input_mix_weight_down.weight, rnd(gen, 320, HC * HIDDEN, std=0.02))
        set_weight(gr.input_mix_weight_up.weight, rnd(gen, HC * HIDDEN, 320, std=0.05))
        if use_combine:
            set_weight(gr.block_inject_weight.weight, rnd(gen, HC, HC * HIDDEN, std=0.02))
        h = residual_input(gen, rows)
        block_out = rnd(gen, rows, HIDDEN, std=1.0)
        with torch.no_grad():
            n = gr.hc_norm(h)
            out = gr(h)
            if use_combine:
                x, hyper_input, inj = out
                injection = block_out.unsqueeze(-2) * inj.unsqueeze(-1)
                h_new = hyper_input + injection.flatten(-2)
            else:
                x = out
        w.tensor(f"hc.{name}.norm_w", gr.hc_norm.weight.data)
        w.tensor(f"hc.{name}.down_w", gr.input_mix_weight_down.weight.data)
        w.tensor(f"hc.{name}.up_w", gr.input_mix_weight_up.weight.data)
        w.tensor(f"hc.{name}.h", h)
        w.tensor(f"hc.{name}.n", n)
        w.tensor(f"hc.{name}.x", x)
        if use_combine:
            w.tensor(f"hc.{name}.inject_w", gr.block_inject_weight.weight.data)
            w.tensor(f"hc.{name}.inj", inj)
            w.tensor(f"hc.{name}.block_out", block_out)
            w.tensor(f"hc.{name}.h_new", h_new)
    # embedding gather + 4x repeat (Qwen4ExpTextModel: hidden_states.repeat(1, 1, hc_count), :1014)
    vocab, ids = 4096, torch.tensor([5, 0, 4095, 17, 17, 2048, 999], dtype=torch.int32)
    table = rnd(gen, vocab, HIDDEN, std=0.5)
    w.tensor("embed.table", table)
    w.tensor("embed.ids", ids)
    w.tensor("embed.out", table[ids.long()].repeat(1, HC))


# ---- router + MoE block -----------------------------------------------------------------------------
def hf_route(logits_bf16, k, norm_topk=True, stable=False):
    """Qwen3NextTopKRouter.forward after the linear. stable=True: ties to the lower expert id."""
    probs = torch.softmax(logits_bf16.float(), dim=-1)
    if stable:
        order = torch.sort(-probs, dim=-1, stable=True).indices[:, :k]
        vals = torch.gather(probs, 1, order)
        ids = order
    else:
        vals, ids = torch.topk(probs, k, dim=-1)
    if norm_topk:
        vals = vals / vals.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32), vals.to(BF).float()


def gen_route(w):
    gen = torch.Generator().manual_seed(23)
    k = 10
    cases = {}
    # (a) random logits, E = 16 and the real E = 512
    cases["rand16"] = (rnd(gen, 40, 16, std=1.5), 16)
    cases["rand512"] = (rnd(gen, 24, 512, std=1.5), 512)
    # (b) crafted ties, E = 16
    rows = []
    rows.append(torch.zeros(16))                                          # all equal: ids 0..9
    r = torch.randn(16, generator=gen) * 2
    r[3] = r[11] = r.max() + 1.0                                          # tie at the very top
    rows.append(r)
    r = torch.arange(16, dtype=torch.float32) * 0.5                       # strictly ordered
    rows.append(r)
    r = torch.arange(16, dtype=torch.float32) * 0.5
    r = r[torch.randperm(16, generator=gen)]
    rows.append(r)
    r = torch.randn(16, generator=gen) * 2                                # boundary: 10th == 11th
    order = torch.argsort(-r)
    r[order[10]] = r[order[9]]
    rows.append(r)
    r = torch.randn(16, generator=gen) * 2                                # boundary: 10th == 11th == 12th
    order = torch.argsort(-r)
    r[order[10]] = r[order[9]]
    r[order[11]] = r[order[9]]
    rows.append(r)
    r = torch.randn(16, generator=gen) * 2                                # boundary tie with higher id first
    order = torch.argsort(-r)
    lo, hi = sorted((int(order[9]), int(order[10])))
    r[lo] = r[hi] = r[order[9]]
    rows.append(r)
    r = torch.full((16,), -3.0)                                           # 9 distinct + 7 tied below
    r[:9] = torch.arange(9, dtype=torch.float32)[torch.randperm(9, generator=gen)]
    rows.append(r)
    r = torch.full((16,), 2.0)                                            # 10 tied on top, 6 below
    r[torch.randperm(16, generator=gen)[:6]] = -1.0
    rows.append(r)
    r = torch.randn(16, generator=gen) * 40                               # big spread: probs underflow
    rows.append(r)
    cases["ties16"] = (torch.stack(rows).to(BF), 16)
    # (c) heavy ties at E = 512: logits on a 0.25 grid
    g = torch.round(torch.randn(16, 512, generator=gen) * 2.0 * 4) / 4
    cases["ties512"] = (g.to(BF), 512)
    for name, (logits, E) in cases.items():
        ids, wts = hf_route(logits, k, stable=True)
        ids_t, wts_t = hf_route(logits, k, stable=False)
        differ = int((ids != ids_t).any(dim=-1).sum())
        w.param(f"route.{name}.rows", logits.shape[0])
        w.param(f"route.{name}.E", E)
        w.param(f"route.{name}.torch_topk_ties_differ", differ)
        w.tensor(f"route.{name}.logits", logits)
        w.tensor(f"route.{name}.ids", ids)
        w.tensor(f"route.{name}.weights", wts)
        counts = torch.bincount(ids.flatten().long(), minlength=E).to(torch.int32)
        w.tensor(f"route.{name}.counts", counts)
        print(f"route.{name}: rows={logits.shape[0]} E={E}: torch.topk differs from the stable rule on {differ} rows")


def gen_moe(w, cfg):
    gen = torch.Generator().manual_seed(31)
    moe = m.Qwen4ExpTextSparseMoeBlock(cfg).to(BF).eval()
    E = cfg.num_experts
    set_weight(moe.gate.weight, rnd(gen, E, HIDDEN, std=0.03))
    set_weight(moe.experts.gate_up_proj, rnd(gen, E, 1280, HIDDEN, std=0.02))
    set_weight(moe.experts.down_proj, rnd(gen, E, HIDDEN, 640, std=0.02))
    set_weight(moe.shared_expert.gate_proj.weight, rnd(gen, 640, HIDDEN, std=0.02))
    set_weight(moe.shared_expert.up_proj.weight, rnd(gen, 640, HIDDEN, std=0.02))
    set_weight(moe.shared_expert.down_proj.weight, rnd(gen, HIDDEN, 640, std=0.02))
    set_weight(moe.shared_expert_gate.weight, rnd(gen, 1, HIDDEN, std=0.02))
    # fp32 "truth" for the same weights: HF's bf16 expert-by-expert index_add_ accumulation is itself rounded at
    # every step, ours accumulates in fp32 and rounds once; the test reports both errors against this.
    moe32 = copy.deepcopy(moe).float().eval()
    w.param("moe.E", E)
    w.param("moe.k", cfg.num_experts_per_tok)
    w.tensor("moe.router_w", moe.gate.weight.data)
    w.tensor("moe.gate_up_w", moe.experts.gate_up_proj.data)
    w.tensor("moe.down_w", moe.experts.down_proj.data)
    w.tensor("moe.shared_gate_proj_w", moe.shared_expert.gate_proj.weight.data)
    w.tensor("moe.shared_up_proj_w", moe.shared_expert.up_proj.weight.data)
    w.tensor("moe.shared_down_proj_w", moe.shared_expert.down_proj.weight.data)
    w.tensor("moe.shared_gate_w", moe.shared_expert_gate.weight.data)
    for tag, rows in (("t70", 70), ("t1", 1)):
        x = rnd(gen, rows, HIDDEN, std=0.7)
        with torch.no_grad():
            logits, scores, idx = moe.gate(x)
            y = moe(x.view(1, rows, HIDDEN)).view(rows, HIDDEN)
            y32 = moe32(x.float().view(1, rows, HIDDEN)).view(rows, HIDDEN)
        w.param(f"moe.{tag}.rows", rows)
        w.tensor(f"moe.{tag}.x", x)
        w.tensor(f"moe.{tag}.logits", logits)
        w.tensor(f"moe.{tag}.ids", idx.to(torch.int32))
        w.tensor(f"moe.{tag}.weights", scores.float())
        w.tensor(f"moe.{tag}.y", y)
        w.tensor(f"moe.{tag}.y_f32", y32)


# ---- GDN sigmoid gated norm -------------------------------------------------------------------------
def gen_gdn(w):
    gen = torch.Generator().manual_seed(41)
    norm = m.Qwen4ExpTextRMSNormGated(128, eps=RMS_EPS, activation="sigmoid").to(BF).eval()
    set_weight(norm.weight, rnd(gen, 128, std=0.1, mean=1.0))
    tokens, heads = 7, 48
    rows = tokens * heads
    x = rnd(gen, rows, 128, std=1.5)
    z = rnd(gen, rows, 128, std=2.0)
    with torch.no_grad():
        y = norm(x, z)
        y_silu = m.Qwen4ExpTextRMSNormGated(128, eps=RMS_EPS, activation="silu").to(BF)
        set_weight(y_silu.weight, norm.weight.data)
        y_silu = y_silu(x, z)
    w.param("gdn.rows", rows)
    w.param("gdn.eps", RMS_EPS)
    w.tensor("gdn.weight", norm.weight.data)
    w.tensor("gdn.x", x)
    w.tensor("gdn.z", z)
    w.tensor("gdn.y_sigmoid", y)
    w.tensor("gdn.y_silu", y_silu)


# ---- PLE layer --------------------------------------------------------------------------------------
def gen_ple(w, cfg):
    gen = torch.Generator().manual_seed(53)
    ple = m.Qwen4ExpTextPLELayer(cfg, 1, 0).to(BF).eval()
    C = HC * HIDDEN
    set_weight(ple.key_proj.weight, rnd(gen, C, HIDDEN, std=0.02))
    set_weight(ple.value_proj.weight, rnd(gen, HIDDEN, HIDDEN, std=0.02))
    for n in (ple.norm_key, ple.norm_query, ple.norm_conv):
        set_weight(n.weight, rnd(gen, C, std=0.1))
    set_weight(ple.conv1d.weight, rnd(gen, C, 1, 4, std=0.4))
    set_weight(ple.ple_embedding.ngram_embedding.weight, rnd(gen, ple.ple_embedding.ngram_embedding.weight.shape[0], 160, std=0.5))
    T = 19
    ids = torch.randint(0, 4090, (1, T), generator=gen)
    ids[0, 5] = cfg.eos_token_id
    h = residual_input(gen, T).view(1, T, C)
    with torch.no_grad():
        emb = ple.ple_embedding(ids, None)
        out = ple(h, ids, None)
        h_small = (h.float() * 2.0 ** -8).to(BF)
        out_small = ple(h_small, ids, None)
        # The same layer driven through the HF cache, chunks 1 + 6 + 1 + 11: must equal the one-shot run
        # (this is what pins down the conv-state semantics the C++ PleLayer implements).
        cache = DynamicCache(config=cfg)
        parts, pos = [], 0
        for n in (1, 6, 1, 11):
            parts.append(ple(h[:, pos:pos + n], ids[:, pos:pos + n], cache))
            pos += n
        chunked = torch.cat(parts, dim=1)
    cache_diff = float((chunked.float() - out.float()).abs().max())
    print(f"ple: HF cached 1+6+1+11 vs one-shot max abs diff = {cache_diff}")
    if cache_diff != 0.0:
        raise SystemExit("ple: HF chunked and one-shot outputs differ; the zero-initialised 9-step state model is wrong")
    w.param("ple.T", T)
    w.param("ple.eps", RMS_EPS)
    w.tensor("ple.key_proj_w", ple.key_proj.weight.data)
    w.tensor("ple.value_proj_w", ple.value_proj.weight.data)
    w.tensor("ple.norm_key_w", ple.norm_key.weight.data)
    w.tensor("ple.norm_query_w", ple.norm_query.weight.data)
    w.tensor("ple.norm_conv_w", ple.norm_conv.weight.data)
    w.tensor("ple.conv1d_w", ple.conv1d.weight.data.view(C, 4))
    w.tensor("ple.ids", ids.view(T).to(torch.int32))
    w.tensor("ple.emb", emb.view(T, HIDDEN))
    w.tensor("ple.h", h.view(T, C))
    w.tensor("ple.out", out.view(T, C))               # PLE output; the layer does h + out
    w.tensor("ple.h_new", (h + out).view(T, C))       # bf16 add as in the decoder layer
    # A residual 2^-8 times smaller (the gate is scale invariant): the final h + out add then no longer hides out's last
    # bits under h's coarse bf16 spacing, so this case compares the PLE output itself.
    w.tensor("ple.h_small", h_small.view(T, C))
    w.tensor("ple.h_small_new", (h_small + out_small).view(T, C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:\models\r4dx\q4exp-fixtures\SmallKernels")
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = make_config()
    w = Writer(args.out)
    gen_hc(w, cfg)
    gen_route(w)
    gen_moe(w, cfg)
    gen_gdn(w)
    gen_ple(w, cfg)
    w.finish()
    print(f"wrote {len(w.index)} tensors to {args.out}")


if __name__ == "__main__":
    main()
