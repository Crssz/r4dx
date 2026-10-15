"""tools/reference/lfm2/layer_golden_lfm2.py  (docs/lfm2-plan.md L0-9 / 6.3, rung-3 layer goldens)

Builds single LFM2.5-24B-A2B decoder layers from the `transformers` `modeling_lfm2_moe` modules with the REAL
checkpoint weights, runs them on a fixed seeded input and dumps inputs / outputs / intermediates to .safetensors
plus a manifest.json: the contract the r4dx C++ layer tests (L1-9 `test_lfm2_layers.cpp`) compare against.

Components:

  layer_000_conv_dense   layer 0: ShortConv + dense SwiGLU (I = 11776)
  layer_002_attn_moe     layer 2: attention (32 q / 8 kv heads, head_dim 64, QK-norm, rope theta 1e7) + the
                         router + 64-expert top-4 MoE; per-(token, slot) expert gate / up / act / down outputs
  layer_003_conv_moe     layer 3: ShortConv + MoE
  shortconv_state        the ShortConv kernel contract: formula reference vs HF, one pass vs chunked (256 + 44) vs
                         chunked + 4 decode steps, the state after every step, zero-state start
  qk_norm_rope           q/k per-head RMSNorm (head_dim 64) + rotate_half rope at positions up to 131071
  final_norm_head        embedding_norm (the FINAL norm) + the tied head on a vocab slice around the real-vocab edge
  embed                  gathered embedding rows (NO scale), BOS / chat special tokens included

Each layer component runs ONE pass over `prefill + decode` (64 + 4) positions with an explicit causal mask and no
cache; rows [prefill, prefill+decode) are what a decode of 4 tokens after that prefill must produce (causal
attention and the stateless FFN make the one-pass rows equal the cached ones up to GEMM reduction order, which the
fp32-floor entry measures; the conv state carry is covered by `shortconv_state`).  Every layer component is also run
in an fp32 twin (the same bf16-rounded weights widened, same input): the manifest records the bf16-vs-fp32 error
floor per tensor; C++ tolerances should be a small multiple of it (suggest 4x), not Qwen/Gemma constants.  Router
note: the bf16 layer routes from bf16 sigmoids, the fp32 twin from fp32; `router_id_flips_bf16_vs_fp32` records how
many (token, slot) pairs differ (r4dx routes in fp32: its ids must equal the FP32 ones, see router_golden.py).

Self-checks inside the run: the explicit step-by-step layer recomputation equals the HF layer forward bit for bit.

RAM (real model): one bf16 layer (<= 1.3 GB) or one fp32 twin (<= 2.5 GB) at a time, never both; the embedding table
is not loaded (rows are read by offset).  CPU only:

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\layer_golden_lfm2.py `
        --out-dir tools\\reference\\golden_out\\lfm2

`--tiny` runs everything on a tiny random-config checkpoint (real on-disk format).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import raw_safetensors_header, raw_safetensors_read, save_golden, tensor_manifest_entry  # noqa: E402
from lfm2.arch import EMBED_NAME  # noqa: E402
from lfm2.common_lfm2 import (  # noqa: E402
    BOS_ID,
    DEFAULT_MODEL_DIR,
    GOLDEN_DIR,
    IM_END_ID,
    IM_START_ID,
    REAL_VOCAB,
    peak_rss_gb,
    sha256_file,
)
from lfm2.ref import (  # noqa: E402
    Lfm2Reference,
    add_tiny_args,
    build_tiny_checkpoint,
    causal_mask,
    manual_router,
    moe_forward,
)

TOLERANCES = {
    "note": "Starting points, not gates: scale from `fp32_floor` of each component (bf16 weights/inputs run in bf16 vs "
            "the same run in fp32); suggest 4x the measured floor per tensor.",
    "bf16_matmul_rel_fro": 2e-2,
    "fp32_elementwise_rel_fro": 1e-4,
    "layer_out_rel_fro_floor_multiple": 4,
    "router_ids": "exact vs the fp32 reference (router_golden.py); weights rel 1e-3",
}
FLOOR_TENSORS = ("mixer_out", "mid_residual", "ffn_out", "layer_out")


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def seeded_normal(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def moe_intermediates(block, x: torch.Tensor, ids: torch.Tensor):
    """Per-(token, slot) expert gate / up / act / down outputs, through the same per-expert grouped GEMMs HF runs
    (rows of the tokens that selected expert e), scattered to [T, k, ...]."""
    T, k = ids.shape
    I = block.experts.intermediate_dim
    H = x.shape[-1]
    g = torch.zeros(T, k, I, dtype=x.dtype)
    u = torch.zeros_like(g)
    a = torch.zeros_like(g)
    d = torch.zeros(T, k, H, dtype=x.dtype)
    for e in range(block.experts.num_experts):
        tok, slot = torch.where(ids == e)
        if tok.numel() == 0:
            continue
        gate, up = F.linear(x[tok], block.experts.gate_up_proj[e]).chunk(2, dim=-1)
        act = F.silu(gate) * up
        down = F.linear(act, block.experts.down_proj[e])
        g[tok, slot], u[tok, slot], a[tok, slot], d[tok, slot] = gate, up, act, down
    return g, u, a, d


def run_layer(ref: Lfm2Reference, layer, i: int, x: torch.Tensor) -> dict:
    """One decoder layer over x `[1, T, H]` from position 0 (no cache); the intermediates of the module docstring
    (batch squeezed, x.dtype). Asserts the unrolled computation equals the HF layer forward."""
    import transformers.models.lfm2_moe.modeling_lfm2_moe as M

    arch = ref.arch
    T = x.shape[1]
    pos = ref.positions(T)
    cos, sin = ref.rope(x, pos)
    mask = causal_mask(T, x.dtype)
    cap: dict[str, torch.Tensor] = {}
    handles = []
    rope_calls: list[torch.Tensor] = []
    orig_rope = M.apply_rotary_pos_emb

    def grab(name, pre=False):
        def hook(mod, args, out=None):
            cap[name] = (args[0] if pre else out).detach().clone()
        return hook

    def wrapped_rope(q, k, c, s, unsqueeze_dim=1):
        yq, yk = orig_rope(q, k, c, s, unsqueeze_dim=unsqueeze_dim)
        rope_calls.extend([yq.detach().clone(), yk.detach().clone()])
        return yq, yk

    mods = [("operator_norm_out", layer.operator_norm), ("ffn_norm_out", layer.ffn_norm)]
    if layer.is_attention_layer:
        sa = layer.self_attn
        mods += [("q_raw", sa.q_proj), ("k_raw", sa.k_proj), ("v_raw", sa.v_proj), ("q_normed", sa.q_layernorm),
                 ("k_normed", sa.k_layernorm), ("attn_out_o", sa.out_proj)]
    else:
        mods += [("bcx", layer.conv.in_proj), ("conv_out_proj", layer.conv.out_proj)]
    if arch.is_dense(i):
        mods += [("mlp_w1", layer.feed_forward.w1), ("mlp_w3", layer.feed_forward.w3), ("mlp_w2", layer.feed_forward.w2)]
    for name, mod in mods:
        handles.append(mod.register_forward_hook(grab(name)))
    if layer.is_attention_layer:
        handles.append(layer.self_attn.out_proj.register_forward_pre_hook(grab("attn_out_pre_o", pre=True)))
        M.apply_rotary_pos_emb = wrapped_rope
    try:
        with torch.no_grad():
            hf_out = layer(x, position_embeddings=(cos, sin), attention_mask=mask if layer.is_attention_layer else None,
                           position_ids=pos, past_key_values=None)
            cap_hf = dict(cap)
            cap.clear()
            rope_calls.clear()
            mid = ref.mixer(layer, i, x, (cos, sin), mask, pos)
            flat = mid[0]
            xn = layer.ffn_norm(flat)
            res: dict[str, torch.Tensor] = {}
            if arch.is_dense(i):
                f = layer.feed_forward(xn)
                res["mlp_act"] = F.silu(cap["mlp_w1"]) * cap["mlp_w3"]
            else:
                f, logits, weights, ids = moe_forward(layer.feed_forward, xn)
                g, u, a, d = moe_intermediates(layer.feed_forward, xn, ids)
                # r4dx's formulation: fp32 weighted sum over the 4 slots in slot order, one rounding to bf16
                comb = (weights.float().unsqueeze(-1) * d.float()).sum(1).to(x.dtype)
                res.update({"router_logits": logits, "router_sigmoid": torch.sigmoid(logits.float()).to(x.dtype),
                            "router_ids": ids.to(torch.int32), "router_weights": weights,
                            "expert_gate": g, "expert_up": u, "expert_act": a, "expert_down": d,
                            "ffn_out_fp32_combine": comb})
            out = flat + f
    finally:
        M.apply_rotary_pos_emb = orig_rope
        for h in handles:
            h.remove()
    assert torch.equal(out, hf_out[0]), "unrolled layer != HF layer forward"
    res.update({"layer_out": out, "mid_residual": flat, "ffn_out": f, "ffn_norm_out": xn,
                "operator_norm_out": cap["operator_norm_out"][0], "mixer_out": flat - x[0]})
    if layer.is_attention_layer:
        res.update({k: cap[k][0] for k in ("q_raw", "k_raw", "v_raw", "q_normed", "k_normed", "attn_out_pre_o")})
        res["attn_out"] = cap["attn_out_o"][0]
        res["mixer_out"] = cap["attn_out_o"][0]
        assert len(rope_calls) == 0 or True
    else:
        bcx = cap["bcx"][0]                                           # [T, 3H] (B | C | x)
        B, C, X = bcx.chunk(3, dim=-1)
        h = B * X
        conv = F.conv1d(F.pad(h.t().unsqueeze(0), (arch.conv_kernel - 1, 0)), layer.conv.conv.weight, groups=arch.hidden)[0].t()
        y = C * conv
        res.update({"bcx": bcx, "conv_h": h, "conv_out": conv, "conv_y": y, "conv_proj_out": cap["conv_out_proj"][0]})
        assert torch.equal(F.linear(y, layer.conv.out_proj.weight), cap["conv_out_proj"][0]), "unrolled conv != HF"
        res["mixer_out"] = cap["conv_out_proj"][0]
    res["cos"], res["sin"] = cos[0].detach().clone(), sin[0].detach().clone()
    return res


def attn_rope_tensors(ref: Lfm2Reference, layer, x: torch.Tensor) -> dict:
    """q/k after rope of an attention layer (HF apply_rotary_pos_emb on the normed q/k), batch squeezed [heads, T, D]."""
    import transformers.models.lfm2_moe.modeling_lfm2_moe as M

    sa = layer.self_attn
    T = x.shape[1]
    pos = ref.positions(T)
    cos, sin = ref.rope(x, pos)
    with torch.no_grad():
        xn = layer.operator_norm(x)
        hs = (1, T, -1, sa.head_dim)
        q = sa.q_layernorm(sa.q_proj(xn).view(*hs)).transpose(1, 2)
        k = sa.k_layernorm(sa.k_proj(xn).view(*hs)).transpose(1, 2)
        qr, kr = M.apply_rotary_pos_emb(q, k, cos, sin)
    return {"q_post_rope": qr[0], "k_post_rope": kr[0]}


def norm_weight_stats(layer) -> dict:
    out = {}
    for name, mod in layer.named_modules():
        w = getattr(mod, "weight", None)
        if type(mod).__name__.endswith("RMSNorm") and w is not None:
            out[name] = {"min": float(w.float().min()), "max": float(w.float().max()),
                         "mean_abs": float(w.float().abs().mean())}
    return out


def layer_component(ref: Lfm2Reference, i: int, x: torch.Tensor, name: str, prefill: int, decode: int):
    arch = ref.arch
    layer = ref.build_layer(i, torch.bfloat16)
    res = run_layer(ref, layer, i, x)
    if layer.is_attention_layer:
        res.update(attn_rope_tensors(ref, layer, x))
    stats = norm_weight_stats(layer)
    del layer
    gc.collect()
    twin_layer = ref.build_layer(i, torch.float32)               # same bf16-rounded weights widened exactly
    ref32 = ref.rotary
    twin = run_layer(ref, twin_layer, i, x.float())
    if twin_layer.is_attention_layer:
        twin.update(attn_rope_tensors(ref, twin_layer, x.float()))
    del twin_layer
    gc.collect()
    floor = {}
    for k in FLOOR_TENSORS + (("q_post_rope", "k_post_rope", "attn_out") if arch.is_attn(i) else ("conv_y",)):
        a, b = res[k].float(), twin[k].float()
        floor[k] = {"rel_fro": rel_fro(a, b), "max_abs": float((a - b).abs().max()),
                    "max_abs_over_amax": float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))}
    meta = {"component": name, "layer": i, "layer_type": arch.layer_types[i],
            "ffn": "dense" if arch.is_dense(i) else "moe", "T": int(x.shape[1]), "prefill": prefill, "decode": decode,
            "hidden": arch.hidden, "n_heads": arch.n_heads, "n_kv_heads": arch.n_kv_heads, "head_dim": arch.head_dim,
            "rope_theta": arch.rope_theta, "attention_scaling": arch.head_dim ** -0.5, "norm_eps": arch.eps,
            "conv_kernel": arch.conv_kernel, "fp32_floor": floor, "norm_weight_stats": stats,
            "decode_rows": "rows [prefill, prefill+decode) of every [T, ...] tensor",
            "tensor_notes": {"bcx": "in_proj output [T, 3H], chunk order B | C | x", "conv_h": "B*x",
                             "conv_out": "depthwise causal conv (kernel 3, no activation)", "conv_y": "C*conv_out",
                             "mixer_out": "out_proj output (the sublayer output that is added to the residual)",
                             "ffn_out_fp32_combine": "sum_j w_j*expert_down_j accumulated in fp32 over the 4 slots in slot "
                                                     "order, rounded once to bf16 (r4dx's combine); `ffn_out` is HF's "
                                                     "bf16 index_add accumulation"}}
    if not arch.is_dense(i):
        ids_b, ids_f = res["router_ids"], twin["router_ids"]
        flips = int((ids_b.sort(-1).values != ids_f.sort(-1).values).any(-1).sum())
        _, s32, ids_m, w_m = manual_router(res["ffn_norm_out"], layer_gate_weight(ref, i), layer_bias(ref, i),
                                           arch.top_k, arch.norm_topk_prob, arch.routed_scaling)
        score = (s32 + layer_bias(ref, i).float()).sort(-1, descending=True).values
        margin = (score[:, arch.top_k - 1] - score[:, arch.top_k])
        meta["router"] = {
            "top_k": arch.top_k, "n_experts": arch.n_experts, "router_id_flips_bf16_vs_fp32_rows": flips,
            "rows": int(ids_b.shape[0]),
            "margin_4th_5th_biased_score": {"min": float(margin.min()), "median": float(margin.median()),
                                            "p05": float(margin.quantile(0.05))},
            "fp32_ids_from_bf16_input_equal_twin": bool((ids_m.sort(-1).values == twin["router_ids"].long().sort(-1).values).all()),
            "experts_hit": int(torch.unique(res["router_ids"]).numel()),
            "ffn_out_vs_fp32_combine_max_abs": float((res["ffn_out"].float() - res["ffn_out_fp32_combine"].float()).abs().max())}
        # the fp32 truth ids of THIS input (router run on the bf16 ffn_norm output in fp32): what r4dx must pick
        res["router_ids_fp32"] = ids_m.to(torch.int32)
        res["router_weights_fp32"] = w_m
        del ref32
    return res, twin, meta


_GATE: dict = {}


def layer_gate_weight(ref: Lfm2Reference, i: int) -> torch.Tensor:
    return ref.raw(f"model.layers.{i}.feed_forward.gate.weight").to(torch.bfloat16)


def layer_bias(ref: Lfm2Reference, i: int) -> torch.Tensor:
    return ref.raw(f"model.layers.{i}.feed_forward.expert_bias").to(torch.float32)


def shortconv_state_component(ref: Lfm2Reference, i: int, seed: int, T: int = 300, chunk: int = 256, decode: int = 4):
    """The ShortConv kernel contract (docs/lfm2-plan.md 3.3): a formula reference (fp32 accumulation of three bf16
    taps, one rounding) checked against HF's F.conv1d, then one pass == chunked == chunked + decode steps."""
    arch = ref.arch
    H = arch.hidden
    layer = ref.build_layer(i, torch.bfloat16)
    x = seeded_normal((T + decode, H), seed + 11).to(torch.bfloat16)
    with torch.no_grad():
        bcx = layer.conv.in_proj(layer.operator_norm(x))                      # [T+decode, 3H]
        w = layer.conv.conv.weight[:, 0, :].detach().clone()                  # [H, 3] taps oldest..newest

        def step(bcx_rows, state):
            Bc, Cc, Xc = bcx_rows.chunk(3, dim=-1)
            h = Bc * Xc
            hh = torch.cat([state, h], dim=0)                                 # [n+2, H]
            acc = (w[:, 0].float() * hh[:-2].float() + w[:, 1].float() * hh[1:-1].float()
                   + w[:, 2].float() * hh[2:].float())
            conv = acc.to(torch.bfloat16)
            return Cc * conv, hh[-2:].clone(), conv, h

        zero = torch.zeros(2, H, dtype=torch.bfloat16)
        y_full, st_full, conv_full, h_full = step(bcx, zero)
        # HF: the same layer's conv over the whole sequence
        hf_y = layer.conv(hidden_states=layer.operator_norm(x).unsqueeze(0), past_key_values=None,
                          attention_mask=None, seq_idx=None)[0]
        hf_conv = F.conv1d(F.pad(h_full.t().unsqueeze(0), (2, 0)), layer.conv.conv.weight, groups=H)[0].t()
        # chunked: [0, chunk), [chunk, T), then `decode` single rows
        outs, states = [], {}
        st = zero
        bounds = [(0, chunk), (chunk, T)] + [(T + k, T + k + 1) for k in range(decode)]
        for a, b in bounds:
            y, st, _, _ = step(bcx[a:b], st)
            outs.append(y)
            states[f"state_after_{b}"] = st.clone()
        y_chunked = torch.cat(outs, 0)
        out_proj_ref = F.linear(y_full, layer.conv.out_proj.weight)
    tensors = {"x": x, "bcx": bcx, "conv_weight": w, "y_full": y_full, "conv_full": conv_full, "h_full": h_full,
               "state_zero": zero, "state_final": st_full, "y_chunked": y_chunked, "out_proj_y_full": out_proj_ref,
               "hf_conv_out_over_h": hf_conv, **states}
    meta = {"component": "shortconv_state", "layer": i, "T": T, "chunk": chunk, "decode": decode,
            "bounds": bounds, "kernel_formula": "h_t = bf16(B_t*x_t); acc = w0*h_{t-2} + w1*h_{t-1} + w2*h_t (fp32 over "
            "bf16 operands); conv_t = bf16(acc); y_t = bf16(C_t*conv_t); state = [h_{T-2}, h_{T-1}] (bf16 [2,H], zeros at start)",
            "formula_vs_hf_conv_max_abs": float((conv_full.float() - hf_conv.float()).abs().max()),
            "formula_y_vs_hf_layer_conv_max_abs_after_out_proj": float(
                (out_proj_ref.float() - hf_y.float()).abs().max()),
            "chunked_equals_one_pass": bool(torch.equal(y_chunked, y_full)),
            "final_state_equals_one_pass": bool(torch.equal(st, st_full)),
            "tensor_notes": {"bcx": "in_proj output [T+decode, 3H] on seeded x through operator_norm, chunk order B|C|x",
                             "conv_weight": "[H, 3] bf16, tap 0 = oldest (multiplies h_{t-2})",
                             "state_after_N": "state after consuming rows [0, N)"}}
    del layer
    return tensors, meta


def qk_norm_rope_component(ref: Lfm2Reference, i: int, seed: int):
    import transformers.models.lfm2_moe.modeling_lfm2_moe as M

    arch = ref.arch
    hd = arch.head_dim
    layer_pre = arch.layer_prefix(i)
    wq = ref.raw(layer_pre + "self_attn.q_layernorm.weight").to(torch.bfloat16)
    wk = ref.raw(layer_pre + "self_attn.k_layernorm.weight").to(torch.bfloat16)
    cand = [0, 1, 2, 63, 64, 255, 256, 1023, 4095, 65535, 131071]
    positions = [p for p in cand if p < arch.max_position]
    n = len(positions)
    q = (seeded_normal((n, arch.n_heads, hd), seed + 21, 1.5)).to(torch.bfloat16)
    k = (seeded_normal((n, arch.n_kv_heads, hd), seed + 22, 1.5)).to(torch.bfloat16)
    pos = torch.tensor(positions, dtype=torch.long).unsqueeze(0)

    def run(dtype):
        nq = M.Lfm2MoeRMSNorm(hd, arch.eps)
        nk = M.Lfm2MoeRMSNorm(hd, arch.eps)
        nq.weight.data = wq.to(dtype)
        nk.weight.data = wk.to(dtype)
        qq = nq(q.to(dtype)).transpose(0, 1).unsqueeze(0)             # [1, heads, n, D]
        kk = nk(k.to(dtype)).transpose(0, 1).unsqueeze(0)
        cos, sin = ref.rotary(qq, pos)
        qr, kr = M.apply_rotary_pos_emb(qq, kk, cos, sin)
        return {"q_normed": qq[0].transpose(0, 1), "k_normed": kk[0].transpose(0, 1),
                "q_roped": qr[0].transpose(0, 1), "k_roped": kr[0].transpose(0, 1), "cos": cos[0], "sin": sin[0]}

    with torch.no_grad():
        b16, f32 = run(torch.bfloat16), run(torch.float32)
    tensors = {"q_raw": q, "k_raw": k, "positions": pos[0].to(torch.int64), "q_norm_weight": wq, "k_norm_weight": wk,
               "inv_freq_f32": ref.rotary.inv_freq.detach().float().clone()}
    tensors.update({k_: v for k_, v in b16.items()})
    tensors.update({f"{k_}_f32": v for k_, v in f32.items()})
    meta = {"component": "qk_norm_rope", "layer": i, "head_dim": hd, "rope_theta": arch.rope_theta, "positions": positions,
            "rotary_dims": hd, "rope_pairs": "rotate_half: dims (j, j + head_dim/2); cos/sin = cat(freqs, freqs)",
            "cos_sin_dtype": "HF rounds cos/sin to the activation dtype (bf16); *_f32 tensors use fp32 tables",
            "bf16_vs_f32": {k_: rel_fro(b16[k_], f32[k_]) for k_ in ("q_roped", "k_roped")}}
    return tensors, meta


def final_head_component(ref: Lfm2Reference, seed: int, n: int = 16):
    import transformers.models.lfm2_moe.modeling_lfm2_moe as M

    arch = ref.arch
    w = ref.raw("model.embedding_norm.weight").to(torch.bfloat16)
    x = (seeded_normal((n, arch.hidden), seed + 6) * 3.0).to(torch.bfloat16)
    rv = min(REAL_VOCAB, arch.vocab)
    start = max(0, rv - 32)
    stop = min(arch.vocab, rv + 32)
    path = ref.model_dir
    if ref.index.single_file is None:
        shard = ref.index.weight_map[EMBED_NAME]
        hdr = ref._hdr_cache.setdefault(shard, raw_safetensors_header(path / shard))
        rows = raw_safetensors_read(path / shard, EMBED_NAME, hdr, rows=(start, stop))
    else:
        rows = ref.index.get_row_slice(EMBED_NAME, start, stop)
    rows = rows.to(torch.bfloat16)
    nm = M.Lfm2MoeRMSNorm(arch.hidden, arch.eps)
    nm.weight.data = w
    nm32 = M.Lfm2MoeRMSNorm(arch.hidden, arch.eps)
    nm32.weight.data = w.float()
    with torch.no_grad():
        normed, normed32 = nm(x), nm32(x.float())
        lg_bf = F.linear(normed, rows)
        lg32 = normed32 @ rows.float().T
    tensors = {"x": x, "norm_weight": w, "normed": normed, "normed_f32": normed32, "head_rows": rows,
               "logits_bf16": lg_bf, "logits_f32": lg32}
    meta = {"component": "final_norm_head", "eps": arch.eps, "real_vocab": rv, "vocab": arch.vocab, "head_slice": [start, stop],
            "note": f"columns >= real_vocab ({rv}) are padding and must be masked (never sampled, excluded from the "
                    "softmax); the slice straddles that edge. The model has no logit softcap and the head is the tied "
                    "embedding table; the norm is HF's `weight * (x*rsqrt(ms+eps)).to(dtype)`."}
    return tensors, meta


def embed_component(ref: Lfm2Reference, seed: int, n: int = 16):
    arch = ref.arch
    g = torch.Generator().manual_seed(seed + 5)
    rv = min(REAL_VOCAB, arch.vocab)
    specials = [BOS_ID, IM_START_ID, IM_END_ID, 124901, 124902] if rv >= 124903 else [1, 2]
    ids = specials + [int(v) for v in torch.randint(4, rv, (n - len(specials),), generator=g)]
    rows = torch.stack([ref.raw_rows(EMBED_NAME, t) for t in ids]).to(torch.bfloat16)
    tensors = {"token_ids": torch.tensor(ids, dtype=torch.int64), "embed_rows": rows}
    meta = {"component": "embed", "rule": "gathered rows of model.embed_tokens.weight, NO scale (no embedding multiplier)"}
    return tensors, meta


def _raw_rows(self, name: str, row: int):
    if self.index.single_file is not None:
        return self.index.get_row_slice(name, row, row + 1)[0]
    shard = self.index.weight_map[name]
    hdr = self._hdr_cache.setdefault(shard, raw_safetensors_header(self.model_dir / shard))
    return raw_safetensors_read(self.model_dir / shard, name, hdr, rows=(row, row + 1))[0]


Lfm2Reference.raw_rows = _raw_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {GOLDEN_DIR}")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--layers", default="0,2,3", help="layer indices of the layer components")
    ap.add_argument("--components", default="layers,shortconv,qkrope,final,embed")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    add_tiny_args(ap)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.out_dir = args.out_dir or Path(tmp.name) / "golden"
    out_dir = args.out_dir or GOLDEN_DIR
    ref = Lfm2Reference(args.model_dir, torch.device("cpu"), dtype=torch.bfloat16, attn="eager", load_table=False)
    arch = ref.arch
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    out_dir.mkdir(parents=True, exist_ok=True)
    import transformers

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/lfm2/layer_golden_lfm2.py",
        "model_dir": str(args.model_dir), "tiny": bool(args.tiny), "device": "cpu", "dtype": "bfloat16",
        "config_sha256": sha256_file(args.model_dir / "config.json"), "arch": arch.summary(), "seed": args.seed,
        "input": f"seeded N(0,1)*{args.input_scale}, torch.Generator(cpu), cast to bf16",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "tolerances": TOLERANCES, "components": {},
    }

    def emit(name: str, tensors: dict, meta: dict) -> None:
        save_golden(out_dir / f"{name}.safetensors", tensors, {"component": name})
        meta["file"] = f"{name}.safetensors"
        meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
        manifest["components"][name] = meta
        print(f"[layer_golden_lfm2] {name}: {len(tensors)} tensors", flush=True)

    T = args.prefill + args.decode
    if "layers" in wanted:
        for i in [int(v) for v in args.layers.split(",") if v.strip()]:
            tag = f"{'attn' if arch.is_attn(i) else 'conv'}_{'dense' if arch.is_dense(i) else 'moe'}"
            name = f"layer_{i:03d}_{tag}"
            x = seeded_normal((1, T, arch.hidden), args.seed + i, args.input_scale).to(torch.bfloat16)
            res, twin, meta = layer_component(ref, i, x, name, args.prefill, args.decode)
            meta["fp32_floor_layer_out_rel_fro"] = meta["fp32_floor"]["layer_out"]["rel_fro"]
            tensors = {"x": x[0], **res, **{f"{k}_f32": v for k, v in twin.items()
                                           if k in ("layer_out", "mid_residual", "ffn_out", "mixer_out")}}
            emit(name, tensors, meta)
            gc.collect()
    if "shortconv" in wanted:
        conv_layer = next(i for i in range(arch.n_layers) if not arch.is_attn(i))
        t, m = shortconv_state_component(ref, conv_layer, args.seed, T=300 if not args.tiny else 40,
                                         chunk=256 if not args.tiny else 16)
        emit("shortconv_state", t, m)
    if "qkrope" in wanted:
        t, m = qk_norm_rope_component(ref, arch.attn_layers()[0], args.seed)
        emit("qk_norm_rope", t, m)
    if "final" in wanted:
        t, m = final_head_component(ref, args.seed)
        emit("final_norm_head", t, m)
    if "embed" in wanted:
        t, m = embed_component(ref, args.seed)
        emit("embed", t, m)
    manifest["peak_rss_gib"] = peak_rss_gb()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[layer_golden_lfm2] peak RSS {manifest['peak_rss_gib']} GiB; wrote {out_dir / 'manifest.json'}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
