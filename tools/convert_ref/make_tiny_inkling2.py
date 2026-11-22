#!/usr/bin/env python
"""Generates a TINY inkling2_loop checkpoint in the ORIGINAL naming (model.llm.*) -- the converter's and the model
lane's end-to-end fixture (docs/inkling2-plan.md 4.2) -- and, with --reference, runs the HF reference on it.

  python -I tools/convert_ref/make_tiny_inkling2.py --out tests/convert/data/inkling2_tiny [--reference]

Writes (under --out):
  model.safetensors   every tensor of a real checkpoint's naming / dtypes / layout, at hidden 64 (see TINY below):
                      bf16 non-experts, F32 queries and router bias, NVFP4 experts (U8 + F8_E4M3 + amax + denominator)
                      in layers 0-11 and MXFP8 experts (F8_E4M3 + U8 E8M0) in layers 12-15, three model.visual.* dummies
                      (allow-listed by the converter, unexpected to a text-only HF model)
  config.json         the real config.json's structure (top-level inkling2_loop, nested text_config incl. loop_config,
                      layer_types + local_layer_ids); both layer kinds, full layers at (i + 1) % 4 == 0
  with --reference (needs the reference venv: D:\\venvs\\r4dx-inkling-ref, transformers main with inkling2_loop):
  tokens.json         the fixed 40-token sequence (longer than the window, 8, and than log_scaling_n_floor, 16)
  logits.npy          float32 [40, unpadded_vocab] -- Inkling2LoopForCausalLM(dtype=float32) logits on those tokens,
                      the experts dequantized by HF's own Inkling2LoopDequantizeExperts in float32

Deterministic (seeded numpy); re-running reproduces the files byte for byte. No torch is needed without --reference,
except for the e4m3 conversions (torch.float8_e4m3fn), which the generator does through torch (CPU).
"""
import argparse
import json
import os
import struct

import numpy as np
import torch

TINY = dict(
    hidden=64, layers=16, heads=4, kv_heads=2, head_dim=16, d_rel=4, rel_extent=8, window=8, vocab=112, unpadded=90,
    experts=8, topk=2, inter=32, block=4, loop_start=4, loop_end=12, passes=3, rank=8, conv=4,
    nvfp4_layers=12,  # layers [0, 12) NVFP4, [12, 16) MXFP8 (real: 0-19 / 20-23)
    log_floor=16,
)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
NVFP4_DENOM = 2688.0  # 6 * 448, modelopt's outer scale denominator


def is_full(i, t=TINY):
    return (i + 1) % t["block"] == 0


def config(t=TINY):
    layer_types = ["hybrid" if is_full(i) else "hybrid_sliding" for i in range(t["layers"])]
    text = {
        "model_type": "inkling2_loop_text", "model_max_length": 4096, "torch_dtype": "bfloat16", "eos_token_id": 100,
        "hidden_size": t["hidden"], "num_hidden_layers": t["layers"], "vocab_size": t["vocab"],
        "num_attention_heads": t["heads"], "num_key_value_heads": t["kv_heads"], "head_dim": t["head_dim"],
        "d_rel": t["d_rel"], "rel_extent": t["rel_extent"], "q_bias": False, "o_bias": False,
        "log_scaling_n_floor": t["log_floor"], "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 8.0, "rel_l2_clip_threshold": 4.0,
        "layer_types": layer_types,
        "local_layer_ids": [i for i in range(t["layers"]) if not is_full(i)],
        "sconv_kernel_size": t["conv"], "unpadded_vocab_size": t["unpadded"], "final_logit_softcapping": 30.0,
        "swa_head_dim": t["head_dim"], "swa_num_attention_heads": t["heads"], "swa_num_key_value_heads": t["kv_heads"],
        "sliding_window_size": t["window"], "n_layers_per_attn_res_block": t["block"], "attn_res_normalize_values": True,
        "n_routed_experts": t["experts"], "num_experts_per_tok": t["topk"], "n_shared_experts": 0,
        "intermediate_size": t["inter"], "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
        "use_gate_bias": True,
        "loop_config": {"core_start": t["loop_start"], "core_end": t["loop_end"], "num_passes": t["passes"],
                        "gate_rank": t["rank"], "is_varloop": False},
    }
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"], "model_type": "inkling2_loop",
        "eos_token_id": 100, "text_config": text,
        "vision_config": {"model_type": "inkling2_vision", "text_hidden_size": t["hidden"]},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


# ---- number formats -------------------------------------------------------------------------------------------

def to_bf16_bytes(x):
    """float32 -> bf16 bytes, round to nearest even (NaN never occurs here)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16).tobytes()


def bf16_round(x):
    u = np.frombuffer(to_bf16_bytes(x), dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(np.shape(x))


def e4m3_codes(x):
    """float32 -> OCP e4m3fn codes (uint8), RNE, saturating at +-448 (torch's float8_e4m3fn cast)."""
    t = torch.from_numpy(np.ascontiguousarray(np.clip(x, -448.0, 448.0), dtype=np.float32))
    return t.to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def e4m3_values(codes):
    return torch.from_numpy(np.ascontiguousarray(codes, dtype=np.uint8)).view(torch.float8_e4m3fn).float().numpy()


def nvfp4_encode(w):
    """w [E, rows, cols] f32 -> (weight U8 [E, rows, cols/2], scale e4m3 codes [E, rows, cols/16], amax [E], denom)."""
    e, rows, cols = w.shape
    amax = np.abs(w).reshape(e, -1).max(axis=1).astype(np.float32)
    factor = (amax / np.float32(NVFP4_DENOM)).astype(np.float32)
    blocks = w.reshape(e, rows, cols // 16, 16)
    bscale = np.abs(blocks).max(axis=-1) / 6.0 / factor[:, None, None]
    scale_codes = e4m3_codes(bscale.astype(np.float32))
    eff = e4m3_values(scale_codes)[..., None] * factor[:, None, None, None]
    eff = np.where(eff == 0, 1.0, eff)
    q = blocks / eff
    mag = np.abs(q)[..., None]
    idx = np.abs(mag - E2M1).argmin(axis=-1).astype(np.uint8)  # nearest e2m1 magnitude
    code = idx | np.where(q < 0, 8, 0).astype(np.uint8)        # sign in bit 3
    code = code.reshape(e, rows, cols)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).astype(np.uint8)  # even element LOW nibble
    return packed, scale_codes, amax, np.float32(NVFP4_DENOM)


def mxfp8_encode(w):
    """OCP-style MXFP8: e = floor(log2(group amax)) - 8, codes = e4m3(w / 2^e) (clips up to 512 -> 448)."""
    e, rows, cols = w.shape
    blocks = w.reshape(e, rows, cols // 32, 32)
    amax = np.abs(blocks).max(axis=-1)
    exp = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 8, -127).astype(np.int64)
    exp = np.clip(exp, -127, 127)
    scale = (exp + 127).astype(np.uint8)
    codes = e4m3_codes(blocks / np.exp2(exp.astype(np.float32))[..., None]).reshape(e, rows, cols)
    return codes, scale


def inject_mxfp8_edges(codes, scale):
    """Groups that exercise the raw copy: a zero group with a nonzero scale, subnormal / -0 codes at scale byte 0,
    a group whose largest code is far below 224 (a re-encode would choose another exponent), the largest code 0x7E."""
    codes[0, 0, 0:32] = 0
    scale[0, 0, 0] = 130
    codes[0, 1, 0:32] = np.array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x81, 0x87, 0x08, 0x88] +
                                 [0x00] * 20, dtype=np.uint8)
    scale[0, 1, 0] = 0
    codes[0, 2, 0:32] = (np.arange(32) % 7 + 1).astype(np.uint8) | np.where(np.arange(32) % 2, 0x80, 0).astype(np.uint8)
    scale[0, 2, 0] = 122
    codes[0, 3, 0:32] = 0x38  # 1.0
    codes[0, 3, 5] = 0x7E     # 448: the group's max code is the largest finite one
    scale[0, 3, 0] = 117


# ---- safetensors writer (bf16 / f8 are not numpy dtypes) --------------------------------------------------------

class Writer:
    def __init__(self):
        self.items = []  # (name, dtype, shape, bytes)

    def bf16(self, name, x):
        x = np.asarray(x, dtype=np.float32)
        self.items.append((name, "BF16", list(x.shape), to_bf16_bytes(x)))

    def f32(self, name, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.items.append((name, "F32", list(x.shape), x.tobytes()))

    def raw(self, name, dtype, x):
        x = np.ascontiguousarray(x, dtype=np.uint8)
        self.items.append((name, dtype, list(x.shape), x.tobytes()))

    def save(self, path):
        header, blob = {"__metadata__": {"format": "pt"}}, bytearray()
        for name, dtype, shape, data in sorted(self.items, key=lambda it: it[0]):
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
            blob += data
        h = json.dumps(header, separators=(",", ":")).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            f.write(bytes(blob))


def build(t=TINY, seed=0x1CE2):
    rng = np.random.default_rng(seed)
    H, I, E = t["hidden"], t["inter"], t["experts"]
    qd, kvd = t["heads"] * t["head_dim"], t["kv_heads"] * t["head_dim"]
    w = Writer()

    def mat(n, k, std=0.08):
        return rng.normal(0.0, std, (n, k)).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.normal(size=n)).astype(np.float32)

    L = "model.llm."
    w.bf16(L + "embed.weight", rng.normal(0.0, 0.5, (t["vocab"], H)))
    w.bf16(L + "embed_norm.weight", norm(H))
    w.bf16(L + "norm.weight", norm(H))
    w.bf16(L + "unembed.weight", mat(t["vocab"], H, 0.1))
    for i in range(t["layers"]):
        full = is_full(i)
        if i >= 1:
            w.f32(L + f"attn_res.attn_queries.{i - 1}", rng.normal(0.0, 0.3, (3, H) if full else (H,)))
        w.f32(L + f"attn_res.mlp_queries.{i}", rng.normal(0.0, 0.3, (H,)))
        p = L + f"layers.{i}."
        w.bf16(p + "attn.wq_du.weight", mat(qd, H))
        w.bf16(p + "attn.gate_attn.weight", mat(qd, H))
        w.bf16(p + "attn.wk_dv.weight", mat(kvd, H))
        w.bf16(p + "attn.wv_dv.weight", mat(kvd, H))
        w.bf16(p + "attn.wo_ud.weight", mat(H, qd))
        w.bf16(p + "attn.wr_du.weight", mat(t["heads"] * t["d_rel"], H))
        w.bf16(p + "attn.attn_sink_proj.weight", mat(t["heads"], H))
        w.bf16(p + "attn.q_l2_clip.scale", norm(t["heads"] * t["head_dim"]).reshape(t["heads"], t["head_dim"]))
        w.bf16(p + "attn.rel_l2_clip.scale", norm(t["heads"] * t["d_rel"]).reshape(t["heads"], t["d_rel"]))
        w.bf16(p + "attn.rel_logits_proj.proj", rng.normal(0.0, 0.3, (t["d_rel"], t["rel_extent"])))
        if full:
            for kv in "kv":
                w.bf16(p + f"attn.{kv}_sconv.static_w", rng.normal(0.0, 0.4, (t["conv"], kvd)))
                w.bf16(p + f"attn.{kv}_sconv.weight_proj", mat(t["kv_heads"] * t["conv"], H, 0.1))
            for n in "qkv":
                w.bf16(p + f"{n}_input_norm.weight", norm(H))
        else:
            for kv in "kv":
                w.bf16(p + f"attn.{kv}_sconv.weight", rng.normal(0.0, 0.4, (kvd, 1, t["conv"])))
            w.bf16(p + "attn_norm.weight", norm(H))
        w.bf16(p + "mlp_norm.weight", norm(H))
        w.bf16(p + "attn_sconv.weight", rng.normal(0.0, 0.4, (H, 1, t["conv"])))
        w.bf16(p + "mlp_sconv.weight", rng.normal(0.0, 0.4, (H, 1, t["conv"])))
        if t["loop_start"] <= i < t["loop_end"]:
            for c in ("attn", "k", "v", "mlp"):
                w.bf16(p + f"controllers.{c}.down_weight", mat(t["rank"], 2 * H, 0.1))
                w.bf16(p + f"controllers.{c}.forget_weight", mat(H, t["rank"], 0.2))
                w.bf16(p + f"controllers.{c}.gain_weight", mat(H, t["rank"], 0.2))
        w.bf16(p + "mlp.gate.weight", mat(E, H, 0.2))
        w.f32(p + "mlp.gate.bias", rng.normal(0.0, 0.1, (E,)))
        # experts: f32 source weights -> the checkpoint encoding (rows of w13 INTERLEAVED gate, up, gate, up, ...)
        w13 = rng.normal(0.0, 0.12, (E, 2 * I, H)).astype(np.float32)
        w2 = rng.normal(0.0, 0.12, (E, H, I)).astype(np.float32)
        for name, src in (("w13", w13), ("w2", w2)):
            if i < t["nvfp4_layers"]:
                packed, sc, amax, denom = nvfp4_encode(src)
                w.raw(p + f"mlp.experts.{name}_weight", "U8", packed)
                w.raw(p + f"mlp.experts.{name}_weight_scale", "F8_E4M3", sc)
                w.f32(p + f"mlp.experts.{name}_weight_amax", amax)
                w.f32(p + f"mlp.experts.{name}_weight_outer_scale_denominator", np.array([denom], dtype=np.float32))
            else:
                codes, scale = mxfp8_encode(src)
                if i == t["nvfp4_layers"] and name == "w13":
                    inject_mxfp8_edges(codes, scale)
                w.raw(p + f"mlp.experts.{name}_weight", "F8_E4M3", codes)
                w.raw(p + f"mlp.experts.{name}_weight_scale", "U8", scale)
    nl = t["loop_end"] - t["loop_start"]
    w.f32(L + "attn_res.output_query", rng.normal(0.0, 0.3, (H,)))
    w.f32(L + "middle_cycle_cross_pass_queries", rng.normal(0.0, 0.3, (2 * nl, H)))
    w.f32(L + "middle_cycle_cross_pass_kv_queries", rng.normal(0.0, 0.3, (2 * nl, H)))
    # allow-listed dummies (a real checkpoint has 422 model.visual.* tensors)
    w.bf16("model.visual.backbone.ln_pre.weight", norm(16))
    w.bf16("model.visual.backbone.ln_pre.bias", norm(16))
    w.bf16("model.visual.proj_out.weight", mat(H, 16))
    return w


def reference(out, t=TINY):
    import transformers
    from transformers import Inkling2LoopForCausalLM

    cfg = transformers.AutoConfig.from_pretrained(out).text_config
    model = Inkling2LoopForCausalLM.from_pretrained(out, config=cfg, dtype=torch.float32, output_loading_info=True)
    model, info = model
    missing, unexpected = sorted(info["missing_keys"]), sorted(info["unexpected_keys"])
    print("missing:", missing)
    print("unexpected:", unexpected)
    assert not missing, missing
    assert all("visual" in k for k in unexpected), unexpected  # text-only model: only the ViT dummies are unexpected
    model.eval()
    g = np.random.default_rng(7)
    tokens = g.integers(0, t["unpadded"], size=40).tolist()
    with torch.no_grad():
        logits = model(torch.tensor([tokens])).logits[0].float().numpy()
    assert logits.shape == (40, t["unpadded"]) and np.isfinite(logits).all()
    np.save(os.path.join(out, "logits.npy"), logits.astype(np.float32))
    with open(os.path.join(out, "tokens.json"), "w") as f:
        json.dump({"tokens": tokens, "note": "Inkling2LoopForCausalLM float32 logits [40, unpadded_vocab] -> logits.npy"}, f)
    print("logits", logits.shape, "range", logits.min(), logits.max(), "transformers", transformers.__version__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build().save(os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(config(), f, indent=2)
    print("wrote", a.out, os.path.getsize(os.path.join(a.out, "model.safetensors")), "bytes")
    if a.reference:
        reference(a.out)


if __name__ == "__main__":
    main()
