#!/usr/bin/env python3
"""Dependency-free (stdlib only: struct, json, random -- NO numpy/torch/safetensors) generator for a tiny synthetic
lfm2_moe checkpoint, the input of r4dx-convert's LFM2 branch tests (docs/lfm2-plan.md 4, L1-2 / L1-5).

    python make_tiny_lfm2.py --out-dir <dir> [--seed 1] [--shards 1] [--experts-style per-expert|fused]

Writes `<dir>/model.safetensors` (or `model-0000i-of-0000N.safetensors` + `model.safetensors.index.json` with
--shards N, like the real 10-shard checkpoint) and the FLAT `<dir>/config.json` (lfm2_moe has no text_config).

Layers: 4 = conv + dense FFN, conv + dense FFN, attention + MoE, conv + MoE (attention at i % 4 == 2, the real
pattern; num_dense_layers 2). The real tensor names, exactly as in the LiquidAI/LFM2.5-24B-A2B header (7610 tensors):
model.embed_tokens.weight, model.embedding_norm.weight (the FINAL norm), NO lm_head (tied); per layer
operator_norm / ffn_norm; conv layers conv.in_proj / conv.conv ([H, 1, 3]) / conv.out_proj; attention layers
self_attn.{q,k,v,out}_proj + {q,k}_layernorm; dense layers feed_forward.{w1,w2,w3}; MoE layers feed_forward.gate,
feed_forward.expert_bias (F32 [E]) and the PER-EXPERT feed_forward.experts.{e}.{w1,w2,w3}.weight
(`--experts-style fused` writes transformers 5's fused experts.gate_up_proj [E, 2I, H] / experts.down_proj [E, H, I]
instead, which r4dx-convert must refuse by name). All bf16 values are bf16-exact; expert_bias is F32.

Dimensions keep every TP=2 rule of Lfm2Config::Shard legal (K % 256 per rank, rows % 16): hidden 512, 8 heads x 64
(head_dim = hidden / heads, 2 kv heads, GQA 4), dense intermediate 512, 8 experts of 512, top-2, vocab 64.
tests/convert/test_lfm2_layout.cpp builds the same checkpoint in C++ (it must not need Python to run); this script is
for manual runs (`r4dx-convert --input <dir> --output tiny.r4dx --layouts bf16 --lm-head bf16`) and `--list` cross-checks.
"""
import argparse
import json
import random
import struct
from array import array
from pathlib import Path

HIDDEN, HEADS, KV, HEAD_DIM = 512, 8, 2, 64
DENSE_I, MOE_I, EXPERTS, TOPK, VOCAB, CONV_L = 512, 512, 8, 2, 64, 3
LAYERS, NUM_DENSE = 4, 2


def f32_to_bf16(f):
    """Top 16 bits of an IEEE-754 fp32, round-to-nearest-even (r4dx::core::FloatToBf16)."""
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return (bits >> 16) | 0x0040
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16


def is_attn(i):
    return i % 4 == 2


def tensor_specs(experts_style="per-expert"):
    """[(name, dtype, shape, kind)] with kind in {"w", "norm", "bias"}."""
    specs = []
    for i in range(LAYERS):
        L = f"model.layers.{i}."
        specs.append((L + "operator_norm.weight", "BF16", [HIDDEN], "norm"))
        specs.append((L + "ffn_norm.weight", "BF16", [HIDDEN], "norm"))
        if is_attn(i):
            specs.append((L + "self_attn.q_proj.weight", "BF16", [HEADS * HEAD_DIM, HIDDEN], "w"))
            specs.append((L + "self_attn.k_proj.weight", "BF16", [KV * HEAD_DIM, HIDDEN], "w"))
            specs.append((L + "self_attn.v_proj.weight", "BF16", [KV * HEAD_DIM, HIDDEN], "w"))
            specs.append((L + "self_attn.out_proj.weight", "BF16", [HIDDEN, HEADS * HEAD_DIM], "w"))
            specs.append((L + "self_attn.q_layernorm.weight", "BF16", [HEAD_DIM], "norm"))
            specs.append((L + "self_attn.k_layernorm.weight", "BF16", [HEAD_DIM], "norm"))
        else:
            specs.append((L + "conv.in_proj.weight", "BF16", [3 * HIDDEN, HIDDEN], "w"))
            specs.append((L + "conv.conv.weight", "BF16", [HIDDEN, 1, CONV_L], "w"))
            specs.append((L + "conv.out_proj.weight", "BF16", [HIDDEN, HIDDEN], "w"))
        if i < NUM_DENSE:
            specs.append((L + "feed_forward.w1.weight", "BF16", [DENSE_I, HIDDEN], "w"))
            specs.append((L + "feed_forward.w2.weight", "BF16", [HIDDEN, DENSE_I], "w"))
            specs.append((L + "feed_forward.w3.weight", "BF16", [DENSE_I, HIDDEN], "w"))
        else:
            specs.append((L + "feed_forward.gate.weight", "BF16", [EXPERTS, HIDDEN], "w"))
            specs.append((L + "feed_forward.expert_bias", "F32", [EXPERTS], "bias"))
            if experts_style == "fused":
                specs.append((L + "feed_forward.experts.gate_up_proj", "BF16", [EXPERTS, 2 * MOE_I, HIDDEN], "w"))
                specs.append((L + "feed_forward.experts.down_proj", "BF16", [EXPERTS, HIDDEN, MOE_I], "w"))
            else:
                for e in range(EXPERTS):
                    E = L + f"feed_forward.experts.{e}."
                    specs.append((E + "w1.weight", "BF16", [MOE_I, HIDDEN], "w"))
                    specs.append((E + "w2.weight", "BF16", [HIDDEN, MOE_I], "w"))
                    specs.append((E + "w3.weight", "BF16", [MOE_I, HIDDEN], "w"))
    specs.append(("model.embed_tokens.weight", "BF16", [VOCAB, HIDDEN], "w"))
    specs.append(("model.embedding_norm.weight", "BF16", [HIDDEN], "norm"))
    return specs


def config():
    return {
        "architectures": ["Lfm2MoeForCausalLM"],
        "bos_token_id": 124894,
        "conv_L_cache": CONV_L,
        "conv_bias": False,
        "dtype": "bfloat16",
        "eos_token_id": 124900,
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE_I,
        "layer_types": ["full_attention" if is_attn(i) else "conv" for i in range(LAYERS)],
        "max_position_embeddings": 4096,
        "model_type": "lfm2_moe",
        "moe_intermediate_size": MOE_I,
        "norm_eps": 1e-05,
        "norm_topk_prob": True,
        "num_attention_heads": HEADS,
        "num_dense_layers": NUM_DENSE,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV,
        "pad_token_id": 124893,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True,
        "use_cache": True,
        "use_expert_bias": True,
        "vocab_size": VOCAB,
    }


def write_safetensors(path, entries):
    """entries: [(name, dtype, shape, bytes)]."""
    header, data = {}, bytearray()
    for name, dtype, shape, raw in entries:
        begin = len(data)
        data += raw
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(data)]}
    h = json.dumps(header).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + bytes(data))
    return len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1, help="split the tensors into N files + model.safetensors.index.json")
    ap.add_argument("--experts-style", choices=["per-expert", "fused"], default="per-expert")
    ap.add_argument("--list", action="store_true", help="print the tensor names, dtypes and shapes and exit")
    a = ap.parse_args()
    specs = tensor_specs(a.experts_style)
    if a.list:
        for name, dtype, shape, _ in specs:
            print(name, dtype, shape)
        return
    if a.out_dir is None:
        ap.error("--out-dir is required")
    rng = random.Random(a.seed)
    # A pool of bf16-exact values per kind, sampled per element (C-level random.choices): fast, still random.
    pool_w = [f32_to_bf16(0.05 * rng.gauss(0, 1)) for _ in range(4096)]
    pool_n = [f32_to_bf16(1.0 + 0.2 * rng.gauss(0, 1)) for _ in range(1024)]
    entries = []
    for name, dtype, shape, kind in specs:
        n = 1
        for d in shape:
            n *= d
        if kind == "bias":  # expert_bias: F32, small values around 0 (a selection bias)
            vals = array("f", [0.1 * rng.gauss(0, 1) for _ in range(n)])
            raw = vals.tobytes()
        else:
            vals = array("H", rng.choices(pool_n if kind == "norm" else pool_w, k=n))
            raw = vals.tobytes()
        entries.append((name, dtype, shape, raw))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    if a.shards <= 1:
        total = write_safetensors(a.out_dir / "model.safetensors", entries)
    else:
        weight_map = {}
        per = (len(entries) + a.shards - 1) // a.shards
        for s in range(a.shards):
            part = entries[s * per:(s + 1) * per]
            fn = f"model-{s + 1:05d}-of-{a.shards:05d}.safetensors"
            total += write_safetensors(a.out_dir / fn, part)
            for name, _, _, _ in part:
                weight_map[name] = fn
        (a.out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    (a.out_dir / "config.json").write_text(json.dumps(config(), indent=2))
    print(f"wrote {a.out_dir} ({len(specs)} tensors, {total} data bytes, {a.shards} shard(s), {a.experts_style})")


if __name__ == "__main__":
    main()
