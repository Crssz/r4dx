#!/usr/bin/env python3
"""Dependency-free (stdlib only: struct, json, random -- NO numpy/torch/safetensors) generator for a tiny synthetic
dense Llama checkpoint, the input of r4dx-convert's Llama branch tests (docs/nanbeige-plan.md 4, N1-3).

    python make_tiny_llama.py --out-dir <dir> [--seed 1] [--shards 1]

Writes `<dir>/model.safetensors` (or `model-0000i-of-0000N.safetensors` + `model.safetensors.index.json` with
--shards N, like the real 14-shard checkpoint) and the FLAT `<dir>/config.json`.

Tensor names are the stock transformers LlamaForCausalLM state_dict (UNVERIFIED against the Nanbeige4.3-33B header
until N0-2): model.embed_tokens.weight, model.norm.weight, an UNTIED lm_head.weight; per layer input_layernorm /
post_attention_layernorm, self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj. All values are bf16-exact.

Dimensions keep every TP=2 rule of LlamaConfig::Shard legal (row-parallel K % 256 per rank, rows % 16) at the REAL
model's attention geometry: head_dim 128 and GQA 7 (28 q heads / 4 kv heads -> 14 / 2 per rank, o_proj K 1792 per
rank = 7 x 256), so the N5 h128_gqa7 kernel can be tested on it. hidden 1792 is deliberately != heads x head_dim
(3584): the config's explicit head_dim key must be honoured (transformers 5 still requires hidden % heads == 0:
1792 = 28 x 64; and 1792 = 14 x 128 keeps every K whole w8a16 groups). intermediate 512, vocab 512, 3 layers.
"""
import argparse
import json
import random
import struct
from array import array
from pathlib import Path

HIDDEN, HEADS, KV, HEAD_DIM = 1792, 28, 4, 128
FFN, VOCAB, LAYERS = 512, 512, 3


def f32_to_bf16(f):
    """Top 16 bits of an IEEE-754 fp32, round-to-nearest-even (r4dx::core::FloatToBf16)."""
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return (bits >> 16) | 0x0040
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16


def tensor_specs():
    """[(name, dtype, shape, kind)] with kind in {"w", "norm"}."""
    specs = []
    for i in range(LAYERS):
        L = f"model.layers.{i}."
        specs.append((L + "input_layernorm.weight", "BF16", [HIDDEN], "norm"))
        specs.append((L + "post_attention_layernorm.weight", "BF16", [HIDDEN], "norm"))
        specs.append((L + "self_attn.q_proj.weight", "BF16", [HEADS * HEAD_DIM, HIDDEN], "w"))
        specs.append((L + "self_attn.k_proj.weight", "BF16", [KV * HEAD_DIM, HIDDEN], "w"))
        specs.append((L + "self_attn.v_proj.weight", "BF16", [KV * HEAD_DIM, HIDDEN], "w"))
        specs.append((L + "self_attn.o_proj.weight", "BF16", [HIDDEN, HEADS * HEAD_DIM], "w"))
        specs.append((L + "mlp.gate_proj.weight", "BF16", [FFN, HIDDEN], "w"))
        specs.append((L + "mlp.up_proj.weight", "BF16", [FFN, HIDDEN], "w"))
        specs.append((L + "mlp.down_proj.weight", "BF16", [HIDDEN, FFN], "w"))
    specs.append(("model.embed_tokens.weight", "BF16", [VOCAB, HIDDEN], "w"))
    specs.append(("model.norm.weight", "BF16", [HIDDEN], "norm"))
    specs.append(("lm_head.weight", "BF16", [VOCAB, HIDDEN], "w"))
    return specs


def config():
    # The real config's keys (transformers 4.x spelling: rope_theta + rope_scaling null), tiny sizes.
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": FFN,
        "max_position_embeddings": 4096,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 120000000,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": False,
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
    ap.add_argument("--list", action="store_true", help="print the tensor names, dtypes and shapes and exit")
    a = ap.parse_args()
    specs = tensor_specs()
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
        vals = array("H", rng.choices(pool_n if kind == "norm" else pool_w, k=n))
        entries.append((name, dtype, shape, vals.tobytes()))
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
    print(f"wrote {a.out_dir} ({len(specs)} tensors, {total} data bytes, {a.shards} shard(s))")


if __name__ == "__main__":
    main()
