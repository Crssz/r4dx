#!/usr/bin/env python3
"""Dependency-free (stdlib only: struct, json, random -- NO numpy/torch/safetensors) generator for a tiny
synthetic gemma4_unified checkpoint, the input of r4dx-convert's Gemma branch tests.

    python make_tiny_gemma.py --out-dir <dir> [--layers 2] [--seed 1]

Writes `<dir>/model.safetensors` (ONE file, no `model.safetensors.index.json`, like the real Huihui
checkpoint) and `<dir>/config.json`. Layers alternate sliding..., full at i % 6 == 5 by default; the
default 2 layers are one sliding + one full. Real tensor names (tools/reference/gemma/tensor_names.json),
including `layer_scalar` ([1]) and NO `v_proj` / `v_norm` on a full layer (attention_k_eq_v), plus the
non-text tensors (`model.vision_embedder.*`, `model.embed_vision.*`, `model.embed_audio.*`) at tiny shapes.
No `lm_head` (tied). All values are bf16-exact.

Dimensions are the smallest that keep every converter constraint of the real model: hidden 768 (rotation
block 256 x 3), 4 query heads, head_dim 64 (sliding) / 128 (full), 2 / 1 KV heads, intermediate 512 (the
q2ab mlp.down Hadamard block), vocab 64 -- so `--rotate q2ab`, `--layouts w4a16` and `--kv-calib` all work
on it. tests/convert/test_gemma_layout.cpp builds the same checkpoint in C++ (it must not need Python to
run); this script exists for manual runs (`r4dx-convert --input <dir> --output tiny.r4dx --layouts bf16
--lm-head bf16`) and to cross-check that file's tensor set (`--list`).
"""
import argparse
import json
import random
import struct
from pathlib import Path

HIDDEN, HEADS, KV_S, KV_F, HD_S, HD_F, INTER, VOCAB = 768, 4, 2, 1, 64, 128, 512, 64


def f32_to_bf16(f):
    """Top 16 bits of an IEEE-754 fp32, round-to-nearest-even (r4dx::core::FloatToBf16)."""
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return (bits >> 16) | 0x0040
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16


def bf16_exact(f):
    return struct.unpack("<f", struct.pack("<I", f32_to_bf16(f) << 16))[0]


def tensor_specs(layers):
    """[(name, shape, kind)] with kind in {"w", "norm", "scalar"}."""
    specs = []
    for i in range(layers):
        full = i % 6 == 5
        L = f"model.language_model.layers.{i}."
        hd, kv = (HD_F, KV_F) if full else (HD_S, KV_S)
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            specs.append((L + n + ".weight", [HIDDEN], "norm"))
        specs.append((L + "self_attn.q_proj.weight", [HEADS * hd, HIDDEN], "w"))
        specs.append((L + "self_attn.k_proj.weight", [kv * hd, HIDDEN], "w"))
        if not full:
            specs.append((L + "self_attn.v_proj.weight", [kv * hd, HIDDEN], "w"))
        specs.append((L + "self_attn.o_proj.weight", [HIDDEN, HEADS * hd], "w"))
        specs.append((L + "self_attn.q_norm.weight", [hd], "norm"))
        specs.append((L + "self_attn.k_norm.weight", [hd], "norm"))
        specs.append((L + "mlp.gate_proj.weight", [INTER, HIDDEN], "w"))
        specs.append((L + "mlp.up_proj.weight", [INTER, HIDDEN], "w"))
        specs.append((L + "mlp.down_proj.weight", [HIDDEN, INTER], "w"))
        specs.append((L + "layer_scalar", [1], "scalar"))
    specs.append(("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN], "w"))
    specs.append(("model.language_model.norm.weight", [HIDDEN], "norm"))
    specs.append(("model.vision_embedder.patch_ln1.weight", [16], "w"))
    specs.append(("model.vision_embedder.patch_dense.weight", [8, 16], "w"))
    specs.append(("model.vision_embedder.pos_embedding", [4, 2, 8], "w"))
    specs.append(("model.embed_vision.embedding_projection.weight", [8, 8], "w"))
    specs.append(("model.embed_audio.embedding_projection.weight", [8, 4], "w"))
    return specs


def config(layers):
    return {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "eos_token_id": [1, 106],
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": HIDDEN,
            "num_hidden_layers": layers,
            "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(layers)],
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_S,
            "num_global_key_value_heads": KV_F,
            "head_dim": HD_S,
            "global_head_dim": HD_F,
            "intermediate_size": INTER,
            "vocab_size": VOCAB,
            "attention_k_eq_v": True,
            "tie_word_embeddings": True,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="print the tensor names and shapes and exit")
    a = ap.parse_args()
    specs = tensor_specs(a.layers)
    if a.list:
        for name, shape, _ in specs:
            print(name, shape)
        return
    if a.out_dir is None:
        ap.error("--out-dir is required")
    rng = random.Random(a.seed)
    header, data = {}, bytearray()
    for name, shape, kind in specs:
        n = 1
        for d in shape:
            n *= d
        if kind == "norm":
            vals = [bf16_exact(1.0 + 0.2 * rng.gauss(0, 1)) for _ in range(n)]
        elif kind == "scalar":
            vals = [bf16_exact(0.5 + 0.25 * rng.random())]
        else:
            vals = [bf16_exact(0.05 * rng.gauss(0, 1)) for _ in range(n)]
        begin = len(data)
        for v in vals:
            data += struct.pack("<H", f32_to_bf16(v))
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [begin, len(data)]}
    h = json.dumps(header).encode("utf-8")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + bytes(data))
    (a.out_dir / "config.json").write_text(json.dumps(config(a.layers), indent=2))
    print(f"wrote {a.out_dir} ({len(specs)} tensors, {len(data)} data bytes)")


if __name__ == "__main__":
    main()
