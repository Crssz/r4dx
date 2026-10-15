"""tools/reference/lfm2/make_tiny_lfm2.py -- a TINY random lfm2_moe checkpoint with the real tensor names (lane F's own
generator for the quant-tool tests; lane C's tools/convert_ref/make_tiny_lfm2.py is the converter test's, this one
is independent so the lanes do not collide).

    python make_tiny_lfm2.py --out D:\\r4dx-scratch\\lfm2\\tiny\\ckpt [--expert-style per-expert|fused] [--seed N]

Geometry (every K and N a multiple of 128, so the trellis quantizer accepts all of it): hidden 256, 4 heads x
head_dim 64 (q 256, kv 2 x 64 = 128), 4 layers = conv+dense, conv+dense, attention+MoE, conv+MoE, 8 experts top-2 with
moe_intermediate 256, dense intermediate 512, vocab 512 (tied head). Key layout = the REAL checkpoint's
(docs/lfm2-plan.md 2.1 / lane_common FACTS): `model.layers.N.feed_forward.experts.E.{w1,w2,w3}.weight` per expert,
`feed_forward.gate.weight`, fp32 `feed_forward.expert_bias`. `--expert-style fused` writes the transformers-5 3-D
`experts.gate_up_proj` / `experts.down_proj` instead (the loader must handle both).

Two shards + model.safetensors.index.json (like the real one), bf16 except expert_bias (F32). The weights are random,
but the router is given a skewed bias so some experts are RARE (the Hessian shrinkage path gets exercised).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 64: NOT in config.json (the real config has no head_dim either)
N_LAYERS = 4
LAYER_TYPES = ["conv", "conv", "full_attention", "conv"]
NUM_DENSE = 2
N_EXPERTS = 8
TOP_K = 2
MOE_I = 256
DENSE_I = 512
VOCAB = 512
CONV_L = 3

CONFIG = {
    "architectures": ["Lfm2MoeForCausalLM"],
    "bos_token_id": 1, "conv_L_cache": CONV_L, "conv_bias": False, "dtype": "bfloat16", "eos_token_id": 2,
    "hidden_size": HIDDEN, "initializer_range": 0.02, "intermediate_size": DENSE_I,
    "layer_types": LAYER_TYPES, "max_position_embeddings": 4096, "model_type": "lfm2_moe",
    "moe_intermediate_size": MOE_I, "norm_eps": 1e-05, "norm_topk_prob": True,
    "num_attention_heads": HEADS, "num_dense_layers": NUM_DENSE, "num_experts": N_EXPERTS,
    "num_experts_per_tok": TOP_K, "num_hidden_layers": N_LAYERS, "num_key_value_heads": KV_HEADS,
    "pad_token_id": 0, "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "routed_scaling_factor": 1.0, "tie_word_embeddings": True, "transformers_version": "5.12.1",
    "use_cache": True, "use_expert_bias": True, "vocab_size": VOCAB,
}


def tensors(seed: int, style: str) -> dict:
    """name -> numpy array (float32; the writer casts to bf16 except expert_bias)."""
    rng = np.random.default_rng(seed)

    def w(*shape, std=0.05):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    t = {"model.embed_tokens.weight": w(VOCAB, HIDDEN, std=0.5),
         "model.embedding_norm.weight": norm(HIDDEN)}
    for i, lt in enumerate(LAYER_TYPES):
        p = f"model.layers.{i}."
        t[p + "operator_norm.weight"] = norm(HIDDEN)
        t[p + "ffn_norm.weight"] = norm(HIDDEN)
        if lt == "conv":
            t[p + "conv.in_proj.weight"] = w(3 * HIDDEN, HIDDEN)
            t[p + "conv.conv.weight"] = w(HIDDEN, 1, CONV_L, std=0.3)
            t[p + "conv.out_proj.weight"] = w(HIDDEN, HIDDEN)
        else:
            t[p + "self_attn.q_proj.weight"] = w(HEADS * HEAD_DIM, HIDDEN)
            t[p + "self_attn.k_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
            t[p + "self_attn.v_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
            t[p + "self_attn.out_proj.weight"] = w(HIDDEN, HEADS * HEAD_DIM)
            t[p + "self_attn.q_layernorm.weight"] = norm(HEAD_DIM)
            t[p + "self_attn.k_layernorm.weight"] = norm(HEAD_DIM)
        f = p + "feed_forward."
        if i < NUM_DENSE:
            t[f + "w1.weight"] = w(DENSE_I, HIDDEN)
            t[f + "w3.weight"] = w(DENSE_I, HIDDEN)
            t[f + "w2.weight"] = w(HIDDEN, DENSE_I)
        else:
            t[f + "gate.weight"] = w(N_EXPERTS, HIDDEN, std=0.3)
            # a skew: the last experts get a negative bias so they are rarely selected (rare-expert path)
            bias = np.zeros(N_EXPERTS, dtype=np.float32)
            bias[-2], bias[-1] = -0.45, -1.5
            bias[0] = 0.3
            t[f + "expert_bias"] = bias
            gate = [w(MOE_I, HIDDEN) for _ in range(N_EXPERTS)]
            up = [w(MOE_I, HIDDEN) for _ in range(N_EXPERTS)]
            down = [w(HIDDEN, MOE_I) for _ in range(N_EXPERTS)]
            if style == "per-expert":
                for e in range(N_EXPERTS):
                    t[f"{f}experts.{e}.w1.weight"] = gate[e]
                    t[f"{f}experts.{e}.w3.weight"] = up[e]
                    t[f"{f}experts.{e}.w2.weight"] = down[e]
            else:
                t[f + "experts.gate_up_proj"] = np.stack([np.concatenate([g, u], 0) for g, u in zip(gate, up)])
                t[f + "experts.down_proj"] = np.stack(down)
    return t


def write(out: Path, seed: int = 20261014, style: str = "per-expert") -> Path:
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    t = tensors(seed, style)
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    weight_map, total = {}, 0
    for fn, ns in shards.items():
        d = {}
        for n in ns:
            a = torch.from_numpy(t[n])
            d[n] = a.float().contiguous() if n.endswith("expert_bias") else a.to(torch.bfloat16).contiguous()
            total += d[n].numel() * d[n].element_size()
            weight_map[n] = fn
        save_file(d, str(out / fn), metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--expert-style", choices=["per-expert", "fused"], default="per-expert")
    a = ap.parse_args()
    out = write(a.out, a.seed, a.expert_style)
    print(f"[tiny-lfm2] wrote {out} ({a.expert_style} experts, seed {a.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
