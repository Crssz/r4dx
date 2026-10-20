"""Random-weight `tiny-q35moe`: the real Ornith-1.5-122B (`qwen3_5_moe`) geometry at 4 layers, in the real checkpoint format
(docs/qwen35moe-plan.md M2).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\tiny.py build [--out E:\\models\\r4dx\\tiny-q35moe]

Config: the real `text_config` with num_hidden_layers 4 (GDN, GDN, GDN, full attention), num_experts 16 (top-8),
vocab_size 4096, eos_token_id 4095. Everything else (hidden 3072, GDN 16/64 x 128, attention 32/2 x 256, expert width
1024, shared expert 1024, rope) is the real value, so the kernels see the real shapes. The vision tower is built by HF
(depth 1) but its weights are not saved, and neither is the MTP head: the tiny checkpoint is text only.

Output directory (`<out>`):
    hf/config.json                       real key names, dtype bfloat16 (vision depth 1)
    hf/generation_config.json
    hf/model-00001-of-00001.safetensors  every text tensor under the real checkpoint's names, bf16, FUSED experts:
                                           model.language_model.embed_tokens.weight
                                           model.language_model.layers.{i}.mlp.experts.gate_up_proj   [E, 2 * 1024, 3072]
                                           model.language_model.layers.{i}.mlp.experts.down_proj      [E, 3072, 1024]
                                           model.language_model.layers.{i}.mlp.gate.weight, .shared_expert.*, ...
                                           model.language_model.norm.weight, lm_head.weight
    hf/model.safetensors.index.json
    DONE                                 marker written last

Weights are drawn from a seeded generator; scales are chosen so the router's top-8 of 16 differs per token, the GDN gates
are non-degenerate and the logits are not flat. `load_hf_model(dir)` loads the result with transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

SEED = 20261020
DEFAULT_OUT = Path(r"E:\models\r4dx\tiny-q35moe")
REAL_CONFIG = Path(r"E:\models\r4dx\Ornith-1.5-122B\config.json")
TINY_EOS = 4095
TINY_VOCAB = 4096
TINY_LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
TINY_EXPERTS = 16

# (mean, std) per tensor class.
SCALES = {
    "embed_tokens": (0.0, 0.03),
    "lm_head": (0.0, 0.05),
    "router": (0.0, 0.06),
    "shared_gate_up": (0.0, 0.012),
    "shared_down": (0.0, 0.012),
    "shared_expert_gate": (0.0, 0.02),
    "expert_gate_up": (0.0, 0.0145),
    "expert_down": (0.0, 0.012),
    "gdn_qkv": (0.0, 0.0195),
    "gdn_z": (0.0, 0.0185),
    "gdn_out": (0.0, 0.0185),
    "gdn_ab": (0.0, 0.02),
    "gdn_conv": (0.0, 0.03),
    "gdn_norm": (1.0, 0.03),       # RMSNormGated: plain weight, ones-centred
    "dt_bias": (-3.0, 1.5),
    "attn_q": (0.0, 0.0179),
    "attn_k": (0.0, 0.0179),
    "attn_v": (0.0, 0.0179),
    "attn_o": (0.0, 0.0179),
    "qk_norm": (0.0, 0.1),         # (1 + w) form
    "layer_norm": (0.0, 0.1),      # (1 + w) form
}


def build_config() -> dict:
    cfg = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    t = cfg["text_config"]
    t["num_hidden_layers"] = len(TINY_LAYERS)
    t["layer_types"] = list(TINY_LAYERS)
    t["num_experts"] = TINY_EXPERTS
    t["vocab_size"] = TINY_VOCAB
    t["eos_token_id"] = TINY_EOS
    cfg["vision_config"]["depth"] = 1
    return cfg


def _normal(shape, key, gen):
    mean, std = SCALES[key]
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * std + mean).to(torch.bfloat16)


def draw_tensors(cfg: dict, seed: int = SEED) -> dict[str, torch.Tensor]:
    """Every text tensor of the checkpoint under its real name, bf16 (A_log / dt_bias too: HF keeps them as stored)."""
    t = cfg["text_config"]
    gen = torch.Generator().manual_seed(seed)
    H = t["hidden_size"]
    E = t["num_experts"]
    I = t["moe_intermediate_size"]
    SI = t["shared_expert_intermediate_size"]
    key_dim = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    val_dim = t["linear_num_value_heads"] * t["linear_value_head_dim"]
    nv = t["linear_num_value_heads"]
    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    p = "model.language_model."
    out: dict[str, torch.Tensor] = {}
    out[p + "embed_tokens.weight"] = _normal((t["vocab_size"], H), "embed_tokens", gen)
    for i, kind in enumerate(t["layer_types"]):
        L = f"{p}layers.{i}."
        out[L + "input_layernorm.weight"] = _normal((H,), "layer_norm", gen)
        out[L + "post_attention_layernorm.weight"] = _normal((H,), "layer_norm", gen)
        if kind == "linear_attention":
            a = L + "linear_attn."
            out[a + "in_proj_qkv.weight"] = _normal((2 * key_dim + val_dim, H), "gdn_qkv", gen)
            out[a + "in_proj_z.weight"] = _normal((val_dim, H), "gdn_z", gen)
            out[a + "in_proj_b.weight"] = _normal((nv, H), "gdn_ab", gen)
            out[a + "in_proj_a.weight"] = _normal((nv, H), "gdn_ab", gen)
            out[a + "conv1d.weight"] = _normal((2 * key_dim + val_dim, 1, t["linear_conv_kernel_dim"]), "gdn_conv", gen)
            out[a + "A_log"] = torch.log(torch.rand(nv, generator=gen) * 15.99 + 0.01).to(torch.bfloat16)
            out[a + "dt_bias"] = _normal((nv,), "dt_bias", gen)
            out[a + "norm.weight"] = _normal((t["linear_value_head_dim"],), "gdn_norm", gen)
            out[a + "out_proj.weight"] = _normal((H, val_dim), "gdn_out", gen)
        else:
            a = L + "self_attn."
            out[a + "q_proj.weight"] = _normal((2 * heads * hd, H), "attn_q", gen)  # per-head [q | gate]
            out[a + "k_proj.weight"] = _normal((kv * hd, H), "attn_k", gen)
            out[a + "v_proj.weight"] = _normal((kv * hd, H), "attn_v", gen)
            out[a + "o_proj.weight"] = _normal((H, heads * hd), "attn_o", gen)
            out[a + "q_norm.weight"] = _normal((hd,), "qk_norm", gen)
            out[a + "k_norm.weight"] = _normal((hd,), "qk_norm", gen)
        m = L + "mlp."
        out[m + "gate.weight"] = _normal((E, H), "router", gen)
        out[m + "experts.gate_up_proj"] = _normal((E, 2 * I, H), "expert_gate_up", gen)
        out[m + "experts.down_proj"] = _normal((E, H, I), "expert_down", gen)
        out[m + "shared_expert.gate_proj.weight"] = _normal((SI, H), "shared_gate_up", gen)
        out[m + "shared_expert.up_proj.weight"] = _normal((SI, H), "shared_gate_up", gen)
        out[m + "shared_expert.down_proj.weight"] = _normal((H, SI), "shared_down", gen)
        out[m + "shared_expert_gate.weight"] = _normal((1, H), "shared_expert_gate", gen)
    out[p + "norm.weight"] = _normal((H,), "layer_norm", gen)
    out["lm_head.weight"] = _normal((t["vocab_size"], H), "lm_head", gen)
    return out


def build(out_dir: Path, force: bool = False) -> Path:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    hf = out_dir / "hf"
    if (out_dir / "DONE").exists() and not force:
        print(f"[tiny-q35moe] {out_dir} already built (use --force)")
        return hf
    if hf.exists():
        shutil.rmtree(hf)
    hf.mkdir(parents=True)
    (out_dir / "DONE").unlink(missing_ok=True)
    cfg = build_config()
    (hf / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    gen_cfg = {"bos_token_id": TINY_EOS, "eos_token_id": [TINY_EOS - 1, TINY_EOS], "pad_token_id": TINY_EOS, "do_sample": False}
    (hf / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2), encoding="utf-8")
    tensors = draw_tensors(cfg)
    fname = "model-00001-of-00001.safetensors"
    save_file(tensors, str(hf / fname), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": {k: fname for k in tensors}}
    (hf / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (out_dir / "DONE").write_text("ok\n")
    print(f"[tiny-q35moe] wrote {len(tensors)} tensors, {total / 2**30:.2f} GiB, to {hf}")
    return hf


def load_hf_model(hf_dir: str | Path):
    """The tiny checkpoint as an HF Qwen3_5MoeForConditionalGeneration (bf16, CPU, sdpa attention, eager experts)."""
    from safetensors.torch import load_file
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    hf_dir = Path(hf_dir)
    cfg = Qwen3_5MoeConfig.from_pretrained(str(hf_dir))
    cfg._attn_implementation = "sdpa"
    for sub in (cfg.text_config,):
        try:
            sub.experts_implementation = "eager"
        except Exception:
            pass
    with torch.device("meta"):
        model = Qwen3_5MoeForConditionalGeneration(cfg)
    sd = load_file(str(hf_dir / "model-00001-of-00001.safetensors"))
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("model.visual.")]
    if bad or unexpected:
        raise RuntimeError(f"tiny-q35moe load: missing {bad[:5]} unexpected {list(unexpected)[:5]}")
    # buffers that to_empty() left uninitialised (rotary inv_freq)
    for mod in model.modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "rope_init_fn"):
            inv, scale = mod.rope_init_fn(mod.config, "cpu")
            mod.inv_freq = inv
            mod.original_inv_freq = inv
            mod.attention_scaling = scale
    model.eval()
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(Path(args.out), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
