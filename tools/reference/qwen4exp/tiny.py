"""Random-weight `tiny-q4exp`: the real qwen4_exp geometry at 4 layers, in the real checkpoint format (plan 4.11 item 2).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\tiny.py build [--out E:\\models\\r4dx\\tiny-q4exp]

Config: the real `text_config` of Swift-1.5-Qwen3.8-Flash-Next with num_hidden_layers 4 (GDN, GDN, GDN, QSA),
num_experts 16 (top-10), vocab_size 4096, ngram_vocab_size_base 2003, split_ngram_parts 4, indexer_budget 64
(ratio 4: top-16 blocks, dense-exact up to 67 tokens), ple_layer_ids [2] (layer index 1), eos_token_id 4095.
Everything else (hidden 2560, GDN 16/48 x 128, attention 24/2 x 256, expert width 640, hc 4, rope) is the real
value. The vision tower is built by HF (depth 1) but its weights are not saved: the checkpoint is text only and
HF re-initialises the visual tower on load, which does not touch the text path.

Output directory (`<out>`):
    hf/config.json                      real key names, `quantization_config` absent, dtype bfloat16
    hf/generation_config.json
    hf/model-00001-of-00001.safetensors every tensor of the real checkpoint's names, bf16 (no weight_packed):
                                          model.language_model.embed_tokens.weight
                                          model.language_model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
                                          model.language_model.layers.{i}.mlp.gate.weight, .shared_expert.*, ...
                                          ...attn_hyper_connection.* / mlp_hyper_connection.* (hc_norm, input_mix_*, block_inject_weight)
                                          model.language_model.layers.1.ple.* incl. the I64 buffers layer_multipliers,
                                            ngram_heads_offsets, ngram_heads_vocab_sizes
                                          model.language_model.hyper_connection_mixer.*, lm_head.weight
    hf/embedding-model-0000K-of-00002.safetensors  the 4 n-gram shards
                                          model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..3}.weight
    hf/model.safetensors.index.json     weight_map over all files (HF needs it: it finds the shards through it)
    sidecar.r4dx-ngram                  `ngram_sidecar.py repack` of hf/
    DONE                                marker written last

Weights are drawn from a seeded generator at the scales measured on the real checkpoint (see SCALES): router
logits spread so the top-10 of 16 differs per token, non-trivial hyper-connections, a non-zero PLE conv (HF
inits it to zeros), A_log / dt_bias as in the real GDN layers, (1 + w) norm weights near zero.

`load_hf_model(dir)` loads the result with transformers (eager experts: the per-expert loop the reference uses).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 20261010
DEFAULT_OUT = Path(r"E:\models\r4dx\tiny-q4exp")
REAL_REPO = Path(r"E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ")
TINY_EOS = 4095
TINY_VOCAB = 4096
NGRAM_PARTS = 4
EMBEDDING_FILES = 2

# (mean, std) per tensor class; measured on the real checkpoint layers 0-3 / 40 unless noted.
SCALES = {
    "embed_tokens": (0.0, 0.02),            # real 0.009; a little larger so the PLE/embedding path is not buried
    "lm_head": (0.0, 0.05),                 # real 0.013; wider so the 4096-way logits are not flat
    "hc_norm": (0.0, 0.4),                  # real std 0.4-1.0, (1 + w) form
    "hc_down": (0.0, 0.07),
    "hc_up": (0.0, 0.07),
    "hc_inject": (0.0, 0.03),
    "router": (0.0, 0.06),                  # real 0.02 over 512 experts; wider so 16 experts rank differently per token
    "shared_gate_up": (0.0, 0.012),
    "shared_down": (0.0, 0.012),
    "shared_expert_gate": (0.0, 0.008),
    "expert_gate": (0.0, 0.0135),
    "expert_up": (0.0, 0.0145),
    "expert_down": (0.0, 0.012),
    "gdn_qkv": (0.0, 0.0195),
    "gdn_z": (0.0, 0.0185),
    "gdn_out": (0.0, 0.0185),
    "gdn_ab": (0.0, 0.02),
    "gdn_conv": (0.0, 0.03),
    "gdn_norm": (1.0, 0.03),                # RMSNormGated: plain weight, ones-centred
    "dt_bias": (-3.0, 1.5),
    "attn_q": (0.0, 0.0179),
    "attn_k": (0.0, 0.0198),
    "attn_v": (0.0, 0.0206),
    "attn_o": (0.0, 0.019),
    "attn_qk_norm": (0.28, 0.1),            # (1 + w) form, real mean 0.28
    "indexer_qk": (0.0, 0.0287),
    "indexer_norm": (-0.04, 0.07),
    "ple_key": (0.0, 0.014),
    "ple_value": (0.0, 0.02),
    "ple_conv": (0.005, 0.02),              # HF inits it to zeros; the real one is not
    "ple_norm": (-0.1, 0.1),
    "ngram_table": (0.0, 0.03),             # real 0.0075; larger so a PLE contribution is visible in bf16
}


def build_config_dict(base: dict | None = None) -> dict:
    """config.json of the tiny model: the real one with the text_config overrides and no quantization_config."""
    if base is None:
        base = json.loads((REAL_REPO / "config.json").read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(base))
    cfg.pop("quantization_config", None)
    tc = cfg["text_config"]
    tc.update(
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_experts=16,
        num_experts_per_tok=10,
        vocab_size=TINY_VOCAB,
        ngram_vocab_size_base=2003,
        split_ngram_parts=NGRAM_PARTS,
        indexer_budget=64,
        ple_layer_ids=[2],
        eos_token_id=TINY_EOS,
        bos_token_id=TINY_EOS,
        dtype="bfloat16",
    )
    cfg["vision_config"]["depth"] = 1
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        cfg.pop(key, None)  # real ids (248053..) are outside the tiny vocab
    cfg["dtype"] = "bfloat16"
    return cfg


def build_generation_config() -> dict:
    return {"bos_token_id": TINY_EOS, "do_sample": False, "eos_token_id": [TINY_EOS], "pad_token_id": TINY_EOS}


def _normal(shape, mean_std, gen) -> torch.Tensor:
    mean, std = mean_std
    return torch.randn(shape, generator=gen, dtype=torch.float32) * std + mean


def _scale_key(name: str) -> str:
    """SCALES key of a text-model parameter name (without the `model.language_model.` prefix)."""
    rules = [
        (r"^embed_tokens\.weight$", "embed_tokens"),
        (r"hc_norm\.weight$", "hc_norm"),
        (r"input_mix_weight_down\.weight$", "hc_down"),
        (r"input_mix_weight_up\.weight$", "hc_up"),
        (r"block_inject_weight\.weight$", "hc_inject"),
        (r"mlp\.gate\.weight$", "router"),
        (r"shared_expert\.(gate|up)_proj\.weight$", "shared_gate_up"),
        (r"shared_expert\.down_proj\.weight$", "shared_down"),
        (r"shared_expert_gate\.weight$", "shared_expert_gate"),
        (r"linear_attn\.in_proj_qkv\.weight$", "gdn_qkv"),
        (r"linear_attn\.in_proj_z\.weight$", "gdn_z"),
        (r"linear_attn\.out_proj\.weight$", "gdn_out"),
        (r"linear_attn\.in_proj_[ab]\.weight$", "gdn_ab"),
        (r"linear_attn\.conv1d\.weight$", "gdn_conv"),
        (r"linear_attn\.norm\.weight$", "gdn_norm"),
        (r"linear_attn\.dt_bias$", "dt_bias"),
        (r"self_attn\.q_proj\.weight$", "attn_q"),
        (r"self_attn\.k_proj\.weight$", "attn_k"),
        (r"self_attn\.v_proj\.weight$", "attn_v"),
        (r"self_attn\.o_proj\.weight$", "attn_o"),
        (r"self_attn\.[qk]_norm\.weight$", "attn_qk_norm"),
        (r"indexer\.index_qk_proj\.weight$", "indexer_qk"),
        (r"indexer\.[qk]_layernorm\.weight$", "indexer_norm"),
        (r"ple\.key_proj\.weight$", "ple_key"),
        (r"ple\.value_proj\.weight$", "ple_value"),
        (r"ple\.conv1d\.weight$", "ple_conv"),
        (r"ple\.norm_(key|query|conv)\.weight$", "ple_norm"),
        (r"ple_embedding\.ngram_embedding\.weight$", "ngram_table"),
    ]
    for pattern, key in rules:
        if re.search(pattern, name):
            return key
    raise KeyError(f"no init rule for text-model parameter {name}")


def init_weights(model, seed: int = SEED) -> None:
    """Overwrite every text parameter with a seeded draw at the SCALES above (fp32 draw, cast to the param dtype)."""
    gen = torch.Generator().manual_seed(seed)
    text = model.model.language_model
    with torch.no_grad():
        # parameters in a fixed order: named_parameters() is registration order, which is deterministic
        for name, p in text.named_parameters():
            if name.endswith("A_log"):
                a = torch.empty(p.shape).uniform_(0.01, 16.0, generator=gen)
                p.copy_(a.log())
            elif name.endswith("experts.gate_up_proj"):
                inter = p.shape[1] // 2
                p[:, :inter] = _normal(p[:, :inter].shape, SCALES["expert_gate"], gen).to(p.dtype)
                p[:, inter:] = _normal(p[:, inter:].shape, SCALES["expert_up"], gen).to(p.dtype)
            elif name.endswith("experts.down_proj"):
                p.copy_(_normal(p.shape, SCALES["expert_down"], gen))
            else:
                p.copy_(_normal(p.shape, SCALES[_scale_key(name)], gen))
        mean, std = SCALES["lm_head"]
        model.lm_head.weight.copy_(torch.randn(model.lm_head.weight.shape, generator=gen) * std + mean)


def build_model(seed: int = SEED):
    """Fresh random tiny model (bf16, eager experts) from build_config_dict()."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    config = Qwen4ExpConfig(**build_config_dict())
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen4ExpForConditionalGeneration(config)
    init_weights(model, seed)
    for p in model.parameters():  # not model.to(): that would round the fp32 rotary inv_freq buffers to bf16
        p.data = p.data.to(torch.bfloat16)
    return model.eval()


def checkpoint_tensors(model) -> dict[str, torch.Tensor]:
    """The model as checkpoint-format tensors (real names): per-expert unfused linears, n-gram shards, no visual tower."""
    sd = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    num_parts = model.config.text_config.split_ngram_parts
    for name, t in sd.items():
        if name.startswith("model.visual."):
            continue
        if name.endswith(".mlp.experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            inter = t.shape[1] // 2
            for e in range(t.shape[0]):
                out[f"{base}{e}.gate_proj.weight"] = t[e, :inter].contiguous()
                out[f"{base}{e}.up_proj.weight"] = t[e, inter:].contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            base = name[: -len("down_proj")]
            for e in range(t.shape[0]):
                out[f"{base}{e}.down_proj.weight"] = t[e].contiguous()
        elif name.endswith(".ngram_embedding.weight"):
            base = name[: -len("weight")]
            if t.shape[0] % num_parts:
                raise RuntimeError(f"{name}: {t.shape[0]} rows do not split into {num_parts} shards")
            for i, part in enumerate(torch.chunk(t, num_parts, dim=0)):
                out[f"{base}shard_{i}.weight"] = part.contiguous()
        else:
            out[name] = t.contiguous()
    return out


def write_checkpoint(model, hf_dir: Path) -> None:
    from safetensors.torch import save_file

    hf_dir.mkdir(parents=True, exist_ok=True)
    tensors = checkpoint_tensors(model)
    shard_re = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
    main_name = "model-00001-of-00001.safetensors"
    emb_names = [f"embedding-model-{i + 1:05d}-of-{EMBEDDING_FILES:05d}.safetensors" for i in range(EMBEDDING_FILES)]
    files: dict[str, dict[str, torch.Tensor]] = {main_name: {}, **{n: {} for n in emb_names}}
    weight_map: dict[str, str] = {}
    for name, t in tensors.items():
        m = shard_re.search(name)
        file = emb_names[int(m.group(1)) * EMBEDDING_FILES // NGRAM_PARTS] if m else main_name
        files[file][name] = t
        weight_map[name] = file
    total = 0
    for file, content in files.items():
        save_file(content, str(hf_dir / file), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in content.values())
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (hf_dir / "config.json").write_text(json.dumps(build_config_dict(), indent=2) + "\n", encoding="utf-8")
    (hf_dir / "generation_config.json").write_text(json.dumps(build_generation_config(), indent=2) + "\n",
                                                   encoding="utf-8")


def load_hf_model(model_dir: str | Path, dtype=torch.bfloat16, experts_implementation: str = "eager"):
    """Qwen4ExpForConditionalGeneration from a checkpoint-format dir (the tiny one), eval mode, CPU."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=dtype, experts_implementation=experts_implementation
    )
    return model.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build", help="write hf/, sidecar.r4dx-ngram and DONE")
    bp.add_argument("--out", default=str(DEFAULT_OUT))
    bp.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    out = Path(args.out)
    done = out / "DONE"
    done.unlink(missing_ok=True)
    hf_dir = out / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    model = build_model(args.seed)
    write_checkpoint(model, hf_dir)
    del model

    from qwen4exp import ngram_sidecar

    header = ngram_sidecar.repack(hf_dir, out / "sidecar.r4dx-ngram")
    done.write_text(f"tiny-q4exp seed {args.seed} rows {header.rows} sha256 {header.body_sha256.hex()}\n")
    print(f"wrote {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
