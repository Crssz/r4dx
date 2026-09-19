"""tools/reference/kv_calibrate.py

Prototype of the static fp8 KV descale calibration described in docs/container-format.md
("KV descale tables"): for one full-attention text layer, run a calibration text through that
layer's q/k/v projection + rope path (real weights when the shard is downloaded, else seeded
random init) and record the per-kv-head amax of K (post-rope, i.e. exactly what the paged fp8
cache would store) and V (no rope applied to V).

Emits JSON: {"<layer_idx>": {"k_amax": [kv_heads floats], "v_amax": [kv_heads floats]}}.

This is a PROTOTYPE, not the converter's real calibration pass: it feeds the calibration tokens'
raw embeddings straight into one layer's input_layernorm + self_attn (skipping every preceding
layer), so the hidden-state distribution it calibrates against is not what layer N actually sees
mid-stack. The real converter calibration must run the full 64-layer stack up to layer N. This
tool exists to pin down the math (which tensor, which axis, e4m3 recipe) and the JSON contract
ahead of that; see "How the converter will consume this" below and docs/validation.md.

Usage:
    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\reference\\kv_calibrate.py ^
        --device cuda --layer 3 --out tools\\reference\\kv_calibrate_out\\kv_descale.json

See tools/reference/README.md for the full option list and expected runtime.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ShardIndex,
    force_eager,
    load_module_state,
    load_text_config,
    resolve_device,
    set_seed,
    sha256_file,
)

# A few hundred tokens of varied English prose (numbers, punctuation, code-ish tokens) so the
# calibration set isn't degenerate. Repeated/truncated to --num-tokens after tokenization.
DEFAULT_CALIBRATION_TEXT = """
The gated delta network keeps a running state of shape [H, K, V] per sequence, updated one chunk
at a time. Each of the 48 linear-attention layers repeats this recurrence, while every fourth
layer runs full multi-head attention instead, with head_dim=256 and 4 key/value heads shared
across 24 query heads via grouped-query attention. Positions are encoded with a partial rotary
embedding: only the first quarter of each head's 256 dimensions rotate, using an interleaved
multi-axis scheme (mrope) with sections [11, 11, 10] over temporal, height, and width position
ids and a base theta of 1e7. Static per-head fp8 e4m3 descale factors are computed once, from a
short calibration pass, rather than dynamically at every decode step -- trading a small amount of
accuracy for a paged KV cache with no per-token scale bookkeeping. In 2026, on a single Radeon AI
PRO R9700, chasing 120 tokens/second on a 27-billion-parameter model means every one of these
choices has to pay for itself in bandwidth saved, not just FLOPs. def forward(self, hidden): x =
self.norm(hidden); q, k, v = self.qkv(x).chunk(3, dim=-1); return self.attn(q, k, v). The quick
brown fox jumps over the lazy dog, 1234567890 times, while pi is approximately 3.14159265358979.
""".strip()


def build_calibration_ids(tokenizer, num_tokens: int) -> torch.Tensor:
    ids: list[int] = []
    while len(ids) < num_tokens:
        ids.extend(tokenizer(DEFAULT_CALIBRATION_TEXT, add_special_tokens=False)["input_ids"])
    return torch.tensor(ids[:num_tokens], dtype=torch.long)


def gather_embedding_rows(index: ShardIndex, name: str, ids: torch.Tensor) -> torch.Tensor:
    """Read only the rows we need out of the (huge, [vocab, hidden]) embedding table."""
    from safetensors import safe_open

    shard = index.weight_map[name]
    rows = []
    with safe_open(str(index.model_dir / shard), framework="pt", device="cpu") as f:
        sl = f.get_slice(name)
        for tid in ids.tolist():
            rows.append(sl[tid : tid + 1, :])
    return torch.cat(rows, dim=0)


def causal_additive_mask(seq_len: int, device, dtype) -> torch.Tensor:
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(k_pos > q_pos, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def mrope_position_ids(total_len: int, device) -> torch.Tensor:
    idx = torch.arange(total_len, device=device)
    return idx.view(1, 1, total_len).expand(3, 1, total_len).clone()


def run_calibration(model_dir: Path, layer_idx: int, device, dtype, seed: int, num_tokens: int):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers import AutoTokenizer

    full_config, text_config = load_text_config(model_dir)
    if text_config.layer_types[layer_idx] != "full_attention":
        raise ValueError(
            f"layer {layer_idx} is {text_config.layer_types[layer_idx]!r}, not full_attention -- "
            "KV descale calibration only applies to the 16 full-attention layers (r4d's paged fp8 "
            "KV cache is attention-only; GDN layers have their own fp32 recurrent state instead)."
        )
    index = ShardIndex.load(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    calib_ids = build_calibration_ids(tokenizer, num_tokens)

    hidden_size = text_config.hidden_size
    embed_name = "model.language_model.embed_tokens.weight"
    if index.available(embed_name):
        embeds = gather_embedding_rows(index, embed_name, calib_ids)
        embed_source = "safetensors (row-gathered)"
    else:
        set_seed(seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        embeds = torch.randn(num_tokens, hidden_size, generator=gen) * 0.02
        embed_source = f"random_init(seed={seed})"
    hidden = embeds.unsqueeze(0).to(device=device, dtype=dtype)

    set_seed(seed + 1)
    layer_config = copy.deepcopy(text_config)
    layer = m.Qwen3_5DecoderLayer(layer_config, layer_idx).to(device=device, dtype=dtype)
    layer.eval()
    hf_prefix = f"model.language_model.layers.{layer_idx}."
    loaded, missing, error = load_module_state(layer, index, hf_prefix)
    weights_source = "safetensors" if loaded else f"random_init(seed={seed + 1})"

    rotary = m.Qwen3_5TextRotaryEmbedding(text_config).to(device=device)
    pos_ids = mrope_position_ids(num_tokens, device)
    cos, sin = rotary(hidden, pos_ids)
    mask = causal_additive_mask(num_tokens, device, dtype)

    captured = {}
    original_rope = m.apply_rotary_pos_emb

    def capture_rope(q, k, cos_, sin_, unsqueeze_dim=1):
        q_embed, k_embed = original_rope(q, k, cos_, sin_, unsqueeze_dim)
        captured["k_post_rope"] = k_embed.detach().clone()  # [batch, kv_heads, T, head_dim]
        return q_embed, k_embed

    def capture_v(_module, _inputs, output):
        captured["v_raw"] = output.detach().clone()  # [batch, T, kv_heads*head_dim]

    m.apply_rotary_pos_emb = capture_rope
    v_handle = layer.self_attn.v_proj.register_forward_hook(capture_v)
    try:
        with torch.no_grad():
            normed = layer.input_layernorm(hidden)
            layer.self_attn(
                hidden_states=normed,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                past_key_values=None,
            )
    finally:
        m.apply_rotary_pos_emb = original_rope
        v_handle.remove()

    kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    k = captured["k_post_rope"][0]  # [kv_heads, T, head_dim]
    v = captured["v_raw"][0].view(num_tokens, kv_heads, head_dim).transpose(0, 1)  # [kv_heads, T, head_dim]

    k_amax = k.float().abs().amax(dim=(1, 2)).tolist()
    v_amax = v.float().abs().amax(dim=(1, 2)).tolist()

    return {
        "layer_idx": layer_idx,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "num_calibration_tokens": num_tokens,
        "k_amax": k_amax,
        "v_amax": v_amax,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "weights_source": weights_source,
        "missing_weight_keys": missing,
        "load_error": error,
        "embed_source": embed_source,
        "seed": seed,
        "config_sha256": sha256_file(model_dir / "config.json") if (model_dir / "config.json").exists() else None,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "caveat": (
            "PROTOTYPE: hidden_states fed to this layer are the calibration tokens' raw embeddings "
            "(text.embed_tokens), not the true mid-stack activation that layer_idx actually sees -- "
            "every preceding layer's transform is skipped. Good enough to pin down the amax/JSON "
            "contract; not good enough to ship as the real per-model calibration."
        ),
        "converter_consumption": (
            "The converter's real calibration pass (full 64-layer stack, real prompts) writes "
            "text.layers.{i}.attn.k_descale / .v_descale as fp32[kv_heads] = amax / fp8_e4m3_max "
            "(~448.0), one scalar per kv head, replacing docs/container-format.md's placeholder "
            "1.0 -- see r4d.h's R4DArgs.k_descale/.v_descale and docs/container-format.md 'KV "
            "descale tables'. r4d_kv_write_paged_fp8_hnd (src/kernels, not yet implemented) divides "
            "each K/V element by its head's descale before the fp8 cast on write; the attention "
            "kernel multiplies back by the same descale when it dequantizes for the QK^T / PV "
            "matmuls. A too-small descale clips (saturates at +-448 with the amax outlier's tail "
            "cut off); a too-large one wastes fp8's few mantissa bits -- hence calibrating off "
            "amax rather than guessing a global constant. NOTE: r4d.h declares the *runtime* "
            "k_descale/v_descale as (num_seqs, kv_heads), indexed seq*kv_heads+kvh -- this "
            "per-layer fp32[kv_heads] vector (one value per kv head, shared by every sequence, "
            "since calibration is not per-sequence) must be broadcast to all num_seqs rows when "
            "the loader builds that runtime array, not stored once and reused as a [kv_heads] "
            "buffer."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=3, help="a full_attention layer index (0,4,8,... are linear_attention; 3,7,11,... are full_attention)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "kv_calibrate_out" / "kv_descale.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    result = run_calibration(args.model_dir, args.layer, device, dtype, args.seed, args.num_tokens)

    # Merge into any existing calibration file rather than overwriting it -- a real container
    # needs all 16 full-attention layers' descale tables, calibrated one `--layer` at a time.
    out: dict = {}
    if args.out.exists():
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                out = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[kv_calibrate] WARNING: {args.out} exists but couldn't be parsed ({exc!r}); overwriting it")
            out = {}
    out[str(args.layer)] = result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[kv_calibrate] layer {args.layer}: k_amax={result['k_amax']} v_amax={result['v_amax']}")
    print(f"[kv_calibrate] wrote {args.out} ({len(out)} layer(s) calibrated so far)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
