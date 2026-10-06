"""CPU-only golden for the Gemma 4 (gemma4_unified) mask / rope / embed-scale semantics.

Builds tiny configs, calls the HF mask and rope functions on CPU (no model weights, no GPU),
and dumps what they actually produce to <out-dir>/gemma/mask_semantics.json. The statements in
docs/gemma4-semantics.md are backed by these dumps.

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\rope_masks_golden.py

Default --out-dir is tools/reference/golden_out (gitignored like the other goldens).
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import masking_utils  # noqa: E402
from transformers.models.gemma4_unified import modeling_gemma4_unified as M  # noqa: E402
from transformers.models.gemma4_unified.configuration_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
)

T = 8
WINDOW = 4
MODELS_ROOT = Path(os.environ.get("R4DX_MODELS_ROOT", r"E:\models"))
DEFAULT_MODEL_DIR = MODELS_ROOT / "Huihui-gemma-4-12B-it-abliterated"


def tiny_config(sliding_window=WINDOW):
    # last layer is forced to full_attention by the config; layer 0 sliding, layer 1 full.
    text = Gemma4UnifiedTextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=sliding_window,
        layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=30.0,
        max_position_embeddings=64, use_bidirectional_attention="vision",
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    )
    cfg = Gemma4UnifiedConfig(text_config=text)
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"
    return cfg


def to_bool(mask):
    """Float eager mask (0 = attend, min = blocked) or bool mask -> [T,T] list of 0/1 (row = query)."""
    if mask is None:
        return None
    m = mask[0, 0]
    allowed = (m == 0) if m.dtype.is_floating_point else m
    return allowed.int().tolist()


def masks_generate_path(cfg, mm_ids):
    """What `model.generate()` uses: Gemma4UnifiedForConditionalGeneration.create_masks_for_generate."""
    emb = torch.zeros(1, T, cfg.text_config.hidden_size)
    out = M.Gemma4UnifiedForConditionalGeneration.create_masks_for_generate(
        cfg, emb, torch.ones(1, T, dtype=torch.long), None, torch.arange(T)[None],
        mm_token_type_ids=None if mm_ids is None else torch.tensor([mm_ids]),
    )
    return {k: to_bool(v) for k, v in out.items()}


def masks_forward_path(cfg, mm_ids):
    """What a plain `Gemma4UnifiedModel.forward(...)` builds when no mask dict is passed
    (modeling_gemma4_unified.py: block_sequence_ids -> masking_utils.create_masks_for_generate)."""
    emb = torch.zeros(1, T, cfg.text_config.hidden_size)
    bsi = torch.full((1, T), -1)
    if mm_ids is not None:
        bsi = M.get_block_sequence_ids_for_mask(torch.tensor([mm_ids]), device=emb.device)
    out = masking_utils.create_masks_for_generate(
        config=cfg.get_text_config(), inputs_embeds=emb, attention_mask=torch.ones(1, T, dtype=torch.long),
        past_key_values=None, position_ids=torch.arange(T)[None], block_sequence_ids=bsi,
    )
    return {k: to_bool(v) for k, v in out.items()}, bsi.tolist()


def real_rope_and_scale():
    """12B text config: inv_freq per layer type, proportional layout check, embed-scale dtype."""
    cfgj = json.load(open(Path(os.environ.get("R4DX_MODEL_DIR", str(DEFAULT_MODEL_DIR))) / "config.json"))
    text = Gemma4UnifiedTextConfig(**cfgj["text_config"])
    rot = M.Gemma4UnifiedTextRotaryEmbedding(text)
    res = {"rope_type": dict(rot.rope_type), "max_position_embeddings": text.max_position_embeddings}
    for lt in ("sliding_attention", "full_attention"):
        f = getattr(rot, f"{lt}_inv_freq")
        res[lt] = {
            "inv_freq_len": f.numel(),
            "attention_scaling": getattr(rot, f"{lt}_attention_scaling"),
            "inv_freq_first4": f[:4].tolist(),
            "inv_freq_last_nonzero_index": int((f != 0).nonzero().max()),
            "n_zero": int((f == 0).sum()),
        }
    # Explicit formula for the full layer vs HF cos/sin applied to a random vector.
    hd, base, nrot = 512, 1e6, 64
    ref_if = torch.zeros(hd // 2)
    ref_if[:nrot] = 1.0 / base ** (torch.arange(0, 2 * nrot, 2, dtype=torch.float) / hd)
    res["full_inv_freq_matches_formula"] = bool(torch.equal(ref_if, rot.full_attention_inv_freq))
    pos = torch.tensor([[0, 5, 1000]])
    x = torch.randn(1, 3, 1, hd, generator=torch.Generator().manual_seed(0))
    cos, sin = rot(x, pos, "full_attention")
    got = M.apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2)
    exp = x.clone()
    for t in range(3):
        p = float(pos[0, t])
        for i in range(nrot):  # pairs (i, i+256), identity elsewhere
            a, b = x[0, t, 0, i].item(), x[0, t, 0, i + hd // 2].item()
            ang = p * ref_if[i].item()
            exp[0, t, 0, i] = a * math.cos(ang) - b * math.sin(ang)
            exp[0, t, 0, i + hd // 2] = b * math.cos(ang) + a * math.sin(ang)
    res["full_rope_pairs_i_i_plus_256_for_i_lt_64_identity_elsewhere_max_abs_err"] = float((got - exp).abs().max())
    # Sliding: full-rotary neox pairs (i, i+128), theta 1e4, head_dim 256.
    hd_s = 256
    s_if = 1.0 / 1e4 ** (torch.arange(0, hd_s, 2, dtype=torch.float) / hd_s)
    res["sliding_inv_freq_matches_formula"] = bool(torch.equal(s_if, rot.sliding_attention_inv_freq))
    # Embed scale: float32 tensor(sqrt(3840)) cast to bf16 before the multiply.
    s = torch.tensor(text.hidden_size ** 0.5)
    res["embed_scale"] = {"f32": s.item(), "bf16": s.to(torch.bfloat16).item(), "sqrt_3840": text.hidden_size ** 0.5}
    return res


def bos_and_layer_types():
    """BOS behaviour of the assembled tokenizer dir and the 5:1 layer pattern in config.json."""
    mdir = Path(os.environ.get("R4DX_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    tdir = Path(os.environ.get("R4DX_TOKENIZER_DIR", str(mdir) + "-tok"))
    res = {}
    cfgj = json.load(open(mdir / "config.json"))
    lt = cfgj["text_config"]["layer_types"]
    res["full_attention_layers"] = [i for i, t in enumerate(lt) if t == "full_attention"]
    res["pattern_is_5_to_1_full_at_5_11_to_47"] = res["full_attention_layers"] == list(range(5, 48, 6)) and len(lt) == 48
    if (tdir / "tokenizer.json").exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tdir))
        res["hi_default"] = tok("hi").input_ids
        res["hi_no_special"] = tok("hi", add_special_tokens=False).input_ids
        res["bos_literal_hi"] = tok("<bos>hi").input_ids
        res["tokenizer_json_post_processor"] = json.load(open(tdir / "tokenizer.json", encoding="utf8"))["post_processor"]
        rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
        res["chat_rendered"] = rendered
        res["chat_ids_head"] = tok(rendered).input_ids[:4]
        res["raw_adds_bos"] = res["hi_default"][:1] == [2]
    return res


def sliding_boundary():
    """dist = q - kv; which distances the sliding mask admits (window 4 -> dist 0..3)."""
    cfg = tiny_config()
    m = masks_generate_path(cfg, None)["sliding_attention"]
    q = T - 1
    return {"window": WINDOW, "query": q, "allowed_kv": [k for k in range(T) if m[q][k]],
            "allowed_dists": [q - k for k in range(T) if m[q][k]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "golden_out"))
    a = ap.parse_args()
    cfg = tiny_config()
    cases = {
        "text_only": [0] * T,
        "one_image_2_to_5": [0, 0, 1, 1, 1, 1, 0, 0],
        "two_images_1_2_and_4_5": [0, 1, 1, 0, 1, 1, 0, 0],
        "audio_3_not_a_vision_block": [0, 0, 3, 3, 3, 3, 0, 0],
    }
    out = {
        "transformers": transformers.__version__, "T": T, "sliding_window": WINDOW,
        "layers": {"0": "sliding_attention", "1": "full_attention"},
        "mask_row_is_query_col_is_key_1_is_attend": True,
        "generate_path": {}, "forward_path": {}, "block_sequence_ids": {},
    }
    for name, ids in cases.items():
        out["generate_path"][name] = masks_generate_path(cfg, ids)
        fwd, bsi = masks_forward_path(cfg, ids)
        out["forward_path"][name] = fwd
        out["block_sequence_ids"][name] = bsi[0]
    out["sliding_boundary"] = sliding_boundary()
    out["bos_and_layer_types"] = bos_and_layer_types()
    out["rope_and_scale"] = real_rope_and_scale()
    g = out["generate_path"]["one_image_2_to_5"]
    f = out["forward_path"]["one_image_2_to_5"]
    out["findings"] = {
        "generate_full_layer_is_pure_causal": g["full_attention"] == out["generate_path"]["text_only"]["full_attention"],
        "generate_sliding_layer_has_bidirectional_image_block": g["sliding_attention"][2][5] == 1,
        "forward_full_layer_has_bidirectional_image_block": f["full_attention"][2][5] == 1,
        "forward_sliding_layer_has_bidirectional_image_block": f["sliding_attention"][2][5] == 1,
        "audio_block_stays_causal": out["generate_path"]["audio_3_not_a_vision_block"]["sliding_attention"][2][5] == 0,
        "two_images_do_not_see_each_other_ahead": out["generate_path"]["two_images_1_2_and_4_5"]["sliding_attention"][2][4] == 0,
    }
    d = Path(a.out_dir) / "gemma"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mask_semantics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["findings"], indent=1))
    print(json.dumps(out["sliding_boundary"]))
    print(json.dumps(out["rope_and_scale"], indent=1))
    print("wrote", d / "mask_semantics.json")


if __name__ == "__main__":
    main()

