"""CPU-only golden for the Gemma 4 bidirectional image mask (docs/gemma4-plan.md M2, semantics doc section 2).

For a set of token layouts (text / image soft tokens with boi / eoi, one or several images, an image
touching the window edge, images at the sequence start and end) this asks transformers' own mask code --
Gemma4UnifiedForConditionalGeneration.create_masks_for_generate with mm_token_type_ids, the path
generate() takes -- for the sliding and the full layer masks of a tiny config, and writes them as text to
<out-dir>/gemma/vision_masks.txt:

    case <name> <T> <window>
    types <T ints: mm_token_type_ids, 1 = image soft token, 0 = text/boi/eoi>
    sliding <T lines of T 0/1 chars (row = query, col = key)>
    full    <T lines>

tests/vision/test_gemma_vision_cpu.cpp rebuilds each mask from the type row with
r4dx::vision::BuildDenseMask (blocks = maximal runs of type 1) and compares cell by cell: sliding layers get
the block overlay, full layers stay causal (the semantics-doc decision, and what the libr4d / reference
attention klimit_ext implements). Prefill only: decode keeps mm_token_type_ids dropped, i.e. causal.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\vision_masks_golden.py
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch  # noqa: E402

assert not torch.cuda.is_available(), "GPU must stay hidden"

from transformers.models.gemma4_unified import modeling_gemma4_unified as M  # noqa: E402
from transformers.models.gemma4_unified.configuration_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
)


def tiny_config(window):
    text = Gemma4UnifiedTextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=window,
        layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=30.0,
        max_position_embeddings=256, use_bidirectional_attention="vision",
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    )
    cfg = Gemma4UnifiedConfig(text_config=text)
    cfg._attn_implementation = "eager"
    cfg.get_text_config()._attn_implementation = "eager"
    return cfg


def to_rows(mask, T):
    m = mask[0, 0]
    allowed = (m == 0) if m.dtype.is_floating_point else m
    return ["".join("1" if v else "0" for v in row) for row in allowed.int().tolist()]


def layout(spec):
    """spec: list of ('t', n) text, ('i', n) a boi + n image tokens + eoi."""
    types = []
    for kind, n in spec:
        if kind == "t":
            types += [0] * n
        else:
            types += [0] + [1] * n + [0]
    return types


CASES = [
    ("text_only", 5, [("t", 12)]),
    ("one_image_small_window", 4, [("t", 3), ("i", 4), ("t", 3)]),
    ("one_image_big_window", 64, [("t", 5), ("i", 9), ("t", 6)]),
    ("two_images", 6, [("t", 2), ("i", 5), ("t", 3), ("i", 4), ("t", 2)]),
    ("block_wider_than_window", 3, [("t", 2), ("i", 9), ("t", 2)]),
    ("image_first_and_last", 5, [("i", 6), ("t", 4), ("i", 3)]),
    ("adjacent_images", 8, [("i", 3), ("i", 4), ("t", 1)]),
    ("single_token_image", 4, [("t", 3), ("i", 1), ("t", 3)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "golden_out"))
    a = ap.parse_args()
    lines = []
    for name, window, spec in CASES:
        types = layout(spec)
        T = len(types)
        cfg = tiny_config(window)
        emb = torch.zeros(1, T, cfg.text_config.hidden_size)
        out = M.Gemma4UnifiedForConditionalGeneration.create_masks_for_generate(
            cfg, emb, torch.ones(1, T, dtype=torch.long), None, torch.arange(T)[None],
            mm_token_type_ids=torch.tensor([types]),
        )
        lines.append(f"case {name} {T} {window}")
        lines.append("types " + " ".join(str(t) for t in types))
        lines.append("sliding")
        lines += to_rows(out["sliding_attention"], T)
        lines.append("full")
        lines += to_rows(out["full_attention"], T)
        print(f"{name}: T={T} window={window}")
    d = Path(a.out_dir) / "gemma"
    d.mkdir(parents=True, exist_ok=True)
    (d / "vision_masks.txt").write_text("\n".join(lines) + "\n")
    print("wrote", d / "vision_masks.txt")


if __name__ == "__main__":
    sys.exit(main())
