"""tools/reference/mrope_layer_golden.py

Golden for ONE real full-attention decoder layer driven by 3-AXIS mrope position ids, i.e. the
layer-level half of the vision milestone's text-side splicing validation (docs/vision.md
"Text-side splicing", stage 4 deliverable 4b).

Why this exists separately from layer_golden.py: that golden's `mrope_position_ids` collapses all
three (t,h,w) streams to the same sequential index, which is correct for a text-only prompt and
therefore cannot distinguish `r4dx_rope_partial_mrope3_bf16`'s per-frequency-bin stream selection
from the single-row kernel it replaces. This one drives the SAME layer, from the SAME real
weights, with the position ids `Qwen3_5Model.get_rope_index` actually produces for a prompt
containing an image -- so the t/h/w rows genuinely differ and the recomposition is exercised.
And because a full bf16 27B reference does not fit on this card, a single layer is the largest
piece of the real stack that CAN be compared numerically at all.

The hidden states are seeded random for both the text and the image rows. That is deliberate and
sufficient: the splice's own correctness (do the merger's rows land at the placeholder positions?)
is a memcpy checked structurally on the C++ side, while what a layer-level numeric golden can
actually falsify is the ROPE -- whether each of the 32 frequency bins reads the position stream
the reference says it does. Feeding the real tower's output instead would change no arithmetic in
this layer and would make the golden depend on the whole vision tower.

Usage (reference venv's python; GPU only via HIP_VISIBLE_DEVICES=1, per CLAUDE.md):
    $env:HIP_VISIBLE_DEVICES='1'
    <reference-venv>\\Scripts\\python.exe tools\\reference\\mrope_layer_golden.py ^
        --out-dir tools\\reference\\golden_out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ShardIndex,
    load_module_state,
    force_eager,
    load_text_config,
    resolve_device,
    save_golden,
    set_seed,
    tensor_manifest_entry,
)
from layer_golden import build_layer, causal_additive_mask  # noqa: E402
from rope_index_golden import _RopeIndexShim  # noqa: E402


def build_prompt(image_grid, text_before, text_after, image_token_id, vision_start_token_id,
                 vision_end_token_id, merge):
    """The same `build_case` layout rope_index_golden.py uses, inlined for one image: a text run,
    <vision_start> (a TEXT-type token), the merged image placeholders, <vision_end>, a text run."""
    t, h, w = image_grid
    n_merged = t * (h // merge) * (w // merge)
    input_ids = list(range(1000, 1000 + text_before))
    types = [0] * text_before
    input_ids.append(vision_start_token_id)
    types.append(0)
    image_start = len(input_ids)
    input_ids.extend([image_token_id] * n_merged)
    types.extend([1] * n_merged)
    input_ids.append(vision_end_token_id)
    types.append(0)
    input_ids.extend(range(2000, 2000 + text_after))
    types.extend([0] * text_after)
    return input_ids, types, image_start, n_merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "golden_out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attn-layer", type=int, default=3,
                    help="layer index to run (must be a full_attention layer; matches "
                          "layer_golden.py's own default so the two goldens are comparable)")
    ap.add_argument("--decode-len", type=int, default=4)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers.cache_utils import Cache, DynamicLayer

    full_config, text_config = load_text_config(args.model_dir)
    index = ShardIndex.load(args.model_dir)
    merge = full_config.vision_config.spatial_merge_size

    # A 10x16-patch image merges to 5x8 = 40 placeholder tokens: small enough that the whole
    # prompt is one 64-token r4dx prefill chunk, non-square so the h and w streams cannot be
    # swapped without changing the answer, and with max(h,w)//merge = 8 != 40 so the post-image
    # advance rule is genuinely exercised (a "advance by token count" bug shifts everything after).
    image_grid = (1, 10, 16)
    input_ids, types, image_start, n_merged = build_prompt(
        image_grid, text_before=6, text_after=15,
        image_token_id=full_config.image_token_id,
        vision_start_token_id=full_config.vision_start_token_id,
        vision_end_token_id=full_config.vision_end_token_id, merge=merge)
    prefill_len = len(input_ids)
    total_len = prefill_len + args.decode_len

    shim = _RopeIndexShim(full_config, m)
    prompt_pos, deltas = shim.get_rope_index(
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([types], dtype=torch.int32),
        image_grid_thw=torch.tensor([list(image_grid)], dtype=torch.long),
        video_grid_thw=None, attention_mask=None)
    delta = int(deltas[0, 0].item())

    # Decode positions continue at `sequence index + delta` on all three rows -- the rule
    # docs/vision.md states and Model::RopePositionsHost implements.
    decode_pos = torch.arange(prefill_len, total_len, dtype=torch.long) + delta
    position_ids = torch.cat(
        [prompt_pos[:, 0, :], decode_pos.view(1, -1).expand(3, -1)], dim=1).unsqueeze(1)
    position_ids = position_ids.to(device)  # [3, 1, total_len]

    set_seed(args.seed)
    layer_config = type(text_config).from_dict(text_config.to_dict())
    layer_config.layer_types = list(text_config.layer_types)
    layer_config.layer_types[args.attn_layer] = "full_attention"
    force_eager(layer_config)  # same dispatch layer_golden.py's own attention case uses
    layer = build_layer(m, layer_config, args.attn_layer, device, dtype)
    hf_prefix = f"model.language_model.layers.{args.attn_layer}."
    loaded, missing, error = load_module_state(layer, index, hf_prefix)
    if not loaded:
        raise RuntimeError(
            f"this golden needs the REAL layer-{args.attn_layer} weights; load_module_state "
            f"reported error={error} missing={missing[:5]}")

    hidden_size = text_config.hidden_size
    full_hidden = torch.randn(1, total_len, hidden_size,
                              generator=torch.Generator(device="cpu").manual_seed(args.seed))
    full_hidden = full_hidden.to(device=device, dtype=dtype)
    # Scale the image rows differently from the text rows so a splice that silently wrote the
    # wrong rows (or the right rows at the wrong offset) changes the layer output measurably --
    # a uniform random block would hide an off-by-one in the span's own offset.
    with torch.no_grad():
        full_hidden[:, image_start:image_start + n_merged, :] *= 2.0

    rotary = m.Qwen3_5TextRotaryEmbedding(text_config).to(device=device)
    cos, sin = rotary(full_hidden, position_ids)

    cache = Cache(layers=[DynamicLayer() for _ in range(args.attn_layer + 1)])
    tensors: dict[str, torch.Tensor] = {
        "prefill_hidden_states": full_hidden[0, :prefill_len],
        "decode_hidden_states": full_hidden[0, prefill_len:],
        "mrope_position_ids": position_ids[:, 0, :].to(torch.int64).cpu(),
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "mm_token_type_ids": torch.tensor(types, dtype=torch.int32),
        "image_grid_thw": torch.tensor([list(image_grid)], dtype=torch.int64),
    }

    sink: dict[str, torch.Tensor] = {}
    h_attn = layer.self_attn.register_forward_hook(
        lambda _m, _i, o: sink.__setitem__("attention_output",
                                           (o[0] if isinstance(o, tuple) else o).detach().clone()))
    # Same apply_rotary_pos_emb monkeypatch layer_golden.py uses -- transformers exposes no hook
    # point between q_norm/k_norm (pre-rope) and the attention call (post-rope).
    original_rope = m.apply_rotary_pos_emb

    def capture_rope(q, k, cos_, sin_, unsqueeze_dim=1):
        q_e, k_e = original_rope(q, k, cos_, sin_, unsqueeze_dim)
        sink["attn_q_post_rope"] = q_e.detach().clone()
        sink["attn_k_post_rope"] = k_e.detach().clone()
        return q_e, k_e

    m.apply_rotary_pos_emb = capture_rope
    try:
        with torch.no_grad():
            for stage, lo, hi in (("prefill", 0, prefill_len), ("decode", prefill_len, total_len)):
                sink.clear()
                out = layer(
                    hidden_states=full_hidden[:, lo:hi, :],
                    position_embeddings=(cos[:, lo:hi], sin[:, lo:hi]),
                    attention_mask=causal_additive_mask(hi - lo, hi, lo, device, dtype),
                    position_ids=None,
                    past_key_values=cache,
                )
                for k, v in sink.items():
                    tensors[f"{stage}_{k}"] = v[0] if v.dim() > 0 and v.shape[0] == 1 else v
                tensors[f"{stage}_layer_output"] = out[0]
    finally:
        m.apply_rotary_pos_emb = original_rope
        h_attn.remove()

    tensors = {k: (v.detach().cpu() if v.is_cuda else v.detach()) for k, v in tensors.items()}
    out_path = args.out_dir / ("mrope_layer_%03d.safetensors" % args.attn_layer)
    save_golden(out_path, tensors, extra_meta={"component": "mrope_layer"})

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "device": str(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "seed": args.seed,
        "layer_idx": args.attn_layer,
        "file": out_path.name,
        "prefill_len": prefill_len,
        "decode_len": args.decode_len,
        "image_start": image_start,
        "num_merged_tokens": n_merged,
        "image_grid_thw": list(image_grid),
        "spatial_merge_size": merge,
        "mrope_position_delta": delta,
        "mrope_section": list(text_config.rope_parameters.get("mrope_section", [11, 11, 10])),
        "weights_source": "safetensors",
        "missing_weight_keys": missing,
        "semantics": {
            "mrope_position_ids": "[3, prefill_len+decode_len] int64 -- get_rope_index's own rows "
            "for the prompt, then `sequence index + mrope_position_delta` on all three rows for "
            "the decode steps.",
            "prefill_attention_output": "the ATTENTION SUBLAYER's output (o_proj's result, before "
            "the residual add) -- the same tensor tests/model/attention/test_attn_layer.cpp "
            "compares against for the text-only golden.",
        },
        "tensors": {k: tensor_manifest_entry(v) for k, v in tensors.items()},
    }
    with open(args.out_dir / "mrope_layer_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[mrope_layer_golden] prefill_len={prefill_len} image_start={image_start} "
          f"n_merged={n_merged} delta={delta}")
    print(f"[mrope_layer_golden] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
