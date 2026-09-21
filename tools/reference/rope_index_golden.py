"""tools/reference/rope_index_golden.py

Golden for the TEXT-side mrope position ids of a prompt that contains images, i.e. exactly what
`Qwen3_5Model.get_rope_index` + `Qwen3_5Model.get_vision_position_ids` produce -- called as the
REAL, unmodified `transformers` methods (bound to a tiny shim that only carries `.config`, since
those two methods touch nothing else on the model), so this golden cannot drift from the reference
implementation the way a re-derivation in this file would.

No model weights, no GPU, no image data: `get_rope_index` consumes `input_ids` (for shape only),
`mm_token_type_ids` (0=text, 1=image, 2=video) and `image_grid_thw`. That makes this the cheapest
golden in tools/reference -- it runs on any machine in under a second -- and it covers the single
easiest place in the whole vision path to introduce an off-by-one that still produces
plausible-looking output (docs/vision.md "Text-side splicing").

Cases: text only; text + 1 image; text + 2 images with DIFFERENT grids; an image at sequence
position 0 (no leading text run); an image as the final tokens (no trailing text run).

Usage (reference venv's python; --device is not a parameter, this is CPU-only):
    <reference-venv>\\Scripts\\python.exe tools\\reference\\rope_index_golden.py ^
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
from common import DEFAULT_MODEL_DIR, save_golden, tensor_manifest_entry  # noqa: E402


class _RopeIndexShim:
    """`Qwen3_5Model.get_rope_index` and `.get_vision_position_ids` are pure functions of their
    arguments plus `self.config.vision_config.spatial_merge_size` and `self`'s own
    `get_vision_position_ids` -- nothing else on the model is touched. Binding them to this shim
    runs the reference code verbatim without constructing a 27B nn.Module."""

    def __init__(self, config, m):
        self.config = config
        self.get_vision_position_ids = m.Qwen3_5Model.get_vision_position_ids.__get__(self)
        self.get_rope_index = m.Qwen3_5Model.get_rope_index.__get__(self)


def build_case(
    name: str,
    segments: list[tuple[str, object]],
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Assembles one synthetic prompt from `segments`, each either ("text", n_tokens) or
    ("image", (t, h, w)). An image segment expands to the real layout the processor emits:
    <vision_start> (a TEXT-type token), then (h/merge)*(w/merge) image-placeholder tokens carrying
    mm_token_type_id 1, then <vision_end> (text again) -- so the grouping `get_rope_index` does over
    contiguous runs of equal type sees exactly what it sees in production."""
    input_ids: list[int] = []
    token_types: list[int] = []
    grids: list[list[int]] = []
    layout: list[dict] = []
    next_text_id = 1000
    for kind, value in segments:
        start = len(input_ids)
        if kind == "text":
            for _ in range(int(value)):
                input_ids.append(next_text_id)
                token_types.append(0)
                next_text_id += 1
            layout.append({"kind": "text", "start": start, "length": int(value)})
        elif kind == "image":
            t, h, w = value
            n_merged = (h // spatial_merge_size) * (w // spatial_merge_size) * t
            input_ids.append(vision_start_token_id)
            token_types.append(0)
            input_ids.extend([image_token_id] * n_merged)
            token_types.extend([1] * n_merged)
            input_ids.append(vision_end_token_id)
            token_types.append(0)
            grids.append([t, h, w])
            layout.append(
                {"kind": "image", "start": start, "grid_thw": [t, h, w], "num_merged_tokens": n_merged}
            )
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    return (
        name,
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([token_types], dtype=torch.int32),
        torch.tensor(grids, dtype=torch.long) if grids else None,
        layout,
    )


def verify_prompt(shim, dump_path: Path) -> int:
    """Checks the ENGINE's own 3-axis rope rows for a REAL rendered prompt against the unmodified
    reference `get_rope_index` (docs/vision.md "Text-side splicing", stage 4 deliverable 4a).

    `dump_path` is what `tests/vision/tool_vision_chat --dump-prompt` writes: the token ids the
    engine actually fed (after the processor's `<|image_pad|>` expansion), the derived
    mm_token_type_ids, the image grids, and `engine_position_ids` -- the [3, seq] rows
    `Model::PrefillMultimodal` really handed the rope kernel, not a re-derivation. The five
    committed cases above are synthetic; this is the same comparison against a prompt that came
    out of the real chat template and the real tokenizer, where an off-by-one in the placeholder
    expansion or the span offsets would show up and a synthetic fixture never could."""
    with open(dump_path, "r", encoding="utf-8") as f:
        dump = json.load(f)

    input_ids = torch.tensor([dump["input_ids"]], dtype=torch.long)
    types = torch.tensor([dump["mm_token_type_ids"]], dtype=torch.int32)
    grid_flat = dump["image_grid_thw"]
    grid_thw = (torch.tensor(grid_flat, dtype=torch.long).view(-1, 3) if grid_flat else None)
    seq_len = input_ids.shape[1]

    ref_positions, ref_deltas = shim.get_rope_index(
        input_ids, types, image_grid_thw=grid_thw, video_grid_thw=None, attention_mask=None
    )
    ref = ref_positions[:, 0, :].reshape(-1).tolist()
    got = list(dump["engine_position_ids"])
    ref_delta = int(ref_deltas[0, 0].item())
    got_delta = int(dump["mrope_position_delta"])

    mismatches = [i for i, (a, b) in enumerate(zip(ref, got)) if a != b]
    print(f"[verify_prompt] {dump_path}: seq_len={seq_len} "
          f"images={0 if grid_thw is None else grid_thw.shape[0]}")
    if len(got) != len(ref):
        print(f"[verify_prompt] FAIL: engine produced {len(got)} values, reference {len(ref)}")
        return 1
    if mismatches:
        i = mismatches[0]
        print(f"[verify_prompt] FAIL: {len(mismatches)} mismatch(es); first at flat index {i} "
              f"(axis {i // seq_len}, token {i % seq_len}): engine={got[i]} reference={ref[i]}")
        return 1
    if got_delta != ref_delta:
        print(f"[verify_prompt] FAIL: delta engine={got_delta} reference={ref_delta}")
        return 1
    print(f"[verify_prompt] OK: all {len(ref)} position ids match the reference exactly, "
          f"delta={ref_delta}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "golden_out")
    ap.add_argument("--verify-prompt", type=Path, default=None,
                    help="instead of regenerating the golden, check one tool_vision_chat "
                          "--dump-prompt JSON against the reference (see verify_prompt above)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    try:
        config = Qwen3_5Config.from_pretrained(str(args.model_dir))
        config_source = str(args.model_dir)
    except Exception as exc:  # no checkpoint on this machine -- the defaults carry the same ids
        config = Qwen3_5Config()
        config_source = f"Qwen3_5Config() defaults ({type(exc).__name__}: {exc})"

    if args.verify_prompt is not None:
        return verify_prompt(_RopeIndexShim(config, m), args.verify_prompt)

    merge = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id
    vision_end_token_id = config.vision_end_token_id
    shim = _RopeIndexShim(config, m)

    # (28,28) is vision_golden.py's square test image, (26,38) its non-square one -- the same two
    # grids, so the C++ side can cross-check preprocessing and position ids against each other.
    specs = [
        ("text_only", [("text", 17)]),
        ("text_one_image", [("text", 6), ("image", (1, 28, 28)), ("text", 5)]),
        (
            "text_two_images_different_grids",
            [("text", 4), ("image", (1, 28, 28)), ("text", 3), ("image", (1, 26, 38)), ("text", 7)],
        ),
        ("image_first", [("image", (1, 26, 38)), ("text", 9)]),
        ("image_last", [("text", 11), ("image", (1, 28, 28))]),
    ]

    tensors: dict[str, torch.Tensor] = {}
    components: dict[str, dict] = {}
    for name, segments in specs:
        _, input_ids, token_types, grid_thw, layout = build_case(
            name, segments, image_token_id, vision_start_token_id, vision_end_token_id, merge
        )
        position_ids, deltas = shim.get_rope_index(
            input_ids, token_types, image_grid_thw=grid_thw, video_grid_thw=None, attention_mask=None
        )
        case_tensors = {
            f"{name}.input_ids": input_ids.to(torch.int64),
            f"{name}.mm_token_type_ids": token_types.to(torch.int32),
            f"{name}.position_ids": position_ids.to(torch.int64),
            f"{name}.mrope_position_deltas": deltas.to(torch.int64),
        }
        if grid_thw is not None:
            case_tensors[f"{name}.image_grid_thw"] = grid_thw.to(torch.int64)
        tensors.update(case_tensors)
        components[name] = {
            "status": "ok",
            "seq_len": int(input_ids.shape[1]),
            "layout": layout,
            "num_images": 0 if grid_thw is None else int(grid_thw.shape[0]),
            "mrope_position_delta": int(deltas[0, 0].item()),
            "max_position": int(position_ids.max().item()),
            "tensors": {k: tensor_manifest_entry(v) for k, v in case_tensors.items()},
        }
        print(
            f"[rope_index_golden] {name}: seq_len={components[name]['seq_len']} "
            f"max_pos={components[name]['max_position']} delta={components[name]['mrope_position_delta']}"
        )

    out_path = args.out_dir / "rope_index.safetensors"
    save_golden(out_path, tensors, extra_meta={"component": "rope_index"})

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_source": config_source,
        "spatial_merge_size": merge,
        "image_token_id": image_token_id,
        "vision_start_token_id": vision_start_token_id,
        "vision_end_token_id": vision_end_token_id,
        "file": out_path.name,
        "semantics": {
            "position_ids": "[3, batch=1, seq_len] int64 -- the (t,h,w) mrope rows "
            "docs/architecture.md's [11,11,10]-section rope kernel consumes. A text run of length L "
            "starting at current_pos gets current_pos+arange(L) on ALL THREE rows; an image run gets "
            "t=current_pos constant and h/w ranging over the MERGED (h/merge, w/merge) grid, each "
            "offset by current_pos; after an image run current_pos advances by "
            "max(grid_h, grid_w)//merge -- the larger merged spatial extent, NOT the image's token "
            "count.",
            "mrope_position_deltas": "[batch, 1] int64 -- position_ids.max()+1 - seq_len. Decode "
            "step i (0-based, after the prompt) uses position seq_len + i + delta on all three rows.",
            "mm_token_type_ids": "0=text, 1=image, 2=video. The <vision_start>/<vision_end> tokens "
            "around an image are TEXT (0), only the image-placeholder tokens are 1.",
        },
        "components": components,
    }
    manifest_path = args.out_dir / "rope_index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[rope_index_golden] wrote {out_path}")
    print(f"[rope_index_golden] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
