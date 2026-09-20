"""tools/reference/vision_golden.py

Builds the Qwen3_5 vision tower (`Qwen3_5VisionModel`) straight from the reference `transformers`
install's `modeling_qwen3_5.py` -- real weights (`model.visual.*`, all 333 tensors present in the
real checkpoint), real preprocessing (`transformers.AutoImageProcessor`, `Qwen2VLImageProcessor`
per this checkpoint's `preprocessor_config.json`) -- runs a real image through it, and dumps every
intermediate `src/model`'s eventual vision-tower implementation needs to diff against. This is the
vision-tower analogue of `layer_golden.py` for the text decoder layers; see docs/validation.md
rung 3 and docs/status.md's vision-tower entry.

Usage:
    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\reference\\vision_golden.py ^
        --device cuda --out-dir tools\\reference\\golden_out

See tools/reference/README.md for the option list. Pass --image <path> for a real photo; without
it, a deterministic synthetic RGB test pattern is generated and saved next to the golden output
(<out-dir>/vision_test_image.png) so the run is reproducible and so a later end-to-end description
check (docs/validation.md rung 5) has a real, inspectable image to point r4dx-cli --image at.
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
    resolve_device,
    save_golden,
    set_seed,
    sha256_file,
    tensor_manifest_entry,
)

TOLERANCES = {
    "bf16_matmul_rel_err": 2e-2,
    "fp32_elementwise_rel_err": 1e-4,
    "note": "Same convention as layer_golden.py's TOLERANCES (docs/validation.md rung 3). "
    "bf16_matmul_rel_err for anything ending in a r4d_gemm_*/r4d_attn_vit_h72_bf16 call "
    "(patch_embed's conv3d-as-matmul, qkv/proj, mlp fc1/fc2, merger fc1/fc2); "
    "fp32_elementwise_rel_err for LayerNorm/rope/interpolation/gelu, computed in fp32 by the "
    "reference (Qwen3_5VisionAttention's apply_rotary_pos_emb_vision explicitly upcasts to "
    "float32; nn.LayerNorm accumulates in the input dtype but its variance/mean epsilon math is "
    "well-conditioned at fp32-comparable precision for hidden_size=1152). No layout variants: "
    "docs/container-format.md's vision.* tensors are bf16 passthrough only, never quantized, so "
    "there is exactly one numeric path to validate (bf16 compute vs this fp32/bf16 reference), "
    "unlike the text side's mxfp4/w4a16/w4a8 fan-out.",
}


def make_synthetic_image(width: int, height: int, seed: int):
    """A deterministic, non-trivial RGB test pattern: a horizontal/vertical gradient plus a
    checkerboard overlay plus a centered circle -- enough structure that patch embedding and
    position-embedding interpolation are exercised across smoothly-varying and sharp-edge content,
    unlike flat noise (which would make a resize/normalize bug in the preprocessing path much
    harder to notice by eye when someone inspects <out-dir>/vision_test_image.png later)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    grad_r = (xx / max(width - 1, 1) * 255).astype(np.float32)
    grad_g = (yy / max(height - 1, 1) * 255).astype(np.float32)
    checker = (((xx // 16) + (yy // 16)) % 2 * 255).astype(np.float32)
    cy, cx = height / 2.0, width / 2.0
    radius = min(width, height) / 3.0
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(np.float32) * 255

    r = 0.5 * grad_r + 0.3 * checker + 0.2 * circle
    g = 0.5 * grad_g + 0.3 * checker + 0.2 * (255 - circle)
    b = 0.4 * (255 - grad_r) + 0.3 * (255 - grad_g) + 0.3 * checker
    noise = rng.normal(0, 4.0, size=(height, width, 3))
    img = np.stack([r, g, b], axis=-1) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


class Rung3VisionCapture:
    """Forward hooks + a module-level `apply_rotary_pos_emb_vision` monkeypatch, capturing block 0's
    (the first encoder layer's) full intermediate chain at kernel granularity, mirroring how
    layer_golden.py's GdnCapture/attention hooks expose what a plain nn.Module forward hides."""

    def __init__(self, m):
        self.m = m
        self.record: dict[str, torch.Tensor] = {}
        self._orig_rope = m.apply_rotary_pos_emb_vision

    def __enter__(self):
        rec = self.record

        def wrapped_rope(q, k, cos, sin):
            q_out, k_out = self._orig_rope(q, k, cos, sin)
            # Only capture the first call (block 0) -- every block reuses the same cos/sin table,
            # this hook fires once per block per forward pass.
            if "attn_q_post_rope" not in rec:
                rec["attn_q_post_rope"] = q_out.detach().clone()
                rec["attn_k_post_rope"] = k_out.detach().clone()
            return q_out, k_out

        self.m.apply_rotary_pos_emb_vision = wrapped_rope

        def hook(name):
            def _fn(module, inputs, output):
                if name not in rec:
                    rec[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()

            return _fn

        block0 = self.m_ref.blocks[0]
        self._handles = [
            block0.norm1.register_forward_hook(hook("block0_norm1_out")),
            block0.attn.qkv.register_forward_hook(hook("block0_attn_qkv_raw")),
            block0.attn.proj.register_forward_hook(hook("block0_attn_proj_out")),
            block0.norm2.register_forward_hook(hook("block0_norm2_out")),
            block0.mlp.linear_fc1.register_forward_hook(hook("block0_mlp_fc1_out")),
            block0.mlp.linear_fc2.register_forward_hook(hook("block0_mlp_fc2_out")),
        ]
        return self

    def bind_model(self, model):
        self.m_ref = model
        return self

    def __exit__(self, *exc):
        self.m.apply_rotary_pos_emb_vision = self._orig_rope
        for h in self._handles:
            h.remove()


def run_vision_tower(
    m,
    vision_config,
    index: ShardIndex,
    device,
    dtype,
    image,
    seed: int,
) -> tuple[dict, dict]:
    from transformers import AutoImageProcessor

    set_seed(seed)
    vision_config._attn_implementation = "eager"
    model = m.Qwen3_5VisionModel(vision_config)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    loaded, missing, error = load_module_state(model, index, "model.visual.")
    weights_source = "safetensors" if loaded else f"random_init(seed={seed})"

    proc = AutoImageProcessor.from_pretrained(str(index.model_dir))
    processed = proc(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float32)
    grid_thw = processed["image_grid_thw"].to(device=device)

    tensors: dict[str, torch.Tensor] = {
        "pixel_values": pixel_values.detach().clone(),
        "image_grid_thw": grid_thw.detach().clone().to(torch.int64),
    }

    # Block-by-block: capture every encoder layer's OUTPUT (27 tensors, [num_patches, 1152] each --
    # ~3.6 MB fp32 apiece at a 28x28-patch image, ~97 MB total, fine for a gitignored golden dump)
    # by wrapping each block's forward, plus block 0's full internal chain via Rung3VisionCapture.
    block_outputs: dict[str, torch.Tensor] = {}
    orig_blocks_forward = [blk.forward for blk in model.blocks]

    def make_block_wrapper(idx, orig_fwd):
        def wrapped(*args, **kwargs):
            out = orig_fwd(*args, **kwargs)
            block_outputs[f"block_{idx:02d}_output"] = out.detach().clone()
            return out

        return wrapped

    for i, blk in enumerate(model.blocks):
        blk.forward = make_block_wrapper(i, orig_blocks_forward[i])

    cap = Rung3VisionCapture(m).bind_model(model)
    with torch.no_grad(), cap:
        result = model(hidden_states=pixel_values, grid_thw=grid_thw)

    for i, blk in enumerate(model.blocks):
        blk.forward = orig_blocks_forward[i]

    tensors.update(block_outputs)
    tensors.update(cap.record)
    tensors["last_hidden_state"] = result.last_hidden_state.detach().clone()
    tensors["merger_output"] = result.pooler_output.detach().clone()

    meta = {
        "status": "ok",
        "weights_source": weights_source,
        "missing_keys": missing[:20],
        "num_missing_keys": len(missing),
        "image_size": list(image.size),
        "grid_thw": grid_thw.detach().cpu().tolist(),
        "num_patches": int(pixel_values.shape[0]),
        "num_merged_tokens": int(result.pooler_output.shape[0]),
        "architecture_notes": {
            "patch_embed": "Conv3d(in_channels=3, embed_dim=1152, kernel=[temporal_patch_size=2, "
            "patch_size=16, patch_size=16], stride=same) applied to hidden_states.view(-1, 3, 2, 16, "
            "16), i.e. a dense [1536 -> 1152] matmul per patch row once flattened -- pixel_values is "
            "already patchified+flattened by the image processor, r4dx's stb_image path must "
            "reproduce that exact patchify+normalize (mean/std 0.5/0.5/0.5, resize to a multiple of "
            "patch_size*spatial_merge_size=32 per side within [shortest_edge, longest_edge]) before "
            "this matmul, not a Conv3d kernel.",
            "pos_embed": "bilinear-interpolated from a learned (num_grid_per_side=48)^2 x 1152 "
            "table via get_vision_interpolation_indices_and_weights (4-tap gather+weighted-sum, "
            "align_corners=True) -- NOT a fixed sinusoidal table, must be replicated exactly "
            "(interp_indices/interp_weights are not dumped here since r4dx's own patch-embed test "
            "should recompute them from grid_thw the same way; flag if this turns out insufficient).",
            "rope": "axial 2D rope, head_dim=72, first half of head_dim rotates h-position freqs, "
            "second half rotates w-position freqs (see Qwen3_5VisionRotaryEmbedding."
            "recomposition_frequencies), theta=10000 (vision's own rope_theta, DIFFERENT from the "
            "text side's 1e7) -- full head_dim rotates (no partial_rotary_factor, unlike text).",
            "attention": "dense non-causal attention per image via cu_seqlens (one segment per "
            "image here, single-image golden) -- exactly r4d_attn_vit_h72_bf16's contract.",
            "merger": "hidden_states.view(-1, hidden_size*spatial_merge_size**2) [784,1152] -> "
            "[196,4608] relies on the image processor already emitting patches in spatial-merge-"
            "block order (2x2 blocks contiguous), THEN LayerNorm(4608) -> Linear(4608,4608) -> GELU "
            "-> Linear(4608,5120). No reorder happens in the model itself.",
        },
    }
    return tensors, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "golden_out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image", type=Path, default=None, help="real image path; default: synthetic")
    ap.add_argument("--image-width", type=int, default=448)
    ap.add_argument("--image-height", type=int, default=448)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    full_config = Qwen3_5Config.from_pretrained(str(args.model_dir))
    vision_config = full_config.vision_config
    config_sha256 = sha256_file(args.model_dir / "config.json")
    index = ShardIndex.load(args.model_dir)

    if args.image is not None:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        image_source = str(args.image)
    else:
        image = make_synthetic_image(args.image_width, args.image_height, args.seed)
        saved_path = args.out_dir / "vision_test_image.png"
        image.save(saved_path)
        image_source = f"synthetic(seed={args.seed}, size={args.image_width}x{args.image_height}) -> {saved_path.name}"

    tensors, meta = run_vision_tower(m, vision_config, index, device, dtype, image, args.seed)

    out_path = args.out_dir / "vision_tower.safetensors"
    save_golden(out_path, tensors, extra_meta={"component": "vision_tower"})
    meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
    meta["file"] = out_path.name
    meta["image_source"] = image_source

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "config_sha256": config_sha256,
        "device": str(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "seed": args.seed,
        "tolerances": TOLERANCES,
        "components": {"vision_tower": meta},
    }
    manifest_path = args.out_dir / "vision_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[vision_golden] vision_tower: status={meta['status']} weights={meta['weights_source']}")
    print(f"[vision_golden] grid_thw={meta['grid_thw']} patches={meta['num_patches']} merged_tokens={meta['num_merged_tokens']}")
    print(f"[vision_golden] wrote {out_path}")
    print(f"[vision_golden] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
