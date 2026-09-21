"""tools/reference/vision_golden.py

Builds the Qwen3_5 vision tower (`Qwen3_5VisionModel`) straight from the reference `transformers`
install's `modeling_qwen3_5.py` -- real weights (`model.visual.*`, all 333 tensors present in the
real checkpoint), real preprocessing (`transformers.AutoImageProcessor`, `Qwen2VLImageProcessorFast`
per this checkpoint's `preprocessor_config.json`) -- runs real images through it, and dumps every
intermediate `src/vision`/`src/model`'s eventual vision-tower implementation needs to diff against.
This is the vision-tower analogue of `layer_golden.py` for the text decoder layers; see
docs/validation.md rung 3 and docs/status.md's vision-tower entry.

Three cases are produced, each its own `.safetensors` file + `vision_manifest.json` component:

  vision_tower             one square 448x448 image (grid 28x28, no resize -- smart_resize is the
                           identity at a multiple-of-32 side). The deep case: every encoder block's
                           output plus block 0's internal chain, as before.
  vision_tower_nonsquare   one 613x409 (WxH) image -> 608x416, i.e. width DOWN-sampled and height
                           UP-sampled by smart_resize, h != w, neither side a multiple of 32 to
                           begin with. This is the case that actually exercises the resampler and
                           the h!=w position/interpolation math; the square case does not resize at
                           all. Front-end tensors + merger output only (the 27 per-block dumps
                           would be ~100 MB of redundant coverage).
  vision_tower_two_image   both images in ONE processor call and ONE tower forward: `cu_seqlens`
                           has two segments with different grids, `pixel_values` is the
                           concatenation, and the pos-embed/rope index math has to restart per
                           image. Front-end tensors + merger output.

Usage (reference venv's python; --device cuda additionally requires $env:HIP_VISIBLE_DEVICES='1'):
    <reference-venv>\\Scripts\\python.exe tools\\reference\\vision_golden.py ^
        --device cuda --out-dir tools\\reference\\golden_out

See tools/reference/README.md for the option list. Pass --image/--image2 for real photos; without
them, deterministic synthetic RGB test patterns are generated and saved next to the golden output
(<out-dir>/vision_test_image.png, <out-dir>/vision_test_image_nonsquare.png) so the run is
reproducible, so the C++ preprocessing unit test has the exact same pixels to decode, and so a
later end-to-end description check (docs/validation.md rung 5) has a real, inspectable image to
point r4dx-cli --image at.

The `vision_tower` component additionally carries three small CPU-only fixture groups that pin the
preprocessing claims nothing else in this golden would catch: PIL's own `convert("RGB")` output for
an alpha-ramp RGBA and a greyscale PNG (`make_channel_conversion_fixtures`), four random-noise
resize pairs run through the reference's exact `tvF.resize` call (`make_resampler_fixtures`), and
the same picture written as BMP/GIF/JPEG with PIL's decode of each (`make_format_fixtures`).
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
    "unlike the text side's mxfp4/w4a16/w4a8 fan-out. `pixel_values` is NOT covered by either "
    "number: host-side preprocessing is integer-exact against this golden (the reference's own "
    "uint8 resampler is reproducible bit-for-bit -- see docs/vision.md 'Resampling'), so its test "
    "asserts equality, not a tolerance.",
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


def make_channel_conversion_fixtures(out_dir: Path, seed: int) -> tuple[dict, dict]:
    """An RGBA image with a full alpha ramp and an 8-bit greyscale image, saved as PNGs, plus what
    the reference's own `do_convert_rgb` step (PIL's `Image.convert("RGB")`) turns each of them
    into. This is the ground truth for the claim docs/vision.md makes about alpha -- that PIL DROPS
    it rather than compositing over a background -- which is otherwise a reading of PIL's source
    that nothing in this repo would catch if it were wrong, and which decides whether every pixel
    of a transparent PNG is right or badly wrong. `tests/vision/test_preprocess.cpp` asserts
    stb_image's req_comp=3 output is byte-identical to these."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    h, w = 48, 64
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    rgba[..., 1] = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    rgba[..., 2] = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    # Alpha sweeps the full 0..255 range, including fully transparent pixels whose RGB would be
    # unrecoverable if anything composited them.
    yy, xx = np.mgrid[0:h, 0:w]
    rgba[..., 3] = ((xx * 255) // max(w - 1, 1)).astype(np.uint8)
    rgba_path = out_dir / "vision_test_image_rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)

    grey = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    grey_path = out_dir / "vision_test_image_grey.png"
    Image.fromarray(grey, mode="L").save(grey_path)

    tensors = {
        "rgba_pil_rgb": torch.from_numpy(
            np.array(Image.open(rgba_path).convert("RGB"), dtype=np.uint8).copy()
        ),
        "grey_pil_rgb": torch.from_numpy(
            np.array(Image.open(grey_path).convert("RGB"), dtype=np.uint8).copy()
        ),
    }
    meta = {
        "rgba_png": rgba_path.name,
        "grey_png": grey_path.name,
        "note": "PIL Image.convert('RGB') output for each. RGBA -> RGB drops the alpha band (no "
        "compositing); L -> RGB replicates the single channel. stb_image with req_comp=3 must "
        "agree byte for byte.",
    }
    return tensors, meta


def make_resampler_fixtures(out_dir: Path, seed: int) -> tuple[dict, dict]:
    """Random-noise resize fixtures: four (in -> out) size pairs run through the EXACT call the
    reference backend makes (`tvF.resize(uint8_chw, [h, w], BICUBIC, antialias=True)` -- confirmed
    by instrumenting `TorchvisionBackend.resize` during a real processor call, which sees a uint8
    CPU tensor and BICUBIC), with the input saved as a PNG so the C++ side decodes the same bytes.

    Random noise, not the structured test pattern, on purpose: it is the worst case for any
    rounding or kernel disagreement, so a resampler that differs from the reference in the cubic
    coefficient, the intermediate dtype or the pass order cannot hide in smooth gradients. The two
    golden photographs only exercise one resize between them (448x448 does not resize at all), and
    neither exercises upscaling or a strong downscale where antialiasing widens the kernel most --
    these four pairs do. docs/vision.md "Resampling" quotes the numbers this fixture pins."""
    import numpy as np
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvF

    # (in_w, in_h, out_w, out_h): both down / the real smart_resize pair (w down, h up) / both up /
    # a strong downscale, where support = 2*scale spans many more taps.
    cases = [(640, 416, 608, 384), (613, 409, 608, 416), (140, 100, 288, 224), (1280, 960, 320, 256)]
    rng = np.random.default_rng(seed + 7)
    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for i, (iw, ih, ow, oh) in enumerate(cases):
        noise = rng.integers(0, 256, size=(ih, iw, 3), dtype=np.uint8)
        png = out_dir / f"vision_resize_noise_{i}.png"
        Image.fromarray(noise, mode="RGB").save(png)
        # Decode the PNG back rather than resizing the in-memory array: the C++ test resizes what
        # stb_image decodes, so the golden must start from the same bytes on disk.
        src = np.array(Image.open(png).convert("RGB"), dtype=np.uint8)
        chw = torch.from_numpy(np.ascontiguousarray(np.transpose(src, (2, 0, 1))))
        out = tvF.resize(chw, [oh, ow], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tensors[f"resize_noise_{i}_out"] = torch.from_numpy(
            np.ascontiguousarray(np.transpose(out.numpy(), (1, 2, 0)))
        )
        entries.append({"png": png.name, "tensor": f"resize_noise_{i}_out",
                        "in_wh": [iw, ih], "out_wh": [ow, oh]})
    meta = {
        "cases": entries,
        "call": "tvF.resize(uint8 CHW tensor, [out_h, out_w], "
        "interpolation=InterpolationMode.BICUBIC, antialias=True) -- the same call "
        "TorchvisionBackend.resize makes for this checkpoint (resample=PILImageResampling.BICUBIC "
        "= 3, antialias defaulted True, uint8 CPU tensor in and out).",
        "note": "Output tensors are HWC uint8 so the C++ test compares them directly against its "
        "own row-major interleaved buffer. Equality is expected, not a tolerance.",
    }
    return tensors, meta


def make_format_fixtures(out_dir: Path, seed: int) -> tuple[dict, dict]:
    """The same small picture written as BMP, GIF and JPEG, plus PIL's `convert("RGB")` output for
    each. src/vision/image_decode.h claims stb_image handles PNG/JPEG/BMP/GIF-first-frame; only PNG
    was ever actually exercised. BMP and GIF are lossless given the file (GIF through a 256-colour
    palette both decoders read identically), so those must match byte for byte; JPEG cannot -- stb's
    IDCT is not libjpeg's -- so its test asserts a small bound and reports the measured error."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed + 11)
    h, w = 40, 56
    yy, xx = np.mgrid[0:h, 0:w]
    # Smooth gradients plus a few flat blocks: compressible enough that JPEG stays close, and
    # few enough distinct colours that the GIF palette is lossless.
    img = np.stack(
        [
            (xx * 255 // max(w - 1, 1)).astype(np.uint8),
            (yy * 255 // max(h - 1, 1)).astype(np.uint8),
            (((xx // 8) + (yy // 8)) % 2 * 200).astype(np.uint8),
        ],
        axis=-1,
    )
    img[4:12, 4:20] = rng.integers(0, 256, size=(3,), dtype=np.uint8)
    pil = Image.fromarray(img, mode="RGB")

    tensors: dict[str, torch.Tensor] = {}
    entries = []
    for ext, tensor_name, kwargs in [
        ("bmp", "bmp_pil_rgb", {}),
        ("gif", "gif_pil_rgb", {}),
        ("jpg", "jpeg_pil_rgb", {"quality": 95}),
    ]:
        path = out_dir / f"vision_test_image_format.{ext}"
        pil.save(path, **kwargs)
        decoded = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        tensors[tensor_name] = torch.from_numpy(decoded.copy())
        entries.append({"file": path.name, "tensor": tensor_name})
    meta = {
        "cases": entries,
        "note": "PIL's own convert('RGB') of each file. stb_image (req_comp=3) must match BMP and "
        "GIF exactly; JPEG only approximately (different IDCT), so its test bounds the error "
        "instead of demanding equality.",
    }
    return tensors, meta


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


class FrontEndCapture:
    """The tensors between `pixel_values` and encoder block 0, which a plain forward hides but
    which a from-scratch implementation has to reproduce one at a time before any block output can
    possibly match: the patch-embed GEMM's own output and the rope cos/sin table the blocks are
    handed. (Block 0's own input is captured by wrapping its `forward` in run_case instead of by a
    forward-pre-hook -- Qwen3_5VisionBlock derives from GradientCheckpointingLayer, which overrides
    `__call__`, so a module-level pre-hook on it is not a reliable capture point.)"""

    def __init__(self, model):
        self.model = model
        self.record: dict[str, torch.Tensor] = {}

    def __enter__(self):
        rec = self.record

        def patch_embed_hook(module, inputs, output):
            rec.setdefault("patch_embed_out", output.detach().clone())

        def rope_hook(module, inputs, output):
            if "rope_cos" not in rec:
                rec["rope_cos"] = output[0].detach().clone()
                rec["rope_sin"] = output[1].detach().clone()

        self._handles = [
            self.model.patch_embed.register_forward_hook(patch_embed_hook),
            self.model.rotary_pos_emb.register_forward_hook(rope_hook),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def vision_index_tensors(model, vision_config, grid_thw) -> dict[str, torch.Tensor]:
    """The three index/weight tensors `Qwen3_5VisionModel.forward` derives from `grid_thw` alone,
    computed by calling the REAL `transformers.vision_utils` helpers with exactly the arguments the
    model passes them -- plus the position embedding they produce. r4dx recomputes all of these from
    `grid_thw` on the host; dumping them (rather than only the tensors downstream of them) is what
    lets a C++ unit test localize an index bug to the index math instead of only seeing a wrong
    block-0 input."""
    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=model.num_grid_per_side,
        mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners,
        spatial_merge_size=vision_config.spatial_merge_size,
    )
    # Exactly Qwen3_5VisionModel.forward's own line -- a 4-tap gather out of the learned table
    # weighted by interp_weights, summed over the taps.
    pos_embeds = (model.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)

    # cu_seqlens via the model's OWN call (`get_vision_attention_seqlens`, not the bare
    # `get_vision_cu_seqlens` an earlier version of this script dumped). The wrapper only forwards
    # `merge_temporal` (False here, i.e. one segment per FRAME) and adds `max_seqlen`, so the two
    # agree today -- but the golden should be what the model actually hands its attention, not a
    # lookalike, so the assert below is what keeps that true if the wrapper ever grows a rule.
    cu_seqlens, _max_seqlen = get_vision_attention_seqlens(grid_thw, vision_config)
    assert torch.equal(cu_seqlens, get_vision_cu_seqlens(grid_thw)), (
        "get_vision_attention_seqlens no longer agrees with get_vision_cu_seqlens for this config; "
        "src/vision/vision_index.cpp's BuildCuSeqlens reproduces the latter's rule"
    )
    return {
        "interp_indices": interp_indices.detach().clone().to(torch.int64),
        "interp_weights": interp_weights.detach().clone().to(torch.float32),
        "pos_embeds": pos_embeds.detach().clone().to(torch.float32),
        "vision_position_ids": get_vision_position_ids(grid_thw, model.spatial_merge_size)
        .detach()
        .clone()
        .to(torch.int64),
        "cu_seqlens": cu_seqlens.detach().clone().to(torch.int32),
    }


def run_case(
    m,
    model,
    vision_config,
    images: list,
    proc,
    device,
    capture_blocks: bool,
) -> tuple[dict, dict]:
    """One processor call + one tower forward over `images` (a list, so the two-image case is the
    same code path as the single-image ones, not a special case)."""
    processed = proc(images=images, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float32)
    grid_thw = processed["image_grid_thw"].to(device=device)

    tensors: dict[str, torch.Tensor] = {
        "pixel_values": pixel_values.detach().clone(),
        "image_grid_thw": grid_thw.detach().clone().to(torch.int64),
    }
    tensors.update(vision_index_tensors(model, vision_config, grid_thw))

    block_outputs: dict[str, torch.Tensor] = {}
    block_input: dict[str, torch.Tensor] = {}
    orig_blocks_forward = [blk.forward for blk in model.blocks]

    def make_block_wrapper(idx, orig_fwd):
        def wrapped(*args, **kwargs):
            if idx == 0:
                # patch_embed_out + the interpolated position embedding, in the model's own dtype.
                block_input.setdefault("block_input", args[0].detach().clone())
            out = orig_fwd(*args, **kwargs)
            if capture_blocks:
                # Every encoder layer's OUTPUT (27 tensors, [num_patches, 1152] each -- ~1.8 MB
                # bf16 apiece at a 28x28-patch image, fine for a gitignored golden dump).
                block_outputs[f"block_{idx:02d}_output"] = out.detach().clone()
            return out

        return wrapped

    for i, blk in enumerate(model.blocks):
        if i == 0 or capture_blocks:
            blk.forward = make_block_wrapper(i, orig_blocks_forward[i])

    cap = Rung3VisionCapture(m).bind_model(model)
    front = FrontEndCapture(model)
    with torch.no_grad(), front:
        if capture_blocks:
            with cap:
                result = model(hidden_states=pixel_values, grid_thw=grid_thw)
        else:
            result = model(hidden_states=pixel_values, grid_thw=grid_thw)

    for i, blk in enumerate(model.blocks):
        blk.forward = orig_blocks_forward[i]
    if capture_blocks:
        tensors.update(block_outputs)
        tensors.update(cap.record)

    tensors.update(block_input)
    tensors.update(front.record)
    tensors["last_hidden_state"] = result.last_hidden_state.detach().clone()
    tensors["merger_output"] = result.pooler_output.detach().clone()

    meta = {
        "status": "ok",
        "image_sizes": [list(im.size) for im in images],
        "grid_thw": grid_thw.detach().cpu().tolist(),
        "num_patches": int(pixel_values.shape[0]),
        "num_merged_tokens": int(result.pooler_output.shape[0]),
        "captured_blocks": bool(capture_blocks),
    }
    return tensors, meta


ARCHITECTURE_NOTES = {
    "preprocessing": "Qwen2VLImageProcessorFast (this checkpoint's configured processor) over a "
    "TorchvisionBackend: PIL convert('RGB') (which DROPS an alpha channel, it does not composite "
    "it), smart_resize to a multiple of patch_size*merge_size=32 within [shortest_edge=65536, "
    "longest_edge=16777216] TOTAL PIXELS, then tvF.resize(uint8, BICUBIC, antialias=True) -- an "
    "integer, two-pass, uint8-intermediate resampler with Pillow's a=-0.5 cubic kernel, NOT a "
    "float bilinear resize (see docs/vision.md 'Resampling'), then rescale+normalize with "
    "mean/std 0.5 fused as (u8 - 127.5)/127.5 in fp32, then patchify into [num_patches, "
    "channel*temporal*patch*patch = 1536] in 2x2 spatial-merge-block-major patch order with the "
    "single frame duplicated across temporal_patch_size=2.",
    "patch_embed": "Conv3d(in_channels=3, embed_dim=1152, kernel=[temporal_patch_size=2, "
    "patch_size=16, patch_size=16], stride=same) applied to hidden_states.view(-1, 3, 2, 16, 16), "
    "i.e. a dense [1536 -> 1152] matmul per patch row once flattened.",
    "pos_embed": "bilinear-interpolated from a learned (num_grid_per_side=48)^2 x 1152 table via "
    "get_vision_interpolation_indices_and_weights(mode='bilinear', align_corners=True, "
    "spatial_merge_size=2) -- NOT a fixed sinusoidal table. This golden dumps interp_indices/"
    "interp_weights/pos_embeds AND the raw pos_embed table, so a from-scratch implementation can "
    "be tested against the index math itself rather than only against its downstream effect.",
    "rope": "axial 2D rope, head_dim=72, inv_freq over head_dim/4=18 frequencies, theta=10000 "
    "(vision's own rope_theta, DIFFERENT from the text side's 1e7); cos/sin are "
    "cat([cos_h, cos_w, cos_h, cos_w]) -> [num_patches, 72], i.e. the FULL head_dim rotates (no "
    "partial_rotary_factor, unlike text).",
    "attention": "dense non-causal attention per image via cu_seqlens (one segment per image -- "
    "the two-image case here has two) -- exactly r4d_attn_vit_h72_bf16's contract.",
    "merger": "hidden_states.view(-1, hidden_size*spatial_merge_size**2) relies on the image "
    "processor already emitting patches in spatial-merge-block order (2x2 blocks contiguous), THEN "
    "LayerNorm(1152) -> view -> Linear(4608,4608) -> GELU -> Linear(4608,5120). No reorder happens "
    "in the model itself.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "golden_out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--image", type=Path, default=None, help="real image for case 1; default: synthetic")
    ap.add_argument("--image-width", type=int, default=448)
    ap.add_argument("--image-height", type=int, default=448)
    ap.add_argument("--image2", type=Path, default=None, help="real image for case 2; default: synthetic")
    # 613x409 on purpose: neither side is a multiple of 32, and smart_resize takes it to 608x416,
    # i.e. width DOWN and height UP in the same call, with h != w afterwards.
    ap.add_argument("--image2-width", type=int, default=613)
    ap.add_argument("--image2-height", type=int, default=409)
    # The attention implementation the REFERENCE tower runs. "eager" is the default and is what
    # every committed golden was produced with. It is worth knowing what it costs: eager attention
    # (transformers' own `eager_attention_forward`) computes `torch.matmul(q, k^T)` in bf16, so the
    # attention SCORES are rounded to bf16 before the `* scaling`, rounded again after it, and the
    # softmax probabilities are rounded to bf16 again before the P@V matmul. r4dx's
    # r4d_attn_vit_h72_bf16 keeps all three in fp32/f16, i.e. it is strictly MORE accurate than the
    # golden it is measured against. `--attn-impl sdpa` regenerates the same case through torch's
    # SDPA instead (fp32 accumulation, no bf16 score round-trip) so the two references can be
    # diffed against each other and that cost measured rather than assumed -- see docs/vision.md
    # "Why the block-26 disagreement is not an r4dx error". Output is written to --out-dir exactly
    # as usual, so point it at a scratch directory when comparing.
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    # Limits the run to the named cases (default: all three). Only useful with --attn-impl or when
    # iterating on one case; the committed golden is always a full run.
    ap.add_argument("--cases", default="", help="comma-separated subset of case names")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers import AutoImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    full_config = Qwen3_5Config.from_pretrained(str(args.model_dir))
    vision_config = full_config.vision_config
    config_sha256 = sha256_file(args.model_dir / "config.json")
    index = ShardIndex.load(args.model_dir)

    def load_or_make(path, width, height, filename):
        if path is not None:
            from PIL import Image

            return Image.open(path).convert("RGB"), str(path)
        img = make_synthetic_image(width, height, args.seed)
        saved = args.out_dir / filename
        img.save(saved)
        return img, f"synthetic(seed={args.seed}, size={width}x{height}) -> {saved.name}"

    image1, image1_source = load_or_make(args.image, args.image_width, args.image_height, "vision_test_image.png")
    image2, image2_source = load_or_make(
        args.image2, args.image2_width, args.image2_height, "vision_test_image_nonsquare.png"
    )

    set_seed(args.seed)
    vision_config._attn_implementation = args.attn_impl
    model = m.Qwen3_5VisionModel(vision_config).to(device=device, dtype=dtype).eval()
    loaded, missing, error = load_module_state(model, index, "model.visual.")

    # `.to(dtype=bfloat16)` casts non-persistent BUFFERS too, which silently rounds
    # Qwen3_5VisionRotaryEmbedding's `inv_freq` table to bf16's ~3 significant decimal digits
    # (1/theta^(2/36) becomes 0.597656 instead of 0.599484). That defeats the rope module's own
    # explicit fp32 intent -- its forward wraps the cos/sin computation in
    # maybe_autocast(enabled=False) and calls .float() on both operands -- and at a 38-wide patch
    # grid it is worth up to 0.047 absolute on cos/sin, an order of magnitude more than any bf16
    # matmul noise downstream of it. r4dx's own rope kernels compute inv_freq in fp32 on device
    # (src/kernels/src/r4dx_kernels.hip), as does every mixed-precision training setup, so the
    # golden is generated with the table restored to fp32: otherwise every block output in it
    # would encode a quantization artifact no r4dx implementation should reproduce. See
    # docs/vision.md "Rope" for the measured numbers.
    # Recomputed, not widened: .float() on the already-rounded bf16 buffer would only restore the
    # dtype, not the digits. This calls the rope module's OWN parameter computation.
    fp32_inv_freq, _ = m.Qwen3_5VisionRotaryEmbedding.compute_axial_rope_parameters(vision_config)
    model.rotary_pos_emb.inv_freq = fp32_inv_freq.to(device=device, dtype=torch.float32)
    weights_source = "safetensors" if loaded else f"random_init(seed={args.seed})"

    proc = AutoImageProcessor.from_pretrained(str(index.model_dir))

    cases = [
        ("vision_tower", [image1], [image1_source], True),
        ("vision_tower_nonsquare", [image2], [image2_source], False),
        ("vision_tower_two_image", [image1, image2], [image1_source, image2_source], False),
    ]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = wanted - {c[0] for c in cases}
        if unknown:
            raise SystemExit(f"--cases: unknown case name(s) {sorted(unknown)}")
        cases = [c for c in cases if c[0] in wanted]

    components: dict[str, dict] = {}
    for name, images, sources, capture_blocks in cases:
        tensors, meta = run_case(m, model, vision_config, images, proc, device, capture_blocks)
        if name == "vision_tower":
            # The learned [2304, 1152] table itself, dumped once (5.3 MB bf16): the C++ pos-embed
            # index test gathers from THIS rather than from the container, so it can run without a
            # converted container on disk.
            tensors["pos_embed_table"] = model.pos_embed.weight.detach().clone()
            tensors["rope_inv_freq"] = model.rotary_pos_emb.inv_freq.detach().clone().to(torch.float32)
            channel_tensors, channel_meta = make_channel_conversion_fixtures(args.out_dir, args.seed)
            tensors.update(channel_tensors)
            meta["channel_conversion"] = channel_meta
            resize_tensors, resize_meta = make_resampler_fixtures(args.out_dir, args.seed)
            tensors.update(resize_tensors)
            meta["resampler_fixtures"] = resize_meta
            format_tensors, format_meta = make_format_fixtures(args.out_dir, args.seed)
            tensors.update(format_tensors)
            meta["format_fixtures"] = format_meta

        out_path = args.out_dir / f"{name}.safetensors"
        save_golden(out_path, tensors, extra_meta={"component": name})
        meta.update(
            {
                "file": out_path.name,
                "weights_source": weights_source,
                "missing_keys": missing[:20],
                "num_missing_keys": len(missing),
                "load_error": error,
                "image_sources": sources,
                "tensors": {k: tensor_manifest_entry(v) for k, v in tensors.items()},
            }
        )
        components[name] = meta
        print(
            f"[vision_golden] {name}: grid_thw={meta['grid_thw']} patches={meta['num_patches']} "
            f"merged_tokens={meta['num_merged_tokens']} -> {out_path.name}"
        )

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "config_sha256": config_sha256,
        "device": str(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "attn_impl": args.attn_impl,
        "seed": args.seed,
        "tolerances": TOLERANCES,
        "architecture_notes": ARCHITECTURE_NOTES,
        "components": components,
    }
    manifest_path = args.out_dir / "vision_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[vision_golden] weights={weights_source} missing_keys={len(missing)}")
    print(f"[vision_golden] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
