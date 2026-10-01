"""CPU-only golden for the Gemma 4 (gemma4_unified) image preprocessing (docs/gemma4-plan.md M2).

Runs transformers' Gemma4UnifiedImageProcessor (the checkpoint's own processor_config.json values:
rescale 1/255, no mean/std, bicubic antialiased resize, 16 px patches merged 3x3 into 48x48x3 HWC
rasters, positions (x, y)) on a few synthetic uint8 images and writes, per case, into
<out-dir>/gemma/vision_preproc/:

    case<N>.rgb   int32 width, int32 height, then height*width*3 uint8 (HWC, RGB)
    case<N>.pix   int32 max_soft_tokens, int32 n_tokens, then n_tokens*6912 float32
                  (the real, unpadded merged patches; HF pads to max_soft_tokens with zeros)
    case<N>.pos   int32 n_tokens*2 (x, y) of the real merged patches
    manifest.json shapes and the HF-side numbers

tests/vision/test_gemma_preprocess.cpp compares src/vision/gemma_vision.cpp's PreprocessGemmaImage against
these byte for byte. With --check it also re-derives the tensors with an independent numpy path from
the HF-resized image (resize via the same torch kernel; rescale / patchify / merge in numpy) and asserts
bit equality, which pins the two facts the C++ side relies on: the rescale is float32(u8) * float32(1/255),
and the merged patch is a plain 48x48 HWC raster of the merged-grid cell.

CPU only: HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES are set to '-1' before torch is imported.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\vision_preproc_golden.py --check
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available(), "GPU must stay hidden"

from transformers.models.gemma4_unified.image_processing_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedImageProcessor,
    get_aspect_ratio_preserving_size,
)

# (name, height, width, max_soft_tokens, kind). Sizes cover: a tiny image (UPSCALED: the target is the
# largest 48-multiple size inside the budget, not an upper bound only), square, non-square, up- and
# down-scaling, a very wide strip, and the other soft-token budgets. (480x672 is not a no-resize case:
# the budget is 645120 px, so it also grows.)
CASES = [
    ("sq48", 48, 48, 280, "noise"),
    ("grow480x672", 480, 672, 280, "noise"),
    ("wide", 300, 777, 280, "grad"),
    ("tall", 901, 233, 280, "noise"),
    ("big", 1080, 1920, 280, "grad"),            # downscale, hits the budget
    ("strip", 64, 2000, 280, "noise"),            # one row of merged patches
    ("small_up", 100, 130, 280, "noise"),
    ("exact672x912", 672, 912, 280, "noise"),     # already the target size: no resize at all
    ("t70", 512, 512, 70, "noise"),
    ("t140", 640, 480, 140, "grad"),
    ("t560", 1000, 1000, 560, "noise"),
]


def make_image(h, w, kind, seed):
    rng = np.random.RandomState(seed)
    if kind == "noise":
        return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)),
                    ((xx + yy) * 255 // max(h + w - 2, 1))], axis=-1).astype(np.uint8)
    img[::7, :, :] = rng.randint(0, 256, size=(len(range(0, h, 7)), w, 3), dtype=np.uint8)
    return img


def numpy_path(img_hwc, max_soft_tokens, pooling=3, patch=16):
    """Independent re-derivation from the HF-resized uint8 image."""
    import torchvision.transforms.v2.functional as tvF

    h, w = img_hwc.shape[:2]
    th, tw = get_aspect_ratio_preserving_size(h, w, patch, max_soft_tokens * pooling ** 2, pooling)
    t = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()
    if (th, tw) != (h, w):
        t = tvF.resize(t, size=[th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    rs = t.permute(1, 2, 0).numpy()  # [th, tw, 3] uint8
    f = rs.astype(np.float32) * np.float32(1.0 / 255.0)
    side = pooling * patch
    gh, gw = th // side, tw // side
    pix = f.reshape(gh, side, gw, side, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, side * side * 3)
    pos = np.array([[x, y] for y in range(gh) for x in range(gw)], dtype=np.int32)
    return pix, pos, (th, tw, gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "golden_out"))
    ap.add_argument("--check", action="store_true", help="also verify an independent numpy re-derivation")
    args = ap.parse_args()
    out = Path(args.out_dir) / "gemma" / "vision_preproc"
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, (name, h, w, mst, kind) in enumerate(CASES):
        img = make_image(h, w, kind, seed=1000 + i)
        proc = Gemma4UnifiedImageProcessor(max_soft_tokens=mst)
        res = proc(images=[torch.from_numpy(img).permute(2, 0, 1)], return_tensors="pt")
        pv = res["pixel_values"][0]            # [mst, 6912], zero padded
        pp = res["image_position_ids"][0]      # [mst, 2], -1 padded
        n = int(res["num_soft_tokens_per_image"][0])
        pix = pv[:n].to(torch.float32).numpy()
        pos = pp[:n].to(torch.int32).numpy()
        assert (pp[n:] == -1).all() and (pv[n:] == 0).all()
        assert pix.shape == (n, 6912), pix.shape
        if args.check:
            npix, npos, dims = numpy_path(img, mst)
            assert npix.shape == pix.shape, (name, npix.shape, pix.shape)
            assert np.array_equal(npix.view(np.uint32), pix.view(np.uint32)), f"{name}: pixel bits differ"
            assert np.array_equal(npos, pos), f"{name}: positions differ"
        with open(out / f"case{i}.rgb", "wb") as f:
            f.write(struct.pack("<ii", w, h))
            f.write(img.tobytes())
        with open(out / f"case{i}.pix", "wb") as f:
            f.write(struct.pack("<ii", mst, n))
            f.write(np.ascontiguousarray(pix).tobytes())
        with open(out / f"case{i}.pos", "wb") as f:
            f.write(np.ascontiguousarray(pos).tobytes())
        th, tw = get_aspect_ratio_preserving_size(h, w, 16, mst * 9, 3)
        manifest.append({"case": i, "name": name, "h": h, "w": w, "max_soft_tokens": mst, "n_tokens": n,
                         "target_hw": [th, tw], "grid_hw": [th // 48, tw // 48]})
        print(f"case{i} {name}: {h}x{w} -> {th}x{tw}, {n} tokens, max_soft_tokens {mst}"
              + (" [numpy path bit-equal]" if args.check else ""))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
