"""tiny-q35moe WITH the real vision tower (docs/qwen35moe-vision.md): a fixture for the image path of Qwen35MoeModel.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\tiny_vision.py attach [--out E:\\models\\r4dx\\tiny-q35moe-vision]
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\tiny_vision.py golden [--out ...] [--image PATH ...]

`attach` writes `<out>/hf`: the tiny text checkpoint (hard-linked, 4 layers, 16 experts, vocab 4096) plus a second shard with the 333 REAL
`model.visual.*` tensors of Ornith-1.5-122B (bf16, 0.9 GB) and a config with the real vision_config and tiny special-token ids
(vision_start 4088, vision_end 4089, image 4090, video 4091: the tiny vocabulary has no room for 248053..248057). r4dx-convert turns it
into a container that carries `vision.*` (`--vision on`), exactly as it will for the real model.

`golden` runs HF's Qwen3_5MoeForConditionalGeneration (bf16, CPU, K / V rounded to fp8 like r4dx's cache) over prompts that contain
images and writes `<out>/golden/{case}.safetensors` + `{case}.json`:
    input_ids i64 [T]; pixel_values f32 [P, 1536]; image_grid_thw i64 [n, 3]; image_embeds bf16 [tokens, 3072] (the merger output);
    position_ids i32 [3, T] (get_rope_index); logits bf16 [T, 4096]; layer{i}.residual_out bf16 [T, 3072] (+ embedding);
    json: the greedy continuation (FREE_STEPS tokens, full forward per step with the image) and the mrope delta.
Cases: one_image (64 image tokens, text before and after), two_images (two different grids, text between).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe import tiny, tiny_golden  # noqa: E402

REAL_DIR = Path(r"E:\models\r4dx\Ornith-1.5-122B")
TINY_HF = tiny.DEFAULT_OUT / "hf"
DEFAULT_OUT = Path(r"E:\models\r4dx\tiny-q35moe-vision")
START_ID, END_ID, IMAGE_ID, VIDEO_ID = 4088, 4089, 4090, 4091
FREE_STEPS = 12
SEED = 20261023


def attach(out: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    hf = out / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    # text shard: hard link (same volume) or copy
    text_src = TINY_HF / "model-00001-of-00001.safetensors"
    text_dst = hf / text_src.name
    if not text_dst.exists():
        try:
            os.link(text_src, text_dst)
        except OSError:
            import shutil

            shutil.copyfile(text_src, text_dst)
    # visual shard: every model.visual.* tensor of the real checkpoint (all in shard 1, checked against the index)
    real_index = json.loads((REAL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(k for k in real_index if k.startswith("model.visual."))
    shards = sorted({real_index[k] for k in names})
    tensors = {}
    for sh in shards:
        with safe_open(str(REAL_DIR / sh), framework="pt") as f:
            for k in names:
                if real_index[k] == sh:
                    tensors[k] = f.get_tensor(k).contiguous()
    assert len(tensors) == 333, len(tensors)
    vis_name = "model-visual.safetensors"
    save_file(tensors, str(hf / vis_name), metadata={"format": "pt"})
    index = json.loads((TINY_HF / "model.safetensors.index.json").read_text())
    index["weight_map"].update({k: vis_name for k in tensors})
    index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in tensors.values())
    (hf / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    cfg = json.loads((TINY_HF / "config.json").read_text())
    real_cfg = json.loads((REAL_DIR / "config.json").read_text())
    cfg["vision_config"] = real_cfg["vision_config"]
    cfg["image_token_id"], cfg["video_token_id"] = IMAGE_ID, VIDEO_ID
    cfg["vision_start_token_id"], cfg["vision_end_token_id"] = START_ID, END_ID
    (hf / "config.json").write_text(json.dumps(cfg, indent=2))
    for extra in ("generation_config.json",):
        if (TINY_HF / extra).exists():
            (hf / extra).write_text((TINY_HF / extra).read_text())
    print(f"[tiny_vision] {hf}: {len(tensors)} visual tensors, {sum(t.numel() for t in tensors.values()) * 2 / 2**30:.3f} GiB")


def load_model(hf: Path):
    from safetensors.torch import load_file
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    cfg = Qwen3_5MoeConfig.from_pretrained(str(hf))
    cfg._attn_implementation = "sdpa"
    try:
        cfg.text_config.experts_implementation = "eager"
    except Exception:
        pass
    model = Qwen3_5MoeForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    sd = load_file(str(hf / "model-00001-of-00001.safetensors"))
    sd.update(load_file(str(hf / "model-visual.safetensors")))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing {missing[:5]} unexpected {list(unexpected)[:5]}")
    # .to(bf16) rounds the vision rope's inv_freq buffer (docs/vision.md "Rope"): recompute it in fp32 like vision_golden.py
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m

    fp32_inv, _ = m.Qwen3_5MoeVisionRotaryEmbedding.compute_axial_rope_parameters(cfg.vision_config)
    model.model.visual.rotary_pos_emb.inv_freq = fp32_inv.to(torch.float32)
    return model.eval()


def synthetic_image(w: int, h: int, seed: int):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / max(w - 1, 1) * 255).astype(np.float32) * 0.6 + (((xx // 16) + (yy // 16)) % 2) * 80
    g = (yy / max(h - 1, 1) * 255).astype(np.float32) * 0.6 + (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) / 3) ** 2) * 100
    b = 255 - 0.5 * r - 0.5 * g
    img = np.clip(np.stack([r, g, b], -1) + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


@torch.no_grad()
def run_case(model, proc, images, text_before, text_between, text_after, name, out_dir: Path):
    from safetensors.torch import save_file

    pix = proc(images=images, return_tensors="pt")
    grid = pix["image_grid_thw"]
    merge = model.config.vision_config.spatial_merge_size
    counts = [int(g[0] * g[1] * g[2]) // (merge * merge) for g in grid]
    ids = list(text_before)
    for i, c in enumerate(counts):
        ids += [START_ID] + [IMAGE_ID] * c + [END_ID]
        if i + 1 < len(counts):
            ids += list(text_between)
    ids += list(text_after)
    x = torch.tensor([ids], dtype=torch.long)
    mm = (x == IMAGE_ID).to(torch.int32)
    kw = dict(pixel_values=pix["pixel_values"].to(torch.bfloat16), image_grid_thw=grid, mm_token_type_ids=mm, use_cache=False)

    cap = tiny_golden.Capture(model)
    pos_box = {}
    h = model.model.language_model.register_forward_pre_hook(lambda m, a, k: pos_box.__setitem__("p", k["position_ids"].detach().clone()), with_kwargs=True)
    emb_box = {}
    h2 = model.model.visual.register_forward_hook(lambda m, a, o: emb_box.__setitem__("e", (o.pooler_output if hasattr(o, "pooler_output") else o).detach().clone()))
    try:
        outp = model(input_ids=x, **kw)
    finally:
        cap.close()
        h.remove()
        h2.remove()
    t = dict(cap.t)
    t["input_ids"] = x[0].clone()
    t["pixel_values"] = pix["pixel_values"].to(torch.float32)
    t["image_grid_thw"] = grid.to(torch.int64)
    t["image_embeds"] = emb_box["e"]
    t["position_ids"] = pos_box["p"][:, 0, :].to(torch.int32)
    t["logits"] = outp.logits[0].detach().clone()
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() and k != "pixel_values" else v).contiguous() for k, v in t.items()},
              str(out_dir / f"{name}.safetensors"))
    # greedy continuation: a full forward per step (the image included), no cache
    cur = list(ids)
    new = []
    for _ in range(FREE_STEPS):
        xx = torch.tensor([cur], dtype=torch.long)
        o = model(input_ids=xx, pixel_values=kw["pixel_values"], image_grid_thw=grid,
                  mm_token_type_ids=(xx == IMAGE_ID).to(torch.int32), use_cache=False)
        nxt = int(o.logits[0, -1].float().argmax())
        new.append(nxt)
        cur.append(nxt)
    pmax = int(t["position_ids"].max())
    (out_dir / f"{name}.json").write_text(json.dumps({
        "prompt": ids, "continuation": new, "grid": grid.tolist(), "image_tokens": counts,
        "mrope_delta": pmax + 1 - len(ids), "image_token_id": IMAGE_ID,
    }))
    print(f"[golden] {name}: T={len(ids)} grids={grid.tolist()} tokens={counts} delta={pmax + 1 - len(ids)} free={new[:6]}...")


def golden(out: Path, image_paths: list[str]) -> None:
    from PIL import Image
    from transformers import AutoImageProcessor

    torch.manual_seed(0)
    model = load_model(out / "hf")
    impl = tiny_golden.install_fp8_kv_attention()
    model.config._attn_implementation = impl
    model.config.text_config._attn_implementation = impl
    for layer in model.model.language_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.config._attn_implementation = impl
    proc = AutoImageProcessor.from_pretrained(str(REAL_DIR))
    gdir = out / "golden"
    gdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)
    rnd = lambda n: torch.randint(0, 4000, (n,), generator=g).tolist()  # noqa: E731
    if image_paths:
        imgs = [Image.open(p).convert("RGB") for p in image_paths[:2]]
    else:
        imgs = [synthetic_image(256, 256, 1), synthetic_image(320, 224, 2)]
        imgs[0].save(gdir / "image_a.png")
        imgs[1].save(gdir / "image_b.png")
    run_case(model, proc, imgs[:1], rnd(9), [], rnd(11), "one_image", gdir)
    run_case(model, proc, imgs[:2], rnd(5), rnd(6), rnd(7), "two_images", gdir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["attach", "golden"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--image", action="append", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "attach":
        attach(Path(a.out))
    else:
        golden(Path(a.out), a.image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
