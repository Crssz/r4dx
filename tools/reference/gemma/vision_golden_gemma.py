"""tools/reference/gemma/vision_golden_gemma.py  (M2-1, vision golden)

Real-checkpoint golden for Gemma 4 (gemma4_unified) vision: transformers' own image processor, the REAL
`model.vision_embedder.*` + `model.embed_vision.*` weights, and decoder layers with an image block, all from the
reference `transformers` install (no r4dx code). Two images of different shapes (a non-square one and a wide
one, both at the 70-soft-token budget so the CPU run is quick) in one sequence:

    [bos t t] <|image> A... <image|> [t t] <|image> B... <image|> [t t t]

Components written to <out-dir>/vision_golden_gemma.safetensors (+ .json manifest):

  pixel_values_{A,B}   fp32 [n, 6912]   the unpadded merged patches (== r4dx PreprocessGemmaImage, bit for bit)
  positions_{A,B}      int32 [n, 2]     (x, y) merged-grid cells
  embed_{A,B}          bf16 [n, 3840]   HF Gemma4UnifiedVisionEmbedder pooler_output (bf16, CPU or GPU): the
                                        UNSCALED rows spliced into the residual stream
  token_ids            int32 [T]        the layout above (placeholders are 258880)
  mm_token_type_ids    int32 [T]        1 on image soft tokens
  x_in                 bf16 [T, 3840]   inputs_embeds: text rows = embed_tokens[id] * bf16(sqrt(3840)) (= 62.0),
                                        image rows = embed_{A,B} (masked_scatter, NOT scaled)
  klimit_ext           int32 [T]        per-row absolute key limit the r4dx sliding kernels take (-1 = causal)
  mask_sliding / mask_full  uint8 [T, T] the masks create_masks_for_generate built (1 = attend)
  layer0_block         bf16 [T, 3840]   layer 0 (sliding, window 1024) output with the image-block mask
  layer0_causal        bf16 [T, 3840]   the same layer with the plain causal sliding mask (control: must differ)
  layer5_full          bf16 [T, 3840]   layer 5 (full attention, head_dim 512, k_eq_v) output: causal
  layer5_full_if_bidir bf16 [T, 3840]   layer 5 with the (rejected) forward()-path mask that also overlays the block
                                        on full layers, so a real-image run can A/B the one-line mask switch

GPU commands (the user's; ask before running anything on a GPU):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\vision_golden_gemma.py --device cuda
    # whole-model logits for the same sequence (the end-to-end reference; ~24 GB of weights, GPU only):
    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\vision_golden_gemma.py --device cuda --full-logits

`--device cpu` (the default) is fine and needs ~3 GB: only the 10 vision tensors, the embedding rows of the 11
text ids and layers 0 and 5 are read, all with the non-mmap raw reader (ShardIndex), never the whole checkpoint.
The C++ test that consumes it is tests/model/test_gemma_vision_gpu.cpp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "--device" not in sys.argv or sys.argv[sys.argv.index("--device") + 1:][:1] in ([], ["cpu"]):
    # A CPU run must never see a GPU (rule: GPUs are busy with long jobs).
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gemma.arch import EMBED_NAME  # noqa: E402
from gemma.common_gemma import DEFAULT_MODEL_DIR, GOLDEN_DIR, resolve_device, save_golden  # noqa: E402
from gemma.ref import GemmaReference  # noqa: E402

BOI, EOI, IMG = 255999, 258882, 258880
LEN_TEXT = {"pre": [2, 105, 2364], "mid": [107, 4000], "post": [5000, 6000, 7000]}


def make_image(h, w, seed):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)),
                    ((xx * 3 + yy * 5) % 256)], axis=-1).astype(np.uint8)
    img = np.where(rng.rand(h, w, 1) < 0.15, rng.randint(0, 256, size=(h, w, 3)), img).astype(np.uint8)
    return img


def build_embedder(ref: GemmaReference, device):
    from transformers.models.gemma4_unified import modeling_gemma4_unified as M

    cfg = ref.config
    emb = M.Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config)
    idx = ref.index
    state = {}
    for k in ("patch_ln1.weight", "patch_ln1.bias", "patch_dense.weight", "patch_dense.bias", "patch_ln2.weight",
              "patch_ln2.bias", "pos_norm.weight", "pos_norm.bias"):
        state[k] = idx.get_tensor("model.vision_embedder." + k)
    state["pos_embedding"] = idx.get_tensor("model.vision_embedder.pos_embedding")
    state["multimodal_embedder.embedding_projection.weight"] = idx.get_tensor(
        "model.embed_vision.embedding_projection.weight")
    emb = emb.to(torch.bfloat16)
    missing = emb.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys, missing
    return emb.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(GOLDEN_DIR))
    ap.add_argument("--max-soft-tokens", type=int, default=70)
    ap.add_argument("--full-logits", action="store_true", help="also run ALL layers (GPU only) and dump the last-row logits")
    a = ap.parse_args()
    device = resolve_device(a.device)
    if device.type == "cpu":
        assert not torch.cuda.is_available(), "a CPU run must not see a GPU"
    if a.full_logits and device.type != "cuda":
        sys.exit("--full-logits needs --device cuda (the 12B model does not fit the CPU RAM budget)")

    from transformers.models.gemma4_unified import modeling_gemma4_unified as M
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor

    ref = GemmaReference(a.model_dir, device, load_table=False, verbose=False)
    emb = build_embedder(ref, device)

    proc = Gemma4UnifiedImageProcessor(max_soft_tokens=a.max_soft_tokens)
    images = {"A": make_image(336, 240, 11), "B": make_image(192, 480, 12)}   # non-square, tall and wide
    tensors, meta = {}, {"max_soft_tokens": a.max_soft_tokens, "transformers": __import__("transformers").__version__}
    embeds = {}
    for k, img in images.items():
        res = proc(images=[torch.from_numpy(img).permute(2, 0, 1)], return_tensors="pt")
        n = int(res["num_soft_tokens_per_image"][0])
        pv = res["pixel_values"][0][:n].to(torch.float32)
        pp = res["image_position_ids"][0][:n].to(torch.int32)
        with torch.no_grad():
            out = emb(pixel_values=pv[None].to(device), image_position_ids=pp[None].to(device).long(),
                      return_dict=True).pooler_output[0]
        tensors[f"pixel_values_{k}"] = pv
        tensors[f"positions_{k}"] = pp
        tensors[f"embed_{k}"] = out.to(torch.bfloat16).cpu()
        embeds[k] = out.to(torch.bfloat16)
        meta[f"image_{k}"] = {"hw": list(img.shape[:2]), "soft_tokens": n,
                              "embed_rms": float(out.float().pow(2).mean().sqrt())}

    # ---- the sequence ----
    ids, types, spans = [], [], {}
    ids += LEN_TEXT["pre"]
    types += [0] * len(LEN_TEXT["pre"])
    for k, between in (("A", LEN_TEXT["mid"]), ("B", LEN_TEXT["post"])):
        n = embeds[k].shape[0]
        ids += [BOI]
        types += [0]
        spans[k] = (len(ids), len(ids) + n)
        ids += [IMG] * n
        types += [1] * n
        ids += [EOI]
        types += [0]
        ids += between
        types += [0] * len(between)
    T = len(ids)
    meta["T"], meta["spans"] = T, spans

    # inputs_embeds: scaled text rows, UNSCALED image rows (masked_scatter)
    scale = torch.tensor(ref.arch.hidden ** 0.5).to(torch.bfloat16)
    rows = []
    for t in ids:
        if t == IMG:
            rows.append(torch.zeros(ref.arch.hidden, dtype=torch.bfloat16))
        else:
            rows.append(ref.index.get_row_slice(EMBED_NAME, t, t + 1)[0].to(torch.bfloat16))
    x = (torch.stack(rows) * scale)[None].to(device)
    ids_t = torch.tensor(ids)
    mask_img = (ids_t == IMG)[None, :, None].to(device).expand_as(x)
    x = x.masked_scatter(mask_img, torch.cat([embeds["A"], embeds["B"]]).to(device))
    for k, (s, e) in spans.items():
        assert torch.equal(x[0, s:e].cpu(), embeds[k].cpu()), "image rows must be the unscaled embedder rows"

    # masks as generate() builds them
    types_t = torch.tensor([types])
    masks = M.Gemma4UnifiedForConditionalGeneration.create_masks_for_generate(
        ref.config, x, torch.ones(1, T, dtype=torch.long, device=device), None, torch.arange(T, device=device)[None],
        mm_token_type_ids=types_t.to(device))
    masks_causal = M.Gemma4UnifiedForConditionalGeneration.create_masks_for_generate(
        ref.config, x, torch.ones(1, T, dtype=torch.long, device=device), None, torch.arange(T, device=device)[None],
        mm_token_type_ids=None)

    def as_bool(m):
        allowed = (m[0, 0] == 0) if m.dtype.is_floating_point else m[0, 0]
        return allowed.to(torch.uint8).cpu()

    assert not torch.equal(as_bool(masks["sliding_attention"]), as_bool(masks_causal["sliding_attention"]))
    assert torch.equal(as_bool(masks["full_attention"]), as_bool(masks_causal["full_attention"])), \
        "the generate() path keeps full layers causal (semantics doc section 2)"
    # klimit_ext as the r4dx sliding kernels take it
    klimit = [-1] * T
    for s, e in spans.values():
        for p in range(s, e):
            klimit[p] = e - 1
    sl = as_bool(masks["sliding_attention"]).numpy()
    for q in range(T):
        lo = max(0, q - ref.arch.window + 1)
        hi = max(q, klimit[q])
        want = np.zeros(T, np.uint8)
        want[lo:hi + 1] = 1
        assert (sl[q] == want).all(), f"klimit_ext + window does not reproduce HF's sliding mask at row {q}"

    # forward()-path mask (block overlay on full layers too) for the A/B switch
    bsi = M.get_block_sequence_ids_for_mask(types_t, device=device)
    from transformers import masking_utils
    fwd = masking_utils.create_masks_for_generate(
        config=ref.config.get_text_config(), inputs_embeds=x, attention_mask=torch.ones(1, T, dtype=torch.long, device=device),
        past_key_values=None, position_ids=torch.arange(T, device=device)[None], block_sequence_ids=bsi)

    pos = torch.arange(T, device=device)[None]

    def run(i, mask_dict):
        layer = ref.layer(i)
        lt = ref.layer_types[i]
        cos, sin = ref.rotary(x, pos, lt)
        from collections import UserDict
        with torch.no_grad():
            out = layer(x, shared_kv_states=UserDict(), position_embeddings=(cos, sin),
                        attention_mask=mask_dict[lt], position_ids=pos, past_key_values=None)
        return (out[0] if isinstance(out, tuple) else out)[0].to(torch.bfloat16).cpu()

    l0_block = run(0, masks)
    l0_causal = run(0, masks_causal)
    l5 = run(5, masks)
    l5_bidir = run(5, fwd)
    rel = lambda p, q: float((p.float() - q.float()).norm() / q.float().norm())  # noqa: E731
    meta["layer0_block_vs_causal_rel_fro"] = rel(l0_block, l0_causal)
    meta["layer5_full_vs_bidir_rel_fro"] = rel(l5, l5_bidir)
    assert meta["layer0_block_vs_causal_rel_fro"] > 1e-3, "the block mask must change layer 0's output"

    tensors.update({
        "token_ids": torch.tensor(ids, dtype=torch.int32), "mm_token_type_ids": torch.tensor(types, dtype=torch.int32),
        "x_in": x[0].cpu(), "klimit_ext": torch.tensor(klimit, dtype=torch.int32),
        "mask_sliding": as_bool(masks["sliding_attention"]), "mask_full": as_bool(masks["full_attention"]),
        "layer0_block": l0_block, "layer0_causal": l0_causal, "layer5_full": l5, "layer5_full_if_bidir": l5_bidir,
    })

    if a.full_logits:
        h = x
        with torch.no_grad():
            for i in range(ref.n_layers):
                layer = ref.layer(i)
                lt = ref.layer_types[i]
                cos, sin = ref.rotary(h, pos, lt)
                from collections import UserDict
                out = layer(h, shared_kv_states=UserDict(), position_embeddings=(cos, sin), attention_mask=masks[lt],
                            position_ids=pos, past_key_values=None)
                h = out[0] if isinstance(out, tuple) else out
                ref._layers.pop(i, None)  # keep VRAM flat
            last = ref.final_norm(h[0, -1:])
            logits = ref.logits_rows(last, mode="fp32")
        tensors["last_row_logits_fp32"] = logits.float().cpu()
        meta["full_logits"] = "last row, generate()-path masks (sliding-only block overlay)"

    out_dir = Path(a.out_dir)
    save_golden(out_dir / "vision_golden_gemma.safetensors", tensors, {"meta": meta})
    (out_dir / "vision_golden_gemma.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print("wrote", out_dir / "vision_golden_gemma.safetensors")


if __name__ == "__main__":
    sys.exit(main())
