"""HF-side check of an Ornith image conversation (docs/qwen35moe-vision.md): is the text r4dx generated for an image what the bf16
reference model would say, token by token?

    $env:HIP_VISIBLE_DEVICES='1'
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\vision_check.py --tokens <cli --dump-token-ids file> `
        --prompt-len <the CLI's "prefill: N tok"> --image <path> [--image <path2>] [--max-pixels 1000000] --out report.json

The 122B does not fit anywhere in bf16, so the reference is the layer-streaming one (tools/reference/qwen35moe/ref.py: each layer's
bf16 weights are built from the checkpoint shards, run, and dropped). One teacher-forced pass over the whole token sequence
(prompt with its expanded image placeholders + r4dx's generated tokens):
  * the vision tower (HF Qwen3_5MoeVisionModel, bf16, real weights) on HF's own preprocessing of the image(s) -> merged rows;
  * the rows replace the placeholder embeddings, the 3-axis rope positions come from HF's get_rope_index;
  * every decoder layer, the final norm, the lm_head over the GENERATED positions.
Per generated token it reports whether HF's argmax is the token r4dx emitted, HF's log-probability of r4dx's token and whether it is in
HF's top 5. r4dx runs 3.4-bit weights (Rung 4: top-1 agreement with HF 88% on text, docs/qwen35moe-kl-gate.md), so the numbers are to be
read against that, and against the two CONTROLS run in the same pass (every layer is loaded once for all three streams):
  no_mrope   the same embeddings, plain 1-D positions (what an image-unaware rope would do);
  no_image   the placeholder embeddings stay (the image rows are not spliced), mrope positions as in `hf`.
A real splice / rope error shows as `hf` agreement collapsing towards the controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "reference"))

from common import ShardIndex, load_module_state  # noqa: E402
from qwen35moe.ref import Qwen35MoeStreamingReference  # noqa: E402

DEFAULT_MODEL = Path(r"E:\models\r4dx\Ornith-1.5-122B")


def encode_images(model_dir: Path, image_paths: list[str], max_pixels: int, device):
    from PIL import Image
    from transformers import AutoImageProcessor
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    cfg = Qwen3_5MoeConfig.from_pretrained(str(model_dir))
    vc = cfg.vision_config
    vc._attn_implementation = "sdpa"
    tower = m.Qwen3_5MoeVisionModel(vc).to(device=device, dtype=torch.bfloat16).eval()
    loaded, missing, err = load_module_state(tower, ShardIndex.load(model_dir), "model.visual.")
    if not loaded:
        raise RuntimeError(f"vision weights: missing {missing[:5]} {err}")
    fp32_inv, _ = m.Qwen3_5MoeVisionRotaryEmbedding.compute_axial_rope_parameters(vc)
    tower.rotary_pos_emb.inv_freq = fp32_inv.to(device=device, dtype=torch.float32)
    proc = AutoImageProcessor.from_pretrained(str(model_dir))
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    pix = proc(images=imgs, size={"shortest_edge": 65536, "longest_edge": max_pixels}, return_tensors="pt")
    with torch.no_grad():
        out = tower(hidden_states=pix["pixel_values"].to(device=device, dtype=torch.float32), grid_thw=pix["image_grid_thw"].to(device))
    rows = out.pooler_output.detach().to(torch.bfloat16)
    del tower
    torch.cuda.empty_cache()
    return rows, pix["image_grid_thw"], cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", required=True, help="the CLI's --dump-token-ids file (prompt + generated ids)")
    ap.add_argument("--prompt-len", type=int, required=True, help="tokens of the prompt (the CLI's `prefill: N tok`)")
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    seg = json.loads(Path(a.tokens).read_text())["segments"][0]["token_ids"]
    ids = [int(t) for t in seg]
    P = a.prompt_len
    T = len(ids)
    print(f"[vision_check] {T} tokens: prompt {P} + generated {T - P}", flush=True)

    model_dir = Path(a.model_dir)
    ref = Qwen35MoeStreamingReference(model_dir, a.device, verbose=False)
    dev = ref.device
    rows, grid, cfg = encode_images(model_dir, a.image, a.max_pixels, dev)
    image_id = cfg.image_token_id
    n_ph = sum(1 for t in ids[:P] if t == image_id)
    print(f"[vision_check] tower: grid {grid.tolist()} -> {rows.shape[0]} rows; the prompt has {n_ph} placeholders", flush=True)
    if n_ph != rows.shape[0]:
        raise SystemExit(f"placeholder count {n_ph} != tower rows {rows.shape[0]}: r4dx and HF preprocess the image differently")

    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m

    x = torch.tensor([ids], dtype=torch.long, device=dev)
    mm = (x == image_id).to(torch.int32)
    shim = types.SimpleNamespace(config=cfg)
    shim.get_vision_position_ids = lambda *aa, **kk: m.Qwen3_5MoeModel.get_vision_position_ids(shim, *aa, **kk)
    pos_mrope, _ = m.Qwen3_5MoeModel.get_rope_index(shim, x, mm, image_grid_thw=grid.to(dev))
    pos_plain = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)

    emb = ref.embed(ids).unsqueeze(0)  # placeholder embeddings
    emb_img = emb.clone()
    emb_img[0, x[0] == image_id] = rows.to(dev)
    variants = {
        "hf": (emb_img, pos_mrope),
        "no_mrope": (emb_img, pos_plain),
        "no_image": (emb, pos_mrope),
    }
    hidden = {k: v[0] for k, v in variants.items()}
    pe = {k: ref._rotary_emb()(hidden[k], v[1]) for k, v in variants.items()}
    for i in range(ref.n_layers):
        layer = ref._build_layer(i)
        for k in hidden:
            hidden[k] = ref.layer_forward(layer, hidden[k], pe[k])
        del layer
        if i % 8 == 7:
            print(f"[vision_check] layer {i + 1}/{ref.n_layers}", flush=True)

    report = {"tokens": T, "prompt": P, "generated": T - P, "image_rows": int(rows.shape[0]), "grid": grid.tolist(), "variants": {}}
    for k in hidden:
        rows_h = ref.final_norm(hidden[k])[:, P - 1:T - 1]  # positions whose next token r4dx generated
        lg = ref.logits_rows(rows_h, "hf-bf16")[0]
        lp = torch.log_softmax(lg, dim=-1)
        tgt = torch.tensor(ids[P:], device=lp.device)
        top1 = lp.argmax(-1)
        tok_lp = lp.gather(1, tgt[:, None])[:, 0]
        top5 = lp.topk(5, dim=-1).indices
        r = {
            "top1_agree": float((top1 == tgt).float().mean()),
            "mean_logprob_of_r4dx_token": float(tok_lp.mean()),
            "in_hf_top5": float((top5 == tgt[:, None]).any(-1).float().mean()),
            "perplexity_of_r4dx_text": math.exp(-float(tok_lp.mean())),
        }
        report["variants"][k] = r
        print(f"[vision_check] {k:9s} top-1 agreement {100 * r['top1_agree']:.1f}%  in HF top-5 {100 * r['in_hf_top5']:.1f}%  "
              f"mean log-prob {r['mean_logprob_of_r4dx_token']:.3f} (ppl {r['perplexity_of_r4dx_text']:.2f})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
