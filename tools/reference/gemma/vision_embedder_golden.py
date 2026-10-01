"""CPU-only golden for the Gemma 4 encoder-free vision embedder on RANDOM weights (docs/gemma4-plan.md M2).

Builds a tiny Gemma4UnifiedVisionEmbedder (same module chain as the 12B: LayerNorm(patch_dim) -> Linear
(+bias) -> LayerNorm -> + pos_embedding[x, 0] + pos_embedding[y, 1] -> LayerNorm(pos_norm) -> RMSNorm
(no weight, eps 1e-6) -> Linear (no bias)) with seeded random bf16 weights, runs the transformers module in
bf16 on CPU, and re-derives it with a numpy implementation (fp32 math, bf16 rounding after every op, which
is how the C++ host reference r4dx::vision::GemmaEmbedHost is written). Dumps to
<out-dir>/gemma/vision_embedder.bin:

    int32 patch_dim, mm_dim, out_dim, posemb, n
    float32 (bf16-representable): ln1_w[P] ln1_b[P] dense_w[D,P] dense_b[D] ln2_w[D] ln2_b[D]
        pos_table[posemb,2,D] pos_norm_w[D] pos_norm_b[D] proj_w[O,D] pixels[n,P]
    int32 positions[n,2] (x, y)
    float32 out_torch[n,O] out_numpy[n,O]

tests/vision/test_gemma_embedder_golden.cpp checks the C++ host reference against both. The script asserts
numpy vs torch itself (max relative error well inside a couple of bf16 ulps, with a few 1-ulp flips from the
different fp32 summation order of the GEMMs allowed). It also asserts the splice facts the engine relies on:
the embedder output is NOT scaled by sqrt(hidden), and the placeholder rows of inputs_embeds are overwritten
(masked_scatter) rather than added to.

CPU only: HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES are set to '-1' before torch is imported.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\vision_embedder_golden.py
"""
import argparse
import os
import struct
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available(), "GPU must stay hidden"

from transformers.models.gemma4_unified import modeling_gemma4_unified as M  # noqa: E402
from transformers.models.gemma4_unified.configuration_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)

PATCH, POOL = 2, 2           # model_patch_size 4 -> patch_dim 4*4*3 = 48
MM_DIM, OUT_DIM, POSEMB = 40, 24, 16
N = 11


def bf16(x):
    """fp32 numpy -> bf16-rounded fp32 (round to nearest even), the torch rounding."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16).to(torch.float32).numpy()


def layer_norm(x, w, b, eps=1e-5):
    x = x.astype(np.float64)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps) * w + b
    return bf16(y.astype(np.float32))


def numpy_embedder(P, pixels, pos):
    x = bf16(pixels)
    h = layer_norm(x, P["ln1_w"], P["ln1_b"])
    h = bf16((h.astype(np.float64) @ P["dense_w"].T.astype(np.float64) + P["dense_b"]).astype(np.float32))
    h = layer_norm(h, P["ln2_w"], P["ln2_b"])
    pe = bf16(P["pos_table"][pos[:, 0], 0] + P["pos_table"][pos[:, 1], 1])
    h = bf16(h + pe)
    h = layer_norm(h, P["pos_norm_w"], P["pos_norm_b"])
    ss = (h.astype(np.float64) ** 2).mean(-1, keepdims=True)
    h = bf16((h / np.sqrt(ss + 1e-6)).astype(np.float32))
    return bf16((h.astype(np.float64) @ P["proj_w"].T.astype(np.float64)).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "golden_out"))
    a = ap.parse_args()

    torch.manual_seed(1234)
    vcfg = Gemma4UnifiedVisionConfig(patch_size=PATCH, pooling_kernel_size=POOL, mm_embed_dim=MM_DIM,
                                     mm_posemb_size=POSEMB, output_proj_dims=MM_DIM)
    tcfg = Gemma4UnifiedTextConfig(vocab_size=64, hidden_size=OUT_DIM, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=16,
                                   num_global_key_value_heads=1, attention_k_eq_v=True,
                                   layer_types=["sliding_attention", "full_attention"])
    emb = M.Gemma4UnifiedVisionEmbedder(vcfg, tcfg).to(torch.bfloat16)
    P_DIM = vcfg.model_patch_size ** 2 * 3
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in emb.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln1.weight") or name.endswith("ln2.weight"):
                p.copy_((1.0 + 0.2 * torch.randn(p.shape, generator=g)).to(torch.bfloat16))
            elif name.endswith(".bias"):
                p.copy_((0.2 * torch.randn(p.shape, generator=g)).to(torch.bfloat16))
            elif "pos_embedding" in name:
                p.copy_((1.0 * torch.randn(p.shape, generator=g)).to(torch.bfloat16))
            else:
                p.copy_((0.5 / np.sqrt(p.shape[-1]) * torch.randn(p.shape, generator=g)).to(torch.bfloat16))
    emb.eval()

    pixels = torch.rand(1, N, P_DIM, generator=g)  # [0, 1) like rescaled u8
    xs = torch.randint(0, POSEMB, (N,), generator=g)
    ys = torch.randint(0, POSEMB, (N,), generator=g)
    pos = torch.stack([xs, ys], dim=-1)[None]
    with torch.no_grad():
        out = emb(pixel_values=pixels, image_position_ids=pos, return_dict=True).pooler_output[0]
    out_torch = out.to(torch.float32).numpy()

    f = lambda t: t.detach().to(torch.float32).numpy()  # noqa: E731
    P = {
        "ln1_w": f(emb.patch_ln1.weight), "ln1_b": f(emb.patch_ln1.bias),
        "dense_w": f(emb.patch_dense.weight), "dense_b": f(emb.patch_dense.bias),
        "ln2_w": f(emb.patch_ln2.weight), "ln2_b": f(emb.patch_ln2.bias),
        "pos_table": f(emb.pos_embedding),
        "pos_norm_w": f(emb.pos_norm.weight), "pos_norm_b": f(emb.pos_norm.bias),
        "proj_w": f(emb.multimodal_embedder.embedding_projection.weight),
    }
    px = pixels[0].numpy()
    pn = pos[0].numpy()
    out_np = numpy_embedder(P, px, pn)

    # numpy vs torch: same ops, different fp32 summation order inside the GEMMs -> a few 1-ulp flips.
    err = np.abs(out_np - out_torch)
    scale = np.maximum(np.abs(out_torch), 2.0 ** -6)
    rel = (err / scale).max()
    flips = float((err > 0).mean())
    print(f"numpy vs torch: max rel err {rel:.4g}, differing elements {flips:.3%}")
    assert rel < 2.0 ** -5, "numpy reference disagrees with the HF module"

    # Splice facts: masked_scatter overwrites placeholder rows with the UNSCALED embedder output.
    ids = torch.tensor([[5, 6, 7, 7, 7, 8]])
    text_emb = torch.randn(1, 6, OUT_DIM)
    mask = (ids == 7)[..., None].expand_as(text_emb)
    scattered = text_emb.masked_scatter(mask, out[:3].to(torch.float32))
    assert torch.equal(scattered[0, 2:5], out[:3].to(torch.float32)), "placeholder rows must equal the embedder rows"
    assert torch.equal(scattered[0, :2], text_emb[0, :2]), "text rows untouched"

    d = Path(a.out_dir) / "gemma"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "vision_embedder.bin", "wb") as fh:
        fh.write(struct.pack("<5i", P_DIM, MM_DIM, OUT_DIM, POSEMB, N))
        for k in ("ln1_w", "ln1_b", "dense_w", "dense_b", "ln2_w", "ln2_b", "pos_table", "pos_norm_w",
                  "pos_norm_b", "proj_w"):
            fh.write(np.ascontiguousarray(P[k], dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(px, dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(pn, dtype=np.int32).tobytes())
        fh.write(np.ascontiguousarray(out_torch, dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(out_np, dtype=np.float32).tobytes())
    print("wrote", d / "vision_embedder.bin")


if __name__ == "__main__":
    sys.exit(main())
