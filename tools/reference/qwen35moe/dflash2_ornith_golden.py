"""Golden for the Ornith-1.5-122B DFlash2 drafter: the ACTUAL vendored z-lab `DFlash2DraftModel` (zlab_dflash_model.py) with the
REAL drafter weights (incoai/Ornith-1.5-122B-DFlash2), on CPU in float32, fed random target features and a random small-vocabulary
target embedding / lm_head. It pins what r4dx's DflashDraft (converted container, `r4dx-convert --dflash-hf`) must reproduce:
the draft block's final normed hidden state, the per-position top-16 candidates of the shared lm_head, and the selector's greedy
path (`propose`, temperature 0).

Writes <out>/ : manifest.json, features.bin (bf16 [n_ctx, 8 * 3072]), embed.bin (bf16 [V, 3072], also the lm_head),
x_final_normed.bin (f32 [15, 3072]: block positions 1..15), cand.bin (i32 [15, 16]), unary.bin (f32 [15, 16]), path.bin (i32 [15]).

Run (system Python 3.12 with torch + transformers 5.x; CPU only):
  python tools/reference/qwen35moe/dflash2_ornith_golden.py --drafter E:/models/r4dx/Ornith-1.5-122B-DFlash2 --out E:/models/r4dx/q35moe-dflash2-golden
Checked by tests/model/qwen35moe/test_q35moe_dflash2_golden.cpp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
import zlab_dflash_model as zl  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen3Config  # noqa: E402


def bf16_round(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()


def to_bf16_bits(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(a).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--n-ctx", type=int, default=40)
    ap.add_argument("--anchor", type=int, default=17)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg_json = json.loads((Path(a.drafter) / "config.json").read_text())
    qc = Qwen3Config(**{k: v for k, v in cfg_json.items() if k not in ("architectures", "model_type", "dtype", "transformers_version")})
    qc._attn_implementation = "sdpa"
    model = zl.DFlash2DraftModel(qc).eval()
    sd = load_file(str(Path(a.drafter) / "model.safetensors"))
    sd = {(k + ".weight" if k.startswith("candidate_selector.") and k.endswith("_codebook") else k): v.to(torch.float32) for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys and not res.missing_keys, (res.unexpected_keys, res.missing_keys)
    model = model.float()

    rng = np.random.RandomState(a.seed)
    H = qc.hidden_size
    B = int(cfg_json["dflash_config"]["block_size"])
    n_taps = len(cfg_json["dflash_config"]["target_layer_ids"])
    mask_id = a.vocab - 1
    embed = bf16_round((rng.randn(a.vocab, H) * 0.5).astype(np.float32))
    feats = bf16_round(rng.randn(a.n_ctx, n_taps * H).astype(np.float32))

    ids = torch.tensor([[a.anchor] + [mask_id] * (B - 1)])
    noise = torch.nn.functional.embedding(ids, torch.from_numpy(embed))
    with torch.no_grad():
        h = model(target_hidden=torch.from_numpy(feats)[None], noise_embedding=noise,
                  position_ids=torch.arange(a.n_ctx + B)[None], use_cache=False)[:, 1 - B:, :]
        head = lambda x: x @ torch.from_numpy(embed).T  # noqa: E731
        path, candidates, _ = model.propose(h, torch.tensor([a.anchor]), head, 0.0)
        logits = model.compute_logits(h, head)
        unary, cand = torch.topk(logits, 16, dim=-1, sorted=True)
    # HF's own bf16-vs-fp32 noise on the same inputs: the floor any bf16 implementation (r4dx's) is judged against
    model_bf = zl.DFlash2DraftModel(qc).eval()
    model_bf.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    model_bf = model_bf.to(torch.bfloat16)
    with torch.no_grad():
        hb = model_bf(target_hidden=torch.from_numpy(feats).to(torch.bfloat16)[None],
                      noise_embedding=noise.to(torch.bfloat16), position_ids=torch.arange(a.n_ctx + B)[None],
                      use_cache=False)[:, 1 - B:, :].to(torch.float32)
    noise_rel = float(((hb - h).norm() / h.norm()).item())
    print(f"HF bf16 vs HF fp32 final normed hidden RelL2: {noise_rel:.3e}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    to_bf16_bits(feats).tofile(out / "features.bin")
    to_bf16_bits(embed).tofile(out / "embed.bin")
    h[0].numpy().astype(np.float32).tofile(out / "x_final_normed.bin")
    cand[0].numpy().astype(np.int32).tofile(out / "cand.bin")
    unary[0].numpy().astype(np.float32).tofile(out / "unary.bin")
    path[0].numpy().astype(np.int32).tofile(out / "path.bin")
    (out / "manifest.json").write_text(json.dumps({
        "schema": "q35moe_dflash2_golden_v1", "vocab": a.vocab, "mask_id": mask_id, "n_ctx": a.n_ctx, "anchor": a.anchor,
        "block": B, "hidden": H, "taps": n_taps, "seed": a.seed, "path": path[0].tolist()}, indent=2))
    print("path:", path[0].tolist())
    print("wrote", out)


if __name__ == "__main__":
    main()
