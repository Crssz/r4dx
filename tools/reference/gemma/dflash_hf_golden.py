"""HF/z-lab golden for the Gemma 4 DFlash v1 drafter (CPU only).

Runs the ACTUAL vendored z-lab `DFlashDraftModel` (zlab_dflash_model.py, transformers 5.x Qwen3 classes) on a tiny
random model and compares its draft hidden state / argmax tokens to the numpy v1 reference (dflash_v1_ref.py) that
the r4dx dflash2 identity-conv forward is checked against. This closes the circularity of dflash_v1_ref.py: the
semantics now come from z-lab's code, not from the same author's reading of it.

It also pins the embedding scale: dflash_generate builds the block with `F.embedding(ids, target_weight) * scale`
(`_raw_input_embeddings`, scale = dflash_config/config `input_embedding_scale`, default 1.0), i.e. it BYPASSES the
target's scaled embedding module.

Run: <ref venv>/Scripts/python.exe tools/reference/gemma/dflash_hf_golden.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

assert not torch.cuda.is_available()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_v1_ref as v1  # noqa: E402
import zlab_dflash_model as zl  # noqa: E402
from transformers import Qwen3Config  # noqa: E402


def build(seed, window_cfg, layer_types):
    rng = np.random.RandomState(seed)
    cfg = v1.make_cfg(window_cfg)
    cfg["layers"] = len(layer_types)
    sd = v1.make_v1_weights(rng, cfg["hidden"], cfg["layers"], cfg["ffn"], cfg["heads"], cfg["kv_heads"], cfg["hd"],
                            len(cfg["target_layers"]))
    qc = Qwen3Config(
        hidden_size=cfg["hidden"], intermediate_size=cfg["ffn"], num_hidden_layers=cfg["layers"],
        num_attention_heads=cfg["heads"], num_key_value_heads=cfg["kv_heads"], head_dim=cfg["hd"],
        rms_norm_eps=cfg["eps"], vocab_size=cfg["vocab"], layer_types=layer_types, sliding_window=window_cfg,
        use_sliding_window=True, max_window_layers=cfg["layers"], attention_bias=False, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": cfg["theta"]}, num_target_layers=48,
        dflash_config={"mask_token_id": cfg["mask_id"], "target_layer_ids": [1, 10]}, block_size=cfg["block"],
        final_logit_softcapping=30.0, attn_implementation="eager")
    model = zl.DFlashDraftModel(qc).eval()
    missing = model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return rng, cfg, sd, model


def zl_block(model, cfg, embed, feats, anchor, n_ctx, scale):
    B = cfg["block"]
    ids = torch.tensor([[anchor] + [cfg["mask_id"]] * (B - 1)])
    noise = zl._raw_input_embeddings(SimpleTarget(embed), ids, scale)
    with torch.no_grad():
        h = model(target_hidden=torch.from_numpy(feats)[None], noise_embedding=noise,
                  position_ids=torch.arange(n_ctx + B)[None], use_cache=False)[:, 1 - B:, :]
        logits = model.compute_logits(h, lambda x: x @ torch.from_numpy(embed).T)
    return h[0].numpy(), logits[0].argmax(-1).tolist()


class SimpleTarget(torch.nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embed))

    def get_input_embeddings(self):
        return self.emb


def numpy_block(sd, cfg, embed, feats, anchor, n_ctx, windows, causal, scale):
    toks, xf, _ = v1.v1_draft_tokens(sd, cfg, embed, embed, feats, np.arange(n_ctx), anchor, windows, 30.0, scale,
                                     causal=causal)
    return xf[1:], toks


def case(seed, n_ctx, scale):
    layer_types = ["sliding_attention", "sliding_attention", "full_attention"]
    window = 24
    rng, cfg, sd, model = build(seed, window, layer_types)
    embed = (rng.randn(cfg["vocab"], cfg["hidden"]) * 0.5).astype(np.float32)
    feats = rng.randn(n_ctx, len(cfg["target_layers"]) * cfg["hidden"]).astype(np.float32)
    h_hf, tok_hf = zl_block(model, cfg, embed, feats, 17, n_ctx, scale)
    # z-lab: is_causal = (layer_type == "sliding_attention"); sliding layers windowed, the full layer unrestricted.
    windows = [window, window, None]
    causal = [True, True, False]
    h_np, tok_np = numpy_block(sd, cfg, embed, feats, 17, n_ctx, windows, causal, scale)
    err = float(np.max(np.abs(h_hf - h_np)))
    # ... and the all-non-causal reading the r4dx runtime implements today (documented gap)
    h_nc, tok_nc = numpy_block(sd, cfg, embed, feats, 17, n_ctx, windows, [False] * 3, scale)
    err_nc = float(np.max(np.abs(h_hf - h_nc)))
    return err, tok_hf == tok_np, err_nc, tok_hf == tok_nc


def test_hf_matches_numpy_v1_with_zlab_causality():
    for seed, n_ctx, scale in [(1, 40, 1.0), (2, 8, 1.0), (3, 90, 1.0), (4, 40, 62.0)]:
        err, same, err_nc, same_nc = case(seed, n_ctx, scale)
        print(f"[hf golden] seed={seed} n_ctx={n_ctx} scale={scale}: zlab-causality err={err:.2e} tokens_equal={same}"
              f" | all-non-causal err={err_nc:.2e} tokens_equal={same_nc}")
        assert err < 5e-3 * max(1.0, scale) and same, (seed, err, same)


def test_noncausal_reading_is_a_real_difference():
    err, _, err_nc, _ = case(5, 40, 1.0)
    assert err_nc > 50 * err, "sliding layers are causal in z-lab; an all-non-causal block must visibly differ"


if __name__ == "__main__":
    test_hf_matches_numpy_v1_with_zlab_causality()
    test_noncausal_reading_is_a_real_difference()
    print("PASS")

