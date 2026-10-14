"""tools/reference/lfm2/router_golden.py  (docs/lfm2-plan.md L0-9 / 3.5 / 6.3: the router kernel contract)

Goldens for `r4dx_lfm2_router_f32`: per row, 64 dot products of length 2048 (bf16 x bf16, fp32 accumulate),
`s = sigmoid(logit)` in fp32, `score = s + expert_bias[e]` (fp32), top-4 by score (ties -> LOWER expert index),
weights `s[sel] / (sum(s[sel]) + 1e-6) * routed_scaling_factor`, all fp32.  The contract implementation is
`lfm2.ref.manual_router` (independent of HF; cross-checked here against the HF `Lfm2MoeTopKRouter` run in fp32).

Cases (each a .safetensors; tolerance: ids EXACT, weights rel 1e-3, logits rel 1e-3):

  real_layer_002 / real_layer_003   the real gate weight + F32 expert_bias of layers 2 / 3 on the REAL ffn_norm
                                    outputs of the first --tokens tokens of the first chat-corpus segment run
                                    through layers 0..3 (bf16 stack): what the router actually sees. The manifest
                                    records the biased 4th-5th margin distribution (L0-5), the rows where the bias
                                    changes the selected set vs the unbiased top-4, and the rows where HF-bf16
                                    routing differs from fp32 routing.
  tie                               synthetic: two experts a < b with IDENTICAL gate rows and identical bias, the bias
                                    tuned so that on many rows the pair straddles the 4th/5th boundary: the lower index
                                    must be selected and the higher one not.
  bias_selects                      synthetic: a large expert_bias makes the selected set differ from the unbiased top-4
                                    on most rows; the weights must still be the UNBIASED sigmoids.

`ids` are stored in descending-score slot order (the order `weights` follows); a kernel that returns the set in another
order should compare as sets and sum the weighted outputs. CPU only.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\router_golden.py `
        --out-dir tools\\reference\\golden_out\\lfm2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import save_golden, tensor_manifest_entry  # noqa: E402
from lfm2.common_lfm2 import DEFAULT_MODEL_DIR, GOLDEN_DIR, KL_CORPUS_DIR, peak_rss_gb  # noqa: E402
from lfm2.layer_golden_lfm2 import seeded_normal  # noqa: E402
from lfm2.ref import Lfm2Reference, add_tiny_args, build_tiny_checkpoint, manual_router, causal_mask  # noqa: E402


def hf_route(arch, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dtype):
    """HF Lfm2MoeTopKRouter in `dtype` (bf16: stock HF numerics; fp32: the truth)."""
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeTopKRouter
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig

    cfg = Lfm2MoeConfig(hidden_size=arch.hidden, num_experts=arch.n_experts, num_experts_per_tok=arch.top_k,
                        norm_topk_prob=arch.norm_topk_prob, routed_scaling_factor=arch.routed_scaling,
                        use_expert_bias=arch.use_expert_bias, num_hidden_layers=1, layer_types=["conv"])
    r = Lfm2MoeTopKRouter(cfg)
    r.weight.data = weight.to(dtype).clone()
    with torch.no_grad():
        logits, w, ids = r(x.to(dtype), bias.float())
    return logits, w, ids


def stats_entry(arch, x, W, b) -> tuple[dict, dict]:
    logits, s, ids, w = manual_router(x, W, b, arch.top_k, arch.norm_topk_prob, arch.routed_scaling)
    # contract vs HF-fp32 router (same operands widened)
    _, w_h, ids_h = hf_route(arch, x.float(), W.float(), b, torch.float32)
    contract_equals_hf_fp32 = bool((ids.sort(-1).values == ids_h.sort(-1).values).all()) and \
        float((w.sort(-1).values - w_h.sort(-1).values).abs().max()) < 1e-6
    _, w_b, ids_b = hf_route(arch, x, W, b, torch.bfloat16)
    flips = int((ids.sort(-1).values != ids_b.sort(-1).values).any(-1).sum())
    score = (s + b.float()).sort(-1, descending=True).values
    margin = score[:, arch.top_k - 1] - score[:, arch.top_k]
    unb = torch.sort(s, dim=-1, descending=True).indices[:, :arch.top_k]
    bias_changes = int((unb.sort(-1).values != ids.sort(-1).values).any(-1).sum())
    meta = {"rows": int(x.shape[0]), "contract_equals_hf_fp32": contract_equals_hf_fp32,
            "rows_hf_bf16_ids_differ_from_fp32": flips,
            "margin_4th_5th_biased": {q: float(margin.quantile(p)) for q, p in
                                      (("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("median", 0.5), ("p95", 0.95))},
            "rows_margin_below_1e-3": int((margin < 1e-3).sum()), "rows_margin_below_1e-2": int((margin < 1e-2).sum()),
            "rows_bias_changes_selected_set": bias_changes,
            "weights_sum_before_norm_mean": float(s.gather(-1, ids).sum(-1).mean()),
            "sigmoid_selected_mean": float(s.gather(-1, ids).mean()), "sigmoid_all_mean": float(s.mean()),
            "experts_hit": int(torch.unique(ids).numel())}
    t = {"router_logits_f32": logits, "router_sigmoid_f32": s, "ids": ids.to(torch.int32), "weights_f32": w,
         "ids_hf_bf16": ids_b.to(torch.int32), "weights_hf_bf16": w_b.float()}
    return meta, t


def real_router_inputs(ref: Lfm2Reference, ids: list[int], layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """ffn_norm outputs `[T, H]` (bf16) of MoE layers `layers` for the token ids, running layers 0..max(layers)."""
    with torch.no_grad():
        h = torch.stack([ref.raw_rows("model.embed_tokens.weight", t) for t in ids]).to(torch.bfloat16).unsqueeze(0)
        T = h.shape[1]
        pos = ref.positions(T)
        pe = ref.rope(h, pos)
        mask = causal_mask(T, h.dtype)
        got = {}
        for i in range(max(layers) + 1):
            layer = ref.build_layer(i, torch.bfloat16)
            mid = ref.mixer(layer, i, h, pe, mask if layer.is_attention_layer else None, pos)
            if i in layers:
                got[i] = layer.ffn_norm(mid[0]).clone()
            out, _ = ref.ffn(layer, i, mid[0])
            h = out.unsqueeze(0)
            del layer
    return got


def tie_case(arch, W0: torch.Tensor, b0: torch.Tensor, seed: int, rows: int = 256):
    E, H = arch.n_experts, arch.hidden
    x = seeded_normal((rows, H), seed + 31).to(torch.bfloat16)
    W = W0.clone()
    a, b = 5, min(E - 1, 9)
    W[b] = W[a]
    bias = b0.clone() * 0.0
    others = [e for e in range(E) if e not in (a, b)]
    s_others = torch.sigmoid(x.float() @ W[others].float().T) + bias[others]          # [rows, E-2]
    s_pair = torch.sigmoid(x.float() @ W[a].float())                                  # [rows]
    best, best_n = 0.0, -1
    for B in torch.linspace(-1.0, 1.0, 801):
        pair = s_pair + B
        above = (s_others > pair.unsqueeze(1)).sum(1)
        n = int((above == arch.top_k - 1).sum())
        if n > best_n:
            best, best_n = float(B), n
    bias[a] = bias[b] = best
    logits, s, ids, w = manual_router(x, W, bias, arch.top_k, arch.norm_topk_prob, arch.routed_scaling)
    assert torch.equal(logits[:, a], logits[:, b]), "identical gate rows must give bit-identical logits"
    straddle = ((ids == a).any(1) & ~(ids == b).any(1))
    wrong = ((ids == b).any(1) & ~(ids == a).any(1))
    assert int(wrong.sum()) == 0, "lower index must win ties"
    assert int(straddle.sum()) > 0, "tie case did not produce a boundary tie row"
    t = {"x": x, "weight": W, "bias": bias, "router_logits_f32": logits, "router_sigmoid_f32": s,
         "ids": ids.to(torch.int32), "weights_f32": w, "straddle_rows": straddle.to(torch.int32)}
    meta = {"case": "tie", "tied_experts": [a, b], "bias_value": best, "rows": rows,
            "rows_tie_straddles_4th_5th_a_selected_b_not": int(straddle.sum()),
            "rule": "ties -> lower expert index: expert a is selected and b is not on every straddle row"}
    return t, meta


def bias_case(arch, W0: torch.Tensor, seed: int, rows: int = 256):
    E, H = arch.n_experts, arch.hidden
    x = seeded_normal((rows, H), seed + 41).to(torch.bfloat16)
    bias = torch.zeros(E)
    g = torch.Generator().manual_seed(seed + 42)
    bias[torch.randperm(E, generator=g)[: max(2, E // 8)]] = 2.0       # a few boosted experts dominate selection
    logits, s, ids, w = manual_router(x, W0, bias, arch.top_k, arch.norm_topk_prob, arch.routed_scaling)
    unb = torch.sort(s, dim=-1, descending=True).indices[:, :arch.top_k]
    changed = int((unb.sort(-1).values != ids.sort(-1).values).any(-1).sum())
    assert changed > rows // 2
    t = {"x": x, "weight": W0, "bias": bias, "router_logits_f32": logits, "router_sigmoid_f32": s,
         "ids": ids.to(torch.int32), "weights_f32": w}
    meta = {"case": "bias_selects", "rows": rows, "rows_selected_set_differs_from_unbiased": changed,
            "rule": "selection uses s + bias, weights use the UNBIASED s of the selected experts (then normalized)"}
    return t, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {GOLDEN_DIR}")
    ap.add_argument("--tokens", type=int, default=256, help="rows of real router input per layer")
    ap.add_argument("--corpus", type=Path, default=KL_CORPUS_DIR / "chat_lfm2.json")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", default="2,3")
    ap.add_argument("--threads", type=int, default=8)
    add_tiny_args(ap)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.out_dir = args.out_dir or Path(tmp.name) / "golden"
    out_dir = args.out_dir or GOLDEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = Lfm2Reference(args.model_dir, torch.device("cpu"), dtype=torch.bfloat16, attn="eager", load_table=False)
    arch = ref.arch
    layers = tuple(int(v) for v in args.layers.split(",") if v.strip())
    if args.tiny:
        g = torch.Generator().manual_seed(args.seed)
        ids = [1] + [int(v) for v in torch.randint(4, 150, (args.tokens - 1,), generator=g)]
        ids = ids[:64]
        src = "tiny random ids"
    else:
        seg = json.loads(args.corpus.read_text(encoding="utf-8"))["segments"][0]
        ids = seg["token_ids"][: args.tokens]
        src = f"{args.corpus.name}:{seg['name']}[:{len(ids)}]"
    manifest = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/lfm2/router_golden.py",
                "model_dir": str(args.model_dir), "tiny": bool(args.tiny), "arch": arch.summary(), "real_input": src,
                "tolerances": {"ids": "exact vs `ids` (fp32 contract)", "weights": "rel 1e-3", "logits": "rel 1e-3"},
                "components": {}}

    def emit(name, tensors, meta):
        save_golden(out_dir / f"{name}.safetensors", tensors, {"component": name})
        meta["file"] = f"{name}.safetensors"
        meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
        manifest["components"][name] = meta
        print(f"[router_golden] {name}: rows={meta.get('rows')}", flush=True)

    xs = real_router_inputs(ref, ids, layers)
    for i in layers:
        W = ref.raw(f"model.layers.{i}.feed_forward.gate.weight").to(torch.bfloat16)
        b = ref.raw(f"model.layers.{i}.feed_forward.expert_bias").to(torch.float32)
        meta, t = stats_entry(arch, xs[i], W, b)
        assert meta["contract_equals_hf_fp32"], f"layer {i}: manual router != HF fp32 router"
        meta.update({"case": "real", "layer": i})
        emit(f"router_real_layer_{i:03d}", {"x": xs[i], "weight": W, "bias": b, **t}, meta)
    i0 = layers[0]
    W0 = ref.raw(f"model.layers.{i0}.feed_forward.gate.weight").to(torch.bfloat16)
    b0 = ref.raw(f"model.layers.{i0}.feed_forward.expert_bias").to(torch.float32)
    t, m = tie_case(arch, W0, b0, args.seed)
    emit("router_tie", t, m)
    t, m = bias_case(arch, W0, args.seed)
    emit("router_bias_selects", t, m)
    manifest["peak_rss_gib"] = peak_rss_gb()
    (out_dir / "router_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[router_golden] wrote {out_dir / 'router_manifest.json'}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
