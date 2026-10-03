"""tools/reference/gemma/layer_golden_gemma.py  (M0-8, rung-3 layer goldens)

Builds single Gemma 4 decoder layers straight from the reference `transformers` install's
`modeling_gemma4_unified.py` with the REAL checkpoint weights, runs them on a fixed seeded input
and dumps inputs / outputs / intermediates to .safetensors plus a manifest.json: the contract the
r4dx C++ layer tests (`test_gemma_attn_layer.cpp`, M1-21) compare against (docs/validation.md rung 3).

Components (docs/gemma4-plan.md 6.3):

  layer_000_sliding    layer 0: head_dim 256, 16 q / 8 kv heads, window 1024, rope theta 1e4
  layer_005_full       layer 5: head_dim 512, 1 kv head, k_eq_v (no v_proj; V = raw k_proj output ->
                       v_norm), proportional rope theta 1e6 (64 rotated dims), attention scaling 1.0
  sliding_ring_wrap    the sliding layer over `ring_prefill` (1500) + `decode` positions: the last
                       decode rows attend through a window that has wrapped the 1024-slot ring
  embed_scale          gathered embedding rows * bf16(sqrt(hidden)) (= 62.0 for 3840); text only
  final_norm_softcap   final RMSNorm + tied lm_head on a vocab slice + softcap 30*tanh(x/30), in the
                       HF-faithful bf16 and the fp32 variants, plus a pure softcap probe

Each layer component runs ONE pass over `prefill + decode` (64 + 4) positions with an explicit causal
(+ sliding) mask and no cache. Rows [0, prefill) are the prefill output; rows [prefill, prefill+decode)
are what a decode of 4 tokens after that prefill must produce (causal attention makes the one-pass
rows equal the cached ones, up to GEMM reduction order, which the fp32-twin floor below measures).

Intermediates per layer: raw / normed / post-rope q, k, v (layer 5: `v_raw` is the k_proj output),
attention output before and after o_proj, the four layernorm outputs, mid residual, mlp gate / up / act /
down, pre-`layer_scalar` and layer output, plus cos / sin, `layer_scalar`, and fp32 `inv_freq` of both
layer types. Every component is also run in an fp32 twin (same bf16-rounded weights and input) and
the manifest records the bf16-vs-fp32 error floor: tolerances for the C++ tests should be a small multiple
of it, not the Qwen constants. `--perturb-norms` adds seeded noise to the norm weights (only meant for
random weights: the real ones are already far from 1; the manifest records their spread).

GPU: the real run is handed to the user (docs/gemma4-plan.md M0-8):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\layer_golden_gemma.py --device cuda --out-dir tools\\reference\\golden_out\\gemma

`--device cpu` also works (slow for the 1500-position component). `--tiny` runs everything on a tiny
random-config checkpoint on CPU (the M0-6 smoke). See common_gemma.py for the device rule.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import tempfile
from collections import UserDict
from pathlib import Path

if "--tiny" in sys.argv:  # a smoke must never see a GPU
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from gemma.arch import EMBED_NAME, FULL, SLIDING, GemmaArch  # noqa: E402
from gemma.common_gemma import (  # noqa: E402
    BOS_ID,
    DEFAULT_MODEL_DIR,
    GOLDEN_DIR,
    perturb_norm_weights,
    resolve_device,
    save_golden,
    sha256_file,
    tensor_manifest_entry,
)
from gemma.ref import GemmaReference, add_tiny_args, additive_mask, build_tiny_checkpoint  # noqa: E402

TOLERANCES = {
    "note": "Starting points, not gates: scale from fp32_floor in each component's `fp32_floor` entry "
            "(bf16 weights/inputs run in bf16 vs the same run in fp32). Gemma has no Qwen-style constants: "
            "the C++ tests should use a small multiple (suggest 4x) of the measured floor per tensor.",
    "bf16_matmul_rel_fro": 2e-2,
    "fp32_elementwise_rel_fro": 1e-4,
    "layer_out_rel_fro_floor_multiple": 4,
}
FLOOR_TENSORS = ("q_post_rope", "k_post_rope", "attn_out", "mid_residual", "mlp_down", "pre_scalar", "layer_out")


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def seeded_normal(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def run_layer(layer, rotary, arch: GemmaArch, i: int, x: torch.Tensor, record: bool = True) -> dict:
    """One decoder layer over x `[1, T, H]` from position 0 (no cache). Returns the tensors of the
    module docstring (batch squeezed) in x.dtype; `layer_out` always, the rest when `record`."""
    import transformers.models.gemma4_unified.modeling_gemma4_unified as M

    T = x.shape[1]
    lt = arch.layer_types[i]
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    cos, sin = rotary(x, pos, lt)
    mask = additive_mask(T, T, 0, arch.window if lt == SLIDING else None, x.device, x.dtype)
    cap: dict[str, torch.Tensor] = {}
    handles = []
    rope_calls: list[torch.Tensor] = []
    orig_rope = M.apply_rotary_pos_emb

    def grab(name, pre=False):
        def hook(mod, args, out=None):
            t = args[0] if pre else out
            cap[name] = t.detach().clone()
        return hook

    def wrapped_rope(t, c, s, unsqueeze_dim=1):
        y = orig_rope(t, c, s, unsqueeze_dim=unsqueeze_dim)
        rope_calls.append(y.detach().clone())
        return y

    if record:
        sa = layer.self_attn
        for name, mod in (("q_raw", sa.q_proj), ("k_raw", sa.k_proj), ("q_normed", sa.q_norm),
                          ("k_normed", sa.k_norm), ("v_normed", sa.v_norm), ("attn_out", sa.o_proj),
                          ("input_ln", layer.input_layernorm), ("post_attn_ln", layer.post_attention_layernorm),
                          ("pre_ff_ln", layer.pre_feedforward_layernorm),
                          ("post_ff_ln", layer.post_feedforward_layernorm),
                          ("mlp_gate", layer.mlp.gate_proj), ("mlp_up", layer.mlp.up_proj),
                          ("mlp_down", layer.mlp.down_proj)):
            if mod is not None:
                handles.append(mod.register_forward_hook(grab(name)))
        if sa.v_proj is not None:
            handles.append(sa.v_proj.register_forward_hook(grab("v_raw")))
        handles.append(sa.o_proj.register_forward_pre_hook(grab("attn_out_pre_o", pre=True)))
        if isinstance(layer.mlp.act_fn, torch.nn.Module):
            handles.append(layer.mlp.act_fn.register_forward_hook(grab("mlp_act")))
        M.apply_rotary_pos_emb = wrapped_rope
    try:
        out = layer(x, shared_kv_states=UserDict(), position_embeddings=(cos, sin), attention_mask=mask,
                    position_ids=pos, past_key_values=None)
    finally:
        M.apply_rotary_pos_emb = orig_rope
        for h in handles:
            h.remove()
    res = {"layer_out": out[0].detach().clone()}
    if not record:
        return res
    if "v_raw" not in cap:  # full layer (k_eq_v): V is the raw k_proj output
        cap["v_raw"] = cap["k_raw"]
    if "mlp_act" not in cap:
        cap["mlp_act"] = layer.mlp.act_fn(cap["mlp_gate"])
    if len(rope_calls) != 2:
        raise RuntimeError(f"expected 2 apply_rotary_pos_emb calls (q, k), saw {len(rope_calls)}")
    cap["q_post_rope"], cap["k_post_rope"] = rope_calls
    cap["mid_residual"] = x + cap["post_attn_ln"]
    cap["pre_scalar"] = cap["mid_residual"] + cap["post_ff_ln"]
    cap["layer_scalar"] = layer.layer_scalar.detach().clone()
    cap["cos"], cap["sin"] = cos[0].detach().clone(), sin[0].detach().clone()
    res.update({k: (v[0] if v.dim() > 1 and v.shape[0] == 1 and k not in ("layer_scalar",) else v)
                for k, v in cap.items()})
    return res


def fp32_floor(layer, rotary, arch, i, x, got: dict) -> dict:
    """bf16-vs-fp32-twin error of the recorded tensors (same weights rounded to bf16, same input)."""
    twin = copy.deepcopy(layer).float()
    want = run_layer(twin, rotary, arch, i, x.float(), record=True)
    out = {}
    for k in FLOOR_TENSORS:
        a, b = got[k].float(), want[k].float()
        out[k] = {"rel_fro": rel_fro(a, b), "max_abs": float((a - b).abs().max()),
                  "max_abs_over_amax": float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))}
    return out


def norm_weight_stats(layer) -> dict:
    stats = {}
    for name, mod in layer.named_modules():
        w = getattr(mod, "weight", None)
        if type(mod).__name__.endswith("RMSNorm") and w is not None:
            d = (w.float() - 1.0).abs()
            stats[name] = {"min": float(w.float().min()), "max": float(w.float().max()),
                           "mean": float(w.float().mean()), "mean_abs_dev_from_1": float(d.mean())}
    return stats


def tensors_to_save(res: dict, x: torch.Tensor, extra: dict | None = None) -> dict:
    t = {"x": x[0].detach().clone()}
    t.update({k: v for k, v in res.items()})
    if extra:
        t.update(extra)
    return t


def layer_component(ref: GemmaReference, i: int, x: torch.Tensor, name: str, prefill: int, decode: int,
                    perturb: int | None) -> tuple[dict, dict]:
    arch = ref.arch
    layer = ref.build_layer(i)
    if perturb is not None:
        perturb_norm_weights(layer, perturb)
    res = run_layer(layer, ref.rotary, arch, i, x)
    floor = fp32_floor(layer, ref.rotary, arch, i, x, res)
    meta = {
        "component": name, "layer": i, "layer_type": arch.layer_types[i], "head_dim": arch.head_dim_of(i),
        "n_heads": arch.n_heads, "n_kv_heads": arch.kv_heads_of(i), "has_v_proj": arch.has_v_proj(i),
        "k_eq_v": arch.k_eq_v and arch.is_full(i), "rope_theta": arch.rope_theta(i),
        "rotary_dims": arch.rotary_dims(i), "window": arch.window if arch.layer_types[i] == SLIDING else None,
        "attention_scaling": 1.0, "T": int(x.shape[1]), "prefill": prefill, "decode": decode,
        "layer_scalar": float(res["layer_scalar"].float().item()), "fp32_floor": floor,
        "norm_weight_stats": norm_weight_stats(layer),
        "decode_rows": "rows [prefill, prefill+decode) of every [T, ...] tensor",
    }
    return res, meta


def ring_component(ref: GemmaReference, i: int, x: torch.Tensor, name: str, decode: int, seed: int) -> tuple[dict, dict]:
    """The sliding layer over ring_prefill + decode positions. Saves x, layer_out, attn_out and the
    K/V actually attended; q only for the last rows. Also checks the window claim: replacing every
    input row that lies before the last row's window must not change the last row's output."""
    arch = ref.arch
    layer = ref.build_layer(i)
    res = run_layer(layer, ref.rotary, arch, i, x)
    T = x.shape[1]
    p = T - 1
    outside = p - arch.window  # positions [0, outside] are outside the last row's window
    x2 = x.clone()
    if outside >= 0:
        x2[:, : outside + 1] = seeded_normal(x2[:, : outside + 1].shape, seed + 99).to(x.dtype).to(x.device)
    res2 = run_layer(layer, ref.rotary, arch, i, x2, record=False)
    diff = float((res2["layer_out"][p].float() - res["layer_out"][p].float()).abs().max())
    keep = {k: res[k] for k in ("layer_out", "attn_out", "k_post_rope", "v_normed", "cos", "sin")}
    keep["q_post_rope_tail"] = res["q_post_rope"][-2 * decode:].clone()
    meta = {"component": name, "layer": i, "layer_type": arch.layer_types[i], "T": int(T),
            "ring_slots": arch.window, "positions_wrapped": max(0, T - arch.window), "decode": decode,
            "window_independence_max_abs_diff_last_row": diff,
            "window_independence_note": "output of the last position after replacing all input rows before its "
                                        "window; 0 means the layer really attends to <= `window` keys",
            "q_post_rope_tail": f"last {2 * decode} rows of q_post_rope"}
    return keep, meta


def embed_component(ref: GemmaReference, seed: int, n: int = 16) -> tuple[dict, dict]:
    arch = ref.arch
    g = torch.Generator().manual_seed(seed + 5)
    ids = [BOS_ID] + [int(v) for v in torch.randint(4, arch.vocab, (n - 1,), generator=g)]
    rows = torch.stack([ref.index.get_row_slice(EMBED_NAME, t, t + 1)[0] for t in ids]).to(ref.device, ref.dtype)
    scale32 = torch.tensor(arch.hidden ** 0.5, dtype=torch.float32)
    scale_bf = scale32.to(ref.dtype)
    scaled = rows * scale_bf.to(ref.device)
    tensors = {"token_ids": torch.tensor(ids, dtype=torch.int64), "embed_rows": rows, "embed_scaled": scaled,
               "scale_f32": scale32.reshape(1), "scale_bf16": scale_bf.reshape(1)}
    meta = {"component": "embed_scale", "scale_f32": float(scale32), "scale_bf16": float(scale_bf.float()),
            "rule": "HF: embedding(ids) * embed_scale.to(weight.dtype), i.e. rows * bf16(sqrt(hidden)); "
                    "image / audio embeddings are NOT scaled"}
    return tensors, meta


def final_component(ref: GemmaReference, seed: int, vocab_slice: int = 4096, n: int = 16) -> tuple[dict, dict]:
    arch = ref.arch
    vs = min(vocab_slice, arch.vocab)
    x = (seeded_normal((n, arch.hidden), seed + 6) * 3.0).to(ref.dtype).to(ref.device)
    normed = ref.final_norm(x)
    w = ref.index.get_row_slice(EMBED_NAME, 0, vs).to(ref.device, ref.dtype)
    cap = arch.softcap
    raw_bf = torch.nn.functional.linear(normed, w)
    raw32 = normed.float() @ w.float().T
    probe = torch.linspace(-200.0, 200.0, 257)
    tensors = {"x": x, "normed": normed, "logits_raw_bf16": raw_bf, "logits_raw_f32": raw32,
               "cap_probe_in": probe, "cap_probe_out_f32": torch.tanh(probe / cap) * cap,
               "cap_probe_out_bf16": torch.tanh(probe.to(ref.dtype) / cap) * cap}
    if cap:
        tensors["logits_capped_hf_bf16"] = torch.tanh(raw_bf / cap) * cap          # HF: capped in bf16
        tensors["logits_capped_f32"] = torch.tanh(raw32 / cap) * cap               # more accurate variant
    meta = {"component": "final_norm_softcap", "vocab_slice": vs, "softcap": cap, "eps": arch.eps,
            "note": "HF applies the softcap to the lm_head output in ITS dtype (bf16): logits_capped_hf_bf16. "
                    "logits_capped_f32 is the fp32 variant (lm_head accumulated in fp32); the difference is the "
                    "bf16 spacing near |x|=30 (0.125)",
            "capped_hf_vs_f32_max_abs": float((tensors.get("logits_capped_hf_bf16", raw_bf).float()
                                               - tensors.get("logits_capped_f32", raw32)).abs().max())}
    return tensors, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {GOLDEN_DIR}")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sliding-layer", type=int, default=None, help="default 0 (tiny: 0)")
    ap.add_argument("--full-layer", type=int, default=None, help="default 5 (tiny: 1)")
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--ring-prefill", type=int, default=None,
                    help="prefill length of the ring-wrap component (default 1500 for window 1024, else 2*window+3)")
    ap.add_argument("--components", default="layers,ring,embed,final",
                    help="comma list of: layers (sliding+full), ring, embed, final")
    ap.add_argument("--perturb-norms", action="store_true",
                    help="seeded noise on every norm weight (random-weight runs only; recorded)")
    ap.add_argument("--input-scale", type=float, default=1.0, help="std of the seeded N(0,1) layer input")
    add_tiny_args(ap)
    args = ap.parse_args(argv)

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.device = "cpu"
        args.perturb_norms = False  # the tiny builder already perturbed them
        if args.out_dir is None:
            args.out_dir = Path(tmp.name) / "golden"
    out_dir = args.out_dir or GOLDEN_DIR
    device = resolve_device(args.device)
    arch = GemmaArch.from_model_dir(args.model_dir)
    sl = args.sliding_layer if args.sliding_layer is not None else 0
    fl = args.full_layer if args.full_layer is not None else (1 if args.tiny else 5)
    if arch.layer_types[sl] != SLIDING or arch.layer_types[fl] != FULL:
        raise SystemExit(f"layer {sl} is {arch.layer_types[sl]}, layer {fl} is {arch.layer_types[fl]}: "
                         "need a sliding and a full layer")
    ring_prefill = args.ring_prefill or (1500 if arch.window == 1024 else 2 * arch.window + 3)
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    perturb = args.seed + 100 if args.perturb_norms else None

    print(f"[layer_golden_gemma] {arch.summary()['layer_types']} window={arch.window} on {device}, "
          f"out {out_dir}", flush=True)
    ref = GemmaReference(args.model_dir, device, load_table=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/gemma/layer_golden_gemma.py",
        "model_dir": str(args.model_dir), "tiny": bool(args.tiny), "device": str(device), "dtype": "bfloat16",
        "config_sha256": sha256_file(args.model_dir / "config.json"),
        "weights_file_bytes": (args.model_dir / "model.safetensors").stat().st_size
        if (args.model_dir / "model.safetensors").is_file() else None,
        "arch": arch.summary(), "seed": args.seed, "input": f"seeded N(0,1)*{args.input_scale}, torch.Generator(cpu), cast to bf16",
        "perturb_norms_seed": perturb, "torch_version": torch.__version__, "tolerances": TOLERANCES,
        "inv_freq": {}, "components": {},
    }
    import transformers

    manifest["transformers_version"] = transformers.__version__
    inv = {}
    for lt in (SLIDING, FULL):
        f = getattr(ref.rotary, f"{lt}_inv_freq").detach().float().cpu()
        inv[f"inv_freq_{'sliding' if lt == SLIDING else 'full'}"] = f
        manifest["inv_freq"][lt] = {"len": int(f.numel()), "n_zero": int((f == 0).sum()),
                                    "rope_theta": arch.rope.get(lt, {}).get("rope_theta")}
    save_golden(out_dir / "rope_inv_freq.safetensors", inv, {"note": "fp32 inv_freq per layer type (full: zeros past the rotated dims)"})

    def emit(name: str, tensors: dict, meta: dict) -> None:
        save_golden(out_dir / f"{name}.safetensors", tensors, {"component": name})
        meta["file"] = f"{name}.safetensors"
        meta["tensors"] = {k: tensor_manifest_entry(v) for k, v in tensors.items()}
        manifest["components"][name] = meta
        print(f"[layer_golden_gemma] {name}: {len(tensors)} tensors", flush=True)

    T = args.prefill + args.decode
    if "layers" in wanted:
        for idx, tag in ((sl, "sliding"), (fl, "full")):
            x = (seeded_normal((1, T, arch.hidden), args.seed + idx, args.input_scale)).to(ref.dtype).to(device)
            name = f"layer_{idx:03d}_{tag}"
            res, meta = layer_component(ref, idx, x, name, args.prefill, args.decode, perturb)
            meta["fp32_floor_layer_out_rel_fro"] = meta["fp32_floor"]["layer_out"]["rel_fro"]
            emit(name, tensors_to_save(res, x), meta)
    if "ring" in wanted:
        Tr = ring_prefill + args.decode
        x = (seeded_normal((1, Tr, arch.hidden), args.seed + 7, args.input_scale)).to(ref.dtype).to(device)
        keep, meta = ring_component(ref, sl, x, "sliding_ring_wrap", args.decode, args.seed)
        emit("sliding_ring_wrap", {"x": x[0].clone(), **keep}, meta)
    if "embed" in wanted:
        t, meta = embed_component(ref, args.seed)
        emit("embed_scale", t, meta)
    if "final" in wanted:
        t, meta = final_component(ref, args.seed)
        emit("final_norm_softcap", t, meta)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[layer_golden_gemma] wrote {out_dir / 'manifest.json'}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
