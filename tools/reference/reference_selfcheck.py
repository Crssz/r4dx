"""tools/reference/reference_selfcheck.py

Validates the *reference half* of the Rung 4 KL measurement -- `full_logits_golden.py` -- against
things it does not share code with. A KL number is only as good as the bf16 side it is measured
against, and `full_logits_golden.py` is not an ordinary transformers forward: it builds the model
skeleton on the `meta` device and streams each `Qwen3_5DecoderLayer`'s weights in and out of VRAM,
gathers embedding rows straight from the shard mmap, and evaluates the `[248320, 5120]` lm_head a
vocabulary block at a time. Every one of those is a place a faithful-looking forward can go subtly
wrong, and `--cross-check` (its `--impl model` vs `--impl manual` comparison) cannot see any of
them, because both compositions share `build_layer`, `gather_embedding_rows` and `logits_fp32`.

Two checks:

  --truncated N   (default 4, the important one) Build a REAL, fully resident `Qwen3_5TextModel`
                  with `num_hidden_layers = N` -- ordinary nn.Module, ordinary `load_state_dict`
                  from the same safetensors shards, ordinary `nn.Linear` lm_head over the whole
                  vocabulary -- run it the normal way, and compare it to `StreamingReference` with
                  `max_layers=N` AT EVERY POSITION: hidden states, full-vocab logits, argmax, and
                  the per-position KL between the two log-softmax rows. Nothing on the control side
                  imports the streaming code's helpers. This is what proves the streaming machinery
                  is faithful rather than merely self-consistent. N=4 is chosen because
                  `layer_types[:4]` is `[linear, linear, linear, full]` for this checkpoint, i.e.
                  both the GDN and the full-attention path are exercised, and because a resident
                  4-layer model plus a resident lm_head fits in ~5.5 GiB.

  --noise-floor   Run the FULL depth (all 64 layers) through both compositions (`--impl model` and
                  `--impl manual`) and report the per-position KL between them. That is the bf16
                  reduction-order noise floor of the reference at the depth the real measurement
                  runs at -- the number below which no quantization KL can be interpreted. Slow
                  (two full streaming passes), so it defaults to a short prefix.

Usage (reference venv, HIP device 1):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\reference_selfcheck.py `
        --tokens tools\\reference\\kl_corpus\\tokens.json --truncated 4 --tokens-n 48
    <venv>\\Scripts\\python.exe tools\\reference\\reference_selfcheck.py `
        --tokens tools\\reference\\kl_corpus\\tokens.json --noise-floor --noise-n 256

Peak VRAM: ~5.4 GiB for the truncated control, ~1.6 GiB for the streaming/noise-floor passes.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DEFAULT_MODEL_DIR, load_text_config, resolve_device  # noqa: E402
import full_logits_golden as fg  # noqa: E402

TEXT_PREFIX = "model.language_model."
LM_HEAD_NAME = "lm_head.weight"


def read_tensor(model_dir: Path, weight_map: dict, name: str) -> torch.Tensor:
    from safetensors import safe_open

    if name not in weight_map:
        raise KeyError(f"{name!r} is not in this checkpoint's model.safetensors.index.json")
    with safe_open(str(model_dir / weight_map[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).clone()  # see common.get_tensor's dangling-mmap note


def build_control(model_dir: Path, weight_map: dict, n_layers: int, token_ids: list[int], device):
    """A plain resident `Qwen3_5TextModel` of `n_layers` layers, loaded the ordinary way, plus its
    input embeddings for `token_ids`. Deliberately shares no code with `full_logits_golden`."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    _, text_config = load_text_config(model_dir)
    cfg = copy.deepcopy(text_config)
    cfg.num_hidden_layers = n_layers
    cfg.layer_types = list(text_config.layer_types)[:n_layers]
    print(f"[selfcheck] control: {n_layers} layers, layer_types={cfg.layer_types}")

    with torch.device("meta"):
        ctrl = modeling.Qwen3_5TextModel(cfg)
    sd = {}
    for key in ctrl.state_dict().keys():
        if key == "embed_tokens.weight":
            continue  # 2.5 GiB we do not need resident on the card; indexed on the host below
        sd[key] = read_tensor(model_dir, weight_map, TEXT_PREFIX + key).to(device=device,
                                                                          dtype=torch.bfloat16)
    table = read_tensor(model_dir, weight_map, TEXT_PREFIX + "embed_tokens.weight")
    embeds = table[torch.tensor(token_ids, dtype=torch.long)].to(device=device,
                                                                 dtype=torch.bfloat16).unsqueeze(0)
    del table
    # A 1-row placeholder keeps `load_state_dict(strict=True)` honest without the 2.5 GiB table;
    # the module is never called, because the forward below is given `inputs_embeds`.
    ctrl.embed_tokens = torch.nn.Embedding(1, cfg.hidden_size, device="meta")
    sd["embed_tokens.weight"] = torch.zeros(1, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    ctrl.load_state_dict(sd, assign=True, strict=True)
    ctrl.eval()
    return ctrl, embeds


def compare(label: str, hidden_ctrl, logits_ctrl, hidden_test, logits_test, n_pos: int) -> dict:
    hc = hidden_ctrl.float().cpu()
    hs = hidden_test.float().cpu()
    lc = logits_ctrl.double().cpu()
    ls = logits_test.double().cpu()
    dh = (hs - hc).abs()
    dl = (ls - lc).abs()
    per_pos_h = dh.max(dim=1).values
    per_pos_l = dl.max(dim=1).values
    lsm_c = torch.log_softmax(lc, dim=-1)
    lsm_s = torch.log_softmax(ls, dim=-1)
    kl = (lsm_c.exp() * (lsm_c - lsm_s)).sum(dim=1)
    agree = int((lc.argmax(dim=1) == ls.argmax(dim=1)).sum())
    h_scale = float(hc.abs().max())
    # bf16 keeps 8 explicit mantissa bits, so one ulp at the largest hidden magnitude is the scale a
    # faithful-but-differently-ordered reduction is allowed to differ by.
    ulp = h_scale * 2 ** -8
    out = {
        "impl": label, "positions": n_pos,
        "hidden_max_abs_diff": float(dh.max()), "hidden_abs_max": h_scale, "bf16_ulp": ulp,
        "hidden_worst_position": int(per_pos_h.argmax()),
        "logits_max_abs_diff": float(dl.max()), "logits_abs_max": float(lc.abs().max()),
        "logits_worst_position": int(per_pos_l.argmax()),
        "argmax_agree": agree,
        "kl_max": float(kl.max()), "kl_mean": float(kl.mean()),
    }
    print(f"\n--- streaming --impl {label} vs the resident control")
    print(f"  hidden : max|diff| {out['hidden_max_abs_diff']:.4e} at position "
          f"{out['hidden_worst_position']}  (|hidden|max {h_scale:.3f}, one bf16 ulp there "
          f"{ulp:.4f})")
    print(f"  logits : max|diff| {out['logits_max_abs_diff']:.4e} at position "
          f"{out['logits_worst_position']}  (|logit|max {out['logits_abs_max']:.3f})")
    print(f"  argmax agreement : {agree}/{n_pos} positions")
    print(f"  per-position KL(control || streaming) : max {out['kl_max']:.3e}, "
          f"mean {out['kl_mean']:.3e}")
    return out


def run_truncated(model_dir: Path, weight_map: dict, token_ids: list[int], n_layers: int,
                  device, chunk: int) -> dict:
    t0 = time.perf_counter()
    ctrl, embeds = build_control(model_dir, weight_map, n_layers, token_ids, device)
    with torch.no_grad():
        hidden_ctrl = ctrl(inputs_embeds=embeds, use_cache=False).last_hidden_state[0].clone()
        lm_w = read_tensor(model_dir, weight_map, LM_HEAD_NAME).to(device=device,
                                                                   dtype=torch.bfloat16)
        logits_ctrl = torch.nn.functional.linear(hidden_ctrl, lm_w).float().cpu()
    hidden_ctrl = hidden_ctrl.cpu()
    del lm_w, ctrl, embeds
    torch.cuda.empty_cache() if device.type == "cuda" else None
    ctrl_peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    print(f"[selfcheck] control forward done in {time.perf_counter() - t0:.1f}s, "
          f"peak VRAM {ctrl_peak:.2f} GiB")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ref = fg.StreamingReference(model_dir, device, max_layers=n_layers)
    results = []
    for impl in ("model", "manual"):
        with torch.no_grad():
            h = ref.forward_hidden(token_ids, impl=impl)
            lg = ref.logits_fp32(h, chunk=chunk).cpu()
        results.append(compare(impl, hidden_ctrl, logits_ctrl, h.cpu(), lg, len(token_ids)))
        del h, lg
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stream_peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    print(f"[selfcheck] streaming peak VRAM {stream_peak:.2f} GiB")
    return {"n_layers": n_layers, "positions": len(token_ids), "comparisons": results,
            "control_peak_vram_gib": ctrl_peak, "streaming_peak_vram_gib": stream_peak}


def run_noise_floor(model_dir: Path, token_ids: list[int], device, chunk: int) -> dict:
    ref = fg.StreamingReference(model_dir, device)
    print(f"[selfcheck] noise floor: {ref.n_layers} layers, {len(token_ids)} positions, "
          f"V={ref.vocab_size}")
    rows = {}
    for impl in ("model", "manual"):
        t0 = time.perf_counter()
        with torch.no_grad():
            h = ref.forward_hidden(token_ids, impl=impl)
            lg = ref.logits_fp32(h, chunk=chunk)
            rows[impl] = torch.log_softmax(lg.double(), dim=-1).cpu()
        del h, lg
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[selfcheck]   --impl {impl} in {time.perf_counter() - t0:.1f}s")
    a, b = rows["model"], rows["manual"]
    kl = (a.exp() * (a - b)).sum(dim=1)
    kl_rev = (b.exp() * (b - a)).sum(dim=1)
    agree = int((a.argmax(1) == b.argmax(1)).sum())
    out = {
        "n_layers": ref.n_layers, "positions": len(token_ids),
        "kl_mean": float(kl.mean()), "kl_median": float(kl.median()),
        "kl_p99": float(torch.quantile(kl, 0.99)), "kl_max": float(kl.max()),
        "kl_max_position": int(kl.argmax()), "kl_reverse_mean": float(kl_rev.mean()),
        "top1_agreement_pct": 100.0 * agree / len(token_ids), "top1_agree": agree,
    }
    print(f"\n[selfcheck] reference self-noise at full depth, per-position KL(model || manual):")
    print(f"  mean {out['kl_mean']:.6f}  median {out['kl_median']:.6f}  p99 {out['kl_p99']:.6f}  "
          f"max {out['kl_max']:.6f} at position {out['kl_max_position']}")
    print(f"  reverse direction mean {out['kl_reverse_mean']:.6f}")
    print(f"  top-1 self-agreement {out['top1_agreement_pct']:.2f}% ({agree}/{len(token_ids)})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path,
                    default=Path(__file__).parent / "kl_corpus" / "tokens.json")
    ap.add_argument("--segment", default=None, help="which segment to take tokens from (default: the first)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--truncated", type=int, default=4,
                    help="layers for the resident control (0 disables this check)")
    ap.add_argument("--tokens-n", type=int, default=48, help="positions for the truncated control")
    ap.add_argument("--noise-floor", action="store_true",
                    help="also run the full-depth --impl model vs --impl manual noise floor")
    ap.add_argument("--noise-n", type=int, default=256, help="positions for the noise floor")
    ap.add_argument("--lm-head-chunk", type=int, default=32768)
    ap.add_argument("--out", type=Path, default=None, help="write the results as JSON here")
    args = ap.parse_args()

    device = resolve_device(args.device)  # enforces $env:HIP_VISIBLE_DEVICES == '1' for cuda
    torch.set_grad_enabled(False)
    with open(args.tokens, "r", encoding="utf-8") as f:
        doc = json.load(f)
    segs = doc["segments"]
    seg = segs[0] if args.segment is None else next(s for s in segs if s["name"] == args.segment)
    print(f"[selfcheck] model_dir={args.model_dir} segment={seg['name']} device={device}")

    with open(args.model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    run = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "model_dir": str(args.model_dir), "segment": seg["name"], "device": str(device),
           "torch_version": torch.__version__}
    if args.truncated:
        run["truncated_control"] = run_truncated(
            args.model_dir, weight_map, seg["token_ids"][: args.tokens_n], args.truncated,
            device, args.lm_head_chunk)
    if args.noise_floor:
        run["noise_floor"] = run_noise_floor(
            args.model_dir, seg["token_ids"][: args.noise_n], device, args.lm_head_chunk)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        print(f"\n[selfcheck] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
