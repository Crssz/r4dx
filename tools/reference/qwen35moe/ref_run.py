"""`ref_run.py`: layer-major runs of the qwen3_5_moe streaming reference over MANY token segments, resumable in GPU slices.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\ref_run.py logprobs --tokens T.json --out-dir D --work W
        [--dtype bf16|fp32] [--record-kv kv_amax.json] [--budget-seconds 840] [--device cuda]
    ... ref_run.py kvcalib --tokens T.json --out kvcalib.json --work W [--budget-seconds 840]

Both run every segment through layer 0, then through layer 1, ... (each layer's 5.3 GB of weights is read from the shards ONCE for all
segments, not once per segment as `Qwen35MoeStreamingReference.run_layers` would), keeping one residual [1, T, 3072] per segment on the
device. After every layer the elapsed time is checked against `--budget-seconds`: past it the residuals go to `<work>\\resid.pt` and the
process exits 75 ("run me again"; pipeline.ps1 / the drivers around it use that convention), so a GPU lock can be released between slices.

* `logprobs` writes, per segment, the Rung 4 artifact of docs/validation.md ("Rung 4 tooling"): `<name>.logprobs.f16`
  (little-endian float16 [T-1, V], row i = log_softmax(logits at position i), clamped at -1e4 in fp32 before the cast) and
  `<name>.meta.json` (T, V, dtype, rows, source "reference", torch_dtype, sha256_of_token_ids_json, n_layers). bf16 = the stock HF bf16
  stack (lm_head in bf16, then widened); fp32 = the same modules in float32 ("truth"). Each segment is scored from position 0.
* `kvcalib` records, for the 12 full-attention layers, the per-kv-head absolute maximum of K (after q/k-norm and rope: what the paged fp8
  cache stores) and of V (raw v_proj output), over every segment, plus the per-segment maxima; the merged JSON is
  `{"<layer>": {"k_amax": [kv_heads], "v_amax": [kv_heads]}}`, which `r4dx-convert --kv-calib` turns into descale = amax / 448.
  `logprobs --record-kv` records the same numbers for the segments it scores (held-out text: how far past the calibrated amax does it go).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import sha256_file  # noqa: E402
from qwen35moe.arch import FULL  # noqa: E402
from qwen35moe.ref import Qwen35MoeStreamingReference  # noqa: E402

EXIT_MORE = 75
LOGPROB_CLAMP = -1e4
DEFAULT_MODEL = Path(r"E:\models\r4dx\Ornith-1.5-122B")


def token_ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids], separators=(",", ":")).encode("utf-8")).hexdigest()


def load_segments(path: Path, names: list[str] | None) -> list[tuple[str, list[int]]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = [(s["name"], [int(t) for t in s["token_ids"]]) for s in doc["segments"] if names is None or s["name"] in names]
    if names is not None:
        missing = set(names) - {n for n, _ in segs}
        if missing:
            raise SystemExit(f"segments {sorted(missing)} are not in {path}")
    if not segs:
        raise SystemExit(f"no segments in {path}")
    return segs


class KvRecorder:
    """Registers an attention implementation that records the per-kv-head amax of the K / V the attention module receives (K: after
    q/k-norm and rope, V: raw) and then runs HF's own sdpa path unchanged."""

    NAME = "r4dx_kv_record"

    def __init__(self):
        self.k: dict[int, dict[str, torch.Tensor]] = {}
        self.v: dict[int, dict[str, torch.Tensor]] = {}
        self.current = ""

    def install(self, ref: Qwen35MoeStreamingReference) -> None:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        rec = self

        def attention(module, query, key, value, attention_mask, **kwargs):
            li = int(module.layer_idx)
            ka = key.detach().abs().amax(dim=(0, 2, 3)).float().cpu()    # key / value: [1, kv_heads, T, head_dim]
            va = value.detach().abs().amax(dim=(0, 2, 3)).float().cpu()
            rec.k.setdefault(li, {})[rec.current] = ka
            rec.v.setdefault(li, {})[rec.current] = va
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

        AttentionInterface.register(self.NAME, attention)
        ref.text_config._attn_implementation = self.NAME

    def merged(self) -> dict:
        out = {}
        for li in sorted(self.k):
            ks = torch.stack(list(self.k[li].values()))
            vs = torch.stack(list(self.v[li].values()))
            out[str(li)] = {"k_amax": [float(x) for x in ks.amax(0)], "v_amax": [float(x) for x in vs.amax(0)],
                            "segments": int(ks.shape[0]),
                            "per_segment": {n: {"k": [float(x) for x in self.k[li][n]], "v": [float(x) for x in self.v[li][n]]}
                                            for n in self.k[li]}}
        return out

    def state(self) -> dict:
        return {"k": {li: {n: t.tolist() for n, t in d.items()} for li, d in self.k.items()},
                "v": {li: {n: t.tolist() for n, t in d.items()} for li, d in self.v.items()}}

    def load_state(self, st: dict) -> None:
        self.k = {int(li): {n: torch.tensor(x) for n, x in d.items()} for li, d in st["k"].items()}
        self.v = {int(li): {n: torch.tensor(x) for n, x in d.items()} for li, d in st["v"].items()}


def run(args, mode: str) -> int:
    t_start = time.perf_counter()
    dtype = torch.float32 if getattr(args, "dtype", "bf16") == "fp32" else torch.bfloat16
    segs = load_segments(Path(args.tokens), args.segments.split(",") if getattr(args, "segments", None) else None)
    if mode == "logprobs":
        bad = [n for n, _ in segs if "/" in n or "\\" in n]
        if bad:
            raise SystemExit(f"segment names become file names: {bad[:3]} contain a path separator")
    ref = Qwen35MoeStreamingReference(args.model_dir, args.device, dtype=dtype)
    device, arch = ref.device, ref.arch
    mix =getattr(args, "mix", None)
    if mix:
        # weight-only diagnostic: every trellis linear of the mix replaced by its reconstruction (kl_mix.MixReader), the rest stock
        import re

        from qwen35moe.kl_mix import MixReader
        from qwen35moe.ref import CheckpointReader

        lo, _, hi = (args.mix_layers or "").partition("-")
        layer_set = set(range(int(lo), int(hi or lo) + 1)) if lo else None
        pat = re.compile(args.mix_match) if args.mix_match else None

        class FilteredMix(MixReader):
            """MixReader that reconstructs only the tensors of --mix-layers whose name matches --mix-match (the rest stay stock)."""

            def get(self, name, rows=None):
                m = re.search(r"layers\.(\d+)\.", name)
                use = (layer_set is None or (m is not None and int(m.group(1)) in layer_set)) and (pat is None or pat.search(name))
                return super().get(name, rows) if use else CheckpointReader.get(self, name, rows)

        ref.reader = FilteredMix(args.model_dir, mix, device)
        ref.reader.weight_dtype = dtype
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ident = {"mode": mode, "dtype": str(dtype), "tokens": str(Path(args.tokens).resolve()), "tokens_sha256": sha256_file(Path(args.tokens)),
             "segments": [n for n, _ in segs], "model_dir": str(args.model_dir), "mix": str(mix) if mix else None,
             "mix_layers": getattr(args, "mix_layers", None), "mix_match": getattr(args, "mix_match", None)}
    kv = KvRecorder() if (mode == "kvcalib" or getattr(args, "record_kv", None)) else None
    if kv is not None:
        kv.install(ref)

    state_path, resid_path = work / "state.json", work / "resid.pt"
    next_layer = 0
    resid: list[torch.Tensor] = []
    if state_path.exists() and not args.fresh:
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st["ident"] != ident:
            raise SystemExit(f"{state_path} belongs to another run ({st['ident']}); use --fresh or another --work")
        next_layer = st["next_layer"]
        if kv is not None:
            kv.load_state(st["kv"])
        saved = torch.load(resid_path, map_location="cpu")
        resid = [t.to(device) for t in saved]
        print(f"[ref_run] resumed at layer {next_layer}/{arch.n_layers}", flush=True)
    else:
        resid = [ref.embed(ids).unsqueeze(0) for _, ids in segs]
    pes = [None] * len(segs)

    for i in range(next_layer, arch.n_layers):
        t0 = time.perf_counter()
        layer = ref._build_layer(i)
        t_load = time.perf_counter() - t0
        with torch.no_grad():
            for s, (name, ids) in enumerate(segs):
                if pes[s] is None:
                    pes[s] = ref.position_embeddings(len(ids), resid[s])
                if kv is not None:
                    kv.current = name
                resid[s] = ref.layer_forward(layer, resid[s], pes[s])
        del layer
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(f"[ref_run] layer {i} ({arch.layer_types[i]}): load {t_load:.1f}s, total {time.perf_counter() - t0:.1f}s "
              f"({sum(len(x) for _, x in segs)} tokens in {len(segs)} segments)", flush=True)
        more = i + 1 < arch.n_layers
        if more and args.budget_seconds and time.perf_counter() - t_start > args.budget_seconds:
            torch.save([t.cpu() for t in resid], resid_path)
            doc = {"ident": ident, "next_layer": i + 1}
            if kv is not None:
                doc["kv"] = kv.state()
            state_path.write_text(json.dumps(doc), encoding="utf-8")
            print(f"[ref_run] budget reached after layer {i}; state saved; run again", flush=True)
            return EXIT_MORE

    if mode == "kvcalib":
        merged = kv.merged()
        want = [i for i in range(arch.n_layers) if arch.layer_types[i] == FULL]
        if sorted(int(k) for k in merged) != want:
            raise SystemExit(f"recorded layers {sorted(merged)} != the full-attention layers {want}")
        out = Path(args.out)
        out.write_text(json.dumps({k: {"k_amax": v["k_amax"], "v_amax": v["v_amax"]} for k, v in merged.items()}, indent=1), encoding="utf-8")
        meta = {"tool": "tools/reference/qwen35moe/ref_run.py kvcalib", "ident": ident, "tokens": sum(len(x) for _, x in segs),
                "segments": len(segs), "longest_segment": max(len(x) for _, x in segs), "layers": merged,
                "convention": "descale = amax / 448 (r4dx-convert --kv-calib); K after q/k-norm + rope, V raw; bf16 HF stack"}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[ref_run] wrote {out}: {len(merged)} layers, {meta['tokens']} tokens", flush=True)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for s, (name, ids) in enumerate(segs):
                hidden = ref.final_norm(resid[s])[0]
                mode_l = "fp32" if dtype == torch.float32 else "hf-bf16"
                with open(out_dir / f"{name}.logprobs.f16", "wb") as fh:
                    for r0 in range(0, len(ids) - 1, 256):
                        logits = ref.logits_rows(hidden[r0:min(r0 + 256, len(ids) - 1)], mode_l)          # [rows, V] fp32
                        lp = F.log_softmax(logits.float(), dim=-1).clamp_min(LOGPROB_CLAMP).to(torch.float16).cpu().numpy()
                        fh.write(lp.astype("<f2").tobytes())
                meta = {"T": len(ids), "V": int(logits.shape[-1]), "dtype": "float16", "rows": len(ids) - 1, "source": "reference",
                        "torch_dtype": str(dtype).replace("torch.", ""), "sha256_of_token_ids_json": token_ids_sha256(ids),
                        "n_layers": arch.n_layers, "clamp_min": LOGPROB_CLAMP, "tool": "tools/reference/qwen35moe/ref_run.py logprobs"}
                (out_dir / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
                print(f"[ref_run] {name}: {len(ids) - 1} rows x {meta['V']}", flush=True)
                del logits, lp, hidden
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if kv is not None:
            Path(args.record_kv).write_text(json.dumps(kv.merged(), indent=1), encoding="utf-8")
    for p in (state_path, resid_path):
        if p.exists():
            p.unlink()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--tokens", required=True)
        p.add_argument("--work", required=True, help="resume state (residuals, partial maxima)")
        p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
        p.add_argument("--device", default="cuda")
        p.add_argument("--segments", default=None, help="comma list of segment names (default all)")
        p.add_argument("--budget-seconds", type=float, default=840.0, help="0 = run to the end")
        p.add_argument("--fresh", action="store_true")

    p = sub.add_parser("logprobs")
    common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--record-kv", default=None, help="also write the per-layer K / V amax of these segments here")
    p.add_argument("--mix", default=None, help="a mix / oracle dir (weights_override.json): the HF stack with the trellis linears "
                                                "replaced by their reconstructions (weight-only diagnostic, lm_head stays bf16)")
    p.add_argument("--mix-layers", default=None, help="with --mix: reconstruct only these layers, 'A-B' or 'A'")
    p.add_argument("--mix-match", default=None, help="with --mix: reconstruct only tensors whose HF name matches this regex "
                                                      "(e.g. '(self_attn|linear_attn)' = class p, 'experts.gate_up_proj', 'experts.down_proj')")
    p.set_defaults(fn=lambda a: run(a, "logprobs"))
    p = sub.add_parser("kvcalib")
    common(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=lambda a: run(a, "kvcalib"))
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
