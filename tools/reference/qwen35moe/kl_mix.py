"""`kl_mix.py`: KL of a trellis mix against the bf16 checkpoint through the HF reference (docs/qwen35moe-plan.md M4/M5).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\kl_mix.py --mix E:\\models\\r4dx\\q35moe-tiny\\mix
        [--model-dir E:\\models\\r4dx\\tiny-q35moe\\hf] [--tokens tokens.json | --golden-prompts] [--max-tokens 300] [--device cpu|cuda]

Runs `Qwen35MoeStreamingReference` twice over the same token sequences, once on the checkpoint and once with every trellis linear
of the mix REPLACED by its reconstruction from the stored bits (`trellis_quant.reconstruct`: words + sign vectors -> W^T, exactly what
the container's kernels decode, up to the f16 activations), router / shared expert / norms / lm_head untouched (bf16, as in the
container; the container's w4a16 lm_head is NOT modelled). Reports, per sequence and pooled, KL(p_bf16 || p_mix), top-1
agreement of the final logits (the plan's gate is on logits).

On the real model this is the weight-only quality estimate to run on the layers' mix before the runtime exists (a layer-streaming
forward of 250 GB of weights: hours per sequence, so use few short sequences); on the tiny random model it checks that the whole
chain -- Hessians, oracle, allocation, manifest, reconstruction -- gives a network that still computes (KL falls K3 -> K4).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import trellis_quant as tq  # noqa: E402
from qwen35moe.ref import CheckpointReader, Qwen35MoeStreamingReference  # noqa: E402


class MixReader(CheckpointReader):
    """CheckpointReader whose trellis linears come out of the mix: `get(name)` of a quantized `.weight`, and the two fused expert
    tensors (stacked from the per-expert reconstructions), are the reconstructed weights in bf16."""

    def __init__(self, model_dir, mix_dir, device):
        super().__init__(model_dir)
        man = json.loads((Path(mix_dir) / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
        if man.get("format") != tq.OVERRIDE_FORMAT:
            raise ValueError(f"{mix_dir}: not a weights-override manifest")
        self.mix_dir = Path(mix_dir)
        self.tensors: dict = man["tensors"]
        self.device = device
        self._cb = None
        self._files: dict = {}

    def _record_weight(self, name: str) -> torch.Tensor:
        """[N, K] bf16 reconstruction of record `name` (= '<base>.weight')."""
        from safetensors import safe_open

        r = self.tensors[name]
        path = Path(r["file"])
        path = path if path.is_absolute() else self.mix_dir / path        # an oracle dir keeps file names relative
        f = self._files.get(str(path))
        if f is None:
            f = self._files[str(path)] = safe_open(str(path), framework="pt", device="cpu")
        if self._cb is None:
            self._cb = torch.from_numpy(tq.codebook_np(r["codebook"]).copy()).to(self.device)
        words, suh, svh = (f.get_tensor(name + s).to(self.device) for s in (".trellis", ".suh", ".svh"))
        what = tq.reconstruct(words, suh, svh, r["K"], self._cb)       # [k, n] (in, out)
        return what.T.contiguous().to(torch.bfloat16).cpu()

    def get(self, name, rows=None):
        if rows is None:
            if name in self.tensors:
                return self._record_weight(name)
            if name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj"):
                prefix = name[:-len("gate_up_proj")] if name.endswith("gate_up_proj") else name[:-len("down_proj")]
                probe = prefix + "0.gate_proj.weight" if name.endswith("gate_up_proj") else prefix + "0.down_proj.weight"
                if probe in self.tensors:
                    e = 0
                    out = []
                    while prefix + f"{e}.down_proj.weight" in self.tensors:
                        if name.endswith("gate_up_proj"):
                            out.append(torch.cat([self._record_weight(prefix + f"{e}.gate_proj.weight"),
                                                  self._record_weight(prefix + f"{e}.up_proj.weight")], 0))
                        else:
                            out.append(self._record_weight(prefix + f"{e}.down_proj.weight"))
                        e += 1
                    return torch.stack(out)
        return super().get(name, rows)


def sequences(args, vocab: int, eos: int):
    if args.golden_prompts:
        from qwen35moe import tiny_golden

        for t, ids in tiny_golden.golden_prompts().items():
            if t >= 7:
                yield f"golden T{t}", ids[:args.max_tokens]
        return
    doc = json.loads(Path(args.tokens).read_text(encoding="utf-8"))
    for s in doc["segments"][:args.sequences]:
        yield s["name"], [int(t) for t in s["token_ids"]][:args.max_tokens]


def logprobs(ref: Qwen35MoeStreamingReference, ids):
    return F.log_softmax(ref.forward_logits(ids, mode="fp32" if ref.dtype == torch.float32 else "hf-bf16"), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mix", required=True, help="a mix or oracle directory (weights_override.json)")
    ap.add_argument("--model-dir", default=r"E:\models\r4dx\tiny-q35moe\hf")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--golden-prompts", action="store_true", help="the tiny model's four golden prompts")
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the numbers here as json")
    args = ap.parse_args()
    if not args.golden_prompts and not args.tokens:
        ap.error("--tokens or --golden-prompts")
    ref = Qwen35MoeStreamingReference(args.model_dir, args.device)
    quant = Qwen35MoeStreamingReference(args.model_dir, args.device)
    quant.reader = MixReader(args.model_dir, args.mix, ref.device)
    quant.reader.weight_dtype = quant.dtype
    rows = []
    kl_all, agree_all, n_all = 0.0, 0, 0
    for name, ids in sequences(args, ref.arch.vocab, ref.arch.eos_token_id):
        with torch.no_grad():
            lp, lq = logprobs(ref, ids), logprobs(quant, ids)
            kl = (lp.exp() * (lp - lq)).sum(-1)                       # [T] KL(p || q)
            agree = (lp.argmax(-1) == lq.argmax(-1)).float()
        rows.append({"name": name, "tokens": len(ids), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(agree.mean())})
        kl_all += float(kl.sum())
        agree_all += int(agree.sum())
        n_all += len(ids)
        print(f"[kl_mix] {name}: {len(ids)} tokens, KL mean {kl.mean():.5f} max {kl.max():.4f}, top-1 {agree.mean():.4f}", flush=True)
    res = {"mix": str(args.mix), "model_dir": str(args.model_dir), "tokens": n_all, "kl_mean": kl_all / max(n_all, 1),
           "top1": agree_all / max(n_all, 1), "sequences": rows,
           "note": "weight-only: trellis linears replaced by their reconstructions; lm_head / router / shared expert bf16"}
    print(f"[kl_mix] pooled: {n_all} tokens, mean KL {res['kl_mean']:.5f}, top-1 {res['top1']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
