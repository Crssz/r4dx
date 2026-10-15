"""tools/reference/lfm2/tiny_e2e_golden.py  (docs/lfm2-plan.md L1-8 / L1-10: the tiny end-to-end contract)

The reference side of tests/model/test_lfm2_tiny_e2e.cpp: runs `Lfm2Reference` (tools/reference/lfm2/ref.py) over a
fixed token sequence on the tiny synthetic lfm2_moe checkpoint of tools/convert_ref/make_tiny_lfm2.py (hidden 512,
4 layers = conv+dense, conv+dense, attention+MoE, conv+MoE; 8 experts top-2; vocab 64) and dumps the logits of EVERY
row, from position 0, no cache:

    logits_truth    fp32 [N, V]  dtype=float32 reference (bf16 weights widened, fp32 activations, fp32 router / rope /
                                 experts / head): the TRUTH the r4dx bf16 engine is gated against
    logits_hf_bf16  fp32 [N, V]  stock HF numerics (bf16 everywhere, bf16 router, bf16 head), widened: the noise yardstick
    token_ids       int32 [N]

plus e2e_golden.json (shapes, the per-row KL(truth||hf_bf16) summary, provenance). N = 300 by default: the r4dx test
feeds it in pieces that cross the 256-row chunk boundary (conv state carry, attention over two chunks) and decodes the
tail one token at a time.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\convert_ref\\make_tiny_lfm2.py --out-dir <D>\\ckpt
    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\tiny_e2e_golden.py --model-dir <D>\\ckpt --out-dir <D>
    build\\win-hip\\src\\convert\\r4dx-convert.exe --input <D>\\ckpt --output <D>\\tiny.r4dx --layouts bf16 --lm-head bf16

CPU only, a few seconds, < 1 GB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import save_golden  # noqa: E402
from lfm2.ref import Lfm2Reference  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="make_tiny_lfm2.py output")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    cfg = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vocab = int(cfg["vocab_size"])
    g = torch.Generator().manual_seed(args.seed)
    ids = [int(v) for v in torch.randint(0, vocab, (args.tokens,), generator=g)]

    out = {}
    for name, dtype, mode in (("logits_truth", torch.float32, "fp32"), ("logits_hf_bf16", torch.bfloat16, "hf-bf16")):
        ref = Lfm2Reference(args.model_dir, torch.device("cpu"), dtype=dtype, attn="eager")
        h = ref.forward_hidden(ids)
        out[name] = ref.logits_rows(h, mode=mode).float().contiguous()
        del ref
    lt = torch.log_softmax(out["logits_truth"], -1)
    lb = torch.log_softmax(out["logits_hf_bf16"], -1)
    kl = (lt.exp() * (lt - lb)).sum(-1)
    top1 = float((out["logits_truth"].argmax(-1) == out["logits_hf_bf16"].argmax(-1)).float().mean())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {"token_ids": torch.tensor(ids, dtype=torch.int32), **out}
    save_golden(args.out_dir / "e2e_golden.safetensors", tensors, {"component": "lfm2_tiny_e2e"})
    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/lfm2/tiny_e2e_golden.py",
        "model_dir": str(args.model_dir), "tokens": args.tokens, "vocab": vocab, "seed": args.seed,
        "kl_truth_hf_bf16": {"mean": float(kl.mean()), "max": float(kl.max()), "median": float(kl.median())},
        "top1_truth_vs_hf_bf16": top1,
        "logits_truth_std": float(out["logits_truth"].std()),
        "torch_version": torch.__version__,
    }
    (args.out_dir / "e2e_golden.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[tiny_e2e_golden] wrote {args.out_dir} (N={args.tokens}, V={vocab}, KL(truth||hf-bf16) mean "
          f"{meta['kl_truth_hf_bf16']['mean']:.3e}, top1 {top1:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
