"""tools/reference/lfm2/tiny_trellis_golden.py  (docs/lfm2-plan.md L3-8: the tiny TRELLIS end-to-end contract)

The reference side of tests/model/test_lfm2_tiny_trellis.cpp. Takes the tiny lfm2_moe checkpoint of the tiny_e2e fixture
(tools/convert_ref/make_tiny_lfm2.py), the trellis oracle directory quantized from it (tools/reference/trellis_quant.py
quantize-model --arch lfm2_moe), and the tiny_e2e golden's token ids, and writes:

    ckpt_dequant/            the checkpoint with every oracle-covered linear replaced by its RECONSTRUCTION W_hat
                             (fp32, exactly what the trellis container decodes to), everything else copied as stored
    trellis_golden.safetensors
        token_ids            int32 [N]  (the tiny_e2e golden's, so the two goldens line up row for row)
        logits_truth_q       fp32 [N, V]  Lfm2Reference dtype=float32 on ckpt_dequant: the TRUTH the r4dx trellis
                                          engine is gated against (no quantization error in the gate)
        logits_hf_bf16_q     fp32 [N, V]  stock HF bf16 numerics on ckpt_dequant (W_hat rounded to bf16): the noise yardstick
    trellis_golden.json      provenance + KL(truth_q || hf_bf16_q), and the QUANTIZATION effect KL(truth || truth_q)
                             against the unquantized fp32 truth of e2e_golden.safetensors (informative / a loose gate)

Commands (CPU only, HIP_VISIBLE_DEVICES empty; a few minutes, < 2 GB; <D> = <models root>\\r4dx\\lfm2):

    python -I tools\\reference\\lfm2\\make_tiny_corpus.py --out <D>\\tiny_trellis\\samples.jsonl --vocab 64 --seqs 20 --len 1024
    python -I tools\\reference\\hessian_capture.py --arch lfm2_moe --device cpu --model-dir <D>\\tiny_e2e\\ckpt
        --gen-file <D>\\tiny_trellis\\samples.jsonl --out-dir <D>\\tiny_trellis\\hess --corpus-dir none --calib-txt none
        --wikitext none --code-seqs 0 --seq-len 1024 --max-short-expert-frac 1.0
    python -I tools\\reference\\trellis_quant.py quantize-model --arch lfm2_moe --device cpu --backend cpu --threads 4
        --model-dir <D>\\tiny_e2e\\ckpt --hessian-dir <D>\\tiny_trellis\\hess --out-dir <D>\\tiny_trellis\\oracle --K 4
        --hessian-basis matched
    build\\win-hip\\src\\convert\\r4dx-convert.exe --input <D>\\tiny_e2e\\ckpt --output <D>\\tiny_trellis\\tiny_trellis.r4dx
        --layouts bf16 --lm-head bf16 --trellis-from <D>\\tiny_trellis\\oracle
    python -I tools\\reference\\lfm2\\tiny_trellis_golden.py --model-dir <D>\\tiny_e2e\\ckpt
        --oracle-dir <D>\\tiny_trellis\\oracle --e2e-golden <D>\\tiny_e2e\\e2e_golden.safetensors --out-dir <D>\\tiny_trellis
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import trellis_quant as tq  # noqa: E402
from common import raw_safetensors_header, raw_safetensors_read, save_golden  # noqa: E402
from lfm2.ref import Lfm2Reference  # noqa: E402


def write_dequant_checkpoint(model_dir: Path, oracle_dir: Path, out: Path) -> dict:
    """ckpt_dequant: model_dir's tensors with every manifest tensor replaced by its fp32 reconstruction."""
    from safetensors.torch import save_file

    manifest = tq.load_manifest(oracle_dir)
    if not manifest.get("complete", False):
        raise SystemExit(f"{oracle_dir}: the oracle manifest is not complete")
    files = sorted(p for p in model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"{model_dir}: expected one model.safetensors (the tiny checkpoint), found {len(files)}")
    hdr = raw_safetensors_header(files[0])
    tensors: dict[str, torch.Tensor] = {}
    cb_cache: dict = {}
    replaced, rel = 0, {}
    for name in hdr[0]:
        t = raw_safetensors_read(files[0], name, hdr)
        if name in manifest["tensors"]:
            w_hat = tq.load_override_tensor(manifest, name, torch.device("cpu"), cb_cache).float()
            if tuple(w_hat.shape) != tuple(t.shape):
                raise SystemExit(f"{name}: reconstruction shape {tuple(w_hat.shape)} != checkpoint {tuple(t.shape)}")
            rel[name] = float((w_hat - t.float()).norm() / t.float().norm())
            t = w_hat
            replaced += 1
        tensors[name] = t.contiguous()
    if replaced != len(manifest["tensors"]):
        raise SystemExit(f"only {replaced} of {len(manifest['tensors'])} manifest tensors are in the checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"))
    shutil.copyfile(model_dir / "config.json", out / "config.json")
    return {"replaced": replaced, "rel_weight_err_mean": sum(rel.values()) / len(rel),
            "rel_weight_err_max": max(rel.values())}


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(p_logits.double(), -1)
    lq = torch.log_softmax(q_logits.double(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="the tiny bf16 checkpoint (tiny_e2e\\ckpt)")
    ap.add_argument("--oracle-dir", type=Path, required=True, help="trellis_quant.py quantize-model --out-dir")
    ap.add_argument("--e2e-golden", type=Path, required=True, help="tiny_e2e's e2e_golden.safetensors (token ids, truth)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    from safetensors.torch import load_file

    g = load_file(str(args.e2e_golden))
    ids = [int(v) for v in g["token_ids"].tolist()]
    truth = g["logits_truth"].float()

    dq = args.out_dir / "ckpt_dequant"
    wstats = write_dequant_checkpoint(args.model_dir, args.oracle_dir, dq)

    out = {}
    for name, dtype, mode in (("logits_truth_q", torch.float32, "fp32"), ("logits_hf_bf16_q", torch.bfloat16, "hf-bf16")):
        ref = Lfm2Reference(dq, torch.device("cpu"), dtype=dtype, attn="eager")
        h = ref.forward_hidden(ids)
        out[name] = ref.logits_rows(h, mode=mode).float().contiguous()
        del ref
    if out["logits_truth_q"].shape != truth.shape:
        raise SystemExit(f"shape mismatch: {tuple(out['logits_truth_q'].shape)} vs e2e truth {tuple(truth.shape)}")
    kl_noise = kl_rows(out["logits_truth_q"], out["logits_hf_bf16_q"])
    kl_quant = kl_rows(truth, out["logits_truth_q"])
    top1_quant = float((truth.argmax(-1) == out["logits_truth_q"].argmax(-1)).float().mean())

    tensors = {"token_ids": torch.tensor(ids, dtype=torch.int32), **out}
    save_golden(args.out_dir / "trellis_golden.safetensors", tensors, {"component": "lfm2_tiny_trellis"})
    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/lfm2/tiny_trellis_golden.py",
        "model_dir": str(args.model_dir), "oracle_dir": str(args.oracle_dir), "e2e_golden": str(args.e2e_golden),
        "tokens": len(ids), "vocab": int(truth.shape[1]), "weights": wstats,
        "kl_truth_q_hf_bf16_q": {"mean": float(kl_noise.mean()), "max": float(kl_noise.max())},
        "kl_truth_truth_q": {"mean": float(kl_quant.mean()), "max": float(kl_quant.max()),
                             "median": float(kl_quant.median())},
        "top1_truth_vs_truth_q": top1_quant,
        "torch_version": torch.__version__,
    }
    (args.out_dir / "trellis_golden.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[tiny_trellis_golden] wrote {args.out_dir}: {wstats['replaced']} linears dequantized "
          f"(rel err mean {wstats['rel_weight_err_mean']:.4f}), KL(truth_q||hf_bf16_q) mean {meta['kl_truth_q_hf_bf16_q']['mean']:.3e}, "
          f"KL(truth||truth_q) mean {meta['kl_truth_truth_q']['mean']:.3e} max {meta['kl_truth_truth_q']['max']:.3e}, "
          f"top1 {top1_quant:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
