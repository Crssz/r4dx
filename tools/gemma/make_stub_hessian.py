#!/usr/bin/env python3
"""Write a STUB Gemma Hessian directory for CPU dry runs of the pipeline after the real capture
(`hessian_capture.py --arch gemma4_unified`, a GPU job): `hessian.json` with the exact keys / rms_keys / taps
that capture would write for this checkpoint, and header-only `.hess` files (the 64-byte header with the true
K and a nominal row count, NO matrix body). Enough for `trellis_quant.py quantize-model --dry-run` (which reads
headers only) and for checking plumbing; refused by everything that reads a Hessian body (a real quantize-model,
r4dx-convert --ldlq). The manifest says `"stub": true` so nobody mistakes it for a capture.

    python tools\\gemma\\make_stub_hessian.py --out-dir <dir> [--model-dir <huihui>] [--rows N]

CPU only, no torch.cuda; uses hessian_capture's own tap planner so the stub cannot drift from the capture.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REF = Path(__file__).resolve().parents[1] / "reference"
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF / "gemma"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=450_000, help="nominal row count written to every header")
    args = ap.parse_args()

    import hessian_capture as hc
    from arch_table import resolve_arch, text_config_view
    from common import ShardIndex, sha256_file
    from gemma.common_gemma import DEFAULT_MODEL_DIR
    from imatrix_capture import enumerate_quantized_linears, tensor_shapes, verify_no_k_concat

    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    arch = resolve_arch("gemma4_unified", model_dir)
    tc = text_config_view(arch, model_dir)
    n_layers = int(tc.num_hidden_layers)
    specs = enumerate_quantized_linears(tc, False, False, arch=arch)
    index = ShardIndex.load(model_dir)
    shapes = tensor_shapes(index, sorted({n for s in specs for n in s.hf_names}))
    plans = hc.build_tap_plan(specs, verify_no_k_concat(specs, shapes), arch)
    rms_plans = hc.build_rms_plans({p.file: p.keys for p in plans if p.scope == "layer"}, n_layers,
                                   int(tc.hidden_size), arch)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files, keys = {}, {}
    for p in plans:
        files[p.file] = {"K": p.k, "rows": args.rows, "trace": 0.0}
        keys.update({k: p.file for k in p.keys})
    for rp in rms_plans:
        files[rp.file] = {"K": rp.k, "rows": args.rows, "trace": 0.0}
    for name, v in files.items():
        with open(args.out_dir / name, "wb") as f:
            f.write(hc.HESS_HEADER.pack(hc.HESS_MAGIC, v["K"], hc.HESS_FLAG_PACKED_UPPER, v["rows"], 0.0))
    hc.write_manifest(args.out_dir, files, keys,
                      {"stub": True, "note": "header-only .hess files: for --dry-run only, NOT a capture",
                       "model_dir": str(model_dir), "config_sha256": sha256_file(Path(model_dir) / "config.json"),
                       "arch": arch.name},
                      rms_keys=hc.rms_keys_of(rms_plans))
    print(f"[stub-hessian] {len(files)} header-only file(s), {len(keys)} key(s), "
          f"{len(hc.rms_keys_of(rms_plans))} rms_key(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
