"""Survey of the SMALL tensors of LFM2.5-24B-A2B (docs/lfm2-plan.md L0-5): norm weights (decides the rotation
question, 7.1), router gate weights and expert_bias.  Loads one small tensor at a time (largest: the 64 x 2048
router, 256 KB); never an expert or a projection.  CPU only.  Writes survey_small_tensors.json.

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\survey_small_tensors.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import ShardIndex  # noqa: E402
from lfm2.arch import FINAL_NORM_NAME, Lfm2Arch  # noqa: E402
from lfm2.common_lfm2 import DEFAULT_MODEL_DIR  # noqa: E402


def stats(t: torch.Tensor) -> dict:
    x = t.float()
    a = x.abs()
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "abs_min": float(a.min()),
            "abs_max": float(a.max()), "abs_mean": float(a.mean()), "std": float(x.std()), "n": int(x.numel())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("survey_small_tensors.json"))
    a = ap.parse_args(argv)
    arch = Lfm2Arch.from_model_dir(a.model_dir)
    idx = ShardIndex.load(a.model_dir)
    res = {"operator_norm": {}, "ffn_norm": {}, "q_layernorm": {}, "k_layernorm": {}, "router_gate": {},
           "expert_bias": {}}
    res["final_norm"] = stats(idx.get_tensor(FINAL_NORM_NAME))
    for i in range(arch.n_layers):
        p = arch.layer_prefix(i)
        for k in ("operator_norm", "ffn_norm"):
            res[k][i] = stats(idx.get_tensor(p + k + ".weight"))
        if arch.is_attn(i):
            for k in ("q_layernorm", "k_layernorm"):
                res[k][i] = stats(idx.get_tensor(p + f"self_attn.{k}.weight"))
        if not arch.is_dense(i):
            g = idx.get_tensor(p + "feed_forward.gate.weight").float()
            rn = g.norm(dim=1)
            res["router_gate"][i] = {**stats(g), "row_norm_min": float(rn.min()), "row_norm_max": float(rn.max()),
                                     "row_norm_mean": float(rn.mean())}
            b = idx.get_tensor(p + "feed_forward.expert_bias")
            res["expert_bias"][i] = {**stats(b), "spread_max_minus_min": float(b.max() - b.min()),
                                     "argmax": int(b.argmax()), "argmin": int(b.argmin())}
    def agg(group, key):
        v = [s[key] for s in res[group].values()]
        return {"min": min(v), "max": max(v)}
    res["aggregate"] = {
        "operator_norm_abs_max": agg("operator_norm", "abs_max"), "operator_norm_abs_min": agg("operator_norm", "abs_min"),
        "ffn_norm_abs_max": agg("ffn_norm", "abs_max"), "ffn_norm_abs_min": agg("ffn_norm", "abs_min"),
        "q_layernorm_abs_max": agg("q_layernorm", "abs_max"), "k_layernorm_abs_max": agg("k_layernorm", "abs_max"),
        "expert_bias_abs_max": agg("expert_bias", "abs_max"), "expert_bias_spread": agg("expert_bias", "spread_max_minus_min"),
        "router_row_norm_mean": agg("router_gate", "row_norm_mean"),
    }
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"final_norm": res["final_norm"], "aggregate": res["aggregate"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
