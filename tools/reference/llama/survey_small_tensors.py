"""Survey of the small tensors of Nanbeige4.3-33B (docs/nanbeige-plan.md N0-5): norm weights per layer (the fp32-residual
and rotation evidence, 3.2 / 7) plus the embedding and lm_head row norms.  Loads one tensor at a time; the two 166912 x 7168
tables are reduced in row chunks.  CPU only.  Writes survey_small_tensors.json.

  D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\llama\\survey_small_tensors.py --model-dir <hf dir>
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


def stats(t: torch.Tensor) -> dict:
    x = t.float()
    a = x.abs()
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "abs_min": float(a.min()),
            "abs_max": float(a.max()), "std": float(x.std()), "n": int(x.numel())}


def row_norms(t: torch.Tensor, chunk: int = 8192) -> dict:
    n = torch.cat([t[i:i + chunk].float().norm(dim=1) for i in range(0, t.shape[0], chunk)])
    return {"rows": int(n.numel()), "min": float(n.min()), "max": float(n.max()), "mean": float(n.mean()),
            "argmax": int(n.argmax()), "argmin": int(n.argmin()), "abs_max_elem": float(t.float().abs().max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("survey_small_tensors.json"))
    a = ap.parse_args(argv)
    cfg = json.loads((a.model_dir / "config.json").read_text(encoding="utf-8"))
    idx = ShardIndex.load(a.model_dir)
    res = {"input_layernorm": {}, "post_attention_layernorm": {}}
    for i in range(cfg["num_hidden_layers"]):
        for k in ("input_layernorm", "post_attention_layernorm"):
            res[k][i] = stats(idx.get_tensor(f"model.layers.{i}.{k}.weight"))
    res["final_norm"] = stats(idx.get_tensor("model.norm.weight"))
    res["embed_tokens"] = row_norms(idx.get_tensor("model.embed_tokens.weight"))
    res["lm_head"] = row_norms(idx.get_tensor("lm_head.weight"))

    def agg(group, key):
        v = [s[key] for s in res[group].values()]
        return {"min": min(v), "max": max(v), "layer_of_max": max(range(len(v)), key=v.__getitem__)}
    res["aggregate"] = {f"{g}_{k}": agg(g, k) for g in ("input_layernorm", "post_attention_layernorm")
                        for k in ("abs_max", "abs_min", "mean")}
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("final_norm", "embed_tokens", "lm_head", "aggregate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
