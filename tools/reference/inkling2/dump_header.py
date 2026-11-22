"""Dump the safetensors headers of an Inkling-2-Mini (`inkling2_loop`) checkpoint: names, shapes, dtypes, shards.

Reads ONLY the 8-byte length + JSON header of every shard (and model.safetensors.index.json); no tensor data, no
torch.  CPU only.  (docs/inkling2-plan.md 3.2.)

  python -I dump_header.py --local E:\\models\\OpenUncensored\\Inkling-2-Mini-Uncensored --out tensor_names.json

The summary records, per layer, whether the experts are NVFP4 (U8 codes + F8_E4M3 scale per 16 + amax + outer scale
denominator) or MXFP8 (F8_E4M3 + E8M0 U8 scale per 32), the dtypes present, the tensors of the AttnRes / message queries,
and checks every key against the explicit table of `ref.py` (`audit_against_header`: every name and shape the config
implies exists and every non-`model.visual.*` tensor is consumed).  Exit status 1 on any problem.
"""

from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shard_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n + 8, path.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True, type=Path, help="model dir with *.safetensors (+ index) and config.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("tensor_names.json"))
    a = ap.parse_args(argv)

    shards = sorted(a.local.glob("model-*.safetensors")) or [a.local / "model.safetensors"]
    index = a.local / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.is_file() else None
    tensors, shard_info, meta = {}, {}, None
    for p in shards:
        hdr, data_off, size = shard_header(p)
        meta = hdr.pop("__metadata__", None) or meta
        n_bytes = 0
        for k, v in hdr.items():
            lo, hi = v["data_offsets"]
            n_bytes += hi - lo
            tensors[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": p.name}
        if data_off + n_bytes != size:
            raise SystemExit(f"{p.name}: header {data_off} + data {n_bytes} != file size {size} (truncated download?)")
        shard_info[p.name] = {"file_size": size, "header_bytes": data_off, "n_tensors": len(hdr)}
    tensors = dict(sorted(tensors.items()))
    if weight_map is not None:
        bad = [k for k, s in weight_map.items() if tensors.get(k, {}).get("shard") != s]
        extra = sorted(set(tensors) - set(weight_map))
        if bad or extra:
            raise SystemExit(f"index.json disagrees with shard headers: {bad[:3]} / {extra[:3]}")

    from inkling2.ref import audit_against_header
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    raw = json.loads((a.local / "config.json").read_text(encoding="utf-8"))
    tc = dict(raw["text_config"])
    tc.pop("model_type", None)
    cfg = Inkling2LoopTextConfig(**tc)
    problems = audit_against_header(cfg, {n: (t["dtype"], t["shape"]) for n, t in tensors.items()})
    names = list(tensors)
    fmt = {}
    for L in range(cfg.num_hidden_layers):
        w13 = tensors.get(f"model.llm.layers.{L}.mlp.experts.w13_weight")
        fmt[L] = None if w13 is None else ("NVFP4" if w13["dtype"] == "U8" else "MXFP8")
    layers_of = lambda f: [L for L, v in fmt.items() if v == f]  # noqa: E731
    non_layer = sorted(n for n in names if ".layers." not in n and not n.startswith("model.visual."))
    summary = {
        "source": f"local:{a.local}",
        "n_tensors": len(names),
        "n_shards": len(shards),
        "total_file_bytes": sum(s["file_size"] for s in shard_info.values()),
        "metadata": meta,
        "dtypes": dict(Counter(t["dtype"] for t in tensors.values())),
        "expert_format_by_layer": {"NVFP4": layers_of("NVFP4"), "MXFP8": layers_of("MXFP8")},
        "visual_tensors": sum(1 for n in names if n.startswith("model.visual.")),
        "non_layer_tensors": {n: {"dtype": tensors[n]["dtype"], "shape": tensors[n]["shape"]} for n in non_layer
                              if "attn_queries" not in n and "mlp_queries" not in n},
        "attn_queries_shapes": {n.rsplit(".", 1)[1]: tensors[n]["shape"] for n in non_layer if "attn_queries" in n},
        "layer5_full_names": sorted(n.split(".layers.5.", 1)[1] for n in names if ".layers.5." in n and ".experts." not in n),
        "layer6_loop_names": sorted(n.split(".layers.6.", 1)[1] for n in names if ".layers.6." in n and ".experts." not in n),
        "layer0_names": sorted(n.split(".layers.0.", 1)[1] for n in names if ".layers.0." in n and ".experts." not in n),
        "layer0_expert_tensors": {n.split("experts.", 1)[1]: {"dtype": t["dtype"], "shape": t["shape"]}
                                  for n, t in tensors.items() if ".layers.0.mlp.experts." in n},
        "layer20_expert_tensors": {n.split("experts.", 1)[1]: {"dtype": t["dtype"], "shape": t["shape"]}
                                   for n, t in tensors.items() if ".layers.20.mlp.experts." in n},
        "shards": shard_info,
        "table_audit_problems": problems,
    }
    a.out.write_text(json.dumps({"summary": summary, "tensors": tensors}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
