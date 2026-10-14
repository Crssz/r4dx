"""Dump the safetensors headers of an LFM2.5-24B-A2B (lfm2_moe) checkpoint: names, shapes, dtypes, shards.

Reads ONLY the 8-byte length + JSON header of every shard (and model.safetensors.index.json); no tensor
data, no torch.  CPU only.  (docs/lfm2-plan.md L0-3.)

  python -I dump_header.py --local E:\\models\\LiquidAI\\LFM2.5-24B-A2B --out tensor_names.json

The summary settles plan U4: expert key layout (per-expert w1/w2/w3 vs fused), F32 tensors, lm_head
absence, shard sizes, and checks every key against the architecture table (arch.py).
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lfm2.arch import Lfm2Arch, check_against_header  # noqa: E402


def shard_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n + 8, path.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True, type=Path, help="model dir with *.safetensors (+ index)")
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

    arch = Lfm2Arch.from_model_dir(a.local)
    problems = check_against_header(arch, tensors)
    names = list(tensors)
    expert_keys = [n for n in names if ".experts." in n]
    styles = Counter("per-expert" if re.search(r"\.experts\.\d+\.", n) else "fused" for n in expert_keys)
    f32 = {n: tensors[n] for n in names if tensors[n]["dtype"] != "BF16"}
    summary = {
        "source": f"local:{a.local}",
        "n_tensors": len(names),
        "n_shards": len(shards),
        "total_file_bytes": sum(s["file_size"] for s in shard_info.values()),
        "metadata": meta,
        "dtypes": dict(Counter(t["dtype"] for t in tensors.values())),
        "non_bf16_tensors": {"count": len(f32), "names_sample": list(f32)[:3],
                             "shapes": sorted({tuple(t["shape"]) for t in f32.values()}),
                             "all_expert_bias": all(n.endswith("feed_forward.expert_bias") for n in f32)},
        "expert_key_style": dict(styles),
        "expert_names_layer2": sorted({n.split("experts.", 1)[1].split(".", 1)[1] for n in expert_keys
                                       if ".layers.2." in n}),
        "expert_tensor_shapes_layer2": {k: sorted({tuple(tensors[n]["shape"]) for n in expert_keys
                                                   if ".layers.2." in n and n.endswith(k)})
                                        for k in ("w1.weight", "w2.weight", "w3.weight")},
        "lm_head_present": any(n.startswith("lm_head") for n in names),
        "top_level_names": sorted(n for n in names if ".layers." not in n),
        "embed_shape": tensors.get("model.embed_tokens.weight", {}).get("shape"),
        "conv_weight_shape": tensors.get("model.layers.0.conv.conv.weight", {}).get("shape"),
        "layer0_names": sorted(n.split(".layers.0.", 1)[1] for n in names if ".layers.0." in n),
        "layer2_non_expert_names": sorted(n.split(".layers.2.", 1)[1] for n in names
                                          if ".layers.2." in n and ".experts." not in n),
        "layer3_non_expert_names": sorted(n.split(".layers.3.", 1)[1] for n in names
                                          if ".layers.3." in n and ".experts." not in n),
        "shards": shard_info,
        "arch_table_problems": problems,
        "arch": arch.summary(),
    }
    a.out.write_text(json.dumps({"summary": summary, "tensors": tensors}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
