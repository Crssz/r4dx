"""Dump the safetensors header of a Gemma 4 checkpoint (names, shapes, dtypes).

Reads only the header: from a local model.safetensors, or from the HF hub with an
HTTP range request (no weights downloaded). CPU only, no torch needed.

  python dump_header.py --local D:\\models\\Huihui-gemma-4-12B-it-abliterated --out tensor_names.json
  python dump_header.py --repo huihui-ai/Huihui-gemma-4-12B-it-abliterated --out tensor_names.json
"""
import argparse
import json
import re
import struct
import urllib.request
from collections import Counter
from pathlib import Path


def header_local(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n + 8, path.stat().st_size


def _range(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req) as r:
        return r.read(), r.headers.get("Content-Range")


def header_http(repo, rev="main"):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors"
    raw, cr = _range(url, 0, 7)
    n = struct.unpack("<Q", raw)[0]
    raw, cr = _range(url, 8, 8 + n - 1)
    total = int(cr.split("/")[-1]) if cr else None
    return json.loads(raw), n + 8, total


def layer_of(name):
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", help="model dir containing model.safetensors")
    g.add_argument("--repo", help="HF repo id (header via HTTP range request)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("tensor_names.json")))
    a = ap.parse_args()

    if a.local:
        hdr, data_off, size = header_local(Path(a.local) / "model.safetensors")
        source = f"local:{a.local}"
    else:
        hdr, data_off, size = header_http(a.repo)
        source = f"hf:{a.repo}"
    meta = hdr.pop("__metadata__", None)
    tensors = {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(hdr.items())}
    names = list(tensors)

    layers = sorted({layer_of(n) for n in names if ".language_model.layers." in n} - {None})
    per_layer = {}
    for i in layers:
        pre = f".layers.{i}."
        per_layer[i] = sorted(n.split(pre, 1)[1] for n in names if pre in n and ".language_model." in n)
    layer_types = {}
    for i, ts in per_layer.items():
        layer_types[i] = "full(no v_proj)" if "self_attn.v_proj.weight" not in ts else "sliding"
    prefixes = Counter(".".join(n.split(".")[:2]) for n in names)
    top = Counter(n.split(".")[0] for n in names)
    ls = [n for n in names if n.endswith("layer_scalar")]
    summary = {
        "source": source,
        "file_size": size,
        "header_bytes": data_off,
        "n_tensors": len(names),
        "metadata": meta,
        "layers_found": layers,
        "top_prefixes": dict(top),
        "two_level_prefixes": dict(prefixes),
        "lm_head_present": any(n.startswith("lm_head") for n in names),
        "layer_scalar_example": {n: tensors[n] for n in ls[:3]},
        "layer_scalar_count": len(ls),
        "layer_scalar_shapes": sorted({tuple(tensors[n]["shape"]) for n in ls}),
        "v_proj_absent_layers": [i for i, t in layer_types.items() if t.startswith("full")],
        "k_eq_v_layer_tensor_names": per_layer.get(layers[5]) if len(layers) > 5 else None,
        "sliding_layer_tensor_names": per_layer.get(layers[0]) if layers else None,
        "non_language_tensors": [n for n in names if ".language_model." not in n],
        "dtypes": dict(Counter(t["dtype"] for t in tensors.values())),
    }
    out = {"summary": summary, "tensors": tensors}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sliding_layer_tensor_names",)}, indent=1))


if __name__ == "__main__":
    main()
