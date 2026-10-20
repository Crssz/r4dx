"""A layer prefix of the REAL Ornith-1.5-122B, in the tiny-q35moe layout, so the tiny tests and goldens run on real weights.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\real_prefix.py subset [--layers 4] [--out E:\\models\\r4dx\\q35moe-real4]
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\real_prefix.py golden [--out ...]   (bf16 HF + fp8 KV)

`subset` copies embed_tokens, layers [0, N), the final norm and lm_head of the real checkpoint (E:\\models\\r4dx\\Ornith-1.5-122B)
into `<out>/hf/model-00001-of-00001.safetensors` (+ config.json with num_hidden_layers N and layer_types[:N]), exactly the file
layout tools/reference/qwen35moe/tiny.py writes, so that:
    r4dx-convert --input <out>\\hf --output <out>\\tiny-bf16.r4dx --layouts bf16 --lm-head bf16
    R4DX_Q35MOE_TINY_DIR=<out> test_model_q35moe_tiny.exe
work unchanged (the test file names are the tiny ones). The prefix's logits are not a language model's (N of 48 layers): they only
have to agree with HF's own prefix, which is what the goldens record. `golden` runs HF on the CPU in bf16 with K / V rounded to
fp8 e4m3 (tiny_golden.py --kv-fp8) and needs about 2 x the prefix size in RAM (N = 4: ~25 GB of weights).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe import tiny, tiny_golden  # noqa: E402

REAL = Path(r"E:\models\r4dx\Ornith-1.5-122B")
DEFAULT_OUT = Path(r"E:\models\r4dx\q35moe-real4")


def subset(out: Path, layers: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((REAL / "config.json").read_text(encoding="utf-8"))
    cfg["text_config"]["num_hidden_layers"] = layers
    cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:layers]
    cfg["vision_config"]["depth"] = 1
    index = json.loads((REAL / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers."
    want = []
    for name in index:
        if name in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"):
            want.append(name)
        elif name.startswith(prefix) and int(name[len(prefix):].split(".", 1)[0]) < layers:
            want.append(name)
    hf = out / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for name in want:
        by_shard.setdefault(index[name], []).append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(REAL / shard), framework="pt") as f:
            for n in names:
                tensors[n] = f.get_tensor(n)
        print(f"[subset] {shard}: {len(names)} tensors", flush=True)
    (hf / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (hf / "generation_config.json").write_text((REAL / "generation_config.json").read_text(), encoding="utf-8")
    fname = "model-00001-of-00001.safetensors"
    save_file(tensors, str(hf / fname), metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    (hf / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": {k: fname for k in tensors}}), encoding="utf-8")
    (out / "DONE").write_text("ok\n")
    print(f"[subset] {len(tensors)} tensors, {total / 2**30:.2f} GiB -> {hf}")


def golden(out: Path) -> int:
    # tiny_golden.main reads --model/--out; the prefix is loaded by tiny.load_hf_model, which builds the module for real
    # (a 25 GB prefix needs ~50 GB of RAM at the peak: the state dict plus the freshly initialised parameters).
    argv = ["--kv-fp8", "--model", str(out / "hf"), "--out", str(out / "golden_fp8")]
    return tiny_golden.main(argv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--out", default=str(DEFAULT_OUT))
    g = sub.add_parser("golden")
    g.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    if args.cmd == "subset":
        subset(Path(args.out), args.layers)
        return 0
    return golden(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
