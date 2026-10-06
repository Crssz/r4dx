"""tests/reference/test_fp8_dequant.py

CPU-only test of tools/reference/fp8_dequant.py on a synthetic tiny block-FP8 checkpoint (no real
checkpoint, no GPU). pytest collects the test_* functions; run as a plain script (like test_manifest.py)
it runs them all and prints "OK (<n> tests)":

    python tests\\reference\\test_fp8_dequant.py        (pytest also collects it, if installed)

Covered: ragged shapes (200x300, 128x128, 5x7), both scale names (compressed-tensors `weight_scale`,
DeepSeek `weight_scale_inv`), bit-exact against a straightforward float32 reference, non-fp8 tensors
byte-exact, scale and input_scale tensors dropped, the index and config rewritten, non-weight files
copied, resumability (valid shard skipped, damaged shard redone, no .tmp left), multiple threads, and
the refusals (no scale, both scales, wrong scale shape, orphan scale, 3-D fp8, index lists a shard that
is missing). The diff command's classification of fp8-vs-bf16 differences and --sample-rel-diff.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "reference" / "fp8_dequant.py"
sys.path.insert(0, str(TOOL.parent))
import fp8_dequant as F  # noqa: E402

SHAPES = {"a.weight": (200, 300), "b.weight": (128, 128), "c.weight": (5, 7), "d.weight": (257, 129)}


def _scale(shape, g, dtype):
    s = (torch.rand(math.ceil(shape[0] / 128), math.ceil(shape[1] / 128), generator=g) * 0.02 + 0.001)
    return s.to(dtype)


def make_ckpt(root: Path, naming="weight_scale", two_shards=True, scale_dtype=torch.float32):
    """returns {name: bf16 expected tensor or raw tensor} bookkeeping for the checks."""
    g = torch.Generator().manual_seed(1234)
    root.mkdir(parents=True)
    t1, t2, ref = {}, {}, {}
    for i, (n, shp) in enumerate(SHAPES.items()):
        w = (torch.randn(*shp, generator=g) * 100).clamp(-440, 440).to(torch.float8_e4m3fn)
        sc = _scale(shp, g, scale_dtype)
        sn = n + ("_scale_inv" if naming == "weight_scale_inv" else "_scale")
        (t1 if i % 2 == 0 or not two_shards else t2).update({n: w})
        (t2 if two_shards else t1)[sn] = sc          # scale in the OTHER shard on purpose
        (t2 if two_shards else t1)[n.replace("weight", "input_scale")] = torch.tensor(0.5)
        s = sc.float().repeat_interleave(128, 0)[:shp[0]].repeat_interleave(128, 1)[:, :shp[1]]
        ref[n] = (w.float() * s).to(torch.bfloat16)
    t1["norm.weight"] = torch.randn(300, generator=g).to(torch.bfloat16)
    t2["emb.weight"] = torch.randn(33, 17, generator=g).to(torch.bfloat16)
    t2["A_log"] = torch.randn(4, generator=g)                       # f32 passthrough
    ref["norm.weight"], ref["emb.weight"], ref["A_log"] = t1["norm.weight"], t2["emb.weight"], t2["A_log"]
    save_file(t1, str(root / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(t2, str(root / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    wm = {**{k: "model-00001-of-00002.safetensors" for k in t1}, **{k: "model-00002-of-00002.safetensors" for k in t2}}
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": wm}))
    (root / "config.json").write_text(json.dumps({"a": 1, "text_config": {"x": 2, "quantization_config": {"q": 1}},
                                                  "quantization_config": {"format": "float-quantized"}, "z": [1, 2]}))
    (root / "tokenizer.json").write_text('{"tok": 1}')
    (root / "chat_template.jinja").write_text("{{ x }}")
    (root / ".cache").mkdir()
    (root / ".cache" / "junk").write_text("x")
    return ref, wm


def run(*args, expect=0):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == expect, f"exit {r.returncode} != {expect}\n{r.stdout}\n{r.stderr}"
    return r


def read_all(d: Path):
    out = {}
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def bits(t):
    return t.contiguous().view(torch.uint8).flatten().tolist() if t.dim() else t.view(torch.uint8).tolist()


def check_output(src, out, ref):
    got = read_all(out)
    assert set(got) == set(ref), (sorted(set(got) ^ set(ref)))
    for k, v in ref.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape, k
        assert bits(got[k]) == bits(v), f"{k} not bit-exact"
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(ref)
    assert idx["metadata"]["total_size"] == sum(v.numel() * v.element_size() for v in ref.values())
    for k, shard in idx["weight_map"].items():
        with safe_open(str(out / shard), "pt") as f:
            assert k in f.keys()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg == {"a": 1, "text_config": {"x": 2}, "z": [1, 2]}, cfg
    assert (out / "tokenizer.json").read_text() == '{"tok": 1}'
    assert (out / "chat_template.jinja").read_text() == "{{ x }}"
    assert not (out / ".cache").exists()
    assert not list(out.glob("*.tmp")) and not list((out / F.SIDECAR_DIR).glob("*.tmp"))
    man = json.loads((out / F.MANIFEST).read_text())
    assert man["n_fp8"] == len(SHAPES) and all(not k.endswith(("_scale", "_scale_inv")) for k in man["tensors"])


def test_ragged_both_conventions():
    for naming in ("weight_scale", "weight_scale_inv"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ref, _ = make_ckpt(td / "src", naming)
            run("dequant", "--src", td / "src", "--out", td / "out", "--threads", 2, "--sha256")
            check_output(td / "src", td / "out", ref)
            man = json.loads((td / "out" / F.MANIFEST).read_text())
            e = man["tensors"]["a.weight"]
            assert e["src_dtype"] == "F8_E4M3" and e["scale"].startswith("a.weight_scale")
            assert man["tensors"]["norm.weight"]["scale"] is None
            raw = hashlib.sha256(bytes(bits(ref["a.weight"]))).hexdigest()
            assert e["sha256"] == raw


def test_scale_dtypes_and_single_shard():
    for sd in (torch.bfloat16, torch.float16):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ref, _ = make_ckpt(td / "src", "weight_scale", two_shards=True, scale_dtype=sd)
            run("dequant", "--src", td / "src", "--out", td / "out")
            check_output(td / "src", td / "out", ref)


def test_resume():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref, _ = make_ckpt(td / "src")
        out = td / "out"
        run("dequant", "--src", td / "src", "--out", out)
        s1 = out / "model-00001-of-00002.safetensors"
        m1 = s1.stat().st_mtime_ns
        r = run("dequant", "--src", td / "src", "--out", out)
        assert r.stdout.count("skipped (valid)") == 2 and s1.stat().st_mtime_ns == m1
        data = s1.read_bytes()
        s1.write_bytes(data[:-5])                       # truncated -> redone
        (out / "model-00002-of-00002.safetensors.tmp").write_bytes(b"junk")
        r = run("dequant", "--src", td / "src", "--out", out)
        assert r.stdout.count("skipped (valid)") == 1 and r.stdout.count("written") == 1
        check_output(td / "src", out, ref)
        assert not (out / "model-00002-of-00002.safetensors.tmp").exists()   # stale tmp swept


def rewrite(src: Path, shard: str, fn):
    p = src / shard
    with safe_open(str(p), "pt") as f:
        t = {k: f.get_tensor(k).clone() for k in f.keys()}
    fn(t)
    save_file(t, str(p), metadata={"format": "pt"})
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    idx["weight_map"] = {k: v for k, v in idx["weight_map"].items() if v != shard}
    for k in t:
        idx["weight_map"][k] = shard
    (src / "model.safetensors.index.json").write_text(json.dumps(idx))


def refused(td, mutate, needle):
    src = td / "src"
    make_ckpt(src)
    mutate(src)
    r = run("dequant", "--src", src, "--out", td / "out", expect=2)
    assert needle in r.stderr, r.stderr
    assert not list((td / "out").glob("*.safetensors")) if (td / "out").exists() else True


def test_refusals():
    s2 = "model-00002-of-00002.safetensors"
    s1 = "model-00001-of-00002.safetensors"
    cases = [
        (lambda s: rewrite(s, s2, lambda t: t.pop("a.weight_scale")), "has no scale"),
        (lambda s: rewrite(s, s2, lambda t: t.update({"a.weight_scale_inv": t["a.weight_scale"].clone()})), "both scale names"),
        (lambda s: rewrite(s, s2, lambda t: t.update({"a.weight_scale": torch.ones(3, 3)})), "scale shape"),
        (lambda s: rewrite(s, s2, lambda t: t.update({"a.weight_scale": torch.ones(1, 2)})), "scale shape"),
        (lambda s: rewrite(s, s2, lambda t: t.update({"zzz.weight_scale": torch.ones(1, 1)})), "no fp8 tensor"),
        (lambda s: rewrite(s, s1, lambda t: t.update({"e.weight": torch.zeros(2, 3, 4).to(torch.float8_e4m3fn), "e.weight_scale": torch.ones(1, 1)})), "2-D"),
        (lambda s: (s / s2).unlink(), "not present"),
    ]
    for mutate, needle in cases:
        with tempfile.TemporaryDirectory() as td:
            refused(Path(td), mutate, needle)


def test_diff():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref, _ = make_ckpt(td / "fp8")
        run("dequant", "--src", td / "fp8", "--out", td / "bf16")
        (td / "bf16" / "chat_template.jinja").write_text("{{ y }}")
        run("diff", "--a", td / "bf16", "--b", td / "fp8", "--out", td / "d.json", "--sample-rel-diff", 6)
        d = json.loads((td / "d.json").read_text())
        assert d["missing_in_b"] == [] and d["extra_in_b"] == []
        assert len(d["extra_in_b_scale_expected"]) == 2 * len(SHAPES)
        assert d["shape_mismatch"] == [] and d["dtype_mismatch"] == []
        assert len(d["dtype_mismatch_fp8_expected"]) == len(SHAPES)
        assert d["config"]["raw_equal"] is False and d["config"]["stripped_equal"] is True
        assert d["files"]["differ"] == ["chat_template.jinja", "config.json"]
        assert d["rel_diff"] and all(v == 0.0 for v in d["rel_diff"].values()), d["rel_diff"]
        # a changed tensor shows up in the rel diff
        run("diff", "--a", td / "bf16", "--b", td / "fp8", "--out", td / "d2.json", "--sample-rel-diff", 1,
            "--sample-names", "norm.weight")
        assert list(json.loads((td / "d2.json").read_text())["rel_diff"]) == ["norm.weight"]


def main() -> int:
    fails, n = [], 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                fails.append(name)
                print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"OK ({n} tests)" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
