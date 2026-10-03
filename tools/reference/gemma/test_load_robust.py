"""CPU-only tests of the crash-robust weight loading (g4-load-crash): the no-mmap single-file reader and the
commit-headroom guard. No GPU, no model weights.

    $env:HIP_VISIBLE_DEVICES='-1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\gemma\\test_load_robust.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
assert not torch.cuda.is_available()

from common import ShardIndex, get_tensors_grouped, raw_safetensors_read  # noqa: E402
from gemma import common_gemma as cg  # noqa: E402


@pytest.fixture()
def model_dir(tmp_path):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    tensors = {
        "a.bf16": torch.randn(7, 5, generator=g).to(torch.bfloat16),
        "b.f32": torch.randn(3, 4, generator=g),
        "c.u8": torch.randint(0, 255, (9,), generator=g, dtype=torch.uint8),
        "d.i32": torch.randint(-5, 5, (2, 3, 4), generator=g, dtype=torch.int32),
        "e.scalar": torch.tensor(3.5),
        "f.empty": torch.empty(0, 4),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    return tmp_path, tensors


def test_single_file_index_matches_safetensors_without_mmap(model_dir, monkeypatch):
    d, tensors = model_dir
    import safetensors

    def boom(*a, **k):
        raise AssertionError("safe_open (whole-file mmap) must not be used for a single-file checkpoint")

    monkeypatch.setattr(safetensors, "safe_open", boom)
    idx = ShardIndex.load(d)
    assert idx.single_file is not None and set(idx.weight_map) == set(tensors)
    for n, t in tensors.items():
        got = idx.get_tensor(n)
        assert got.dtype == t.dtype and got.shape == t.shape, n
        assert torch.equal(got, t), n
    assert torch.equal(idx.get_row_slice("a.bf16", 2, 5), tensors["a.bf16"][2:5])
    assert torch.equal(idx.get_row_slice("a.bf16", 5, 99), tensors["a.bf16"][5:])
    grouped = get_tensors_grouped(idx, ["b.f32", "c.u8"])
    assert torch.equal(grouped["b.f32"], tensors["b.f32"]) and torch.equal(grouped["c.u8"], tensors["c.u8"])


def test_returned_tensor_is_independent_of_file(model_dir):
    d, tensors = model_dir
    t = raw_safetensors_read(d / "model.safetensors", "a.bf16")
    t += 1  # writable, owns its buffer
    assert not torch.equal(t, tensors["a.bf16"])
    assert torch.equal(ShardIndex.load(d).get_tensor("a.bf16"), tensors["a.bf16"])


def test_repeated_open_loop(model_dir):
    d, tensors = model_dir
    idx = ShardIndex.load(d)
    for i in range(300):
        n = ["a.bf16", "b.f32", "d.i32"][i % 3]
        assert torch.equal(idx.get_tensor(n), tensors[n])


def test_wait_for_commit_waits_then_proceeds(monkeypatch):
    vals = iter([1 << 30, 2 << 30, 100 << 30])
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    assert cg.wait_for_commit(24 << 30, free_fn=lambda: next(vals), poll_s=1.0)
    assert len(sleeps) == 2


def test_wait_for_commit_times_out(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    t = iter(range(0, 10000, 400))
    monkeypatch.setattr("time.time", lambda: next(t))
    assert cg.wait_for_commit(24 << 30, free_fn=lambda: 0, timeout_s=900) is False


def test_guarded_from_pretrained_retries_once(model_dir, monkeypatch):
    d, _ = model_dir
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    class Fake:
        @classmethod
        def from_pretrained(cls, p, **kw):
            calls.append(p)
            if len(calls) == 1:
                raise OSError("transient")
            return "ok"

    assert cg.guarded_from_pretrained(Fake, d) == "ok" and len(calls) == 2


def test_free_commit_reports_on_windows():
    v = cg.free_commit_bytes()
    assert v is None or v > 0
