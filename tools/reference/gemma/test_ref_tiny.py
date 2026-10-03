"""CPU-only tests of the Gemma reference package (M0-6 "tiny random-config CPU smoke"; no GPU, no
model weights). Run with the reference venv:

    $env:HIP_VISIBLE_DEVICES=''; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\gemma\\test_ref_tiny.py -q
    (or: ...\\python.exe tools\\reference\\gemma\\test_ref_tiny.py)

The scripts' own `--tiny` smokes (layer_golden_gemma / full_logits_gemma / gen_samples / greedy_smoke)
are exercised in test_scripts_tiny.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pytest  # noqa: E402

from gemma import arch as arch_mod  # noqa: E402
from gemma import common_gemma as cg  # noqa: E402


def test_device_rule_default_is_device_1(monkeypatch):
    monkeypatch.delenv("R4DX_REF_ALLOWED_DEVICES", raising=False)
    monkeypatch.delenv("R4DX_ALLOW_GPU0", raising=False)
    assert cg.allowed_devices() == {"1"}
    assert cg.check_visible_devices("1") == ["1"]
    for bad in (None, "", "0", "0,1", "2"):
        with pytest.raises(RuntimeError):
            cg.check_visible_devices(bad)


def test_device_rule_override(monkeypatch):
    monkeypatch.setenv("R4DX_REF_ALLOWED_DEVICES", "0,1")
    assert cg.check_visible_devices("0") == ["0"]
    assert cg.check_visible_devices("0,1") == ["0", "1"]
    assert cg.check_visible_devices(" 1 , 0 ") == ["1", "0"]
    with pytest.raises(RuntimeError):
        cg.check_visible_devices("2")
    with pytest.raises(RuntimeError):
        cg.check_visible_devices("0,2")
    monkeypatch.setenv("R4DX_REF_ALLOWED_DEVICES", "1")
    monkeypatch.setenv("R4DX_ALLOW_GPU0", "1")  # legacy switch still adds device 0
    assert cg.check_visible_devices("0") == ["0"]


def test_cpu_device_never_checks_env(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert cg.resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError):
        cg.resolve_device("cuda")


def _real_config_available() -> bool:
    return (cg.DEFAULT_MODEL_DIR / "config.json").is_file()


@pytest.mark.skipif(not _real_config_available(), reason="Huihui config.json not on this machine")
def test_arch_table_matches_real_config_and_header():
    a = cg.load_arch()
    assert (a.hidden, a.n_layers, a.n_heads, a.vocab, a.window) == (3840, 48, 16, 262144, 1024)
    assert a.full_layers() == [5, 11, 17, 23, 29, 35, 41, 47]
    assert len(a.sliding_layers()) == 40
    assert (a.head_dim_of(0), a.head_dim_of(5), a.kv_heads_of(0), a.kv_heads_of(5)) == (256, 512, 8, 1)
    assert a.has_v_proj(0) and not a.has_v_proj(5)
    assert a.rotary_dims(0) == 256 and a.rotary_dims(5) == 128
    assert (a.rope_theta(0), a.rope_theta(5)) == (1e4, 1e6)
    assert a.eos_ids == (1, 106, 50)  # generation_config.json (plan 9.4)
    header = HERE / "tensor_names.json"
    if header.is_file():
        tensors = cg.read_json(header)["tensors"]
        assert arch_mod.check_against_header(a, tensors) == []


def test_split_model_output():
    s = cg.split_model_output("<|channel>thought\nhmm<channel|>The answer.<turn|>")
    assert s == {"reasoning": "hmm", "content": "The answer.", "closed_thought": True}
    s = cg.split_model_output("Plain answer.<turn|>junk")
    assert s == {"reasoning": None, "content": "Plain answer.", "closed_thought": True}
    s = cg.split_model_output("<|channel>thought\nnever closed")
    assert s["reasoning"] == "never closed" and s["content"] == "" and not s["closed_thought"]


def test_additive_mask_window_semantics():
    import torch
    from gemma.ref import additive_mask

    m = additive_mask(8, 8, 0, 4, torch.device("cpu"), torch.float32)[0, 0] == 0
    assert m[7].tolist() == [False] * 4 + [True] * 4        # query 7 sees keys 4..7 (window 4 incl. self)
    assert m[2].tolist() == [True] * 3 + [False] * 5        # causal, window not yet binding
    full = additive_mask(1, 8, 7, None, torch.device("cpu"), torch.float32)[0, 0] == 0
    assert full[0].tolist() == [True] * 8                      # decode row sees everything


def test_tiny_reference_equals_hf_forward():
    from gemma.ref import selftest_tiny

    with tempfile.TemporaryDirectory() as td:
        res = selftest_tiny(td)
    assert res["max_abs_diff_resident"] == 0.0 and res["max_abs_diff_streaming"] == 0.0


def test_tiny_logprobs_rows_normalize():
    import io

    import numpy as np
    import torch
    from gemma.ref import GemmaReference, build_tiny_checkpoint

    with tempfile.TemporaryDirectory() as td:
        d = build_tiny_checkpoint(td)
        ref = GemmaReference(d, torch.device("cpu"))
        ids = [2, 10, 20, 30, 40, 50, 60, 70, 80]
        for mode in ("fp32", "hf-bf16"):
            buf = io.BytesIO()
            stats = ref.logprobs_to_file(ids, buf, mode=mode, chunk=50, row_block=3)
            arr = np.frombuffer(buf.getvalue(), dtype=np.float16).reshape(len(ids) - 1, ref.vocab_size)
            lse = np.log(np.exp(arr.astype(np.float64)).sum(-1))
            assert np.abs(lse).max() < 2e-3, (mode, np.abs(lse).max())
            assert abs(stats["nll"] - -np.mean([arr[i, ids[i + 1]] for i in range(len(ids) - 1)])) < 1e-3
            assert len(stats["last_top"]) == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
