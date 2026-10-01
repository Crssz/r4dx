"""CPU-only smoke of the four GPU-run scripts (M0-8, M0-9, M0-10) on a tiny random-config checkpoint:
layer_golden_gemma.py, full_logits_gemma.py, gen_samples.py, greedy_smoke.py. Nothing here touches a GPU
or the real weights (HIP/CUDA are hidden before torch is imported).

    $env:HIP_VISIBLE_DEVICES=''; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\gemma\\test_scripts_tiny.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import full_logits_gemma as fl  # noqa: E402
import gen_samples as gs  # noqa: E402
import greedy_smoke as gr  # noqa: E402
import layer_golden_gemma as lg  # noqa: E402
from gemma.ref import GemmaReference, build_tiny_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dir():
    with tempfile.TemporaryDirectory() as td:
        yield build_tiny_checkpoint(Path(td) / "model", 0)


def test_layer_golden_tiny(tmp_path):
    from safetensors.torch import load_file

    assert lg.main(["--tiny", "--out-dir", str(tmp_path)]) == 0
    man = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["components"]) == {"layer_000_sliding", "layer_001_full", "sliding_ring_wrap",
                                      "embed_scale", "final_norm_softcap"}
    full = load_file(str(tmp_path / "layer_001_full.safetensors"))
    # full layer (k_eq_v): v_raw is the k_proj output; sliding has its own v_proj
    assert torch.equal(full["v_raw"], full["k_raw"])
    sl = load_file(str(tmp_path / "layer_000_sliding.safetensors"))
    assert not torch.equal(sl["v_raw"], sl["k_raw"])
    # pre-scalar * layer_scalar == layer output (the scalar is applied once, after the MLP residual)
    for t in (sl, full):
        want = (t["pre_scalar"] * t["layer_scalar"]).to(torch.bfloat16)
        assert torch.equal(want, t["layer_out"])
    # embed scale: bf16(sqrt(hidden)) multiply
    em = load_file(str(tmp_path / "embed_scale.safetensors"))
    assert torch.equal((em["embed_rows"] * em["scale_bf16"]).to(torch.bfloat16), em["embed_scaled"])
    # the ring component's window claim and the softcap probe
    assert man["components"]["sliding_ring_wrap"]["window_independence_max_abs_diff_last_row"] == 0.0
    fc = load_file(str(tmp_path / "final_norm_softcap.safetensors"))
    assert float(fc["cap_probe_out_f32"].abs().max()) < 30.0
    assert man["components"]["layer_000_sliding"]["fp32_floor"]["layer_out"]["rel_fro"] < 0.1


def test_standalone_layer_equals_model_loop(tiny_dir):
    """run_layer (the golden's layer) reproduces what GemmaReference's loop computes for layer i."""
    ref = GemmaReference(tiny_dir, torch.device("cpu"))
    ids = [2, 9, 18, 27, 36, 45, 54, 63, 72]
    rec = {}
    ref.forward_hidden(ids, record=lambda i, h: rec.__setitem__(i, h[0].clone()))
    x = ref.embed(ids).unsqueeze(0)
    for i in range(ref.n_layers):
        out = lg.run_layer(ref.build_layer(i), ref.rotary, ref.arch, i, x)["layer_out"]
        assert torch.equal(out, rec[i]), i
        x = rec[i].unsqueeze(0)


def test_full_logits_tiny_and_kl_report(tmp_path):
    ref_dir, noise_dir = tmp_path / "ref", tmp_path / "noise"
    assert fl.main(["--tiny", "--out-dir", str(ref_dir), "--noise-floor-out", str(noise_dir)]) == 0
    run = json.loads((ref_dir / "reference_run.json").read_text(encoding="utf-8"))
    assert run["mode"] == "A-resident" and run["logits_mode"] == "fp32" and set(run["segments"]) == {"tiny_0", "tiny_1"}
    for name in run["segments"]:
        meta = json.loads((ref_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
        arr = np.fromfile(ref_dir / f"{name}.logprobs.f16", dtype=np.float16).reshape(meta["rows"], meta["V"])
        assert meta["T"] == meta["rows"] + 1 and meta["bos_token_id"] == 2
        assert np.abs(np.log(np.exp(arr.astype(np.float64)).sum(-1))).max() < 2e-3
    kl = REF_DIR_KL(ref_dir, noise_dir)
    assert "tiny_0" in kl and "ALL" in kl


def REF_DIR_KL(ref_dir: Path, test_dir: Path) -> str:
    r = subprocess.run([sys.executable, str(HERE.parent / "kl_report.py"), "--ref-dir", str(ref_dir),
                        "--test-dir", str(test_dir), "--tokens", str(ref_dir / "tokens_used.json")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout


def test_full_logits_refuses_qwen_tokens(tmp_path):
    qwen = HERE.parent / "kl_corpus" / "tokens.json"
    with pytest.raises(SystemExit):
        fl.load_tokens_file(qwen)


def test_bos_rule():
    fl.check_segment("ok", [2, 5, 6], 128, True)
    for bad in ([5, 6], [2, 2, 6]):
        with pytest.raises(SystemExit):
            fl.check_segment("bad", bad, 128, True)
    with pytest.raises(SystemExit):
        fl.check_segment("oob", [2, 500], 128, True)


def test_gen_samples_tiny_resume_shards_merge(tmp_path):
    out = tmp_path / "samples.jsonl"
    assert gs.main(["--tiny", "--out", str(out)]) == 0
    recs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in recs] and len({r["id"] for r in recs}) == len(recs) == 4
    for r in recs:
        assert r["token_ids"][0] == 2 and r["calib_kind"] in ("raw_text", "chat_stream")
        if r["calib_kind"] == "chat_stream":
            last = r["turns"][-1]
            assert r["token_ids"] == r["token_ids"][: last["prompt_len"]] + last["gen_ids"]
        else:
            assert r["token_ids"][1:] == [t for t in r["turns"][-1]["gen_ids"]][: len(r["token_ids"]) - 1]
        assert r["seed"] == gs.sample_seed(r["id"])
        assert r["prompt_sha256"] and r["top_k"] == 8 and r["kl_token_overlap"] == 0
    first = out.read_bytes()
    assert gs.main(["--tiny", "--out", str(out)]) == 0  # resume: nothing new, file untouched
    assert out.read_bytes() == first
    man = json.loads((tmp_path / "gen_manifest.json").read_text(encoding="utf-8"))
    assert man["summary"]["samples"] == 4 and len(man["runs"]) == 2
    # shards partition the processing order; a merge restores the id set
    order = gs.gc.processing_order(gs.TINY_PROMPTS)
    s0, s1 = gs.shard_of(order, "0/2"), gs.shard_of(order, "1/2")
    assert sorted(e["id"] for e in s0 + s1) == sorted(e["id"] for e in order) and not set(
        e["id"] for e in s0) & set(e["id"] for e in s1)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ids0 = {e["id"] for e in s0}
    a.write_text("".join(json.dumps(r) + "\n" for r in recs if r["id"] in ids0), encoding="utf-8")
    b.write_text("".join(json.dumps(r) + "\n" for r in recs if r["id"] not in ids0), encoding="utf-8")
    merged = tmp_path / "merged.jsonl"
    assert gs.main(["--merge", str(a), str(b), "--out", str(merged)]) == 0
    assert [json.loads(x)["id"] for x in merged.read_text(encoding="utf-8").splitlines()] == sorted(r["id"] for r in recs)


def test_gen_samples_resume_truncates_torn_tail_and_rejects_changed_prompt(tmp_path):
    by_id = {e["id"]: e for e in gs.TINY_PROMPTS}
    e = gs.TINY_PROMPTS[0]
    good = json.dumps({"id": e["id"], "prompt_sha256": gs.gc.entry_sha256(e)})
    p = tmp_path / "s.jsonl"
    p.write_bytes((good + "\n" + '{"id": "code/001", "pro').encode())
    assert list(gs.read_existing(p, by_id)) == [e["id"]]
    assert p.read_bytes() == (good + "\n").encode()
    p.write_text(json.dumps({"id": e["id"], "prompt_sha256": "0" * 64}) + "\n")
    with pytest.raises(SystemExit):
        gs.read_existing(p, by_id)


def test_kl_token_overlap():
    runs = {tuple(range(10, 10 + gs.KL_TOKEN_WINDOW))}
    assert gs.kl_token_overlap(list(range(5, 60)), runs) == 1
    assert gs.kl_token_overlap(list(range(100, 160)), runs) == 0


def test_greedy_smoke_tiny_and_compare(tmp_path):
    out = tmp_path / "greedy.json"
    assert gr.main(["--tiny", "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert [p["name"] for p in doc["prompts"]] == [p["name"] for p in gr.PROMPTS]
    think = doc["prompts"][-1]
    assert think["enable_thinking"] is True
    for p in doc["prompts"]:
        assert p["prompt_ids"][0] == 2 and len(p["steps"]) == len(p["gen_ids"])
        assert all(s["gap"] >= 0 for s in p["steps"]) and [s["id"] for s in p["steps"]] == p["gen_ids"]
    test = {p["name"]: p["gen_ids"] for p in doc["prompts"]}
    t = tmp_path / "test.json"
    t.write_text(json.dumps(test), encoding="utf-8")
    assert gr.main(["--compare", str(out), str(t)]) == 0
    first = doc["prompts"][1]
    bad = dict(test)
    bad[first["name"]] = [(first["gen_ids"][0] + 1) % 128] + first["gen_ids"][1:]
    if first["steps"][0]["gap"] >= doc["near_tie_gap"]:     # a divergence at step 0, not at a near tie, fails
        t.write_text(json.dumps(bad), encoding="utf-8")
        assert gr.main(["--compare", str(out), str(t)]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
