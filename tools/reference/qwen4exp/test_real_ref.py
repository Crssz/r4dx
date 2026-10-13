"""CPU tests of the real-model reference driver (real_ref.py, noise_floor.py) on the tiny-q4exp checkpoint:

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe -m pytest tools\\reference\\qwen4exp\\test_real_ref.py -q

Needs `E:\\models\\r4dx\\tiny-q4exp` (tiny.py build; the golden test also its `golden/`); a missing fixture skips. No GPU.

* resumable layer-range checkpoints: a jobset run one layer per step (every step a fresh `run_step`, state and residual
  read back from disk, sha256 verified) gives logprobs files byte-identical to the single-step run, and to
  `Qwen4ExpStreamingReference.logprobs_to_file` of the plain streaming reference (both tie rules, T = 70 and 300);
  the residual checkpoint after layer k equals `run_layers` truncated to k layers.
* corrupt residual checkpoints and ids/rule changes are refused.
* the logprobs file round-trips `kl_report.py`: a self-comparison has KL 0 and top-1 100%; hf vs recent goes through
  `noise_floor.noise_floor` and gives finite, non-negative numbers; `noise_floor prefix` of a dump with itself is 0.
* the unbuffered random-access n-gram reader returns the bytes of the memmap reader (random, first and last rows).
* golden jobs reproduce the tiny HF goldens (tiny_golden.py) tensor for tensor; calib jobs give the amax of the golden's
  post-rope K and V.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import kl_report  # noqa: E402
from qwen4exp import noise_floor, real_ref, tiny, tiny_golden  # noqa: E402
from qwen4exp.ngram_ids import NGramHash  # noqa: E402
from qwen4exp.ngram_rows import SidecarNGramRows  # noqa: E402
from qwen4exp.ref import Qwen4ExpStreamingReference, token_ids_sha256  # noqa: E402

TINY = tiny.DEFAULT_OUT
HF_DIR = TINY / "hf"
SIDECAR = TINY / "sidecar.r4dx-ngram"
GOLDEN = TINY / "golden"
LENGTHS = (70, 300)

pytestmark = pytest.mark.skipif(not (TINY / "DONE").exists(), reason=f"{TINY} not built (tiny.py build)")


@pytest.fixture(scope="module")
def prompts():
    return tiny_golden.golden_prompts()


def tiny_jobset(root: Path, name: str, kind: str, jobs: list[dict], **kw) -> dict:
    js = real_ref.base_jobset(root, name, kind, HF_DIR, SIDECAR, **kw)
    js["attn_query_chunk"] = 4096
    js["jobs"] = jobs
    real_ref.atomic_write_json(real_ref.jobset_path(root, name), js)
    return js


def logprob_jobs(root: Path, prompts: dict, sub: str) -> list[dict]:
    jobs = []
    for t in LENGTHS:
        for rule in ("hf", "recent"):
            jobs.append({"name": f"{rule}.T{t}", "kind": "logprobs", "rule": rule, "segment": f"T{t}", "ids": prompts[t],
                         "out_dir": str(root / sub / rule)})
    return jobs


def run_until_done(root: Path, js: dict, max_layers: int | None, limit: int = 20) -> int:
    steps = 0
    while True:
        rc = real_ref.run_step(root, js, "cpu", 60.0, max_layers, 1, 0.0, verbose=False)
        steps += 1
        if rc == real_ref.EXIT_DONE:
            return steps
        assert rc == real_ref.EXIT_MORE and steps < limit


@pytest.fixture(scope="module")
def runs(tmp_path_factory, prompts):
    """The same jobs run in one step and one layer per step."""
    root = tmp_path_factory.mktemp("real_ref")
    out = {}
    for label, max_layers in (("single", None), ("split", 1)):
        js = tiny_jobset(root, label, "logprobs", logprob_jobs(root, prompts, label))
        out[label] = (js, run_until_done(root, js, max_layers))
    return root, out


def test_split_run_is_bit_identical_to_single_step(runs):
    root, out = runs
    assert out["single"][1] == 1
    assert out["split"][1] == 4, "tiny model has 4 layers: one layer per step"
    for t in LENGTHS:
        for rule in ("hf", "recent"):
            a = (root / "single" / rule / f"T{t}.logprobs.f16").read_bytes()
            b = (root / "split" / rule / f"T{t}.logprobs.f16").read_bytes()
            assert a == b, f"T={t} {rule}"
            ma = json.loads((root / "single" / rule / f"T{t}.meta.json").read_text())
            mb = json.loads((root / "split" / rule / f"T{t}.meta.json").read_text())
            assert ma["sha256_of_token_ids_json"] == mb["sha256_of_token_ids_json"]
            assert ma["n_layers"] == mb["n_layers"] == 4
            assert ma["mean_nll_next_token"] == mb["mean_nll_next_token"]


@pytest.mark.parametrize("rule", ("hf", "recent"))
def test_layer_major_equals_plain_streaming_reference(runs, prompts, rule, tmp_path):
    root, _ = runs
    ref = Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule=rule, ngram_sidecar=SIDECAR)
    for t in LENGTHS:
        p = tmp_path / f"{rule}.T{t}.f16"
        with open(p, "wb") as f:
            ref.logprobs_to_file(prompts[t], f, mode="fp32", chunk=32768, row_block=128)
        got = (root / "split" / rule / f"T{t}.logprobs.f16").read_bytes()
        assert got == p.read_bytes(), f"T={t} {rule}: layer-major run differs from Qwen4ExpStreamingReference.logprobs_to_file"


def test_residual_checkpoint_equals_truncated_run_layers(tmp_path, prompts):
    js = tiny_jobset(tmp_path, "ckpt", "logprobs", logprob_jobs(tmp_path, prompts, "o"))
    assert real_ref.run_step(tmp_path, js, "cpu", 60.0, 2, 1, 0.0, verbose=False) == real_ref.EXIT_MORE
    for job in js["jobs"]:
        st = json.loads((real_ref.work_dir(tmp_path, js, job) / "state.json").read_text())
        assert st["layers_done"] == 2 and st["resid_file"] == "resid_L2.bf16" and not st["finalized"]
        assert not (real_ref.work_dir(tmp_path, js, job) / "resid_L0.bf16").exists()
        got = real_ref.load_residual(tmp_path, js, job, st, torch.device("cpu"))
        ref = Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule=job["rule"], ngram_sidecar=SIDECAR, max_layers=2)
        want = ref.run_layers(job["ids"])
        assert torch.equal(got, want), job["name"]


def test_corrupt_checkpoint_and_changed_jobs_are_refused(tmp_path, prompts):
    js = tiny_jobset(tmp_path, "bad", "logprobs", logprob_jobs(tmp_path, prompts, "o")[:1])
    assert real_ref.run_step(tmp_path, js, "cpu", 60.0, 1, 1, 0.0, verbose=False) == real_ref.EXIT_MORE
    job = js["jobs"][0]
    d = real_ref.work_dir(tmp_path, js, job)
    f = d / "resid_L1.bf16"
    raw = bytearray(f.read_bytes())
    raw[1000] ^= 0x01
    f.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="corrupt checkpoint"):
        real_ref.run_step(tmp_path, js, "cpu", 60.0, 1, 1, 0.0, verbose=False)
    other = dict(job, ids=[(t + 1) % 4000 for t in job["ids"]])
    with pytest.raises(SystemExit, match="different ids/rule"):
        real_ref.load_state(tmp_path, js, other)
    with pytest.raises(SystemExit, match="different jobs"):
        real_ref.write_jobset(tmp_path, dict(js, jobs=[other]))


def test_logprobs_roundtrip_kl_report_self_comparison_is_zero(runs, prompts):
    root, _ = runs
    for rule in ("hf", "recent"):
        d = root / "split" / rule
        meta = json.loads((d / "T70.meta.json").read_text())
        T = len(prompts[70])
        assert (meta["T"], meta["rows"], meta["dtype"], meta["V"]) == (T, T - 1, "float16", 4096)
        assert meta["sha256_of_token_ids_json"] == token_ids_sha256(prompts[70]) == kl_report.token_ids_sha256(prompts[70])
        assert (d / "T70.logprobs.f16").stat().st_size == (T - 1) * 4096 * 2
        run = json.loads((d / "reference_run.json").read_text())
        assert run["n_layers"] == 4 and run["debug_max_layers"] is None and set(run["segments"]) == {"T70", "T300"}
        used = json.loads((d / "tokens_used.json").read_text())
        assert [s["name"] for s in used["segments"]] == ["T70", "T300"]
        res = kl_report.compare_segment("T70", prompts[70], d, d, 8, strict=True)
        assert res["mean_kl"] == 0.0 and res["max_kl"] == 0.0 and res["top1_agreement_pct"] == 100.0
        assert res["problems"] == [] and res["ref_identity"]["n_layers"] == 4
        rows = np.fromfile(d / "T70.logprobs.f16", dtype="<f2").reshape(T - 1, 4096).astype(np.float64)
        assert np.abs(np.exp(rows).sum(axis=1) - 1.0).max() < 5e-3, "fp16 rows are normalized log-probabilities"


def test_noise_floor_and_prefix_check(runs, prompts, tmp_path):
    root, _ = runs
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"segments": [{"name": f"T{t}", "token_ids": prompts[t]} for t in LENGTHS]}))
    nf = noise_floor.noise_floor(root / "split" / "hf", root / "split" / "recent", tokens)
    total = nf["overall"]
    assert total["rows"] == sum(t - 1 for t in LENGTHS) and np.isfinite(total["mean_kl"]) and total["mean_kl"] >= 0.0
    # T = 70 > the tiny model's dense-exact range (67 tokens): the rules may select differently, T = 70 rows before 67 are equal
    assert nf["results"][0]["_kl"][:60].max() == 0.0
    same = noise_floor.noise_floor(root / "split" / "recent", root / "split" / "recent", tokens)
    assert same["overall"]["mean_kl"] == 0.0
    res = noise_floor.prefix_check(root / "split" / "recent", "T300", root / "split" / "recent", "T300", prompts[300])
    assert res["mean_kl"] == 0.0 and res["top1_agreement_pct"] == 100.0


def test_unbuffered_ngram_reader_equals_memmap():
    nh = NGramHash.from_config_file(HF_DIR / "config.json")
    mm = SidecarNGramRows(SIDECAR, nh)
    un = real_ref.UnbufferedSidecarRows(SIDECAR, nh, threads=4)
    rng = np.random.default_rng(1)
    ids = rng.integers(0, nh.padded_rows, size=(37, 16), dtype=np.int64)
    ids[0, :4] = [0, 1, nh.padded_rows - 1, nh.padded_rows - 2]
    ids[1, :3] = [5, 5, 6]  # duplicates and neighbours
    assert torch.equal(un.rows(ids), mm.rows(ids))
    dense = np.arange(nh.padded_rows - 40, nh.padded_rows, dtype=np.int64).reshape(-1, 4)  # the file's last rows (EOF span)
    assert torch.equal(un.rows(dense), mm.rows(dense))
    with pytest.raises(IndexError):
        un.rows(np.array([nh.padded_rows]))


@pytest.mark.skipif(not (GOLDEN / "meta.json").exists(), reason="no tiny golden (tiny_golden.py)")
def test_golden_jobs_reproduce_tiny_goldens(tmp_path, prompts):
    from safetensors.torch import load_file

    jobs = [{"name": f"{rule}.T70", "kind": "golden", "rule": rule, "segment": "T70", "ids": prompts[70], "out_dir": str(tmp_path / "g")}
            for rule in ("hf", "recent")]
    js = tiny_jobset(tmp_path, "gold", "golden", jobs, golden_layers=[0, 1, 3])
    run_until_done(tmp_path, js, 2)
    for rule in ("hf", "recent"):
        got = load_file(str(tmp_path / "g" / f"T70.{rule}.safetensors"))
        want = tiny_golden.load_golden(GOLDEN, 70, rule)
        bad = []
        for name, t in got.items():
            if name in ("embedding", "input_ids", "mixer_out", "logits") or name.split(".")[0] in ("layer0", "layer1", "layer3"):
                if name not in want or not torch.equal(t, want[name]):
                    bad.append(name)
        assert not bad, f"{rule}: tensors differ from the tiny golden: {bad}"
        assert {"layer1.ple_ngram_ids", "layer3.index_mask", "layer3.attn_k", "logits", "mixer_out"} <= set(got)
        assert not any(k.startswith("layer2.") for k in got)


@pytest.mark.skipif(not (GOLDEN / "meta.json").exists(), reason="no tiny golden (tiny_golden.py)")
def test_calib_job_amax_equals_golden_kv(tmp_path, prompts):
    docs = [prompts[70], prompts[300]]
    jobs = [{"name": f"{n:03d}", "kind": "calib", "rule": "recent", "segment": f"doc{n}", "ids": ids, "out_dir": str(tmp_path)}
            for n, ids in enumerate(docs)]
    js = tiny_jobset(tmp_path, "cal", "calib", jobs, out_json=str(tmp_path / "kv_calib.json"), tokenizer_provenance={"mode": "tiny"},
                     corpus_meta=[{"name": j["segment"], "tokens": len(j["ids"])} for j in jobs])
    run_until_done(tmp_path, js, 2)
    calib = json.loads((tmp_path / "kv_calib.json").read_text())
    assert list(calib) == ["3"]
    k_amax = torch.zeros(2)
    v_amax = torch.zeros(2)
    for t in (70, 300):
        g = tiny_golden.load_golden(GOLDEN, t, "recent")
        k_amax = torch.maximum(k_amax, g["layer3.attn_k"].float().abs().amax(dim=(0, 2)))
        v_amax = torch.maximum(v_amax, g["layer3.attn_v"].float().abs().amax(dim=(0, 2)))
    assert calib["3"]["k_amax"] == k_amax.tolist()
    assert calib["3"]["v_amax"] == v_amax.tolist()
    assert calib["3"]["num_calibration_tokens"] == 370 and calib["3"]["tail_exact"] is True


# ---- fp32 truth (ref.py dtype=float32) -------------------------------------------------------------------------------------


def test_fp32_stack_runs_and_tracks_bf16(prompts):
    ref16 = Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="recent", ngram_sidecar=SIDECAR)
    ref32 = Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="recent", ngram_sidecar=SIDECAR, dtype=torch.float32)
    assert ref32.source_tag == "truth-fp32" and ref16.source_tag == "reference"
    assert ref32.segment_meta()["compute_dtype"] == "float32" and ref16.segment_meta()["compute_dtype"] == "bfloat16"
    ids = prompts[70]
    hidden = ref32.run_layers(ids)
    assert hidden.dtype == torch.float32 and torch.isfinite(hidden).all()
    lg32 = ref32.forward_logits(ids, mode="fp32")
    lg16 = ref16.forward_logits(ids, mode="fp32")
    assert lg32.dtype == torch.float32 and torch.isfinite(lg32).all()
    rel = (lg32 - lg16).abs().max().item() / lg32.std().item()
    assert 0.0 < rel < 0.5, f"fp32 vs bf16 logits differ by {rel:.3f} std (bf16 noise expected, not a different model)"
    with pytest.raises(ValueError, match="hf-bf16"):
        ref32.forward_logits(ids, mode="hf-bf16")
    with pytest.raises(ValueError, match="dtype"):
        Qwen4ExpStreamingReference(HF_DIR, "cpu", dtype=torch.float16)


REAL_DIR = real_ref.DEFAULT_MODEL_DIR


@pytest.mark.skipif(not (REAL_DIR / "model.safetensors.index.json").exists(), reason="real checkpoint not present")
def test_fp32_dequant_of_a_real_expert_linear():
    from qwen4exp.ref import CheckpointReader

    rd = CheckpointReader(REAL_DIR)
    base = "model.language_model.layers.5.mlp.experts.17.down_proj"
    w16 = rd.linear_weight(base, torch.device("cpu"))
    assert w16.dtype == torch.bfloat16
    rd.weight_dtype = torch.float32
    rd.exact_dequant = False
    wr = rd.linear_weight(base, torch.device("cpu"))
    rd.exact_dequant = True
    wx = rd.linear_weight(base, torch.device("cpu"))
    assert wr.dtype == wx.dtype == torch.float32
    assert torch.equal(wr, w16.float()), "rounded fp32 dequant == the bf16 dequant, widened"
    assert torch.equal(wx.to(torch.bfloat16), w16), "unrounded fp32 dequant rounds to the bf16 dequant"
    assert (wx != wr).any(), "the unrounded weight differs from the bf16 one somewhere"
    assert (wx - wr).abs().max().item() <= wx.abs().max().item() * 2.0 ** -8


def test_fp32_jobset_split_run_is_bit_identical_to_single_step(tmp_path, prompts):
    outs = {}
    for label, max_layers in (("single", None), ("split", 1)):
        jobs = [{"name": "recent.T70", "kind": "logprobs", "rule": "recent", "segment": "T70", "ids": prompts[70],
                 "out_dir": str(tmp_path / label)}]
        js = tiny_jobset(tmp_path, label, "logprobs", jobs, dtype="float32", fp32_exact_weights=True)
        run_until_done(tmp_path, js, max_layers)
        outs[label] = (tmp_path / label / "T70.logprobs.f16").read_bytes()
        meta = json.loads((tmp_path / label / "T70.meta.json").read_text())
        assert meta["source"] == "truth-fp32" and meta["torch_dtype"] == "float32" and meta["dtype"] == "float16"
    assert outs["single"] == outs["split"]
    ref = Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="recent", ngram_sidecar=SIDECAR, dtype=torch.float32)
    p = tmp_path / "plain.f16"
    with open(p, "wb") as f:
        ref.logprobs_to_file(prompts[70], f, mode="fp32", chunk=32768, row_block=128)
    assert p.read_bytes() == outs["single"]
    st = json.loads((real_ref.work_dir(tmp_path, json.loads(real_ref.jobset_path(tmp_path, "split").read_text()), {"name": "recent.T70"})
                     / "state.json").read_text())
    assert st["finalized"] and st["resid_file"] is None
