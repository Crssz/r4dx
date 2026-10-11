"""CPU tests of the qwen4_exp streaming reference (arch.py, ref.py, ngram_rows.py) on the tiny-q4exp checkpoint:

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe -m pytest tools\\reference\\qwen4exp\\test_ref.py -q

Needs `E:\\models\\r4dx\\tiny-q4exp` (tiny.py build) and, for the golden test, its `golden/` (tiny_golden.py); a missing
fixture skips. No GPU.

* `hf` rule: logits of the layer-by-layer reference == plain `Qwen4ExpForConditionalGeneration` forward, bit for bit
  (T = 1, 7, 70, 300; HF with use_cache=False, sdpa attention, eager per-expert MoE loop).
* `recent` rule: every golden activation of tiny_golden.py (layer inputs and outputs, final mixer, logits) bit for bit.
* the sidecar memmap gives the same rows and logits as the in-RAM table.
* MoE routing counts per layer sum to T * top_k and equal the golden top-k ids' histogram.
* a block-chunked QSA layer (attn_query_chunk < T) agrees with the unchunked one to bf16 noise.
* linear-input hooks name only checkpoint linears and see every routed token once per expert linear.
* the arch table matches the tiny and (when present) the real checkpoint index.
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

from qwen4exp import ref as ref_mod  # noqa: E402
from qwen4exp import tiny, tiny_golden  # noqa: E402
from qwen4exp.arch import Qwen4ExpArch  # noqa: E402
from qwen4exp.common_qwen4exp import DEFAULT_MODEL_DIR  # noqa: E402
from qwen4exp.ngram_ids import NGramHash  # noqa: E402
from qwen4exp.ngram_rows import SidecarNGramRows, open_ngram_rows  # noqa: E402

TINY = tiny.DEFAULT_OUT
HF_DIR = TINY / "hf"
SIDECAR = TINY / "sidecar.r4dx-ngram"
GOLDEN = TINY / "golden"
LENGTHS = tiny_golden.PROMPT_LENGTHS

pytestmark = pytest.mark.skipif(not (TINY / "DONE").exists(), reason=f"{TINY} not built (tiny.py build)")


@pytest.fixture(scope="module")
def prompts():
    return tiny_golden.golden_prompts()


@pytest.fixture(scope="module")
def hf_model():
    return tiny.load_hf_model(HF_DIR)


@pytest.fixture(scope="module")
def ref_hf():
    return ref_mod.Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="hf")


@pytest.fixture(scope="module")
def ref_recent():
    return ref_mod.Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="recent")


@pytest.mark.parametrize("t", LENGTHS)
def test_hf_rule_equals_plain_hf_bitwise(t, prompts, hf_model, ref_hf):
    ids = prompts[t]
    with torch.no_grad():
        want = hf_model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    got = ref_hf.forward_logits(ids, mode="hf-bf16")
    diff = (got - want).abs().max().item()
    print(f"T={t}: streaming vs plain HF max |logit diff| {diff}")
    assert torch.equal(got, want), f"max |diff| {diff}"


@pytest.mark.skipif(not (GOLDEN / "meta.json").exists(), reason="no golden (tiny_golden.py)")
@pytest.mark.parametrize("t", LENGTHS)
def test_recent_rule_equals_golden_bitwise(t, prompts, ref_recent):
    gold = tiny_golden.load_golden(GOLDEN, t, "recent")
    cap = ref_mod.LayerCapture()
    got = ref_recent.forward_logits(prompts[t], mode="hf-bf16", hooks=cap)
    assert torch.equal(torch.tensor(prompts[t]), gold["input_ids"])
    for i in range(ref_recent.n_layers):
        for name, ours in (("residual_in", cap.inputs[i]), ("residual_out", cap.outputs[i])):
            want = gold[f"layer{i}.{name}"]
            assert torch.equal(ours, want), f"T={t} layer {i} {name}: max |diff| {(ours.float() - want.float()).abs().max()}"
    hidden = ref_recent.forward_hidden(prompts[t])
    assert torch.equal(hidden, gold["mixer_out"]), "mixer output"
    diff = (got - gold["logits"].float()).abs().max().item()
    print(f"T={t}: recent vs golden max |logit diff| {diff}")
    assert torch.equal(got, gold["logits"].float())


@pytest.mark.skipif(not SIDECAR.exists(), reason="no sidecar")
def test_sidecar_equals_table(prompts, ref_hf):
    nh = NGramHash.from_config_file(HF_DIR / "config.json")
    table = open_ngram_rows(HF_DIR, nh)  # in-RAM
    side = SidecarNGramRows(SIDECAR, nh)
    shards = open_ngram_rows(HF_DIR, nh, table_max_bytes=0)  # lazy shard reads
    ids, _ = nh.ngram_ids(np.array(prompts[300], dtype=np.int64))
    a, b, c = table.rows(ids), side.rows(ids), shards.rows(ids)
    assert torch.equal(a, b) and torch.equal(a, c)
    ref_side = ref_mod.Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="hf", ngram_sidecar=SIDECAR)
    for t in (7, 70):
        want = ref_hf.forward_logits(prompts[t])
        got = ref_side.forward_logits(prompts[t])
        assert torch.equal(got, want), f"T={t}"
    assert "sidecar" in ref_side.ngram.describe() and "table" in ref_hf.ngram.describe()


@pytest.mark.parametrize("t", (7, 70))
def test_routing_counts_sum(t, prompts, ref_hf):
    hooks = ref_mod.RoutingCounts()
    ref_hf.run_layers(prompts[t], hooks)
    top_k = ref_hf.arch.top_k
    assert sorted(hooks.counts) == list(range(ref_hf.n_layers))
    for i, counts in hooks.counts.items():
        assert counts.shape == (ref_hf.arch.n_experts,)
        assert int(counts.sum()) == t * top_k, f"layer {i}"
        if (GOLDEN / "meta.json").exists():
            ids = tiny_golden.load_golden(GOLDEN, t, "hf")[f"layer{i}.topk_ids"]
            assert torch.equal(counts, torch.bincount(ids.flatten(), minlength=ref_hf.arch.n_experts))


def test_linear_input_hooks(prompts, ref_hf):
    t = 7
    seen: dict[str, int] = {}
    layers: dict[str, int] = {}
    counts = ref_mod.RoutingCounts()

    def on_linear(layer, name, x):
        seen[name] = seen.get(name, 0) + x.shape[0]
        layers[name] = layer
        assert x.dim() == 2

    hooks = ref_mod.CallbackHooks(linear_input=on_linear, moe_routing=counts.moe_routing)
    ref_hf.run_layers(prompts[t], hooks)
    arch = ref_hf.arch
    for i in range(arch.n_layers):
        known = {*arch.quantized_linears(i), *(n[: -len(".weight")] for n in arch.dense_tensors(i) if n.endswith(".weight"))}
        mine = {n for n, l in layers.items() if l == i}
        assert mine <= known, f"layer {i}: unknown linear names {sorted(mine - known)[:3]}"
        p = arch.layer_prefix(i)
        want = ["attn_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.block_inject_weight", "mlp.gate",
                "mlp.shared_expert.down_proj", "mlp.shared_expert_gate"]
        want += ["linear_attn.in_proj_qkv", "linear_attn.out_proj", "linear_attn.in_proj_a"] if arch.layer_types[i] == "linear_attention" else [
            "self_attn.q_proj", "self_attn.o_proj", "self_attn.indexer.index_qk_proj"]
        if i in arch.ple_layers():
            want += ["ple.key_proj", "ple.value_proj"]
        for w in want:
            assert p + w in mine, f"layer {i}: no input for {w}"
        for e in range(arch.n_experts):
            n = int(counts.counts[i][e])
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert seen.get(arch.expert_base(i, e, proj), 0) == n, f"layer {i} expert {e} {proj}"


def test_chunked_qsa_matches_unchunked(prompts, ref_recent, monkeypatch):
    t = 300
    want = ref_recent.forward_logits(prompts[t])
    chunked = ref_mod.Qwen4ExpStreamingReference(HF_DIR, "cpu", tie_rule="recent", attn_query_chunk=64)
    calls = []
    real_select = ref_mod.qsa_indexer.select_tokens
    monkeypatch.setattr(ref_mod.qsa_indexer, "select_tokens", lambda *a, **k: calls.append(a[0].shape[1]) or real_select(*a, **k))
    got = chunked.forward_logits(prompts[t])
    assert calls == [64, 64, 64, 64, 44], calls  # the one QSA layer ran 5 query blocks
    diff = (got - want).abs()
    agree = float((got.argmax(-1) == want.argmax(-1)).float().mean())
    print(f"chunked(64) vs unchunked: max {diff.max():.4g} mean {diff.mean():.3g} top-1 agreement {agree:.4f}")
    assert float(diff.max()) < 0.25 and agree > 0.97


def test_arch_matches_tiny_index():
    arch = Qwen4ExpArch.from_config(HF_DIR / "config.json")
    wm = json.loads((HF_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    assert arch.check_against_weight_map(wm) == []
    assert arch.layer_types == ("linear_attention",) * 3 + ("full_attention",) and arch.ple_layers() == (1,)
    # a name the table does not know and a missing tensor are both reported
    bad = dict(wm)
    bad.pop("lm_head.weight")
    bad["model.language_model.layers.0.linear_attn.bogus.weight"] = "x"
    problems = arch.check_against_weight_map(bad)
    assert any("missing lm_head.weight" in p for p in problems) and any("unexpected" in p for p in problems)


@pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "model.safetensors.index.json").exists(), reason="real checkpoint not present")
def test_arch_matches_real_index():
    arch = Qwen4ExpArch.from_config(DEFAULT_MODEL_DIR / "config.json")
    wm = json.loads((DEFAULT_MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    assert arch.check_against_weight_map(wm) == []
    assert arch.n_layers == 48 and arch.n_experts == 512 and arch.ple_layers() == (1,)
    assert sum(t == "full_attention" for t in arch.layer_types) == 12


def test_full_logits_cli_tiny_feeds_kl_report(tmp_path, ref_recent):
    """The CLI's dump has kl_report.py's format, matches the reference's own log-softmax and is KL 0 against itself."""
    import subprocess

    from qwen4exp import full_logits_qwen4exp as cli

    assert cli.main(["--tiny", "--out-dir", str(tmp_path), "--tie-rule", "recent"]) == 0
    doc = json.loads((tmp_path / "tokens_used.json").read_text())
    for seg in doc["segments"]:
        ids, name = seg["token_ids"], seg["name"]
        meta = json.loads((tmp_path / f"{name}.meta.json").read_text())
        assert (meta["T"], meta["V"], meta["rows"], meta["dtype"]) == (len(ids), 4096, len(ids) - 1, "float16")
        assert meta["sha256_of_token_ids_json"] == ref_mod.token_ids_sha256(ids)
        assert meta["n_layers"] == 4 and meta["debug_max_layers"] is None
        rows = np.fromfile(tmp_path / f"{name}.logprobs.f16", dtype="<f2").reshape(len(ids) - 1, 4096)
        want = torch.log_softmax(ref_recent.forward_logits(ids, mode="fp32")[:-1], dim=-1).clamp_min(-1e4)
        err = np.abs(rows.astype(np.float32) - want.numpy()).max()
        print(f"{name}: max |file - recomputed logprob| {err:.4g}")
        assert err < 0.02
    report = subprocess.run([sys.executable, str(HERE.parent / "kl_report.py"), "--ref-dir", str(tmp_path), "--test-dir",
                             str(tmp_path), "--tokens", str(tmp_path / "tokens_used.json")], capture_output=True, text=True)
    assert report.returncode == 0, report.stdout + report.stderr
    assert "| **ALL** | 108 | **0.00000**" in report.stdout, report.stdout
