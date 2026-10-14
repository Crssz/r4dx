"""CPU tests of the LFM2 reference scripts on a tiny random-config checkpoint in the REAL on-disk format
(no real weights, no GPU, a few seconds):

  - ref.selftest_tiny: the streaming layer loop is bit-equal to HF `Lfm2MoeForCausalLM.from_pretrained`
  - full_logits_lfm2_cpu.py truth + bf16sdpa dumps: file sizes, padding-column mask, row = independent forward,
    batching (alone vs lockstep) exactness, resume, noise_term.json (+ router agreement), kl_report gate CLI
  - layer_golden_lfm2.py / router_golden.py / greedy_ref_lfm2.py: see their own sections below

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\test_scripts_tiny.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import kl_report as K  # noqa: E402
from lfm2 import ref as R  # noqa: E402
from lfm2.common_lfm2 import LOGPROB_CLAMP, token_ids_sha256  # noqa: E402

REAL_V = 150


def make_tokens(tmp: Path, seed: int = 5) -> tuple[Path, Path]:
    g = torch.Generator().manual_seed(seed)
    def seg(name, group, T, prompt):
        ids = [1] + [int(v) for v in torch.randint(4, REAL_V, (T - 1,), generator=g)]
        return {"name": name, "group": group, "token_ids": ids, "score_mask": [0] * prompt + [1] * (T - prompt)}
    chat = {"tokenizer_arch": "lfm2", "segments": [seg("c_en", "english", 40, 8), seg("c_py", "python", 33, 6),
                                                    seg("c_tool", "tool", 37, 9)]}
    raw = {"tokenizer_arch": "lfm2", "segments": [{"name": "r1", "token_ids": [1] + [
        int(v) for v in torch.randint(4, REAL_V, (45,), generator=g)]}]}
    pc, pr = tmp / "chat.json", tmp / "raw.json"
    pc.write_text(json.dumps(chat)), pr.write_text(json.dumps(raw))
    return pc, pr


def run_full(model: Path, variant: str, out: Path, tokens: list[Path], extra: list[str] | None = None):
    import full_logits_lfm2_cpu as F

    argv = ["--variant", variant, "--model-dir", str(model), "--out-dir", str(out), "--bos-id", "1",
            "--real-vocab", str(REAL_V), "--threads", "2", "--max-tokens", "0"]
    for t in tokens:
        argv += ["--tokens", str(t)]
    assert F.main(argv + (extra or [])) == 0


def read_rows(d: Path, name: str, rows: int, V: int) -> np.ndarray:
    return np.fromfile(d / f"{name}.logprobs.f16", dtype="<f2").reshape(rows, V).astype(np.float64)


def test_selftest_and_dumps():
    print(json.dumps(R.selftest_tiny(tempfile.mkdtemp()), indent=1))
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        model = R.build_tiny_checkpoint(tmp / "model", 0)
        pc, pr = make_tokens(tmp)
        V = R.tiny_config().vocab_size
        run_full(model, "truth", tmp / "truth", [pc, pr])
        run_full(model, "bf16sdpa", tmp / "bf16", [pc, pr], ["--truth-dir", str(tmp / "truth")])
        chat = json.loads(pc.read_text())
        for s in chat["segments"] + json.loads(pr.read_text())["segments"]:
            ids, n = s["token_ids"], s["name"]
            meta = json.loads((tmp / "truth" / f"{n}.meta.json").read_text())
            assert meta["sha256_of_token_ids_json"] == token_ids_sha256(ids) and meta["V"] == V
            lp = read_rows(tmp / "truth", n, len(ids) - 1, V)
            assert (lp[:, REAL_V:] == LOGPROB_CLAMP).all(), "padding columns must hold the clamp"
            assert np.abs(np.exp(lp[:, :REAL_V]).sum(1) - 1).max() < 5e-3
        # one row against an independent forward of the same reference class (fp32)
        ref = R.Lfm2Reference(model, torch.device("cpu"), dtype=torch.float32, attn="eager")
        ids = chat["segments"][1]["token_ids"]
        h = ref.forward_hidden(ids)
        want = R.masked_logprobs(ref.logits_rows(h, "fp32"), REAL_V)[:-1].numpy().astype(np.float64)
        got = read_rows(tmp / "truth", "c_py", len(ids) - 1, V)
        assert np.abs(got - want).max() < 2e-2, np.abs(got - want).max()   # fp16 dump rounding only
        # lockstep batching == the segment run alone
        run_full(model, "truth", tmp / "truth_alone", [pc], ["--segment", "c_py"])
        alone = read_rows(tmp / "truth_alone", "c_py", len(ids) - 1, V)
        assert np.abs(alone - got).max() < 2e-2
        # resume: a second run over a finished checkpoint reproduces the dump bit for bit
        before = (tmp / "truth" / "c_en.logprobs.f16").read_bytes()
        run_full(model, "truth", tmp / "truth", [pc, pr])
        assert (tmp / "truth" / "c_en.logprobs.f16").read_bytes() == before
        # noise term + router agreement
        nt = json.loads((tmp / "bf16" / "noise_term.json").read_text())
        names = [g["name"] for g in nt["groups"]]
        assert "chat-ALL" in names and "chat-tool" in names and "chat-code" in names, names
        a = next(g for g in nt["groups"] if g["name"] == "chat-ALL")
        assert a["noise"]["mean_kl"] >= 0 and set(a["implied"]) == {"0.0025", "0.005", "0.01"}
        ra = nt["router_id_agreement_truth_vs_bf16sdpa"]
        assert set(ra["per_layer"]) == {"2", "3"} and 0.0 <= ra["overall_same_expert_set_frac"] <= 1.0
        # kl_report gate: the epsilon is explicit; the bf16sdpa dump passes against itself as the test side
        res = K.gate_gemma_fp32(tmp / "truth", tmp / "bf16", tmp / "bf16", pc, slack=0.0, gate_name="lfm2-fp32")
        assert res["pass"], K.gate_gemma_fp32_markdown(res)
        print("noise", {g["name"]: round(g["noise"]["mean_kl"], 6) for g in nt["groups"]})
        print("router agreement", round(ra["overall_same_expert_set_frac"], 3))


def test_greedy_matches_hf_nocache():
    """greedy_ref_lfm2.py (cache-less, layer-streamed, lockstep) == a plain HF no-cache greedy loop, prompt by
    prompt (ids exactly, the recorded top-2 gaps positive); and --compare accepts / rejects as specified."""
    import greedy_ref_lfm2 as G
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM

    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "g.json"
        assert G.main(["--tiny", "--max-new-tokens", "10", "--out", str(out), "--threads", "2"]) == 0
        doc = json.loads(out.read_text())
        model_dir = R.build_tiny_checkpoint(Path(t) / "m", 0)
        hf = Lfm2MoeForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        for p in doc["prompts"]:
            ids = list(p["prompt_ids"])
            want = []
            for _ in range(len(p["gen_ids"])):
                with torch.no_grad():
                    lg = hf(input_ids=torch.tensor([ids + want]), use_cache=False).logits[0, -1].float()[:150]
                want.append(int(lg.argmax()))
            assert want == p["gen_ids"], (p["name"], want, p["gen_ids"])
            assert all(s["gap"] >= 0 for s in p["steps"]) and len(p["steps"]) == len(p["gen_ids"])
        # --compare: identical passes; a divergence at step 0 with a large gap fails; at a tiny gap passes
        test = {p["name"]: list(p["gen_ids"]) for p in doc["prompts"]}
        tp = Path(t) / "test.json"
        tp.write_text(json.dumps(test))
        assert G.compare(out, tp, 16) == 0
        bad = {k: ([(v[0] + 1) % 150] + v[1:]) for k, v in test.items()}
        tp.write_text(json.dumps(bad))
        for p in doc["prompts"]:
            p["steps"][0]["gap"] = 5.0
        out.write_text(json.dumps(doc))
        assert G.compare(out, tp, 16) == 1
        for p in doc["prompts"]:
            p["steps"][0]["gap"] = 0.01
        out.write_text(json.dumps(doc))
        assert G.compare(out, tp, 16) == 0


def test_tiny_flag_runs():
    import full_logits_lfm2_cpu as F

    assert F.main(["--variant", "truth", "--tiny", "--threads", "2"]) == 0
    assert F.main(["--variant", "bf16sdpa", "--tiny", "--threads", "2"]) == 0


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("PASS")
