"""tests/reference/test_arch_gemma.py -- the architecture table of the quantization tools and their Gemma 4 port
(docs/gemma4-plan.md M1-25 / M1-28 prerequisites). CPU only, no GPU, no real checkpoint (the real 677-tensor
header is tools/reference/gemma/tensor_names.json); needs torch + a transformers that has gemma4_unified
(5.18 in D:\\venvs\\r4dx-gemma-ref), else it prints SKIP and exits 0.

    (a) arch_table.QWEN is exactly the tables hessian_capture / trellis_quant / imatrix_capture carried
        (their module-level constants are aliases of it), and GEMMA against the real header: every linear of
        every layer, no v_proj exactly on the 8 full layers, the converter's gemma_layout.cpp audit, the tap
        plan (L{i}.in / out / mlp_in / mlp_mid + lm_head, K from the header), the rms plans (input_layernorm and
        pre_feedforward_layernorm).
    (b) the token-id corpus (gen_samples.py's JSONL): rejected skipped, a usable sample sharing a token run
        with the KL set refused, truncation to --seq-len, --gen-max-seqs.
    (c) hessian_capture.py --arch gemma4_unified --dry-run and imatrix_capture.py --arch gemma4_unified
        --dry-run on a tiny random checkpoint (the CLIs, as subprocesses, with no GPU env).
    (d) a real CPU capture through the tool's own run_capture (GemmaCaptureRef over GemmaReference): every file
        equals an independent hook-based second moment of the same HF layer -- q_proj / o_proj / gate_proj /
        down_proj inputs, gate_proj's input being pre_feedforward_layernorm's output (not
        post_attention_layernorm's), lm_head's the post-final-norm hidden; the full layer has no v_proj tap;
        the weightless rms taps equal E[rms(x)^T rms(x)] of the norms' inputs with scale w (offset 0) and pass
        evaluate_rms_gates; the shared-input gate sees k / v / up on the sliding layer and k / up on the full one;
        the gates pass; structural_zero_channels reads the single-file checkpoint with offset 0.
    (e) rotation_oracle: the file reader and fingerprint, fold / Hessian transform invariants
        (tr(W H W^T) kept, the division path equals the rms path, Q orthogonal), and -- when r4dx-convert is built
        ($env:R4DX_CONVERT_EXE or the worktree build) -- the fold against the converter's own folded bf16 bytes and
        its --rotation-out fingerprint.
    (f) trellis_quant.py on the tiny Gemma checkpoint: model_linears (single file, no v on the full layer),
        quantize-model --dry-run, and -- with r4dx-convert -- a rotated CPU oracle run whose manifest r4dx-convert
        imports and verifies (the whole chain: oracle fold + Hessian transform -> manifest rotation -> converter
        fingerprint check -> reconstruction against fold(W)).

Plain script, no pytest:  <venv>\\Scripts\\python.exe tests\\reference\\test_arch_gemma.py
Exits 0 and prints "OK (<n> checks)"; 1 and every failed check otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "tools" / "reference"
sys.path.insert(0, str(REF))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.pop("HIP_VISIBLE_DEVICES", None)

try:
    import numpy as np
    import torch
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration  # noqa: F401
except Exception as e:  # noqa: BLE001
    print(f"SKIP: {type(e).__name__}: {e} (needs torch + transformers with gemma4_unified)")
    raise SystemExit(0)

import arch_table as at  # noqa: E402
import hessian_capture as hc  # noqa: E402
import imatrix_capture as ic  # noqa: E402
import rotation_oracle as ro  # noqa: E402
import trellis_quant as tq  # noqa: E402
from gemma import capture as gcap  # noqa: E402
from gemma.ref import GemmaReference, build_tiny_checkpoint  # noqa: E402

CHECKS = 0
FAILS: list[str] = []
PY = sys.executable


def check(cond, what: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(what)
        print(f"FAIL: {what}")


def run_cli(script: str, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.update(env_extra or {})
    return subprocess.run([PY, str(REF / script), *args], capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=900)


def convert_exe() -> Path | None:
    cands = [os.environ.get("R4DX_CONVERT_EXE", "")] + [str(p) for p in
                                                       sorted(REPO.glob("build/*/src/convert/r4dx-convert.exe"))]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


# --------------------------------------------------------------------------------------------
# (a) tables
# --------------------------------------------------------------------------------------------

def header_types(tensors: dict) -> list[str]:
    n = 1 + max(int(k.split(".")[3]) for k in tensors if k.startswith(at.LAYER_PREFIX))
    return [at.SLIDING if f"{at.LAYER_PREFIX}{i}.self_attn.v_proj.weight" in tensors else at.FULL for i in range(n)]


def test_tables() -> None:
    q = at.QWEN
    check(hc.NORM_BEFORE == {"linear_attn.in_proj_qkv": "input_layernorm", "self_attn.q_proj": "input_layernorm",
                             "mlp.gate_proj": "post_attention_layernorm"}, "QWEN norm_before is the old NORM_BEFORE")
    check(hc.SHARED_WITH == {"linear_attn.in_proj_z": "linear_attn.in_proj_qkv", "self_attn.k_proj": "self_attn.q_proj",
                             "self_attn.v_proj": "self_attn.q_proj"}, "QWEN shared_with is the old SHARED_WITH")
    check(hc.TAP_OF == {"linear_attn.in_proj_qkv": "in", "self_attn.q_proj": "in", "linear_attn.out_proj": "out",
                        "self_attn.o_proj": "out", "mlp.gate_proj": "mlp_in", "mlp.down_proj": "mlp_mid"},
          "QWEN tap_of is the old TAP_OF")
    check(list(hc.SHARED_GATE) == ["linear_attention", "full_attention"] and
          hc.SHARED_GATE["full_attention"][0] == ("self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"]),
          "QWEN shared_gate is the old SHARED_GATE (layer type order included)")
    check(hc.RMS_NORMS == (("input_layernorm", "in"), ("post_attention_layernorm", "mlp_in")) and
          hc.RMS_KEY_SUFFIXES["in"] == ("gdn.in_proj_qkv", "gdn.in_proj_z", "attn.qg", "attn.k", "attn.v"),
          "QWEN rms tables are the old RMS_NORMS / RMS_KEY_SUFFIXES")
    check(tq.LINEARS["self_attn.q_proj"] == ("attn.qg", "in") and tq.QGROUPS["linear_attn.in_proj_z"] == "linear_attn.qkvz" and
          tq.module_list("full_attention") == tq.ATTN_MODULES + tq.MLP_MODULES and
          tq.module_list("linear_attention") == tq.GDN_MODULES + tq.MLP_MODULES,
          "trellis_quant's LINEARS / QGROUPS / module_list are the Qwen entry")
    check(q.norm_offset == 1.0 and at.GEMMA.norm_offset == 0.0, "norm offsets: Qwen 1 + w, Gemma w")
    check(at.get_arch(None) is at.QWEN and at.get_arch("gemma4_unified") is at.GEMMA and
          at.get_arch("gemma4_unified_text") is at.GEMMA, "get_arch")

    hdr = json.loads((REF / "gemma" / "tensor_names.json").read_text(encoding="utf-8"))["tensors"]
    types = header_types(hdr)
    check(len(types) == 48 and [i for i, t in enumerate(types) if t == at.FULL] == [5, 11, 17, 23, 29, 35, 41, 47],
          "the real header: 48 layers, full attention (no v_proj) at 5, 11, ..., 47")
    g = at.GEMMA
    bad = []
    for i, lt in enumerate(types):
        mods = g.module_list(lt)
        for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            if m not in mods or g.hf_name(i, m) not in hdr:
                bad.append((i, m))
        if ("self_attn.v_proj" in mods) != (g.hf_name(i, "self_attn.v_proj") in hdr):
            bad.append((i, "v_proj"))
        linear_weights = {k for k in hdr if k.startswith(f"{at.LAYER_PREFIX}{i}.") and k.endswith("_proj.weight")}
        if linear_weights != {g.hf_name(i, m) for m in mods}:
            bad.append((i, "linear set"))
    check(not bad, f"GEMMA.module_list covers exactly the header's linears of every layer: {bad[:4]}")

    specs = ic.enumerate_quantized_linears(SimpleNamespace(layer_types=types), False, False, arch=g)
    check(len(specs) == 40 * 6 + 8 * 5 + 1, f"enumerate (gemma): 281 specs ({len(specs)})")
    check(specs[-1].key == "lm_head" and specs[-1].hf_names == [at.GEMMA.lm_head_name] and specs[-1].tap == "final_norm_out",
          "the tied lm_head is the embedding table fed by the post-final-norm hidden")
    check(not any(s.key.endswith(".attn.v") and s.key.startswith("text.layers.5.") for s in specs) and
          any(s.key == "text.layers.4.attn.v" for s in specs), "no attn.v spec on a full layer, one on a sliding layer")
    shapes = {n: list(hdr[n]["shape"]) for s in specs for n in s.hf_names}
    expect_k = ic.verify_no_k_concat(specs, shapes)
    audit = ic.audit_converter_source(g.converter_source(), arch=g.converter_arch)
    check(audit["ok"], f"the converter audit of gemma_layout.cpp passes: {audit['missing'] + audit['unexpected']}")
    plans = hc.build_tap_plan(specs, expect_k, g)
    by = {p.file: p for p in plans}
    check(len(plans) == 48 * 4 + 1, f"tap plan: 48 x (in, out, mlp_in, mlp_mid) + lm_head ({len(plans)} files)")
    check(by["L00.in.hess"].keys == ["text.layers.0.attn.q", "text.layers.0.attn.k", "text.layers.0.attn.v"] and
          by["L05.in.hess"].keys == ["text.layers.5.attn.q", "text.layers.5.attn.k"] and by["L05.in.hess"].k == 3840,
          "L00.in serves q/k/v, L05.in (full) only q/k; K 3840")
    check(by["L00.out.hess"].k == 4096 and by["L05.out.hess"].k == 8192 and by["L00.mlp_mid.hess"].k == 15360 and
          by["L00.mlp_in.hess"].keys == ["text.layers.0.mlp.gate_up"] and by["L00.mlp_in.hess"].module == "mlp.gate_proj" and
          by["lm_head.hess"].k == 3840 and by["lm_head.hess"].scope == "lm_head",
          "o_proj K 4096 / 8192, down K 15360, gate_up on gate_proj's input, lm_head 3840")
    check(g.norm_before["mlp.gate_proj"] == "pre_feedforward_layernorm" and hc.norm_weight_name("layer", 7, "mlp.gate_proj", g) ==
          f"{at.LAYER_PREFIX}7.pre_feedforward_layernorm.weight" and hc.norm_weight_name("layer", 7, "mlp.gate_proj") ==
          f"{at.LAYER_PREFIX}7.post_attention_layernorm.weight", "mlp_in's norm: pre_feedforward_layernorm (Gemma) vs post_attention (Qwen)")
    post = {p.file: p.keys for p in plans if p.scope == "layer"}
    rms = hc.build_rms_plans(post, 48, 3840, g)
    check(len(rms) == 96 and rms[0].norm == "input_layernorm" and rms[1].norm == "pre_feedforward_layernorm" and
          rms[1].file == "L00.mlp_in.rms.hess" and rms[1].keys == ["text.layers.0.mlp.gate_up"],
          "rms plans: input_layernorm (in) and pre_feedforward_layernorm (mlp_in) per layer")
    try:
        hc.build_rms_plans({"L00.in.hess": ["text.layers.0.gdn.in_proj_qkv"]}, 1, 3840, g)
        check(False, "a gdn key under a Gemma rms tap is refused")
    except SystemExit:
        check(True, "a gdn key under a Gemma rms tap is refused")


# --------------------------------------------------------------------------------------------
# tiny checkpoint + token corpus
# --------------------------------------------------------------------------------------------

def write_samples(path: Path, n: int = 12, vocab: int = 128, seed: int = 5, length: int = 50, extra=()) -> None:
    rng = np.random.default_rng(seed)
    cats = list(gcap.GEN_CATEGORIES)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            cat = cats[i % len(cats)]
            ids = [2] + [int(x) for x in rng.integers(4, vocab, size=length + (i % 3) * 5)]
            f.write(json.dumps({"id": f"{cat}/{i // len(cats):03d}", "category": cat, "rejected": False,
                                "token_ids": ids, "kl_token_overlap": 0}) + "\n")
        for e in extra:
            f.write(json.dumps(e) + "\n")


def test_corpus(tmp: Path) -> None:
    p = tmp / "samples.jsonl"
    write_samples(p, extra=[{"id": "chat/999", "category": "chat", "rejected": True, "token_ids": [2, 5, 6]}])
    seqs, sources = gcap.build_tokens_corpus(p, 48)
    check(len(seqs) == 12 and all(len(s.token_ids) <= 48 and s.source == "tokens" for s in seqs),
          "tokens corpus: one sequence per usable sample, truncated to --seq-len; the rejected one is skipped")
    check(sum(s["samples_rejected"] for s in sources) == 1 and {s["name"] for s in sources} == set(gcap.GEN_CATEGORIES),
          "tokens corpus: per-category sources with the rejected count")
    seqs2, _ = gcap.build_tokens_corpus(p, 48, max_seqs=1)
    check(len(seqs2) == 5, "--gen-max-seqs 1: one sequence per category")
    seqs3, _ = gcap.build_tokens_corpus(p, 48)
    check([s.token_ids for s in seqs] == [s.token_ids for s in seqs3], "tokens corpus is deterministic")
    bad = tmp / "bad.jsonl"
    write_samples(bad, n=1, extra=[{"id": "code/900", "category": "code", "rejected": False, "token_ids": [2, 9, 9, 9],
                                     "kl_token_overlap": 3}])
    try:
        gcap.build_tokens_corpus(bad, 48)
        check(False, "a usable sample overlapping the KL tokens is refused")
    except SystemExit as e:
        check("kl_token_overlap" in str(e), "a usable sample overlapping the KL tokens is refused")
    nid = tmp / "noid.jsonl"
    nid.write_text(json.dumps({"category": "chat", "rejected": False, "token_ids": [1, 2]}) + "\n", encoding="utf-8")
    try:
        gcap.build_tokens_corpus(nid, 48)
        check(False, "a line without an id is refused")
    except SystemExit:
        check(True, "a line without an id is refused")


# --------------------------------------------------------------------------------------------
# (c) CLIs, dry run
# --------------------------------------------------------------------------------------------

def test_dry_runs(tmp: Path, ckpt: Path, samples: Path) -> None:
    out = tmp / "hess-dry"
    r = run_cli("hessian_capture.py", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--gen-file", str(samples),
                "--seq-len", "48", "--rms-taps", "--dry-run", "--out-dir", str(out))
    text = r.stdout + r.stderr
    check(r.returncode == 0, f"hessian_capture.py --arch gemma4_unified --dry-run passes on the CPU (rc {r.returncode}) {text[-400:] if r.returncode else ''}")
    check(all(s in text for s in ("arch gemma4_unified", "gemma_layout.cpp", "L00.in.hess", "L01.in.hess", "L01.out.hess",
                                  "L03.mlp_mid.hess", "lm_head.hess", "L00.mlp_in.rms.hess", "--dry-run: nothing captured")),
          "dry-run: the audit of gemma_layout.cpp, the tap files of every layer, lm_head and the rms taps are listed")
    check("L01.in.hess" in text and "attn.v" not in text.split("L01.in.hess")[1].split("\n")[0],
          "dry-run: the full layer's in tap has no attn.v key")
    check(not out.exists(), "dry-run writes nothing")
    r = run_cli("hessian_capture.py", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--dry-run", "--out-dir", str(out))
    check(r.returncode != 0 and "gen-file" in (r.stdout + r.stderr), "no --gen-file for gemma: refused, naming it")
    r = run_cli("imatrix_capture.py", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--dry-run")
    check(r.returncode == 0 and "--dry-run: PASS" in r.stdout and "lm_head" in r.stdout, f"imatrix_capture.py --arch gemma4_unified --dry-run passes ({(r.stdout + r.stderr)[-300:]})")
    r = run_cli("imatrix_capture.py", "--arch", "gemma4_unified", "--model-dir", str(ckpt))
    check(r.returncode != 0 and "not implemented" in (r.stdout + r.stderr), "imatrix_capture.py --arch gemma4_unified without --dry-run says it is not implemented")


# --------------------------------------------------------------------------------------------
# (d) a real CPU capture
# --------------------------------------------------------------------------------------------

def second_moment(xs: list) -> np.ndarray:
    acc = None
    n = 0
    for x in xs:
        f = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
        acc = f.T @ f if acc is None else acc + f.T @ f
        n += f.shape[0]
    return (acc / n).numpy()


def rel(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def test_capture(tmp: Path, ckpt: Path, samples: Path) -> Path:
    dev = torch.device("cpu")
    arch = at.get_arch("gemma4_unified")
    text_config = at.text_config_view(arch, ckpt)
    check(list(text_config.layer_types) == [at.SLIDING, at.FULL, at.SLIDING, at.FULL], "tiny checkpoint: s F s F")
    specs = ic.enumerate_quantized_linears(text_config, False, False, arch=arch)
    index = hc.ShardIndex.load(ckpt) if hasattr(hc, "ShardIndex") else None
    from common import ShardIndex

    index = ShardIndex.load(ckpt)
    shapes = ic.tensor_shapes(index, sorted({n for s in specs for n in s.hf_names}))
    expect_k = ic.verify_no_k_concat(specs, shapes)
    plans = hc.build_tap_plan(specs, expect_k, arch)
    rms_plans = hc.build_rms_plans({p.file: p.keys for p in plans if p.scope == "layer"}, 4, int(text_config.hidden_size), arch)
    seqs, sources = gcap.build_tokens_corpus(samples, 64)
    total = sum(len(s.token_ids) for s in seqs)

    out = tmp / "hess"
    out.mkdir()
    ref = gcap.GemmaCaptureRef(ckpt, dev, dtype=torch.bfloat16, verbose=False)
    st = hc.run_capture(ref, seqs, plans, specs, 4, out, dev, False, rms_plans=rms_plans, arch=arch)
    check(set(st.written) == {p.file for p in plans} and set(st.rms_written) == {rp.file for rp in rms_plans},
          f"run_capture writes every planned tap and rms tap ({len(st.written)} + {len(st.rms_written)} files)")
    check(all(w["rows"] == total for w in st.written.values()), f"every tap saw all {total} tokens")

    # An independent second moment of the SAME HF layers through hooks.
    ref2 = GemmaReference(ckpt, dev, resident=False, verbose=False)
    caps: dict = {}
    hidden = [ref2.embed(s.token_ids).unsqueeze(0) for s in seqs]
    cap_ref = gcap.GemmaCaptureRef(ckpt, dev, dtype=torch.bfloat16, verbose=False)
    for i in range(4):
        layer = ref2.build_layer(i)
        hooks = []

        def grab(name, kind="in"):
            def pre(_m, inputs):
                caps.setdefault((i, name), []).append(inputs[0].detach().clone())

            def post(_m, inputs, output):
                caps.setdefault((i, name), []).append(output.detach().clone())
            return pre if kind == "in" else post
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(grab("q_in")))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(grab("o_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(grab("gate_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(grab("down_in")))
        hooks.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(grab("pff_x")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(grab("post_attn_out", "out")))
        hooks.append(layer.pre_feedforward_layernorm.register_forward_hook(grab("pff_out", "out")))
        hooks.append(layer.input_layernorm.register_forward_pre_hook(grab("ln_x")))
        nxt = []
        for j, h in enumerate(hidden):
            nxt.append(cap_ref.capture_layer_forward(layer, i, h))
        for hk in hooks:
            hk.remove()
        hidden = nxt
        del layer
    hdec = {}
    for p in plans:
        if p.scope == "layer":
            hdec[p.file] = {"in": "q_in", "out": "o_in", "mlp_in": "gate_in", "mlp_mid": "down_in"}[p.file.split(".")[1]]
    worst = 0.0
    for p in plans:
        if p.scope != "layer":
            continue
        h_file, rows, _ = hc.read_hess_file(out / p.file)
        want = second_moment(caps[(p.layer, hdec[p.file])])
        worst = max(worst, rel(h_file.astype(np.float64), want))
    check(worst < 2e-4, f"every layer tap equals the independent hook-based second moment (worst relative {worst:.2e})")
    # gate_proj's input IS pre_feedforward_layernorm's output, and is NOT post_attention_layernorm's.
    same = all(torch.equal(a, b) for a, b in zip(caps[(0, "gate_in")], caps[(0, "pff_out")]))
    h_gate = hc.read_hess_file(out / "L00.mlp_in.hess")[0].astype(np.float64)
    h_post = second_moment(caps[(0, "post_attn_out")])
    check(same and rel(h_gate, h_post) > 0.05, f"mlp_in = pre_feedforward_layernorm's output, not post_attention_layernorm's "
          f"(difference to the post-attention tap {rel(h_gate, h_post):.2f})")
    # lm_head: the post-final-norm hidden, before the softcap.
    final = [ref2.final_norm(h) for h in hidden]
    h_lm = hc.read_hess_file(out / "lm_head.hess")[0].astype(np.float64)
    check(rel(h_lm, second_moment(final)) < 2e-4, "lm_head.hess is the second moment of the post-final-norm hidden")
    # rms taps: E[rms(x)^T rms(x)] over the norms' inputs; scale = w (offset 0)
    worst = 0.0
    for rp in rms_plans:
        xs = caps[(rp.layer, "ln_x" if rp.norm == "input_layernorm" else "pff_x")]
        r = [x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + 1e-6, -0.5) for x in xs]
        h_file = hc.read_hess_file(out / rp.file)[0].astype(np.float64)
        worst = max(worst, rel(h_file, second_moment(r)))
    check(worst < 2e-4, f"the rms taps equal E[rms(x)^T rms(x)] of the norms' inputs (worst {worst:.2e})")
    w0 = index.get_tensor(f"{at.LAYER_PREFIX}1.pre_feedforward_layernorm.weight").float().numpy()
    check(np.array_equal(st.rms_scale["L01.mlp_in.rms.hess"], w0), "the rms gate's scale is w (offset 0), not 1 + w")
    gates_rms = hc.evaluate_rms_gates(rms_plans, st.rms_written, out, {f: w["rows"] for f, w in st.written.items()}, st.rms_scale)
    check(gates_rms["ok"], f"evaluate_rms_gates passes with offset 0: {[k for k in hc.RMS_GATES if not gates_rms[k]]}")
    sg = st.shared_gate.results
    check(sorted(sg) == ["layer0:mlp.up_proj", "layer0:self_attn.k_proj", "layer0:self_attn.v_proj", "layer1:mlp.up_proj",
                         "layer1:self_attn.k_proj"] and all(v["same_storage"] and v["elementwise_equal"] for v in sg.values()),
          f"shared-input gate: k/v/up on the sliding layer, k/up (no v) on the full one: {sorted(sg)}")
    audit = ic.audit_converter_source(arch.converter_source(), arch=arch.converter_arch)
    structural = hc.structural_zero_channels(ckpt, {p.file: (p.scope, p.layer, p.module) for p in plans}, arch)
    shared_expected = [ref.layer_types.index(t) for t in arch.shared_gate if t in ref.layer_types]
    gates = hc.evaluate_gates(audit, [s.key for s in specs], plans, st, [k for p in plans for k in p.keys], shared_expected, structural)
    check(gates["ok"] and gates["complete"], f"evaluate_gates passes on the Gemma set: { {k: v for k, v in gates.items() if k.endswith('_ok') and not v} }")
    check(structural["L00.in.hess"] == [] and "L00.out.hess" not in structural and structural["lm_head.hess"] == [],
          "structural_zero_channels: single-file checkpoint, offset 0, no norm in front of o_proj / down_proj")
    # the manifest, as main() writes it
    files = dict(st.written)
    keys = {k: p.file for p in plans for k in p.keys}
    for rp in rms_plans:
        files[rp.file] = st.rms_written[rp.file]
    hc.write_manifest(out, files, keys, {"arch": "gemma4_unified", "model_dir": str(ckpt),
                                         "config_sha256": "0" * 64, "taps": {}, "gates": gates},
                      rms_keys=hc.rms_keys_of(rms_plans))
    man = json.loads((out / "hessian.json").read_text(encoding="utf-8"))
    check(man["arch"] == "gemma4_unified" and "text.layers.0.attn.q" in man["rms_keys"] and
          man["keys"]["lm_head"] == "lm_head.hess", "the manifest carries arch, keys (lm_head included) and rms_keys")
    return out


def test_regate_structural(tmp: Path, ckpt: Path) -> None:
    from safetensors.torch import load_file, save_file

    d = tmp / "ckpt-dead"
    shutil.copytree(ckpt, d)
    sd = load_file(str(d / "model.safetensors"))
    k = f"{at.LAYER_PREFIX}0.pre_feedforward_layernorm.weight"
    sd[k][3] = 0.0
    save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
    z = hc.structural_zero_channels(d, {"L00.mlp_in.hess": ("layer", 0, "mlp.gate_proj")}, at.GEMMA)
    check(z == {"L00.mlp_in.hess": [3]}, "a Gemma norm weight of exactly 0 is a dead channel (offset 0): channel 3 found")
    z = hc.structural_zero_channels(d, {"L00.mlp_in.hess": ("layer", 0, "mlp.gate_proj")}, at.QWEN) if False else None


# --------------------------------------------------------------------------------------------
# (e) rotation_oracle
# --------------------------------------------------------------------------------------------

def make_rotation_file(path: Path, hidden: int = 768, kind: str = "q2ab", seed: int = 11, k_down: int = 512,
                       k_o: int = 256, k_o_full: int = 512) -> None:
    """A rotation file in --rotation-out's format with Python-drawn tensors (the identities hold for any
    signs and any orthogonal mix)."""
    rng = np.random.default_rng(seed)
    block = min(hidden & -hidden, 1024)
    nblk = hidden // block
    t = {"rotation.signs": rng.choice([-1.0, 1.0], size=hidden).astype(np.float32),
         "rotation.mix": np.linalg.qr(rng.standard_normal((nblk, nblk)))[0].astype(np.float32)}
    order = ["rotation.signs", "rotation.mix"]
    if kind == "q2ab":
        t["rotation.had_down_signs"] = rng.choice([-1.0, 1.0], size=k_down).astype(np.float32)
        t["rotation.had_o_signs"] = rng.choice([-1.0, 1.0], size=k_o).astype(np.float32)
        t["rotation.had_o_full_signs"] = rng.choice([-1.0, 1.0], size=k_o_full).astype(np.float32)
        order += ["rotation.had_down_signs", "rotation.had_o_signs", "rotation.had_o_full_signs"]
    fp = {"kind": kind, "seed": seed, "tensors_sha256": ro.fingerprint_of(t, order), "hidden": hidden, "block": block}
    rot = {"kind": kind, "seed": seed, "hidden": hidden, "block": block, "out_fold": "had_only"}
    if nblk != 5:
        rot["nblk"] = nblk
    if kind == "q2ab":
        rot["had"] = {"down": 512, "o": 256, "o_full": 256}
    hdr, data, off = {}, b"", 0
    for n in order:
        raw = np.ascontiguousarray(t[n], dtype="<f4").tobytes()
        hdr[n] = {"dtype": "F32", "shape": list(t[n].shape), "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    hdr["__metadata__"] = {"format": "r4dx-rotation", "rotation": json.dumps(rot), "fingerprint": json.dumps(fp),
                           "config_sha256": "0" * 64}
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


def test_rotation_oracle(tmp: Path) -> Path:
    f = tmp / "rot-py.safetensors"
    make_rotation_file(f)
    rot = ro.RotationSpec(f)
    check(rot.kind == "q2ab" and rot.hidden == 768 and rot.nblk == 3 and rot.fingerprint()["tensors_sha256"] == rot.recorded["tensors_sha256"],
          "RotationSpec reads the file and re-derives the fingerprint")
    bad = tmp / "rot-bad.safetensors"
    raw = bytearray(f.read_bytes())
    raw[-1] ^= 0x40
    bad.write_bytes(bytes(raw))
    try:
        ro.RotationSpec(bad)
        check(False, "an edited rotation file is refused")
    except ValueError as e:
        check("hash" in str(e) or "orthogonal" in str(e) or "+-1" in str(e), "an edited rotation file is refused")
    torch.manual_seed(0)
    x = torch.randn(5, 768, dtype=torch.float64)
    q = rot.apply_q(x)
    check(abs(float(q.norm() / x.norm()) - 1) < 1e-6, "x Q keeps the norm (Q orthogonal to the fp32 mix's ~1e-8)")
    # a dense Q from the operator, checked orthogonal and equal to the row-wise application
    qd = rot.apply_q(torch.eye(768, dtype=torch.float64))
    check(float((qd @ qd.T - torch.eye(768, dtype=torch.float64)).abs().max()) < 1e-6 and float((x @ qd - q).abs().max()) < 1e-12,
          "the dense Q is orthogonal and x Q equals x @ Q")

    # fold / Hessian transform: the proxy loss tr(W H W^T) is invariant, per module kind
    rng = np.random.default_rng(3)

    def spd(k):
        a = rng.standard_normal((k, k * 2))
        return torch.from_numpy((a @ a.T / (2 * k)).astype(np.float32))
    norm_w = torch.from_numpy(rng.uniform(0.6, 1.4, 768).astype(np.float32))
    cases = [("sliding_attention", "self_attn.q_proj", 256, 768), ("sliding_attention", "mlp.gate_proj", 512, 768),
             ("sliding_attention", "self_attn.o_proj", 768, 256), ("full_attention", "self_attn.o_proj", 768, 512),
             ("sliding_attention", "mlp.down_proj", 768, 512)]
    for lt, mod, n, k in cases:
        w = torch.from_numpy(rng.standard_normal((n, k)).astype(np.float32))
        if rot.is_in_proj(mod):
            # H of the (rms(x) * w) the unfolded linear sees is D H_rms D; the folded input is rms(x) Q
            h_rms = spd(k)
            d = norm_w.double()
            h_post = (d[:, None] * h_rms.double() * d[None, :]).float()
            wf = rot.fold_weight(w, lt, mod, norm_w, 0.0)
            hp_rms = rot.transform_hessian(h_rms, lt, mod, norm_w, 0.0, is_rms=True)
            hp_div = rot.transform_hessian(h_post, lt, mod, norm_w, 0.0, is_rms=False)
            loss0 = float(torch.trace((w.double() * d[None, :]) @ h_rms.double() @ (w.double() * d[None, :]).T))
            loss1 = float(torch.trace(wf.double() @ hp_rms.double() @ wf.double().T))
            check(abs(loss0 - loss1) / loss0 < 1e-5, f"{mod}: tr(W' H' W'^T) = tr(W D H_rms D W^T) ({loss0:.4g} vs {loss1:.4g})")
            check(rel(hp_div.numpy(), hp_rms.numpy()) < 2e-4 and float((hp_rms - hp_rms.T).abs().max()) == 0.0,
                  f"{mod}: the division path equals the rms path; H' is exactly symmetric")
        else:
            h = spd(k)
            wf = rot.fold_weight(w, lt, mod)
            hp = rot.transform_hessian(h, lt, mod)
            loss0 = float(torch.trace(w.double() @ h.double() @ w.double().T))
            loss1 = float(torch.trace(wf.double() @ hp.double() @ wf.double().T))
            check(abs(loss0 - loss1) / loss0 < 1e-5, f"{lt[:4]} {mod}: tr(W Hb H Hb^T W^T) = tr(W H W^T)")
    try:
        rot.transform_hessian(spd(768), "sliding_attention", "self_attn.q_proj", torch.zeros(768), 0.0, is_rms=False)
        check(False, "a dead norm channel refuses the division path")
    except ValueError as e:
        check("--rms-taps" in str(e), "a dead norm channel refuses the division path, pointing at the rms taps")
    # q2a: out-projections are not folded
    f2 = tmp / "rot-q2a.safetensors"
    make_rotation_file(f2, kind="q2a")
    r2 = ro.RotationSpec(f2)
    w = torch.randn(768, 512)
    check(torch.equal(r2.fold_weight(w, "sliding_attention", "mlp.down_proj"), w) and r2.had_site("full_attention", "self_attn.o_proj") is None,
          "q2a: o / down are not folded")
    return f


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def read_container(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    data = memoryview(raw)[8 + n:]
    out = {}
    for k, e in hdr.items():
        if k != "__metadata__":
            b, d = e["data_offsets"]
            out[k] = bytes(data[b:d])
    return hdr, out


def make_convert_ckpt(tmp: Path, exe: Path) -> Path:
    d = tmp / "ckpt-768"
    r = subprocess.run([PY, str(REPO / "tools" / "convert_ref" / "make_tiny_gemma.py"), "--out-dir", str(d), "--layers", "2", "--full-every", "2"],
                       capture_output=True, text=True)
    check(r.returncode == 0, f"make_tiny_gemma.py writes the converter's tiny checkpoint ({r.stderr[-200:]})")
    return d


def test_rotation_vs_converter(tmp: Path, exe: Path) -> None:
    d = make_convert_ckpt(tmp, exe)
    rot_file = tmp / "rot-cpp.safetensors"
    r = subprocess.run([str(exe), "--input", str(d), "--rotate", "q2ab", "--rotation-out", str(rot_file)], capture_output=True, text=True)
    check(r.returncode == 0 and rot_file.is_file(), f"r4dx-convert --rotation-out writes the file ({r.stderr[-200:]})")
    if not rot_file.is_file():
        return
    rot = ro.RotationSpec(rot_file)  # re-derives the fingerprint: equals the converter's, or this raises
    check(rot.kind == "q2ab" and rot.hidden == 768 and rot.nblk == 3,
          "the converter's --rotation-out file passes RotationSpec's fingerprint check (Python and C++ hash alike)")
    out = tmp / "bf16-rot.r4dx"
    r = subprocess.run([str(exe), "--input", str(d), "--output", str(out), "--layouts", "bf16", "--lm-head", "bf16", "--rotate", "q2ab",
                        "--threads", "2"], capture_output=True, text=True)
    check(r.returncode == 0, f"a --rotate q2ab bf16 conversion ({r.stderr[-200:]})")
    _hdr, ct = read_container(out)
    from common import ShardIndex

    idx = ShardIndex.load(d)
    worst_frac, worst_ulp = 0.0, 0
    for layer, lt in ((0, at.SLIDING), (1, at.FULL)):
        for mod, base in (("self_attn.q_proj", "attn.q"), ("mlp.gate_proj", None), ("self_attn.o_proj", "attn.o"), ("mlp.down_proj", "mlp.down")):
            w = idx.get_tensor(f"{at.LAYER_PREFIX}{layer}.{mod}.weight")
            if mod == "mlp.gate_proj":
                up = idx.get_tensor(f"{at.LAYER_PREFIX}{layer}.mlp.up_proj.weight")
                wf = torch.cat([rot.fold_weight(w, lt, mod, idx.get_tensor(f"{at.LAYER_PREFIX}{layer}.pre_feedforward_layernorm.weight"), 0.0),
                                rot.fold_weight(up, lt, "mlp.up_proj", idx.get_tensor(f"{at.LAYER_PREFIX}{layer}.pre_feedforward_layernorm.weight"), 0.0)])
                name = f"text.layers.{layer}.mlp.gate_up.bf16.w"
            else:
                nw = idx.get_tensor(f"{at.LAYER_PREFIX}{layer}.input_layernorm.weight") if rot.is_in_proj(mod) else None
                wf = rot.fold_weight(w, lt, mod, nw, 0.0)
                name = f"text.layers.{layer}.{base}.bf16.w"
            got = np.frombuffer(ct[name], dtype="<u2").astype(np.int64)
            want = (wf.contiguous().view(torch.int32).numpy().astype(np.int64) + 0x7FFF + ((wf.contiguous().view(torch.int32).numpy().astype(np.int64) >> 16) & 1)) >> 16
            want = (want & 0xFFFF).reshape(-1)
            diff = np.abs(got - want.reshape(-1))
            worst_frac = max(worst_frac, float((diff > 0).mean()))
            worst_ulp = max(worst_ulp, int(diff.max()))
    check(worst_frac < 1e-3 and worst_ulp <= 1,
          f"rotation_oracle.fold_weight equals the converter's folded bf16 weights (differing elements {worst_frac:.1e}, max {worst_ulp} ulp)")


# --------------------------------------------------------------------------------------------
# (f) trellis_quant on the tiny checkpoint
# --------------------------------------------------------------------------------------------

def test_trellis_model_linears(tmp: Path, ckpt: Path) -> None:
    lin = tq.model_linears(ckpt)  # arch auto-detected
    names = [t["name"] for t in lin]
    check(len(lin) == 7 + 6 + 7 + 6 and f"{at.LAYER_PREFIX}1.self_attn.v_proj.weight" not in names and
          f"{at.LAYER_PREFIX}0.self_attn.v_proj.weight" in names,
          f"model_linears (single-file checkpoint): 26 linears, no v_proj on the full layers ({len(lin)})")
    check({t["qgroup"] for t in lin if t["layer"] == 1 and t["module"].startswith("self_attn.") and t["module"] != "self_attn.o_proj"} == {"self_attn.qkv"},
          "the full layer's q / k share the qkv allocator group")
    rates = tq.allocate(lin, 4.5)
    check(set(rates) == set(names) and set(rates.values()) <= {4.0, 5.0}, "allocate works on the Gemma list")
    check([m for m in tq.module_list("full_attention", at.GEMMA)] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
                                                                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
          "module_list(full_attention, GEMMA)")


def write_synth_hessians(out: Path, ckpt: Path, with_rms: bool, seed: int = 21) -> None:
    """A hessian.json + .hess set for the tiny checkpoint's taps, synthetic SPD matrices (an AR(1) covariance
    with per-channel scales) -- enough for the oracle's own code, which never looks at where they came from."""
    out.mkdir(parents=True, exist_ok=True)
    arch = at.GEMMA
    tc = at.text_config_view(arch, ckpt)
    specs = ic.enumerate_quantized_linears(tc, False, False, arch=arch)
    from common import ShardIndex

    idx = ShardIndex.load(ckpt)
    shapes = ic.tensor_shapes(idx, sorted({n for s in specs for n in s.hf_names}))
    plans = hc.build_tap_plan(specs, ic.verify_no_k_concat(specs, shapes), arch)
    rng = np.random.default_rng(seed)
    files, keys = {}, {}
    rms_files, rms_keys = {}, {}

    def make(k, scale_seed):
        i = np.arange(k)
        cov = 0.8 ** np.abs(i[:, None] - i[None, :])
        sc = np.exp(0.3 * np.random.default_rng(scale_seed).standard_normal(k))
        return (cov * sc[:, None] * sc[None, :]).astype(np.float64)
    for n, p in enumerate(plans):
        if p.scope == "lm_head":
            continue
        h = make(p.k, seed + n)
        files[p.file] = hc.write_hess_file(out / p.file, h, 4096)
        for key in p.keys:
            keys[key] = p.file
        if with_rms and p.file.split(".")[1] in ("in", "mlp_in"):
            rf = hc.rms_file_name(p.file)
            rms_files[rf] = hc.write_hess_file(out / rf, make(p.k, seed + 1000 + n), 4096)
            for key in p.keys:
                rms_keys[key] = rf
    hc.write_manifest(out, {**files, **rms_files}, keys, {"arch": "gemma4_unified", "synthetic": True},
                      rms_keys=rms_keys if with_rms else None)


def test_trellis_dry_run(tmp: Path, ckpt: Path, rot_file: Path) -> None:
    hd = tmp / "hess-synth"
    write_synth_hessians(hd, ckpt, with_rms=True)
    r = run_cli("trellis_quant.py", "quantize-model", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--hessian-dir", str(hd),
                "--out-dir", str(tmp / "oracle-dry"), "--K", "4", "--hessian-basis", "matched", "--dry-run")
    text = r.stdout + r.stderr
    check(r.returncode == 0 and "arch gemma4_unified" in text and "L00 sliding in" in text and "L01 full_at" in text and "--dry-run: OK" in text,
          f"trellis_quant.py quantize-model --arch gemma4_unified --dry-run passes ({text[-300:]})")
    check(not (tmp / "oracle-dry").exists(), "the dry-run writes nothing")
    r = run_cli("trellis_quant.py", "quantize-model", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--hessian-dir", str(hd),
                "--out-dir", str(tmp / "oracle-dry"), "--K", "4", "--hessian-basis", "matched", "--dry-run", "--rotation", str(rot_file))
    check(r.returncode == 0 and "rotation q2ab" in r.stdout and "H' = Q^T H_rms Q" in r.stdout and "H' = Hb^T H Hb (o_full)" in r.stdout,
          f"--dry-run --rotation lists the fold: rms Q^T H Q for in-projections, Hb^T H Hb per site ({(r.stdout + r.stderr)[-300:]})")
    shutil.rmtree(hd / "L00.in.hess", ignore_errors=True)
    os.remove(hd / "L00.in.hess")
    r = run_cli("trellis_quant.py", "quantize-model", "--arch", "gemma4_unified", "--model-dir", str(ckpt), "--hessian-dir", str(hd),
                "--out-dir", str(tmp / "oracle-dry"), "--K", "4", "--dry-run")
    check(r.returncode != 0 and "MISSING" in r.stdout and "L00.in.hess" in r.stdout, "a missing Hessian file fails the dry-run, naming it")
    r = run_cli("trellis_quant.py", "quantize-model", "--model-dir", str(ckpt), "--hessian-dir", str(hd), "--out-dir", str(tmp / "o"),
                "--K", "4", "--rotation", str(rot_file), "--arch", "qwen3_5", "--dry-run")
    check(r.returncode != 0 and "Gemma 4 only" in (r.stdout + r.stderr), "--rotation with --arch qwen3_5 is refused")


def test_e2e_rotated(tmp: Path, exe: Path) -> None:
    """The chain: converter --rotation-out -> oracle (rotated, CPU, tiny) -> r4dx-convert --rotate --trellis-from."""
    d = tmp / "ckpt-768"
    if not d.is_dir():
        make_convert_ckpt(tmp, exe)
    rot_file = tmp / "rot-e2e.safetensors"
    r = subprocess.run([str(exe), "--input", str(d), "--rotate", "q2ab", "--rotation-out", str(rot_file)], capture_output=True, text=True)
    check(r.returncode == 0, "e2e: --rotation-out")
    hd = tmp / "hess-e2e"
    write_synth_hessians(hd, d, with_rms=True)
    # drop the rms tap of layer 1's in taps: that group takes the division path (the checkpoint has no dead channel)
    man = json.loads((hd / "hessian.json").read_text(encoding="utf-8"))
    for key in [k for k in man["rms_keys"] if k.startswith("text.layers.1.attn.")]:
        del man["rms_keys"][key]
    (hd / "hessian.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    oracle = tmp / "oracle-e2e"
    t0 = time.perf_counter()
    r = run_cli("trellis_quant.py", "quantize-model", "--arch", "gemma4_unified", "--model-dir", str(d), "--hessian-dir", str(hd), "--out-dir", str(oracle),
                "--K", "4", "--hessian-basis", "matched", "--device", "cpu", "--backend", "cpu", "--rotation", str(rot_file))
    dt_s = time.perf_counter() - t0
    check(r.returncode == 0, f"e2e: the rotated CPU oracle run finishes ({dt_s:.0f}s) {(r.stdout + r.stderr)[-600:] if r.returncode else ''}")
    if r.returncode != 0:
        return
    m = json.loads((oracle / "weights_override.json").read_text(encoding="utf-8"))
    rot = ro.RotationSpec(rot_file)
    check(m["complete"] and m["rotation"] == rot.fingerprint() and m["arch"] == "gemma4_unified" and len(m["tensors"]) == 13,
          f"e2e: the manifest is complete (13 tensors), arch gemma4_unified, rotation = the file's fingerprint")
    recs = m["tensors"]
    check(recs[f"{at.LAYER_PREFIX}0.mlp.gate_proj.weight"]["hessian"].get("rms") is True and
          recs[f"{at.LAYER_PREFIX}1.self_attn.q_proj.weight"]["hessian"].get("rms") is None and
          recs[f"{at.LAYER_PREFIX}1.self_attn.o_proj.weight"]["hessian"]["rotated"] == "q2ab",
          "e2e: in-projections with an rms tap used it, layer 1's attention took the division path, o_proj is Hadamard-folded")
    out = tmp / "e2e.r4dx"
    r = subprocess.run([str(exe), "--input", str(d), "--output", str(out), "--rotate", "q2ab", "--trellis-from", str(oracle), "--threads", "2"],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    check(r.returncode == 0 and "pass 13/13" in text.replace("\n", " ") or (r.returncode == 0 and "13/13 HF tensors within tolerance" in text),
          f"e2e: r4dx-convert --rotate q2ab --trellis-from imports the oracle and its reconstruction check passes against fold(W) ({text[-500:]})")
    # the same oracle without --rotate is refused, and an unrotated run of another seed too
    r = subprocess.run([str(exe), "--input", str(d), "--output", str(tmp / "e2e-no.r4dx"), "--trellis-from", str(oracle)], capture_output=True, text=True)
    check(r.returncode != 0 and "ROTATED" in (r.stdout + r.stderr), "e2e: the rotated oracle without --rotate is refused")
    r = subprocess.run([str(exe), "--input", str(d), "--output", str(tmp / "e2e-seed.r4dx"), "--rotate", "q2ab", "--rotation-seed", "9",
                        "--trellis-from", str(oracle)], capture_output=True, text=True)
    check(r.returncode != 0 and "rotation." in (r.stdout + r.stderr), "e2e: another --rotation-seed is refused by the fingerprint")


# --------------------------------------------------------------------------------------------


def main() -> int:
    tests = []
    with tempfile.TemporaryDirectory(prefix="r4dx_arch_gemma_") as td:
        tmp = Path(td)
        try:
            test_tables()
            ckpt = build_tiny_checkpoint(tmp / "ckpt", seed=0, window=4, layers=4, vocab=128)
            samples = tmp / "samples.jsonl"
            write_samples(samples, n=12, length=60)
            test_corpus(tmp)
            test_dry_runs(tmp, ckpt, samples)
            test_capture(tmp, ckpt, samples)
            test_regate_structural(tmp, ckpt)
            rot_file = test_rotation_oracle(tmp)
            test_trellis_model_linears(tmp, ckpt)
            test_trellis_dry_run(tmp, ckpt, rot_file)
            exe = convert_exe()
            if exe is None:
                print("SKIP (e), (f) against r4dx-convert: no R4DX_CONVERT_EXE / build/*/src/convert/r4dx-convert.exe")
            else:
                test_rotation_vs_converter(tmp, exe)
                test_e2e_rotated(tmp, exe)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            FAILS.append("exception")
    if torch.cuda.is_initialized():
        FAILS.append("torch.cuda was initialized: this test must stay on the CPU")
    if FAILS:
        print(f"FAILED {len(FAILS)} of {CHECKS} checks")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
