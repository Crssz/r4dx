"""tests/reference/test_hessian_rms.py

CPU-only test of hessian_capture.py's weightless rms taps (`--rms-taps` / `--rms-only`,
docs/quant2.md 3.2). Needs NO checkpoint and NO GPU (it never calls torch.cuda, and checks at the end
that CUDA/HIP was never initialized): a tiny random two-layer Qwen3_5 stack (one linear_attention and
one full_attention layer, hidden 64) from the reference venv's transformers is driven layer-major on
the CPU through the SAME functions the tool uses -- run_capture, evaluate_rms_gates,
finish_rms_only / merge_rms_manifest, write_manifest, regate -- with a stand-in for
full_logits_golden.StreamingReference. Layer 1's post_attention_layernorm has a dead channel
((1 + w) == -1 + 1 == 0), like the real checkpoint's layer 7 channel 3994.

    (a) rms_weightless == Qwen3_5RMSNorm._norm(x.float()) bit for bit, and the module's output is
        (rms_weightless(x) * (1 + w)).type_as(x); norm_eps reads .eps and .variance_epsilon
    (b) each L{i}.{in,mlp_in}.rms.hess equals E[rms(x)^T rms(x)] computed independently in fp64
        from the norm inputs of a separate forward; rows = tokens
    (c) a full capture with rms taps: the rms gates pass (consistency worst << 2%); the dead channel
        is exactly 0 on the post-norm diagonal and > 0 on the rms one
    (d) --rms-only's finish on a base manifest: files + rms_keys + rms_capture added, LF only, and
        every other byte unchanged (stripping the rms entries re-serializes the base exactly)
    (e) failures merge nothing (hessian.json byte-identical, hessian_rms.failed.json written): a
        different corpus (consistency), a zeroed rms diagonal channel (no exemption), a post-norm
        rows mismatch, a manifest that changed during the capture
    (f) corpus_mismatches, build_rms_plans, rms_file_name and write_manifest's rms_keys refusals
    (g) --regate keeps rms_keys and re-gates the rms files with no dead-channel exemption

Plain script, no pytest dependency, like test_manifest.py:

    python tests\\reference\\test_hessian_rms.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_REFERENCE_DIR = REPO_ROOT / "tools" / "reference"

try:
    import numpy as np
    import torch
    import transformers  # noqa: F401
except ImportError as e:
    print(f"SKIP: {e} -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

sys.path.insert(0, str(TOOLS_REFERENCE_DIR))
import hessian_capture as hc  # noqa: E402

torch.set_num_threads(min(6, os.cpu_count() or 1))

CHECKS = 0
FAILURES: list[str] = []


def check(cond, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(label)
        print(f"FAIL {label}")


def raises(exc_type, needle: str, fn, label: str) -> None:
    try:
        fn()
    except exc_type as e:
        check(needle in str(e), f"{label} (message {str(e)[:160]!r} lacks {needle!r})")
        return
    check(False, f"{label} (did not raise {exc_type.__name__})")


# ---- the tiny model ------------------------------------------------------------------------------

HIDDEN = 64
DEAD = 13  # layer 1 post_attention_layernorm: (1 + w) == 0


def tiny_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"], rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25})
    cfg._attn_implementation = "eager"
    return cfg


class TinyRef:
    """What run_capture uses of full_logits_golden.StreamingReference, on the CPU: .device,
    .layer_types, .build_layer(i) (deterministic per i), .embed(ids), ._manual_rope, ._manual_mask."""

    def __init__(self, cfg):
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
        from full_logits_golden import StreamingReference

        self.m = m
        self.text_config = cfg
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.layer_types = list(cfg.layer_types)
        self.n_layers = cfg.num_hidden_layers
        g = torch.Generator().manual_seed(7)
        self.table = torch.randn(cfg.vocab_size, HIDDEN, generator=g).to(torch.bfloat16)
        self._rope = StreamingReference._manual_rope
        self._mask = StreamingReference._manual_mask

    def _manual_rope(self, T):
        return self._rope(self, T)

    def _manual_mask(self, T):
        return self._mask(self, T)

    def build_layer(self, i: int):
        layer = self.m.Qwen3_5DecoderLayer(self.text_config, i)
        g = torch.Generator().manual_seed(1000 + i)
        with torch.no_grad():
            for name, p in layer.named_parameters():
                if name.endswith("layernorm.weight"):
                    v = (0.3 * torch.randn(p.shape, generator=g)).clamp(-0.8, 0.8)
                    if i == 1 and name.startswith("post_attention_layernorm"):
                        v[DEAD] = -1.0
                else:
                    v = 0.15 * torch.randn(p.shape, generator=g)
                p.copy_(v)
        layer = layer.to(torch.bfloat16).eval()
        for p in layer.parameters():
            p.requires_grad_(False)
        return layer

    def embed(self, ids):
        return self.table[torch.tensor(ids)].unsqueeze(0)


def corpus(seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [hc.Seq(f"s{j}", "test", rng.integers(0, 97, size=n).tolist())
            for j, n in enumerate((96, 80, 128, 100))]


def post_plans() -> list:
    """The norm-fed post-norm taps of the two layers, as build_tap_plan would group them."""
    return [
        hc.TapPlan("L00.in.hess", "layer", 0, "linear_attn.in_proj_qkv", HIDDEN,
                   ["text.layers.0.gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_z"]),
        hc.TapPlan("L00.mlp_in.hess", "layer", 0, "mlp.gate_proj", HIDDEN, ["text.layers.0.mlp.gate_up"]),
        hc.TapPlan("L01.in.hess", "layer", 1, "self_attn.q_proj", HIDDEN,
                   ["text.layers.1.attn.qg", "text.layers.1.attn.k", "text.layers.1.attn.v"]),
        hc.TapPlan("L01.mlp_in.hess", "layer", 1, "mlp.gate_proj", HIDDEN, ["text.layers.1.mlp.gate_up"]),
    ]


def direct_rms_hessians(ref: TinyRef, seqs) -> dict:
    """E[rms(x)^T rms(x)] in fp64, from norm inputs recorded by an independent layer-major forward."""
    sums = {}
    rows = 0
    hidden = [ref.embed(s.token_ids) for s in seqs]
    with torch.no_grad():
        for i in range(2):
            layer = ref.build_layer(i)
            seen = {}
            hs = [layer.input_layernorm.register_forward_pre_hook(
                      lambda _m, a: seen.setdefault("in", []).append(a[0].detach().clone())),
                  layer.post_attention_layernorm.register_forward_pre_hook(
                      lambda _m, a: seen.setdefault("mlp_in", []).append(a[0].detach().clone()))]
            for j, h in enumerate(hidden):
                T = h.shape[1]
                hidden[j] = layer(hidden_states=h, position_embeddings=ref._manual_rope(T),
                                  attention_mask=None if i == 0 else ref._manual_mask(T),
                                  position_ids=None, past_key_values=None)
            for hd in hs:
                hd.remove()
            for tap, xs in seen.items():
                x = torch.cat([t.reshape(-1, HIDDEN) for t in xs]).double()
                r = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ref.text_config.rms_norm_eps)
                sums[f"L{i:02d}.{tap}.rms.hess"] = (r.t() @ r / r.shape[0]).numpy()
                rows = r.shape[0]
    return sums, rows


def base_manifest(out: Path, st) -> Path:
    """A validated post-norm set's hessian.json, as the full capture writes it (the part the rms
    merge must leave alone)."""
    keys = {k: p.file for p in post_plans() for k in p.keys}
    extra = {"tool": "tests/reference/test_hessian_rms.py", "config_sha256": "0" * 64,
             "corpus": {"sequences": 4, "tokens": 404, "seq_len": 2048, "sources": []},
             "taps": {p.file: {"scope": p.scope, "layer": p.layer, "module": p.module,
                               "tap": "test", "keys": p.keys} for p in post_plans()},
             "gates": {"ok": True, "trace": 5099.702477902174, "unicode": "ü"},
             "caveat": "x"}
    return hc.write_manifest(out, st.written, keys, extra)


# ---- (a) -----------------------------------------------------------------------------------------


def test_formula() -> None:
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    norm = m.Qwen3_5RMSNorm(HIDDEN, eps=1e-6)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        norm.weight.copy_(0.3 * torch.randn(HIDDEN, generator=g))
    norm = norm.to(torch.bfloat16)
    x = (3.0 * torch.randn(2, 17, HIDDEN, generator=g)).to(torch.bfloat16)
    r = hc.rms_weightless(x, hc.norm_eps(norm))
    check(r.dtype == torch.float32, "(a) rms_weightless is fp32")
    check(torch.equal(r, norm._norm(x.float())), "(a) rms_weightless == Qwen3_5RMSNorm._norm(x.float()) bitwise")
    with torch.no_grad():
        y = norm(x)
    check(torch.equal(y, (r * (1.0 + norm.weight.float())).type_as(x)),
          "(a) the module's output is (rms_weightless(x) * (1 + w)).type_as(x): the rms tap drops "
          "exactly the weight and the cast")
    check(hc.norm_eps(norm) == 1e-6, "(a) norm_eps reads Qwen3_5RMSNorm.eps")
    gated = m.Qwen3_5RMSNormGated(16, eps=3e-5)
    check(hc.norm_eps(gated) == 3e-5, "(a) norm_eps reads Qwen3_5RMSNormGated.variance_epsilon")
    raises(SystemExit, "neither", lambda: hc.norm_eps(torch.nn.Identity()), "(a) norm_eps refuses a module without one")


# ---- (b)-(e), (g) --------------------------------------------------------------------------------


def run(ref, seqs, out: Path, plans, rms_plans, shared_gate=True):
    out.mkdir(parents=True, exist_ok=True)
    return hc.run_capture(ref, seqs, plans, [], 2, out, torch.device("cpu"), False,
                          rms_plans=rms_plans, shared_gate=shared_gate)


def rms_plans_for(plans) -> list:
    return hc.build_rms_plans({p.file: p.keys for p in plans}, 2, HIDDEN)


def test_capture(tmp: Path) -> None:
    cfg = tiny_config()
    ref = TinyRef(cfg)
    seqs = corpus(11)
    n_tok = sum(len(s.token_ids) for s in seqs)
    rms = rms_plans_for(post_plans())
    check([rp.file for rp in rms] == ["L00.in.rms.hess", "L00.mlp_in.rms.hess", "L01.in.rms.hess",
                                      "L01.mlp_in.rms.hess"] and
          [rp.norm for rp in rms] == ["input_layernorm", "post_attention_layernorm"] * 2 and
          rms[2].keys == ["text.layers.1.attn.k", "text.layers.1.attn.qg", "text.layers.1.attn.v"],
          "(b) build_rms_plans: one rms tap per norm-fed post-norm tap, same keys")

    # (c) a full capture: post-norm taps + rms taps in one forward.
    full = tmp / "full"
    st = run(ref, seqs, full, post_plans(), rms)
    check(set(st.written) == {p.file for p in post_plans()} and set(st.rms_written) == {rp.file for rp in rms},
          "(c) run_capture wrote the post-norm files to .written and the rms files to .rms_written")
    check(all(st.rms_eps[f] == cfg.rms_norm_eps for f in st.rms_eps), "(c) each rms tap used the norm's own eps")
    check(all(w["rows"] == n_tok for w in st.rms_written.values()), "(c) rms rows == corpus tokens")
    check(st.rms_scale["L01.mlp_in.rms.hess"][DEAD] == 0.0 and
          (st.rms_scale["L01.in.rms.hess"] != 0.0).all(), "(c) recorded (1 + w): 0 at the dead channel only")

    direct, rows = direct_rms_hessians(ref, seqs)
    ok = rows == n_tok
    worst = 0.0
    for f, h64 in direct.items():
        h, r_rows, _ = hc.read_hess_file(full / f)
        worst = max(worst, float(np.abs(h.astype(np.float64) - h64).max() / np.abs(h64).max()))
        ok = ok and r_rows == n_tok
    check(ok and worst < 1e-5, f"(b) every rms file == fp64 E[rms(x)^T rms(x)] of the norm inputs "
                               f"(worst {worst:.2e} of max|H|)")

    gates = hc.evaluate_rms_gates(rms, st.rms_written, full, {f: w["rows"] for f, w in st.written.items()},
                                  st.rms_scale)
    hc.print_rms_gates(gates)
    check(gates["ok"], "(c) all rms gates pass on a consistent capture")
    w = gates["rms_consistency_worst"]["rel"]
    check(0.0 < w < 0.005, f"(c) consistency worst {w:.2e}: bf16 rounding of the post-norm input only")
    d_post = hc.read_hess_diag(full / "L01.mlp_in.hess")
    d_rms = hc.read_hess_diag(full / "L01.mlp_in.rms.hess")
    check(d_post[DEAD] == 0.0 and d_rms[DEAD] > 0.0 and
          gates["rms_per_file"]["L01.mlp_in.rms.hess"]["dead_channels"] == [DEAD],
          "(c) dead channel: post-norm diagonal exactly 0, rms diagonal > 0, excluded from the ratio")

    # (d) --rms-only's finish: merge into a base manifest.
    base_dir = tmp / "merge"
    shutil.copytree(full, base_dir)
    base_path = base_manifest(base_dir, st)
    base_bytes = base_path.read_bytes()
    doc = json.loads(base_bytes)
    rc = hc.finish_rms_only(base_dir, doc, hashlib.sha256(base_bytes).hexdigest(), rms, st,
                            generated_at="test")
    merged_bytes = base_path.read_bytes()
    merged = json.loads(merged_bytes)
    want_rms_keys = {k: rp.file for rp in rms for k in rp.keys}
    check(rc == 0 and merged["rms_keys"] == dict(sorted(want_rms_keys.items())) and
          len(merged["rms_keys"]) == 7, "(d) merged: rms_keys maps the 7 in-projection keys")
    check(all(merged["files"][rp.file] == {"K": HIDDEN, "rows": n_tok, "trace": st.rms_written[rp.file]["trace"]}
              for rp in rms), "(d) merged: the rms files listed under files with K, rows, trace")
    check(merged["rms_capture"]["gates"]["ok"] and merged["rms_capture"]["generated_at"] == "test" and
          merged["rms_capture"]["eps"] == [cfg.rms_norm_eps] and
          set(merged["rms_capture"]["taps"]) == set(want_rms_keys.values()),
          "(d) merged: rms_capture records gates, eps, taps and the provenance passed in")
    check(list(merged)[:5] == ["format", "version", "files", "keys", "rms_keys"] and
          list(merged)[-1] == "rms_capture" and
          [k for k in merged if k not in ("rms_keys", "rms_capture")] == list(doc),
          "(d) merged: field order kept, rms_keys after keys, rms_capture last")
    check(b"\r\n" not in merged_bytes and not (base_dir / hc.RMS_FAILED_MANIFEST_NAME).exists() and
          not list(base_dir.glob("*.tmp")), "(d) merged: LF only, no failed report, no temp file")
    files, keys, extra = hc.manifest_without_rms(merged)
    again = hc.write_manifest(base_dir, files, keys, extra, manifest_name="stripped.json")
    check(again.read_bytes() == base_bytes,
          "(d) stripping the rms entries re-serializes the base manifest byte for byte")

    # (e) failures merge nothing.
    def expect_refused(label, out_dir, st_run, doc_bytes, needle_gate, mutate=None):
        path = out_dir / hc.MANIFEST_NAME
        sha = hashlib.sha256(doc_bytes).hexdigest()
        if mutate:
            mutate()
        before = path.read_bytes()
        rc = hc.finish_rms_only(out_dir, json.loads(doc_bytes), sha, rms, st_run, generated_at="test")
        report = out_dir / hc.RMS_FAILED_MANIFEST_NAME
        rep = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        check(rc == 1 and path.read_bytes() == before and report.exists() and
              rep.get("rms_capture", {}).get("gates", {}).get(needle_gate) is False and
              rep["rms_capture"]["gates"]["ok"] is False and len(rep["rms_keys"]) == 7,
              f"(e) {label}: refused ({needle_gate} false), hessian.json unchanged, "
              f"{hc.RMS_FAILED_MANIFEST_NAME} written")
        if report.exists():
            report.unlink()

    # A different corpus (same token count): the rms taps saw other activations.
    other = tmp / "other"
    other.mkdir()
    for p in post_plans():
        shutil.copy2(full / p.file, other / p.file)
    base_manifest(other, st)
    st_other = run(ref, corpus(12), other, [], rms, shared_gate=False)
    check(not st_other.written and set(st_other.rms_written) == {rp.file for rp in rms},
          "(e) an rms-only run_capture writes the rms files and nothing else")
    expect_refused("a different corpus", other, st_other, (other / hc.MANIFEST_NAME).read_bytes(),
                   "rms_consistency_ok")

    # A zeroed rms diagonal channel -- what hooking a norm's OUTPUT would give on a dead channel.
    zero = tmp / "zero"
    shutil.copytree(full, zero)
    base_manifest(zero, st)
    h, r_rows, _ = hc.read_hess_file(zero / "L00.in.rms.hess")
    h[5, :] = 0.0
    h[:, 5] = 0.0
    st_zero = hc.CaptureState(rms_written=dict(st.rms_written), rms_scale=st.rms_scale, rms_eps=st.rms_eps)
    st_zero.rms_written["L00.in.rms.hess"] = hc.write_hess_file(zero / "L00.in.rms.hess", h, r_rows)
    expect_refused("an rms file with a zero diagonal channel (no exemption)", zero, st_zero,
                   (zero / hc.MANIFEST_NAME).read_bytes(), "rms_min_diag_ok")

    # The post-norm file's rows differ (a different corpus length).
    rows_dir = tmp / "rows"
    shutil.copytree(full, rows_dir)
    base_manifest(rows_dir, st)
    d = json.loads((rows_dir / hc.MANIFEST_NAME).read_bytes())
    d["files"]["L01.in.hess"]["rows"] += 1
    (rows_dir / hc.MANIFEST_NAME).write_bytes(json.dumps(d).encode())
    expect_refused("post-norm rows != rms rows", rows_dir, st, (rows_dir / hc.MANIFEST_NAME).read_bytes(),
                   "rms_rows_match_ok")

    # hessian.json changed while the capture ran.
    moved = tmp / "moved"
    shutil.copytree(full, moved)
    base_manifest(moved, st)
    orig = (moved / hc.MANIFEST_NAME).read_bytes()
    expect_refused("hessian.json changed during the capture", moved, st, orig, "manifest_unchanged_ok",
                   mutate=lambda: (moved / hc.MANIFEST_NAME).write_bytes(orig + b" "))

    # (g) --regate: keeps rms_keys; re-gates the rms files with no dead-channel exemption.
    test_regate(tmp, ref, st, rms, full)


def write_tiny_checkpoint(ref: TinyRef, model_dir: Path) -> None:
    """Only what structural_zero_channels reads: the norm weights, bf16, in one shard + index."""
    from safetensors.torch import save_file

    model_dir.mkdir(parents=True)
    tensors = {}
    for i in range(2):
        layer = ref.build_layer(i)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"model.language_model.layers.{i}.{norm}.weight"] = \
                getattr(layer, norm).weight.detach().clone().contiguous()
    save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: "model-00001-of-00001.safetensors" for n in tensors}}))


def test_regate(tmp: Path, ref: TinyRef, st, rms, full: Path) -> None:
    model_dir = tmp / "ckpt"
    write_tiny_checkpoint(ref, model_dir)
    gates = hc.evaluate_rms_gates(rms, st.rms_written, full, {f: w["rows"] for f, w in st.written.items()},
                                  st.rms_scale)
    # The post-norm gates as a capture from before the dead-channel rule recorded them: the dead
    # channel of L01.mlp_in.hess unexplained, so the set was written as hessian.failed.json.
    all_keys = [k for p in post_plans() for k in p.keys]
    post_gates = hc.evaluate_gates({"ok": True}, all_keys, post_plans(), st, all_keys, [0, 1], {})
    check(not post_gates["ok"] and post_gates["nonpositive_diag_unexplained"] == {"L01.mlp_in.hess": [DEAD]},
          "(g) setup: without the structural rule the dead channel fails the post-norm diag gate")

    def failed_set(name: str) -> Path:
        d = tmp / name
        shutil.copytree(full, d)
        files = dict(st.written)
        files.update(st.rms_written)
        extra = {"taps": {p.file: {"scope": p.scope, "layer": p.layer, "module": p.module}
                          for p in post_plans()},
                 "gates": dict(post_gates),
                 "rms_capture": hc.rms_capture_record("test --rms-taps", rms, st, gates)}
        hc.write_manifest(d, files, {k: p.file for p in post_plans() for k in p.keys}, extra,
                          manifest_name=hc.FAILED_MANIFEST_NAME, rms_keys=hc.rms_keys_of(rms))
        return d

    good = failed_set("regate_ok")
    rc = hc.regate(good, model_dir)
    out = json.loads((good / hc.MANIFEST_NAME).read_text(encoding="utf-8")) \
        if (good / hc.MANIFEST_NAME).exists() else {}
    check(rc == 0 and out.get("rms_keys") == hc.rms_keys_of(rms) and
          out["gates"]["structural_zero_channels"] == {"L01.mlp_in.hess": [DEAD]} and
          "L01.mlp_in.rms.hess" in out["files"] and out["rms_capture"]["gates"]["ok"],
          "(g) --regate promotes a --rms-taps set: rms_keys and the rms files kept; the dead channel "
          "exempted on the post-norm file only")

    bad = failed_set("regate_bad")
    h, r_rows, _ = hc.read_hess_file(bad / "L01.mlp_in.rms.hess")
    h[DEAD, :] = 0.0
    h[:, DEAD] = 0.0
    hc.write_hess_file(bad / "L01.mlp_in.rms.hess", h, r_rows)
    rc = hc.regate(bad, model_dir)
    check(rc == 1 and not (bad / hc.MANIFEST_NAME).exists(),
          "(g) --regate refuses an rms file that is zero on the dead channel (no exemption for rms)")

    lying = failed_set("regate_lying")
    d = json.loads((lying / hc.FAILED_MANIFEST_NAME).read_text(encoding="utf-8"))
    d["rms_capture"]["gates"]["ok"] = False
    (lying / hc.FAILED_MANIFEST_NAME).write_text(json.dumps(d), encoding="utf-8")
    raises(SystemExit, "rms_capture.gates.ok", lambda: hc.regate(lying, model_dir),
           "(g) --regate refuses rms_keys whose rms gates did not pass")


# ---- (f) -----------------------------------------------------------------------------------------


def test_manifest_rules(tmp: Path) -> None:
    check(hc.rms_file_name("L07.mlp_in.hess") == "L07.mlp_in.rms.hess", "(f) rms_file_name pairs by name")
    raises(ValueError, "not a post-norm", lambda: hc.rms_file_name("L07.mlp_in.rms.hess"),
           "(f) rms_file_name refuses an rms file")
    raises(SystemExit, "not an in-projection",
           lambda: hc.build_rms_plans({"L00.in.hess": ["text.layers.0.gdn.out_proj"]}, 1, HIDDEN),
           "(f) build_rms_plans refuses a key that norm does not feed")
    raises(SystemExit, "not an in-projection",
           lambda: hc.build_rms_plans({"L00.mlp_in.hess": ["text.layers.1.mlp.gate_up"]}, 1, HIDDEN),
           "(f) build_rms_plans refuses another layer's key")
    check(hc.build_rms_plans({"L00.in.hess": ["text.layers.0.gdn.in_proj_qkv"]}, 0, HIDDEN) == [],
          "(f) build_rms_plans honours --layers")

    d = tmp / "rules"
    d.mkdir()
    files = {"L00.in.hess": {"K": 8, "rows": 64, "trace": 1.0}, "L00.mlp_in.hess": {"K": 8, "rows": 64, "trace": 1.0},
             "L00.in.rms.hess": {"K": 8, "rows": 64, "trace": 1.0}}
    keys = {"text.layers.0.gdn.in_proj_qkv": "L00.in.hess", "text.layers.0.mlp.gate_up": "L00.mlp_in.hess"}
    for label, rk in (("the wrong tap's rms file", {"text.layers.0.mlp.gate_up": "L00.in.rms.hess"}),
                      ("a post-norm file", {"text.layers.0.gdn.in_proj_qkv": "L00.mlp_in.hess"}),
                      ("an unlisted file", {"text.layers.0.mlp.gate_up": "L00.mlp_in.rms.hess"}),
                      ("a base without keys", {"text.layers.0.attn.qg": "L00.in.rms.hess"})):
        raises(RuntimeError, "rms_keys", lambda rk=rk: hc.write_manifest(d, files, keys, {}, rms_keys=rk),
               f"(f) write_manifest refuses rms_keys naming {label}")
    raises(RuntimeError, "rms_keys=", lambda: hc.write_manifest(d, files, keys, {"rms_keys": {}}),
           "(f) write_manifest refuses rms_keys smuggled in as provenance")
    p = hc.write_manifest(d, files, keys, {}, rms_keys={"text.layers.0.gdn.in_proj_qkv": "L00.in.rms.hess"})
    check(json.loads(p.read_bytes())["rms_keys"] == {"text.layers.0.gdn.in_proj_qkv": "L00.in.rms.hess"},
          "(f) write_manifest writes a correctly paired rms_keys")

    code = {"source": "code", "file_list_method": "git ls-files", "sha256_of_concatenation": "a" * 64,
            "tokens": 32768, "sequences": 16, "window_starts": [0, 5], "file_list": ["x.cpp"]}
    calib = {"source": "calib", "name": "calib", "path": "tools/reference/calib.txt", "sha256": "b" * 64,
             "tokens": 2048, "sequences": 1}
    rec = {"sequences": 17, "tokens": 34816, "seq_len": 2048, "sources": [calib, code]}
    check(hc.corpus_mismatches(rec, [dict(calib), dict(code, file_list_method="git ls-tree abc (--code-rev abc)")],
                               17, 34816, 2048) == [],
          "(f) corpus_mismatches: identical corpus (file_list_method ignored) -> no difference")
    diffs = hc.corpus_mismatches(rec, [calib, dict(code, sha256_of_concatenation="c" * 64)], 17, 34816, 2048)
    check(len(diffs) == 1 and "sha256_of_concatenation" in diffs[0],
          "(f) corpus_mismatches reports a changed code concatenation")
    check(len(hc.corpus_mismatches(rec, [calib], 16, 32768, 1024)) == 4,
          "(f) corpus_mismatches reports seq_len, sequences, tokens and the source count")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="r4dx_test_hessian_rms_"))
    try:
        for fn in (test_formula, lambda: test_manifest_rules(tmp), lambda: test_capture(tmp)):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                check(False, f"unexpected exception in {getattr(fn, '__name__', 'test')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check(not torch.cuda.is_initialized(), "no CUDA/HIP context was created")
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
