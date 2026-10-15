"""tests/reference/test_lfm2_mix.py (ctest `reference_lfm2_mix`) -- the LFM2 mix4.5 ranking and its budget math
(tools/reference/lfm2_mix.py, docs/lfm2-plan.md 7.1). CPU only. The shapes are the REAL model's (40 layers: hidden 2048,
32 q heads / 8 kv heads x head_dim 64, ShortConv in_proj 6144, dense I 11776 on layers 0-1, 64 experts of I 1536 on layers
2-39), built from the config's dims; when the checkpoint's headers are on disk (E:\\models\\LiquidAI\\LFM2.5-24B-A2B) the
closed form is checked against them tensor by tensor.

    (a) the linear list: 10 attention + 30 ShortConv layers, 2 dense + 38 MoE, 4864 expert tensors x ... and the total
        numel equal the closed form (experts 22.95 B = 95.7 % of the weights)
    (b) allocation at 4.5: every non-expert linear K5, experts only K4 / K5, each (layer, group) ONE rate, sum(numel*K)
        <= int(4.5 * total) and within the largest group of it, the experts' average ~4.48; 4.0 is refused (the K5 floor
        alone costs more), 5.0 is uniform K5; deterministic
    (c) priors: first MoE layers (2, 3) and the last two layers above their neighbours, an attention layer above a
        ShortConv layer, down above gate_up; a proxy gap overrides the prior, a zero gap sinks a group
    (d) trellis_quant wiring (torch, CPU): allocate_ranked(LFM2) is lfm2_mix, `--ranking lfm2` without data raises, the
        `mix` CLI over fake K4 / K5 manifests picks the high-gap group and records the ranking

Run with any python 3 that has torch (CPU build is fine). Exits 0 and prints "OK (<n> checks)", else 1.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"   # before torch is imported anywhere
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))

import lfm2_mix as lm  # noqa: E402
from arch_table import LFM2_MODEL_DIR, lfm2_arch  # noqa: E402

CHECKS = 0
FAILS: list = []
H, I_DENSE, I_MOE, E, LAYERS, VOCAB = 2048, 11776, 1536, 64, 40, 128000
ATTN_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print(f"FAIL: {msg}")


def rate_floor(bpw):
    r = math.floor(2 * bpw) / 2 if bpw < 4 else math.floor(bpw)
    return float(min(max(r, 1), 8))


def rate_next(r):
    nr = r + (0.5 if r < 4 else 1.0)
    return None if nr > 8 else nr


SHAPES = {  # module -> (n, k) of the real model
    "self_attn.q_proj": (2048, H), "self_attn.k_proj": (512, H), "self_attn.v_proj": (512, H),
    "self_attn.out_proj": (H, 2048), "conv.in_proj": (3 * H, H), "conv.out_proj": (H, H),
    "feed_forward.w1": (I_DENSE, H), "feed_forward.w3": (I_DENSE, H), "feed_forward.w2": (H, I_DENSE),
}


def real_linears() -> list:
    """The records trellis_quant.model_linears returns, from the real config's dims."""
    arch = lfm2_arch(E)
    cfg = {"layer_types": ["full_attention" if i in ATTN_LAYERS else "conv" for i in range(LAYERS)],
           "num_dense_layers": 2}
    out = []
    for layer, kind in enumerate(arch.layer_kinds(cfg)):
        for idx, m in enumerate(arch.module_list(kind)):
            if m.startswith("feed_forward.experts."):
                n, k = (H, I_MOE) if m.endswith(".w2") else (I_MOE, H)
            else:
                n, k = SHAPES[m]
            out.append({"name": arch.hf_name(layer, m), "layer": layer, "idx": idx, "module": m, "layer_type": kind,
                        "qgroup": arch.qgroups[m], "n": n, "k": k, "numel": n * k})
    return out


def main_checks() -> None:
    lin = real_linears()
    n_exp = [t for t in lin if lm.is_expert(t)]
    total = sum(t["numel"] for t in lin)
    attn = 2048 * H + 512 * H + 512 * H + H * 2048            # 10.49 M
    conv = 3 * H * H + H * H                                  # 16.78 M
    dense = 3 * I_DENSE * H                                   # 72.35 M
    exp = 38 * E * 3 * I_MOE * H                              # 22.95 B
    closed = 10 * attn + 30 * conv + 2 * dense + exp
    check(len(n_exp) == 38 * E * 3, f"{38 * E * 3} expert tensors, got {len(n_exp)}")
    check(total == closed, f"total numel {total} vs closed form {closed}")
    check(abs(sum(t['numel'] for t in n_exp) / total - 0.9696) < 2e-3,
          f"experts are ~97 % of the quantized weights, got {sum(t['numel'] for t in n_exp) / total:.4f}")
    check(abs(exp - 22.95e9) < 0.01e9, f"experts 22.95 B, got {exp / 1e9:.3f}")
    # the real header, tensor by tensor (skipped when the checkpoint is not on disk)
    idx = Path(LFM2_MODEL_DIR) / "model.safetensors.index.json"
    if idx.is_file():
        from common import raw_safetensors_header

        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        hdrs = {}
        bad = 0
        for t in lin:
            f = wm[t["name"]]
            if f not in hdrs:
                hdrs[f] = raw_safetensors_header(Path(LFM2_MODEL_DIR) / f)[0]
            if list(hdrs[f][t["name"]]["shape"]) != [t["n"], t["k"]]:
                bad += 1
        check(bad == 0, f"{bad} tensor(s) whose real header shape differs from the config's closed form")
        print(f"  real headers: {len(lin)} linears checked against {LFM2_MODEL_DIR}")
    else:
        print(f"  SKIP header check: {idx} not found")

    # (b) the budget
    budget = int(4.5 * total)
    rates, report = lm.allocate(lin, 4.5, rate_floor, rate_next)
    check(all(rates[t["name"]] == 5.0 for t in lin if not lm.is_expert(t)), "every non-expert linear at K5")
    check(set(rates[t["name"]] for t in n_exp) == {4.0, 5.0}, "experts at K4 and K5 only")
    used = sum(t["numel"] * rates[t["name"]] for t in lin)
    maxgrp = max(g["numel"] for g in report)
    check(used <= budget, f"used {used} exceeds budget {budget}")
    check(budget - used < maxgrp, f"budget not used: slack {budget - used} >= largest group {maxgrp}")
    avg = lm.avg_rate(lin, rates)
    check(4.5 - maxgrp / total < avg <= 4.5, f"average {avg}")
    exp_avg = sum(t["numel"] * rates[t["name"]] for t in n_exp) / sum(t["numel"] for t in n_exp)
    check(4.46 < exp_avg < 4.5, f"experts' average {exp_avg:.4f} should be about 4.48")
    for g in report:
        check(len(g["K"]) == 1, f"group {g['layer']}/{g['qgroup']} split across rates {g['K']}")
    check(len(report) == 38 * 2, f"2 expert groups per MoE layer, got {len(report)}")
    check(all(g["n_members"] == (2 * E if g["qgroup"] == "moe.gu" else E) for g in report), "group sizes")
    k5 = sum(1 for g in report if g["K"] == [5.0])
    print(f"  4.5 target: avg {avg:.6f} bits over {total} weights, experts avg {exp_avg:.4f}, "
          f"{k5}/{len(report)} expert groups at K5, slack {budget - used} of {maxgrp}")
    try:
        lm.allocate(lin, 4.0, rate_floor, rate_next)
        check(False, "4.0 with a K5 non-expert floor must be refused")
    except ValueError:
        check(True, "")
    r5, _ = lm.allocate(lin, 5.0, rate_floor, rate_next)
    check(set(r5.values()) == {5.0}, "5.0 target is uniform K5")
    r, _ = lm.allocate(lin, 4.5, rate_floor, rate_next)
    check(r == rates, "deterministic")

    # (c) ranking
    sc = lm.group_scores(lin, None, LAYERS)
    check(sc[(2, "moe.gu")] > sc[(4, "moe.gu")] and sc[(3, "moe.gu")] > sc[(4, "moe.gu")], "layers 2 and 3 above layer 4")
    check(sc[(39, "moe.gu")] > sc[(37, "moe.gu")] and sc[(38, "moe.gu")] > sc[(37, "moe.gu")], "last two layers above 37")
    check(sc[(6, "moe.gu")] > sc[(5, "moe.gu")] and sc[(6, "moe.gu")] > sc[(7, "moe.gu")], "attention layer above ShortConv")
    check(sc[(20, "moe.d")] > sc[(20, "moe.gu")], "down above gate_up at the same layer")
    top = max(sc, key=sc.get)
    check(top == (2, "moe.d") or top == (38, "moe.d"), f"top group is a first-MoE / last attention-layer down, got {top}")
    check(rates[f"model.layers.38.feed_forward.experts.0.w2.weight"] == 5.0
          and rates["model.layers.2.feed_forward.experts.63.w2.weight"] == 5.0, "the top downs are K5 in the prior-only mix")
    gaps = {t["name"]: 1.0 for t in lin}
    for t in lin:
        if t["layer"] == 20 and lm.is_expert(t):
            gaps[t["name"]] = 100.0   # data says layer 20's experts are very hard
        if t["layer"] == 38:
            gaps[t["name"]] = 0.0     # and layer 38 gains nothing
    r2, rep2 = lm.allocate(lin, 4.5, rate_floor, rate_next, gaps)
    check(all(r2[t["name"]] == 5.0 for t in lin if t["layer"] == 20 and lm.is_expert(t)), "a large proxy gap promotes layer 20")
    check(all(r2[t["name"]] == 4.0 for t in lin if t["layer"] == 38 and lm.is_expert(t)), "a zero gap leaves layer 38 at K4")
    check(lm.avg_rate(lin, r2) <= 4.5, "data-driven mix within budget")
    check(rep2[0]["layer"] == 20, "data-driven order starts with the high-gap group")


def tq_checks() -> None:
    try:
        import torch  # noqa: F401
        import trellis_quant as tq
        from arch_table import LFM2, QWEN
    except ImportError as e:
        print(f"SKIP (d): {e}")
        return
    check(not torch.cuda.is_available(), "no GPU visible")
    for bpw in (3.5, 4.0, 4.5, 5.0, 6.3):
        check(rate_floor(bpw) == tq.rate_floor(bpw), f"rate_floor copy at {bpw}")
    check(all(rate_next(r) == tq.rate_next(r) for r in (3.0, 3.5, 4.0, 7.0, 8.0)), "rate_next copy")
    lin = real_linears()
    r, rep, used = tq.allocate_ranked(lin, 4.5, LFM2)
    check(used == "lfm2-prior" and rep and r == lm.allocate(lin, 4.5, tq.rate_floor, tq.rate_next)[0], "lfm2 auto = lfm2_mix prior")
    try:
        tq.allocate_ranked(lin, 4.5, QWEN, ranking="lfm2")
        check(False, "lfm2 ranking on a qwen arch must raise")
    except ValueError:
        check(True, "")
    check(tq.parse_hf_name("model.layers.7.feed_forward.experts.12.w2.weight") == (7, "feed_forward.experts.*.w2"),
          "parse_hf_name collapses the expert index")
    check(tq.parse_hf_name("model.language_model.layers.7.mlp.down_proj.weight") == (7, "mlp.down_proj"), "qwen name unchanged")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for K in (4, 5):
            tens = {}
            for t in lin:
                hard = t["layer"] == 7 and lm.is_expert(t)
                p4 = 0.05 if hard else 0.01
                tens[t["name"]] = {"encoding": tq.ENCODING_TRELLIS, "K": float(K), "k": t["k"], "n": t["n"],
                                   "bits": {"total": int(t["numel"] * (K + 0.01)), "trellis": int(t["numel"] * K)},
                                   "proxy": p4 if K == 4 else p4 / 4, "file": "x.safetensors"}
            d = td / f"K{K}"
            d.mkdir()
            doc = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
                   "complete": True, "model_dir": "m", "config_sha256": "c", "hessian_manifest_sha256": "h",
                   "hessian_basis": "matched", "codebook": "mul1", "arch": "lfm2_moe",
                   "recipe": {}, "hessian_dir": "hd", "tensors": tens, "code_sha256": "s"}
            (d / tq.MANIFEST_NAME).write_text(json.dumps(doc), encoding="utf-8")
        orig = tq.model_linears
        tq.model_linears = lambda model_dir, arch=None: [dict(t) for t in lin]
        try:
            class A:
                bpw = 4.5
                src = [str(td / "K4"), str(td / "K5")]
                ranking = "auto"
                out_dir = td / "mix"
            check(tq.cmd_mix(A) == 0, "mix returns 0")
            m = json.loads((td / "mix" / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
            check(m["allocation"].get("ranking") == "lfm2-proxy", f"ranking recorded: {m['allocation'].get('ranking')}")
            check(m["allocation"]["rule"].startswith("tools/reference/lfm2_mix.py"), "rule text")
            kk = {n: r["K"] for n, r in m["tensors"].items()}
            check(all(kk[t["name"]] == 5.0 for t in lin if t["layer"] == 7 and lm.is_expert(t)), "high-gap layer 7 experts K5")
            check(all(kk[t["name"]] == 5.0 for t in lin if not lm.is_expert(t)), "non-experts K5 in the mix")
            check(m["arch"] == "lfm2_moe", "arch carried")
            check((td / "mix" / "mix_ranking.json").is_file(), "mix_ranking.json written")
            avg = sum(t["numel"] * kk[t["name"]] for t in lin) / sum(t["numel"] for t in lin)
            check(4.45 < avg <= 4.5, f"mix average {avg:.4f}")
            A.ranking, A.out_dir = "prior", td / "mix_prior"
            check(tq.cmd_mix(A) == 0, "prior mix returns 0")
            m3 = json.loads((td / "mix_prior" / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
            check(m3["allocation"]["ranking"] == "lfm2-prior", "prior mix recorded")
        finally:
            tq.model_linears = orig


if __name__ == "__main__":
    try:
        main_checks()
        tq_checks()
    except Exception:
        traceback.print_exc()
        FAILS.append("exception")
    if FAILS:
        print(f"{len(FAILS)} failed of {CHECKS}")
        raise SystemExit(1)
    print(f"OK ({CHECKS} checks)")
