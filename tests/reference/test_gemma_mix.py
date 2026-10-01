"""tests/reference/test_gemma_mix.py -- the Gemma 4 mix4.5m ranking and its budget math
(tools/reference/gemma_mix.py, docs/gemma4-plan.md 11). CPU only; the real 677-tensor header
(tools/reference/gemma/tensor_names.json) supplies the shapes, so the byte budget is checked on the real
Gemma linears (40 sliding x 7 + 8 full x 6 = 328 tensors, no v_proj on the full layers).

    (a) the header's linear numel equals an independent closed form from the config dims (3840 hidden, 15360
        ffn, sliding q 4096 / kv 2048, full q 8192 / k 512)
    (b) allocation at 4.5: only K4 / K5, every qgroup one rate, sum(numel*K) <= int(4.5 * total) and within one
        group of it (the budget is used), exact average printed; 4.0 and 5.0 targets are uniform
    (c) ranking: the prior-only scores order as documented (full amp layer > edge > amp > plain), a data gap
        overrides the prior, a zero gap sinks a group, determinism
    (d) trellis_quant wiring (needs torch, CPU build is enough): Qwen's allocate_ranked IS allocate; the
        Gemma `mix` CLI path with fake K4/K5 manifests picks the groups by proxy gap, records the ranking, and
        `--ranking exl3` reproduces EXL3's order; a Qwen manifest's mix keeps its old fields

Run with any python 3 that has torch (CPU build is fine); the GPUs are hidden below. Exits 0 and prints
"OK (<n> checks)", else 1.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"   # before torch is imported anywhere
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))

import gemma_mix as gm  # noqa: E402

CHECKS = 0
FAILS: list = []


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print(f"FAIL: {msg}")


# the allocator's rate steps, copied from trellis_quant (d) asserts they agree
def rate_floor(bpw):
    import math
    r = math.floor(2 * bpw) / 2 if bpw < 4 else math.floor(bpw)
    return float(min(max(r, 1), 8))


def rate_next(r):
    nr = r + (0.5 if r < 4 else 1.0)
    return None if nr > 8 else nr


MODS_SLIDING = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
QGROUP = {"self_attn.q_proj": "self_attn.qkv", "self_attn.k_proj": "self_attn.qkv",
          "self_attn.v_proj": "self_attn.qkv", "self_attn.o_proj": "self_attn.o", "mlp.gate_proj": "mlp.gu",
          "mlp.up_proj": "mlp.gu", "mlp.down_proj": "mlp.d"}


def real_linears() -> list:
    """The records trellis_quant.model_linears returns, built from the real header."""
    hdr = json.loads((REPO_ROOT / "tools" / "reference" / "gemma" / "tensor_names.json").read_text("utf-8"))["tensors"]
    out = []
    for layer in range(48):
        full = (layer + 1) % 6 == 0
        lt = "full_attention" if full else "sliding_attention"
        mods = [m for m in MODS_SLIDING if not (full and m == "self_attn.v_proj")]
        for idx, m in enumerate(mods):
            name = f"model.language_model.layers.{layer}.{m}.weight"
            n, k = hdr[name]["shape"]
            out.append({"name": name, "layer": layer, "idx": idx, "module": m, "layer_type": lt,
                        "qgroup": QGROUP[m], "n": n, "k": k, "numel": n * k})
        # v_proj absent exactly on the full layers
        check((f"model.language_model.layers.{layer}.self_attn.v_proj.weight" in hdr) == (not full),
              f"v_proj presence at layer {layer}")
    return out


def main_checks() -> None:
    lin = real_linears()
    H, F = 3840, 15360
    sliding = 4096 * H + 2 * 2048 * H + H * 4096 + 3 * H * F
    full = 8192 * H + 512 * H + H * 8192 + 3 * H * F
    total = sum(t["numel"] for t in lin)
    check(len(lin) == 40 * 7 + 8 * 6, f"328 linears, got {len(lin)}")
    check(total == 40 * sliding + 8 * full, f"total numel {total} vs closed form {40 * sliding + 8 * full}")

    # (b) the budget
    budget = int(4.5 * total)
    rates, report = gm.allocate(lin, 4.5, rate_floor, rate_next)
    check(set(rates.values()) == {4.0, 5.0}, f"only K4/K5, got {sorted(set(rates.values()))}")
    used = sum(t["numel"] * rates[t["name"]] for t in lin)
    maxgrp = max(g["numel"] for g in report)
    check(used <= budget, f"used {used} exceeds budget {budget}")
    check(budget - used < maxgrp, f"budget not used: slack {budget - used} >= largest group {maxgrp}")
    avg = gm.avg_rate(lin, rates)
    check(4.5 - maxgrp / total < avg <= 4.5, f"average {avg}")
    for g in report:
        check(len(g["K"]) == 1, f"group {g['layer']}/{g['qgroup']} split across rates {g['K']}")
    # every group promotes as a unit and the K5 numel is exactly what the average says
    k5 = sum(t["numel"] for t in lin if rates[t["name"]] == 5.0)
    check(abs(avg - (4.0 + k5 / total)) < 1e-12, "average = 4 + K5 fraction")
    print(f"  4.5 target: avg {avg:.6f} bits over {total} weights, K5 fraction {k5 / total:.4%} "
          f"({sum(1 for g in report if g['K'] == [5.0])}/{len(report)} groups)")
    check(len(report) == 48 * 4, f"4 qgroups per layer (qkv, o, gu, d), got {len(report)}")
    for bpw, want in ((4.0, 4.0), (5.0, 5.0)):
        r, _ = gm.allocate(lin, bpw, rate_floor, rate_next)
        check(set(r.values()) == {want}, f"bpw {bpw} is uniform K{want:g}")
    r, _ = gm.allocate(lin, 4.5, rate_floor, rate_next)
    check(r == rates, "deterministic")

    # (c) ranking
    sc = gm.group_scores(lin, None, 48)
    check(sc[(41, "self_attn.qkv")] > sc[(11, "self_attn.qkv")] > sc[(12, "self_attn.qkv")],
          "prior: full amp layer 41 > full plain 11 > sliding in-band 12")
    check(sc[(41, "self_attn.qkv")] > sc[(40, "self_attn.qkv")], "amp layer above its neighbour")
    check(sc[(0, "mlp.gu")] > sc[(3, "mlp.gu")], "edge layer 0 above layer 3")
    check(sc[(10, "mlp.gu")] > sc[(12, "mlp.gu")] and sc[(23, "mlp.d")] > sc[(22, "mlp.d")], "amp sites above neighbours")
    check(sc[(12, "mlp.d")] > sc[(12, "mlp.gu")], "down above gate/up at the same layer")
    check(gm.kind_weight("self_attn.k_proj", "full_attention") == 1.5 and
          gm.kind_weight("self_attn.k_proj", "sliding_attention") == 1.0, "k_proj weight only on full layers")
    top = max(sc, key=sc.get)
    check(rates[next(t["name"] for t in lin if (t["layer"], t["qgroup"]) == top)] == 5.0, "top group is K5")
    # prior-only: the amp layers' MLPs (3 big tensors) are not starved by cheap attention groups
    check(all(rates[f"model.language_model.layers.{l}.mlp.down_proj.weight"] == 5.0 for l in (10, 23, 29, 41)),
          "amp-site down_proj at K5 in the prior-only mix")

    gaps = {t["name"]: 1.0 for t in lin}
    for t in lin:
        if t["layer"] == 20 and t["module"].startswith("mlp"):
            gaps[t["name"]] = 100.0  # data says layer 20's MLP is very hard
        if t["layer"] == 41:
            gaps[t["name"]] = 0.0    # and layer 41 gains nothing
    r2, rep2 = gm.allocate(lin, 4.5, rate_floor, rate_next, gaps)
    check(all(r2[t["name"]] == 5.0 for t in lin if t["layer"] == 20 and t["module"].startswith("mlp")),
          "a large proxy gap promotes layer 20's MLP")
    check(all(r2[t["name"]] == 4.0 for t in lin if t["layer"] == 41), "a zero gap leaves layer 41 at K4 despite its prior")
    check(gm.avg_rate(lin, r2) <= 4.5, "data-driven mix within budget")
    check(rep2[0]["layer"] == 20, "data-driven order starts with the high-gap group")


def tq_checks() -> None:
    try:
        import torch  # noqa: F401
        import trellis_quant as tq
        from arch_table import GEMMA, QWEN
    except ImportError as e:
        print(f"SKIP (d): {e}")
        return
    import torch
    check(not torch.cuda.is_available(), "no GPU visible")
    check(gm.AMP_LAYERS == {10, 23, 29, 41}, "amp sites")
    for bpw in (3.5, 4.0, 4.5, 5.0, 6.3):
        check(rate_floor(bpw) == tq.rate_floor(bpw), f"rate_floor copy at {bpw}")
    check(all(rate_next(r) == tq.rate_next(r) for r in (3.0, 3.5, 4.0, 7.0, 8.0)), "rate_next copy")

    lin = real_linears()
    # Qwen path: allocate_ranked(QWEN) is allocate, byte for byte, and Gemma + exl3 too
    check(tq.allocate_ranked(lin, 4.5, QWEN)[0] == tq.allocate(lin, 4.5), "qwen arch -> EXL3 allocate")
    r, rep, used = tq.allocate_ranked(lin, 4.5, GEMMA, ranking="exl3")
    check(r == tq.allocate(lin, 4.5) and rep is None and used == "exl3", "--ranking exl3 = allocate")
    r, rep, used = tq.allocate_ranked(lin, 4.5, GEMMA)
    check(used == "gemma-prior" and rep, "gemma auto without data = prior")
    try:
        tq.allocate_ranked(lin, 4.5, QWEN, ranking="gemma")
        check(False, "gemma ranking on a qwen arch must raise")
    except ValueError:
        check(True, "")

    # the mix CLI on fake K4 / K5 manifests (model_linears is swapped for the real-shape records)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        man = {}
        for K in (4, 5):
            tens = {}
            for t in lin:
                gap_hard = t["layer"] == 7 and t["module"].startswith("mlp")
                p4 = 0.05 if gap_hard else 0.01
                tens[t["name"]] = {"encoding": tq.ENCODING_TRELLIS, "K": float(K), "k": t["k"], "n": t["n"],
                                   "bits": {"total": int(t["numel"] * (K + 0.01)), "trellis": int(t["numel"] * K)},
                                   "proxy": p4 if K == 4 else p4 / 4, "file": "x.safetensors"}
            d = td / f"K{K}"
            d.mkdir()
            doc = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
                   "complete": True, "model_dir": "m", "config_sha256": "c", "hessian_manifest_sha256": "h",
                   "hessian_basis": "matched", "codebook": "mul1", "arch": "gemma4_unified",
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
            check(m["allocation"].get("ranking") == "gemma-proxy", f"ranking recorded: {m['allocation']}")
            kk = {n: r["K"] for n, r in m["tensors"].items()}
            check(all(kk[t["name"]] == 5.0 for t in lin if t["layer"] == 7 and t["module"].startswith("mlp")),
                  "hard (high proxy gap) MLP of layer 7 is K5")
            check((td / "mix" / "mix_ranking.json").is_file(), "mix_ranking.json written")
            check(abs(sum(t["numel"] * kk[t["name"]] for t in lin) / sum(t["numel"] for t in lin) - 4.5) < 0.05,
                  "mix average ~4.5")
            A.ranking, A.out_dir = "exl3", td / "mix_exl3"
            check(tq.cmd_mix(A) == 0, "exl3 mix returns 0")
            m2 = json.loads((td / "mix_exl3" / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
            check("ranking" not in m2["allocation"] and m2["allocation"]["rule"].startswith("docs/trellis.md 9"),
                  "exl3 mix keeps the Qwen rule text and fields")
            ex = tq.allocate(lin, 4.5)
            check({n: r["K"] for n, r in m2["tensors"].items()} == ex, "exl3 mix == allocate")
            A.ranking, A.out_dir = "prior", td / "mix_prior"
            check(tq.cmd_mix(A) == 0, "prior mix returns 0")
            m3 = json.loads((td / "mix_prior" / tq.MANIFEST_NAME).read_text(encoding="utf-8"))
            check(m3["allocation"]["ranking"] == "gemma-prior", "prior mix recorded")
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
