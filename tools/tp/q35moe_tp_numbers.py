#!/usr/bin/env python
"""CPU-only numbers behind docs/qwen35moe-tp-design.md (no GPU, no model load).

Reads the shipped allocation (E:\\models\\r4dx\\q35moe\\mix\\allocation.json: per-layer KB of the three classes) and the
Hessian pass's per-expert routed counts (E:\\models\\r4dx\\q35moe\\hessian\\L??\\routing_counts.json: 1,003,172 calibration
tokens, top-8 of 256, per layer) and prints markdown tables:

  1. a byte model of the container (checked against the allocation's own per-layer GiB),
  2. per-rank weight bytes at --tp 2 with every expert TP-split (gate|up rows halves 1024 -> 512, down K 1024 -> 512),
  3. bytes read per decode token per rank: TP-split experts vs expert parallelism (EP) with real routing statistics,
  4. distinct experts touched by an n-row step (n = 16 DFlash2 verify, 64, 256 prefill chunk) and the per-rank bytes.

The routing model is a Plackett-Luce top-8 draw whose item weights are fitted so the per-expert inclusion probability
equals the measured routed share (the real router's co-selection correlations and the correlation between adjacent
tokens are NOT modelled: distinct-expert counts for n rows are therefore an UPPER bound, expert-parallel imbalance is
approximate). Run:  python -I tools\\tp\\q35moe_tp_numbers.py [--root E:\\models\\r4dx\\q35moe]
"""
import argparse
import json
import os

import numpy as np

H = 3072          # hidden
L = 48            # layers
E = 256           # experts
TOPK = 8
INTER = 1024      # routed and shared expert width
VOCAB = 248320
GDN_KEY_HEADS, GDN_VAL_HEADS, GDN_DIM = 16, 64, 128
ATT_Q, ATT_KV, ATT_D = 32, 2, 256
GIB = float(1 << 30)


def lin(N, K, kb, parts=1, world=1, split=None):
    """Device bytes of one trellis linear (words + fp32 sign vectors) for one rank.
    split: None (replicated), 'rows' (N split in `world` equal parts, per segment) or 'cols' (K split)."""
    if split == "rows":
        n, k = N // world, K
        suh, svh = parts * K, N // world          # suh replicated, svh cut by the rank's rows
    elif split == "cols":
        n, k = N, K // world
        suh, svh = K // world, N                  # suh cut by K, svh replicated
    else:
        n, k, suh, svh = N, K, parts * K, N
    return n * k * kb // 8 + 4 * (suh + svh)


def bf16(n_elems, world=1, split=None):
    return 2 * n_elems // (world if split else 1)


def layer_bytes(layer, kb, world=1, decode_experts=None):
    """Bytes of one layer on one rank. decode_experts: None = all 256 experts resident (weights),
    else the number of routed experts whose weights one token reads."""
    gdn = layer % 4 != 3
    kb_p, kb_gu, kb_dn = kb
    w = world
    s = "rows" if w > 1 else None
    c = "cols" if w > 1 else None
    out = {}
    if gdn:
        qkv = (GDN_KEY_HEADS * GDN_DIM) * 2 + GDN_VAL_HEADS * GDN_DIM
        val = GDN_VAL_HEADS * GDN_DIM
        mix = lin(qkv, H, kb_p, 1, w, s) + lin(val, H, kb_p, 1, w, s) + lin(H, val, kb_p, 1, w, c)
        mix += bf16(2 * GDN_VAL_HEADS * H, w, s) + bf16(qkv * 4, w, s) + 4 * 2 * GDN_VAL_HEADS // (w if w > 1 else 1)
        mix += 4 * GDN_DIM
    else:
        qg = 2 * ATT_Q * ATT_D
        kv = ATT_KV * ATT_D
        mix = lin(qg, H, kb_p, 1, w, s) + 2 * lin(kv, H, kb_p, 1, w, s) + lin(H, ATT_Q * ATT_D, kb_p, 1, w, c)
        mix += bf16(2 * ATT_D) + 4 * ATT_KV // (w if w > 1 else 1)
    out["mixer"] = mix
    out["norms"] = 2 * 2 * H
    router = bf16(E * H)                                    # replicated
    shared = 2 * bf16(INTER * H, w, s) + bf16(H * INTER, w, c) + bf16(H)
    out["router+shared"] = router + shared
    gu = lin(2 * INTER, H, kb_gu, 2, w, s)
    dn = lin(H, INTER, kb_dn, 1, w, c)
    n_exp = E if decode_experts is None else decode_experts
    out["experts"] = n_exp * (gu + dn)
    out["expert_gu"], out["expert_dn"] = gu, dn
    return out


def total(lb):
    return lb["mixer"] + lb["norms"] + lb["router+shared"] + lb["experts"]


def fit_plackett_luce(pi, rng, n=20000, iters=4):
    """Weights w so a top-8 Gumbel draw (Plackett-Luce) includes expert e with probability ~ pi[e]."""
    logw = np.log(np.maximum(pi, 1e-6))
    for _ in range(iters):
        g = rng.gumbel(size=(n, E)) + logw
        top = np.argpartition(-g, TOPK, axis=1)[:, :TOPK]
        hit = np.bincount(top.ravel(), minlength=E) / n
        logw += 0.8 * (np.log(np.maximum(pi, 1e-6)) - np.log(np.maximum(hit, 1e-6)))
    return logw


def draw(logw, rng, n):
    g = rng.gumbel(size=(n, E)) + logw
    return np.argpartition(-g, TOPK, axis=1)[:, :TOPK]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\models\r4dx\q35moe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3000)  # n = 1 draws; 1/3 of it at n = 16, 1/10 at 64, 1/50 at 256
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    alloc = json.load(open(os.path.join(a.root, "mix", "allocation.json")))
    kbs = [(int(p["K"]["p"]), int(p["K"]["gu"]), int(p["K"]["down"])) for p in alloc["per_layer"]]
    assert len(kbs) == L

    # ---- 1. byte model vs the allocator's own per-layer GiB ------------------------------------------------------------
    print("### 1. Byte model of the container (TP=1) vs allocation.json\n")
    worst = 0.0
    tot = 0.0
    for i in range(L):
        b = total(layer_bytes(i, kbs[i])) / GIB
        tot += b
        worst = max(worst, abs(b / alloc["per_layer"][i]["gib"] - 1))
    lm_head = (VOCAB * H // 2 + VOCAB * (H // 32) * 4) / GIB
    print(f"layers {tot:.3f} GiB + lm_head (w4a16 g32) {lm_head:.3f} GiB; allocation.json total {alloc['total_gib']:.3f} GiB "
          f"(layer sum {sum(p['gib'] for p in alloc['per_layer']):.3f}); worst per-layer deviation {worst * 100:.2f}% "
          f"(the model matches the allocator's byte count)\n")

    # ---- 2. per-rank weights at --tp 2 -------------------------------------------------------------------------------
    print("### 2. Weights per rank, every linear split (experts TP-split), TP=2\n")
    one = sum(total(layer_bytes(i, kbs[i])) for i in range(L))
    two = sum(total(layer_bytes(i, kbs[i], world=2)) for i in range(L))
    repl = sum(layer_bytes(i, kbs[i], world=2)["norms"] + bf16(E * H) for i in range(L))
    lm2 = lm_head / 2
    print(f"| | GiB |\n|---|--:|\n| layers, TP=1 | {one / GIB:.3f} |\n| layers, one rank at TP=2 | {two / GIB:.3f} |")
    print(f"| of which replicated by design (norms + router) | {repl / GIB:.3f} |")
    print(f"| lm_head one rank (vocab split) | {lm2:.3f} |")
    print(f"| **weights one rank** | **{two / GIB + lm2:.3f}** |")
    gu_sign_repl = sum(E * 4 * 2 * H for _ in range(L)) / GIB
    print(f"| (gate_up sign vector suh, replicated on both ranks: {gu_sign_repl:.3f} GiB) | |\n")

    # ---- 3/4. decode and n-row steps ----------------------------------------------------------------------------------
    print("### 3. Routing statistics (1,003,172 calibration tokens per layer)\n")
    pis = []
    for i in range(L):
        d = json.load(open(os.path.join(a.root, "hessian", f"L{i:02d}", "routing_counts.json")))
        pis.append(np.array(d["tokens_routed"], dtype=np.float64) / d["tokens"])
    pis = np.array(pis)
    print(f"mean inclusion probability {pis.mean():.5f} (uniform = {TOPK / E:.5f}); per-layer max/min expert share "
          f"median {np.median(pis.max(1) / pis.min(1)):.1f}x; hottest expert in any layer {pis.max():.3f}\n")

    parts = {"contiguous 0-127 | 128-255": np.arange(E) // (E // 2),
             "interleaved even | odd": np.arange(E) % 2}

    def balanced(pi):
        order = np.argsort(-pi)
        owner = np.zeros(E, dtype=int)
        load = [0.0, 0.0]
        cnt = [0, 0]
        for e in order:
            r = 0 if (load[0] <= load[1] and cnt[0] < E // 2) or cnt[1] >= E // 2 else 1
            owner[e] = r
            load[r] += pi[e]
            cnt[r] += 1
        return owner

    rows_list = [1, 4, 8, 16, 64, 256]
    res = {n: {"distinct": [], "max_rank": {k: [] for k in list(parts) + ["balanced (by popularity)"]}} for n in rows_list}
    for i in range(L):
        logw = fit_plackett_luce(pis[i], rng)
        owners = dict(parts)
        owners["balanced (by popularity)"] = balanced(pis[i])
        for n in rows_list:
            trials = {1: a.trials, 4: a.trials // 2, 8: a.trials // 2, 16: a.trials // 3, 64: a.trials // 10,
                      256: a.trials // 50}[n]
            dist = []
            mx = {k: [] for k in owners}
            for _ in range(trials):
                sel = np.unique(draw(logw, rng, n).ravel())
                dist.append(len(sel))
                for k, ow in owners.items():
                    c = np.bincount(ow[sel], minlength=2)
                    mx[k].append(c.max())
            res[n]["distinct"].append(np.mean(dist))
            for k in owners:
                res[n]["max_rank"][k].append(np.mean(mx[k]))

    print("### 4. Routed experts read per layer by an n-row step (mean over the 48 layers)\n")
    print("| rows n | distinct experts | TP-split: expert-equivalents per rank | EP contiguous: critical rank | "
          "EP interleaved | EP balanced | worst EP / TP-split |")
    print("|--:|--:|--:|--:|--:|--:|--:|")
    for n in rows_list:
        d = float(np.mean(res[n]["distinct"]))
        tpe = d / 2
        mr = {k: float(np.mean(v)) for k, v in res[n]["max_rank"].items()}
        keys = list(mr)
        print(f"| {n} | {d:.1f} | {tpe:.1f} | {mr[keys[0]]:.1f} | {mr[keys[1]]:.1f} | {mr[keys[2]]:.1f} | "
              f"{max(mr.values()) / tpe:.2f}x |")
    print()

    # ---- 5. bytes per token per rank ---------------------------------------------------------------------------------
    print("### 5. Bytes read per step per rank (mixer + norms + router + shared + routed experts; lm_head excluded)\n")
    print("| step | TP=1 total (one card would read) | TP-split, one rank | ratio | EP-contiguous critical rank (shared expert + mixer still TP-split) |")
    print("|---|--:|--:|--:|--:|")
    for n in [1, 8, 16, 256]:
        t1 = t2 = tep = 0.0
        for i in range(L):
            d = res[n]["distinct"][i]
            b1 = layer_bytes(i, kbs[i], 1, decode_experts=0)
            b2 = layer_bytes(i, kbs[i], 2, decode_experts=0)
            gu1, dn1 = b1["expert_gu"], b1["expert_dn"]
            gu2, dn2 = b2["expert_gu"], b2["expert_dn"]
            base1, base2 = total(b1), total(b2)
            t1 += base1 + d * (gu1 + dn1)
            t2 += base2 + d * (gu2 + dn2)
            mx = res[n]["max_rank"]["contiguous 0-127 | 128-255"][i]
            tep += base2 + mx * (gu1 + dn1)
        print(f"| n = {n} | {t1 / 1e9:.2f} GB | {t2 / 1e9:.2f} GB | {t2 / t1:.3f} | {tep / 1e9:.2f} GB |")
    print()

    # ---- 6. rank shapes ----------------------------------------------------------------------------------------------
    print("### 6. Rank shapes and tile alignment (trellis needs N and K ranges in whole 128-blocks; rank K % 512 for the tuning)\n")
    rows = [
        ("gdn.in_proj_qkv", "rows 3 seg", f"{2 * GDN_KEY_HEADS * GDN_DIM + GDN_VAL_HEADS * GDN_DIM} x {H}",
         [GDN_KEY_HEADS * GDN_DIM // 2] * 2 + [GDN_VAL_HEADS * GDN_DIM // 2], H),
        ("gdn.in_proj_z", "rows", f"{GDN_VAL_HEADS * GDN_DIM} x {H}", [GDN_VAL_HEADS * GDN_DIM // 2], H),
        ("gdn.out_proj", "cols", f"{H} x {GDN_VAL_HEADS * GDN_DIM}", [H], GDN_VAL_HEADS * GDN_DIM // 2),
        ("attn.qg", "rows", f"{2 * ATT_Q * ATT_D} x {H}", [ATT_Q * ATT_D], H),
        ("attn.k / attn.v", "rows", f"{ATT_KV * ATT_D} x {H}", [ATT_KV * ATT_D // 2], H),
        ("attn.o", "cols", f"{H} x {ATT_Q * ATT_D}", [H], ATT_Q * ATT_D // 2),
        ("expert gate_up", "rows 2 seg", f"{2 * INTER} x {H}", [INTER // 2, INTER // 2], H),
        ("expert down", "cols", f"{H} x {INTER}", [H], INTER // 2),
        ("shared gate / up (bf16)", "rows", f"{INTER} x {H}", [INTER // 2], H),
        ("shared down (bf16)", "cols", f"{H} x {INTER}", [H], INTER // 2),
    ]
    print("| linear | split | global N x K | rank N (per segment) | rank K | %128 | rank K %512 |")
    print("|---|---|---|---|--:|---|---|")
    for name, sp, shp, ns, k in rows:
        ok128 = all(n % 128 == 0 for n in ns) and k % 128 == 0
        print(f"| {name} | {sp} | {shp} | {' + '.join(map(str, ns))} | {k} | {'yes' if ok128 else 'NO'} | "
              f"{'yes' if k % 512 == 0 else 'NO'} |")


if __name__ == "__main__":
    main()
