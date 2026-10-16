"""tools/reference/lfm2_mix.py -- the LFM2 (lfm2_moe) mix4.5 ranking (docs/lfm2-plan.md 7.1, task L3-3).

Same machinery as gemma_mix.py (the budget int(bpw * sum numel), the whole-group-fits rule, the sweep until stable)
with three LFM2 specifics:

  * NON-EXPERT linears are pinned at the K5 floor: attention q/k/v/o, ShortConv in_proj/out_proj and the dense FFN
    (layers 0-1) are about 3 % of the weights; promoting them costs 0.03 bpw and they are the sensitive mixers.
    They are counted in the budget first, so the experts' average lands at about 4.48 for a 4.5 target.
  * the allocator's unit is one GROUP per layer role (arch_table's `qgroups`): the gate_up of ALL experts of a layer
    (`moe.gu`) and the down of ALL experts (`moe.d`) are promoted K4 -> K5 whole, so every layer keeps ONE KB per
    expert tensor kind and the per-layer expert table of the L4 kernels stays uniform. Expert-granular promotion is a
    possible later refinement (open in the plan).
  * the priors (docs/lfm2-plan.md 7.1): the first MoE layers (num_dense_layers and the next one: layers 2 and 3 of
    the real model) and the last two layers x1.5 -- the layers that see the least-averaged routing and the head
    interface; the full-attention layers x1.25 (10 of 40: the long-range mixing); the `moe.d` group x1.25 (a down
    projection writes the residual stream). All multiplied; a data term replaces the prior-only mode:

        score(group) = sum_t numel_t * prior_t * gap_t  /  sum_t numel_t
        gap_t        = proxy_t(K_floor) - proxy_t(K_next)   (the K4 and K5 oracle runs' Hessian proxy, manifest "proxy")

    1.0 for every tensor when no oracle data is given (prior-only: `quantize-model --bpw` and `mix --ranking prior`).
    The prior x proxy-gap product is the plan's "first mix"; an empirical per-layer sweep is the repair loop (L3-9).

No torch, no numpy: importable on any python (the CPU unit test uses it).
"""

from __future__ import annotations

import re

#: highest rate the mix may use: only K4 and K5 oracles exist (plan 7: no K6 source is planned), so leftover
#: budget must not buy a K4->K5->K6 second promotion of the top group.
MAX_K = 5.0

#: promoted-unit names (arch_table.lfm2_arch qgroups)
EXPERT_GROUPS = ("moe.gu", "moe.d")
#: the rate of every non-expert linear (the K5 floor of plan 7.1)
NONEXPERT_K = 5.0
W_FIRST_MOE, W_LAST, W_ATTN, W_DOWN = 1.5, 1.5, 1.25, 1.25
LAST_LAYERS = 2
_EXPERT_RE = re.compile(r"feed_forward\.experts\.\d+\.(w1|w3|w2)")


def is_expert(t: dict) -> bool:
    return _EXPERT_RE.fullmatch(t["module"]) is not None


def first_moe_layer(tensors: list) -> int:
    return min(t["layer"] for t in tensors if is_expert(t))


def layer_weight(layer: int, layer_type: str, n_layers: int, first_moe: int) -> float:
    w = 1.0
    if first_moe <= layer <= first_moe + 1:
        w *= W_FIRST_MOE
    if layer >= n_layers - LAST_LAYERS:
        w *= W_LAST
    if layer_type.startswith("full_attention"):
        w *= W_ATTN
    return w


def kind_weight(t: dict) -> float:
    return W_DOWN if t["qgroup"] == "moe.d" else 1.0


def prior(t: dict, n_layers: int, first_moe: int) -> float:
    return layer_weight(t["layer"], t["layer_type"], n_layers, first_moe) * kind_weight(t)


def avg_rate(tensors: list, rates: dict) -> float:
    tot = sum(t["numel"] for t in tensors)
    return sum(t["numel"] * rates[t["name"]] for t in tensors) / tot


def group_scores(tensors: list, gaps: dict | None, n_layers: int | None = None) -> dict:
    """{(layer, qgroup): score} of the PROMOTABLE (expert) groups. `gaps` {tensor name: proxy(K_floor) -
    proxy(K_next)} or None (all 1.0)."""
    n_layers = n_layers or (max(t["layer"] for t in tensors) + 1)
    fm = first_moe_layer(tensors)
    acc: dict = {}
    for t in tensors:
        if not is_expert(t):
            continue
        g = 1.0 if gaps is None else max(float(gaps[t["name"]]), 0.0)
        a = acc.setdefault((t["layer"], t["qgroup"]), [0.0, 0])
        a[0] += t["numel"] * prior(t, n_layers, fm) * g
        a[1] += t["numel"]
    return {k: s / n for k, (s, n) in acc.items()}


def allocate(tensors: list, bpw: float, rate_floor, rate_next, gaps: dict | None = None,
             n_layers: int | None = None, nonexpert_k: float = NONEXPERT_K) -> tuple:
    """Returns (rates {name: K}, report [expert group dicts in promotion-priority order]). Non-expert linears sit at
    `nonexpert_k`; expert groups start at rate_floor(bpw) and are promoted one rate step each, in descending score
    order, whenever the WHOLE group's promotion fits int(bpw * sum numel). Deterministic (ties: layer, first idx)."""
    n_layers = n_layers or (max(t["layer"] for t in tensors) + 1)
    budget = int(bpw * sum(t["numel"] for t in tensors))
    floor = rate_floor(bpw)
    rate = {t["name"]: (floor if is_expert(t) else max(nonexpert_k, floor)) for t in tensors}
    used = sum(t["numel"] * rate[t["name"]] for t in tensors)
    if used > budget:
        raise ValueError(f"bpw {bpw}: the K{max(nonexpert_k, floor):g} non-expert linears and the K{floor:g} expert "
                         f"floor already cost {used / sum(t['numel'] for t in tensors):.4f} bits/weight")
    groups: dict = {}
    for t in sorted(tensors, key=lambda t: (t["layer"], t["idx"])):
        if is_expert(t):
            groups.setdefault((t["layer"], t["qgroup"]), []).append(t)
    score = group_scores(tensors, gaps, n_layers)
    order = sorted(groups, key=lambda k: (-score[k], k[0], groups[k][0]["idx"]))
    changed = True
    while changed:
        changed = False
        for key in order:
            g = groups[key]
            extra = 0.0
            for t in g:
                nr = rate_next(rate[t["name"]])
                extra += 0.0 if (nr is None or nr > MAX_K) else t["numel"] * (nr - rate[t["name"]])
            if extra > 0 and used + extra <= budget:
                for t in g:
                    nr = rate_next(rate[t["name"]])
                    if nr is not None and nr <= MAX_K:
                        rate[t["name"]] = nr
                used += extra
                changed = True
    report = [{"layer": k[0], "qgroup": k[1], "score": score[k], "n_members": len(groups[k]),
               "first_member": groups[k][0]["name"], "numel": sum(t["numel"] for t in groups[k]),
               "K": sorted({rate[t["name"]] for t in groups[k]})} for k in order]
    return rate, report
