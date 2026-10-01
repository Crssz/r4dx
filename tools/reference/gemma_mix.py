"""tools/reference/gemma_mix.py -- the Gemma 4 (gemma4_unified) mix4.5m ranking (docs/gemma4-plan.md 9.12, 11).

Qwen production trellis is mix4.5m = EXL3's allocator (trellis_quant.allocate): every (layer, qgroup) group
starts at the floor rate (K4) and groups are promoted to K5 in a FIXED position order (edge layers first),
because no qwen3_5 module sets q_priority. That ranking is architecture-blind. This module replaces the
ORDER for Gemma and keeps everything else (qgroups as units, floor/next rates, the budget
int(bpw * sum numel), the whole-group-fits rule), so the byte budget and the manifest format are identical.

Score of a group (higher = promoted earlier):

    score(group) = sum_t numel_t * prior_t * gap_t  /  sum_t numel_t          (a numel-weighted mean)
    gap_t        = proxy_t(K_floor) - proxy_t(K_next)    DATA: the oracle's Hessian proxy loss
                                                         tr(E H E^T) / tr(W H W^T) of the K4 and K5 runs
                                                         (manifest tensor record "proxy"), i.e. the relative
                                                         error this promotion removes. 1.0 for every tensor
                                                         when no oracle data is given (prior-only ranking).
    prior_t      = layer_weight(layer, layer_type) * kind_weight(module, layer_type)

Cost of a promotion is numel bits (one more bit per weight), so ranking by the numel-weighted mean of
prior*gap is "relative output-error removed per bit" under the first-order assumption that a linear's output
energy is proportional to its parameter count. That assumption is a heuristic, not a measurement: the final
arbiter is the KL gate (plan 9.9) on the assembled mix, and the weights below are the only place to tune.

Priors (rationale; evidence in tools/reference/gemma/investigate/ and kl/cpu_probe.json, layer10_amp.json):
  * amplification sites L10, L23, L29, L41 (position 1 at L10, full layers L23/L29/L41: bf16 rounding there is
    amplified 5-50x in the residual): x1.5 -- an extra weight error injected at a layer that already
    amplifies perturbations is amplified the same way.
  * full-attention layers (every 6th, L5..L47): x1.25 -- 8 global layers carry the long-range routing, and with
    k_eq_v their k_proj is BOTH key and value, so one K error is counted twice (kind weight below).
  * edge layers (0, 1, 46, 47): x1.5 -- EXL3's own rule (edges first) and the embedding / head interface.
  * massive-activation band L9..L40 (residual 100-300): x1.1 -- a mild tilt; the cpu_probe per-layer relative
    error grows with depth (0.005 at L0, 0.1-0.2 at L30+), so an error late in the band is not discounted.
  * kinds: k_proj of a FULL layer x1.5 (k_eq_v, above); down_proj x1.25 (it writes the massive residual
    channels, and rotation does not smooth its input); everything else 1.0.
All of them are multiplied; the largest is 1.5 * 1.25 * 1.1 * 1.5 = 3.1 (k_proj, full amp layer 41 in-band).
The data term gap_t varies by far more than these (K4 proxies differ by 10x between tensor kinds in the Qwen
oracle), so the priors mostly break ties and tilt the order; they matter alone only in the prior-only mode
(quantize-model --bpw with no oracle, or `mix --ranking prior`).

No torch, no numpy: importable on any python (the CPU unit test uses it).
"""

from __future__ import annotations

AMP_LAYERS = frozenset({10, 23, 29, 41})
MASSIVE_BAND = (9, 40)          # inclusive
EDGE = 2                        # layers < EDGE and >= n_layers - EDGE
W_AMP, W_FULL, W_EDGE, W_BAND = 1.5, 1.25, 1.5, 1.1
W_K_FULL, W_DOWN = 1.5, 1.25


def layer_weight(layer: int, layer_type: str, n_layers: int) -> float:
    w = 1.0
    if layer in AMP_LAYERS:
        w *= W_AMP
    if layer_type == "full_attention":
        w *= W_FULL
    if layer < EDGE or layer >= n_layers - EDGE:
        w *= W_EDGE
    if MASSIVE_BAND[0] <= layer <= MASSIVE_BAND[1]:
        w *= W_BAND
    return w


def kind_weight(module: str, layer_type: str) -> float:
    if module == "self_attn.k_proj" and layer_type == "full_attention":
        return W_K_FULL
    if module == "mlp.down_proj":
        return W_DOWN
    return 1.0


def prior(t: dict, n_layers: int) -> float:
    return layer_weight(t["layer"], t["layer_type"], n_layers) * kind_weight(t["module"], t["layer_type"])


def avg_rate(tensors: list, rates: dict) -> float:
    """Numel-weighted mean of the per-tensor rate: what the budget int(bpw * total) bounds."""
    tot = sum(t["numel"] for t in tensors)
    return sum(t["numel"] * rates[t["name"]] for t in tensors) / tot


def group_scores(tensors: list, gaps: dict | None, n_layers: int) -> dict:
    """{(layer, qgroup): score}. `gaps` {tensor name: proxy(K_floor) - proxy(K_next)} or None (all 1.0)."""
    acc: dict = {}
    for t in tensors:
        g = 1.0 if gaps is None else max(float(gaps[t["name"]]), 0.0)
        a = acc.setdefault((t["layer"], t.get("qgroup") or t["name"]), [0.0, 0])
        a[0] += t["numel"] * prior(t, n_layers) * g
        a[1] += t["numel"]
    return {k: s / n for k, (s, n) in acc.items()}


def allocate(tensors: list, bpw: float, rate_floor, rate_next, gaps: dict | None = None,
             n_layers: int | None = None) -> tuple:
    """trellis_quant.allocate with the Gemma order. Returns (rates {name: K}, report [group dicts, in
    promotion-priority order]). Same budget, same whole-group-fits rule and same sweep-until-stable loop as
    EXL3's; only the order differs: by descending score (ties: layer, first idx, so it is deterministic)."""
    n_layers = n_layers or (max(t["layer"] for t in tensors) + 1)
    budget = int(bpw * sum(t["numel"] for t in tensors))
    rate = {t["name"]: rate_floor(bpw) for t in tensors}
    used = sum(t["numel"] * rate[t["name"]] for t in tensors)
    groups: dict = {}
    for t in sorted(tensors, key=lambda t: (t["layer"], t["idx"])):
        groups.setdefault((t["layer"], t.get("qgroup") or t["name"]), []).append(t)
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
                extra += 0.0 if nr is None else t["numel"] * (nr - rate[t["name"]])
            if extra > 0 and used + extra <= budget:
                for t in g:
                    nr = rate_next(rate[t["name"]])
                    if nr is not None:
                        rate[t["name"]] = nr
                used += extra
                changed = True
    report = [{"layer": k[0], "qgroup": k[1], "score": score[k], "members": [t["name"] for t in groups[k]],
               "numel": sum(t["numel"] for t in groups[k]),
               "K": sorted({rate[t["name"]] for t in groups[k]})} for k in order]
    return rate, report
