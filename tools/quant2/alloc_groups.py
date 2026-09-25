"""docs/quant2.md sections 5 and 5.3 (Q3): pick per-tensor precisions under a byte budget.

The Milestone 11 method, automated. Each candidate is ONE change measured on its own on top of a
baseline container: a set of linears (e.g. one class in one depth half) given another precision --
kind "group": moved from the build's default w4a16 group to 32 (more scale bytes, less KL) or 128
(fewer bytes, more KL) by `r4dx-convert --w4a16-group-rule "<regex>=<g>"`; kind "keep": written in
bf16 by adding it to the recipe's `--keep-bf16` (a spender); kind "unkeep": a set the recipe keeps in
bf16 quantized like the rest by taking it out of `--keep-bf16` (a saver) -- converted and measured
by rung 4 (tools/quant2/group_sweep.ps1 does both and writes the input file). So every candidate's
option is one precision (g32 / g64 / g128 / bf16) for one set of linears. This script then:

  1. prices every candidate against the baseline:
       spender (delta_gib > 0):  nats of KL RECOVERED per GiB spent   = (kl_base - kl) / delta_gib
       saver   (delta_gib < 0):  nats of KL COST per GiB saved        = (kl - kl_base) / -delta_gib
     a spender that did not lower KL is dropped (nothing to buy); a saver whose KL did not rise is a
     "free" saving and is always taken. Unkeep savers are ordinary savers;
  2. fills the budget greedily: spenders best-first, each one funded -- when it does not fit in what
     is left -- by savers whose cost per GiB is below its own gain per GiB. Of those, the set that
     frees enough bytes for the fewest nats is taken (cheapest-per-GiB first with any saver the
     shortfall does not need dropped again, or one saver large enough on its own, whichever costs
     less), and only when its TOTAL KL cost is below the spender's total KL gain -- a trade only when
     it is profitable, so a large saver is never bought to fund a small spender. Bytes a funding set
     frees beyond the shortfall stay in the budget for later spenders. Two candidates whose LINEAR
     SETS overlap (the same tensors at two groups; a bf16 keep of four layers inside a g32 half of the
     same class) exclude each other: each was measured with the other's linears at the baseline's
     precision, so their deltas do not add. The sets are the ones the sweep recorded from the
     converter's output (`linears`); a file without them (an older sweep's) falls back to "the same
     bases_regex". A baseline already over budget first takes the cheapest savers until it fits. A
     spender that cannot be funded (or only at a loss) is skipped and the fill goes on, so a smaller
     spender behind it can still use what is left; the first one skipped is reported as the budget
     stop. A saver no pick needs any more (a forced one the budget later covers) is dropped again;
  2b. the exchange pass: the fill decides overlaps by ORDER (free savers first, then spenders by
     nats/GiB), so a free g128 half, or a g32 half a little ahead by nats/GiB, would lock out a bf16
     keep inside it that recovers far more. So every candidate left out because it overlaps a pick
     is priced as a swap -- the picks it overlaps out (their bytes and KL refunded), it in, the
     shortfall funded by the cheapest disjoint savers (judged by total KL alone), the fill re-run on
     what that frees, unneeded savers dropped -- and the swap that lowers the predicted KL most is
     made; repeated until none does. A local search: every step lowers the first-order KL and
     stays within budget, but a trade needing two swaps at once is not seen;
  2c. the exhaustive step: the first-order optimum -- the lowest summed dKL over every set of
     pairwise disjoint candidates within budget -- computed exactly by overlap clusters (each
     cluster's disjoint subsets enumerated, the clusters combined on the bytes / dKL Pareto
     frontier; the default list's clusters have at most three candidates). It replaces the picks
     only when strictly better (then its savers are marked as funding and every spender's reason is
     "exhaustive optimum"), so the fill's order and reasons stand whenever they are already optimal.
     Skipped, and said so, past 100,000 disjoint subsets in one cluster or 5 million combinations in
     one merge (the local search's picks then stand);
  3. reports the cliff: the ratio between consecutive picks' nats-per-GiB (and from the last pick
     before it to the budget stop). Past a large drop, each further GiB buys little --
     `--cliff-ratio R` stops the picks before the first drop of at least R;
  4. prints the predicted KL (the sum of the individual deltas -- first order only: the combined
     container must be measured) and the r4dx-convert flags for the chosen set: one
     `--w4a16-group-rule` per group pick, and -- when a keep / unkeep was picked -- ONE `--keep-bf16`
     that REPLACES the recipe's (r4dx-convert's flag is single-valued): the recipe's regex plus the
     keeps minus the unkeeps, composed like group_sweep.ps1 does, with the linear set it must resolve
     to (check it against the converted container's r4dx_convert_run.keep_bf16_linears). That set is
     checked here first, with Python's re (the same answers as the converter's ECMAScript for these
     regexes): over every linear the file names (the baseline's quantized and kept linears, every
     candidate's), the merged regex must match exactly the expected ones -- a recipe that is not the
     one the baseline was converted with fails here, loudly, instead of in the combined container.

Input JSON (group_sweep.ps1's candidates.json):
  {"baseline": {"kl": 0.0385, "weights_gib": 16.4065, "keep_bf16_linears": [...],
                "quantized_linears": [...]},  # optional: the baseline's w4a16 linears
   "budget_gib": 16.4065,                     # optional; --budget-gib wins; default = baseline's
   "recipe": ["--layouts", "w4a16", ..., "--keep-bf16", "<regex>"],   # the BASELINE's convert_args
   "candidates": [{"name": "mlp.down.L0-31.g32", "kind": "group",
                   "bases_regex": "^text\\.layers\\.(...)\\.mlp\\.down$",
                   "group": 32, "delta_gib": 0.043, "kl": 0.0371,
                   "linears": ["text.layers.0.mlp.down", ...]}, ...]}
`kind` defaults to "group" (older files); "group" needs `group`; "unkeep" may give the group its
linears land at. Extra keys are ignored. `budget_gib` is the TOTAL weight budget (docs/quant2.md gate
G8: weight bytes <= v6), so the default -- the baseline's own weights -- is "equal bytes".

  python tools/quant2/alloc_groups.py D:\\models\\r4dx\\q3-sweep\\candidates.json
  python tools/quant2/alloc_groups.py candidates.json --budget-gib 16.5 --cliff-ratio 4 --json-out picks.json
  python tools/quant2/alloc_groups.py --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field

GROUPS = (32, 64, 128)
KINDS = ("group", "keep", "unkeep")
EPS = 1e-12


@dataclass
class Cand:
    name: str
    bases_regex: str
    group: int  # group: its w4a16 group; unkeep: the group its linears land at (0 = not given); keep: 0
    delta_gib: float
    kl: float
    kind: str = "group"  # how the change is made: "group" | "keep" | "unkeep"
    linears: frozenset[str] = frozenset()  # the exact set, or {"regex:<bases_regex>"} (older files)
    dkl: float = 0.0  # kl - kl_base: < 0 recovered, > 0 cost
    rate: float = 0.0  # spender: nats recovered per GiB; saver: nats cost per GiB saved
    role: str = ""  # "spender" | "saver" | "free" | "dropped"

    @property
    def precision(self) -> str:
        if self.kind == "keep":
            return "bf16"
        return f"g{self.group}" if self.group else "w4a16"


@dataclass
class Pick:
    cand: Cand
    why: str  # "free", "forced", "funds <name>", "best", ...


def spend_order(c: Cand) -> tuple[float, str]:
    """Spenders best nats/GiB first; ties by name (deterministic)."""
    return (-c.rate, c.name)


def rate_order(c: Cand) -> tuple[float, str]:
    """Savers (and free ones) cheapest nats/GiB first; ties by name."""
    return (c.rate, c.name)


@dataclass
class Result:
    picks: list[Pick] = field(default_factory=list)
    stop: Cand | None = None  # best-ranked spender the budget could not fund
    stop_reason: str = ""
    stop_after: int = 0  # picked spenders ranked before `stop` (the cliff's last-pick -> stop ratio)
    unfunded: list[tuple[Cand, str]] = field(default_factory=list)  # every skipped spender, why
    excluded: list[tuple[Cand, str]] = field(default_factory=list)  # candidate, the pick(s) it overlaps
    exchanges: list[str] = field(default_factory=list)  # the exchange pass's swaps, in order
    optimum_note: str = ""  # what the exhaustive step (2c) found
    weights_gib: float = 0.0
    kl_pred: float = 0.0
    dropped: list[Cand] = field(default_factory=list)

    def spenders(self) -> list[Cand]:
        """The picked spenders in the fill's order (best nats/GiB first), wherever the exchange
        pass left them in `picks`."""
        return sorted((p.cand for p in self.picks if p.cand.role == "spender"), key=spend_order)


def load(obj: dict) -> tuple[dict, list[Cand], float | None]:
    base = obj.get("baseline")
    if not isinstance(base, dict) or "kl" not in base or "weights_gib" not in base:
        raise ValueError("input needs baseline: {kl, weights_gib}")
    cands = []
    seen = set()
    for i, c in enumerate(obj.get("candidates", [])):
        kind = str(c.get("kind", "group"))
        if kind not in KINDS:
            raise ValueError(f"candidate #{i} ({c.get('name', '?')}): kind '{kind}' is not one of {KINDS}")
        for k in ("name", "bases_regex", "delta_gib", "kl") + (("group",) if kind == "group" else ()):
            if k not in c:
                raise ValueError(f"candidate #{i} ({c.get('name', '?')}) has no '{k}'")
        group = 0
        if kind != "keep" and c.get("group") is not None:
            group = int(c["group"])
            if group not in GROUPS:
                raise ValueError(f"candidate {c['name']}: group {c['group']} is not one of {GROUPS}")
        if c["name"] in seen:
            raise ValueError(f"duplicate candidate name {c['name']}")
        seen.add(c["name"])
        if "linears" in c and isinstance(c["linears"], list):
            if not c["linears"]:
                raise ValueError(f"candidate {c['name']}: empty 'linears'")
            lin = frozenset(str(b) for b in c["linears"])
        else:
            lin = frozenset({"regex:" + str(c["bases_regex"])})
        cands.append(Cand(str(c["name"]), str(c["bases_regex"]), group, float(c["delta_gib"]),
                          float(c["kl"]), kind, lin))
    exact = [c.name for c in cands if not any(b.startswith("regex:") for b in c.linears)]
    if exact and len(exact) != len(cands):
        raise ValueError("some candidates have a 'linears' set and some do not (a mixed or stale "
                         "sweep): overlaps cannot be decided -- re-collect with group_sweep.ps1 -Kl")
    return base, cands, obj.get("budget_gib")


def price(base: dict, cands: list[Cand]) -> None:
    for c in cands:
        c.dkl = c.kl - float(base["kl"])
        if c.delta_gib > EPS:
            if c.dkl < 0:
                c.role, c.rate = "spender", -c.dkl / c.delta_gib
            else:
                c.role, c.rate = "dropped", 0.0
        elif c.delta_gib < -EPS:
            c.role = "free" if c.dkl <= 0 else "saver"
            c.rate = c.dkl / -c.delta_gib
        else:  # no byte change: take it iff it helps
            c.role, c.rate = ("free", 0.0) if c.dkl < 0 else ("dropped", 0.0)


def cheapest_cover(need: float, pool: list[Cand]) -> list[Cand] | None:
    """Savers from `pool` (sorted cheapest nats/GiB first, pairwise disjoint linear sets) freeing >=
    `need` GiB for the fewest total nats, or None if even all of them do not. Two tries, the cheaper
    wins: cheapest-per-GiB first, then drop again (costliest first) every saver the need does not
    require -- a small cheap-per-GiB saver can be made redundant by a larger one taken after it; and
    a single saver large enough on its own, which can beat a sum of small ones in total nats."""
    picked: list[Cand] = []
    freed = 0.0
    for v in pool:
        if freed >= need - EPS:
            break
        picked.append(v)
        freed += -v.delta_gib
    best = None
    if freed >= need - EPS:
        for v in sorted(picked, key=lambda v: (-v.dkl, v.name)):
            if freed + v.delta_gib >= need - EPS:
                picked.remove(v)
                freed += v.delta_gib
        best = picked
    for v in pool:
        if -v.delta_gib >= need - EPS and (best is None or v.dkl < sum(b.dkl for b in best) - EPS):
            best = [v]
    return best


class _State:
    """One allocation: picks with pairwise disjoint linear sets, and the budget they leave."""

    def __init__(self, remaining: float) -> None:
        self.picks: list[Pick] = []
        self.remaining = remaining
        self.locked: set[str] = set()  # the picks' linears: decided

    def copy(self) -> _State:
        s = _State(self.remaining)
        s.picks = list(self.picks)
        s.locked = set(self.locked)
        return s

    def has(self, c: Cand) -> bool:
        return any(p.cand is c for p in self.picks)

    def disjoint(self, c: Cand) -> bool:
        return not (c.linears & self.locked)

    def take(self, c: Cand, why: str) -> None:
        self.picks.append(Pick(c, why))
        self.locked |= c.linears
        self.remaining -= c.delta_gib

    def drop(self, c: Cand) -> None:
        self.picks = [p for p in self.picks if p.cand is not c]
        self.locked = set().union(*(p.cand.linears for p in self.picks))
        self.remaining += c.delta_gib

    def dkl(self) -> float:
        return sum(p.cand.dkl for p in self.picks)


def _take_free(st: _State, free: list[Cand]) -> None:
    for c in free:
        if not st.has(c) and st.disjoint(c):
            st.take(c, "free (KL did not rise)" if c.delta_gib < 0 else "free (no bytes)")


def _prune(st: _State, floor: float) -> None:
    """Drop every picked saver the budget no longer needs, costliest first: a forced one the fill
    later covered, or one a swap made redundant. Each drop lowers the predicted KL."""
    for v in sorted((p.cand for p in st.picks if p.cand.role == "saver"), key=lambda v: (-v.dkl, v.name)):
        if st.remaining + v.delta_gib >= floor - EPS:
            st.drop(v)


def _fill(st: _State, free: list[Cand], savers: list[Cand], spenders: list[Cand],
          floor: float) -> list[tuple[Cand, str]]:
    """The greedy fill from `st` (step 2 of the module doc): free savers, then every spender that
    is not picked and overlaps no pick, best nats/GiB first, each funded when it does not fit.
    Returns the skipped spenders, why, in that order."""
    _take_free(st, free)
    unfunded: list[tuple[Cand, str]] = []
    for s in spenders:
        if st.has(s) or not st.disjoint(s):  # picked, or its linears are decided (excluded)
            continue
        funding: list[Cand] = []
        why_not = ""
        left = st.remaining - floor
        need = s.delta_gib - left
        if need > EPS:
            pool: list[Cand] = []
            in_pool: set[str] = set()
            for v in savers:  # cheapest nats/GiB first; pairwise disjoint, disjoint from s
                if (st.has(v) or not st.disjoint(v) or (v.linears & s.linears) or v.rate >= s.rate
                        or (v.linears & in_pool)):
                    continue
                pool.append(v)
                in_pool.update(v.linears)
            cover = cheapest_cover(need, pool)
            if cover is None:
                why_not = (f"needs {s.delta_gib:.4f} GiB, {left:.4f} left and the savers "
                           f"cheaper than {s.rate:.4g} nats/GiB cannot fund the rest")
            elif sum(v.dkl for v in cover) >= -s.dkl - EPS:
                why_not = (f"needs {s.delta_gib:.4f} GiB, {left:.4f} left; the cheapest funding "
                           f"({', '.join(v.name for v in cover)}) costs "
                           f"{sum(v.dkl for v in cover):.5f} nats, not less than the "
                           f"{-s.dkl:.5f} it recovers")
            else:
                funding = cover
        if why_not:
            unfunded.append((s, why_not))
            continue  # a smaller spender further down may still fit
        for v in funding:
            st.take(v, f"funds {s.name}")
        st.take(s, "best remaining nats/GiB")
    _prune(st, floor)
    return unfunded


def _best_exchange(st: _State, cands: list[Cand], free: list[Cand], savers: list[Cand],
                   spenders: list[Cand], floor: float
                   ) -> tuple[_State, list[tuple[Cand, str]], str] | None:
    """Step 2b: the swap that lowers the predicted KL most, or None. Each candidate that is not
    picked but overlaps a pick goes in, the picks it overlaps go out, the free savers that are now
    disjoint come in, a shortfall is funded by the cheapest disjoint savers (any nats/GiB: the
    swap is judged by its total), and the fill re-runs on what is left."""
    cur = st.dkl()
    allowed = {c.name for c in spenders}  # --cliff-ratio's limit holds here too
    best: tuple[_State, list[tuple[Cand, str]], str] | None = None
    for x in sorted(cands, key=lambda c: c.name):
        if x.role == "dropped" or st.has(x) or (x.role == "spender" and x.name not in allowed):
            continue
        out = [p.cand for p in st.picks if p.cand.linears & x.linears]
        if not out:  # a plain addition: the fill already priced it
            continue
        t = st.copy()
        for o in out:
            t.drop(o)
        t.take(x, "replaces " + ", ".join(o.name for o in out))
        _take_free(t, free)
        need = floor - t.remaining
        if need > EPS:
            pool: list[Cand] = []
            in_pool: set[str] = set()
            for v in savers:
                if t.has(v) or not t.disjoint(v) or (v.linears & in_pool):
                    continue
                pool.append(v)
                in_pool.update(v.linears)
            cover = cheapest_cover(need, pool)
            if cover is None:
                continue
            for v in cover:
                t.take(v, f"funds {x.name}")
        # List x after what funds it, like the fill does.
        t.picks.sort(key=lambda p: p.cand is x)
        unfunded = _fill(t, free, savers, spenders, floor)
        k = t.dkl()
        if k < cur - EPS and (best is None or k < best[0].dkl() - EPS):
            what = (f"{x.name} in for {', '.join(o.name for o in out)}" if t.has(x) else
                    f"{', '.join(o.name for o in out)} dropped to make room")
            best = (t, unfunded, f"{what}: predicted dKL {cur:+.5f} -> {k:+.5f}")
    return best


MAX_SUBSETS = 100_000  # disjoint subsets of one overlap cluster the exhaustive step enumerates
MAX_MERGE = 5_000_000  # (frontier state, cluster subset) pairs in one merge


def _disjoint_subsets(comp: list[Cand]) -> list[tuple[float, float, tuple[Cand, ...]]] | None:
    """Every subset of `comp` with pairwise disjoint linear sets, as (bytes, dKL, members) -- by
    backtracking, so only those are visited -- or None past MAX_SUBSETS."""
    out: list[tuple[float, float, tuple[Cand, ...]]] = []

    def rec(i: int, chosen: tuple[Cand, ...], locked: frozenset[str], b: float, k: float) -> bool:
        if i == len(comp):
            out.append((b, k, chosen))
            return len(out) <= MAX_SUBSETS
        c = comp[i]
        if not rec(i + 1, chosen, locked, b, k):
            return False
        if not (c.linears & locked):
            return rec(i + 1, chosen + (c,), locked | c.linears, b + c.delta_gib, k + c.dkl)
        return True

    return out if rec(0, (), frozenset(), 0.0, 0.0) else None


def optimum(live: list[Cand], room: float) -> tuple[float, list[Cand]] | str:
    """Step 2c: the lowest summed dKL over every set of `live` candidates with pairwise disjoint
    linear sets and summed delta_gib <= `room` -- exactly, by clusters: candidates are grouped into
    connected components of the overlap graph, each component's disjoint subsets enumerated, and the
    components combined keeping only the (bytes, dKL) Pareto frontier (a state is dropped when
    another has no more bytes and no more dKL: the totals are sums, so none of its completions can
    win). Returns (dKL, the set), or why it was not computed."""
    parent = list(range(len(live)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(live)):
        for j in range(i):
            if live[i].linears & live[j].linears:
                parent[root(i)] = root(j)
    comps: dict[int, list[Cand]] = {}
    for i, c in enumerate(live):
        comps.setdefault(root(i), []).append(c)
    frontier: list[tuple[float, float, tuple[Cand, ...]]] = [(0.0, 0.0, ())]
    for comp in comps.values():
        opts = _disjoint_subsets(comp)
        if opts is None:
            return f"an overlap cluster of {len(comp)} candidates has more than {MAX_SUBSETS} disjoint subsets"
        if len(frontier) * len(opts) > MAX_MERGE:
            return f"more than {MAX_MERGE} combinations in one merge"
        merged = sorted(((b1 + b2, k1 + k2, s1 + s2) for b1, k1, s1 in frontier for b2, k2, s2 in opts),
                        key=lambda t: (t[0], t[1]))
        frontier = []
        for t in merged:  # keep a state only if nothing with fewer bytes has less dKL
            if not frontier or t[1] < frontier[-1][1]:
                frontier.append(t)
    feasible = [t for t in frontier if t[0] <= room + EPS]
    if not feasible:
        return "no set fits the budget"
    return feasible[-1][1], list(feasible[-1][2])


def allocate(base: dict, cands: list[Cand], budget_gib: float,
             spender_limit: set[str] | None = None, exchange: bool = True) -> Result:
    """The greedy fill, then the exchange pass and the exhaustive step (unless `exchange` is
    False). `spender_limit` (names) restricts the spenders considered (--cliff-ratio)."""
    price(base, cands)
    res = Result(dropped=[c for c in cands if c.role == "dropped"])
    room0 = budget_gib - float(base["weights_gib"])
    st = _State(room0)
    free = sorted((c for c in cands if c.role == "free"), key=rate_order)
    savers = sorted((c for c in cands if c.role == "saver"), key=rate_order)
    spenders = sorted((c for c in cands if c.role == "spender"), key=spend_order)
    if spender_limit is not None:
        spenders = [c for c in spenders if c.name in spender_limit]

    _take_free(st, free)
    forced_skipped: list[Cand] = []
    for c in savers:  # a baseline over budget must shed bytes first, cheapest first
        if st.remaining >= -EPS:
            break
        if st.disjoint(c):
            st.take(c, "forced (baseline over budget)")
        else:
            forced_skipped.append(c)
    # No later step may leave less than this: the budget, or -- if even the forced savers could not
    # bring the baseline within it -- where they left it.
    floor = min(0.0, st.remaining)
    unfunded = _fill(st, free, savers, spenders, floor)
    while exchange:
        move = _best_exchange(st, cands, free, savers, spenders, floor)
        if move is None:
            break
        st, unfunded, note = move
        res.exchanges.append(note)
    if exchange:
        # Step 2c: the exchange pass is a local search; the exhaustive optimum replaces its picks
        # only when strictly better (so the fill's order and reasons stand whenever they are optimal).
        opt = optimum(free + savers + spenders, room0 - floor)
        if isinstance(opt, str):
            res.optimum_note = f"exhaustive check skipped ({opt}): the picks are the local search's"
        elif opt[0] < st.dkl() - EPS:
            cur = st.dkl()
            st = _State(room0)
            chosen = opt[1]
            for c in sorted((c for c in chosen if c.role == "free"), key=rate_order):
                st.take(c, "free (KL did not rise)" if c.delta_gib < 0 else "free (no bytes)")
            for c in sorted((c for c in chosen if c.role == "saver"), key=rate_order):
                st.take(c, "funds the spenders (exhaustive optimum)")
            for c in sorted((c for c in chosen if c.role == "spender"), key=spend_order):
                st.take(c, "exhaustive optimum")
            unfunded = _fill(st, free, savers, spenders, floor)  # adds nothing: records why not
            res.exchanges.append(f"exhaustive optimum over the overlap clusters: predicted dKL "
                                 f"{cur:+.5f} -> {st.dkl():+.5f}")
            res.optimum_note = ("exhaustive check: the picks are the first-order optimum at this "
                                "budget (it replaced the local search's)")
        else:
            res.optimum_note = ("exhaustive check: the picks are the first-order optimum at this "
                                "budget (the fill and exchange pass reached it)")

    res.picks = st.picks
    res.unfunded = unfunded
    if unfunded:
        res.stop, res.stop_reason = unfunded[0]
        res.stop_after = sum(1 for c in res.spenders() if spend_order(c) < spend_order(res.stop))
    # Everything a pick's overlap left out, named: spenders, free savers, and savers the forced step
    # wanted (other savers are just unused funding).
    for c in spenders + free + forced_skipped:
        by = [p.cand.name for p in st.picks if p.cand.linears & c.linears]
        if by and not st.has(c):
            res.excluded.append((c, ", ".join(by)))
    res.weights_gib = float(base["weights_gib"]) + sum(p.cand.delta_gib for p in res.picks)
    res.kl_pred = float(base["kl"]) + sum(p.cand.dkl for p in res.picks)
    return res


def cliff(res: Result) -> list[tuple[str, str, float]]:
    """(from, to, ratio) between consecutive picked spenders, then the last pick BEFORE the budget
    stop -> the budget stop (always last, so run()'s --cliff-ratio slice sees only pick pairs)."""
    sp = res.spenders()
    out = [(a.name, b.name, a.rate / b.rate) for a, b in zip(sp, sp[1:])]
    if res.stop is not None and res.stop_after > 0:
        last = sp[res.stop_after - 1]
        out.append((last.name, res.stop.name + " (not funded)", last.rate / res.stop.rate))
    return out


def run(base: dict, cands: list[Cand], budget_gib: float,
        cliff_ratio: float | None) -> tuple[Result, list[tuple[str, str, float]], str]:
    res = allocate(base, cands, budget_gib)
    ratios = cliff(res)
    note = ""
    if cliff_ratio is not None:
        sp = res.spenders()
        for i, (_, _, r) in enumerate(ratios[: max(len(sp) - 1, 0)]):
            if r >= cliff_ratio:
                keep = {c.name for c in sp[: i + 1]}
                note = (f"--cliff-ratio {cliff_ratio}: stopped after {sp[i].name} "
                        f"(next pick {sp[i + 1].name} is x{r:.2f} worse per GiB)")
                res = allocate(base, cands, budget_gib, spender_limit=keep)
                ratios = cliff(res)
                break
    return res, ratios, note


def recipe_keep(recipe: list | None) -> str:
    """The recipe's effective --keep-bf16 (the last one: r4dx-convert's flag is single-valued)."""
    keep = ""
    args = list(recipe or [])
    for i, a in enumerate(args):
        if a == "--keep-bf16" and i + 1 < len(args):
            keep = str(args[i + 1])
    return keep


def merged_keep(recipe_re: str, adds: list[str], removes: list[str]) -> str:
    """ONE --keep-bf16 regex (ECMAScript, regex_search like r4dx-convert): matches iff the recipe's
    regex or one of `adds` matches somewhere and none of `removes` does -- group_sweep.ps1's two
    single-candidate forms, "(?:R)|(?:K)" and "^(?![\\s\\S]*?(?:U))[\\s\\S]*?(?:R)", composed."""
    alts = ([recipe_re] if recipe_re else []) + list(adds)
    if not alts:
        raise ValueError("an unkeep pick needs a --keep-bf16 in the recipe")
    keep = alts[0] if len(alts) == 1 else "|".join(f"(?:{a})" for a in alts)
    if removes:
        rem = removes[0] if len(removes) == 1 else "|".join(f"(?:{r})" for r in removes)
        keep = f"^(?![\\s\\S]*?(?:{rem}))[\\s\\S]*?(?:{keep})"
    return keep


def keep_plan(res: Result, recipe: list | None, base_kept: list | None,
              universe: set[str] | None = None) -> tuple[str | None, list[str] | None]:
    """The merged --keep-bf16 for the picks (None when no keep / unkeep was picked: the recipe's
    stands), and the linear set it must resolve to (None without the baseline's keep list or with an
    older file's regex-only sets). With `universe` (linear names), the regex is checked to match
    exactly that set among them -- ValueError if not (module doc, step 4)."""
    adds = [p.cand for p in res.picks if p.cand.kind == "keep"]
    removes = [p.cand for p in res.picks if p.cand.kind == "unkeep"]
    if not adds and not removes:
        return None, None
    rx = merged_keep(recipe_keep(recipe), [c.bases_regex for c in adds], [c.bases_regex for c in removes])
    exact = all(not any(b.startswith("regex:") for b in c.linears) for c in adds + removes)
    if base_kept is None or not exact:
        return rx, None
    kept = set(base_kept)
    for c in removes:
        kept -= c.linears
    for c in adds:
        kept |= c.linears
    if universe is not None:
        try:
            pat = re.compile(rx)
        except re.error as e:
            print(f"note: the merged --keep-bf16 does not compile in Python's re ({e}); it was not "
                  f"checked here -- check keep_bf16_linears after converting", file=sys.stderr)
        else:
            names = set(universe) | kept
            missed = sorted(b for b in kept if not pat.search(b))
            extra = sorted(b for b in names - kept if pat.search(b))
            if missed or extra:
                raise ValueError(
                    f"the merged --keep-bf16 '{rx}' does not resolve to the expected set: it misses "
                    f"[{', '.join(missed)}] and also matches [{', '.join(extra)}]. It is built from the "
                    f"file's recipe (its last --keep-bf16: '{recipe_keep(recipe)}'), which must be the "
                    f"measured baseline's (its keep_bf16_linears: {len(base_kept)} linear(s)) -- "
                    f"re-collect with group_sweep.ps1 -Kl")
    return rx, sorted(kept)


def keep_universe(base: dict, cands: list[Cand]) -> set[str]:
    """Every linear name the file knows: the baseline's quantized and kept linears, and every
    candidate's exact set -- what keep_plan checks the merged regex over."""
    names = set(base.get("quantized_linears") or []) | set(base.get("keep_bf16_linears") or [])
    for c in cands:
        names |= {b for b in c.linears if not b.startswith("regex:")}
    return names


def flags(res: Result, recipe: list | None = None) -> list[str]:
    out = [f"--w4a16-group-rule '{p.cand.bases_regex}={p.cand.group}'"
           for p in res.picks if p.cand.kind == "group"]
    rx, _ = keep_plan(res, recipe, None)
    if rx is not None:
        out.append(f"--keep-bf16 '{rx}'")
    return out


def report(base: dict, cands: list[Cand], budget_gib: float, res: Result,
           ratios: list[tuple[str, str, float]], note: str, recipe: list | None = None,
           out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"baseline: mean KL {float(base['kl']):.5f}, weights {float(base['weights_gib']):.4f} GiB; "
      f"budget {budget_gib:.4f} GiB; {len(cands)} candidate(s)")
    p("")
    p(f"{'candidate':<32} {'kind':<6} {'to':>5} {'lin':>4} {'dGiB':>9} {'dKL':>10} {'role':<8} {'nats/GiB':>10}")
    for c in sorted(cands, key=lambda c: (c.role, -c.rate if c.role == "spender" else c.rate, c.name)):
        n = "?" if any(b.startswith("regex:") for b in c.linears) else str(len(c.linears))
        p(f"{c.name:<32} {c.kind:<6} {c.precision:>5} {n:>4} {c.delta_gib:>+9.4f} {c.dkl:>+10.5f} "
          f"{c.role:<8} {c.rate:>10.4g}")
    p("")
    p("picks (in order):")
    cum = float(base["weights_gib"])
    for i, pk in enumerate(res.picks, 1):
        cum += pk.cand.delta_gib
        p(f"  {i:>2}. {pk.cand.name:<32} {pk.cand.role:<8} {pk.cand.rate:>9.4g} nats/GiB  "
          f"-> {cum:.4f} GiB   [{pk.why}]")
    if not res.picks:
        p("  (none)")
    for e in res.exchanges:
        p(f"  exchange: {e}")
    if res.optimum_note:
        p(f"  {res.optimum_note}")
    if res.stop is not None:
        p(f"budget stop: {res.stop.name} -- {res.stop_reason}")
    for c, why in res.unfunded[1:]:
        p(f"  also skipped: {c.name} -- {why}")
    for c, by in res.excluded:
        p(f"  excluded: {c.name} ({c.role}) -- its linears overlap {by} (measured apart, the deltas "
          f"do not add; no swap for it lowers the predicted KL)")
    if note:
        p(note)
    p("")
    if ratios:
        p("cliff (ratio of nats/GiB between consecutive spenders):")
        for a, b, r in ratios:
            p(f"  {a} -> {b}: x{r:.2f}")
        worst = max(ratios, key=lambda t: t[2])
        p(f"largest drop: x{worst[2]:.2f} at {worst[0]} -> {worst[1]}")
    else:
        p("cliff: no picked spender to compare (none picked, or one and no budget stop after it)")
    p("")
    p(f"predicted: mean KL {res.kl_pred:.5f} (baseline {float(base['kl']):.5f}, "
      f"{res.kl_pred - float(base['kl']):+.5f}; first order -- measure the combined container), "
      f"weights {res.weights_gib:.4f} GiB (budget {budget_gib:.4f})")
    p("")
    p("r4dx-convert flags for the chosen set (disjoint linear sets, so their order does not matter;")
    p("a --keep-bf16 here REPLACES the recipe's):")
    for f in flags(res, recipe) or ["(no change: the baseline is the answer)"]:
        p("  " + f)
    _, kept = keep_plan(res, recipe, base.get("keep_bf16_linears"), keep_universe(base, cands))
    if kept is not None:
        p(f"  that --keep-bf16 must resolve to these {len(kept)} linear(s) "
          f"(r4dx_convert_run.keep_bf16_linears): {', '.join(kept) if kept else '(none -- drop the flag)'}")


def self_test() -> int:
    fails = []

    def check(label: str, cond: bool) -> None:
        print(f"{label:<74} {'OK' if cond else 'FAIL'}")
        if not cond:
            fails.append(label)

    def mk(budget: float) -> tuple[dict, list[Cand]]:
        obj = {"baseline": {"kl": 0.0400, "weights_gib": 16.0}, "candidates": [
            {"name": "A.g32", "bases_regex": "kA", "group": 32, "delta_gib": 0.05, "kl": 0.0380},
            {"name": "B.g32", "bases_regex": "kB", "group": 32, "delta_gib": 0.10, "kl": 0.0370},
            {"name": "C.g32", "bases_regex": "kC", "group": 32, "delta_gib": 0.05, "kl": 0.0398},
            {"name": "D.g32", "bases_regex": "kD", "group": 32, "delta_gib": 0.40, "kl": 0.0392},
            {"name": "A.g128", "bases_regex": "kA", "group": 128, "delta_gib": -0.03, "kl": 0.0406},
            {"name": "E.g128", "bases_regex": "kE", "group": 128, "delta_gib": -0.10, "kl": 0.0401},
            {"name": "F.g128", "bases_regex": "kF", "group": 128, "delta_gib": -0.05, "kl": 0.0403},
            {"name": "G.g32", "bases_regex": "kG", "group": 32, "delta_gib": 0.05, "kl": 0.0405},
        ]}
        base, cands, _ = load(obj)
        return base, cands

    names = lambda res: [p.cand.name for p in res.picks]  # noqa: E731
    close = lambda a, b: abs(a - b) < 1e-9  # noqa: E731

    # 1. Equal bytes: every spender must be funded by a cheaper saver; C cannot be (no saver is
    #    cheaper than its 0.004 nats/GiB once E and F are spent and A.g128 is locked out by A.g32).
    #    (An older file: no kind, no linears -- group candidates, overlap = the same bases_regex.)
    base, cands = mk(16.0)
    res, ratios, _ = run(base, cands, 16.0, None)
    check("equal bytes: picks E.g128, A.g32, F.g128, B.g32 in that order",
          names(res) == ["E.g128", "A.g32", "F.g128", "B.g32"])
    check("equal bytes: rates A 0.04, B 0.03 nats/GiB; E 0.001, F 0.006 cost/GiB",
          close(cands[0].rate, 0.04) and close(cands[1].rate, 0.03)
          and close(cands[5].rate, 0.001) and close(cands[6].rate, 0.006))
    check("equal bytes: budget stop at C.g32 (first spender it cannot fund)",
          res.stop is not None and res.stop.name == "C.g32")
    check("equal bytes: weights exactly on budget", close(res.weights_gib, 16.0))
    check("equal bytes: predicted KL = 0.04 - 0.002 - 0.003 + 0.0001 + 0.0003",
          close(res.kl_pred, 0.0354))
    check("equal bytes: cliff A->B x1.33 and B->C(not funded) x7.5",
          len(ratios) == 2 and abs(ratios[0][2] - 4 / 3) < 1e-9 and abs(ratios[1][2] - 7.5) < 1e-9)
    check("G.g32 (KL rose) is dropped, never picked",
          any(c.name == "G.g32" for c in res.dropped) and "G.g32" not in names(res))
    check("A.g128 is excluded once A.g32 holds its bases", "A.g128" not in names(res))
    check("older file: kind defaults to group, precision g<group>",
          all(c.kind == "group" for c in cands) and cands[0].precision == "g32")
    check("flags: one --w4a16-group-rule per pick, regex=group",
          flags(res) == ["--w4a16-group-rule 'kE=128'", "--w4a16-group-rule 'kA=32'",
                         "--w4a16-group-rule 'kF=128'", "--w4a16-group-rule 'kB=32'"])

    # 2. A roomier budget: spenders fit on their own until D, which E funds (0.001 < 0.002).
    base, cands = mk(16.5)
    res, ratios, _ = run(base, cands, 16.5, None)
    check("budget 16.5: picks A, B, C, then E funds D",
          names(res) == ["A.g32", "B.g32", "C.g32", "E.g128", "D.g32"])
    check("budget 16.5: no budget stop, weights 16.5", res.stop is None and close(res.weights_gib, 16.5))
    rs = [r for _, _, r in ratios]
    check("budget 16.5: cliff ratios x1.33, x7.5, x2.0",
          len(rs) == 3 and abs(rs[0] - 4 / 3) < 1e-9 and abs(rs[1] - 7.5) < 1e-9 and abs(rs[2] - 2.0) < 1e-9)
    res, ratios, note = run(base, cands, 16.5, cliff_ratio=5.0)
    check("--cliff-ratio 5: stops before C (the x7.5 drop), no saver needed",
          names(res) == ["A.g32", "B.g32"] and "stopped after B.g32" in note)
    check("--cliff-ratio 5: weights 16.15", close(res.weights_gib, 16.15))

    # 3. Baseline over budget: the cheapest saver is forced first, then trades as usual.
    base, cands = mk(15.9)
    res, _, _ = run(base, cands, 15.9, None)
    check("over budget: E forced, then F funds A, then stop at B",
          names(res) == ["E.g128", "F.g128", "A.g32"] and res.stop is not None and res.stop.name == "B.g32")
    check("over budget: ends within budget", res.weights_gib <= 15.9 + 1e-9)
    check("over budget: E's reason says forced", res.picks[0].why.startswith("forced"))

    # 4. A saver that did not raise KL is free and always taken.
    base, cands = mk(16.0)
    cands[6].kl = 0.0399  # F.g128 now improves KL while saving bytes
    res, _, _ = run(base, cands, 16.0, None)
    check("free saver is taken first", res.picks[0].cand.name == "F.g128" and res.picks[0].why.startswith("free"))

    def mk2(budget_delta: float, rows: list[dict], extra: dict | None = None) -> tuple[dict, list[Cand], float]:
        obj = {"baseline": {"kl": 0.0400, "weights_gib": 16.0}, "candidates": rows}
        obj.update(extra or {})
        base, cands, _ = load(obj)
        return base, cands, 16.0 + budget_delta

    # 5. A large saver never funds a small spender at a net KL loss (mlp.gate_up.g128 in one half
    #    frees ~0.166 GiB, attn.o.g32 in one half needs ~0.015): cheaper per GiB, dearer in total.
    base, cands, budget = mk2(0.0, [
        {"name": "S.g32", "bases_regex": "kS", "group": 32, "delta_gib": 0.015, "kl": 0.0396},
        {"name": "V.g128", "bases_regex": "kV", "group": 128, "delta_gib": -0.166, "kl": 0.0420}])
    res, _, _ = run(base, cands, budget, None)
    check("large saver, small spender: no trade at a net KL loss", names(res) == [])
    check("large saver, small spender: S.g32 is the budget stop, reason names the cost",
          res.stop is not None and res.stop.name == "S.g32" and "costs" in res.stop_reason)
    check("large saver, small spender: predicted KL stays the baseline's", close(res.kl_pred, 0.04))

    # 6. The funding set is the cheapest in total nats, not the first by nats/GiB: V1 is cheaper per
    #    GiB but frees far more than S needs and costs more than S recovers; V2 alone covers it.
    base, cands, budget = mk2(0.0, [
        {"name": "S.g32", "bases_regex": "kS", "group": 32, "delta_gib": 0.01, "kl": 0.0395},
        {"name": "V1.g128", "bases_regex": "kV1", "group": 128, "delta_gib": -1.0, "kl": 0.0410},
        {"name": "V2.g128", "bases_regex": "kV2", "group": 128, "delta_gib": -0.02, "kl": 0.0401}])
    res, _, _ = run(base, cands, budget, None)
    check("cheapest cover: V2 funds S, V1 untouched", names(res) == ["V2.g128", "S.g32"])
    check("cheapest cover: net KL gain", res.kl_pred < 0.04)

    # 7. A spender that cannot be funded does not end the fill: a smaller one behind it still fits.
    base, cands, budget = mk2(0.02, [
        {"name": "A.g32", "bases_regex": "kA", "group": 32, "delta_gib": 0.08, "kl": 0.0320},
        {"name": "B.g32", "bases_regex": "kB", "group": 32, "delta_gib": 0.01, "kl": 0.0395}])
    res, ratios, _ = run(base, cands, budget, None)
    check("skip and continue: A.g32 is the budget stop, B.g32 still picked",
          res.stop is not None and res.stop.name == "A.g32" and names(res) == ["B.g32"])
    check("skip and continue: no cliff ratio to a stop with no pick before it", ratios == [])

    # Linear sets for 9-12: mlp.down in 64 layers, attn.k/v in the 16 attention layers.
    down = lambda a, b: [f"text.layers.{i}.mlp.down" for i in range(a, b)]  # noqa: E731
    kv = lambda a, b: [f"text.layers.{i}.attn.{t}" for i in range(a, b) if i % 4 == 3 for t in "kv"]  # noqa: E731
    recipe = ["--layouts", "w4a16", "--keep-bf16", "^R$"]
    base_kept = kv(0, 64)

    # 9. Overlap exclusion by LINEAR SET, not regex text: a bf16 keep of layers 0-3 lies inside the
    #    g32 half of the same class. The better one (the g32 half) is picked; the keep, overlapping,
    #    is excluded; and a saver overlapping a spender never funds it (the g128 half of mlp.down
    #    cannot pay for a keep inside it), while a disjoint one can.
    base, cands, budget = mk2(0.05, [
        {"name": "down.L0-31.g32", "kind": "group", "bases_regex": "kD0", "group": 32,
         "delta_gib": 0.05, "kl": 0.0390, "linears": down(0, 32)},
        {"name": "down.L0-3.bf16", "kind": "keep", "bases_regex": "kK", "delta_gib": 0.48,
         "kl": 0.0380, "linears": down(0, 4)}])
    res, _, _ = run(base, cands, budget, None)
    check("overlap: the g32 half picked, the bf16 keep inside it excluded (not even a stop)",
          names(res) == ["down.L0-31.g32"] and res.stop is None
          and [(c.name, by) for c, by in res.excluded] == [("down.L0-3.bf16", "down.L0-31.g32")])
    base, cands, budget = mk2(0.0, [
        {"name": "down.L0-3.bf16", "kind": "keep", "bases_regex": "kK", "delta_gib": 0.48,
         "kl": 0.0360, "linears": down(0, 4)},
        {"name": "down.L0-31.g128", "kind": "group", "bases_regex": "kD1", "group": 128,
         "delta_gib": -0.60, "kl": 0.0402, "linears": down(0, 32)},
        {"name": "down.L32-63.g128", "kind": "group", "bases_regex": "kD2", "group": 128,
         "delta_gib": -0.60, "kl": 0.0404, "linears": down(32, 64)}])
    res, _, _ = run(base, cands, budget, None)
    check("overlap: an overlapping saver (L0-31.g128, cheaper) never funds the keep; L32-63 does",
          names(res) == ["down.L32-63.g128", "down.L0-3.bf16"])
    check("overlap: the precision column says bf16 / g128",
          [c.precision for c in cands] == ["bf16", "g128", "g128"])

    # 10. A keep spender funded by unkeep savers: equal bytes, +0.2 GiB of bf16 paid for by
    #     quantizing both k/v halves (-0.1123 GiB each at ~0.0036 nats/GiB). The flags REPLACE the
    #     recipe's --keep-bf16 with one regex: the recipe's plus the keep, minus both unkeeps.
    base, cands, budget = mk2(0.0, [
        {"name": "o.L0-31.bf16", "kind": "keep", "bases_regex": "kO", "delta_gib": 0.2,
         "kl": 0.0380, "linears": ["text.layers.3.attn.o", "text.layers.7.attn.o"]},
        {"name": "kv.L0-31.g64", "kind": "unkeep", "bases_regex": "kU1", "group": 64,
         "delta_gib": -0.1123, "kl": 0.0404, "linears": kv(0, 32)},
        {"name": "kv.L32-63.g64", "kind": "unkeep", "bases_regex": "kU2", "group": 64,
         "delta_gib": -0.1123, "kl": 0.0404, "linears": kv(32, 64)}],
        {"recipe": recipe})
    res, _, _ = run(base, cands, budget, None)
    check("keep funded by unkeeps: both k/v halves fund o.L0-31.bf16",
          names(res) == ["kv.L0-31.g64", "kv.L32-63.g64", "o.L0-31.bf16"]
          and res.picks[0].why == "funds o.L0-31.bf16")
    check("keep funded by unkeeps: unkeeps are savers, the keep a spender",
          [c.role for c in cands] == ["spender", "saver", "saver"])
    check("keep funded by unkeeps: within budget, net KL gain",
          res.weights_gib <= budget + 1e-9 and close(res.kl_pred, 0.0400 - 0.002 + 0.0008))
    check("keep funded by unkeeps: flags = ONE merged --keep-bf16 (recipe + keep - unkeeps)",
          flags(res, recipe) == ["--keep-bf16 '^(?![\\s\\S]*?(?:(?:kU1)|(?:kU2)))[\\s\\S]*?(?:(?:^R$)|(?:kO))'"])
    _, kept = keep_plan(res, recipe, base_kept)
    check("keep funded by unkeeps: the merged flag must resolve to exactly the two attn.o",
          kept == ["text.layers.3.attn.o", "text.layers.7.attn.o"])
    check("merged_keep: one keep / one unkeep reproduce group_sweep.ps1's forms",
          merged_keep("R", ["K"], []) == "(?:R)|(?:K)"
          and merged_keep("R", [], ["U"]) == "^(?![\\s\\S]*?(?:U))[\\s\\S]*?(?:R)")

    # 11. A keep spender that is not worth it: it recovers 0.0003 nats for +0.48 GiB (0.000625
    #     nats/GiB), below every saver's cost per GiB, so nothing can fund it at equal bytes and the
    #     savers are not bought either.
    base, cands, budget = mk2(0.0, [
        {"name": "up.L62-63.bf16", "kind": "keep", "bases_regex": "kG", "delta_gib": 0.48,
         "kl": 0.0397, "linears": ["text.layers.62.mlp.gate_up", "text.layers.63.mlp.gate_up"]},
        {"name": "kv.L0-31.g64", "kind": "unkeep", "bases_regex": "kU1", "group": 64,
         "delta_gib": -0.1123, "kl": 0.0404, "linears": kv(0, 32)},
        {"name": "z.L0-31.g128", "kind": "group", "bases_regex": "kZ", "group": 128,
         "delta_gib": -0.4, "kl": 0.0404, "linears": ["text.layers.0.gdn.in_proj_z"]}],
        {"recipe": recipe})
    res, _, _ = run(base, cands, budget, None)
    check("keep not worth it: nothing picked, the keep is the budget stop",
          names(res) == [] and res.stop is not None and res.stop.name == "up.L62-63.bf16"
          and "cannot fund" in res.stop_reason)
    check("keep not worth it: no --keep-bf16 in the flags (the recipe's stands)", flags(res, recipe) == [])
    res, _, _ = run(base, cands, 16.5, None)
    check("... with the bytes to spare it is bought on its own (no saver)", names(res) == ["up.L62-63.bf16"])

    def brute(base: dict, cands: list[Cand], budget: float) -> float:
        """The first-order optimum: the lowest summed dKL over every pairwise-disjoint subset
        within budget (exponential: small inputs only)."""
        price(base, cands)
        live = [c for c in cands if c.role != "dropped"]
        room = budget - float(base["weights_gib"])
        best = 0.0
        for mask in range(1, 1 << len(live)):
            pick = [live[i] for i in range(len(live)) if mask >> i & 1]
            if sum(c.delta_gib for c in pick) > room + EPS:
                continue
            seen: set[str] = set()
            ok = True
            for c in pick:
                if c.linears & seen:
                    ok = False
                    break
                seen |= c.linears
            if ok:
                best = min(best, sum(c.dkl for c in pick))
        return best

    # 13. The fill decides overlaps by order; the exchange pass by value. A free g128 half (taken
    #     first) overlaps a bf16 keep inside it worth 100x more; an unkeep saver can fund the keep.
    rows13 = [
        {"name": "down.L0-31.g128", "kind": "group", "bases_regex": "kD1", "group": 128,
         "delta_gib": -0.083, "kl": 0.0399, "linears": down(0, 32)},
        {"name": "down.L0-3.bf16", "kind": "keep", "bases_regex": "kK", "delta_gib": 0.4773,
         "kl": 0.0300, "linears": down(0, 4)},
        {"name": "kv.L0-31.g64", "kind": "unkeep", "bases_regex": "kU1", "group": 64,
         "delta_gib": -0.5, "kl": 0.0401, "linears": kv(0, 32)}]
    base, cands, budget = mk2(0.0, rows13)
    greedy = allocate(base, cands, budget, exchange=False)
    check("free saver vs keep: the fill alone keeps the free g128 half and excludes the keep",
          names(greedy) == ["down.L0-31.g128"] and close(greedy.kl_pred, 0.0399))
    res, _, _ = run(base, cands, budget, None)
    check("free saver vs keep: the exchange swaps the keep in, funded by the unkeep",
          names(res) == ["kv.L0-31.g64", "down.L0-3.bf16"] and len(res.exchanges) == 1
          and res.picks[0].why == "funds down.L0-3.bf16"
          and res.picks[1].why == "replaces down.L0-31.g128")
    check("free saver vs keep: predicted KL 0.04 - 0.0099, within budget, the optimum",
          close(res.kl_pred, 0.0301) and res.weights_gib <= budget + 1e-9
          and close(res.kl_pred - 0.04, brute(base, cands, budget)))
    check("free saver vs keep: the free g128 half is reported as excluded (not silently lost)",
          [(c.name, c.role, by) for c, by in res.excluded]
          == [("down.L0-31.g128", "free", "down.L0-3.bf16")])

    # 14. The same on the real names and byte prices (+0.5 GiB over 16.4065): a g32 half a little
    #     ahead by nats/GiB locks out the bf16 keep inside it that recovers 2.7x more and fits the
    #     budget on its own; attn.o's free g128 half locks out attn.o's keep (which cannot be funded).
    ao = [f"text.layers.{i}.attn.o" for i in range(35, 64, 4)]
    obj14 = {"baseline": {"kl": 0.0226, "weights_gib": 16.4065}, "candidates": [
        {"name": "attn.o.L32-63.g128", "kind": "group", "bases_regex": "kO128", "group": 128,
         "delta_gib": -0.00732, "kl": 0.02259, "linears": ao},
        {"name": "mlp.down.L0-31.g32", "kind": "group", "bases_regex": "kD32", "group": 32,
         "delta_gib": 0.16602, "kl": 0.02219, "linears": down(0, 32)},
        {"name": "mlp.down.L0-3.bf16", "kind": "keep", "bases_regex": "kDK", "delta_gib": 0.4773,
         "kl": 0.02149, "linears": down(0, 4)},
        {"name": "attn.o.L32-63.bf16", "kind": "keep", "bases_regex": "kOK", "delta_gib": 0.3369,
         "kl": 0.0221, "linears": ao}]}
    base, cands, _ = load(obj14)
    greedy = allocate(base, cands, 16.9065, exchange=False)
    check("27B names: the fill alone stops at g128 + g32 half, -0.00042, 0.34 GiB unspent",
          names(greedy) == ["attn.o.L32-63.g128", "mlp.down.L0-31.g32"]
          and abs(greedy.kl_pred - 0.0226 + 0.00042) < 1e-9)
    res, _, _ = run(base, cands, 16.9065, None)
    check("27B names: the exchange takes the mlp.down keep instead of its g32 half",
          sorted(names(res)) == ["attn.o.L32-63.g128", "mlp.down.L0-3.bf16"]
          and abs(res.kl_pred - 0.0226 + 0.00112) < 1e-9 and res.weights_gib <= 16.9065 + 1e-9)
    check("27B names: the optimum; the g32 half and attn.o's keep are reported as excluded",
          close(res.kl_pred - 0.0226, brute(base, cands, 16.9065))
          and sorted(c.name for c, _ in res.excluded) == ["attn.o.L32-63.bf16", "mlp.down.L0-31.g32"])
    check("27B names: one exchange did it; the exhaustive step agrees and changes nothing",
          len(res.exchanges) == 1 and res.exchanges[0].startswith("mlp.down.L0-3.bf16 in for mlp.down.L0-31.g32")
          and "reached it" in res.optimum_note)

    # 14b. A trade the exchange pass cannot see (it needs c0.g32 dropped AND c1.g128 swapped): the
    #      exhaustive step finds it and replaces the picks.
    base, cands, budget = mk2(0.0876, [
        {"name": "c0.g32", "bases_regex": "k0", "group": 32, "delta_gib": 0.1174, "kl": 0.038704,
         "linears": down(0, 8)},
        {"name": "c1.g32", "bases_regex": "k1", "group": 32, "delta_gib": 0.0841, "kl": 0.038479,
         "linears": down(8, 16)},
        {"name": "c1.g128", "bases_regex": "k1", "group": 128, "delta_gib": -0.076, "kl": 0.039884,
         "linears": down(8, 16)}])
    greedy = allocate(base, cands, budget, exchange=False)
    res, _, _ = run(base, cands, budget, None)
    check("exhaustive: the fill (and the exchange pass) stop at c1.g128 + c0.g32, -0.001412",
          sorted(names(greedy)) == ["c0.g32", "c1.g128"] and abs(greedy.kl_pred - 0.04 + 0.001412) < 1e-9)
    check("exhaustive: c1.g32 alone (-0.001521) replaces them, reasons say so",
          names(res) == ["c1.g32"] and res.picks[0].why == "exhaustive optimum"
          and res.exchanges[-1].startswith("exhaustive optimum") and abs(res.kl_pred - 0.04 + 0.001521) < 1e-9)
    check("exhaustive: c0.g32 is the budget stop, c1.g128 excluded, cliff c1.g32 -> c0.g32",
          res.stop is not None and res.stop.name == "c0.g32" and res.stop_after == 1
          and [(c.name, by) for c, by in res.excluded] == [("c1.g128", "c1.g32")]
          and [(a, b) for a, b, _ in cliff(res)] == [("c1.g32", "c0.g32 (not funded)")])
    # A whole-layer diagnostic keep merges six class halves into one cluster of 19 (2^19 subsets;
    # 4^6 + 2^6 = 4160 of them disjoint): enumerated by backtracking, not skipped.
    rows = []
    for h in range(6):
        half = [f"h{h}.l{i}" for i in range(8)]
        rows += [{"name": f"h{h}.g32", "bases_regex": f"h{h}", "group": 32, "delta_gib": 0.02,
                  "kl": 0.0399 - 0.00001 * h, "linears": half},
                 {"name": f"h{h}.g128", "bases_regex": f"h{h}", "group": 128, "delta_gib": -0.01,
                  "kl": 0.0401, "linears": half},
                 {"name": f"h{h}.bf16", "kind": "keep", "bases_regex": f"k{h}", "delta_gib": 0.4,
                  "kl": 0.0395, "linears": half[1:3]}]
    rows.append({"name": "layer0.bf16", "kind": "keep", "bases_regex": "L0", "delta_gib": 0.5,
                 "kl": 0.0390, "linears": [f"h{h}.l0" for h in range(6)]})
    base, cands, budget = mk2(0.3, rows)
    res, _, _ = run(base, cands, budget, None)
    live = [c for c in cands if c.role != "dropped"]
    subs = _disjoint_subsets(live)
    check("exhaustive: a 19-candidate cluster (4160 disjoint subsets) is enumerated, not skipped",
          len(live) == 19 and subs is not None and len(subs) == 4160 and "skipped" not in res.optimum_note
          and close(res.kl_pred - 0.04, min(k for b, k, _ in subs if b <= 0.3 + EPS)))

    # 15. The merged --keep-bf16 is checked against the set it must resolve to: a recipe that is not
    #     the baseline's (its keep dropped, e.g. a bare -Kl re-collect) fails loudly, as does one
    #     that keeps more than the baseline did.
    real_re = r"^text\.layers\.[0-9]+\.attn\.[kv]$"
    obj15 = {"baseline": {"kl": 0.0400, "weights_gib": 16.0, "keep_bf16_linears": kv(0, 64),
                          "quantized_linears": down(0, 64) + ao},
             "candidates": [
        {"name": "mlp.down.L0-3.bf16", "kind": "keep", "bases_regex": r"^text\.layers\.[0-3]\.mlp\.down$",
         "delta_gib": 0.4773, "kl": 0.0390, "linears": down(0, 4)},
        {"name": "attn.kv.L0-31.g64", "kind": "unkeep", "group": 64,
         "bases_regex": r"^text\.layers\.([0-9]|[12][0-9]|3[01])\.attn\.[kv]$",
         "delta_gib": -0.1123, "kl": 0.04001, "linears": kv(0, 32)}]}
    base, cands, _ = load(obj15)
    uni = keep_universe(base, cands)
    res, _, _ = run(base, cands, 16.4, None)
    check("keep regex: the unkeep funds the keep", names(res) == ["attn.kv.L0-31.g64", "mlp.down.L0-3.bf16"])
    rx, kept = keep_plan(res, ["--layouts", "w4a16", "--keep-bf16", real_re], kv(0, 64), uni)
    check("keep regex: the baseline's recipe resolves to k/v 32-63 + mlp.down 0-3",
          kept == sorted(kv(32, 64) + down(0, 4)) and re.compile(rx).search("text.layers.35.attn.k") is not None)
    res, _, _ = run(base, cands, 16.5, None)  # room for the keep alone
    for recipe_x, label in ((["--layouts", "w4a16"], "no --keep-bf16 (it came through -ExtraArgs)"),
                            (["--keep-bf16", r"^text\.layers\.[0-9]+\.attn\."], "one that keeps attn.o too")):
        try:
            keep_plan(res, recipe_x, kv(0, 64), uni)
            check(f"keep regex: a recipe with {label} fails loudly", False)
        except ValueError as e:
            check(f"keep regex: a recipe with {label} fails loudly", "does not resolve" in str(e))
    _, kept = keep_plan(res, ["--keep-bf16", real_re], kv(0, 64), uni)
    check("keep regex: keep alone, the right recipe: all 32 k/v + mlp.down 0-3",
          kept == sorted(kv(0, 64) + down(0, 4)))

    # 16. Seeded random inputs shaped like the default list (per class half: a g32 and a g128
    #     option on the whole half and sometimes a keep inside it; separate unkeep savers): the
    #     picks never overlap, stay within budget, never predict worse than the fill alone, and are
    #     the first-order optimum (brute force over every subset).
    rng = random.Random(20260926)
    bad_overlap = bad_budget = bad_worse = 0
    hits = trials = local = 0
    for _ in range(300):
        rows = []
        for k in range(rng.randint(1, 4)):
            half = [f"c{k}.l{i}" for i in range(8)]
            rows.append({"name": f"c{k}.g32", "group": 32, "bases_regex": f"c{k}",
                         "delta_gib": rng.uniform(0.01, 0.2), "kl": 0.04 + rng.uniform(-0.002, 0.0003),
                         "linears": half})
            rows.append({"name": f"c{k}.g128", "group": 128, "bases_regex": f"c{k}",
                         "delta_gib": -rng.uniform(0.005, 0.1), "kl": 0.04 + rng.uniform(-0.0002, 0.001),
                         "linears": half})
            if rng.random() < 0.7:
                rows.append({"name": f"c{k}.bf16", "kind": "keep", "bases_regex": f"k{k}",
                             "delta_gib": rng.uniform(0.1, 0.6), "kl": 0.04 + rng.uniform(-0.004, 0.0002),
                             "linears": half[: rng.randint(1, 4)]})
        for u in range(rng.randint(0, 2)):
            rows.append({"name": f"u{u}", "kind": "unkeep", "group": 64, "bases_regex": f"u{u}",
                         "delta_gib": -rng.uniform(0.05, 0.6), "kl": 0.04 + rng.uniform(-0.0001, 0.001),
                         "linears": [f"u{u}.k"]})
        base, cands, budget = mk2(rng.uniform(0.0, 0.8), rows)
        res = allocate(base, cands, budget)
        greedy = allocate(base, cands, budget, exchange=False)
        seen: set[str] = set()
        for pk in res.picks:
            bad_overlap += bool(pk.cand.linears & seen)
            seen |= pk.cand.linears
        bad_budget += res.weights_gib > budget + 1e-9
        bad_worse += res.kl_pred > greedy.kl_pred + 1e-12
        if len(cands) <= 14:
            trials += 1
            best = brute(base, cands, budget)
            hits += abs(res.kl_pred - 0.04 - best) < 1e-9
            local += not any(e.startswith("exhaustive") for e in res.exchanges)
    check("random: no overlapping picks", bad_overlap == 0)
    check("random: always within budget", bad_budget == 0)
    check("random: never predicts worse than the fill alone", bad_worse == 0)
    check(f"random: the first-order optimum in every brute-forced input ({hits}/{trials})",
          trials >= 250 and hits == trials)
    print(f"  (the fill + exchange pass reached it alone in {local}/{trials})")

    # 12. Input validation.
    for bad, label in (({"candidates": []}, "no baseline"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "bases_regex": "x", "group": 48,
                                         "delta_gib": 1, "kl": 1}]}, "group 48"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "group": 32, "delta_gib": 1, "kl": 1}]},
                        "missing bases_regex"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "kind": "fp8", "bases_regex": "x",
                                         "delta_gib": 1, "kl": 1}]}, "an unknown kind"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "kind": "group", "bases_regex": "x",
                                         "delta_gib": 1, "kl": 1}]}, "a group candidate without group"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "kind": "keep", "bases_regex": "x",
                                         "delta_gib": 1, "kl": 1, "linears": []}]}, "empty linears"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "kind": "keep", "bases_regex": "x",
                                         "delta_gib": 1, "kl": 1, "linears": ["a"]},
                                        {"name": "y", "bases_regex": "y", "group": 32,
                                         "delta_gib": 1, "kl": 1}]}, "mixed exact / regex-only sets")):
        try:
            load(bad)
            check(f"rejects input with {label}", False)
        except ValueError:
            check(f"rejects input with {label}", True)

    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="?", help="candidates JSON (tools/quant2/group_sweep.ps1)")
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="total weight budget in GiB (default: the file's budget_gib, else the "
                         "baseline's weights_gib -- equal bytes)")
    ap.add_argument("--cliff-ratio", type=float, default=None,
                    help="stop before the first consecutive-pick drop in nats/GiB of at least this")
    ap.add_argument("--json-out", default=None, help="also write the picks and flags here")
    ap.add_argument("--self-test", action="store_true", help="run the built-in synthetic checks")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.candidates:
        ap.error("a candidates JSON is required (or --self-test)")
    # utf-8-sig: Windows PowerShell 5.1's Set-Content -Encoding utf8 (group_sweep.ps1) writes a BOM.
    with open(args.candidates, encoding="utf-8-sig") as f:
        obj = json.load(f)
    try:
        base, cands, file_budget = load(obj)
        recipe = obj.get("recipe")
        budget = args.budget_gib if args.budget_gib is not None else (
            float(file_budget) if file_budget is not None else float(base["weights_gib"]))
        res, ratios, note = run(base, cands, budget, args.cliff_ratio)
        report(base, cands, budget, res, ratios, note, recipe)
        keep_rx, keep_set = keep_plan(res, recipe, base.get("keep_bf16_linears"), keep_universe(base, cands))
    except ValueError as e:
        sys.stdout.flush()
        print(f"alloc_groups.py: error: {e}", file=sys.stderr)
        return 2
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"baseline": base, "budget_gib": budget,
                       "picks": [{"name": p.cand.name, "kind": p.cand.kind,
                                  "bases_regex": p.cand.bases_regex, "group": p.cand.group,
                                  "precision": p.cand.precision,
                                  "linears": sorted(p.cand.linears), "delta_gib": p.cand.delta_gib,
                                  "dkl": p.cand.dkl, "rate": p.cand.rate, "role": p.cand.role,
                                  "why": p.why} for p in res.picks],
                       "budget_stop": res.stop.name if res.stop else None,
                       "unfunded": [{"name": c.name, "why": why} for c, why in res.unfunded],
                       "excluded": [{"name": c.name, "role": c.role, "overlaps": by}
                                    for c, by in res.excluded],
                       "exchanges": res.exchanges, "optimum": res.optimum_note,
                       "cliff": [{"from": a, "to": b, "ratio": r} for a, b, r in ratios],
                       "cliff_note": note,
                       "predicted_kl": res.kl_pred, "weights_gib": res.weights_gib,
                       "keep_bf16": keep_rx, "keep_bf16_linears_expected": keep_set,
                       "flags": flags(res, recipe)}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
