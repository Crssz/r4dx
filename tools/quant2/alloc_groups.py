"""docs/quant2.md section 5 (Q3): pick per-tensor w4a16 groups under a byte budget.

The Milestone 11 method, automated. Each candidate is ONE change measured on its own on top of a
baseline container: a set of linears (a container-base regex, e.g. one class in one depth half)
moved from the build's default w4a16 group to 32 (more scale bytes, less KL) or 128 (fewer bytes,
more KL), converted with `r4dx-convert --w4a16-group-rule "<regex>=<g>"` and measured by rung 4
(tools/quant2/group_sweep.ps1 does both and writes the input file). This script then:

  1. prices every candidate against the baseline:
       spender (delta_gib > 0):  nats of KL RECOVERED per GiB spent   = (kl_base - kl) / delta_gib
       saver   (delta_gib < 0):  nats of KL COST per GiB saved        = (kl - kl_base) / -delta_gib
     a spender that did not lower KL is dropped (nothing to buy); a saver whose KL did not rise is a
     "free" saving and is always taken;
  2. fills the budget greedily: spenders best-first, each one funded -- when it does not fit in what
     is left -- by savers whose cost per GiB is below its own gain per GiB. Of those, the set that
     frees enough bytes for the fewest nats is taken (cheapest-per-GiB first with any saver the
     shortfall does not need dropped again, or one saver large enough on its own, whichever costs
     less), and only when its TOTAL KL cost is below the spender's total KL gain -- a trade only when
     it is profitable, so a large saver is never bought to fund a small spender. Bytes a funding set
     frees beyond the shortfall stay in the budget for later spenders. Two candidates with the same
     `bases_regex` are the same tensors at two groups and exclude each other. A baseline already
     over budget first takes the cheapest savers until it fits. A spender that cannot be funded
     (or only at a loss) is skipped and the fill goes on, so a smaller spender behind it can still
     use what is left; the first one skipped is reported as the budget stop;
  3. reports the cliff: the ratio between consecutive picks' nats-per-GiB (and from the last pick
     before it to the budget stop). Past a large drop, each further GiB buys little --
     `--cliff-ratio R` stops the picks before the first drop of at least R;
  4. prints the predicted KL (the sum of the individual deltas -- first order only: the combined
     container must be measured) and the `--w4a16-group-rule` flags for the chosen set.

Input JSON (group_sweep.ps1's candidates.json):
  {"baseline": {"kl": 0.0385, "weights_gib": 16.4065},
   "budget_gib": 16.4065,                     # optional; --budget-gib wins; default = baseline's
   "candidates": [{"name": "mlp.down.L0-31.g32",
                   "bases_regex": "^text\\.layers\\.(...)\\.mlp\\.down$",
                   "group": 32, "delta_gib": 0.043, "kl": 0.0371}, ...]}
Extra keys are ignored. `budget_gib` is the TOTAL weight budget (docs/quant2.md gate G8: weight
bytes <= v6), so the default -- the baseline's own weights -- is "equal bytes".

  python tools/quant2/alloc_groups.py D:\\models\\r4dx\\q3-sweep\\candidates.json
  python tools/quant2/alloc_groups.py candidates.json --budget-gib 16.5 --cliff-ratio 4 --json-out picks.json
  python tools/quant2/alloc_groups.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

GROUPS = (32, 64, 128)
EPS = 1e-12


@dataclass
class Cand:
    name: str
    bases_regex: str
    group: int
    delta_gib: float
    kl: float
    dkl: float = 0.0  # kl - kl_base: < 0 recovered, > 0 cost
    rate: float = 0.0  # spender: nats recovered per GiB; saver: nats cost per GiB saved
    kind: str = ""  # "spender" | "saver" | "free" | "dropped"


@dataclass
class Pick:
    cand: Cand
    why: str  # "free", "forced", "funds <name>", "best", ...


@dataclass
class Result:
    picks: list[Pick] = field(default_factory=list)
    stop: Cand | None = None  # first spender the budget could not fund
    stop_reason: str = ""
    stop_after: int = 0  # spenders picked before `stop` (the cliff's last-pick -> stop ratio)
    unfunded: list[tuple[Cand, str]] = field(default_factory=list)  # every skipped spender, why
    weights_gib: float = 0.0
    kl_pred: float = 0.0
    dropped: list[Cand] = field(default_factory=list)

    def spenders(self) -> list[Cand]:
        return [p.cand for p in self.picks if p.cand.kind == "spender"]


def load(obj: dict) -> tuple[dict, list[Cand], float | None]:
    base = obj.get("baseline")
    if not isinstance(base, dict) or "kl" not in base or "weights_gib" not in base:
        raise ValueError("input needs baseline: {kl, weights_gib}")
    cands = []
    seen = set()
    for i, c in enumerate(obj.get("candidates", [])):
        for k in ("name", "bases_regex", "group", "delta_gib", "kl"):
            if k not in c:
                raise ValueError(f"candidate #{i} ({c.get('name', '?')}) has no '{k}'")
        if int(c["group"]) not in GROUPS:
            raise ValueError(f"candidate {c['name']}: group {c['group']} is not one of {GROUPS}")
        if c["name"] in seen:
            raise ValueError(f"duplicate candidate name {c['name']}")
        seen.add(c["name"])
        cands.append(Cand(str(c["name"]), str(c["bases_regex"]), int(c["group"]),
                          float(c["delta_gib"]), float(c["kl"])))
    return base, cands, obj.get("budget_gib")


def price(base: dict, cands: list[Cand]) -> None:
    for c in cands:
        c.dkl = c.kl - float(base["kl"])
        if c.delta_gib > EPS:
            if c.dkl < 0:
                c.kind, c.rate = "spender", -c.dkl / c.delta_gib
            else:
                c.kind, c.rate = "dropped", 0.0
        elif c.delta_gib < -EPS:
            c.kind = "free" if c.dkl <= 0 else "saver"
            c.rate = c.dkl / -c.delta_gib
        else:  # no byte change: take it iff it helps
            c.kind, c.rate = ("free", 0.0) if c.dkl < 0 else ("dropped", 0.0)


def cheapest_cover(need: float, pool: list[Cand]) -> list[Cand] | None:
    """Savers from `pool` (sorted cheapest nats/GiB first, one per bases_regex) freeing >= `need`
    GiB for the fewest total nats, or None if even all of them do not. Two tries, the cheaper wins:
    cheapest-per-GiB first, then drop again (costliest first) every saver the need does not require
    -- a small cheap-per-GiB saver can be made redundant by a larger one taken after it; and a
    single saver large enough on its own, which can beat a sum of small ones in total nats."""
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


def allocate(base: dict, cands: list[Cand], budget_gib: float,
             spender_limit: set[str] | None = None) -> Result:
    """The greedy fill. `spender_limit` (names) restricts the spenders considered (--cliff-ratio)."""
    price(base, cands)
    res = Result(dropped=[c for c in cands if c.kind == "dropped"])
    locked: set[str] = set()  # bases_regex already decided
    remaining = budget_gib - float(base["weights_gib"])

    def take(c: Cand, why: str) -> None:
        nonlocal remaining
        res.picks.append(Pick(c, why))
        locked.add(c.bases_regex)
        remaining -= c.delta_gib

    key = lambda c: (c.rate, c.name)  # noqa: E731 -- deterministic tie-break by name
    for c in sorted((c for c in cands if c.kind == "free"), key=key):
        if c.bases_regex not in locked:
            take(c, "free (KL did not rise)" if c.delta_gib < 0 else "free (no bytes)")
    savers = sorted((c for c in cands if c.kind == "saver"), key=key)
    for c in savers:  # a baseline over budget must shed bytes first, cheapest first
        if remaining >= -EPS:
            break
        if c.bases_regex not in locked:
            take(c, "forced (baseline over budget)")

    spenders = sorted((c for c in cands if c.kind == "spender"), key=lambda c: (-c.rate, c.name))
    if spender_limit is not None:
        spenders = [c for c in spenders if c.name in spender_limit]
    picked_spenders = 0
    for s in spenders:
        if s.bases_regex in locked:
            continue
        funding: list[Cand] = []
        why_not = ""
        need = s.delta_gib - remaining
        if need > EPS:
            pool, bases = [], set()
            for v in savers:  # cheapest nats/GiB first; one group per set of tensors
                if (v.bases_regex in locked or v.bases_regex == s.bases_regex or v.rate >= s.rate
                        or v.bases_regex in bases):
                    continue
                pool.append(v)
                bases.add(v.bases_regex)
            cover = cheapest_cover(need, pool)
            if cover is None:
                why_not = (f"needs {s.delta_gib:.4f} GiB, {remaining:.4f} left and the savers "
                           f"cheaper than {s.rate:.4g} nats/GiB cannot fund the rest")
            elif sum(v.dkl for v in cover) >= -s.dkl - EPS:
                why_not = (f"needs {s.delta_gib:.4f} GiB, {remaining:.4f} left; the cheapest funding "
                           f"({', '.join(v.name for v in cover)}) costs "
                           f"{sum(v.dkl for v in cover):.5f} nats, not less than the "
                           f"{-s.dkl:.5f} it recovers")
            else:
                funding = cover
        if why_not:
            res.unfunded.append((s, why_not))
            if res.stop is None:
                res.stop, res.stop_reason, res.stop_after = s, why_not, picked_spenders
            continue  # a smaller spender further down may still fit
        for v in funding:
            take(v, f"funds {s.name}")
        take(s, "best remaining nats/GiB")
        picked_spenders += 1

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


def flags(res: Result) -> list[str]:
    return [f"--w4a16-group-rule '{p.cand.bases_regex}={p.cand.group}'" for p in res.picks]


def report(base: dict, cands: list[Cand], budget_gib: float, res: Result,
           ratios: list[tuple[str, str, float]], note: str, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"baseline: mean KL {float(base['kl']):.5f}, weights {float(base['weights_gib']):.4f} GiB; "
      f"budget {budget_gib:.4f} GiB; {len(cands)} candidate(s)")
    p("")
    p(f"{'candidate':<34} {'g':>4} {'dGiB':>9} {'dKL':>10} {'kind':<8} {'nats/GiB':>10}")
    for c in sorted(cands, key=lambda c: (c.kind, -c.rate if c.kind == "spender" else c.rate, c.name)):
        p(f"{c.name:<34} {c.group:>4} {c.delta_gib:>+9.4f} {c.dkl:>+10.5f} {c.kind:<8} {c.rate:>10.4g}")
    p("")
    p("picks (in order):")
    cum = float(base["weights_gib"])
    for i, pk in enumerate(res.picks, 1):
        cum += pk.cand.delta_gib
        p(f"  {i:>2}. {pk.cand.name:<34} {pk.cand.kind:<8} {pk.cand.rate:>9.4g} nats/GiB  "
          f"-> {cum:.4f} GiB   [{pk.why}]")
    if not res.picks:
        p("  (none)")
    if res.stop is not None:
        p(f"budget stop: {res.stop.name} -- {res.stop_reason}")
    for c, why in res.unfunded[1:]:
        p(f"  also skipped: {c.name} -- {why}")
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
    p("r4dx-convert flags for the chosen set (disjoint bases, so their order does not matter):")
    for f in flags(res) or ["(no rules: the baseline is the answer)"]:
        p("  " + f)


def self_test() -> int:
    fails = []

    def check(label: str, cond: bool) -> None:
        print(f"{label:<66} {'OK' if cond else 'FAIL'}")
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

    def mk2(budget_delta: float, rows: list[dict]) -> tuple[dict, list[Cand], float]:
        base, cands, _ = load({"baseline": {"kl": 0.0400, "weights_gib": 16.0}, "candidates": rows})
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

    # 8. Input validation.
    for bad, label in (({"candidates": []}, "no baseline"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "bases_regex": "x", "group": 48,
                                         "delta_gib": 1, "kl": 1}]}, "group 48"),
                       ({"baseline": {"kl": 1, "weights_gib": 1},
                         "candidates": [{"name": "x", "group": 32, "delta_gib": 1, "kl": 1}]},
                        "missing bases_regex")):
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
        base, cands, file_budget = load(json.load(f))
    budget = args.budget_gib if args.budget_gib is not None else (
        float(file_budget) if file_budget is not None else float(base["weights_gib"]))
    res, ratios, note = run(base, cands, budget, args.cliff_ratio)
    report(base, cands, budget, res, ratios, note)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"baseline": base, "budget_gib": budget,
                       "picks": [{"name": p.cand.name, "bases_regex": p.cand.bases_regex,
                                  "group": p.cand.group, "delta_gib": p.cand.delta_gib,
                                  "dkl": p.cand.dkl, "rate": p.cand.rate, "kind": p.cand.kind,
                                  "why": p.why} for p in res.picks],
                       "budget_stop": res.stop.name if res.stop else None,
                       "unfunded": [{"name": c.name, "why": why} for c, why in res.unfunded],
                       "cliff": [{"from": a, "to": b, "ratio": r} for a, b, r in ratios],
                       "cliff_note": note,
                       "predicted_kl": res.kl_pred, "weights_gib": res.weights_gib,
                       "flags": flags(res)}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
