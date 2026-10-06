#!/usr/bin/env python
"""Offline n-gram (prompt-lookup) speculation simulator for r4dx-server request logs.

Reads the JSON Lines file `r4dx-server --request-log <path> --request-log-tokens` writes
(docs/server.md "Request log" > "Token capture"), replays every logged generation against a
prompt-lookup proposer and answers: would an n-gram drafter, alone or in front of DFlash, have paid?

Per generation, the proposer looks at the context (prompt + the tokens generated so far), finds the
longest suffix of it (min-n .. max-n tokens) that occurred earlier -- the most recent occurrence -- and
proposes the up-to-k tokens that followed it. The verify round accepts the longest prefix of the
proposal that equals what the model actually generated next, plus one bonus token (so a round emits
accepted + 1 tokens, and a round with no match emits 1: plain decode).

  * exact for greedy requests (temperature 0): the logged tokens ARE what a speculative decoder would
    have produced, whatever the drafter.
  * for sampled requests it replays the one sampled trajectory that happened. A deterministic proposal
    is accepted with probability p(token) = the chance the sample equals it, so the replay estimates
    the real acceptance without bias but with sampling noise. --greedy-only drops them.

Reported (stdlib only; numpy is not needed):
  tok/round   generated tokens / rounds, the engine's own "tok/round" convention (the first generated
              token comes out of the prefill and is counted; rounds cover the tokens after it).
  round hit   share of rounds whose proposal had at least one accepted token.
  token hit   share of the tokens after the first that sit inside a copy run: a maximal stretch where the
              longest-suffix match's continuation keeps equalling what was generated (k-independent).
  run>=10     share of those tokens inside runs of 10+ tokens.
  hybrid      per round, the n-gram proposal when its match is >= T tokens long, else a DFlash round. A
              DFlash round emits what the log's own DFlash rounds did from that position
              (round_accepted; "positional": the logged round that starts there, or the rest of the
              one that covers the position after an n-gram stretch, so the hybrid keeps the log's
              local difficulty) or always the request's mean logged tok/round ("mean": optimistic
              when the n-gram takes the stretches where DFlash was great; the gate uses the lower of
              the two), and --dflash-tpr when the log has no DFlash rounds.

Timing (what the verdict is about). Speculation only pays if decode gets FASTER, and an n-gram round is
cheaper than a DFlash round (no drafter forward, fewer verify rows), so rounds are priced:
  verify(rows)  = --verify-ms-base + --verify-ms-per-row * rows
  DFlash round  = verify(k+1 = 8 rows) + --drafter-ms           (39.8 ms at the defaults, TP1)
  n-gram round  = verify(len(proposal) + 1 rows)  [+ --inject-ms-per-round in the hybrid: the drafter
                  must be fed the rows it did not see, one InjectFeatures call, weight-read bound so
                  independent of the 1-8 rows]  (no proposal: 1 row = plain decode)
The defaults are the repo's measurements (docs/dflash2.md k sweep: round 39.8 ms at k=7, 0.8 ms per
draft row, InjectFeatures 0.875 ms wall at 64 rows; docs/perf.md top table: decode step 27.28 ms and
8-row verify 32.19 ms on w4a16, 26.93 / 31.59 on trellis mix4.5m, plain 36.69 tok/s = 27.3 ms).
Estimated tok/s = decode tokens (all but the first) / summed round time. A log written with
--dflash-k below 7 prices its rounds at its own draft_k; an --mtp log has no DFlash rounds here.

Gate: BUILD the n-gram path only if the best hybrid's estimated tok/s (the lower of the positional and
mean models) beats DFlash alone as logged by >= --min-gain (default 5%) on at least --min-requests
(default 20) requests of >= --gate-min-gen generated tokens. Fewer requests: NOT ENOUGH DATA. The
2026-10-03 gates (n-gram alone >= 2.5 tok/round at k=7; > 50% of tokens in 10+ token copy runs) are
still printed, as information only: n-gram alone can pass them while the hybrid is slower than DFlash.

  python sim_ngram.py requests.jsonl
  python sim_ngram.py requests.jsonl --k 7,9 --hybrid-t 3,4,5 --greedy-only --rows 0
  python sim_ngram.py requests.jsonl --min-gain 0.10 --min-requests 50 --drafter-ms 6.5
  python sim_ngram.py --selftest
  python sim_ngram.py --write-example example_requests.jsonl
"""
import argparse
import bisect
import json
import random
import sys
import tempfile
import os
from collections import Counter

GATE_K = 7  # kMaxUnsplitDraftK: a verify window is at most 8 rows
GATE_TPR = 2.5
GATE_RUN = 10
GATE_RUN_SHARE = 0.5
DFLASH_TPR = 2.96  # real agent traffic, 2026-10-03
# Round cost model, TP1 Huihui 27B trellis + w4a16 DFlash2 drafter (docs/dflash2.md, docs/perf.md):
# plain decode = 1 row = 27.2 ms, an 8-row verify = 32.8 ms (measured 32.2), a k=7 DFlash round = 39.8 ms.
VERIFY_MS_BASE = 26.4
VERIFY_MS_PER_ROW = 0.8
DRAFTER_MS = 7.0  # 39.8 - verify(8): DraftRound wall 6.2 ms + injection + glue
# InjectFeatures(64 rows) wall, w4a16 drafter. Charged per n-gram round, not per row: the call is
# weight-read / launch bound (docs/dflash2.md: DraftRound runs at the card's bandwidth roof plus ~41 us
# a launch), so injecting the 1-8 rows of an n-gram round costs about what 64 rows do.
INJECT_MS_PER_ROUND = 0.875
MIN_GAIN = 0.05
MIN_REQUESTS = 20
GATE_MIN_GEN = 32
RUN_BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, None)]


# ---- reading the log ---------------------------------------------------------------------------------

def read_requests(paths, stats):
    """Yield one dict per request that carries tokens, prompts rebuilt from prompt_shared.

    A token line stores prompt_ids[prompt_shared:]; the first `prompt_shared` ids are the previous
    token-carrying line's prompt_ids + generated_ids. A line that cannot be rebuilt (the chain is broken
    by an unreadable line, or prompt_shared points past it) is counted in stats and skipped; the chain
    restarts at the next line with prompt_shared == 0.
    """
    for path in paths:
        prev = []  # previous token line's prompt + generated; None = the chain is broken
        with open(path, "r", encoding="utf-8-sig") as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    stats["unreadable_lines"] += 1
                    prev = None  # a torn line (server killed mid-write) may have been a token line
                    continue
                if "prompt_ids" not in rec:
                    stats["lines_without_tokens"] += 1
                    continue
                shared = rec.get("prompt_shared", 0)
                if shared and (prev is None or shared > len(prev)):
                    stats["unrebuildable"] += 1
                    prev = None
                    continue
                prompt = (prev[:shared] if shared else []) + rec["prompt_ids"]
                gen = rec.get("generated_ids") or []
                prev = prompt + gen
                stats["token_lines"] += 1
                rec["_prompt"] = prompt
                rec["_gen"] = gen
                rec["_where"] = "%s:%d" % (os.path.basename(path), lineno)
                yield rec


# ---- the proposer ------------------------------------------------------------------------------------

def build_match_table(prompt, gen, min_n, max_n, window=0):
    """table[i] = (n, pos) for i in 1..len(gen)-1: the proposer's longest-suffix match when it has to
    predict gen[i], i.e. the context is prompt + gen[:i] (its last token sits at index e = P + i - 1 of
    prompt + gen). n is the matched suffix length, pos the index where the matched occurrence ENDS (so
    the proposal continues at pos + 1). None = no match of at least min_n tokens. window > 0 only
    accepts a source within the last `window` tokens. Independent of the round structure, so one
    table serves every k, threshold and hybrid variant.
    """
    P = len(prompt)
    full = prompt + gen
    n_gen = len(gen)
    ns = list(range(min_n, max_n + 1))
    idx = {}
    for m in ns:
        if P >= m:
            # dict(zip(...)) keeps the LAST occurrence of a repeated gram: the most recent one.
            grams = zip(*[full[i:P - m + 1 + i] for i in range(m)])
            idx[m] = dict(zip(grams, range(m - 1, P)))
        else:
            idx[m] = {}
    table = [None] * n_gen
    for i in range(1, n_gen):
        e = P + i - 1
        best = None
        for m in reversed(ns):
            if e + 1 < m:
                continue
            pos = idx[m].get(tuple(full[e - m + 1:e + 1]))
            if pos is not None and (window <= 0 or e - pos <= window):
                best = (m, pos)
                break
        table[i] = best
        for m in ns:  # the suffix ending at e is searchable from the next position on
            if e + 1 >= m:
                idx[m][tuple(full[e - m + 1:e + 1])] = e
    return table


def brute_match_table(prompt, gen, min_n, max_n, window=0):
    """The same table by the definition, O(n^2): the selftest's reference for build_match_table."""
    P = len(prompt)
    full = prompt + gen
    table = [None] * len(gen)
    for i in range(1, len(gen)):
        e = P + i - 1
        best = None
        for m in range(max_n, min_n - 1, -1):
            if e + 1 < m:
                continue
            suffix = full[e - m + 1:e + 1]
            for pos in range(e - 1, m - 2, -1):  # most recent first; the match ends strictly before e
                if full[pos - m + 1:pos + 1] == suffix and (window <= 0 or e - pos <= window):
                    best = (m, pos)
                    break
            if best:
                break
        table[i] = best
    return table


def proposal(full, P, i, match, k):
    """The tokens the proposer would put in the window at gen index i: the continuation of the match,
    cut at k and at what the context already holds (index P + i - 1)."""
    _, pos = match
    return full[pos + 1:min(pos + 1 + k, P + i)]


def accepted(prop, gen, i):
    a = 0
    n = len(gen)
    while a < len(prop) and i + a < n and prop[a] == gen[i + a]:
        a += 1
    return a


# ---- one request -------------------------------------------------------------------------------------

def sim_ngram(prompt, gen, table, k):
    """Standalone n-gram rounds at draft cap k. Returns (rounds, rounds_with_proposal, rounds_hit,
    verified_rows): a round verifies len(proposal) + 1 rows, 1 without a proposal."""
    P, n = len(prompt), len(gen)
    full = prompt + gen
    i = 1
    rounds = proposed = hit = rows = 0
    while i < n:
        m = table[i]
        a = 0
        rows += 1
        if m is not None:
            prop = proposal(full, P, i, m, k)
            a = accepted(prop, gen, i)
            proposed += 1
            hit += a > 0
            rows += len(prop)
        i += min(a + 1, n - i)
        rounds += 1
    return rounds, proposed, hit, rows


def copy_runs(prompt, gen, table):
    """Lengths of the copy runs (see the module doc), k-independent."""
    full = prompt + gen
    n = len(gen)
    P = len(prompt)
    runs = []
    i = 1
    while i < n:
        m = table[i]
        run = 0
        if m is not None:
            src = m[1] + 1
            while i + run < n and src + run < len(full) and full[src + run] == gen[i + run]:
                run += 1
        if run:
            runs.append(run)
            i += run
        else:
            i += 1
    return runs


def dflash_rounds_of(rec, n):
    """(starts, ends) of the log's DFlash rounds in generated-token index space, or None. Round r
    emitted round_accepted[r] + 1 tokens, the first one starting at index 1. MTP rounds are not
    DFlash rounds (a 4-row verify and a different drafter): the timing model cannot price them, so a
    --mtp log is treated as having none and takes the --dflash-tpr constant."""
    acc = rec.get("round_accepted") or []
    if not acc or rec.get("speculative") != "dflash":
        return None
    starts, ends, s = [], [], 1
    for a in acc:
        starts.append(s)
        s += a + 1
        ends.append(s)
    return starts, ends


def sim_hybrid(prompt, gen, table, k, t_min, rounds_log, tpr):
    """n-gram when the match is >= t_min tokens long, else a DFlash round. rounds_log: dflash_rounds_of
    (positional) or None (mean). Mean mode: a DFlash round emits `tpr` tokens on average (carried
    fractionally), the request's mean logged tokens per round. Positional mode: a DFlash round at
    position i emits the rest of the logged round that covers i (a whole logged round when it starts
    at i): the drafts that round got accepted are the ones a fresh round would, so the hybrid stays
    on the log's own local difficulty. The mean is wrong exactly where the hybrid matters: the n-gram
    takes the stretches where DFlash was also great, and charging the mean to the rest hands the
    hybrid DFlash rounds far better than any it was logged with. The mean only covers a position
    past the last logged round. Returns (rounds, ngram_rounds, dflash_rounds, ngram_rows): the rows
    the n-gram rounds verified."""
    P, n = len(prompt), len(gen)
    full = prompt + gen
    i = 1
    rounds = ng = df = ng_rows = 0
    carry = 0.0
    while i < n:
        m = table[i]
        if m is not None and m[0] >= t_min:
            prop = proposal(full, P, i, m, k)
            emit = accepted(prop, gen, i) + 1
            ng += 1
            ng_rows += len(prop) + 1
        else:
            emit = 0
            if rounds_log is not None:
                starts, ends = rounds_log
                r = bisect.bisect_right(starts, i) - 1
                if r >= 0 and i < ends[r]:
                    emit = ends[r] - i
            if emit == 0:
                carry += tpr
                emit = max(1, int(carry))
                carry -= emit
            df += 1
        i += min(emit, n - i)
        rounds += 1
    return rounds, ng, df, ng_rows


def verify_ms(cfg, rows):
    return cfg["verify_base"] + cfg["verify_row"] * rows


def dflash_round_ms(cfg, k=GATE_K):
    """A DFlash round at draft cap k (the log's draft_k; 7 = 8 rows = the measured 39.8 ms)."""
    return verify_ms(cfg, k + 1) + cfg["drafter_ms"]


def analyse(rec, cfg):
    prompt, gen = rec["_prompt"], rec["_gen"]
    n = len(gen)
    table = build_match_table(prompt, gen, cfg["min_n"], cfg["max_n"], cfg["window"])
    res = {
        "request_id": rec.get("request_id"),
        "where": rec["_where"],
        "prompt_tokens": len(prompt),
        "gen_tokens": n,
        "speculative": rec.get("speculative"),
        "temperature": rec.get("temperature"),
        "greedy": (rec.get("temperature") or 0) <= 0,
    }
    # DFlash as logged (or the constant when the server ran without a drafter). Rounds cover the n - 1
    # tokens after the first, which the prefill emitted.
    log_rounds = dflash_rounds_of(rec, n)
    # what a logged round cost: the server's own draft cap (7 when the log does not say)
    kd = max(1, min(GATE_K, int(rec.get("draft_k") or GATE_K)))
    round_ms = dflash_round_ms(cfg, kd)
    res["dflash_k"] = kd
    if log_rounds is not None:
        res["dflash_rounds"] = len(log_rounds[0])
        res["dflash_src"] = "log"
        res["dflash_tpr"] = n / len(log_rounds[0])
        mean_emit = max(n - 1, 1) / len(log_rounds[0])
        eq_rounds = len(log_rounds[0])
    else:
        res["dflash_rounds"] = None
        res["dflash_src"] = "const"
        res["dflash_tpr"] = cfg["dflash_tpr"]
        mean_emit = cfg["dflash_tpr"]
        eq_rounds = max(n - 1, 0) / cfg["dflash_tpr"]
        round_ms = dflash_round_ms(cfg)  # a hypothetical DFlash round: the production k=7
    res["dflash_ms"] = eq_rounds * round_ms
    res["ngram"] = {}
    for k in cfg["ks"]:
        rounds, proposed, hit, rows = sim_ngram(prompt, gen, table, k)
        res["ngram"][k] = {"rounds": rounds, "proposed": proposed, "hit": hit,
                           "ms": rounds * cfg["verify_base"] + rows * cfg["verify_row"]}
    runs = copy_runs(prompt, gen, table)
    res["runs"] = runs
    res["covered"] = sum(runs)
    res["hybrid"] = {}
    for t in cfg["ts"]:
        for mode in ("positional", "mean"):
            r, ng, df, ng_rows = sim_hybrid(
                prompt, gen, table, cfg["hybrid_k"], t, log_rounds if mode == "positional" else None, mean_emit)
            ms = (df * round_ms + ng * (cfg["verify_base"] + cfg["inject_round"]) + ng_rows * cfg["verify_row"])
            res["hybrid"][(t, mode)] = {"rounds": r, "ngram_rounds": ng, "dflash_rounds": df, "ms": ms}
    return res


# ---- aggregation / report ----------------------------------------------------------------------------

def tpr(tokens, rounds):
    return tokens / rounds if rounds else 0.0


def tok_s(tokens, ms):
    return 1000.0 * tokens / ms if ms else 0.0


def timing(results, cfg):
    """Estimated decode speed of a set of requests: tok/s of DFlash as logged, of n-gram alone, and of
    each hybrid (T, mode); the conservative gain per T is the lower of the two modes. best = (T, gain,
    tok/s) of the T with the largest conservative gain (None without requests)."""
    if not results:
        return {"requests": 0, "tokens": 0, "dflash_tps": 0.0, "ngram_tps": {}, "hybrid": {}, "best": None}
    toks = sum(max(r["gen_tokens"] - 1, 0) for r in results)
    base = tok_s(toks, sum(r["dflash_ms"] for r in results))
    out = {"requests": len(results), "tokens": toks, "dflash_tps": base, "ngram_tps": {}, "hybrid": {}, "best": None}
    for k in cfg["ks"]:
        out["ngram_tps"][k] = tok_s(toks, sum(r["ngram"][k]["ms"] for r in results))
    for t in cfg["ts"]:
        per = {}
        for mode in ("positional", "mean"):
            per[mode] = tok_s(toks, sum(r["hybrid"][(t, mode)]["ms"] for r in results))
        cons = min(per.values())
        gain = cons / base - 1 if base else 0.0
        out["hybrid"][t] = {"positional": per["positional"], "mean": per["mean"], "conservative": cons, "gain": gain}
        if out["best"] is None or gain > out["best"][1]:
            out["best"] = (t, gain, cons)
    return out


def aggregate(results, cfg):
    agg = {"requests": len(results), "tokens": sum(r["gen_tokens"] for r in results)}
    after_first = sum(max(r["gen_tokens"] - 1, 0) for r in results)
    agg["tokens_after_first"] = after_first
    agg["dflash_tpr"] = tpr(agg["tokens"], sum(
        (r["dflash_rounds"] if r["dflash_rounds"] else r["gen_tokens"] / r["dflash_tpr"]) for r in results))
    agg["dflash_logged"] = sum(1 for r in results if r["dflash_src"] == "log")
    agg["ngram"] = {}
    for k in cfg["ks"]:
        rounds = sum(r["ngram"][k]["rounds"] for r in results)
        agg["ngram"][k] = {
            "tpr": tpr(agg["tokens"], rounds),
            "mean_request_tpr": (sum(tpr(r["gen_tokens"], r["ngram"][k]["rounds"]) for r in results) / len(results)),
            "round_hit": tpr(sum(r["ngram"][k]["hit"] for r in results), rounds),
            "proposal_rate": tpr(sum(r["ngram"][k]["proposed"] for r in results), rounds),
        }
    runs = [x for r in results for x in r["runs"]]
    covered = sum(runs)
    agg["token_hit"] = tpr(covered, after_first)
    agg["run_share"] = tpr(sum(x for x in runs if x >= GATE_RUN), after_first)
    hist = []
    for lo, hi in RUN_BUCKETS:
        toks = sum(x for x in runs if x >= lo and (hi is None or x <= hi))
        hist.append(((lo, hi), sum(1 for x in runs if x >= lo and (hi is None or x <= hi)), toks, tpr(toks, after_first)))
    agg["run_hist"] = hist
    agg["hybrid"] = {}
    for key in results[0]["hybrid"]:
        rounds = sum(r["hybrid"][key]["rounds"] for r in results)
        ng = sum(r["hybrid"][key]["ngram_rounds"] for r in results)
        agg["hybrid"][key] = {"tpr": tpr(agg["tokens"], rounds), "ngram_share": tpr(ng, rounds)}
    # the timing verdict: the gate set is the requests long enough to count, split by sampling
    gate_set = [r for r in results if r["gen_tokens"] >= cfg["gate_min_gen"]]
    agg["gate_requests"] = len(gate_set)
    agg["timing_all"] = timing(results, cfg)
    agg["timing_gate"] = timing(gate_set, cfg)
    agg["timing_greedy"] = timing([r for r in gate_set if r["greedy"]], cfg)
    agg["timing_sampled"] = timing([r for r in gate_set if not r["greedy"]], cfg)
    return agg


def decide(agg, cfg):
    """The verdict: (label, reason). label is BUILD, DO NOT BUILD or NOT ENOUGH DATA."""
    have, need = agg["gate_requests"], cfg["min_requests"]
    best = agg["timing_gate"]["best"]
    if have < need:
        return "NOT ENOUGH DATA", "%d request(s) with >= %d generated tokens, need %d" % (have, cfg["gate_min_gen"], need)
    if best is not None and best[1] >= cfg["min_gain"]:
        return "BUILD", "best hybrid T=%d is %+.1f%% faster than DFlash alone (>= %+.1f%%)" % (
            best[0], 100 * best[1], 100 * cfg["min_gain"])
    return "DO NOT BUILD", "best hybrid%s vs DFlash alone, needs >= %+.1f%%" % (
        " (T=%d) is %+.1f%%" % (best[0], 100 * best[1]) if best else " n/a", 100 * cfg["min_gain"])


def bucket_name(lo, hi):
    return "%d+" % lo if hi is None else ("%d" % lo if lo == hi else "%d-%d" % (lo, hi))


def fmt_pct(x):
    return "%5.1f%%" % (100 * x)


def report(results, agg, cfg, rows, out=sys.stdout):
    w = out.write
    ks = cfg["ks"]
    k0 = GATE_K
    if rows != 0:
        shown = results if rows < 0 else results[:rows]
        w("per request (tok/round = generated tokens / rounds)\n")
        w("%4s %-18s %8s %6s %-6s %5s %9s %9s %8s %8s %9s\n" % (
            "#", "request", "prompt", "gen", "spec", "temp", "dflash", "ngram k=%d" % k0, "rnd-hit", "tok-hit", "run>=%d" % GATE_RUN))
        for i, r in enumerate(shown, 1):
            n = max(r["gen_tokens"] - 1, 0)
            nk = r["ngram"][k0]
            rid = (r["request_id"] or "?")[-18:]
            w("%4d %-18s %8d %6d %-6s %5s %8.2f%s %9.2f %8s %8s %9s\n" % (
                i, rid, r["prompt_tokens"], r["gen_tokens"], r["speculative"] or "-",
                "%g" % r["temperature"] if r["temperature"] is not None else "-",
                r["dflash_tpr"], "" if r["dflash_src"] == "log" else "*",
                tpr(r["gen_tokens"], nk["rounds"]), fmt_pct(tpr(nk["hit"], nk["rounds"])),
                fmt_pct(tpr(r["covered"], n)), fmt_pct(tpr(sum(x for x in r["runs"] if x >= GATE_RUN), n))))
        if len(shown) < len(results):
            w("  ... %d more (--rows 0 prints all)\n" % (len(results) - len(shown)))
        if any(r["dflash_src"] != "log" for r in shown):
            w("  * no DFlash rounds in the log for this request: the --dflash-tpr constant\n")
        w("\n")

    greedy = sum(1 for r in results if r["greedy"])
    w("overall: %d requests (%d greedy, %d sampled), %d generated tokens\n" % (
        agg["requests"], greedy, agg["requests"] - greedy, agg["tokens"]))
    w("  proposer: longest suffix %d..%d tokens, most recent occurrence%s\n" % (
        cfg["min_n"], cfg["max_n"], ", source within %d tokens" % cfg["window"] if cfg["window"] else ""))
    w("  DFlash as logged: %.2f tok/round (%d of %d requests have logged rounds, the rest use %.2f)\n" % (
        agg["dflash_tpr"], agg["dflash_logged"], agg["requests"], cfg["dflash_tpr"]))
    for k in ks:
        a = agg["ngram"][k]
        w("  n-gram alone, k=%d%s: %.2f tok/round (mean over requests %.2f), proposal in %s of rounds, "
          "round hit %s\n" % (k, "" if k <= GATE_K else " (beyond the 8-row window: what-if)", a["tpr"],
                              a["mean_request_tpr"], fmt_pct(a["proposal_rate"]).strip(), fmt_pct(a["round_hit"]).strip()))
    w("  copy runs: %s of the tokens after the first are inside a run, %s inside runs of %d+ tokens\n" % (
        fmt_pct(agg["token_hit"]).strip(), fmt_pct(agg["run_share"]).strip(), GATE_RUN))
    w("    run length: " + ", ".join("%s: %d runs / %s of tokens" % (bucket_name(*b), c, fmt_pct(s).strip())
                                      for b, c, _, s in agg["run_hist"]) + "\n")
    w("  hybrid (n-gram k=%d when match >= T, else DFlash), tok/round vs DFlash alone %.2f"
      " (rounds are not equally long: see the timing below):\n" % (cfg["hybrid_k"], agg["dflash_tpr"]))
    for t in cfg["ts"]:
        parts = []
        for mode in ("positional", "mean"):
            h = agg["hybrid"][(t, mode)]
            parts.append("%s %.2f (%+.1f%%, n-gram in %s of rounds)" % (
                mode, h["tpr"], 100 * (h["tpr"] / agg["dflash_tpr"] - 1) if agg["dflash_tpr"] else 0.0,
                fmt_pct(h["ngram_share"]).strip()))
        w("    T=%d: %s\n" % (t, "; ".join(parts)))

    # ---- timing: the part the verdict rests on
    dm = dflash_round_ms(cfg)
    w("\ntiming model (ms; --verify-ms-base/--verify-ms-per-row/--drafter-ms/--inject-ms-per-round):\n")
    w("  verify(rows) = %.2f + %.3f * rows; plain decode (1 row) %.1f; DFlash round = verify(%d rows) + drafter %.2f = %.1f;"
      " hybrid n-gram round = verify(proposal + 1 rows) + %.3f injection\n" % (
          cfg["verify_base"], cfg["verify_row"], verify_ms(cfg, 1), GATE_K + 1, cfg["drafter_ms"], dm, cfg["inject_round"]))

    def tline(label, tm):
        if not tm["requests"]:
            w("  %-34s no requests\n" % label)
            return
        w("  %-34s %3d requests, %6d decode tokens: DFlash alone %.1f tok/s\n" % (
            label, tm["requests"], tm["tokens"], tm["dflash_tps"]))
        for k in ks:
            w("      n-gram alone k=%d (info): %.1f tok/s (%+.1f%%)\n" % (
                k, tm["ngram_tps"][k], 100 * (tm["ngram_tps"][k] / tm["dflash_tps"] - 1) if tm["dflash_tps"] else 0.0))
        for t in cfg["ts"]:
            h = tm["hybrid"][t]
            w("      hybrid T=%d: positional %.1f, mean %.1f tok/s -> %+.1f%% (lower of the two)%s\n" % (
                t, h["positional"], h["mean"], 100 * h["gain"], "  <- best" if tm["best"] and tm["best"][0] == t else ""))

    w("estimated decode speed, requests with >= %d generated tokens (the gate set):\n" % cfg["gate_min_gen"])
    tline("all", agg["timing_gate"])
    tline("  greedy (exact)", agg["timing_greedy"])
    tline("  sampled (estimate)", agg["timing_sampled"])
    if agg["timing_all"]["requests"] != agg["timing_gate"]["requests"]:
        w("(all %d requests, including the short ones, for reference:)\n" % agg["timing_all"]["requests"])
        tline("all requests", agg["timing_all"])
    if agg["dflash_logged"] < agg["requests"]:
        w("  note: %d request(s) have no logged DFlash rounds; DFlash alone is priced at --dflash-tpr %.2f for them\n" % (
            agg["requests"] - agg["dflash_logged"], cfg["dflash_tpr"]))

    label, reason = decide(agg, cfg)
    best = agg["timing_gate"]["best"]
    have = agg["gate_requests"]
    w("\ngate (decides): the best hybrid must beat DFlash alone by >= %.1f%% on >= %d requests of >= %d tokens\n" % (
        100 * cfg["min_gain"], cfg["min_requests"], cfg["gate_min_gen"]))
    w("  requests with >= %d tokens: %d vs >= %d ........ %s\n" % (
        cfg["gate_min_gen"], have, cfg["min_requests"], "PASS" if have >= cfg["min_requests"] else "FAIL"))
    if best is not None:
        w("  best hybrid T=%d: %+.1f%% vs >= %+.1f%% ........ %s\n" % (
            best[0], 100 * best[1], 100 * cfg["min_gain"], "PASS" if best[1] >= cfg["min_gain"] else "FAIL"))
    exact = " (only greedy requests are exact; %d sampled are estimates)" % (agg["requests"] - greedy) if agg["requests"] > greedy else ""
    w("info, the 2026-10-03 gates (they no longer decide: n-gram alone passing them does not make the hybrid faster)%s:\n" % exact)
    tpr7 = agg["ngram"][GATE_K]["tpr"]
    a_ok = tpr7 >= GATE_TPR
    b_ok = agg["run_share"] > GATE_RUN_SHARE
    w("  A  n-gram k=%d: %.2f tok/round vs %.1f ........ %s\n" % (GATE_K, tpr7, GATE_TPR, "PASS" if a_ok else "FAIL"))
    w("  B  tokens in runs of %d+: %s vs > %d%% ........ %s\n" % (
        GATE_RUN, fmt_pct(agg["run_share"]).strip(), int(100 * GATE_RUN_SHARE), "PASS" if b_ok else "FAIL"))
    verdict_text = {"BUILD": "BUILD the n-gram hybrid", "DO NOT BUILD": "DO NOT BUILD the n-gram hybrid",
                    "NOT ENOUGH DATA": "NOT ENOUGH DATA to decide"}[label]
    w("verdict: %s (%s)\n" % (verdict_text, reason))
    return label


def to_jsonable(results, agg):
    def conv(o):
        if isinstance(o, dict):
            return {(" ".join(map(str, k)) if isinstance(k, tuple) else str(k)): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o
    slim = [dict({k: v for k, v in r.items() if k != "runs"}, runs=Counter(r["runs"]).most_common()) for r in results]
    return {"overall": conv(agg), "requests": conv(slim)}


# ---- self-test / example ---------------------------------------------------------------------------

def _rand_tokens(rng, n, vocab):
    return [rng.randrange(vocab) for _ in range(n)]


def synthetic_records(seed=7):
    """A few requests with known structure, as (record fields, prompt, generated). The conversation
    shape of an agent: request 2's prompt is request 1's prompt + reply + a new tool result."""
    rng = random.Random(seed)
    vocab = 248320
    block = _rand_tokens(rng, 60, vocab)  # text the model will quote
    pre = _rand_tokens(rng, 1, vocab)[0]
    # 1: a pure quote: the prompt holds [pre block...], the reply is the block again (after `pre`).
    p1 = _rand_tokens(rng, 80, vocab) + [pre] + block + _rand_tokens(rng, 30, vocab) + [pre]
    g1 = list(block[:40])
    # 2: the next turn: shares p1 + g1, then a new tool result; the reply is fresh text (no copying).
    p2 = p1 + g1 + _rand_tokens(rng, 50, vocab)
    g2 = _rand_tokens(rng, 30, vocab)
    # 3: unrelated conversation, sampled, a reply that repeats a 5-token phrase three times.
    phrase = _rand_tokens(rng, 5, vocab)
    p3 = _rand_tokens(rng, 100, vocab)
    g3 = _rand_tokens(rng, 8, vocab) + phrase + _rand_tokens(rng, 3, vocab) + phrase + _rand_tokens(rng, 3, vocab) + phrase
    recs = [
        dict(prompt=p1, gen=g1, temperature=0.0, speculative="dflash", acc=[3, 2, 4, 0, 1, 5, 2, 1, 3, 0, 2, 4]),
        dict(prompt=p2, gen=g2, temperature=0.0, speculative="dflash", acc=[0, 1, 0, 2, 0, 0, 1, 3, 0, 1, 0, 0, 2, 1, 0, 2]),
        dict(prompt=p3, gen=g3, temperature=0.6, speculative="none", acc=[]),
    ]
    return recs


def encode_log(recs):
    """The lines r4dx-server writes for `recs` (only the fields the simulator and a reader need, in the
    writer's delta encoding: prompt_ids past the part shared with the previous prompt + generated)."""
    lines = []
    prev = []
    for i, r in enumerate(recs):
        p, g = r["prompt"], r["gen"]
        shared = 0
        for a, b in zip(prev, p):
            if a != b:
                break
            shared += 1
        acc = list(r["acc"])
        # drafted per round: the sizes are not simulated, a dflash k=7 round proposes up to 7
        line = {
            "ts": "2026-10-08T10:%02d:00.000+07:00" % i,
            "request_id": "chatcmpl-synthetic%04d" % i,
            "endpoint": "chat/completions", "stream": True, "http_status": 200, "error_status": None,
            "finish_reason": "stop", "cancelled": False, "temperature": r["temperature"],
            "prompt_tokens": len(p), "completion_tokens": len(g),
            "speculative": r["speculative"],
            "draft_k": 7 if r["speculative"] == "dflash" else None,
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": None,
            "prompt_shared": shared, "prompt_ids": p[shared:], "generated_ids": g,
            "round_drafted": [7] * len(acc), "round_accepted": acc,
        }
        lines.append(json.dumps(line, separators=(",", ":")))
        prev = p + g
    return lines


def _traffic_rec(i, prompt, gen, temperature, acc):
    return {"_prompt": prompt, "_gen": gen, "_where": "selftest:%d" % i, "request_id": "selftest%03d" % i,
            "temperature": temperature, "speculative": "dflash", "round_accepted": acc}


def _copy_traffic(n_req, gen_len=80, seed=11):
    """Heavy copy traffic: every reply quotes a block of the prompt, and DFlash (as logged) only gets 3
    tokens per round on it, so an n-gram round (8 tokens, no drafter) is both longer and cheaper."""
    rng = random.Random(seed)
    out = []
    for i in range(n_req):
        block = _rand_tokens(rng, gen_len, 248320)
        prompt = _rand_tokens(rng, 200, 248320) + block + _rand_tokens(rng, 20, 248320)
        acc = [2] * ((gen_len - 1) // 3) + ([(gen_len - 1) % 3 - 1] if (gen_len - 1) % 3 else [])
        out.append(_traffic_rec(i, prompt, list(block), 0.0 if i % 2 == 0 else 0.7, acc))
    return out


def _dflash_great_traffic(n_req, seed=12):
    """The smoke-like case: DFlash already gets 6 tokens per round, and the reply is fresh text over a small
    vocabulary, so n-gram matches of 3+ tokens exist by chance and are almost never right."""
    rng = random.Random(seed)
    out = []
    for i in range(n_req):
        prompt = _rand_tokens(rng, 300, 12)
        gen = _rand_tokens(rng, 120, 12)
        out.append(_traffic_rec(i, prompt, gen, 0.0 if i % 3 else 0.7, [5] * 20))
    return out


def _steal_traffic(n_req, seed=21):
    """The n-gram takes the 80 quoted tokens where DFlash was great (10 logged rounds of 8) and leaves the
    80 fresh ones where it got 4 per round: the hybrid is only a little faster than DFlash alone. Charging
    the request's mean tok/round to the DFlash rounds after the stretch (5.3 per round here, above the
    4 the log shows there) overstates the hybrid's speed by 17%."""
    rng = random.Random(seed)
    out = []
    for i in range(n_req):
        block = _rand_tokens(rng, 80, 248320)
        prompt = _rand_tokens(rng, 200, 248320) + block + _rand_tokens(rng, 20, 248320)
        out.append(_traffic_rec(i, prompt, block + _rand_tokens(rng, 80, 248320), 0.0, [7] * 10 + [3] * 20))
    return out


def _partial_copy_traffic(n_req, period=8, seed=22):
    """The smoke log's shape: a quoted block with every `period`-th token changed. The n-gram alone gets
    ~2.7 tok/round (the old gate A passes) while DFlash gets 6, so any hybrid that hands rounds to the
    n-gram is slower."""
    rng = random.Random(seed)
    out = []
    for i in range(n_req):
        block = _rand_tokens(rng, 120, 248320)
        prompt = _rand_tokens(rng, 200, 248320) + block + _rand_tokens(rng, 20, 248320)
        gen = [_rand_tokens(rng, 1, 248320)[0] if j % period == period - 1 else t for j, t in enumerate(block)]
        out.append(_traffic_rec(i, prompt, gen, 0.0, [5] * 20))
    return out


def selftest_verdicts(check):
    """The timing-aware verdict: heavy copy traffic -> BUILD; DFlash already great -> DO NOT BUILD; too few
    requests (or too short ones) -> NOT ENOUGH DATA."""
    cfg = make_cfg(build_parser().parse_args(["--hybrid-t", "3,4,5"]))

    def run(recs):
        results = [analyse(r, cfg) for r in recs]
        agg = aggregate(results, cfg)
        return results, agg, decide(agg, cfg)

    check(abs(dflash_round_ms(cfg) - 39.8) < 1e-9 and abs(verify_ms(cfg, 1) - 27.2) < 1e-9,
          "default timing model: round %.2f ms, plain %.2f ms" % (dflash_round_ms(cfg), verify_ms(cfg, 1)))
    # (a) heavy copy traffic: the hybrid is much faster than DFlash alone
    results, agg, (label, _) = run(_copy_traffic(24))
    best = agg["timing_gate"]["best"]
    check(label == "BUILD" and best[1] > 0.5, "copy traffic: %s, best %r" % (label, best))
    check(agg["timing_greedy"]["requests"] == 12 and agg["timing_sampled"]["requests"] == 12,
          "greedy/sampled split: %d/%d" % (agg["timing_greedy"]["requests"], agg["timing_sampled"]["requests"]))
    # (b) DFlash already great, chance n-gram matches only cost time: not worth building
    results, agg, (label, _) = run(_dflash_great_traffic(24))
    best = agg["timing_gate"]["best"]
    check(label == "DO NOT BUILD" and best[1] < 0.05, "DFlash-great traffic: %s, best %r" % (label, best))
    check(any(h["tpr"] < agg["dflash_tpr"] for h in agg["hybrid"].values()), "hybrid should lose tok/round here")
    # (b2) the regression the old gates had: the n-gram alone passes gate A (>= 2.5 tok/round at k=7) but
    # DFlash gets 6, so the hybrid is much slower. The verdict must follow the time, not the gate.
    results, agg, (label, _) = run(_partial_copy_traffic(24))
    best = agg["timing_gate"]["best"]
    check(agg["ngram"][GATE_K]["tpr"] >= GATE_TPR and label == "DO NOT BUILD" and best[1] < 0,
          "n-gram passes gate A, hybrid slower: %s, tpr %.2f, best %r" % (label, agg["ngram"][GATE_K]["tpr"], best))
    # (b3) the n-gram takes the stretches where DFlash was great: after one the DFlash rounds must come from
    # the log (the rest of the covering round), not from the request's mean, which says BUILD here.
    results, agg, (label, _) = run(_steal_traffic(24))
    h = agg["timing_gate"]["hybrid"][3]
    check(label == "DO NOT BUILD" and 0 <= h["gain"] < 0.05 and h["mean"] > 1.1 * h["positional"],
          "stolen rounds: %s, positional %.1f mean %.1f tok/s, gain %.3f" % (label, h["positional"], h["mean"], h["gain"]))
    # (b4) pricing: a --dflash-k 3 log is priced at 4 rows, an --mtp log is not a DFlash log
    rec = _copy_traffic(1)[0]
    rec["draft_k"] = 3
    r = analyse(rec, cfg)
    check(abs(r["dflash_ms"] - len(rec["round_accepted"]) * dflash_round_ms(cfg, 3)) < 1e-6
          and dflash_round_ms(cfg, 3) < dflash_round_ms(cfg), "draft_k pricing")
    rec = _copy_traffic(1)[0]
    rec["speculative"] = "mtp"
    check(analyse(rec, cfg)["dflash_src"] == "const", "an mtp log must not be priced as DFlash rounds")
    # injection is a cost: a free injection never makes a hybrid slower, a dear one never faster
    cheap = make_cfg(build_parser().parse_args(["--inject-ms-per-round", "0"]))
    dear = make_cfg(build_parser().parse_args(["--inject-ms-per-round", "5"]))
    recs = _copy_traffic(1)
    check(analyse(recs[0], cheap)["hybrid"][(3, "positional")]["ms"] < analyse(recs[0], cfg)["hybrid"][(3, "positional")]["ms"]
          < analyse(recs[0], dear)["hybrid"][(3, "positional")]["ms"], "injection cost must raise the hybrid time")
    # (c) too few requests, or requests too short to count: no verdict, however good the numbers look
    results, agg, (label, _) = run(_copy_traffic(5))
    check(label == "NOT ENOUGH DATA" and agg["timing_gate"]["best"][1] > 0.5, "5 copy requests: %s" % label)
    short = _copy_traffic(24, gen_len=20)
    results, agg, (label, _) = run(short)
    check(label == "NOT ENOUGH DATA" and agg["gate_requests"] == 0, "short requests: %s, %d counted" % (label, agg["gate_requests"]))
    # the threshold is a flag: the same copy traffic fails a 10x gain requirement
    cfg2 = make_cfg(build_parser().parse_args(["--min-gain", "10"]))
    agg2 = aggregate([analyse(r, cfg2) for r in _copy_traffic(24)], cfg2)
    check(decide(agg2, cfg2)[0] == "DO NOT BUILD", "--min-gain 10 must reject")
    # the positional model equals the log exactly when the hybrid never leaves DFlash (T above any match)
    cfg3 = make_cfg(build_parser().parse_args(["--hybrid-t", "99"]))
    r3 = analyse(_copy_traffic(1)[0], cfg3)
    check(abs(r3["hybrid"][(99, "positional")]["ms"] - r3["dflash_ms"]) < 1e-6, "T=99 positional != DFlash as logged")


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # 1. the optimised proposer equals the by-definition one on random small-vocab text (many repeats).
    rng = random.Random(1)
    for trial in range(60):
        vocab = rng.choice([3, 5, 12])
        prompt = _rand_tokens(rng, rng.randrange(0, 60), vocab)
        gen = _rand_tokens(rng, rng.randrange(1, 60), vocab)
        window = rng.choice([0, 0, 7, 25])
        min_n, max_n = rng.choice([(2, 5), (1, 3), (3, 3)])
        check(build_match_table(prompt, gen, min_n, max_n, window) == brute_match_table(prompt, gen, min_n, max_n, window),
              "match table differs from the brute-force definition (trial %d)" % trial)

    # 2. a quote of 40 tokens at k=7: after the first token every round accepts 7 and emits 8:
    #    39 tokens after the first -> 5 rounds -> 40 / 5 = 8 tok/round. A reply with no repeats: 1.0.
    recs = synthetic_records()
    prompt, gen = recs[0]["prompt"], recs[0]["gen"]
    table = build_match_table(prompt, gen, 2, 5)
    check(sim_ngram(prompt, gen, table, 7) == (5, 5, 5, 40), "quote k=7: %r" % (sim_ngram(prompt, gen, table, 7),))
    check(sim_ngram(prompt, gen, table, 3)[0] == 10, "quote k=3 rounds: %r" % (sim_ngram(prompt, gen, table, 3),))
    check(copy_runs(prompt, gen, table) == [39], "quote runs: %r" % (copy_runs(prompt, gen, table),))
    p2, g2 = recs[1]["prompt"], recs[1]["gen"]
    t2 = build_match_table(p2, g2, 2, 5)
    check(sim_ngram(p2, g2, t2, 7) == (29, 0, 0, 29), "fresh text: %r" % (sim_ngram(p2, g2, t2, 7),))
    # the repeated 5-token phrase: the 2nd and 3rd copies are found after their first two tokens, so
    # each contributes a run of the remaining 3 (the run stops where the filler differs)
    p3, g3 = recs[2]["prompt"], recs[2]["gen"]
    t3 = build_match_table(p3, g3, 2, 5)
    check(copy_runs(p3, g3, t3) == [3, 3], "phrase runs: %r" % (copy_runs(p3, g3, t3),))

    # 3. hybrid: T above the longest match = DFlash alone, positional = the logged tok/round exactly.
    log_rounds = dflash_rounds_of({"round_accepted": recs[0]["acc"], "speculative": "dflash"}, len(gen))
    r_never = sim_hybrid(prompt, gen, table, 7, 99, log_rounds, DFLASH_TPR)
    check(r_never[1] == 0 and r_never[0] == len(recs[0]["acc"]) + (0), "hybrid without n-gram rounds: %r" % (r_never,))
    r_always = sim_hybrid(prompt, gen, table, 7, 2, log_rounds, DFLASH_TPR)
    check(r_always[0] <= 5 + 1 and r_always[2] <= 1, "hybrid T=2 on a quote: %r" % (r_always,))
    r_const = sim_hybrid(p2, g2, t2, 7, 2, None, 3.0)  # no matches: pure constant DFlash, 29 tokens / 3
    check(r_const[1] == 0 and r_const[0] == 10, "constant-tpr hybrid: %r" % (r_const,))

    # 4. the log round trip: the writer's delta encoding is rebuilt exactly, torn and orphaned lines skipped.
    lines = encode_log(recs)
    check(json.loads(lines[1])["prompt_shared"] == len(recs[0]["prompt"]) + len(recs[0]["gen"]),
          "request 2 should share request 1's prompt + reply")
    check(len(json.loads(lines[1])["prompt_ids"]) == 50, "only the new tool result is stored")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:2]) + "\n")
            f.write('{"torn": \n')                # a torn line breaks the chain ...
            f.write(lines[2] + "\n")              # ... but request 3 is whole (prompt_shared == 0)
            f.write(json.dumps({"http_status": 400}) + "\n")
        stats = Counter()
        got = list(read_requests([path], stats))
        check([r["_prompt"] for r in got] == [x["prompt"] for x in recs], "rebuilt prompts differ")
        check([r["_gen"] for r in got] == [x["gen"] for x in recs], "rebuilt generations differ")
        check(stats["unreadable_lines"] == 1 and stats["lines_without_tokens"] == 1 and stats["unrebuildable"] == 0,
              "stats %r" % (dict(stats),))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join([lines[1]]) + "\n")  # request 2 alone: its head is in a line we do not have
        stats = Counter()
        check(list(read_requests([path], stats)) == [] and stats["unrebuildable"] == 1, "orphan not skipped")

        # 5. end to end on the synthetic log: the report runs and the numbers hang together.
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        cfg = make_cfg(build_parser().parse_args(["--k", "7,9", "--hybrid-t", "2,3,4"]))
        results = [analyse(r, cfg) for r in read_requests([path], Counter())]
        agg = aggregate(results, cfg)
        check(agg["requests"] == 3 and agg["tokens"] == 40 + 30 + len(recs[2]["gen"]), "aggregate counts")
        check(agg["ngram"][7]["tpr"] > 1.0 and agg["ngram"][9]["tpr"] >= agg["ngram"][7]["tpr"] - 1e-9,
              "k=9 must not be worse than k=7 on these requests: %r" % (agg["ngram"],))
        logged = sum(len(r["acc"]) for r in recs[:2])
        check(abs(agg["dflash_tpr"] - (40 + 30 + len(recs[2]["gen"])) / (logged + (len(recs[2]["gen"]) / DFLASH_TPR))) < 1e-9,
              "aggregate dflash tok/round")
        import io
        buf = io.StringIO()
        report(results, agg, cfg, rows=-1, out=buf)
        text = buf.getvalue()
        check("verdict:" in text and "per request" in text and "hybrid" in text, "report text")
        check("NOT ENOUGH DATA" in text, "3 requests are not enough data to decide")
        sys.stdout.write(text)
    selftest_verdicts(check)
    if failures:
        sys.stderr.write("SELFTEST FAILED:\n  " + "\n  ".join(failures) + "\n")
        return 1
    sys.stdout.write("\nselftest passed\n")
    return 0


# ---- main --------------------------------------------------------------------------------------------

def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def make_cfg(args):
    ks = sorted(set(parse_ints(args.k)) | {GATE_K})
    ts = parse_ints(args.hybrid_t)
    return {"min_n": args.min_n, "max_n": args.max_n, "ks": ks, "ts": ts, "window": args.window,
            "hybrid_k": GATE_K, "dflash_tpr": args.dflash_tpr,
            "verify_base": args.verify_ms_base, "verify_row": args.verify_ms_per_row, "drafter_ms": args.drafter_ms,
            "inject_round": args.inject_ms_per_round, "min_gain": args.min_gain, "min_requests": args.min_requests,
            "gate_min_gen": args.gate_min_gen}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="request log file(s), written with --request-log-tokens")
    ap.add_argument("--min-n", type=int, default=2, help="shortest suffix to match (default 2)")
    ap.add_argument("--max-n", type=int, default=5, help="longest suffix to match (default 5)")
    ap.add_argument("--k", default="7,9", help="draft lengths to report, comma separated (default 7,9; 7 is always run: it is the gate's)")
    ap.add_argument("--hybrid-t", default="3,4,5", help="hybrid match-length thresholds T, comma separated (default 3,4,5)")
    ap.add_argument("--dflash-tpr", type=float, default=DFLASH_TPR,
                    help="DFlash tok/round for requests whose log has no DFlash rounds (default %.2f)" % DFLASH_TPR)
    ap.add_argument("--window", type=int, default=0, help="only match sources within the last N tokens (default 0 = whole context)")
    ap.add_argument("--min-gen", type=int, default=8, help="skip requests that generated fewer tokens (default 8)")
    ap.add_argument("--greedy-only", action="store_true", help="only temperature <= 0 requests (the exact ones)")
    ap.add_argument("--rows", type=int, default=30, help="per-request rows to print (default 30, 0 none, -1 all)")
    ap.add_argument("--json", metavar="PATH", help="also write the per-request and overall numbers as JSON")
    g = ap.add_argument_group("timing model and gate (defaults: the repo's TP1 measurements, see the module doc)")
    g.add_argument("--verify-ms-base", type=float, default=VERIFY_MS_BASE,
                   help="verify forward, fixed part, ms (default %.1f)" % VERIFY_MS_BASE)
    g.add_argument("--verify-ms-per-row", type=float, default=VERIFY_MS_PER_ROW,
                   help="verify forward, ms per row in the window (default %.1f)" % VERIFY_MS_PER_ROW)
    g.add_argument("--drafter-ms", type=float, default=DRAFTER_MS,
                   help="DFlash drafter forward + glue per round, ms (default %.1f: a k=7 round is then 39.8)" % DRAFTER_MS)
    g.add_argument("--inject-ms-per-round", type=float, default=INJECT_MS_PER_ROUND,
                   help="hybrid only: drafter feature injection after each n-gram round, ms (default %.3f)" % INJECT_MS_PER_ROUND)
    g.add_argument("--min-gain", type=float, default=MIN_GAIN,
                   help="BUILD needs the best hybrid this much faster than DFlash alone, as a fraction (default %.2f)" % MIN_GAIN)
    g.add_argument("--min-requests", type=int, default=MIN_REQUESTS,
                   help="... on at least this many requests (default %d); fewer: NOT ENOUGH DATA" % MIN_REQUESTS)
    g.add_argument("--gate-min-gen", type=int, default=GATE_MIN_GEN,
                   help="... of at least this many generated tokens (default %d)" % GATE_MIN_GEN)
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic log and exit")
    ap.add_argument("--write-example", metavar="PATH", help="write the small synthetic example log and exit")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.write_example:
        with open(args.write_example, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(encode_log(synthetic_records())) + "\n")
        print("wrote", args.write_example)
        return 0
    if not args.logs:
        ap.error("give a request log (or --selftest)")
    if not (1 <= args.min_n <= args.max_n):
        ap.error("need 1 <= --min-n <= --max-n")

    cfg = make_cfg(args)
    stats = Counter()
    results = []
    for rec in read_requests(args.logs, stats):
        if rec.get("http_status") != 200 or rec.get("error_status") is not None:
            stats["failed_requests"] += 1
            continue
        if len(rec["_gen"]) < args.min_gen:
            stats["too_short"] += 1
            continue
        if args.greedy_only and (rec.get("temperature") or 0) > 0:
            stats["sampled_skipped"] += 1
            continue
        results.append(analyse(rec, cfg))
    sys.stderr.write("log: %s\n" % ", ".join("%s=%d" % kv for kv in sorted(stats.items())))
    if not results:
        sys.stderr.write("no request to simulate (was the server started with --request-log-tokens?)\n")
        return 2
    agg = aggregate(results, cfg)
    report(results, agg, cfg, args.rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(results, agg), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
