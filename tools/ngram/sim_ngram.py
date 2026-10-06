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
              (round_accepted; "positional") or the request's mean logged tok/round ("mean"), and
              --dflash-tpr when the log has no DFlash rounds. An n-gram round is counted as one round
              like a DFlash round, although it is cheaper (no drafter forward).

Gate (from the 2026-10-03 research): build the n-gram path if standalone n-gram reaches >= 2.5 tok/round
at k=7 (the verify window is capped at 8 rows), or if more than 50% of the generated tokens sit in copy
runs of 10+ tokens.

  python sim_ngram.py requests.jsonl
  python sim_ngram.py requests.jsonl --k 7,9 --hybrid-t 3,4,5 --greedy-only --rows 0
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
    """Standalone n-gram rounds at draft cap k. Returns (rounds, rounds_with_proposal, rounds_hit)."""
    P, n = len(prompt), len(gen)
    full = prompt + gen
    i = 1
    rounds = proposed = hit = 0
    while i < n:
        m = table[i]
        a = 0
        if m is not None:
            prop = proposal(full, P, i, m, k)
            a = accepted(prop, gen, i)
            proposed += 1
            hit += a > 0
        i += min(a + 1, n - i)
        rounds += 1
    return rounds, proposed, hit


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
    """(starts, ends) of the log's DFlash/MTP rounds in generated-token index space, or None. Round r
    emitted round_accepted[r] + 1 tokens, the first one starting at index 1."""
    acc = rec.get("round_accepted") or []
    if not acc or rec.get("speculative") not in ("dflash", "mtp"):
        return None
    starts, ends, s = [], [], 1
    for a in acc:
        starts.append(s)
        s += a + 1
        ends.append(s)
    return starts, ends


def sim_hybrid(prompt, gen, table, k, t_min, rounds_log, tpr):
    """n-gram when the match is >= t_min tokens long, else a DFlash round. rounds_log: dflash_rounds_of
    (positional) or None, in which case a DFlash round emits `tpr` tokens on average (carried
    fractionally). Returns (rounds, ngram_rounds, dflash_rounds)."""
    P, n = len(prompt), len(gen)
    full = prompt + gen
    i = 1
    rounds = ng = df = 0
    carry = 0.0
    while i < n:
        m = table[i]
        if m is not None and m[0] >= t_min:
            a = accepted(proposal(full, P, i, m, k), gen, i)
            emit = a + 1
            ng += 1
        else:
            if rounds_log is not None:
                starts, ends = rounds_log
                r = bisect.bisect_right(starts, i) - 1
                emit = ends[r] - i if r >= 0 and ends[r] > i else 1
            else:
                carry += tpr
                emit = max(1, int(carry))
                carry -= emit
            df += 1
        i += min(emit, n - i)
        rounds += 1
    return rounds, ng, df


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
    # DFlash as logged (or the constant when the server ran without a drafter).
    log_rounds = dflash_rounds_of(rec, n)
    if log_rounds is not None:
        res["dflash_rounds"] = len(log_rounds[0])
        res["dflash_src"] = "log"
        res["dflash_tpr"] = n / len(log_rounds[0])
        mean_tpr = res["dflash_tpr"]
    else:
        res["dflash_rounds"] = None
        res["dflash_src"] = "const"
        res["dflash_tpr"] = cfg["dflash_tpr"]
        mean_tpr = cfg["dflash_tpr"]
    res["ngram"] = {}
    for k in cfg["ks"]:
        rounds, proposed, hit = sim_ngram(prompt, gen, table, k)
        res["ngram"][k] = {"rounds": rounds, "proposed": proposed, "hit": hit}
    runs = copy_runs(prompt, gen, table)
    res["runs"] = runs
    res["covered"] = sum(runs)
    res["hybrid"] = {}
    for t in cfg["ts"]:
        for mode in ("positional", "mean"):
            if mode == "positional":
                r, ng, df = sim_hybrid(prompt, gen, table, cfg["hybrid_k"], t, log_rounds, cfg["dflash_tpr"])
            else:
                r, ng, df = sim_hybrid(prompt, gen, table, cfg["hybrid_k"], t, None, mean_tpr)
            res["hybrid"][(t, mode)] = {"rounds": r, "ngram_rounds": ng, "dflash_rounds": df}
    return res


# ---- aggregation / report ----------------------------------------------------------------------------

def tpr(tokens, rounds):
    return tokens / rounds if rounds else 0.0


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
    return agg


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
            w("  * no DFlash/MTP rounds in the log for this request: the --dflash-tpr constant\n")
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
    w("  hybrid (n-gram k=%d when match >= T, else DFlash), tok/round vs DFlash alone %.2f:\n" % (cfg["hybrid_k"], agg["dflash_tpr"]))
    best = None
    for t in cfg["ts"]:
        parts = []
        for mode in ("positional", "mean"):
            h = agg["hybrid"][(t, mode)]
            parts.append("%s %.2f (%+.1f%%, n-gram in %s of rounds)" % (
                mode, h["tpr"], 100 * (h["tpr"] / agg["dflash_tpr"] - 1) if agg["dflash_tpr"] else 0.0,
                fmt_pct(h["ngram_share"]).strip()))
            if mode == "positional" and (best is None or h["tpr"] > best[1]):
                best = (t, h["tpr"])
        w("    T=%d: %s\n" % (t, "; ".join(parts)))
    exact = " (only greedy requests are exact; %d sampled are estimates)" % (agg["requests"] - greedy) if agg["requests"] > greedy else ""
    w("\ngate (n-gram alone at k=%d >= %.1f tok/round, or > %d%% of tokens in runs of %d+)%s\n" % (
        GATE_K, GATE_TPR, int(100 * GATE_RUN_SHARE), GATE_RUN, exact))
    tpr7 = agg["ngram"][GATE_K]["tpr"]
    a_ok = tpr7 >= GATE_TPR
    b_ok = agg["run_share"] > GATE_RUN_SHARE
    w("  A  n-gram k=%d: %.2f tok/round vs %.1f ........ %s\n" % (GATE_K, tpr7, GATE_TPR, "PASS" if a_ok else "FAIL"))
    w("  B  tokens in runs of %d+: %s vs > %d%% ........ %s\n" % (
        GATE_RUN, fmt_pct(agg["run_share"]).strip(), int(100 * GATE_RUN_SHARE), "PASS" if b_ok else "FAIL"))
    if best is not None:
        w("  info: best hybrid T=%d gives %.2f tok/round, %+.1f%% over DFlash alone (n-gram rounds skip the drafter forward)\n" % (
            best[0], best[1], 100 * (best[1] / agg["dflash_tpr"] - 1) if agg["dflash_tpr"] else 0.0))
    w("verdict: %s\n" % ("BUILD the n-gram path (gate met)" if (a_ok or b_ok) else
                         "DO NOT BUILD standalone n-gram (gate not met); judge a hybrid on the uplift above"))
    return a_ok or b_ok


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
    check(sim_ngram(prompt, gen, table, 7) == (5, 5, 5), "quote k=7: %r" % (sim_ngram(prompt, gen, table, 7),))
    check(sim_ngram(prompt, gen, table, 3)[0] == 10, "quote k=3 rounds: %r" % (sim_ngram(prompt, gen, table, 3),))
    check(copy_runs(prompt, gen, table) == [39], "quote runs: %r" % (copy_runs(prompt, gen, table),))
    p2, g2 = recs[1]["prompt"], recs[1]["gen"]
    t2 = build_match_table(p2, g2, 2, 5)
    check(sim_ngram(p2, g2, t2, 7) == (29, 0, 0), "fresh text: %r" % (sim_ngram(p2, g2, t2, 7),))
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
        cfg = make_cfg(argparse.Namespace(min_n=2, max_n=5, k="7,9", hybrid_t="2,3,4", dflash_tpr=DFLASH_TPR, window=0))
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
        sys.stdout.write(text)
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
            "hybrid_k": GATE_K, "dflash_tpr": args.dflash_tpr}


def main(argv=None):
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
    ap.add_argument("--selftest", action="store_true", help="run on a synthetic log and exit")
    ap.add_argument("--write-example", metavar="PATH", help="write the small synthetic example log and exit")
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
