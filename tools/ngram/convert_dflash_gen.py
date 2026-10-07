#!/usr/bin/env python
"""Convert tool_dflash_traindata generations (gen_p*.jsonl) to the request-log token format sim_ngram.py reads.

PROXY use only: the generations are the production target's (Huihui trellis) own greedy / sampled streams from
the production speculative DFlash2 k=7 loop, but the prompts are the dflash-ft training/eval mix (synthetic
agent trajectories, local code, prose), not the user's real oh-my-pi traffic.

Input record (tests/model/tool_dflash_traindata.cpp, mode gen/both):
  id, source, [kind | shape | lang | tag], think, temperature, top_p, seed, max_new, hit_eos,
  ids (prompt token ids), gen (generated ids, the first one comes out of the prefill; the EOS that ended the
  sequence is appended), dflash_rounds / dflash_drafted / dflash_accepted (counters),
  drafts (only with --log-drafts 1 and temperature 0: per round the k drafted token ids).

Output line (what r4dx-server --request-log-tokens writes, only the fields the replay reads, prompt_shared 0):
  request_id, http_status 200, error_status null, finish_reason, temperature, prompt_tokens, completion_tokens,
  speculative "dflash", draft_k, prompt_shared 0, prompt_ids, generated_ids,
  round_drafted + round_accepted   when `drafts` is there: per round the drafted chain is matched against the
                                   generation (greedy: the verifier accepts exactly the longest equal prefix);
  dflash_round_count               otherwise (sampled, or greedy without drafts): the counter, so the
                                   simulator prices DFlash alone at that many rounds and uses the request's
                                   mean tok/round for the hybrid's DFlash rounds (sim_ngram.py "round count"),
  plus source / group / kind / think / max_new (ignored by the simulator, kept for slicing).

A generation that ran into max_new ends in a round the loop cut: the logged counter counts the whole round's
accepted tokens while the kept generation holds only part of them, so a replay of `drafts` against `gen`
undercounts that last round. The last round's acceptance is then taken from the counter (only the last
round: the others reproduce the counters exactly, which the converter checks and counts).

  python convert_dflash_gen.py --out all.jsonl --split-dir by_group  gen_p0.jsonl [gen_p1.jsonl ...]
  python convert_dflash_gen.py --selftest
"""
import argparse
import json
import os
import sys
from collections import Counter

# Coarse groups for the proxy report. agent = prompts shaped like a coding-agent turn (system prompt, tool
# list, tool output in the context); code = code-writing prompts over the local repo / libraries; prose =
# plain text and chat. Everything else falls to its own source name.
GROUPS = {
    "heldout-agent": "agent", "heldout-agentnew": "agent", "agentsyn": "agent", "toolsyn": "agent",
    "repo": "code", "codelib": "code",
    "prose": "prose", "corpus": "prose", "kl": "prose", "heldout-corpus": "prose", "heldout-mtp": "prose",
}


def reconstruct_rounds(gen, drafts, counted_accepted):
    """Per-round accepted counts from the drafted chains. Round r starts at generated index s (1 + the tokens
    the earlier rounds emitted: gen[0] came out of the prefill); the chain's token j is accepted while it
    equals gen[s + j]. Returns (accepted per round, status) with status "exact" (sum == the counter),
    "last-fixed" (the cut last round taken from the counter) or "mismatch" (accepted is then None)."""
    acc = []
    s = 1
    for d in drafts:
        a = 0
        while a < len(d) and s + a < len(gen) and d[a] == gen[s + a]:
            a += 1
        acc.append(a)
        s += a + 1
    if counted_accepted is None or sum(acc) == counted_accepted:
        return acc, "exact"
    if acc:
        fixed = counted_accepted - sum(acc[:-1])
        if acc[-1] < fixed <= len(drafts[-1]) and s >= len(gen):  # the generation ended inside the last round
            return acc[:-1] + [fixed], "last-fixed"
    return None, "mismatch"


def convert_record(rec, stats):
    gen = rec["gen"]
    out = {
        "request_id": rec["id"],
        "endpoint": "chat/completions", "stream": True, "http_status": 200, "error_status": None,
        "finish_reason": "stop" if rec.get("hit_eos") else "length", "cancelled": False,
        "temperature": rec.get("temperature", 0.0),
        "prompt_tokens": len(rec["ids"]), "completion_tokens": len(gen),
        "speculative": "dflash", "draft_k": 7,
        "top_p": rec.get("top_p", 1.0), "top_k": 0, "min_p": 0.0, "seed": rec.get("seed"),
        "prompt_shared": 0, "prompt_ids": rec["ids"], "generated_ids": gen,
        "source": rec.get("source"), "group": GROUPS.get(rec.get("source"), rec.get("source")),
        "kind": rec.get("kind") or rec.get("shape") or rec.get("lang"),
        "think": rec.get("think"), "max_new": rec.get("max_new"),
    }
    drafts = rec.get("drafts")
    if drafts:
        acc, status = reconstruct_rounds(gen, drafts, rec.get("dflash_accepted"))
        stats["rounds_" + status] += 1
        if acc is not None:
            out["round_drafted"] = [len(d) for d in drafts]
            out["round_accepted"] = acc
    else:
        stats["no_drafts"] += 1
    if rec.get("dflash_rounds"):
        out["dflash_round_count"] = int(rec["dflash_rounds"])
    if "round_accepted" in out:
        stats["with_round_accepted"] += 1
    elif "dflash_round_count" in out:
        stats["with_round_count"] += 1
    else:
        stats["no_dflash_data"] += 1
    return out


def convert(paths, stats):
    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                if "gen" not in rec or "ids" not in rec:
                    stats["skipped_no_gen"] += 1
                    continue
                if len(rec["gen"]) < 2:
                    stats["skipped_short"] += 1
                    continue
                yield convert_record(rec, stats)


def write_lines(path, recs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def selftest():
    fails = []

    def check(c, m):
        if not c:
            fails.append(m)
    gen = [9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    # round 1 (s=1): chain 1,2,3,0 -> 3 accepted (emits 4); round 2 (s=5): chain 5,6,7,8 -> 4 accepted, bonus 10 (emits 5, s=10)
    # round 3 (s=10): chain 11,12,13,14 -> gen ends after 13: 3 matched, the counter says 4 (cut by max_new)
    drafts = [[1, 2, 3, 0], [5, 6, 7, 8], [11, 12, 13, 14]]
    acc, st = reconstruct_rounds(gen, drafts, 3 + 4 + 4)
    check(acc == [3, 4, 3 + 1] and st == "last-fixed", "cut last round: %r %s" % (acc, st))
    acc, st = reconstruct_rounds(gen, drafts, 3 + 4 + 3)
    check(acc == [3, 4, 3] and st == "exact", "exact: %r %s" % (acc, st))
    acc, st = reconstruct_rounds(gen, drafts, 3 + 4 + 9)
    check(acc is None and st == "mismatch", "mismatch: %r %s" % (acc, st))
    acc, st = reconstruct_rounds(gen, drafts, 99)
    check(st == "mismatch", "mismatch 2")
    stats = Counter()
    rec = {"id": "agentsyn-git-1", "source": "agentsyn", "kind": "git", "think": True, "temperature": 0.0, "max_new": 13,
           "hit_eos": False, "ids": [5, 6, 7], "gen": gen, "dflash_rounds": 3, "dflash_accepted": 11, "drafts": drafts}
    o = convert_record(rec, stats)
    check(o["round_accepted"] == [3, 4, 4] and o["prompt_shared"] == 0 and o["generated_ids"] == gen
          and o["group"] == "agent" and o["dflash_round_count"] == 3 and o["finish_reason"] == "length", "convert greedy: %r" % o)
    rec2 = dict(rec, temperature=1.0, id="prose-1", source="prose")
    del rec2["drafts"]
    o = convert_record(rec2, stats)
    check("round_accepted" not in o and o["dflash_round_count"] == 3 and o["group"] == "prose", "convert sampled: %r" % o)
    # the converted lines feed the simulator
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import sim_ngram
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        write_lines(p, [convert_record(rec, Counter()), o])
        got = list(sim_ngram.read_requests([p], Counter()))
        check(len(got) == 2 and got[0]["_gen"] == gen and got[0]["_prompt"] == [5, 6, 7], "sim reads the output")
        cfg = sim_ngram.make_cfg(sim_ngram.build_parser().parse_args([]))
        r0, r1 = (sim_ngram.analyse(g, cfg) for g in got)
        check(r0["dflash_src"] == "log" and r0["dflash_rounds"] == 3 and r1["dflash_src"] == "count", "sim sources %r %r" % (r0["dflash_src"], r1["dflash_src"]))
    if fails:
        sys.stderr.write("SELFTEST FAILED:\n  " + "\n  ".join(fails) + "\n")
        return 1
    print("selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="*", help="gen_p*.jsonl written by tool_dflash_traindata (mode gen / both)")
    ap.add_argument("--out", metavar="PATH", help="one file with every converted request")
    ap.add_argument("--split-dir", metavar="DIR", help="also write group_<group>.jsonl and source_<source>.jsonl here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.inputs or not (args.out or args.split_dir):
        ap.error("give input files and --out and/or --split-dir")
    stats = Counter()
    recs = list(convert(args.inputs, stats))
    if args.out:
        write_lines(args.out, recs)
    if args.split_dir:
        by = {}
        for r in recs:
            by.setdefault("group_%s" % r["group"], []).append(r)
            by.setdefault("source_%s" % r["source"], []).append(r)
        for name, rs in sorted(by.items()):
            write_lines(os.path.join(args.split_dir, name + ".jsonl"), rs)
    sys.stderr.write("converted %d requests: %s\n" % (len(recs), ", ".join("%s=%d" % kv for kv in sorted(stats.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
