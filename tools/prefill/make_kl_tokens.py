"""tools/prefill/make_kl_tokens.py -- builds the long-prefix KL token file (the shared tokens.json
format tests/model/tool_teacher_forced_logprobs reads; see tools/reference/make_tokens_json.py).

One segment per (kind, length): exactly T = <length> tokens -- a (T - tail) token prefix and a
`tail`-token continuation (default 256). The engine is run with `--tail-rows <tail>`, so the prefix
goes through ONE chunked Prefill call and only the continuation's next-token distributions are
dumped (rows T-tail-1 .. T-2). Kinds:

  prose   repo docs prose from a seeded start; the continuation is the text that follows
  code    repo source files (seeded order) back to back; the continuation is the code that follows
  recall  a prose prefix, then a VERBATIM copy of 256 tokens that appeared at 10% depth of the
          prefix -- predicting it well needs attention to reach ~0.9*T tokens back, so it is the
          kind most sensitive to approximate (sparse / compressed) attention

Raw text, no chat template, no BOS (the checkpoint adds none), canonical tokenizer.json ids.

Usage:
  python tools\\prefill\\make_kl_tokens.py --lengths 8k,32k,64k,128k `
      --out <models root>\\r4dx\\prefill-m0\\kl\\tokens_long.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pf_common as C

KINDS = ["prose", "code", "recall"]


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids], separators=(",", ":")).encode()).hexdigest()


def prose_stream(tok: C.Tok, paras: list[str], seed: int, need: int) -> list[int]:
    start = random.Random(seed).randrange(len(paras))
    n_chars = 0
    k = 0
    while n_chars < need * 4 + 4000:  # ~3.2 characters per token in these docs
        n_chars += len(paras[(start + k) % len(paras)]) + 2
        k += 1
    ids = tok.encode("\n\n".join(paras[(start + j) % len(paras)] for j in range(k)))
    while len(ids) < need:
        k += 200
        ids = tok.encode("\n\n".join(paras[(start + j) % len(paras)] for j in range(k)))
    return ids[:need]


def code_stream(tok: C.Tok, files, seed: int, need: int) -> list[int]:
    order = list(files)
    random.Random(seed).shuffle(order)
    text = ""
    ids: list[int] = []
    for rel, body in order:
        text += f"===== FILE: {rel} =====\n{body.rstrip()}\n\n"
        if len(text) > need * 3:
            ids = tok.encode(text)
            if len(ids) >= need:
                return ids[:need]
    ids = tok.encode(text)
    if len(ids) < need:
        raise SystemExit(f"code corpus too small for {need} tokens")
    return ids[:need]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="8k,32k,64k,128k")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_OUT / "kl" / "tokens_long.json")
    args = ap.parse_args(argv)
    tok = C.Tok()
    paras = C.load_prose_paragraphs()
    files = C.load_code_files()
    segments = []
    for length in C.parse_lengths(args.lengths):
        T = C.LENGTHS[length]
        n_prefix = T - args.tail
        for kind in [k for k in args.kinds.split(",") if k]:
            seed = args.seed + 1000 * KINDS.index(kind) + T
            if kind == "prose":
                ids = prose_stream(tok, paras, seed, T)
            elif kind == "code":
                ids = code_stream(tok, files, seed, T)
            elif kind == "recall":
                pre = prose_stream(tok, paras, seed, n_prefix)
                a = int(0.1 * n_prefix)
                ids = pre + pre[a:a + args.tail]
            else:
                raise SystemExit(f"unknown kind {kind!r}")
            assert len(ids) == T
            name = f"{kind}_{length}"
            segments.append({"name": name, "token_ids": [int(i) for i in ids],
                             "prefix_tokens": n_prefix, "tail_rows": args.tail})
            print(f"[kl-tokens] {name:<14} T={T} prefix={n_prefix} tail={args.tail} sha={ids_sha(ids)[:16]} "
                  f"tail text={tok.decode(ids[n_prefix:n_prefix + 24])!r}", flush=True)
    doc = {"tokenizer": tok.t.describe(), "tokenizer_mode": "canonical",
           "tokenizer_provenance": tok.t.provenance(), "add_special_tokens": False,
           "chat_template": False, "tail_rows": args.tail, "seed": args.seed,
           "builder": "tools/prefill/make_kl_tokens.py", "segments": segments}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    print(f"[kl-tokens] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(segments)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
