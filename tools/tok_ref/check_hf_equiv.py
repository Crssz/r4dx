#!/usr/bin/env python
"""Audits that the two HF tokenizer front-ends for Gemma 4 agree, so the golden file (and r4dx's C++
tokenizer) is never checked against a front-end-specific quirk (docs/gemma4-plan.md 5.1 item 2):

  raw   = tokenizers.Tokenizer.from_file(tokenizer.json)      (the Rust backend, no overrides)
  auto  = transformers.AutoTokenizer.from_pretrained(dir)     (runtime GemmaTokenizer class)

Compared for every text (encode with special tokens parsed and not parsed, then decode with
skip_special_tokens True and False): token ids and decoded strings. Texts come from the KL corpus
(tools/reference/kl_corpus/*.txt), the repo docs, wikitext-2 if present, newline / space / tab runs
and a seeded fuzzer. CPU only.

    python tools\\tok_ref\\check_hf_equiv.py [--model-dir DIR] [--fuzz N] [--emit-jsonl PATH]

--emit-jsonl writes {"text", "ids"} lines (raw-tokenizer ids, special tokens not parsed) that
`bench_tokenizer --check PATH` replays through the C++ tokenizer, giving a large-corpus agreement
check of r4dx against HF beyond the committed golden file.

Exit status 1 if any text differs.
"""
import argparse
import glob
import json
import os
import random
import sys

MODELS_ROOT = os.environ.get("R4DX_MODELS_ROOT", r"E:\models")
DEFAULT_DIR = os.environ.get("R4DX_GEMMA_TOKENIZER_DIR",
                             os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated-tok"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIKITEXT = os.path.join(MODELS_ROOT, "wikitext-2-raw", "wiki.test.raw")


def corpus_texts(max_wiki_bytes):
    texts = []
    for p in sorted(glob.glob(os.path.join(REPO, "tools", "reference", "kl_corpus", "*.txt"))):
        texts.append((os.path.basename(p), open(p, encoding="utf-8", errors="replace").read()))
    for p in sorted(glob.glob(os.path.join(REPO, "docs", "*.md"))):
        texts.append((os.path.relpath(p, REPO), open(p, encoding="utf-8", errors="replace").read()))
    if os.path.exists(WIKITEXT):
        raw = open(WIKITEXT, encoding="utf-8", errors="replace").read()[:max_wiki_bytes]
        # paragraph-sized chunks plus the whole thing as one (very long, multi-line) document
        texts.append(("wikitext_all", raw))
        for i in range(0, len(raw), 4000):
            texts.append((f"wikitext_{i}", raw[i:i + 4000]))
    return texts


def fuzz_texts(n, seed):
    rng = random.Random(seed)
    alphabet = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + [" "] * 14 + ["\n"] * 6 +
                ["\t", "\r", ".", ",", "!", "?", "'", '"', "-", "_", "/", "(", ")", "{", "}", "<", ">", "|"] +
                list("\u00e9\u00fc\u00f1\u4f60\u597d\u0e01\u0e34\u0e49\u0e32\U0001F600\u2581\u2018\u2019\ue000\U0001FAE0"))
    out = []
    for i in range(n):
        out.append((f"fuzz_{i}", "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 400)))))
    for k in list(range(1, 400)):
        out.append((f"newlines_{k}", "\n" * k))
        out.append((f"nl_ab_{k}", "a" + "\n" * k + "b"))
    for k in range(1, 130):
        out.append((f"spaces_{k}", " " * k))
        out.append((f"tabs_{k}", "\t" * k))
    for k in (1, 2, 30, 31, 32, 33, 62, 63, 64, 93, 94, 95, 200):
        out.append((f"nl_sp_{k}", ("\n" + " ") * k))
        out.append((f"sp_nl_{k}", (" " + "\n") * k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--fuzz", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--wiki-bytes", type=int, default=400000)
    ap.add_argument("--emit-jsonl")
    args = ap.parse_args()

    from tokenizers import Tokenizer as RawTokenizer
    from transformers import AutoTokenizer

    raw = RawTokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(args.model_dir)

    texts = corpus_texts(args.wiki_bytes) + fuzz_texts(args.fuzz, args.seed)
    bad = 0
    n_bytes = 0
    emit = open(args.emit_jsonl, "w", encoding="utf-8") if args.emit_jsonl else None
    for name, text in texts:
        n_bytes += len(text.encode("utf-8"))
        for parse_special in (False, True):
            raw.encode_special_tokens = not parse_special
            r = raw.encode(text, add_special_tokens=False).ids
            a = auto.encode(text, add_special_tokens=False, split_special_tokens=not parse_special)
            if r != a:
                bad += 1
                first = next((i for i in range(min(len(r), len(a))) if r[i] != a[i]), min(len(r), len(a)))
                print(f"ENCODE MISMATCH {name} parse_special={parse_special}: raw {len(r)} ids, auto {len(a)} ids, "
                      f"first diff at {first}: raw {r[first:first + 5]} auto {a[first:first + 5]}")
                continue
            for skip in (True, False):
                dr = raw.decode(r, skip_special_tokens=skip)
                da = auto.decode(a, skip_special_tokens=skip)
                if dr != da:
                    bad += 1
                    print(f"DECODE MISMATCH {name} parse_special={parse_special} skip={skip}")
            if emit is not None and not parse_special:
                emit.write(json.dumps({"name": name, "text": text, "ids": r}, ensure_ascii=False) + "\n")
    if emit is not None:
        emit.close()
    print(f"{len(texts)} texts, {n_bytes / 1e6:.2f} MB: {bad} mismatch(es) between raw tokenizers and AutoTokenizer")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
