#!/usr/bin/env python
"""Fuzzes r4dx's Inkling-2 tokenizer (o200k splitter + BPE + ignore_merges) against HF `tokenizers`.

Writes a JSONL corpus of random texts with the HF ids, then runs the C++ checker `tokenizer_fuzz_inkling2.exe` over it.
CPU only. Run with the reference venv, -I:

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\tok_ref\\fuzz_inkling2.py [--n 100000] [--seed 1]
        [--exe build\\win-hip\\tests\\tokenizer\\tokenizer_fuzz_inkling2.exe] [--corpus <path>]

Text sources: structured alphabets (ASCII upper/lower/digits/punctuation/apostrophes/slashes, every whitespace code
point, CR/LF, combining marks, Lt/Lm/Lo/Nl/No/Han/Thai/Hindi/emoji/ZWJ pools), contraction fragments, whole-range random
code points (surrogates excluded; unassigned ones kept), and special-token strings with parse_special both ways.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import unicodedata

MODEL_DIR = os.environ.get("R4DX_INKLING2_TOKENIZER_DIR", r"E:\models\OpenUncensored\Inkling-2-Mini-Uncensored")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_pools():
    pools = {
        "lower": list("abcdefghijklmnopqrstuvwxyz"), "upper": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "digit": list("0123456789"), "punct": list("!\"#$%&()*+,-.:;<=>?@[\\]^_`{|}~"), "apos": ["'", "'", "'"],
        "slash": ["/", "/"], "ws": [" ", " ", " ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\u00a0", "\u2003", "\u2028",
                                      "\u2029", "\u0085", "\u3000", "\u1680", "\u202f", "\u200a"],
        "marks": ["\u0301", "\u0300", "\u0308", "\u0e31", "\u0e48", "\u093f", "\u094d", "\u064e", "\u20dd", "\u0903"],
        "titlecase": ["\u01c5", "\u01c8", "\u01cb", "\u01f2", "\u1f88"], "modifier": ["\u02b0", "\u02bb", "\u02c6", "\u3005", "\u30fc"],
        "other_letters": ["\u05d0", "\u0627", "\u0e01", "\u0915", "\u3042", "\u30ab", "\uac00", "\u4e2d", "\u6587", "\u00aa", "\u00ba", "\u0294"],
        "uni_lower": list("\u00e9\u00fc\u00f1\u00df\u03b1\u03b2\u0430\u0431\u0101\u1e9d"),
        "uni_upper": list("\u00c9\u00dc\u00d1\u1e9e\u0391\u0392\u0410\u0411\u0100\u0130"),
        "numbers_other": ["\u00b2", "\u00bd", "\u2160", "\u2460", "\u0661", "\u0e51", "\uff11", "\u096a"],
        "emoji": ["\U0001F600", "\U0001F468", "\u200d", "\U0001F469", "\ufe0f", "\U0001F1F9", "\U0001F44B", "\U0001F3FD", "\u2764", "\u2705"],
        "han": list("\u4f60\u597d\u4e16\u754c\u6211\u7231") + ["\U00020bb7", "\u3400"],
        "symbols_other": ["\u2014", "\u2019", "\u201c", "\u20ac", "\u00a9", "\u2192", "\u00b7", "\u2022", "\u3001", "\u3002", "\uff0c"],
        "long_s": ["\u017f"],
    }
    return pools


def random_codepoint(rng):
    while True:
        cp = rng.choice([rng.randrange(0x20, 0x3000), rng.randrange(0x3000, 0x20000), rng.randrange(0x1F000, 0x1FB00),
                         rng.randrange(0xA0, 0x800), rng.randrange(0x20000, 0x30000)])
        if 0xD800 <= cp <= 0xDFFF:
            continue
        return chr(cp)


FRAGMENTS = ["'s", "'S", "'t", "'T", "'re", "'RE", "'ve", "'VE", "'m", "'M", "'ll", "'LL", "'d", "'D", "'rE", "'Ll", "'\u017f",
             "n't", "N'T", "it's", "IT'S", "I'm", "x'", "'", "''", "'q", "/", "//", "!/", "!\n", ".\r\n", " \n", "  \n\n"]
SPECIALS = ["<|message_user|>", "<|message_model|>", "<|end_message|>", "<|content_text|>", "<|content_model_end_sampling|>",
            "<|unused_200007|>", "<|endoftext|>", "<|content_invoke_tool_json|>"]


def gen_text(rng, pools):
    names = list(pools)
    kind = rng.random()
    if kind < 0.55:
        # few alphabets mixed, so runs of one class get long enough to exercise the regex backtracking
        chosen = rng.sample(names, rng.randint(1, 5))
        out = []
        for _ in range(rng.randint(1, 60)):
            p = pools[rng.choice(chosen)]
            out.append(rng.choice(p) * (rng.choice([1, 1, 1, 2, 3, 4]) if rng.random() < 0.3 else 1))
        return "".join(out)
    if kind < 0.75:
        out = []
        for _ in range(rng.randint(1, 30)):
            r = rng.random()
            if r < 0.3:
                out.append(rng.choice(FRAGMENTS))
            elif r < 0.6:
                out.append("".join(rng.choice(pools[rng.choice(["lower", "upper", "digit", "uni_lower", "uni_upper", "marks"])])
                                   for _ in range(rng.randint(1, 8))))
            else:
                out.append(rng.choice(pools[rng.choice(["ws", "punct", "apos", "slash", "digit"])]))
        return "".join(out)
    if kind < 0.9:
        return "".join(random_codepoint(rng) for _ in range(rng.randint(1, 30)))
    # words and sentences
    words = ["Hello", "world", "CamelCase", "HTTPServer", "don't", "I'M", "x", "42", "1234567", "caf\u00e9", "\u4f60\u597d", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
             "snake_case", "A", "ALLCAPS", "mIxEd", "\u00c9COLE"]
    out = []
    for _ in range(rng.randint(1, 25)):
        out.append(rng.choice(words))
        out.append(rng.choice([" ", " ", " ", "  ", "\n", "\r\n", ", ", ". ", "\t", "", "/", "-", "'"]))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "win-hip", "tests", "tokenizer", "tokenizer_fuzz_inkling2.exe"))
    ap.add_argument("--corpus", default=os.path.join(os.environ.get("TEMP", "."), "inkling2_fuzz_corpus.jsonl"))
    args = ap.parse_args()

    from tokenizers import Tokenizer as RawTokenizer
    from transformers import AutoTokenizer
    raw = RawTokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(MODEL_DIR)
    rng = random.Random(args.seed)
    pools = build_pools()
    n_auto_checked = 0
    with open(args.corpus, "w", encoding="utf-8") as f:
        for i in range(args.n):
            text = gen_text(rng, pools)
            parse_special = False
            if rng.random() < 0.05:
                parts = [text, rng.choice(SPECIALS), gen_text(rng, pools)]
                rng.shuffle(parts)
                text = "".join(parts)
                parse_special = rng.random() < 0.5
            raw.encode_special_tokens = not parse_special
            ids = raw.encode(text, add_special_tokens=False).ids
            if i % 50 == 0:  # AutoTokenizer agrees with the raw backend (checked on a sample)
                assert ids == auto.encode(text, add_special_tokens=False, split_special_tokens=not parse_special)
                n_auto_checked += 1
            f.write(json.dumps({"text": text, "parse_special": parse_special, "ids": ids}, ensure_ascii=False) + "\n")
    print(f"corpus: {args.n} cases at {args.corpus} (AutoTokenizer cross-checked on {n_auto_checked}), unicode {unicodedata.unidata_version}")
    r = subprocess.run([args.exe, args.corpus, MODEL_DIR])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
