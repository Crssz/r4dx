#!/usr/bin/env python
"""Tokenizer microbench, HF side (docs/gemma4-plan.md M1-11). CPU only.

Builds a benchmark corpus of real text (wikitext-2 chunks, repo docs, source files, the KL corpus's
Thai prose), writes it as JSONL {"name", "text", "ids"} (ids = raw `tokenizers` ids, special tokens
not parsed) for the C++ side (`bench_tokenizer --corpus`), and times the HF Rust backend on it:

  encode         tokenizers.Tokenizer.encode(text), one document at a time, single thread
  encode_batch   tokenizers.Tokenizer.encode_batch (HF's multi-threaded path; informational)
  encode_auto    transformers AutoTokenizer.encode (what a Python server would call; informational)
  decode         tokenizers.Tokenizer.decode(ids, skip_special_tokens=True), whole document
  stream_decode  tokenizers.decoders.DecodeStream.step(), one id at a time (if the installed
                 tokenizers has it) -- the closest HF analogue of r4dx's StreamDecoder::push

    python tools\\tok_ref\\bench_hf.py [--model-dir DIR] [--corpus-out PATH] [--repeats N]

MB/s counts UTF-8 bytes of the corpus text; Mtok/s counts ids. Best of --repeats passes.
"""
import argparse
import glob
import json
import os
import sys
import time

MODELS_ROOT = os.environ.get("R4DX_MODELS_ROOT", r"E:\models")
DEFAULT_DIR = os.environ.get("R4DX_GEMMA_TOKENIZER_DIR",
                             os.path.join(MODELS_ROOT, "Huihui-gemma-4-12B-it-abliterated-tok"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIKITEXT = os.path.join(MODELS_ROOT, "wikitext-2-raw", "wiki.test.raw")


def build_corpus(wiki_bytes):
    docs = []
    for p in sorted(glob.glob(os.path.join(REPO, "tools", "reference", "kl_corpus", "*.txt"))):
        docs.append((os.path.basename(p), open(p, encoding="utf-8", errors="replace").read()))
    for p in sorted(glob.glob(os.path.join(REPO, "docs", "*.md"))):
        docs.append((os.path.relpath(p, REPO), open(p, encoding="utf-8", errors="replace").read()))
    for pat in ("src/**/*.cpp", "src/**/*.h", "tools/**/*.py"):
        for p in sorted(glob.glob(os.path.join(REPO, pat), recursive=True))[:60]:
            if "vendor" in p or os.path.getsize(p) > 200000:
                continue
            docs.append((os.path.relpath(p, REPO), open(p, encoding="utf-8", errors="replace").read()))
    if os.path.exists(WIKITEXT):
        wiki = open(WIKITEXT, encoding="utf-8", errors="replace").read()[:wiki_bytes]
        for i in range(0, len(wiki), 4000):
            docs.append((f"wikitext_{i}", wiki[i:i + 4000]))
        docs.append(("wikitext_long", wiki[:200000]))
    return [(n, t) for n, t in docs if t]


def best_of(fn, repeats):
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None or dt < best else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--corpus-out", default=os.path.join(MODELS_ROOT, "r4dx", "tok_bench", "corpus_gemma.jsonl"))
    ap.add_argument("--wiki-bytes", type=int, default=1500000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    import tokenizers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    import transformers

    raw = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(args.model_dir)

    # encode_special_tokens=True means special-token surface text is NOT recognized (ordinary text),
    # i.e. r4dx's Tokenizer::encode(text, parse_special=false); the corpus (repo docs) contains such text.
    raw.encode_special_tokens = True
    docs = build_corpus(args.wiki_bytes)
    texts = [t for _, t in docs]
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    ids = [raw.encode(t, add_special_tokens=False).ids for t in texts]
    ntok = sum(len(i) for i in ids)
    os.makedirs(os.path.dirname(args.corpus_out), exist_ok=True)
    with open(args.corpus_out, "w", encoding="utf-8") as f:
        for (name, text), i in zip(docs, ids):
            f.write(json.dumps({"name": name, "text": text, "ids": i}, ensure_ascii=False) + "\n")
    print(f"corpus: {len(docs)} docs, {nbytes / 1e6:.3f} MB utf-8, {ntok} tokens -> {args.corpus_out}")
    print(f"tokenizers {tokenizers.__version__}, transformers {transformers.__version__}")

    def report(label, dt, with_tokens=True):
        mb = nbytes / 1e6 / dt
        tok = ntok / 1e6 / dt
        print(f"{label:14s} {mb:8.2f} MB/s   {tok:7.3f} Mtok/s   ({dt * 1000:.1f} ms per pass)")
        return {"label": label, "MB_per_s": mb, "Mtok_per_s": tok, "ms": dt * 1000}

    results = []
    results.append(report("encode", best_of(lambda: [raw.encode(t, add_special_tokens=False) for t in texts], args.repeats)))
    results.append(report("encode_batch", best_of(lambda: raw.encode_batch(texts, add_special_tokens=False), args.repeats)))
    results.append(report("encode_auto", best_of(lambda: [auto.encode(t, add_special_tokens=False, split_special_tokens=True) for t in texts],
                                                  max(1, args.repeats // 2))))
    results.append(report("decode", best_of(lambda: [raw.decode(i, skip_special_tokens=True) for i in ids], args.repeats)))

    try:
        from tokenizers.decoders import DecodeStream

        def stream_all():
            for doc_ids in ids:
                st = DecodeStream(skip_special_tokens=True)
                for i in doc_ids:
                    st.step(raw, i)
        results.append(report("stream_decode", best_of(stream_all, max(1, args.repeats // 2))))
    except Exception as e:  # noqa: BLE001
        print(f"stream_decode: not available in this tokenizers build ({e})")

    print("JSON " + json.dumps({"tokenizers": tokenizers.__version__, "bytes": nbytes, "tokens": ntok,
                                 "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
