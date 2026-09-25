"""tools/reference/make_tokens_json.py

Builds the shared **tokens file** that both halves of the KL comparison consume -- the reference
half (`full_logits_golden.py`) and the r4dx-engine half -- so that the two are guaranteed to be
scoring byte-identical token sequences.

Format (see "Shared file format" in tools/reference/README.md):

    {"tokenizer": "<description>", "tokenizer_mode": "canonical" | "hf-auto",
     "tokenizer_provenance": {...common.RefTokenizer.provenance()...},
     "add_special_tokens": false, "chat_template": false, "max_tokens": N,
     "segments": [{"name": "<str>", "token_ids": [int, ...]}, ...]}

Token ids are the raw file text encoded by `common.RefTokenizer` in `--tokenizer`'s mode:
**no chat template, no BOS, no EOS**. Special-token literals written in the text would be
recognized as their ids (r4dx's chat path, parse_special=true; r4dx-server's raw /v1/completions
would spell them out) -- the KL corpus has none. Each segment is evaluated independently from a
fresh context (position 0), so a segment is exactly one self-contained sequence.

`--tokenizer` (docs/quant2.md 3.4):

- `canonical` (the default): the checkpoint's tokenizer.json, as r4dx tokenizes the NFC text of
  every corpus here (a file that is not NFC is refused). A new tokens file should be this.
- `hf-auto`: transformers 5.17's AutoTokenizer, which splits Thai combining marks off their
  consonants (refused under a transformers that does not). `kl_corpus/tokens.json` (every KL
  number before 2026-09-26) was made this way and records no mode; `--tokenizer hf-auto`
  regenerates its token ids exactly.

Both commands below reproduce their file's token ids and every field it has; the output adds
`tokenizer_mode` and `tokenizer_provenance`, so it is not byte-identical to a file made before
those fields existed (tokens.json, tokens_thai_canon.json). KL pairing goes by the token ids'
sha256, and the r4dx half ignores unknown keys.

The default never silently changes an existing file's tokenization: when `--out` already exists
and its mode (`common.tokens_file_tokenizer_mode`: the recorded field, else the r4dx-cli dump
shape, the exact canonical description, or a known legacy file's ids -- tokens.json is hf-auto --
else unknown) differs from this run's, the run is refused unless `--force` is given. So
re-running the command that made tokens.json without `--tokenizer` stops and says to pass
`--tokenizer hf-auto` (reproduce it) or `--force` (re-tokenize it canonically).

Usage:

    <venv>\\Scripts\\python.exe tools\\reference\\make_tokens_json.py --tokenizer hf-auto `
        --corpus-dir tools\\reference\\kl_corpus --max-tokens 1024 `
        --out tools\\reference\\kl_corpus\\tokens.json
    <venv>\\Scripts\\python.exe tools\\reference\\make_tokens_json.py --tokenizer canonical `
        --file thai_prose_canon=tools\\reference\\kl_corpus\\thai_prose.txt --max-tokens 1024 `
        --out tools\\reference\\kl_corpus\\tokens_thai_canon.json

No GPU, no model weights -- only the tokenizer files under `--model-dir` are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_TOKENIZER_MODE,
    TOKENIZER_HELP,
    TOKENIZER_MODES,
    load_ref_tokenizer,
    refuse_tokenizer_mode_change,
    tokens_file_tokenizer_mode,
)


def token_ids_sha256(token_ids) -> str:
    """The `sha256_of_token_ids_json` sidecar field both halves must agree on: sha256 of the
    compact JSON array of the segment's token ids (`json.dumps(ids, separators=(",", ":"))`,
    UTF-8). Kept byte-identical to `full_logits_golden.token_ids_sha256`."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="checkpoint directory whose tokenizer is used")
    ap.add_argument("--corpus-dir", type=Path, default=Path(__file__).parent / "kl_corpus",
                    help="directory of .txt files; each becomes one segment named after its stem")
    ap.add_argument("--file", action="append", default=None, metavar="NAME=PATH",
                    help="explicit segment (repeatable); overrides --corpus-dir when given")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="truncate every segment to exactly this many tokens (0 = no truncation)")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="fail if any segment is shorter than this (default: --max-tokens)")
    ap.add_argument("--tokenizer", choices=TOKENIZER_MODES, default=DEFAULT_TOKENIZER_MODE,
                    help=TOKENIZER_HELP + " Recorded as the file's tokenizer_mode. "
                         "kl_corpus/tokens.json is hf-auto.")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing --out whose tokenizer mode differs from this run's "
                         "or cannot be determined")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "kl_corpus" / "tokens.json")
    args = ap.parse_args(argv)

    if args.file:
        sources = []
        for spec in args.file:
            if "=" not in spec:
                raise SystemExit(f"--file expects NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            sources.append((name, Path(path)))
    else:
        sources = sorted((p.stem, p) for p in args.corpus_dir.glob("*.txt"))
    if not sources:
        raise SystemExit(f"no .txt files found in {args.corpus_dir}")

    if args.out.exists() and not args.force:
        with open(args.out, "r", encoding="utf-8") as f:
            old_mode = tokens_file_tokenizer_mode(json.load(f))
        refuse_tokenizer_mode_change("make_tokens", str(args.out), [old_mode], args.tokenizer, False)

    tok = load_ref_tokenizer(args.model_dir, args.tokenizer)
    min_tokens = args.max_tokens if args.min_tokens is None else args.min_tokens
    print(f"[make_tokens] tokenizer: {tok.mode} ({tok.describe()})")

    segments = []
    for name, path in sources:
        text = path.read_text(encoding="utf-8")
        ids = tok.encode(text)
        full = len(ids)
        if full < min_tokens:
            raise SystemExit(f"segment {name!r} ({path}) tokenizes to {full} tokens, "
                             f"fewer than the required minimum {min_tokens}")
        if args.max_tokens:
            ids = ids[: args.max_tokens]
        segments.append({"name": name, "token_ids": [int(i) for i in ids]})
        print(f"[make_tokens] {name:<16} {path.name:<20} {full:>6} tokens -> {len(ids)} "
              f"(sha256 {token_ids_sha256(ids)[:16]}...)")

    doc = {
        "tokenizer": tok.describe(),
        "tokenizer_mode": tok.mode,
        # The versions and tokenizer.json sha256 behind the mode (hf-auto's ids depend on the
        # installed transformers; RefTokenizer refuses one that does not split).
        "tokenizer_provenance": tok.provenance(),
        "add_special_tokens": False,
        "chat_template": False,
        "max_tokens": args.max_tokens,
        "segments": segments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"[make_tokens] wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB, "
          f"{len(segments)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
