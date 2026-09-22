"""tools/reference/make_tokens_json.py

Builds the shared **tokens file** that both halves of the KL comparison consume -- the reference
half (`full_logits_golden.py`) and the r4dx-engine half -- so that the two are guaranteed to be
scoring byte-identical token sequences.

Format (see "Shared file format" in tools/reference/README.md):

    {"tokenizer": "<hf path or name>",
     "segments": [{"name": "<str>", "token_ids": [int, ...]}, ...]}

Token ids come from the checkpoint's own HF `AutoTokenizer` with `add_special_tokens=False` on the
raw file text: **no chat template, no BOS, no EOS**. Each segment is evaluated independently from a
fresh context (position 0), so a segment is exactly one self-contained sequence.

Usage:

    <venv>\\Scripts\\python.exe tools\\reference\\make_tokens_json.py `
        --corpus-dir tools\\reference\\kl_corpus --max-tokens 1024 `
        --out tools\\reference\\kl_corpus\\tokens.json

No GPU, no model weights -- only the tokenizer files under `--model-dir` are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DEFAULT_MODEL_DIR  # noqa: E402


def token_ids_sha256(token_ids) -> str:
    """The `sha256_of_token_ids_json` sidecar field both halves must agree on: sha256 of the
    compact JSON array of the segment's token ids (`json.dumps(ids, separators=(",", ":"))`,
    UTF-8). Kept byte-identical to `full_logits_golden.token_ids_sha256`."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="checkpoint directory whose AutoTokenizer is used")
    ap.add_argument("--corpus-dir", type=Path, default=Path(__file__).parent / "kl_corpus",
                    help="directory of .txt files; each becomes one segment named after its stem")
    ap.add_argument("--file", action="append", default=None, metavar="NAME=PATH",
                    help="explicit segment (repeatable); overrides --corpus-dir when given")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="truncate every segment to exactly this many tokens (0 = no truncation)")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="fail if any segment is shorter than this (default: --max-tokens)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "kl_corpus" / "tokens.json")
    args = ap.parse_args()

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

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    min_tokens = args.max_tokens if args.min_tokens is None else args.min_tokens

    segments = []
    for name, path in sources:
        text = path.read_text(encoding="utf-8")
        ids = tok(text, add_special_tokens=False)["input_ids"]
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
        "tokenizer": str(args.model_dir),
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
