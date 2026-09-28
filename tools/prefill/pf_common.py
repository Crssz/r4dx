"""tools/prefill/pf_common.py -- shared helpers for the long-context (prefill) evaluation kit.

Everything here is local and deterministic: the checkpoint's own tokenizer.json and
chat_template.jinja (through tools/reference/common.py's RefTokenizer, canonical mode -- the ids
r4dx itself produces), and text from this repository's own docs and sources. Nothing is downloaded.

Text sources (paths are relative to the repo root; files are read sorted, CRLF -> LF, NFC):
  prose: docs/**/*.md, README.md, third_party/libr4d/**/*.md
  code:  tools/**/*.py, src/**/*.{h,cpp}, tests/**/*.{h,cpp}, third_party/libr4d/**/*.{hip,h,hpp,cpp,py}

The prompts built from them are frozen to disk (build_tasks.py / make_kl_tokens.py write them
under the output directory with sha256s), so later edits to the repo never change a measurement:
every run reads the frozen files, not the live sources.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "reference"))
from common import DEFAULT_MODEL_DIR, load_ref_tokenizer  # noqa: E402

DEFAULT_OUT = Path(r"D:\models\r4dx\prefill-m0")

#: Length tags -> target prompt length in tokens (the whole chat-templated prompt).
LENGTHS = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}

PROSE_GLOBS = [("docs", "**/*.md"), (".", "README.md"), ("third_party/libr4d", "**/*.md")]
CODE_GLOBS = [
    ("tools", "**/*.py"),
    ("src", "**/*.h"), ("src", "**/*.cpp"),
    ("tests", "**/*.h"), ("tests", "**/*.cpp"),
    ("third_party/libr4d", "**/*.hip"), ("third_party/libr4d", "**/*.h"),
    ("third_party/libr4d", "**/*.hpp"), ("third_party/libr4d", "**/*.cpp"),
    ("third_party/libr4d", "**/*.py"),
]
_SKIP_PARTS = {"build", "build-win", ".git", "__pycache__", "golden_out", "golden"}


def parse_lengths(spec: str) -> list[str]:
    tags = [t.strip() for t in spec.split(",") if t.strip()]
    for t in tags:
        if t not in LENGTHS:
            raise SystemExit(f"unknown length tag {t!r}; known: {', '.join(LENGTHS)}")
    return tags


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return unicodedata.normalize("NFC", s)


def _collect(globs) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for root, pattern in globs:
        base = REPO / root
        if not base.exists():
            continue
        for p in sorted(base.glob(pattern)):
            rel = p.relative_to(REPO).as_posix()
            if rel in seen or not p.is_file():
                continue
            if any(part in _SKIP_PARTS for part in p.relative_to(REPO).parts):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            seen.add(rel)
            out.append((rel, clean_text(text)))
    out.sort()
    return out


def load_code_files() -> list[tuple[str, str]]:
    """(relpath, text) for every source file, sorted by path."""
    return [(r, t) for r, t in _collect(CODE_GLOBS) if t.strip()]


def load_prose_paragraphs() -> list[str]:
    """Paragraphs (blank-line separated) of the prose sources, in file order. Fenced code blocks
    and table rows are dropped so the haystack reads as prose; very short paragraphs are dropped."""
    paras: list[str] = []
    for _, text in _collect(PROSE_GLOBS):
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for block in re.split(r"\n\s*\n", text):
            lines = [ln for ln in block.split("\n") if not ln.lstrip().startswith("|")]
            b = "\n".join(lines).strip()
            if len(b) < 80:
                continue
            # Long paragraphs are cut at sentence ends into <= ~600-character pieces, so the
            # length fit (whole pieces) lands within ~150 tokens of its target.
            piece = ""
            for sent in re.split(r"(?<=[.!?])\s+", b):
                if piece and len(piece) + len(sent) > 600:
                    paras.append(piece)
                    piece = sent
                else:
                    piece = f"{piece} {sent}" if piece else sent
            if piece:
                paras.append(piece)
    return paras


def prose_vocabulary(min_len: int = 4, max_len: int = 10) -> list[str]:
    """Distinct lowercase alphabetic words of the prose sources, sorted."""
    words = set()
    for p in load_prose_paragraphs():
        for w in re.findall(r"\b[a-z]+\b", p):
            if min_len <= len(w) <= max_len:
                words.add(w)
    return sorted(words)


class Tok:
    """The checkpoint tokenizer (canonical ids) plus the chat template, as r4dx-cli/-server render
    a single user turn with thinking off."""

    def __init__(self, model_dir: Path | None = None):
        self.t = load_ref_tokenizer(model_dir or DEFAULT_MODEL_DIR, "canonical")

    def encode(self, text: str) -> list[int]:
        return self.t.encode(text)

    def decode(self, ids) -> str:
        return self.t.decode(ids)

    def render_user(self, content: str) -> str:
        return self.t.render_chat([{"role": "user", "content": content}],
                                  add_generation_prompt=True, enable_thinking=False)

    def prompt_tokens(self, content: str) -> int:
        return len(self.encode(self.render_user(content)))


def fit_units(n_max: int, build, count, target: int) -> tuple[int, object, int]:
    """Largest n in [1, n_max] with count(build(n)) <= target (count is monotone in n). Returns
    (n, build(n), count)."""
    lo, hi = 1, n_max
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        obj = build(mid)
        c = count(obj)
        if c <= target:
            best = (mid, obj, c)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError(f"even one unit exceeds the {target}-token target")
    return best


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
