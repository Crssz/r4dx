"""tools/quant2/gen_corpus.py

The SELF-GENERATED part of Hessian corpus v2 (docs/quant2.md 2.1, "Corpus"): every prompt of
`corpus_v2_prompts.json` is sent to r4dx-server's OpenAI API (docs/server.md), and one realistic
document or conversation per prompt is appended to `samples.jsonl`, which
`hessian_capture.py --gen-file` later reads.

**Why generate text at all.** The held-out KL eval (`tools/reference/kl_corpus/`) has Thai as its
worst segment by 2x, while Thai is ~1% of corpus v1 (calib.txt + kv_calib_corpus + WikiText-2 +
this repo's code, 172k tokens). Half of the remaining gap was measured to be corpus sampling
(rows/K ~10 for `mlp.down`'s K = 17408) and half domain shift, so v2 adds more rows AND the missing
domains: Thai prose, chat turns rendered through the real template (thinking blocks, im_start/im_end),
code in other languages, other scripts. Nothing is downloaded: our own quantized model writes it.
That is sound because the text only has to be realistic INPUT. The Hessians are measured later by
running the bf16 checkpoint over it, so the quantized model that wrote the text never enters `H`.

**The contract** (one JSON object per line, UTF-8, LF; `hessian_capture.py --gen-file` reads it):

    id               "<category>/<NNN>", unique, the sort key
    category         thai_prose | english_prose | chat | code | multilingual
    format           "raw" | "chat"
    enable_thinking  bool (only meaningful for "chat"; always false for "raw")
    messages         the FULL conversation, prompt messages AND every generated assistant turn; an
                     assistant turn carries "content" and, when thinking was on, "reasoning_content"
                     (the server's split at the first </think>, docs/server.md "reasoning_content")
    text             "raw": the final assistant content, stripped; null for "chat"
    finish_reason    of the final generation ("stop" | "length")
    prompt_tokens, completion_tokens   summed over the sample's generations (server usage)
    seed, temperature, top_p           the sampling settings (seed = the first turn's)
    rejected         bool; reject_reason string | null

The capture uses only `rejected == false` lines: "raw" -> `text` tokenized with no special tokens;
"chat" -> the checkpoint's own chat template over `messages` (enable_thinking as recorded,
add_generation_prompt=False). Extra fields, which a reader may ignore: `top_k`, `min_p`,
`max_tokens`, `turn_seeds`, `turn_finish_reasons`, `timings` (per turn, the server's `timings`
subset), `model`, `elapsed_s`, `prompt_sha256` (of the prompt entry, see "Resume").

**Requests.** Non-streaming `POST /v1/chat/completions` with the entry's messages, `max_tokens`,
`seed`, `chat_template_kwargs.enable_thinking` and the checkpoint's own recommended sampling
(C:\\AI\\models\\Qwen3.8-27B\\README.md "Best Practices"; thinking on is also its
generation_config.json):

    thinking off:  temperature 0.7, top_p 0.8,  top_k 20, min_p 0
    thinking on:   temperature 1.0, top_p 0.95, top_k 20, min_p 0

`top_k` is sent explicitly because r4dx-server's default is 0 (off). The card's other non-default,
`presence_penalty` 1.5 with thinking off, is NOT sent: r4dx-server does not implement it and would
ignore it silently (docs/server.md, `supported_parameters`), so the thinking-off text is sampled
without it. `reasoning_effort` is NOT sent either: the template's default (xhigh) then puts the
same system sentence into the prompt here as it will when the capture renders `messages` again, so
the captured token stream is the served one. `max_tokens` is per turn and, with thinking on, holds
the thought AND the answer; `load_prompts` requires `turns x max_tokens + PROMPT_RESERVE_TOKENS <=
SERVED_MAX_CTX`, so no turn's budget is cut by the server's context (it clamps silently).
The seed of turn `t` is `sample_seed(id, t)`: the low 31 bits of sha256("r4dx-corpus-v2:<id>#<t>"),
so a sample's seeds depend on its id only, never on order, resume or --limit. A two-turn chat sends
turn 1, appends the answer (with its `reasoning_content`, which the server replays into the
template, docs/server.md "Multi-turn"), appends the entry's follow-up and sends turn 2.

**Rejection** (the line is still written, with `rejected: true`, so a resume does not redo it and the
capture can report it). A turn is rejected for:

- empty output: no content after stripping; with thinking on and a non-empty `reasoning_content`
  this means `</think>` never came before `max_tokens` ("reasoning never closed"). A two-turn chat
  stops there: a follow-up to an empty answer is not a realistic conversation.
- a "raw" document shorter than `MIN_RAW_CHARS`, or containing a think tag (thinking is off for
  every raw entry, so a tag is a server or template fault).
- `reasoning_content` present with thinking off (the same kind of fault).
- degenerate repetition, in `content` or `reasoning_content`: a unit of >= `TANDEM_MIN_PERIOD`
  characters repeated >= `TANDEM_MIN_COPIES` times back to back (`tandem_repeat`), or a unique
  `NGRAM_N`-gram ratio below `NGRAM_MIN_UNIQUE` (`unique_ngram_ratio`, whitespace collapsed, so
  code indentation does not count as repetition).
- "kl overlap": `content` or `reasoning_content` shares a `KL_WINDOW`-character stretch (whitespace
  collapsed) with `tools/reference/kl_corpus/*.txt` outside that file's `#include` / Python import
  lines (`kl_segments`). hessian_capture.py's own disjointness gate refuses the WHOLE run on any
  sample sharing 50 characters with the same runs of lines; 48 here is stricter, so a sample kept
  here passes there. The include/import lines are skipped for the reason the gate skips them: the
  model.cpp / layer_golden.py excerpts open with the sorted standard includes/imports any C++ or
  Python file shares (idiom, not eval text), and rejecting a generated script for its header would
  only drop realistic code the capture accepts. The prompts themselves are held to the same check
  at load. A `--kl-dir` with no *.txt text is refused (exit 2) like hessian_capture.py's `kl_index`
  refuses it: both checks would pass vacuously.

**Resume.** `--out` is opened for append and every line is flushed and fsync'ed before the next
request. On start every id already in the file is skipped; a torn LAST line (a crash mid-write) is
truncated away and regenerated; any other malformed line, a duplicate id, an id that is not in the
prompt file, a line whose `prompt_sha256` differs from its entry's current hash (the prompt was
edited after the line was written), or one whose recorded sampling differs from `SAMPLING` (the
settings were changed after it was written) stops the run before any request: mixing two prompt
versions or two samplers under one id set is never silent. A transport failure (connection error,
timeout, HTTP 5xx or 429) is retried once; an HTTP 4xx other than 429 is not (the same body would be
refused again). Either way the sample is NOT written, so the next run retries it, and this run
moves on: a refusal can belong to one entry alone (a prompt past `--max-ctx`, a template render
error). After `--max-consecutive-failures` failed samples in a row the run aborts: exit 1 when the
last failure was a transport failure, exit 2 when it was a refusal (refusals in a row mean the
requests themselves are wrong). After a transport failure it also aborts at once when `/health`
stops answering (exit 1).

**A sample to drop by hand** (e.g. one `hessian_capture.py`'s disjointness gate names; with the
stricter window here that should only happen when kl_corpus/ changed after generation): set
its line's `"rejected": true` and give `"reject_reason"` a short "<kind>: <detail>" (e.g.
`"kl overlap: named by hessian_capture.py"`); a line without a reason is counted as
"unspecified" and warned about. Deleting the line to regenerate it does NOT help: its seeds depend
on its id only, so the same server and container write the same text again.

**Manifest.** `gen_manifest.json` (next to `--out` unless `--manifest`) is rewritten at the end of
every run, also on abort or Ctrl+C: the prompt file's sha256, the server's `/health` and
`/v1/models` answers, the container paths and server command line the launcher passes in, counts
(per category, per reject reason, finish reasons), token totals (overall and per category, all and
accepted), speculative acceptance, the samples file's sha256, and every run's wall time (the list
is carried over from the previous manifest, so a resumed corpus keeps its whole history).

**Order.** Categories are interleaved in proportion to their sizes (`processing_order`), so a run
cut short -- or a `--limit` smoke -- still has every category in its planned share.

Usage (stdlib only; any python >= 3.9; tools/quant2/gen_corpus.ps1 starts the server and runs this):

    python tools\\quant2\\gen_corpus.py --dry-run
    python tools\\quant2\\gen_corpus.py --server http://127.0.0.1:18080 `
        --out D:\\models\\r4dx\\corpus-v2\\samples.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import platform
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "corpus_v2_prompts.json"
DEFAULT_OUT = Path(r"D:\models\r4dx\corpus-v2\samples.jsonl")
DEFAULT_SERVER = "http://127.0.0.1:18080"
MANIFEST_NAME = "gen_manifest.json"

PROMPTS_FORMAT = "r4dx-corpus-v2-prompts"
PROMPTS_VERSION = 1
MANIFEST_FORMAT = "r4dx-corpus-v2-gen"
MANIFEST_VERSION = 1

#: Contract order; also the tie-break of `processing_order`.
CATEGORIES = ("thai_prose", "english_prose", "chat", "code", "multilingual")
#: Every category but "chat" is a raw document: the capture tokenizes its text without a template.
FORMAT_OF = {"thai_prose": "raw", "english_prose": "raw", "chat": "chat", "code": "raw",
             "multilingual": "raw"}
ID_RE = re.compile(r"^(%s)/(\d{3})$" % "|".join(CATEGORIES))
ENTRY_KEYS = {"id", "category", "format", "enable_thinking", "max_tokens", "messages", "followups"}
#: The launcher's server runs --max-ctx 8192 (gen_corpus.ps1). r4dx-server clamps a turn's
#: max_tokens SILENTLY to what the context has left (engine.cpp, `ctx_budget`), so a budget the
#: context cannot hold would quietly become a shorter one.
SERVED_MAX_CTX = 8192
#: What `load_prompts` sets aside for an entry's rendered prompt -- system, user turn, every
#: follow-up, the template's control tokens and (thinking on) the xhigh system sentence -- when it
#: checks `turns x max_tokens + PROMPT_RESERVE_TOKENS <= SERVED_MAX_CTX` (turn t's prompt holds the
#: earlier turns' output, so the sum bounds the last turn's context). Measured with the checkpoint's
#: tokenizer on corpus_v2_prompts.json (each follow-up after an empty assistant turn, generation
#: prompt on): at most 443 tokens (chat/076), 197 for a two-turn chat (chat/013). The slack covers
#: a replayed turn tokenizing a few tokens longer than it was generated.
PROMPT_RESERVE_TOKENS = 1024
MAX_TOKENS_MIN = 64

#: The checkpoint's recommended sampling (README.md "Best Practices"; thinking on is also its
#: generation_config.json), per enable_thinking. top_k is explicit because r4dx-server's own default
#: is 0 (off). The card's presence_penalty 1.5 (thinking off) is not sent: r4dx-server ignores it.
#: Recorded on every line and checked on resume (`check_existing`), so a change here after a partial
#: run is refused instead of mixed into one corpus.
SAMPLING = {
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}
SEED_DOMAIN = "r4dx-corpus-v2"

#: The held-out KL corpus. No prompt (refused at load) and no kept generated text (rejected as "kl
#: overlap") may share a `KL_WINDOW`-character stretch, whitespace collapsed, with it outside its
#: include/import lines (next): calibrating on the eval text would flatter the result.
#: hessian_capture.py's gate on the rendered samples uses 50 over the same runs of lines.
KL_CORPUS_DIR = REPO_ROOT / "tools" / "reference" / "kl_corpus"
KL_WINDOW = 48
#: The lines of a KL file neither check compares against: hessian_capture.py's KL_BOILERPLATE_LINE
#: and KL_IMPORT_CONTINUATION, copied verbatim because this script is stdlib only (the reader
#: imports numpy); test_gen_corpus.py (k) checks that `kl_segments` and the capture's
#: `kl_gate_segments` split every kl_corpus file into the same runs. An `#include <x>` / `#include
#: "x"` directive or a Python import statement (`import a.b as c, d`, `from a import b, c`, `from a
#: import (b, c)`, every line of a `from a import (` block up to its `)`), each with an optional
#: trailing comment. `from __future__ import annotations\n\nimport argparse` alone is 50 normalized
#: characters of layer_golden.py's header, and a generated CLI script may well open with it.
_PY_NAMES = r"\w+(?:\s+as\s+\w+)?(?:\s*,\s*\w+(?:\s+as\s+\w+)?)*\s*,?"
KL_BOILERPLATE_LINE = re.compile(
    r"\s*(?:#\s*include\s*(?:<[^<>]*>|\"[^\"]*\")\s*(?://.*|/\*.*)?"
    r"|import\s+[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*\s*(?:#.*)?"
    rf"|from\s+[\w.]+\s+import\s+(?:\*|{_PY_NAMES}|\(\s*{_PY_NAMES}\s*\)|(?P<open>\()\s*(?:{_PY_NAMES})?)"
    r"\s*(?:#.*)?)")
KL_IMPORT_CONTINUATION = re.compile(rf"\s*(?:{_PY_NAMES})?\s*(?P<close>\))?\s*(?:#.*)?")

#: Rejection thresholds (module docstring, "Rejection").
MIN_RAW_CHARS = 200
TANDEM_MIN_PERIOD = 32
TANDEM_MIN_COPIES = 4
NGRAM_N = 20
#: Measured on 1,039 real 6,000-char chunks of this repo (C++/HIP/Python/PowerShell/docs, Thai and
#: English calibration prose): the lowest ratio was 0.452 (a block of near-identical C++ test
#: assertions) and none repeated a 32-char unit 4x back to back. A counter-only loop ("Step 1: <same
#: sentence>", "Step 2: ...") scores ~0.25. Exact loops are the tandem check's job.
NGRAM_MIN_UNIQUE = 0.35
#: Below this many n-grams the ratio says nothing; short texts are judged by the tandem check only.
NGRAM_MIN_TOTAL = 400

#: Server `timings` fields kept per turn (docs/server.md "timings").
TIMING_KEYS = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second",
               "draft_n", "draft_n_accepted")
#: Record fields that legitimately differ between two runs of the same sample.
VOLATILE_FIELDS = ("elapsed_s", "timings")


class PromptFileError(Exception):
    """The prompt file is malformed; nothing was sent."""


class OutFileError(Exception):
    """The existing samples file cannot be resumed safely; nothing was sent."""


class KlCorpusError(Exception):
    """--kl-dir holds no *.txt text: every disjointness check would pass vacuously."""


class RequestFailed(Exception):
    """The server refused a request (HTTP 4xx other than 429): the same body would be refused
    again, so it is not retried."""


class ResponseInvalid(Exception):
    """The server answered 200 with a body that does not have the chat.completion shape."""


class TransportFailed(Exception):
    """No usable answer after the one retry (connection error, timeout, 5xx, 429)."""


# --------------------------------------------------------------------------------------------
# Logging (stdout, plus an optional append-only log file)
# --------------------------------------------------------------------------------------------


class Log:
    def __init__(self, path: Path | None = None):
        self.file = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, "a", encoding="utf-8", newline="\n")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.file is not None:
            self.file.write(msg + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


# --------------------------------------------------------------------------------------------
# The prompt file
# --------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_sha256(entry: dict) -> str:
    """Hash of one prompt entry's canonical JSON: what a samples line was generated from."""
    canon = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canon.encode("utf-8"))


def sample_seed(sample_id: str, turn: int = 0) -> int:
    """The seed of turn `turn` of sample `sample_id`: 31 bits of sha256, a function of the id only."""
    d = hashlib.sha256(f"{SEED_DOMAIN}:{sample_id}#{turn}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") & 0x7FFFFFFF


def validate_entry(e, where: str) -> list[str]:
    """Every problem with one prompt entry, as strings (empty = valid)."""
    if not isinstance(e, dict):
        return [f"{where}: not an object"]
    p: list[str] = []
    missing = sorted(ENTRY_KEYS - {"followups"} - set(e))
    if missing:
        p.append(f"{where}: missing {missing}")
    unknown = sorted(set(e) - ENTRY_KEYS)
    if unknown:
        p.append(f"{where}: unknown keys {unknown}")
    if missing:
        return p
    sid, cat = e["id"], e["category"]
    m = ID_RE.match(sid) if isinstance(sid, str) else None
    if not m:
        p.append(f"{where}: id {sid!r} is not '<category>/<NNN>'")
    elif m.group(1) != cat:
        p.append(f"{where}: id {sid!r} does not start with its category {cat!r}")
    if not isinstance(cat, str) or cat not in FORMAT_OF:
        p.append(f"{where}: category {cat!r} is not one of {list(CATEGORIES)}")
    elif e["format"] != FORMAT_OF[cat]:
        p.append(f"{where}: category {cat} must have format {FORMAT_OF[cat]!r}, not {e['format']!r}")
    if not isinstance(e["enable_thinking"], bool):
        p.append(f"{where}: enable_thinking must be a bool")
    elif e["enable_thinking"] and e["format"] == "raw":
        p.append(f"{where}: a raw document is generated with thinking off (its text is the answer only)")
    mt = e["max_tokens"]
    fu = e.get("followups")
    turns = 1 + (len(fu) if isinstance(fu, list) else 0)
    most = (SERVED_MAX_CTX - PROMPT_RESERVE_TOKENS) // turns
    if not (isinstance(mt, int) and not isinstance(mt, bool) and MAX_TOKENS_MIN <= mt <= most):
        p.append(f"{where}: max_tokens must be an int in [{MAX_TOKENS_MIN}, {most}] for {turns} "
                 f"turn(s): turns x max_tokens + {PROMPT_RESERVE_TOKENS} (the prompt's reserve) must "
                 f"fit the served --max-ctx {SERVED_MAX_CTX}")
    msgs = e["messages"]
    if not isinstance(msgs, list) or not msgs:
        p.append(f"{where}: messages must be a non-empty list")
    else:
        for i, msg in enumerate(msgs):
            if not (isinstance(msg, dict) and set(msg) == {"role", "content"}
                    and isinstance(msg["content"], str) and msg["content"].strip()):
                p.append(f"{where}: messages[{i}] must be exactly {{role, content}} with non-empty text")
                continue
            if msg["role"] == "system":
                if i != 0:
                    p.append(f"{where}: messages[{i}]: a system message must come first")
            elif msg["role"] != "user":
                p.append(f"{where}: messages[{i}]: role {msg['role']!r} (the prompt holds system/user "
                         "only; the generator writes every assistant turn)")
        if isinstance(msgs[-1], dict) and msgs[-1].get("role") != "user":
            p.append(f"{where}: the last prompt message must be the user's")
    if "followups" in e:
        f = e["followups"]
        if e["format"] != "chat":
            p.append(f"{where}: followups are for format 'chat' only")
        if not (isinstance(f, list) and f and all(isinstance(x, str) and x.strip() for x in f)):
            p.append(f"{where}: followups must be a non-empty list of non-empty strings")
    return p


def entry_texts(e: dict) -> list[str]:
    return [m["content"] for m in e["messages"]] + list(e.get("followups", []))


def _collapse(text: str) -> str:
    return " ".join(text.split())


def kl_segments(text: str) -> list[str]:
    """The runs of consecutive lines of one KL file that are not boilerplate (KL_BOILERPLATE_LINE and
    the continuation lines of a `from a import (` block), each whitespace-collapsed -- its non-empty
    collapsed lines joined by one space: the texts of hessian_capture.kl_gate_segments. A file
    without boilerplate is one run, `_collapse(text)`; no shingle spans a boilerplate line."""
    segs: list[str] = []
    pieces: list[str] = []
    in_import = False
    for line in text.split("\n"):
        if in_import:
            m = KL_IMPORT_CONTINUATION.fullmatch(line)
            if m:
                in_import = m.group("close") is None
                continue
            in_import = False  # not an import line after all: an unclosed `(`, read on as text
        m = KL_BOILERPLATE_LINE.fullmatch(line)
        if m:
            if pieces:
                segs.append(" ".join(pieces))
                pieces = []
            in_import = m.group("open") is not None
            continue
        piece = _collapse(line)
        if piece:
            pieces.append(piece)
    if pieces:
        segs.append(" ".join(pieces))
    return segs


def kl_shingle_index(kl_dir: Path, window: int = KL_WINDOW) -> tuple[dict[str, str], dict[str, str]]:
    """({every `window`-char substring of each `kl_segments` run of each kl_dir/*.txt: its file
    name}, {file name: sha256}). Raises KlCorpusError when that yields no shingle at all (kl_dir
    missing, no *.txt in it, or only empty ones): the check would pass vacuously, so a mistyped
    --kl-dir is refused here, as hessian_capture.py's `kl_index` refuses it, not found at capture
    time."""
    shingles: dict[str, str] = {}
    shas: dict[str, str] = {}
    files = sorted(Path(kl_dir).glob("*.txt")) if Path(kl_dir).is_dir() else []
    for path in files:
        raw = path.read_bytes()
        shas[path.name] = sha256_bytes(raw)
        for t in kl_segments(raw.decode("utf-8", errors="replace")):
            for i in range(len(t) - window + 1):
                shingles.setdefault(t[i:i + window], path.name)
    if not shingles:
        raise KlCorpusError(f"--kl-dir {kl_dir}: no *.txt text of {window}+ characters to check "
                            f"against ({len(files)} file(s)); the KL disjointness check would pass "
                            f"vacuously -- point it at the held-out KL corpus ({KL_CORPUS_DIR})")
    return shingles, shas


def first_kl_overlap(text: str, shingles: dict[str, str],
                     window: int = KL_WINDOW) -> tuple[str, str] | None:
    """(the first shared substring, its kl file) or None."""
    t = _collapse(text)
    for i in range(len(t) - window + 1):
        f = shingles.get(t[i:i + window])
        if f is not None:
            return t[i:i + window], f
    return None


def kl_overlaps(entries: list[dict], kl_dir: Path, window: int = KL_WINDOW) -> list[str]:
    """Prompts sharing a `window`-character stretch (whitespace collapsed) with the KL corpus."""
    shingles, _ = kl_shingle_index(kl_dir, window)
    hits = []
    for e in entries:
        for text in entry_texts(e):
            hit = first_kl_overlap(text, shingles, window)
            if hit:
                hits.append(f"{e['id']}: shares {hit[0]!r} with {kl_dir}/{hit[1]}")
                break
    return hits


def load_prompts(path: Path, kl_dir: Path | None = KL_CORPUS_DIR) -> tuple[list[dict], str]:
    """(entries, sha256 of the file). Raises PromptFileError listing every problem, KlCorpusError
    when `kl_dir` (None: no KL check, for tests) has no text to check the prompts against."""
    raw = Path(path).read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise PromptFileError(f"{path}: not UTF-8 JSON ({e})") from None
    if not isinstance(doc, dict):
        raise PromptFileError(f"{path}: the top level must be an object")
    if doc.get("format") != PROMPTS_FORMAT or doc.get("version") != PROMPTS_VERSION:
        raise PromptFileError(f"{path}: format/version must be {PROMPTS_FORMAT!r}/{PROMPTS_VERSION}")
    entries = doc.get("samples")
    if not isinstance(entries, list) or not entries:
        raise PromptFileError(f"{path}: 'samples' must be a non-empty list")
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        where = f"samples[{i}]" + (f" ({e.get('id')})" if isinstance(e, dict) else "")
        problems += validate_entry(e, where)
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            if e["id"] in seen:
                problems.append(f"{where}: duplicate id")
            seen.add(e["id"])
    if not problems and kl_dir is not None:
        problems += kl_overlaps(entries, Path(kl_dir))
    if problems:
        raise PromptFileError(f"{path}: {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return entries, sha256_bytes(raw)


def processing_order(entries: list[dict]) -> list[dict]:
    """Categories interleaved in proportion to their sizes; ids ascending within a category."""
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)
    keyed = []
    for cat, es in by_cat.items():
        es = sorted(es, key=lambda x: x["id"])
        for i, e in enumerate(es):
            keyed.append(((i + 0.5) / len(es), CATEGORIES.index(cat), e["id"], e))
    keyed.sort(key=lambda k: k[:3])
    return [k[3] for k in keyed]


def turn_count(e: dict) -> int:
    return 1 + len(e.get("followups", []))


def prompt_table(entries: list[dict]) -> list[str]:
    rows = ["| category | format | samples | thinking | two-turn | max_tokens budget |",
            "|---|---|--:|--:|--:|--:|"]
    tot = [0, 0, 0, 0]
    for cat in CATEGORIES:
        es = [e for e in entries if e["category"] == cat]
        if not es:
            continue
        n, th = len(es), sum(e["enable_thinking"] for e in es)
        tt = sum(turn_count(e) > 1 for e in es)
        budget = sum(e["max_tokens"] * turn_count(e) for e in es)
        tot = [tot[0] + n, tot[1] + th, tot[2] + tt, tot[3] + budget]
        rows.append(f"| {cat} | {FORMAT_OF[cat]} | {n} | {th} | {tt} | {budget:,} |")
    rows.append(f"| **total** | | **{tot[0]}** | {tot[1]} | {tot[2]} | {tot[3]:,} |")
    return rows


# --------------------------------------------------------------------------------------------
# Degenerate-output detection
# --------------------------------------------------------------------------------------------


def tandem_repeat(text: str, min_period: int = TANDEM_MIN_PERIOD,
                  min_copies: int = TANDEM_MIN_COPIES) -> tuple[int, int, int] | None:
    """A back-to-back repetition: `(period, copies, start)` when some unit of `period >=
    min_period` characters occurs `copies >= min_copies` times in a row from character `start`
    (i.e. `text[start : start + (min_copies-1)*period] == text[start+period : start +
    min_copies*period]`); None when there is none. `copies` is extended as far as the run goes.

    Exhaustive, in roughly linear time: if such a run exists at (i, p), the `min_period`-gram at i
    also occurs at i + p, so testing every pair of occurrences (a, b) with `b - a >= min_period`
    of every gram that occurs at least `min_copies` times finds it. Real text has few such grams
    (none of the 1,039 repo chunks behind NGRAM_MIN_UNIQUE has a run), and a degenerate text
    returns at its first pair. A short-period loop ("ab ab ab ...") is caught through the
    multiples of its period that are >= min_period.
    """
    n = len(text)
    k = min_period
    if n < k * min_copies:
        return None
    occ: dict[str, list[int]] = {}
    for i in range(n - k + 1):
        occ.setdefault(text[i:i + k], []).append(i)
    span_copies = min_copies - 1
    for positions in occ.values():
        if len(positions) < min_copies:
            continue
        for x, a in enumerate(positions):
            for b in positions[x + 1:]:
                p = b - a
                if p < k:
                    continue
                if a + min_copies * p > n:
                    break
                if text[a:a + span_copies * p] == text[b:b + span_copies * p]:
                    copies = min_copies
                    while (a + (copies + 1) * p <= n and text[a + (copies - 1) * p:a + copies * p]
                           == text[a + copies * p:a + (copies + 1) * p]):
                        copies += 1
                    return p, copies, a
    return None


def unique_ngram_ratio(text: str, n: int = NGRAM_N) -> float:
    """Distinct / total character n-grams, whitespace runs collapsed to one space (so indentation
    and blank lines are not "repetition"); 1.0 when there are fewer than NGRAM_MIN_TOTAL n-grams."""
    t = _collapse(text)
    total = len(t) - n + 1
    if total < NGRAM_MIN_TOTAL:
        return 1.0
    return len({t[i:i + n] for i in range(total)}) / total


def degenerate_reason(text: str) -> str | None:
    tr = tandem_repeat(text)
    if tr is not None:
        p, copies, start = tr
        return f"a {p}-char unit repeated {copies}x back to back from char {start}"
    r = unique_ngram_ratio(text)
    if r < NGRAM_MIN_UNIQUE:
        return f"unique {NGRAM_N}-gram ratio {r:.3f} < {NGRAM_MIN_UNIQUE}"
    return None


def check_turn(fmt: str, thinking: bool, turn: int, content: str, reasoning: str | None,
               has_reasoning_key: bool) -> str | None:
    """The reject reason for one generated turn ("<kind>: <detail>"), or None to keep it."""
    where = f"turn {turn + 1}"
    if not thinking and has_reasoning_key:
        return f"unexpected reasoning_content: thinking was off ({where})"
    body = content.strip()
    if not body:
        if thinking and reasoning and reasoning.strip():
            return f"reasoning never closed: no content after {len(reasoning)} reasoning chars ({where})"
        return f"empty output: no content ({where})"
    if fmt == "raw":
        if len(body) < MIN_RAW_CHARS:
            return f"too short: {len(body)} chars < {MIN_RAW_CHARS} ({where})"
        if "<think>" in body or "</think>" in body:
            return f"think tag in a raw document: <think> or </think> in content ({where})"
    for field, text in (("content", content), ("reasoning_content", reasoning or "")):
        why = degenerate_reason(text)
        if why:
            return f"degenerate {field}: {why} ({where})"
    return None


#: The kind of a rejected line whose reject_reason is missing or blank (a line rejected by hand).
UNSPECIFIED_REJECT = "unspecified"


def reject_kind(reason) -> str:
    """The "<kind>" of a "<kind>: <detail>" reject_reason; UNSPECIFIED_REJECT for a missing, blank
    or non-string one (a line marked rejected by hand without a reason must not crash the manifest)."""
    if not isinstance(reason, str) or not reason.strip():
        return UNSPECIFIED_REJECT
    return reason.split(":", 1)[0].strip()


# --------------------------------------------------------------------------------------------
# HTTP client (stdlib urllib; proxies bypassed -- the server is on loopback)
# --------------------------------------------------------------------------------------------


def _http_error_detail(e: urllib.error.HTTPError) -> str:
    try:
        body = e.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- best effort, the status code is the message
        return str(e.reason)
    try:
        err = json.loads(body).get("error", {})
        return str(err.get("message") or body)[:400]
    except (ValueError, AttributeError):
        return body[:400]


class Client:
    def __init__(self, base_url: str, timeout: float, retry_wait: float, log=print):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_wait = retry_wait
        self.log = log
        # An HTTP(S)_PROXY in the environment must not route loopback traffic.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _once(self, method: str, path: str, body: dict | None, timeout: float) -> dict:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json; charset=utf-8",
                                              "Accept": "application/json"})
        with self.opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def request(self, method: str, path: str, body: dict | None = None,
                timeout: float | None = None) -> dict:
        """One request, retried ONCE on a transport failure; 4xx (not 429) raises RequestFailed."""
        timeout = self.timeout if timeout is None else timeout
        last = ""
        for attempt in (1, 2):
            try:
                return self._once(method, path, body, timeout)
            except urllib.error.HTTPError as e:
                detail = _http_error_detail(e)
                if 400 <= e.code < 500 and e.code != 429:
                    raise RequestFailed(f"HTTP {e.code} from {method} {path}: {detail}") from None
                last = f"HTTP {e.code}: {detail}"
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, socket.timeout, ValueError) as e:
                # ValueError covers a body that is not UTF-8 JSON (json.JSONDecodeError,
                # UnicodeDecodeError): a torn response, retried like a dropped connection.
                last = f"{type(e).__name__}: {e}"
            if attempt == 1:
                self.log(f"[gen] {method} {path} failed ({last}); retrying once in {self.retry_wait:g} s")
                time.sleep(self.retry_wait)
        raise TransportFailed(f"{method} {path} failed twice; last error: {last}")

    def healthy(self) -> bool:
        try:
            return self._once("GET", "/health", None, 10.0).get("status") == "ok"
        except Exception:  # noqa: BLE001 -- any failure means "not healthy"
            return False


# --------------------------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------------------------


def request_body(entry: dict, messages: list[dict], turn: int) -> dict:
    thinking = entry["enable_thinking"]
    return {"messages": messages, "max_tokens": entry["max_tokens"],
            "seed": sample_seed(entry["id"], turn), "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking}, **SAMPLING[thinking]}


def _parse_choice(resp: dict) -> tuple[dict, str | None]:
    try:
        choice = resp["choices"][0]
        msg = choice["message"]
        if not isinstance(msg, dict):
            raise TypeError("message is not an object")
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError("content is not a string")
        return msg, choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as e:
        raise ResponseInvalid(f"not a chat.completion ({e}): {json.dumps(resp)[:300]}") from None


def kl_reason(turn: int, content: str, reasoning: str | None, shingles: dict[str, str]) -> str | None:
    for field, text in (("reasoning_content", reasoning or ""), ("content", content)):
        hit = first_kl_overlap(text, shingles) if shingles else None
        if hit:
            return (f"kl overlap: {field} shares {hit[0]!r} with kl_corpus/{hit[1]} "
                    f"(turn {turn + 1})")
    return None


def generate_sample(entry: dict, client: Client, kl_shingles: dict[str, str] | None = None) -> dict:
    """Every turn of one entry -> one contract record. Raises TransportFailed / RequestFailed /
    ResponseInvalid; never returns a partial record."""
    thinking = entry["enable_thinking"]
    messages = [dict(m) for m in entry["messages"]]
    followups = entry.get("followups", [])
    seeds, finishes, timings = [], [], []
    prompt_tokens = completion_tokens = 0
    reason = None
    model = None
    content = ""
    t0 = time.monotonic()
    for turn in range(1 + len(followups)):
        if turn > 0:
            messages.append({"role": "user", "content": followups[turn - 1]})
        body = request_body(entry, messages, turn)
        resp = client.request("POST", "/v1/chat/completions", body)
        msg, finish = _parse_choice(resp)
        content = msg.get("content") or ""
        assistant = {"role": "assistant", "content": content}
        reasoning = None
        if thinking:
            rc = msg.get("reasoning_content")
            reasoning = rc if isinstance(rc, str) else ""
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
        usage = resp.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        t = resp.get("timings") or {}
        timings.append({k: t[k] for k in TIMING_KEYS if k in t})
        seeds.append(body["seed"])
        finishes.append(finish)
        model = resp.get("model", model)
        reason = (check_turn(entry["format"], thinking, turn, content, reasoning,
                             "reasoning_content" in msg)
                  or kl_reason(turn, content, reasoning, kl_shingles or {}))
        if reason:
            break
    samp = SAMPLING[thinking]
    return {
        "id": entry["id"],
        "category": entry["category"],
        "format": entry["format"],
        "enable_thinking": thinking,
        "messages": messages,
        "text": content.strip() if entry["format"] == "raw" else None,
        "finish_reason": finishes[-1],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "seed": seeds[0],
        "temperature": samp["temperature"],
        "top_p": samp["top_p"],
        "rejected": reason is not None,
        "reject_reason": reason,
        "top_k": samp["top_k"],
        "min_p": samp["min_p"],
        "max_tokens": entry["max_tokens"],
        "turn_seeds": seeds,
        "turn_finish_reasons": finishes,
        "timings": timings,
        "model": model,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "prompt_sha256": entry_sha256(entry),
    }


# --------------------------------------------------------------------------------------------
# The samples file: resume and append
# --------------------------------------------------------------------------------------------


def read_existing(path: Path, log=print) -> dict[str, dict]:
    """The records already in `path`, by id. A torn LAST line is truncated away (and a last line
    missing only its newline gets one); anything else malformed raises OutFileError."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    records: dict[str, dict] = {}
    lines = data.split(b"\n")
    offset = 0
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        if last and line == b"":
            break
        if not line.strip():
            offset += len(line) + 1
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
            if not (isinstance(rec, dict) and isinstance(rec.get("id"), str)):
                raise ValueError("not a sample object")
        except (UnicodeDecodeError, ValueError) as e:
            if last:
                with open(path, "r+b") as f:
                    f.truncate(offset)
                log(f"[gen] {path}: dropped a torn last line ({len(line)} bytes: {e}); "
                    "that sample is regenerated")
                break
            raise OutFileError(f"{path}: line {i + 1} is not a sample ({e}); only a torn LAST line "
                               "is repaired automatically") from None
        if rec["id"] in records:
            raise OutFileError(f"{path}: line {i + 1}: duplicate id {rec['id']}")
        records[rec["id"]] = rec
        offset += len(line) + 1
        if last:  # parsed, but the newline never made it to disk
            with open(path, "ab") as f:
                f.write(b"\n")
    return records


#: The fields of an existing line that `summarize` reads, with their types. A hand-edited line that
#: breaks one is refused before any request, not found by a crash while writing the manifest.
RECORD_TYPES = {"category": str, "rejected": bool, "prompt_tokens": int, "completion_tokens": int}


def check_existing(records: dict[str, dict], by_id: dict[str, dict], path: Path, log=print) -> None:
    """Raises OutFileError for lines a resume must not build on: an id not in the prompt file, a
    prompt entry edited after its line was written (`prompt_sha256`), a line sampled with settings
    other than today's `SAMPLING` (a corpus mixing two samplers is never silent), a RECORD_TYPES
    field missing or mistyped. Warns about a rejected line without a reason (module docstring, "A
    sample to drop by hand"); `summarize` counts it as UNSPECIFIED_REJECT."""
    problems = []
    unreasoned = []
    for sid, rec in records.items():
        e = by_id.get(sid)
        if e is None:
            problems.append(f"{sid}: not in the prompt file")
            continue
        if rec.get("prompt_sha256") != entry_sha256(e):
            problems.append(f"{sid}: its prompt entry changed after the line was written")
        want = SAMPLING[e["enable_thinking"]]
        got = {k: rec.get(k) for k in want}
        if got != want:
            problems.append(f"{sid}: sampled with {got}; SAMPLING is now {want}")
        bad = [k for k, t in RECORD_TYPES.items()
               if not isinstance(rec.get(k), t) or (t is int and isinstance(rec.get(k), bool))]
        if bad:
            problems.append(f"{sid}: {', '.join(bad)} missing or not "
                            + ", ".join(RECORD_TYPES[k].__name__ for k in bad))
        elif rec["category"] != e["category"]:
            problems.append(f"{sid}: category {rec['category']!r}, its entry's is {e['category']!r}")
        elif rec["rejected"] and reject_kind(rec.get("reject_reason")) == UNSPECIFIED_REJECT:
            unreasoned.append(sid)
    if problems:
        raise OutFileError(f"{path}: {len(problems)} line(s) do not match the prompt file or this "
                           "generator (use a new --out, or delete those lines to regenerate them):\n  "
                           + "\n  ".join(problems[:40]))
    if unreasoned:
        log(f"[gen] WARNING: {len(unreasoned)} line(s) rejected without a reject_reason, counted as "
            f"{UNSPECIFIED_REJECT!r}: {', '.join(sorted(unreasoned)[:20])}")


def append_record(f, rec: dict) -> None:
    f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def summarize(records: dict[str, dict], entries: list[dict]) -> dict:
    per_cat = {c: {"planned": 0, "done": 0, "accepted": 0, "rejected": 0, "prompt_tokens": 0,
                   "completion_tokens": 0, "completion_tokens_accepted": 0} for c in CATEGORIES}
    for e in entries:
        per_cat[e["category"]]["planned"] += 1
    reasons: dict[str, int] = {}
    finishes: dict[str, int] = {}
    draft_n = draft_acc = 0
    for rec in records.values():
        c = per_cat[rec["category"]]
        c["done"] += 1
        c["prompt_tokens"] += rec["prompt_tokens"]
        c["completion_tokens"] += rec["completion_tokens"]
        if rec["rejected"]:
            c["rejected"] += 1
            k = reject_kind(rec.get("reject_reason"))
            reasons[k] = reasons.get(k, 0) + 1
        else:
            c["accepted"] += 1
            c["completion_tokens_accepted"] += rec["completion_tokens"]
        for fr in rec.get("turn_finish_reasons") or [rec.get("finish_reason")]:
            finishes[str(fr)] = finishes.get(str(fr), 0) + 1
        for t in rec.get("timings") or []:
            draft_n += t.get("draft_n", 0)
            draft_acc += t.get("draft_n_accepted", 0)
    tot = {k: sum(c[k] for c in per_cat.values()) for k in next(iter(per_cat.values()))}
    return {
        "planned": tot["planned"], "done": tot["done"], "accepted": tot["accepted"],
        "rejected": tot["rejected"], "missing": tot["planned"] - tot["done"],
        "complete": tot["done"] == tot["planned"],
        "reject_reasons": dict(sorted(reasons.items())),
        "finish_reasons_per_turn": dict(sorted(finishes.items())),
        "tokens": {"prompt": tot["prompt_tokens"], "completion": tot["completion_tokens"],
                   "completion_accepted": tot["completion_tokens_accepted"]},
        "per_category": per_cat,
        "speculation": {"draft_n": draft_n, "draft_n_accepted": draft_acc,
                        "acceptance": round(draft_acc / draft_n, 4) if draft_n else None},
    }


def write_json_atomic(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def previous_runs(manifest_path: Path, log) -> list[dict]:
    if not manifest_path.exists():
        return []
    try:
        runs = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
        return runs if isinstance(runs, list) else []
    except (ValueError, OSError, AttributeError) as e:
        log(f"[gen] WARNING: {manifest_path} unreadable ({e}); its run history is not carried over")
        return []


def build_manifest(args, entries, prompts_sha, records, runs, server, disjoint) -> dict:
    out = Path(args.out)
    return {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": {"script": "tools/quant2/gen_corpus.py", "sha256": sha256_file(Path(__file__)),
                      "python": sys.version.split()[0], "host": platform.node()},
        "prompts": {"path": str(args.prompts), "sha256": prompts_sha, "entries": len(entries)},
        "samples": {"path": str(out), "sha256": sha256_file(out) if out.exists() else None,
                    "lines": len(records)},
        "server": server,
        "sampling": {"thinking_off": SAMPLING[False], "thinking_on": SAMPLING[True],
                     "source": "the checkpoint's README.md 'Best Practices' (thinking on: also its "
                               "generation_config.json)",
                     "presence_penalty": "not sent: r4dx-server ignores it (the card's thinking-off "
                                         "value is 1.5)",
                     "seed": "turn t of sample <id>: low 31 bits of sha256('r4dx-corpus-v2:<id>#<t>')",
                     "reasoning_effort": "not sent (template default)",
                     "max_tokens": f"per turn, from the prompt entry; turns x max_tokens + "
                                   f"{PROMPT_RESERVE_TOKENS} <= {SERVED_MAX_CTX}"},
        "rejection": {"min_raw_chars": MIN_RAW_CHARS, "tandem_min_period": TANDEM_MIN_PERIOD,
                      "tandem_min_copies": TANDEM_MIN_COPIES, "ngram_n": NGRAM_N,
                      "ngram_min_unique": NGRAM_MIN_UNIQUE, "ngram_min_total": NGRAM_MIN_TOTAL,
                      "kl_disjointness": disjoint},
        "counts": summarize(records, entries),
        "wall_s_total": round(sum(r.get("wall_s", 0.0) for r in runs), 1),
        "runs": runs,
    }


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def fmt_duration(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--server", default=DEFAULT_SERVER, help="r4dx-server base URL")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="samples.jsonl (appended)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help=f"default: {MANIFEST_NAME} next to --out")
    ap.add_argument("--log", type=Path, default=None, help="also append progress lines here")
    ap.add_argument("--limit", type=int, default=0, help="generate at most N samples this run")
    ap.add_argument("--only-category", choices=CATEGORIES, default=None)
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds per request (a non-streaming 7168-token turn must fit)")
    ap.add_argument("--retry-wait", type=float, default=5.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--container", default=None, help="provenance: the target container served")
    ap.add_argument("--dflash-container", default=None, help="provenance: the DFlash2 drafter, if any")
    ap.add_argument("--dflash-k", type=int, default=None, help="provenance")
    ap.add_argument("--server-cmdline", default=None, help="provenance: how the server was started")
    ap.add_argument("--kl-dir", type=Path, default=KL_CORPUS_DIR,
                    help="held-out KL corpus neither the prompts nor kept text may overlap "
                         "(refused when it has no *.txt text)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the prompt file, print the plan and the first request; no HTTP")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    log = Log(args.log)
    try:
        return _run(args, log)
    finally:
        log.close()


def _run(args, log) -> int:
    try:
        entries, prompts_sha = load_prompts(args.prompts, args.kl_dir)
        kl_shingles, kl_shas = kl_shingle_index(args.kl_dir)
    except (PromptFileError, KlCorpusError, OSError) as e:
        log(f"[gen] ERROR: {e}")
        return 2
    by_id = {e["id"]: e for e in entries}
    order = processing_order(entries)
    if args.dry_run:
        for row in prompt_table(entries):
            log(row)
        first = order[0]
        log(f"[gen] prompt file {args.prompts} sha256 {prompts_sha}")
        log(f"[gen] first request ({first['id']}): "
            + json.dumps(request_body(first, first["messages"], 0), ensure_ascii=False)[:600])
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or out.with_name(MANIFEST_NAME)
    try:
        records = read_existing(out, log)
        check_existing(records, by_id, out, log)
    except OutFileError as e:
        log(f"[gen] ERROR: {e}")
        return 2

    disjoint = {"kl_dir": str(args.kl_dir), "window_chars": KL_WINDOW, "kl_files": kl_shas,
                "shingles": len(kl_shingles), "normalization": "whitespace runs -> one space",
                "kl_boilerplate": ("#include and Python import lines are not compared (hessian_capture"
                                   ".py's KL_BOILERPLATE_LINE); no shingle spans one")}

    client = Client(args.server, args.timeout, args.retry_wait, log)
    try:
        health = client.request("GET", "/health", timeout=30.0)
        models = client.request("GET", "/v1/models", timeout=30.0)
    except (TransportFailed, RequestFailed) as e:
        log(f"[gen] ERROR: server {args.server} not usable: {e}")
        return 1
    server = {"url": args.server, "health": health, "models": models, "container": args.container,
              "dflash_container": args.dflash_container, "dflash_k": args.dflash_k,
              "cmdline": args.server_cmdline}

    todo = [e for e in order if e["id"] not in records
            and (args.only_category is None or e["category"] == args.only_category)]
    if args.limit > 0:
        todo = todo[:args.limit]
    runs = previous_runs(manifest_path, log)
    run = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "ended": None, "wall_s": 0.0, "planned": len(todo), "generated": 0, "rejected": 0,
           "failed": 0, "refused": 0, "completion_tokens": 0, "limit": args.limit or None,
           "only_category": args.only_category, "prompts_sha256": prompts_sha, "server": server,
           "exit": None}
    runs.append(run)
    log(f"[gen] {len(records)} of {len(entries)} samples already in {out}; this run: {len(todo)} "
        f"(server {args.server}, model {health.get('model')})")

    t_run = time.monotonic()
    gen_tok = 0
    gen_budget = 0
    consecutive = 0
    code = 0
    try:
        with open(out, "ab") as f:
            for k, e in enumerate(todo, 1):
                t_sample = time.monotonic()
                try:
                    rec = generate_sample(e, client, kl_shingles)
                except (TransportFailed, RequestFailed) as err:
                    # Not written either way, so a later run retries it. A refusal (4xx) can be
                    # this entry's alone (a prompt past --max-ctx, a template render error): the
                    # run moves on, and only refusals in a row stop it (exit 2, the requests are
                    # wrong). After a transport failure a dead /health stops it at once.
                    refused = isinstance(err, RequestFailed)
                    run["failed"] += 1
                    run["refused"] += refused
                    consecutive += 1
                    log(f"[gen] {k}/{len(todo)} {e['id']} {'REFUSED' if refused else 'FAILED'} (not "
                        f"written; a later run retries it): {err}")
                    if not refused and not client.healthy():
                        log("[gen] ABORT: /health does not answer")
                        code = 1
                        break
                    if consecutive >= args.max_consecutive_failures:
                        log(f"[gen] ABORT: {consecutive} consecutive failed samples")
                        code = 2 if refused else 1
                        break
                    continue
                consecutive = 0
                append_record(f, rec)
                records[rec["id"]] = rec
                run["generated"] += 1
                run["rejected"] += rec["rejected"]
                run["completion_tokens"] += rec["completion_tokens"]
                gen_tok += rec["completion_tokens"]
                gen_budget += e["max_tokens"] * turn_count(e)
                wall = time.monotonic() - t_run
                rate = gen_tok / wall if wall > 0 else 0.0
                fill = gen_tok / gen_budget if gen_budget else 0.75
                left = sum(x["max_tokens"] * turn_count(x) for x in todo[k:]) * fill
                eta = fmt_duration(left / rate) if rate > 0 else "?"
                dn = sum(t.get("draft_n", 0) for t in rec["timings"])
                da = sum(t.get("draft_n_accepted", 0) for t in rec["timings"])
                acc = f" acc {100.0 * da / dn:.0f}%" if dn else ""
                el = time.monotonic() - t_sample
                per = f"{rec['completion_tokens'] / el:5.1f}" if el >= 0.01 else "    -"
                status = "REJECT" if rec["rejected"] else "ok"
                log(f"[gen] {k}/{len(todo)} {rec['id']:<18} {status:<6} {rec['completion_tokens']:>5} tok "
                    f"{rec['finish_reason'] or '?':<6} {el:6.1f} s {per} tok/s{acc} | run {gen_tok:,} tok "
                    f"{rate:.1f} tok/s | ETA {eta}"
                    + (f" | {rec['reject_reason']}" if rec["rejected"] else ""))
    except KeyboardInterrupt:
        log("[gen] interrupted; every finished sample is on disk")
        code = 130
    except ResponseInvalid as err:
        log(f"[gen] ABORT: {err}")
        code = 2
    finally:
        run["ended"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        run["wall_s"] = round(time.monotonic() - t_run, 1)
        run["exit"] = {0: "ok", 1: "aborted: transport", 2: "aborted: request",
                       130: "interrupted"}.get(code, str(code))
        manifest = build_manifest(args, entries, prompts_sha, records, runs, server, disjoint)
        write_json_atomic(Path(manifest_path), manifest)
        c = manifest["counts"]
        log(f"[gen] {c['done']}/{c['planned']} samples on disk ({c['accepted']} accepted, "
            f"{c['rejected']} rejected), {c['tokens']['completion']:,} completion tokens; "
            f"this run {run['generated']} generated, {run['failed']} failed "
            f"({run['refused']} refused), {run['wall_s']:.0f} s; "
            f"manifest {manifest_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
