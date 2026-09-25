"""tools/quant2/test_gen_corpus.py

Offline test of gen_corpus.py (corpus v2's self-generated text) against a STUB r4dx-server: an
http.server on a loopback port in a thread, answering /health, /v1/models and
/v1/chat/completions with canned, seed-determined responses (reasoning_content, usage and timings
included) and with scripted faults. No GPU, no real server, no network beyond 127.0.0.1. The
generator is driven in-process through `gen_corpus.main`. Stdlib only, except (k), which needs the
reader's numpy and, for its render half, transformers and the checkpoint's tokenizer; each half is
SKIPPED with a message when what it needs is missing.

    (a) the contract: every field and type on every line; raw `text` is the final content stripped,
        chat `text` is null; usage summed over turns; the request carries the checkpoint's sampling
        for the thinking mode (thinking on = its generation_config.json), enable_thinking,
        max_tokens, stream false and no reasoning_effort
    (b) two-turn chats: the second request replays turn 1 (with its reasoning_content) plus the
        follow-up; the record holds both turns; a turn 1 cut inside its thought sends no second
        request
    (c) resume: --limit, then a torn last line, then a full run -> the torn fragment is dropped
        and regenerated, finished ids are never requested again, every id exactly once
    (d) rejection, kept in the file: degenerate repetition, empty output, an empty answer after a
        closed thought (a thought cut by max_tokens is KEPT, truncated_thought_turn set),
        a too-short document, reasoning_content with thinking off, a think tag in a raw document,
        text overlapping a (synthetic) KL corpus; a line rejected BY HAND without a reason is
        warned about and counted as "unspecified", and the run still writes its manifest
    (e) seeds: sample_seed is pinned, per id and turn, and what the stub received; two fresh runs
        write identical records (up to elapsed_s/timings)
    (f) transport: one retry after a dropped connection; abort after --max-consecutive-failures
        failed samples, at once when /health dies; HTTP 400 is never retried, the run moves past a
        refused sample and aborts (exit 2) only on refusals in a row; a failed sample is never
        written
    (g) refusals before any request: a prompt entry edited after its line was written, a line
        sampled with other SAMPLING, a malformed middle line, a --kl-dir without text; prompt-file
        validation (duplicate id, thinking on a raw entry, an assistant turn in a prompt, a budget
        the served context cannot hold, overlap with a synthetic KL corpus); the KL check skips a
        (synthetic) KL file's #include/import lines like the capture's gate: a generated file
        sharing only that header is kept, one sharing its code is not
    (h) the real corpus_v2_prompts.json: validates (KL guard against the real kl_corpus/ included),
        per-category counts, thinking / two-turn counts, every thinking chat at the largest budget
        the context allows; its table is printed
    (i) the detectors: tandem_repeat and unique_ngram_ratio edge cases, no false positive on this
        tool's own source
    (j) gen_manifest.json: counts, reasons, tokens, sha256s, provenance, run history
    (k) the contract's actual reader: the generated file through tools/reference/hessian_capture.py's
        load_gen_samples (used/rejected split = the generator's), its disjointness gate (the
        generator's KL reject would have been refused there; the kept samples pass; the include/
        import grammar and every real kl_corpus file's runs are the gate's), and, with the
        checkpoint's tokenizer, render_gen_sample + check_rendered_chat on every kept chat

Plain script, no pytest dependency, like tests/reference/test_hessian_rms.py:

    C:\\Users\\pay20\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\quant2\\test_gen_corpus.py

Exits 0 and prints "OK (<n> checks[, <k> skipped])" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# (k)'s tokenizer, if it loads, stays on one thread (the machine may be benchmarking).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_corpus as gc  # noqa: E402

TOOLS_REFERENCE_DIR = HERE.parent / "reference"
#: The checkpoint (tools/reference/common.py's DEFAULT_MODEL_DIR; not imported, it imports torch).
MODEL_DIR = Path(r"C:\AI\models\Qwen3.8-27B")

CHECKS = 0
FAILURES: list[str] = []
SKIPS: list[str] = []


def skip(label: str, why: str) -> None:
    SKIPS.append(label)
    print(f"SKIP {label} ({why})")


def check(cond, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(label)
        print(f"FAIL {label}")


def raises(exc_type, needle: str, fn, label: str) -> None:
    try:
        fn()
    except exc_type as e:
        check(needle in str(e), f"{label} (message {str(e)[:200]!r} lacks {needle!r})")
        return
    check(False, f"{label} (did not raise {exc_type.__name__})")


# --------------------------------------------------------------------------------------------
# The stub server
# --------------------------------------------------------------------------------------------

WORDS = ("river lantern harvest copper meadow signal quiet orbit ledger canvas thunder pepper "
         "window marble engine cotton garden silver bridge candle forest rocket pillow basket mirror "
         "harbor violet saddle kettle museum planet shadow ticket valley walnut yellow zebra anchor "
         "breeze cabin desert falcon glacier helmet island jungle kitten lemon magnet noodle oyster "
         "parrot quartz ribbon spider tunnel umbrella vessel whistle").split()


#: The text of the SYNTHETIC kl dir the generation runs use; the stub echoes it for [[kl-echo]]. (No
#: text of the real tools/reference/kl_corpus/ is ever put into a fixture.)
GEN_KL_SECRET = ("Held-out stand-in: the ferry timetable changes twice a year, and nobody reads the "
                 "small print about holiday sailings until the morning they miss one.")


def prose(seed: int, n: int) -> str:
    rng = random.Random(seed)
    return "เอกสารทดสอบภาษาไทย " + " ".join(rng.choice(WORDS) for _ in range(n)) + "."


class Stub:
    """A scripted r4dx-server. Markers in the LAST user message pick the behaviour:
    [[repeat]] [[empty]] [[short]] [[unclosed]] [[leak-reasoning]] [[think-tag]] [[kl-echo]] -> a
    reply that must be rejected;
    [[drop-once]] -> the first request is dropped without a response; [[always-500]] -> HTTP 500;
    [[bad-400]] -> HTTP 400; [[kill-health]] -> HTTP 500 and /health stops answering "ok"."""

    def __init__(self):
        self.lock = threading.Lock()
        self.posts: list[dict] = []
        self.attempts: dict[str, int] = {}
        self.health_ok = True
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the test output clean
                pass

            def _send(self, code: int, obj) -> None:
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200 if stub.health_ok else 503,
                               {"status": "ok" if stub.health_ok else "down", "model": "stub-model"})
                elif self.path == "/v1/models":
                    self._send(200, {"object": "list",
                                     "data": [{"id": "stub-model", "context_length": 8192}]})
                else:
                    self._send(404, {"error": {"message": "no route"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                status, reply = stub.answer(body)
                if status is None:
                    self.close_connection = True  # dropped: no status line at all
                    return
                self._send(status, reply)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def answer(self, body: dict):
        msgs = body["messages"]
        user = [m for m in msgs if m["role"] == "user"][-1]["content"]
        with self.lock:
            self.posts.append(copy.deepcopy(body))
            self.attempts[user] = self.attempts.get(user, 0) + 1
            attempt = self.attempts[user]
        if "[[drop-once]]" in user and attempt == 1:
            return None, None
        if "[[always-500]]" in user:
            return 500, {"error": {"message": "stub failure"}}
        if "[[kill-health]]" in user:
            self.health_ok = False
            return 500, {"error": {"message": "stub is going down"}}
        if "[[bad-400]]" in user:
            return 400, {"error": {"message": "stub rejects this request"}}
        seed = body["seed"]
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        finish = "stop"
        content = prose(seed, 90)
        reasoning = ("Let me think this through. " + prose(seed + 1, 40)) if thinking else None
        if "[[repeat]]" in user:
            content = "An opening line. " + "The same sentence comes back again and again here. " * 10
        elif "[[empty]]" in user:
            content = "  \n"
        elif "[[short]]" in user:
            content = "Too short."
        elif "[[unclosed]]" in user:
            content, reasoning, finish = "", prose(seed + 2, 150), "length"
        elif "[[leak-reasoning]]" in user:
            reasoning = "a stray thought"
        elif "[[think-tag]]" in user:
            content = prose(seed, 60) + "\n</think>\n" + prose(seed + 3, 30)
        elif "[[kl-echo]]" in user:
            content = prose(seed, 40) + "\n\n" + GEN_KL_SECRET + "\n\n" + prose(seed + 4, 20)
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        completion = 40 + len(msgs)
        return 200, {
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub-model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10 * len(msgs), "completion_tokens": completion,
                      "total_tokens": 10 * len(msgs) + completion},
            "timings": {"prompt_n": 7, "prompt_ms": 1.5, "prompt_per_second": 4666.7,
                        "predicted_n": completion, "predicted_ms": 100.0,
                        "predicted_per_second": completion * 10.0, "draft_n": 8, "draft_n_accepted": 5},
        }

    def posts_for(self, first_user: str) -> list[dict]:
        with self.lock:
            return [p for p in self.posts
                    if [m for m in p["messages"] if m["role"] == "user"][0]["content"] == first_user]


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


def entry(sid, fmt, thinking, user, system="Test system prompt.", followups=None, max_tokens=256):
    cat = sid.split("/")[0]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    e = {"id": sid, "category": cat, "format": fmt, "enable_thinking": thinking,
         "max_tokens": max_tokens, "messages": msgs}
    if followups:
        e["followups"] = followups
    return e


def prompt_doc(entries: list[dict]) -> dict:
    return {"format": gc.PROMPTS_FORMAT, "version": gc.PROMPTS_VERSION, "samples": entries}


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


MAIN_ENTRIES = [
    entry("thai_prose/001", "raw", False, "เขียนเรื่องสั้นสำหรับการทดสอบ [[ok]]"),
    entry("thai_prose/002", "raw", False, "เขียนบทความสำหรับการทดสอบ [[kl-echo]]"),
    entry("english_prose/001", "raw", False, "Write a test essay. [[drop-once]]"),
    entry("english_prose/002", "raw", False, "Write a test note. [[short]]"),
    entry("english_prose/003", "raw", False, "Write a test page. [[think-tag]]"),
    entry("chat/001", "chat", True, "A thinking question for the stub.",
          followups=["A follow-up question for the stub."], max_tokens=512),
    entry("chat/002", "chat", False, "A plain question for the stub.", system=None,
          followups=["And a plain follow-up."]),
    entry("chat/003", "chat", True, "A question whose thinking never ends. [[unclosed]]",
          followups=["This follow-up must never be sent."]),
    entry("chat/004", "chat", False, "One more plain question."),
    entry("code/001", "raw", False, "Write a looping program. [[repeat]]"),
    entry("code/002", "raw", False, "Write a program. [[leak-reasoning]]"),
    entry("multilingual/001", "raw", False, "Escribe algo. [[empty]]"),
]
#: processing_order's rule by hand: key (i + 0.5) / n per category, ties by contract category order.
EXPECTED_ORDER = ["chat/001", "english_prose/001", "thai_prose/001", "code/001", "chat/002",
                  "english_prose/002", "multilingual/001", "chat/003", "thai_prose/002", "code/002",
                  "english_prose/003", "chat/004"]
EXPECTED_REJECT = {"english_prose/002": "too short",
                   "code/001": "degenerate content", "code/002": "unexpected reasoning_content",
                   "multilingual/001": "empty output", "english_prose/003": "think tag in a raw document",
                   "thai_prose/002": "kl overlap"}
CONTRACT_TYPES = {"id": str, "category": str, "format": str, "enable_thinking": bool, "messages": list,
                  "finish_reason": str, "prompt_tokens": int, "completion_tokens": int, "seed": int,
                  "temperature": float, "top_p": float, "rejected": bool}


def run_main(args: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = gc.main(args)
    return code, buf.getvalue()


def read_lines(path: Path) -> list[dict]:
    # "\n" only, as hessian_capture.py reads it: str.splitlines() would also cut at U+2028 & co.,
    # which json.dumps(ensure_ascii=False) leaves unescaped.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def strip_volatile(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in gc.VOLATILE_FIELDS}


# --------------------------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------------------------


def test_detectors() -> None:
    unit = "The pump stops, the valve closes, then it starts. "  # 50 chars
    # The run starts at char 5, not 7: the unit ends in ". ", so its rotation ". The pump ..." repeats
    # from the "." of "Intro." on.
    check(gc.tandem_repeat("Intro. " + unit * 4 + "Outro.") == (50, 4, 5), "(i) 4 copies found")
    check(gc.tandem_repeat("Intro. " + unit * 3 + "Outro.") is None, "(i) 3 copies are not a run")
    check(gc.tandem_repeat("x" + unit * 9)[1] == 9, "(i) copies extend to the end of the run")
    th = "ฝนตกหนักตั้งแต่เช้า ถนนหน้าบ้านมีน้ำท่วมขังจนรถผ่านไม่ได้ "
    check(gc.tandem_repeat(th * 5) is not None, "(i) a Thai unit repeated 5x")
    check(gc.tandem_repeat("ab" * 200)[0] >= gc.TANDEM_MIN_PERIOD, "(i) short-period loop via a multiple")
    near = ("\u0e01" + "x" * 39) * 3 + "\u0f01" + "x" * 39
    check(gc.tandem_repeat(near) is None, "(i) 3 copies + a char differing only in a high byte")
    counter = "".join(f"Step {i}: open the valve and wait for the pressure to settle.\n" for i in range(60))
    check(gc.unique_ngram_ratio(counter) < gc.NGRAM_MIN_UNIQUE, "(i) counter loop has a low n-gram ratio")
    check(f"unique {gc.NGRAM_N}-gram ratio" in (gc.degenerate_reason(counter) or ""),
          "(i) degenerate_reason names the ratio")
    check(gc.unique_ngram_ratio("short text") == 1.0, "(i) short text: ratio 1.0")
    src = Path(gc.__file__).read_text(encoding="utf-8")
    fp = [i for i in range(0, len(src), 6000) if gc.degenerate_reason(src[i:i + 6000])]
    check(not fp, f"(i) no false positive on gen_corpus.py's own source (chunks {fp})")


def test_sampling() -> None:
    check(gc.SAMPLING[False] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
          and gc.SAMPLING[True] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
          f"(a) SAMPLING pinned to the checkpoint card's values ({gc.SAMPLING})")
    cfg = MODEL_DIR / "generation_config.json"
    if not cfg.is_file():
        skip("(a) thinking-on SAMPLING vs generation_config.json", f"no {cfg}")
        return
    g = json.loads(cfg.read_text(encoding="utf-8"))
    check(all(gc.SAMPLING[True][k] == g.get(k) for k in ("temperature", "top_p", "top_k")),
          f"(a) thinking-on SAMPLING is the checkpoint's generation_config.json ({g})")


def test_seeds() -> None:
    s = gc.sample_seed("thai_prose/001", 0)
    check(s == 236546503, f"(e) sample_seed pinned (got {s})")
    check(gc.sample_seed("thai_prose/001", 0) == s, "(e) sample_seed is stable")
    seeds = {gc.sample_seed(f"chat/{i:03d}", t) for i in range(1, 81) for t in (0, 1)}
    check(len(seeds) == 160, "(e) seeds differ across ids and turns")
    check(all(0 <= x < 2 ** 31 for x in seeds), "(e) seeds are non-negative 31-bit")


def test_prompt_validation(tmp: Path) -> None:
    ok = [entry("thai_prose/001", "raw", False, "เขียนบทความสั้น ๆ"),
          entry("chat/001", "chat", True, "Question?", followups=["Next?"])]
    gc.load_prompts(write_json(tmp / "ok.json", prompt_doc(ok)), None)
    check(True, "(g) a valid prompt file loads")
    dup = ok + [copy.deepcopy(ok[0])]
    raises(gc.PromptFileError, "duplicate id",
           lambda: gc.load_prompts(write_json(tmp / "dup.json", prompt_doc(dup)), None), "(g) duplicate id")
    bad = copy.deepcopy(ok)
    bad[0]["enable_thinking"] = True
    raises(gc.PromptFileError, "thinking off",
           lambda: gc.load_prompts(write_json(tmp / "th.json", prompt_doc(bad)), None),
           "(g) thinking on a raw entry")
    bad = copy.deepcopy(ok)
    bad[1]["messages"].append({"role": "assistant", "content": "canned"})
    raises(gc.PromptFileError, "role 'assistant'",
           lambda: gc.load_prompts(write_json(tmp / "as.json", prompt_doc(bad)), None),
           "(g) an assistant turn in a prompt")
    bad = copy.deepcopy(ok)
    bad[0]["id"] = "code/001"
    raises(gc.PromptFileError, "does not start with its category",
           lambda: gc.load_prompts(write_json(tmp / "id.json", prompt_doc(bad)), None),
           "(g) id/category mismatch")
    # The context budget: turns x max_tokens + the prompt's reserve must fit the served --max-ctx.
    one = (gc.SERVED_MAX_CTX - gc.PROMPT_RESERVE_TOKENS)
    fit = copy.deepcopy(ok)
    fit[0]["max_tokens"], fit[1]["max_tokens"] = one, one // 2
    gc.load_prompts(write_json(tmp / "fit.json", prompt_doc(fit)), None)
    check(True, f"(g) the largest budgets load: {one} for one turn, {one // 2} per turn for two")
    for i, mt in ((0, one + 1), (1, one // 2 + 1), (0, gc.MAX_TOKENS_MIN - 1)):
        over = copy.deepcopy(fit)
        over[i]["max_tokens"] = mt
        raises(gc.PromptFileError, "must fit the served --max-ctx",
               lambda: gc.load_prompts(write_json(tmp / "over.json", prompt_doc(over)), None),
               f"(g) max_tokens {mt} for {gc.turn_count(over[i])} turn(s) refused")
    # A --kl-dir without text would make the check vacuous: refused, like hessian_capture's kl_index.
    empty_kl = tmp / "empty_kl"
    empty_kl.mkdir()
    (empty_kl / "notes.md").write_text("not a .txt file\n", encoding="utf-8")
    for d in (tmp / "no_such_kl", empty_kl):
        raises(gc.KlCorpusError, "pass vacuously", lambda: gc.load_prompts(tmp / "ok.json", d),
               f"(g) --kl-dir {d.name} without *.txt text refused")
    # The KL guard, against a SYNTHETIC kl dir: no text of the real kl_corpus/ is ever put in a fixture.
    kl = tmp / "fake_kl"
    kl.mkdir()
    secret = "The held-out passage says that lighthouses were painted in bands so ships could tell them apart."
    (kl / "english_prose.txt").write_text("Preamble.\n" + secret + "\nMore.\n", encoding="utf-8")
    leak = copy.deepcopy(ok)
    leak[1]["followups"] = ["Summarize:\n" + secret.replace(" so ", "  so\n")]
    raises(gc.PromptFileError, "shares",
           lambda: gc.load_prompts(write_json(tmp / "kl.json", prompt_doc(leak)), kl),
           "(g) KL overlap refused (whitespace-insensitive)")
    gc.load_prompts(tmp / "ok.json", kl)
    check(True, "(g) no overlap -> loads with the KL guard on")
    # A KL file's #include / import lines are not compared, as in hessian_capture.py's gate: a
    # generated file sharing only its standard header is kept, one sharing its code is not.
    # (Synthetic stand-ins for the real kl_corpus/ code excerpts; whole-file shingling rejects both.)
    code_kl = tmp / "code_kl"
    code_kl.mkdir()
    cpp_head = "#include <algorithm>\n#include <chrono>\n#include <cstdint>\n#include <cstdio>\n"
    cpp_body = ("static int held_out_checksum(const unsigned char* p, int n) {\n"
                "    int s = 0;\n    while (n--) s = s * 31 + *p++;\n    return s;\n}\n")
    py_head = ("from __future__ import annotations\n\nimport argparse\nimport json\n"
               "from pathlib import (\n    Path,\n    PurePath,\n)\n")
    py_body = "def held_out_order(values):\n    return sorted(set(values), key=lambda v: (len(str(v)), str(v)))\n"
    (code_kl / "cpp_source.txt").write_text("// Held-out excerpt.\n" + cpp_head + "\n" + cpp_body, encoding="utf-8")
    (code_kl / "python_source.txt").write_text('"""Held-out module."""\n\n' + py_head + "\n\n" + py_body,
                                               encoding="utf-8")
    sh, _ = gc.kl_shingle_index(code_kl)
    for name, head, body in (("C++", cpp_head, cpp_body), ("Python", py_head, py_body)):
        mine = head + "\n\nint main_or_def_main_of_a_generated_file_that_shares_only_the_header;\n"
        check(gc.first_kl_overlap(mine, sh) is None, f"(g) a generated {name} file sharing only the KL "
                                                     f"file's include/import lines is kept")
        check(gc.first_kl_overlap(mine + body, sh) is not None,
              f"(g) ... a generated {name} file sharing its code is not")
    check(gc.kl_segments("a\n#include <x>\nb\nfrom q import (\n  r,\n)\nc") == ["a", "b", "c"]
          and gc.kl_segments(" one\n\n two ") == ["one two"],
          "(g) kl_segments: runs between include/import lines (a `from q import (` block to its `)`), "
          "whitespace collapsed")


def test_real_prompt_file() -> None:
    entries, sha = gc.load_prompts(gc.DEFAULT_PROMPTS, gc.KL_CORPUS_DIR)
    check(gc.KL_CORPUS_DIR.is_dir(), "(h) the real kl_corpus/ exists, so the guard really ran")
    counts = {c: sum(e["category"] == c for e in entries) for c in gc.CATEGORIES}
    want = {"thai_prose": 140, "english_prose": 90, "chat": 80, "code": 50, "multilingual": 30}
    check(counts == want, f"(h) per-category counts {counts}")
    chat = [e for e in entries if e["category"] == "chat"]
    check(sum(e["enable_thinking"] for e in chat) == 40, "(h) 40 of 80 chats think")
    check(sum("followups" in e for e in chat) == 25, "(h) 25 two-turn chats")
    room = gc.SERVED_MAX_CTX - gc.PROMPT_RESERVE_TOKENS
    short = [e["id"] for e in chat if e["enable_thinking"] and e["max_tokens"] != room // gc.turn_count(e)]
    check(not short, f"(h) every thinking chat gets the largest budget the context allows (not: {short})")
    for c in gc.CATEGORIES:
        ids = sorted(e["id"] for e in entries if e["category"] == c)
        check(ids == [f"{c}/{i:03d}" for i in range(1, len(ids) + 1)], f"(h) {c} ids are contiguous")
    th = [e for e in entries if e["category"] == "thai_prose"]
    check(all(any("\u0e00" <= ch <= "\u0e7f" for ch in e["messages"][-1]["content"]) for e in th),
          "(h) every thai_prose instruction is written in Thai")
    first = gc.processing_order(entries)[:20]
    check({e["category"] for e in first} == set(gc.CATEGORIES), "(h) the first 20 cover every category")
    print("\n".join(gc.prompt_table(entries)))
    print(f"prompt file sha256 {sha}")


def check_with_reader(out: Path, kl: Path) -> None:
    """(k): the generated `out` through the contract's actual reader, tools/reference/
    hessian_capture.py, with `kl` (the synthetic kl dir of the generation runs) as its KL corpus.
    Without numpy the whole check is skipped; without transformers or the checkpoint's tokenizer,
    the render half is."""
    if str(TOOLS_REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_REFERENCE_DIR))
    try:
        import hessian_capture as hc
    except ImportError as e:
        skip("(k) the hessian_capture.py cross-check", f"it does not import here: {e}")
        return
    # The generator's KL runs are the capture's: the same grammar (copied, gen_corpus.py is stdlib
    # only), and the same runs of every real kl_corpus file (read here at runtime, in memory only).
    check(gc.KL_BOILERPLATE_LINE.pattern == hc.KL_BOILERPLATE_LINE.pattern
          and gc.KL_IMPORT_CONTINUATION.pattern == hc.KL_IMPORT_CONTINUATION.pattern,
          "(k) gen_corpus's include/import grammar is hessian_capture's, verbatim")
    real = sorted(gc.KL_CORPUS_DIR.glob("*.txt"))
    differ = [p.name for p in real if gc.kl_segments(p.read_text(encoding="utf-8"))
              != [t for t, _ in hc.kl_gate_segments(p.read_text(encoding="utf-8"))[0]]]
    check(len(real) >= 4 and not differ,
          f"(k) kl_segments == kl_gate_segments' runs on every real kl_corpus file (differ: {differ})")
    recs = {r["id"]: r for r in read_lines(out)}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            used, rejected, sha = hc.load_gen_samples(out)
    except SystemExit as e:
        check(False, f"(k) load_gen_samples accepts the generated file: {e}\n{buf.getvalue()}")
        return
    check([s["id"] for s in rejected] == sorted(EXPECTED_REJECT)
          and [s["id"] for s in used] == sorted(set(recs) - set(EXPECTED_REJECT)),
          f"(k) load_gen_samples' used/rejected split is the generator's "
          f"(used {[s['id'] for s in used]}, rejected {[s['id'] for s in rejected]})")
    check(sha == hashlib.sha256(out.read_bytes()).hexdigest(), "(k) load_gen_samples' sha256 is the file's")
    kli = hc.kl_index(kl)
    check(bool(hc.find_kl_overlaps([("thai_prose/002", recs["thai_prose/002"]["text"])], kli.shingles)),
          "(k) the text the generator rejected as 'kl overlap' would trip the capture's gate")
    fields = [(s["id"], m[f]) for s in used for m in s["messages"]
              for f in ("content", "reasoning_content") if isinstance(m.get(f), str)]
    hits = hc.find_kl_overlaps(fields, kli.shingles)
    check(not hits, f"(k) no message field of a kept sample trips the capture's gate ({hits})")

    label = "(k) build_gen_corpus: render_gen_sample + check_rendered_chat + the gate"
    if not (MODEL_DIR / "tokenizer.json").is_file():
        skip(label, f"no tokenizer.json under {MODEL_DIR}")
        return
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        skip(label, f"no transformers: {e}")
        return
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _, sources = hc.build_gen_corpus(out, tok, 64, None, kli)
    except SystemExit as e:
        check(False, f"{label}: refused: {e}\n{buf.getvalue()}")
        return
    src = {s["name"]: s for s in sources}
    check(sorted(src) == sorted({r["category"] for r in recs.values()})
          and all(s["sample_ids"] == [u["id"] for u in used if u["category"] == c] for c, s in src.items())
          and sum(s["samples_rejected"] for s in src.values()) == len(EXPECTED_REJECT)
          and all((s["sequences"] > 0) == bool(s["sample_ids"]) for s in src.values()),
          f"{label}: every kept sample rendered and packed "
          f"({ {c: (s['sample_ids'], s['samples_rejected'], s['sequences']) for c, s in src.items()} })")
    check(src["chat"]["formats"] == {"raw": 0, "chat": 4} and src["chat"]["thinking_samples"] == 2,
          f"(k) the kept chats: chat/001 (thinking, two turns), chat/002 (two turns), chat/003 "
          f"(thinking, cut inside its thought), chat/004 "
          f"({src['chat']['formats']}, {src['chat']['thinking_samples']} thinking)")
    c1 = next(u for u in used if u["id"] == "chat/001")
    r1 = hc.render_gen_sample(c1, tok, tok.chat_template)
    check(r1.startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
          and r1.endswith(hc.assistant_turn_rendering(c1["messages"][-1]))
          and r1.count("<think>\n") == 2,
          "(k) chat/001 renders with the xhigh sentence and both thoughts, the final turn last")


def test_generation(tmp: Path, stub: Stub) -> None:
    prompts = write_json(tmp / "prompts.json", prompt_doc(MAIN_ENTRIES))
    by_id = {e["id"]: e for e in MAIN_ENTRIES}
    check([e["id"] for e in gc.processing_order(MAIN_ENTRIES)] == EXPECTED_ORDER, "(c) processing order")
    out = tmp / "run" / "samples.jsonl"
    manifest = out.with_name("gen_manifest.json")
    kl = tmp / "gen_kl"
    kl.mkdir()
    (kl / "english_prose.txt").write_text("Title\n\n" + GEN_KL_SECRET + "\n", encoding="utf-8")
    base = ["--prompts", str(prompts), "--server", stub.url, "--out", str(out), "--retry-wait", "0",
            "--kl-dir", str(kl), "--container", r"D:\fake\target.r4dx",
            "--server-cmdline", "r4dx-server --port 0"]

    # ---- run 1: --limit 3 ------------------------------------------------------------------
    code, log1 = run_main(base + ["--limit", "3"])
    check(code == 0, f"(c) run 1 exit 0 (got {code})\n{log1}")
    recs = read_lines(out)
    check([r["id"] for r in recs] == EXPECTED_ORDER[:3], "(c) --limit 3 wrote the first 3 in order")
    check(len(stub.posts_for(by_id["english_prose/001"]["messages"][-1]["content"])) == 2,
          "(f) a dropped connection is retried exactly once")
    check("retrying once" in log1, "(f) the retry is logged")

    # ---- a crash mid-write: a torn last line --------------------------------------------------
    with open(out, "ab") as f:
        f.write(b'{"id": "chat/002", "category": "chat", "format": "ch')
    n_before = len(stub.posts)

    # ---- run 2: everything else ------------------------------------------------------------
    code, log2 = run_main(base)
    check(code == 0, f"(c) run 2 exit 0 (got {code})\n{log2}")
    check("dropped a torn last line" in log2, "(c) the torn line is reported")
    raw_lines = out.read_bytes().split(b"\n")
    check(raw_lines[-1] == b"" and all(json.loads(x) for x in raw_lines[:-1]),
          "(c) the file is whole lines only, LF-terminated")
    recs = read_lines(out)
    ids = [r["id"] for r in recs]
    check(sorted(ids) == sorted(by_id) and len(ids) == len(set(ids)), "(c) every id exactly once")
    run2_first_users = {[m for m in p["messages"] if m["role"] == "user"][0]["content"]
                        for p in stub.posts[n_before:]}
    for sid in EXPECTED_ORDER[:3]:
        check(by_id[sid]["messages"][-1]["content"] not in run2_first_users,
              f"(c) {sid} was not requested again")
    check(len(stub.posts_for("A plain question for the stub.")) == 2,
          "(c) chat/002 (torn) regenerated: both turns sent once")
    rec = {r["id"]: r for r in recs}

    # ---- (a) the contract ---------------------------------------------------------------------
    for r in recs:
        e = by_id[r["id"]]
        for k, t in CONTRACT_TYPES.items():
            check(isinstance(r.get(k), t) and not (t is int and isinstance(r.get(k), bool)),
                  f"(a) {r['id']}: {k} is {t.__name__}")
        check("text" in r and "reject_reason" in r, f"(a) {r['id']}: text and reject_reason present")
        check(r["category"] == e["category"] and r["format"] == e["format"]
              and r["enable_thinking"] == e["enable_thinking"], f"(a) {r['id']}: identity fields")
        check(r["messages"][:len(e["messages"])] == e["messages"], f"(a) {r['id']}: prompt messages kept")
        check(r["messages"][-1]["role"] == "assistant", f"(a) {r['id']}: ends with the assistant turn")
        if e["format"] == "raw":
            check(r["text"] == r["messages"][-1]["content"].strip(), f"(a) {r['id']}: text = content stripped")
        else:
            check(r["text"] is None, f"(a) {r['id']}: chat text is null")
        posts = stub.posts_for(e["messages"][-1]["content"] if len(e["messages"]) == 1
                               else [m for m in e["messages"] if m["role"] == "user"][0]["content"])
        if r["id"] == "english_prose/001":
            posts = posts[1:]  # the dropped first attempt carries the same body
        want_prompt = sum(10 * len(p["messages"]) for p in posts[-len(r["turn_seeds"]):])
        want_compl = sum(40 + len(p["messages"]) for p in posts[-len(r["turn_seeds"]):])
        check(r["prompt_tokens"] == want_prompt and r["completion_tokens"] == want_compl,
              f"(a) {r['id']}: usage summed over turns ({r['prompt_tokens']}/{r['completion_tokens']})")
        samp = gc.SAMPLING[e["enable_thinking"]]
        check((r["temperature"], r["top_p"], r["top_k"]) == (samp["temperature"], samp["top_p"], samp["top_k"]),
              f"(a) {r['id']}: sampling recorded")
        for t, p in enumerate(posts[-len(r["turn_seeds"]):]):
            check(p["temperature"] == samp["temperature"] and p["top_p"] == samp["top_p"]
                  and p["top_k"] == 20 and p["min_p"] == 0.0 and p["stream"] is False
                  and p["max_tokens"] == e["max_tokens"]
                  and p["chat_template_kwargs"] == {"enable_thinking": e["enable_thinking"]}
                  and "reasoning_effort" not in p, f"(a) {r['id']} turn {t + 1}: request body")
            check(p["seed"] == gc.sample_seed(r["id"], t), f"(e) {r['id']} turn {t + 1}: seed sent")
        check(r["seed"] == gc.sample_seed(r["id"], 0)
              and r["turn_seeds"] == [gc.sample_seed(r["id"], t) for t in range(len(r["turn_seeds"]))],
              f"(e) {r['id']}: seeds recorded")
        check(r["prompt_sha256"] == gc.entry_sha256(e), f"(c) {r['id']}: prompt_sha256")

    # ---- (b) two-turn chats -----------------------------------------------------------------
    r1 = rec["chat/001"]
    check([m["role"] for m in r1["messages"]] == ["system", "user", "assistant", "user", "assistant"],
          "(b) chat/001 holds both turns")
    check(all(isinstance(m.get("reasoning_content"), str) and m["reasoning_content"]
              for m in r1["messages"] if m["role"] == "assistant"), "(b) thinking turns carry reasoning")
    p1, p2 = stub.posts_for("A thinking question for the stub.")
    check(p2["messages"][:3] == r1["messages"][:3] and p2["messages"][3]["content"]
          == "A follow-up question for the stub.", "(b) turn 2 replays turn 1 + the follow-up")
    check(p2["messages"][2]["reasoning_content"] == r1["messages"][2]["reasoning_content"],
          "(b) turn 2 replays turn 1's reasoning_content")
    check(r1["turn_finish_reasons"] == ["stop", "stop"] and len(r1["timings"]) == 2,
          "(b) per-turn finish reasons and timings")
    r2 = rec["chat/002"]
    check([m["role"] for m in r2["messages"]] == ["user", "assistant", "user", "assistant"]
          and all("reasoning_content" not in m for m in r2["messages"]),
          "(b) chat/002: no system, no reasoning_content with thinking off")
    check(len(stub.posts_for("A question whose thinking never ends. [[unclosed]]")) == 1
          and len(rec["chat/003"]["messages"]) == 3,
          "(b) a turn 1 cut inside its thought sends no follow-up")

    # ---- (d) rejection ------------------------------------------------------------------------
    for sid, r in rec.items():
        want = EXPECTED_REJECT.get(sid)
        if want:
            check(r["rejected"] and r["reject_reason"].startswith(want),
                  f"(d) {sid} rejected as {want!r} (got {r['reject_reason']!r})")
        else:
            check(not r["rejected"] and r["reject_reason"] is None, f"(d) {sid} kept ({r['reject_reason']})")
    check(rec["chat/003"]["finish_reason"] == "length" and rec["chat/003"]["truncated_thought_turn"] == 1
          and all(r["truncated_thought_turn"] is None for sid, r in rec.items() if sid != "chat/003"),
          "(d) a thought cut by max_tokens is KEPT: finish_reason length, truncated_thought_turn 1")
    for fin, want in (("length", None), ("stop", "empty answer")):
        got = gc.check_turn("chat", True, 0, "", "a thought", True, fin)
        check(got == want or bool(want and got and got.startswith(want)),
              f"(d) empty answer after a thought, finish {fin!r} -> {want!r} (got {got!r})")

    # ---- (j) the manifest ---------------------------------------------------------------------
    m = json.loads(manifest.read_text(encoding="utf-8"))
    c = m["counts"]
    check((c["planned"], c["done"], c["accepted"], c["rejected"], c["missing"], c["complete"])
          == (12, 12, 6, 6, 0, True), f"(j) counts {c}")
    check(c["reject_reasons"] == {k: 1 for k in EXPECTED_REJECT.values()}, f"(j) reasons {c['reject_reasons']}")
    check(c["tokens"]["completion"] == sum(r["completion_tokens"] for r in recs)
          and c["tokens"]["prompt"] == sum(r["prompt_tokens"] for r in recs), "(j) token totals")
    check(sum(v["completion_tokens"] for v in c["per_category"].values()) == c["tokens"]["completion"],
          "(j) per-category tokens add up")
    check(c["per_category"]["chat"]["completion_tokens_accepted"]
          == sum(r["completion_tokens"] for r in recs if r["category"] == "chat" and not r["rejected"]),
          "(j) accepted tokens per category")
    check(c["speculation"]["draft_n"] == 8 * sum(len(r["timings"]) for r in recs), "(j) draft totals")
    check(m["prompts"]["sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest(), "(j) prompt sha256")
    check(m["samples"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest(), "(j) samples sha256")
    check(m["server"]["container"] == r"D:\fake\target.r4dx" and m["server"]["health"]["status"] == "ok"
          and m["server"]["models"]["data"][0]["id"] == "stub-model", "(j) server provenance")
    check([r["exit"] for r in m["runs"]] == ["ok", "ok"] and [r["generated"] for r in m["runs"]] == [3, 9],
          f"(j) run history {[(r['exit'], r['generated']) for r in m['runs']]}")
    kd = m["rejection"]["kl_disjointness"]
    check(kd["window_chars"] == gc.KL_WINDOW and kd["shingles"] > 0 and list(kd["kl_files"]) == ["english_prose.txt"],
          f"(j) KL disjointness provenance {kd}")
    check("kl_corpus/english_prose.txt" in rec["thai_prose/002"]["reject_reason"],
          "(d) the kl overlap names the kl file")

    # ---- (k) the contract's actual reader --------------------------------------------------------
    check_with_reader(out, kl)

    # ---- (e) determinism: a fresh run writes the same records --------------------------------
    out2 = tmp / "run2" / "samples.jsonl"
    code, log3 = run_main([a if a != str(out) else str(out2) for a in base])
    check(code == 0, f"(e) fresh run exit 0\n{log3}")
    a = {r["id"]: strip_volatile(r) for r in read_lines(out)}
    b = {r["id"]: strip_volatile(r) for r in read_lines(out2)}
    check(a == b, "(e) two fresh runs write identical records (up to elapsed_s/timings)")

    # ---- a finished corpus: nothing to do --------------------------------------------------
    n_before = len(stub.posts)
    code, _ = run_main(base)
    check(code == 0 and len(stub.posts) == n_before, "(c) a complete corpus sends no request")

    # ---- (g) refusals before any request ------------------------------------------------------
    edited = copy.deepcopy(MAIN_ENTRIES)
    next(e for e in edited if e["id"] == "chat/001")["messages"][-1]["content"] += " (edited)"
    p_edit = write_json(tmp / "prompts_edited.json", prompt_doc(edited))
    before = out.read_bytes()
    n_before = len(stub.posts)
    code, log4 = run_main([a if a != str(prompts) else str(p_edit) for a in base])
    check(code == 2 and "chat/001: its prompt entry changed" in log4, f"(g) edited prompt refused\n{log4}")
    check(out.read_bytes() == before and len(stub.posts) == n_before, "(g) ... before any request or write")
    out3 = tmp / "run3" / "samples.jsonl"
    out3.parent.mkdir()
    lines = out.read_bytes().split(b"\n")
    lines[1] = lines[1][:40]
    out3.write_bytes(b"\n".join(lines))
    code, log5 = run_main([a if a != str(out) else str(out3) for a in base])
    check(code == 2 and "line 2 is not a sample" in log5, f"(g) a malformed middle line is refused\n{log5}")

    # A sampler change after lines were written: refused, never mixed into one corpus.
    saved = copy.deepcopy(gc.SAMPLING)
    gc.SAMPLING[True]["temperature"] = 0.6
    n_before = len(stub.posts)
    try:
        code, log6 = run_main(base)
    finally:
        gc.SAMPLING.clear()
        gc.SAMPLING.update(saved)
    check(code == 2 and "chat/001: sampled with" in log6 and "chat/003: sampled with" in log6
          and "chat/002: sampled" not in log6 and out.read_bytes() == before and len(stub.posts) == n_before,
          f"(g) lines sampled with other SAMPLING refused, the thinking ones only, before any request\n{log6}")

    # Lines rejected by hand (the fix for a capture-side KL refusal): with a reason, and without one.
    # The run must still finish and write its manifest; the reasonless line counts as "unspecified".
    out5 = tmp / "run5" / "samples.jsonl"
    out5.parent.mkdir()
    lines = read_lines(out)
    for r in lines:
        if r["id"] == "chat/004":
            r["rejected"] = True  # reject_reason stays null, as on every accepted line
        elif r["id"] == "english_prose/001":
            r["rejected"], r["reject_reason"] = True, "kl overlap: named by hessian_capture.py"
    out5.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode("utf-8"))
    n_before = len(stub.posts)
    code, log7 = run_main([a if a != str(out) else str(out5) for a in base])
    check(code == 0 and len(stub.posts) == n_before, f"(d) hand-rejected lines: exit 0, nothing re-requested\n{log7}")
    check("1 line(s) rejected without a reject_reason" in log7 and "chat/004" in log7,
          "(d) the reasonless hand-rejected line is warned about")
    m5 = json.loads(out5.with_name("gen_manifest.json").read_text(encoding="utf-8"))
    rr = m5["counts"]["reject_reasons"]
    check(m5["counts"]["rejected"] == 8 and rr.get(gc.UNSPECIFIED_REJECT) == 1 and rr.get("kl overlap") == 2
          and m5["runs"][-1]["exit"] == "ok",
          f"(d) the manifest counts them: {m5['counts']['rejected']} rejected, reasons {rr}")
    lines[0]["completion_tokens"] = str(lines[0]["completion_tokens"])
    out5.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines).encode("utf-8"))
    code, log8 = run_main([a if a != str(out) else str(out5) for a in base])
    check(code == 2 and f"{lines[0]['id']}: completion_tokens missing or not int" in log8,
          f"(g) a hand-edited line with a mistyped field is refused, not a manifest crash\n{log8}")

    # ---- read_existing: a last line missing only its newline --------------------------------
    out4 = tmp / "nl.jsonl"
    out4.write_bytes(out.read_bytes().rstrip(b"\n"))
    got = gc.read_existing(out4, lambda _m: None)
    check(len(got) == 12 and out4.read_bytes().endswith(b"}\n"), "(c) a missing final newline is repaired")


def test_failures(tmp: Path, stub: Stub) -> None:
    kl = tmp / "fail_kl"
    kl.mkdir()
    (kl / "english_prose.txt").write_text(GEN_KL_SECRET + "\n", encoding="utf-8")

    def run(entries: list[dict], name: str, *extra: str) -> tuple[int, str, int, Path]:
        p = write_json(tmp / f"{name}.json", prompt_doc(entries))
        out = tmp / name / "samples.jsonl"
        n_before = len(stub.posts)
        code, log = run_main(["--prompts", str(p), "--server", stub.url, "--out", str(out),
                              "--retry-wait", "0", "--kl-dir", str(kl), *extra])
        return code, log, len(stub.posts) - n_before, out

    fail = [entry(f"english_prose/{i:03d}", "raw", False, f"Essay {i}. [[always-500]]") for i in (1, 2, 3)]
    code, log, posts, out = run(fail, "fail", "--max-consecutive-failures", "2")
    check(code == 1, f"(f) repeated 5xx aborts with exit 1 (got {code})\n{log}")
    check(posts == 4, "(f) two samples x (one try + one retry), then abort")
    check(not out.exists() or out.read_bytes() == b"", "(f) failed samples are not written")
    m = json.loads(out.with_name("gen_manifest.json").read_text(encoding="utf-8"))
    check(m["runs"][-1]["exit"] == "aborted: transport" and m["runs"][-1]["failed"] == 2
          and m["counts"]["done"] == 0, "(f) the manifest records the aborted run")

    # A refusal (HTTP 400) can be one entry's alone: never retried, not written, the run moves on --
    # and a resume retries it once and moves on again.
    mixed = [entry("code/001", "raw", False, "Program 1. [[bad-400]]"),
             entry("code/002", "raw", False, "Program 2."), entry("code/003", "raw", False, "Program 3.")]
    for attempt in (1, 2):
        code, log, posts, out = run(mixed, "mixed")
        m = json.loads(out.with_name("gen_manifest.json").read_text(encoding="utf-8"))
        want_posts = 3 if attempt == 1 else 1
        check(code == 0 and posts == want_posts and "code/001 REFUSED" in log and "HTTP 400" in log
              and [r["id"] for r in read_lines(out)] == ["code/002", "code/003"]
              and (m["runs"][-1]["failed"], m["runs"][-1]["refused"], m["counts"]["missing"]) == (1, 1, 1),
              f"(f) run {attempt}: a refused sample is skipped without a retry, the rest is generated "
              f"(exit {code}, {posts} posts)\n{log}")
    bad = [entry(f"code/{i:03d}", "raw", False, f"Program {i}. [[bad-400]]") for i in (1, 2, 3)]
    code, log, posts, out = run(bad, "bad", "--max-consecutive-failures", "2")
    m = json.loads(out.with_name("gen_manifest.json").read_text(encoding="utf-8"))
    check(code == 2 and posts == 2 and "2 consecutive failed samples" in log
          and m["runs"][-1]["exit"] == "aborted: request" and m["counts"]["done"] == 0,
          f"(f) refusals in a row abort with exit 2, one request each (exit {code}, {posts} posts)\n{log}")

    dying = [entry(f"chat/{i:03d}", "chat", False, f"Question {i}. [[kill-health]]") for i in (1, 2, 3)]
    code, log, posts, _ = run(dying, "dying", "--max-consecutive-failures", "3")
    check(code == 1 and posts == 2 and "/health does not answer" in log,
          f"(f) a dead /health aborts after the first failed sample (exit {code})\n{log}")
    stub.health_ok = True

    # A --kl-dir without text: refused before any request (the check would pass vacuously).
    ok = [entry("code/001", "raw", False, "Program 1.")]
    for d in (tmp / "no_kl", tmp / "empty_kl_run"):
        if d.name == "empty_kl_run":
            d.mkdir()
        p = write_json(tmp / "nokl.json", prompt_doc(ok))
        n_before = len(stub.posts)
        code, log = run_main(["--prompts", str(p), "--server", stub.url, "--out", str(tmp / "nokl" / "s.jsonl"),
                              "--kl-dir", str(d)])
        check(code == 2 and "pass vacuously" in log and len(stub.posts) == n_before
              and not (tmp / "nokl" / "s.jsonl").exists(),
              f"(g) --kl-dir {d.name} (no *.txt) refused before any request (exit {code})\n{log}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test_gen_corpus_"))
    stub = Stub()
    try:
        for name, fn in (("detectors", test_detectors), ("sampling", test_sampling), ("seeds", test_seeds),
                         ("prompt validation", lambda: test_prompt_validation(tmp)),
                         ("real prompt file", test_real_prompt_file),
                         ("generation", lambda: test_generation(tmp, stub)),
                         ("failures", lambda: test_failures(tmp, stub))):
            try:
                fn()
            except Exception:  # noqa: BLE001 -- report and keep going, like the sibling tests
                FAILURES.append(f"{name}: exception\n{traceback.format_exc()}")
                print(f"FAIL {name}: exception\n{traceback.format_exc()}")
    finally:
        stub.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    skipped = f", {len(SKIPS)} skipped" if SKIPS else ""
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} of {CHECKS} checks{skipped})")
        return 1
    print(f"OK ({CHECKS} checks{skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
