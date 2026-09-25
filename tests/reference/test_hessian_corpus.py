"""tests/reference/test_hessian_corpus.py

CPU-only test of hessian_capture.py's corpus source 4, the self-generated `samples.jsonl`
(`--gen-file`, docs/quant2.md "Corpus v2"). No GPU (it checks at the end that CUDA/HIP was never
initialized) and no model weights. The rendering and packing checks read the checkpoint's TOKENIZER
and chat template (common.DEFAULT_MODEL_DIR: tokenizer.json, tokenizer_config.json,
chat_template.jinja); without them those checks print SKIP and everything else still runs, with a
one-id-per-character stand-in tokenizer where only raw samples are involved.

Fixture: tests/reference/fixtures/hessian_gen_samples.jsonl, deliberately out of id order:
chat/001 (thinking, two turns), thai_prose/000 (raw), english_prose/001 (REJECTED raw), chat/000
(no thinking, with a system turn), english_prose/002 (raw), code/000 (raw), english_prose/000 (raw).

    (a) load_gen_samples: id order, rejected split off (never validated beyond its header fields);
        every contract refusal
    (b) render_gen_sample with the real template: raw verbatim; a thinking-off chat gets the EMPTY
        think block and equals the enable_thinking=false generation prompt + the generated turn; the
        two-turn thinking chat keeps BOTH thoughts with the final one last, and equals the thinking
        generation prompt + the generated final turn; preserve_thinking=false would drop only the
        earlier thought; check_rendered_chat refuses renders that lost a thought; the token ids are
        the template text's (one id per control token, no BOS) and equal apply_chat_template's own
    (c) packing: per category the concatenation (blank line after a raw document, chats back to
        back) cut into non-overlapping windows, the short tail dropped and reported;
        --gen-max-seqs keeps an evenly spread subset of the same windows; gen_window_starts
    (d) determinism: a rebuild and a shuffled copy of the file give identical sequences and
        sources (only the file's path and sha256 differ)
    (e) provenance: sha256 of the file, of the template and of each concatenation, sample counts,
        ids, reject reasons (counted by the "<kind>" of "<kind>: <detail>"); corpus_mismatches
        clean on the same file (kl_disjointness ignored) and
        naming an edited sample; build_corpus only APPENDS the gen sequences and sources
    (f) the disjointness gate: the fixture passes against the real tools/reference/kl_corpus/;
        excerpts read AT RUNTIME from every kl_corpus/*.txt (only ever written into this test's temp
        dir, deleted at exit) trip it -- in a raw text with its whitespace rewritten, in a raw
        sample's PROMPT, and in a chat's user turn -- naming the sample, where, the file and line;
        EVERY overlapping sample is listed (10, more than the old cap of 8); the refusal survives a
        'strict' cp1252 stdout (a pipe on Windows) with the Thai escaped; 49 characters do not trip
        it; a rejected sample is never checked; ~1.5M characters are scanned in seconds.
        Boilerplate: the #include/import grammar, kl_gate_segments' runs and line numbers; generic
        include/import headers (and the KL code files' own include/import lines) that a naive
        whole-file shingling flags do not trip it; the prose files have no boilerplate line
    (g) sources 1-3 (build_corpus): every token window is checked, an overlap is RECORDED under its
        own source's kl_disjointness (and printed, cp1252-safe), not refused; a code window's hit
        names the repo file holding the shared run
    (h) tokenization (--tokenizer, docs/quant2.md 3.4; needs the checkpoint's tokenizer): the Thai
        probe is 6 ids canonical and 14 hf-auto; canonical is tokenizers.Tokenizer.from_file itself;
        every chat case of tests/tokenizer/golden.json (r4dx's ground truth) renders and encodes to
        its prompt and ids canonically, every non-special encode case matches canonically and every
        compat case matches hf-auto; a Thai chat encodes differently in the two modes while its
        render does not; the fixture's gen windows under both modes (Thai shorter canonically, the
        text and every other category identical); the served-prompt prefix property in both modes;
        a bare transformers tokenizer is refused; provenance: corpus_record's mode,
        recorded_tokenizer_mode, corpus_mismatches, and --rms-only's mode resolution end to end
        (--dry-run on a temp set: a set recording no mode is read as hf-auto and matches, an
        explicit mismatching --tokenizer is refused, a canonical set follows its record, a set
        whose record lies about its tokens is caught by the token counts); canonical refuses
        non-NFC text; hf-auto refuses an AutoTokenizer that does not split (transformers 5.3.0);
        tokens_file_tokenizer_mode never guesses (r4dx-cli dump shape, exact canonical description,
        tokens.json by its ids, else unknown); make_tokens_json.main end to end (canonical default,
        the recorded mode and provenance, tokens.json and tokens_thai_canon.json reproduced byte for
        byte but for the two added fields, the mode guard and --force); the mode guards of
        imatrix_capture, kv_calibrate_full and kv_calibrate (before the GPU rule; no GPU touched)

Plain script, no pytest dependency, like test_hessian_rms.py:

    C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tests\\reference\\test_hessian_corpus.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_REFERENCE_DIR = REPO_ROOT / "tools" / "reference"
FIXTURE = REPO_ROOT / "tests" / "reference" / "fixtures" / "hessian_gen_samples.jsonl"
KL_DIR = TOOLS_REFERENCE_DIR / "kl_corpus"

try:
    import numpy  # noqa: F401  (hessian_capture imports it)
    import torch
    import transformers  # noqa: F401
except ImportError as e:
    print(f"SKIP: {e} -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

sys.path.insert(0, str(TOOLS_REFERENCE_DIR))
import hessian_capture as hc  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []
SKIPS: list[str] = []


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


class CharTokenizer:
    """One id per character, no chat template: enough for build_gen_corpus on raw-only files and for
    build_corpus' text sources. Duck-typed to common.RefTokenizer (which build_corpus requires)."""

    mode = "char (test stand-in)"
    chat_template = None

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))

    def render_chat(self, messages, **kwargs):
        raise AssertionError("CharTokenizer has no chat template")


def captured(fn, encoding: str = "utf-8"):
    """Run fn with sys.stdout on a fresh `encoding` stream with 'strict' errors -- what a pipe or a
    file gets on Windows (cp1252 here), and ctest always pipes. Returns (fn's result, or the
    exception it raised -- SystemExit included, what was printed)."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding=encoding, errors="strict", newline="\n")
    with contextlib.redirect_stdout(stream):
        try:
            res = fn()
        except Exception as e:
            res = e
        except SystemExit as e:
            res = e
    stream.flush()
    return res, buf.getvalue().decode(encoding)


def fixture_lines() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_bytes().decode("utf-8").split("\n") if line.strip()]


def write_jsonl(path: Path, samples: list[dict]) -> Path:
    path.write_bytes("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples).encode("utf-8"))
    return path


def by_id(samples: list[dict]) -> dict[str, dict]:
    return {s["id"]: s for s in samples}


def load_tokenizer(mode: str = "canonical"):
    """The capture's own tokenizer (common.RefTokenizer) in `mode`; canonical is the default mode."""
    from common import DEFAULT_MODEL_DIR, load_ref_tokenizer

    if not (DEFAULT_MODEL_DIR / "tokenizer.json").exists():
        return None, DEFAULT_MODEL_DIR
    return load_ref_tokenizer(DEFAULT_MODEL_DIR, mode), DEFAULT_MODEL_DIR


def skip(label: str) -> None:
    SKIPS.append(label)
    print(f"SKIP {label} (no tokenizer at common.DEFAULT_MODEL_DIR)")


# ---- (a) -----------------------------------------------------------------------------------------


def test_contract(tmp: Path) -> None:
    used, rejected, sha = hc.load_gen_samples(FIXTURE)
    check([s["id"] for s in used] == ["chat/000", "chat/001", "code/000", "english_prose/000",
                                      "english_prose/002", "thai_prose/000"],
          "(a) the non-rejected samples come back sorted by id")
    check([s["id"] for s in rejected] == ["english_prose/001"], "(a) the rejected sample is split off")
    check(sha == hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), "(a) the file's sha256")

    good = by_id(fixture_lines())
    raw = good["english_prose/000"]
    chat = good["chat/000"]
    think = good["chat/001"]

    def refused(label: str, needle: str, lines: list) -> None:
        p = tmp / "bad.jsonl"
        p.write_bytes("".join((x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)) + "\n"
                              for x in lines).encode("utf-8"))
        raises(SystemExit, needle, lambda: hc.load_gen_samples(p), f"(a) refuses {label}")

    def edit(s: dict, **kw) -> dict:
        s = json.loads(json.dumps(s))
        s.update(kw)
        return s

    refused("a line that is not JSON", "bad.jsonl:2: not JSON", [raw, "{not json"])
    refused("a JSON line that is not an object", "not a JSON object", [raw, "[1, 2]"])
    refused("a duplicate id", "duplicate id (first on line 1)", [raw, raw])
    refused("an id whose prefix is not its category", "is not '<category>/<NNN>'",
            [raw, edit(raw, id="code/009")])
    refused("an unknown category", "category 'poetry'", [raw, edit(raw, id="poetry/000", category="poetry")])
    refused("an unknown format", "format 'markdown'", [raw, edit(raw, id="english_prose/005", format="markdown")])
    refused("a non-boolean rejected", "rejected 'no'", [raw, edit(raw, id="english_prose/005", rejected="no")])
    refused("a non-boolean enable_thinking", "enable_thinking None",
            [raw, edit(raw, id="english_prose/005", enable_thinking=None)])
    refused("a raw text that is not the final content stripped", "not the final assistant content stripped",
            [edit(raw, text=raw["text"] + " (edited)")])
    refused("an empty raw text", "non-empty text",
            [edit(raw, text="", messages=[raw["messages"][0], {"role": "assistant", "content": "  "}])])
    refused("messages ending in a user turn", "not the generated assistant turn",
            [edit(chat, messages=chat["messages"][:2])])
    refused("a conversation without a user turn", "no user turn",
            [edit(chat, messages=[chat["messages"][0], chat["messages"][2]])])
    refused("a message without string content", "non-empty list of",
            [edit(chat, messages=chat["messages"][:2] + [{"role": "assistant", "content": None}])])
    refused("a tool turn", "non-empty list of",
            [edit(chat, messages=chat["messages"] + [{"role": "tool", "content": "x"}])])
    refused("a chat sample with a text", "text must be null", [edit(chat, text="x")])
    no_rc = json.loads(json.dumps(think))
    del no_rc["messages"][3]["reasoning_content"]
    refused("a thinking chat whose final turn has no reasoning_content", "no reasoning_content", [no_rc])
    leaked = json.loads(json.dumps(chat))
    leaked["messages"][2]["reasoning_content"] = "a thought"
    refused("a thinking-off chat carrying a thought", "non-empty reasoning_content", [leaked])
    refused("a file where every sample is rejected", "no sample with rejected == false",
            [good["english_prose/001"]])

    # Accepted: an earlier turn of a thinking chat without its thought; a rejected sample whose body
    # breaks every rule (it is counted, never rendered).
    early = json.loads(json.dumps(think))
    del early["messages"][1]["reasoning_content"]
    junk = {"id": "english_prose/009", "category": "english_prose", "format": "raw",
            "enable_thinking": False, "messages": None, "text": None, "rejected": True,
            "reject_reason": "empty output"}
    p = write_jsonl(tmp / "ok.jsonl", [early, junk])
    u, r, _ = hc.load_gen_samples(p)
    check([s["id"] for s in u] == ["chat/001"] and [s["id"] for s in r] == ["english_prose/009"],
          "(a) accepts an earlier thinking turn without reasoning_content and a malformed rejected sample")

    # json.dumps(ensure_ascii=False) leaves U+2028/U+2029/U+0085/\x1c/\v/\f unescaped inside a string;
    # the file is still one sample per "\n"-terminated line (str.splitlines would cut it apart). A
    # CRLF-terminated file reads the same.
    odd = "line sep par\u0085nel\x1cfs\x0bvt\x0cff"
    weird = edit(raw, messages=[raw["messages"][0], {"role": "assistant", "content": odd}], text=odd)
    p = write_jsonl(tmp / "odd.jsonl", [weird, chat])
    check(" " in p.read_text(encoding="utf-8"), "(a) setup: the separators are written unescaped")
    u, _, _ = hc.load_gen_samples(p)
    check([s["id"] for s in u] == ["chat/000", "english_prose/000"] and u[1]["text"] == odd,
          "(a) Unicode line separators inside a string do not split the JSONL line")
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    u, _, _ = hc.load_gen_samples(p)
    check(len(u) == 2 and u[1]["text"] == odd, "(a) CRLF line endings read the same")


# ---- (b) -----------------------------------------------------------------------------------------


def test_render(tok, model_dir: Path) -> None:
    used, _, _ = hc.load_gen_samples(FIXTURE)
    s = by_id(used)
    tpl = tok.chat_template
    jinja = model_dir / "chat_template.jinja"
    check(isinstance(tpl, str) and (not jinja.exists() or tpl == jinja.read_text(encoding="utf-8")),
          "(b) the tokenizer's chat template is the checkpoint's chat_template.jinja")
    for sid in ("thai_prose/000", "english_prose/000", "code/000"):
        check(hc.render_gen_sample(s[sid], tok, tpl) == s[sid]["text"], f"(b) raw {sid} renders verbatim")

    # Thinking off, with a system turn: no reasoning-effort sentence, and the assistant turn carries the
    # EMPTY think block -- which is the template's enable_thinking=false generation prompt.
    c0 = s["chat/000"]
    m = c0["messages"]
    r0 = hc.render_gen_sample(c0, tok, tpl)
    want0 = (f"<|im_start|>system\n{m[0]['content']}<|im_end|>\n<|im_start|>user\n{m[1]['content']}"
             f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{m[2]['content']}<|im_end|>\n")
    check(r0 == want0, "(b) thinking-off chat: system, user, then assistant with the EMPTY think block")
    gp0 = tok.render_chat(m[:-1], chat_template=tpl, add_generation_prompt=True, enable_thinking=False)
    check(r0 == gp0 + m[2]["content"] + "<|im_end|>\n",
          "(b) thinking-off chat == the enable_thinking=false generation prompt + the generated turn")

    # Thinking on, two turns: the system sentence, BOTH thoughts kept, the final turn last.
    c1 = s["chat/001"]
    m = c1["messages"]
    r1 = hc.render_gen_sample(c1, tok, tpl)
    turns = (f"<|im_start|>user\n{m[0]['content']}<|im_end|>\n"
             f"<|im_start|>assistant\n<think>\n{m[1]['reasoning_content']}\n</think>\n\n{m[1]['content']}"
             f"<|im_end|>\n<|im_start|>user\n{m[2]['content']}<|im_end|>\n")
    final = f"<|im_start|>assistant\n<think>\n{m[3]['reasoning_content']}\n</think>\n\n{m[3]['content']}<|im_end|>\n"
    head = r1[:len(r1) - len(turns) - len(final)]
    check(r1.endswith(turns + final) and head.startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
          and head.endswith("<|im_end|>\n") and head.count("<|im_start|>") == 1,
          "(b) thinking chat: the default-effort system turn, then both turns with their thoughts, final last")
    check(r1.endswith(hc.assistant_turn_rendering(m[3])) and final == hc.assistant_turn_rendering(m[3]),
          "(b) the final turn's thinking block survives, in assistant_turn_rendering's form")
    gp1 = tok.render_chat(m[:-1], chat_template=tpl, add_generation_prompt=True, enable_thinking=True)
    check(gp1.endswith("<|im_start|>assistant\n<think>\n") and
          r1 == gp1 + m[3]["reasoning_content"] + "\n</think>\n\n" + m[3]["content"] + "<|im_end|>\n",
          "(b) thinking chat == the thinking generation prompt (which ends in <think>\\n) + the generated turn")

    # What preserve_thinking=false would do (neither the generator nor this script sets it): only the
    # earlier turn loses its thought; the final one keeps it. check_rendered_chat refuses that render,
    # and one that lost the final thought.
    np_ = tok.render_chat(m, chat_template=tpl, add_generation_prompt=False, enable_thinking=True,
                          preserve_thinking=False)
    check(m[1]["reasoning_content"] not in np_ and np_.endswith(final),
          "(b) preserve_thinking=false: the earlier thought is dropped, the final turn's is kept")
    raises(SystemExit, "assistant turn 1/2", lambda: hc.check_rendered_chat(c1, np_),
           "(b) check_rendered_chat refuses a render that dropped an earlier thought")
    lost = r1[:len(r1) - len(final)] + f"<|im_start|>assistant\n{m[3]['content']}<|im_end|>\n"
    raises(SystemExit, "assistant turn 2/2", lambda: hc.check_rendered_chat(c1, lost),
           "(b) check_rendered_chat refuses a render that dropped the final thought")
    raises(SystemExit, "at the end of the conversation", lambda: hc.check_rendered_chat(c1, r1 + "x"),
           "(b) check_rendered_chat refuses anything after the final turn")

    # Tokens: exactly the template text's, one id per control token, no BOS. These chats are English,
    # so transformers' own apply_chat_template(tokenize=True) -- AutoTokenizer, i.e. hf-auto -- gives
    # the same ids as the canonical encode (for Thai it would not: (h)).
    cid = {t: tok.hf.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")}
    for sample, text in ((c0, r0), (c1, r1)):
        ids = tok.encode(text)
        ref = tok.hf.apply_chat_template(sample["messages"], chat_template=tpl, tokenize=True, return_dict=False,
                                         add_generation_prompt=False, enable_thinking=sample["enable_thinking"])
        check(ids == list(ref), f"(b) {sample['id']}: ids == apply_chat_template(tokenize=True)")
        check(all(ids.count(i) == text.count(t) for t, i in cid.items()) and ids[0] == cid["<|im_start|>"],
              f"(b) {sample['id']}: one id per <|im_start|>/<|im_end|>/<think>/</think>, no BOS")


# ---- (c)-(e) -------------------------------------------------------------------------------------


def expected_concat(tok, samples: list[dict]) -> dict[str, str]:
    """Per category: the rendered non-rejected samples in id order, a blank line after a raw one when
    anything follows -- written out independently of build_gen_corpus."""
    out: dict[str, list] = {}
    for s in sorted(samples, key=lambda x: x["id"]):
        if s["rejected"]:
            out.setdefault(s["category"], [])
            continue
        t = s["text"] if s["format"] == "raw" else tok.render_chat(
            s["messages"], add_generation_prompt=False, enable_thinking=s["enable_thinking"])
        out.setdefault(s["category"], []).append((s["format"], t))
    res = {}
    for cat, items in out.items():
        text = ""
        for k, (fmt, t) in enumerate(items):
            text += (hc.GEN_RAW_SEPARATOR if k and items[k - 1][0] == "raw" else "") + t
        res[cat] = text
    return res


def seq_tuples(seqs) -> list[tuple]:
    return [(s.name, s.source, list(s.token_ids)) for s in seqs]


def test_window_starts() -> None:
    check(hc.gen_window_starts(100, 16, None) == [0, 16, 32, 48, 64, 80], "(c) 100 tokens, L=16: 6 windows")
    check(hc.gen_window_starts(100, 16, 3) == [0, 32, 80], "(c) cap 3 of 6: first, middle, last grid window")
    check(hc.gen_window_starts(100, 16, 1) == [0], "(c) cap 1: the first window")
    check(hc.gen_window_starts(100, 16, 6) == hc.gen_window_starts(100, 16, 50) ==
          hc.gen_window_starts(100, 16, None), "(c) a cap at or above the window count changes nothing")
    check(hc.gen_window_starts(15, 16, None) == [] and hc.gen_window_starts(0, 16, 4) == [],
          "(c) shorter than one window: none")
    ok = True
    for n_all in range(1, 40):
        full = hc.gen_window_starts(n_all * 16 + 5, 16, None)
        for cap in range(1, n_all + 1):
            got = hc.gen_window_starts(n_all * 16 + 5, 16, cap)
            ok = ok and len(got) == cap and len(set(got)) == cap and set(got) <= set(full) and \
                got == sorted(got) and got[0] == 0 and (cap == 1 or got[-1] == full[-1])
    check(ok, "(c) every cap: that many distinct windows of the uncapped grid, first and last included")


def test_packing(tok, tmp: Path) -> None:
    L = 16
    seqs, sources = hc.build_gen_corpus(FIXTURE, tok, L)
    lines = fixture_lines()
    concat = expected_concat(tok, lines)
    src = {e["name"]: e for e in sources}
    check(list(src) == sorted(concat) == ["chat", "code", "english_prose", "thai_prose"],
          "(c) one source entry per category, sorted")
    check(concat["english_prose"].count("\n\n") >= 1 and
          concat["english_prose"].startswith(by_id(lines)["english_prose/000"]["text"] + "\n\n"),
          "(c) english_prose: 000 + blank line + 002 (001 is rejected)")
    check("<|im_end|>\n<|im_start|>" in concat["chat"] and "<|im_end|>\n\n<|im_start|>" not in concat["chat"],
          "(c) chat: the two rendered chats back to back")
    ok_all = True
    tails = 0
    for cat, text in concat.items():
        ids = tok.encode(text)
        n_all = len(ids) // L
        want = [(f"gen/{cat}/{w:03d}", "gen", ids[w * L:(w + 1) * L]) for w in range(n_all)]
        got = [t for t in seq_tuples(seqs) if t[0].startswith(f"gen/{cat}/")]
        e = src[cat]
        ok = (got == want and e["sequences"] == n_all and e["tokens"] == n_all * L and
              e["concatenation_tokens"] == len(ids) and e["windows_available"] == n_all and
              e["dropped_tail_tokens"] == len(ids) - n_all * L and
              e["window_starts"] == [w * L for w in range(n_all)] and
              e["sha256_of_concatenation"] == hashlib.sha256(text.encode("utf-8")).hexdigest() and
              e["concatenation_chars"] == len(text))
        check(ok, f"(c) {cat}: {n_all} non-overlapping {L}-token windows of the concatenation "
                  f"({len(ids)} tokens), tail of {len(ids) % L} dropped and recorded")
        ok_all = ok_all and ok
        tails += e["dropped_tail_tokens"]
    check(tails > 0, "(c) the fixture exercises a non-empty dropped tail")

    capped, csrc = hc.build_gen_corpus(FIXTURE, tok, L, max_seqs=2)
    cs = {e["name"]: e for e in csrc}
    ok = True
    for cat, e in src.items():
        want_n = min(2, e["windows_available"])
        c = cs[cat]
        ok = ok and c["sequences"] == want_n and c["max_seqs"] == 2 and \
            set(c["window_starts"]) <= set(e["window_starts"]) and \
            (want_n < 2 or c["window_starts"] == [0, e["window_starts"][-1]]) and \
            c["windows_available"] == e["windows_available"] and \
            c["dropped_tail_tokens"] == e["dropped_tail_tokens"]
    # Each capped window holds exactly the tokens of the uncapped window at the same offset.
    full = {t[0]: t[2] for t in seq_tuples(seqs)}
    for name, _, ids in seq_tuples(capped):
        _, cat, w = name.split("/")
        start = cs[cat]["window_starts"][int(w)]
        ok = ok and ids == full.get(f"gen/{cat}/{start // L:03d}")
    check(ok, "(c) --gen-max-seqs 2: the first and last of each category's windows, same tokens")
    raises(SystemExit, "must be >= 1", lambda: hc.build_gen_corpus(FIXTURE, tok, L, max_seqs=0),
           "(c) --gen-max-seqs 0 is refused")
    none, nsrc = hc.build_gen_corpus(FIXTURE, tok, 1 << 20)
    check(none == [] and all(e["sequences"] == 0 and e["dropped_tail_tokens"] == e["concatenation_tokens"] > 0
                             for e in nsrc),
          "(c) a category shorter than one window contributes nothing, and its tokens are the dropped tail")

    # (d) determinism.
    again, asrc = hc.build_gen_corpus(FIXTURE, tok, L)
    check(seq_tuples(again) == seq_tuples(seqs) and asrc == sources, "(d) a rebuild is identical")
    shuffled = write_jsonl(tmp / "shuffled.jsonl", list(reversed(lines)))
    sh, shsrc = hc.build_gen_corpus(shuffled, tok, L)

    def strip(entries):
        return [{k: v for k, v in e.items() if k not in ("path", "sha256")} for e in entries]
    check(seq_tuples(sh) == seq_tuples(seqs) and strip(shsrc) == strip(sources) and
          all(e["sha256"] != sources[0]["sha256"] for e in shsrc),
          "(d) a shuffled file gives the same sequences and sources (only its path and sha256 differ)")

    # (e) provenance.
    file_sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    tpl_sha = hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest()
    ep, ch = src["english_prose"], src["chat"]
    check(all(e["source"] == "gen" and e["sha256"] == file_sha and e["path"] == str(FIXTURE) for e in sources),
          "(e) every category records the file's path and sha256")
    check(ep["samples"] == 2 and ep["samples_rejected"] == 1 and
          ep["reject_reasons"] == {"degenerate repetition": 1} and
          ep["sample_ids"] == ["english_prose/000", "english_prose/002"] and
          ep["formats"] == {"raw": 2, "chat": 0} and "chat_template_sha256" not in ep,
          "(e) english_prose: 2 used (ids), 1 rejected with its reason, raw only, no template hash")
    # The generator writes "<kind>: <detail>" (the detail of a "kl overlap" is the matched KL run):
    # only the kind is counted, as in gen_manifest.json; a missing reason is "unspecified".
    kinds = fixture_lines()
    for s in kinds:
        if s["id"] == "english_prose/001":
            s["reject_reason"] = "kl overlap: content shares 'a stand-in run' with kl_corpus/x.txt (turn 1)"
            kinds.append(dict(s, id="english_prose/009", reject_reason=None))
            break
    _, ksrc = hc.build_gen_corpus(write_jsonl(tmp / "kinds.jsonl", kinds), tok, L)
    kr = {e["name"]: e for e in ksrc}["english_prose"]
    check(kr["samples_rejected"] == 2 and kr["reject_reasons"] == {"kl overlap": 1, "unspecified": 1},
          f"(e) reject_reasons counts the kind of '<kind>: <detail>', 'unspecified' without one "
          f"({kr['reject_reasons']})")
    check(ch["samples"] == 2 and ch["thinking_samples"] == 1 and ch["formats"] == {"raw": 0, "chat": 2} and
          ch["chat_template_sha256"] == tpl_sha and ch["finish_reasons"] == {"stop": 2},
          "(e) chat: 2 chats (1 thinking), the template's sha256, finish reasons")
    kd = ch["kl_disjointness"]
    check(kd["ok"] and kd["shingle_chars"] == 50 and kd["kl_dir"] == "tools/reference/kl_corpus/" and
          sorted(kd["kl_files"]) == sorted(p.name for p in KL_DIR.glob("*.txt")),
          "(e) kl_disjointness records the gate: 50 chars, every kl_corpus/*.txt by sha256")
    rec = {"sequences": len(seqs), "tokens": sum(len(s.token_ids) for s in seqs), "seq_len": L,
           "sources": sources}
    moved = [dict(e, kl_disjointness={"ok": True, "kl_files": {}}) for e in sources]
    check(hc.corpus_mismatches(rec, moved, rec["sequences"], rec["tokens"], L) == [],
          "(e) corpus_mismatches: the same file matches (kl_disjointness is a HOW field, ignored)")
    edited = fixture_lines()
    for s in edited:
        if s["id"] == "english_prose/000":
            s["messages"][1]["content"] = s["messages"][1]["content"].replace("rocks", "stones")
            s["text"] = s["messages"][1]["content"].strip()
    e_seqs, e_src = hc.build_gen_corpus(write_jsonl(tmp / "edited.jsonl", edited), tok, L)
    diffs = hc.corpus_mismatches(rec, e_src, len(e_seqs), sum(len(s.token_ids) for s in e_seqs), L)
    check(any("gen/english_prose sha256_of_concatenation" in d for d in diffs) and
          any("gen/chat path" in d for d in diffs),
          f"(e) corpus_mismatches names an edited sample's category (and the other file) ({diffs[:3]})")

    # build_corpus: gen sources are APPENDED; without --gen-file nothing changes.
    args = argparse.Namespace(calib_txt=TOOLS_REFERENCE_DIR / "calib.txt", corpus_dir="none",
                              wikitext="none", wikitext_seqs=0, code_seqs=0, seq_len=L, gen_file=None,
                              gen_max_seqs=None)
    a_seqs, a_src = hc.build_corpus(args, tok)
    b_seqs, b_src = hc.build_corpus(argparse.Namespace(**dict(vars(args), gen_file=FIXTURE)), tok)
    n = len(a_seqs)
    check(n == 1 and seq_tuples(b_seqs[:n]) == seq_tuples(a_seqs) and b_src[:len(a_src)] == a_src and
          seq_tuples(b_seqs[n:]) == seq_tuples(seqs) and b_src[len(a_src):] == sources,
          "(e) build_corpus: sources 1-3 untouched, the gen sequences and sources appended after them")
    ad = a_src[0]["kl_disjointness"]
    check(ad["ok"] and ad["overlaps"] == [] and "window" in ad["checked"] and
          ad["kl_files"] == kd["kl_files"],
          "(e) build_corpus: calib.txt's window is checked too and recorded as disjoint")
    c_seqs, _ = hc.build_corpus(argparse.Namespace(**dict(vars(args), gen_file="none")), tok)
    check(seq_tuples(c_seqs) == seq_tuples(a_seqs), "(e) --gen-file none is no gen source")


# ---- (f) -----------------------------------------------------------------------------------------


def kl_texts() -> dict[str, str]:
    """The held-out KL files, read at runtime. Their text only ever goes into this test's temp dir."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(KL_DIR.glob("*.txt"))}


def naive_shingles(texts: dict[str, str], n: int = 50) -> dict[str, tuple[str, int]]:
    """What the gate compared against before the boilerplate rule: every n-char run of each whole
    whitespace-normalized file (find_kl_overlaps' shingle-map shape)."""
    out: dict[str, tuple[str, int]] = {}
    for name, t in texts.items():
        t = hc.ws_normalize(t)
        for i in range(len(t) - n + 1):
            out.setdefault(t[i:i + n], (name, 0))
    return out


def test_boilerplate(kl_idx) -> None:
    b = hc.KL_BOILERPLATE_LINE
    for line in ("#include <vector>", '#include "r4dx/core/error.hpp"', "  # include <a.h>  // why",
                 "import os", "import numpy as np", "import a.b, c as d  # noqa", "from pathlib import Path",
                 "from . import x", "from a.b import c as d, e,", "from a import (b, c)", "from x import *",
                 "from common import (  # noqa: E402"):
        check(b.fullmatch(line), f"(f) boilerplate: {line!r}")
    for line in ("import tariffs on steel", "#include_next <x>", "#define X 1", "print('import os')",
                 "from the river import goods", "importance of imports", "x = 1  # import os", ""):
        check(not b.fullmatch(line), f"(f) not boilerplate: {line!r}")
    segs, skipped = hc.kl_gate_segments("alpha one\n#include <x>\nbeta\nfrom a import (\n  b,\n  c as d,  # e\n)\n"
                                        "gamma\nfrom q import (\nx = 1")
    check([t for t, _ in segs] == ["alpha one", "beta", "gamma", "x = 1"] and skipped == 6,
          f"(f) kl_gate_segments: runs between boilerplate lines, a `from a import (` block to its `)`, an "
          f"unclosed one ends at the first non-name line ({segs}, {skipped})")
    text = "one\n two  three \n\n\tfour\r\n"
    segs, skipped = hc.kl_gate_segments(text)
    t, line_of = segs[0]
    check(len(segs) == 1 and skipped == 0 and t == hc.ws_normalize(text) and len(line_of) == len(t) and
          line_of[t.index("one")] == 1 and line_of[t.index("two")] == 2 and line_of[t.index("three") + 4] == 2 and
          line_of[t.index("four")] == 4,
          f"(f) kl_gate_segments: no boilerplate -> one run == ws_normalize(text), a file line per char ({line_of})")

    kl = kl_texts()
    naive = naive_shingles(kl)
    check(kl_idx.shingles.keys() <= naive.keys() and len(kl_idx.shingles) > 0.8 * len(naive),
          "(f) the boilerplate rule only removes shingles, and few of them")
    bp = kl_idx.boilerplate_lines
    check(bp["english_prose.txt"] == 0 and bp["thai_prose.txt"] == 0 and bp["cpp_source.txt"] > 0 and
          bp["python_source.txt"] > 0, f"(f) boilerplate lines: none in the prose files, some in the code files ({bp})")

    def trips(text: str, shingles) -> bool:
        return bool(hc.find_kl_overlaps([("x/0", text)], shingles))
    # Generic headers any generated C++/Python file may open with: the naive shingling flagged them.
    for header in ("x\n#include <algorithm>\n#include <chrono>\n#include <cstdint>\n",
                   "from __future__ import annotations\n\nimport argparse\nimport json\n"):
        check(trips(header, naive) and not trips(header, kl_idx.shingles),
              f"(f) a generic header trips the naive shingling, not the gate: {header!r}")
    # The KL code files' own single-line include/import lines, in order (read at runtime, in memory only).
    for name, lead in (("cpp_source.txt", ("#include",)), ("python_source.txt", ("import ", "from "))):
        block = "\n".join(line for line in kl[name].split("\n") if line.startswith(lead))
        check(len(hc.ws_normalize(block)) > 100 and trips(block, naive) and not trips(block, kl_idx.shingles),
              f"(f) {name}'s include/import lines ({len(block)} chars) do not trip the gate")


def test_gate(tok, tmp: Path) -> None:
    lines = fixture_lines()
    raw_only = [s for s in lines if s["format"] == "raw"]
    char_tok = CharTokenizer()
    # The fixture (all of it when the real template is here) is disjoint from the real KL corpus.
    kl_idx = hc.kl_index(KL_DIR)
    shingles, shas = kl_idx.shingles, kl_idx.shas
    ok_raw = hc.find_kl_overlaps([(s["id"], s["text"]) for s in raw_only if not s["rejected"]], shingles) == []
    check(ok_raw and len(shas) >= 4 and len(shingles) > 1000 and kl_idx.label == "tools/reference/kl_corpus/",
          f"(f) the fixture's raw texts are disjoint from the {len(shas)} kl_corpus files")
    if tok is not None:
        seqs, _ = hc.build_gen_corpus(FIXTURE, tok, 16)
        check(bool(seqs), "(f) the whole fixture, chats rendered, passes the gate")
    test_boilerplate(kl_idx)
    kl = kl_texts()

    def inject_raw(sid: str, extra: str) -> list[dict]:
        out = json.loads(json.dumps(raw_only))
        for s in out:
            if s["id"] == sid:
                s["messages"][-1]["content"] = s["messages"][-1]["content"] + " " + extra + " "
                s["text"] = s["messages"][-1]["content"].strip()
        return out

    target = {"thai_prose.txt": "thai_prose/000", "english_prose.txt": "english_prose/002",
              "cpp_source.txt": "code/000", "python_source.txt": "code/000"}
    for name, text in kl.items():
        mid = len(text) // 2
        excerpt = text[mid:mid + 160]
        # Whitespace rewritten: every run doubled / turned into CRLF + tab, which normalization undoes.
        mangled = excerpt.replace("\n", "\r\n\t").replace(" ", "  ")
        sid = target.get(name, "english_prose/000")
        p = write_jsonl(tmp / f"kl_{name}.jsonl", inject_raw(sid, mangled))
        raises(SystemExit, f"(first: {sid} vs {name})", lambda p=p: hc.build_gen_corpus(p, char_tok, 16),
               f"(f) an excerpt of kl_corpus/{name} with rewritten whitespace in {sid} trips the gate")

    # The refusal on a 'strict' cp1252 stdout -- a pipe or a file on Windows, which is how ctest runs
    # this: no UnicodeEncodeError, the sample listed, the Thai escaped.
    res, out = captured(lambda: hc.build_gen_corpus(tmp / "kl_thai_prose.txt.jsonl", char_tok, 16), "cp1252")
    check(isinstance(res, SystemExit) and "(first: thai_prose/000 vs thai_prose.txt)" in str(res) and
          "[hessian]   thai_prose/000 text (normalized char" in out and "with thai_prose.txt line " in out and
          "\\u0e" in out,
          f"(f) the Thai refusal on a strict cp1252 stdout: SystemExit, the sample listed, the Thai escaped "
          f"({type(res).__name__}: {str(res)[:120]!r})")

    # A raw sample's PROMPT is checked too: it is not calibrated on, but it steered the text.
    prompt = json.loads(json.dumps(raw_only))
    for s in prompt:
        if s["id"] == "english_prose/000":
            s["messages"][0]["content"] += "\n\nContinue this essay:\n" + kl["english_prose.txt"][300:700]
    p = write_jsonl(tmp / "kl_prompt.jsonl", prompt)
    res, out = captured(lambda: hc.build_gen_corpus(p, char_tok, 16))
    check(isinstance(res, SystemExit) and "(first: english_prose/000 vs english_prose.txt)" in str(res) and
          "[hessian]   english_prose/000 messages[0].content (normalized char" in out,
          f"(f) a KL excerpt in a raw sample's prompt trips the gate, naming where ({str(res)[:120]!r})")

    # EVERY overlapping sample is listed (the refusal used to name 8).
    many = json.loads(json.dumps(raw_only))
    base = by_id(many)["english_prose/000"]
    eng = kl["english_prose.txt"]
    for k in range(10):
        s = json.loads(json.dumps(base))
        s["id"] = f"english_prose/{100 + k}"
        s["messages"][-1]["content"] = "An essay. " + eng[250 * k:250 * k + 150] + " The end."
        s["text"] = s["messages"][-1]["content"].strip()
        many.append(s)
    p = write_jsonl(tmp / "kl_many.jsonl", many)
    res, out = captured(lambda: hc.build_gen_corpus(p, char_tok, 16))
    listed = [k for k in range(10) if f"[hessian]   english_prose/{100 + k} text (" in out]
    check(isinstance(res, SystemExit) and "10 sample(s), every one listed above" in str(res) and len(listed) == 10,
          f"(f) all 10 overlapping samples are listed ({len(listed)} found; {str(res)[:120]!r})")

    # Exactly the threshold: 49 normalized characters (fenced by a character no KL file contains) do
    # not trip it, 50 do.
    thai_norm = hc.ws_normalize(kl["thai_prose.txt"])
    fence = "\u00a4"
    check(all(fence not in t for t in kl.values()), "(f) setup: the fence character is not in kl_corpus")
    at = len(thai_norm) // 3
    while thai_norm[at] == " ":
        at += 1
    for n, want in ((49, False), (50, True)):
        hits = hc.find_kl_overlaps([("x/0", "abc " + fence + thai_norm[at:at + n] + fence + " def")], shingles)
        check(bool(hits) == want and (not hits or hits[0]["kl_file"] == "thai_prose.txt"),
              f"(f) {n} shared normalized characters {'trip' if want else 'do not trip'} the gate")
    # kl_line: the line of kl_corpus/thai_prose.txt the shared run starts on.
    thai_lines = kl["thai_prose.txt"].split("\n")
    line = hits[0]["kl_line"]
    first = hc.ws_normalize(thai_lines[line - 1])
    k = hc.ws_normalize("\n".join(thai_lines[line - 1:])).find(hits[0]["substring"])
    check(first and 0 <= k < len(first), f"(f) kl_line {line} is the file line the shared run starts on")

    # A rejected sample is never checked (nor used).
    rej = json.loads(json.dumps(raw_only))
    for s in rej:
        if s["rejected"]:
            s["messages"][-1]["content"] += " " + kl["cpp_source.txt"][:400]
    p = write_jsonl(tmp / "kl_rejected.jsonl", rej)
    seqs, _ = hc.build_gen_corpus(p, char_tok, 16)
    check(bool(seqs), "(f) KL text in a REJECTED sample does not trip the gate")

    # In a chat's user turn (the rendered text is what is checked, prompts included).
    if tok is not None:
        chat = json.loads(json.dumps(lines))
        for s in chat:
            if s["id"] == "chat/001":
                s["messages"][2]["content"] += "\n" + kl["english_prose.txt"][200:320]
        p = write_jsonl(tmp / "kl_chat.jsonl", chat)
        raises(SystemExit, "(first: chat/001 vs english_prose.txt)", lambda: hc.build_gen_corpus(p, tok, 16),
               "(f) a KL excerpt in a chat's user turn trips the gate")
    else:
        skip("(f) KL excerpt in a chat's user turn")

    # Speed: ~1.5M characters of mixed Thai/Latin text, as ~300 samples.
    rng = random.Random(5)
    alphabet = "abcdefghijklmnopqrstuvwxyz        .,\n" + "".join(chr(c) for c in range(0x0E01, 0x0E2F))
    texts = [(f"t/{k:03d}", "".join(rng.choices(alphabet, k=5000))) for k in range(300)]
    t0 = time.perf_counter()
    hits = hc.find_kl_overlaps(texts, shingles)
    secs = time.perf_counter() - t0
    print(f"[test] find_kl_overlaps: {sum(len(t) for _, t in texts)} chars in {secs:.2f}s")
    check(hits == [] and secs < 20.0, f"(f) ~1.5M characters scanned in {secs:.2f}s (< 20s)")
    raises(SystemExit, "no *.txt", lambda: hc.kl_index(tmp), "(f) an empty KL directory is refused")


# ---- (g) -----------------------------------------------------------------------------------------


def test_windows(tmp: Path) -> None:
    """Sources 1-3 with the one-id-per-character tokenizer: no checkpoint needed."""
    char_tok = CharTokenizer()
    kl = kl_texts()
    fx = by_id(fixture_lines())

    # Two calib files, one quoting kl_corpus/thai_prose.txt (temp dir only).
    cdir = tmp / "calib_dir"
    cdir.mkdir()
    thai = kl["thai_prose.txt"]
    (cdir / "clean.txt").write_text(fx["english_prose/000"]["text"], encoding="utf-8")
    (cdir / "quoted.txt").write_text(fx["thai_prose/000"]["text"] + "\n" + thai[len(thai) // 2:len(thai) // 2 + 200],
                                     encoding="utf-8")
    args = argparse.Namespace(calib_txt="none", corpus_dir=cdir, wikitext="none", wikitext_seqs=0, code_seqs=0,
                              seq_len=1 << 16, gen_file=None, gen_max_seqs=None)
    res, out = captured(lambda: hc.build_corpus(args, char_tok), "cp1252")
    ok = isinstance(res, tuple)
    if ok:
        seqs, sources = res
        kd = {s["name"]: s["kl_disjointness"] for s in sources}
        q = kd["quoted"]["overlaps"]
        ok = ([s.name for s in seqs] == ["calib/clean", "calib/quoted"] and kd["clean"]["ok"] and
              kd["clean"]["overlaps"] == [] and not kd["quoted"]["ok"] and len(q) == 1 and
              q[0]["id"] == "calib/quoted" and q[0]["kl_file"] == "thai_prose.txt" and
              "WARNING disjointness: 1 of 2 window(s)" in out and "[hessian]   calib/quoted (normalized char" in out)
    check(ok, f"(g) a calib window quoting kl_corpus is recorded under ITS source and printed (cp1252-safe), not "
              f"refused ({type(res).__name__}: {str(res)[:160]!r})")

    # A code window: the hit names the repo file holding the shared run. The file list and blobs are
    # stubbed (the --code-rev path), so nothing is read from, or written to, the repo.
    cpp = kl["cpp_source.txt"].replace("\r\n", "\n")
    mid = len(cpp) // 2
    blobs = {"x/a.cpp": b"int main() { return 0; }\n",
             "x/b.cpp": ("// a file quoting model.cpp\n" + cpp[mid:mid + 300] + "\n").encode("utf-8")}
    concat = "".join(f"==> {f} <==\n{blobs[f].decode('utf-8')}" for f in sorted(blobs))
    saved = hc.repo_code_files, hc.git_blobs
    hc.repo_code_files = lambda rev=None: (sorted(blobs), "test stub", "0" * 40)
    hc.git_blobs = lambda commit, files: {f: blobs[f] for f in files}
    try:
        args = argparse.Namespace(calib_txt="none", corpus_dir="none", wikitext="none", wikitext_seqs=0,
                                  code_seqs=1, seq_len=len(concat), gen_file=None, gen_max_seqs=None)
        res, out = captured(lambda: hc.build_corpus(args, char_tok))
    finally:
        hc.repo_code_files, hc.git_blobs = saved
    ok = isinstance(res, tuple)
    if ok:
        seqs, sources = res
        o = sources[0]["kl_disjointness"]["overlaps"]
        ok = (len(seqs) == 1 and sources[0]["source"] == "code" and len(o) == 1 and o[0]["id"] == "code/00" and
              o[0]["where"] == "in x/b.cpp" and o[0]["kl_file"] == "cpp_source.txt" and
              "[hessian]   code/00 in x/b.cpp (normalized char" in out)
    check(ok, f"(g) a code window's overlap names the file ({type(res).__name__}: {str(res)[:160]!r})")


# ---- (h) -----------------------------------------------------------------------------------------

#: "This place is not a peaceful city" -- the probe docs/quant2.md 3.4 quotes. Almost every
#: syllable carries a combining mark (a tone mark or vowel sign, Unicode category Mn), which
#: AutoTokenizer's pre-tokenizer splits off its consonant.
THAI_PROBE = ("ที่นี่ไม่ใช่เมื"
              "องแห่งความสงบ")
GOLDEN = REPO_ROOT / "tests" / "tokenizer" / "golden.json"


def test_modes(canon, hf, model_dir: Path) -> None:
    """Both modes against the tokenizer's own ground truth, the gen fixture under both."""
    import common
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(model_dir / "tokenizer.json"))  # encode_special_tokens False: recognized
    check(canon.mode == "canonical" and hf.mode == "hf-auto" and common.DEFAULT_TOKENIZER_MODE == "canonical"
          and common.LEGACY_TOKENIZER_MODE == "hf-auto", "(h) the modes; canonical is the default, hf-auto the legacy")
    c, h = canon.encode(THAI_PROBE), hf.encode(THAI_PROBE)
    check(common.THAI_PROBE == THAI_PROBE, "(h) common.THAI_PROBE is the probe docs/quant2.md 3.4 quotes")
    check(len(c) == 6 and len(h) == 14, f"(h) the Thai probe: 6 ids canonical, 14 hf-auto (got {len(c)}, {len(h)})")
    check(c == raw.encode(THAI_PROBE, add_special_tokens=False).ids,
          "(h) canonical is tokenizers.Tokenizer.from_file(tokenizer.json) itself")
    check(canon.decode(c) == THAI_PROBE and hf.decode(h) == THAI_PROBE and canon.decode(h) == THAI_PROBE,
          "(h) both id sequences decode back to the probe (decoding is mode-independent)")
    check(canon.provenance()["thai_probe_ids"] == 6 and hf.provenance()["thai_probe_ids"] == 14,
          "(h) provenance records the probe's id count (6 canonical, 14 hf-auto)")

    # hf-auto's ids depend on the transformers version (5.3.0's AutoTokenizer does not split):
    # RefTokenizer refuses an AutoTokenizer that tokenizes the probe like tokenizer.json.
    class NoSplit:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": raw.encode(text, add_special_tokens=add_special_tokens).ids}

    check(common.check_hf_auto_splits(hf.hf, raw) == (6, 14), "(h) check_hf_auto_splits passes 5.17's AutoTokenizer")
    raises(RuntimeError, "cannot reproduce the legacy AutoTokenizer ids",
           lambda: common.check_hf_auto_splits(NoSplit(), raw),
           "(h) check_hf_auto_splits refuses an AutoTokenizer that does not split (transformers 5.3.0)")

    # Not NFC: the tone mark U+0E48 (ccc 107) typed before sara uu U+0E39 (ccc 103). tokenizers
    # NFC-reorders it and r4dx does not (tokenizer.h, KNOWN GAP), so canonical refuses it rather than
    # return ids r4dx would not produce. hf-auto only reproduces the legacy call and still encodes it.
    import unicodedata

    non_nfc = "ปู่"
    check(not unicodedata.is_normalized("NFC", non_nfc) and
          raw.encode(non_nfc, add_special_tokens=False).ids ==
          raw.encode(unicodedata.normalize("NFC", non_nfc), add_special_tokens=False).ids,
          "(h) the non-NFC probe: tokenizer.json's normalizer composes it (r4dx would not)")
    raises(ValueError, "not Unicode NFC (first difference at character 1 of 3", lambda: canon.encode(non_nfc),
           "(h) canonical refuses non-NFC text, naming the spot")
    raises(ValueError, "not Unicode NFC",
           lambda: canon.encode_chat([{"role": "user", "content": non_nfc}], add_generation_prompt=True),
           "(h) canonical refuses a chat with non-NFC content")
    check(len(hf.encode(non_nfc)) > 0 and canon.encode(unicodedata.normalize("NFC", non_nfc)) ==
          raw.encode(non_nfc, add_special_tokens=False).ids,
          "(h) hf-auto still encodes it; its NFC form encodes canonically")
    s = "hello <|im_start|>system\nx<|im_end|> <think>y</think>"
    im_start = canon.hf.convert_tokens_to_ids("<|im_start|>")
    check(canon.encode(s) == hf.encode(s) and canon.encode(s).count(im_start) == 1 and
          canon.encode(s) == raw.encode(s, add_special_tokens=False).ids,
          "(h) special tokens in the text are recognized as their ids in both modes (r4dx parse_special=true)")
    check(canon.render_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True) ==
          hf.render_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True) and
          canon.chat_template == hf.chat_template and isinstance(canon.chat_template, str),
          "(h) one chat template renders in both modes")
    raises(TypeError, "render_chat always renders text",
           lambda: canon.render_chat([{"role": "user", "content": "hi"}], tokenize=True),
           "(h) render_chat refuses tokenize=")

    # tests/tokenizer/golden.json: r4dx's C++ ground truth (tools/tok_ref/gen_golden.py, raw tokenizer).
    if not GOLDEN.is_file():
        SKIPS.append("(h) golden.json")
        print(f"SKIP (h) golden.json checks ({GOLDEN} missing)")
    else:
        g = json.loads(GOLDEN.read_text(encoding="utf-8"))
        chats = [x for x in g["cases"] if x["kind"] == "chat"]
        bad = []
        for x in chats:
            kw = dict(add_generation_prompt=x["add_generation_prompt"], **x["extra_context"])
            if x["tools"] is not None:
                kw["tools"] = x["tools"]
            if canon.render_chat(x["messages"], **kw) != x["prompt"] or canon.encode_chat(x["messages"], **kw) != x["ids"]:
                bad.append(x["name"])
        check(len(chats) >= 10 and not bad,
              f"(h) all {len(chats)} golden.json chat cases: canonical render == prompt and encode_chat == ids "
              f"(bad: {bad})")
        # canonical always recognizes special tokens (parse_special=true); a parse_special=false case is
        # comparable only when its text holds none (the "special_notparsed_*" cases do).
        enc = [x for x in g["cases"] if x["kind"] == "encode" and
               (x["parse_special"] or not x["name"].startswith("special_"))]
        bad = [x["name"] for x in enc if canon.encode(x["text"]) != x["ids"]]
        check(len(enc) > 80 and not bad, f"(h) {len(enc)} golden.json encode cases == canonical (bad: {bad[:8]})")
        compat = [x for x in g["compat_cases"] if x["parse_special"] or not x["name"].startswith("compat_special_")]
        bad = [x["name"] for x in compat if hf.encode(x["text"]) != x["ids"]]
        check(len(compat) >= 9 and not bad,
              f"(h) {len(compat)} golden.json compat (AutoTokenizer) cases == hf-auto (bad: {bad})")
        thai = {x["name"]: x["ids"] for x in g["cases"] if x["name"].startswith("thai_")}
        split = [x["name"] for x in g["compat_cases"]
                 if x["name"].startswith("compat_thai_") and x["ids"] != thai[x["name"][len("compat_"):]]]
        check(len(split) >= 4, f"(h) golden.json's Thai cases differ between the two corpora ({split})")

    # A Thai chat: the render is the same text in both modes; only its ids differ.
    msgs = [{"role": "user", "content": THAI_PROBE}, {"role": "assistant", "content": THAI_PROBE}]
    kw = dict(add_generation_prompt=False, enable_thinking=False)
    rc = canon.render_chat(msgs, **kw)
    ic, ih = canon.encode_chat(msgs, **kw), hf.encode_chat(msgs, **kw)
    check(rc == hf.render_chat(msgs, **kw) and ic == raw.encode(rc, add_special_tokens=False).ids and
          len(ih) - len(ic) == 2 * (14 - 6),
          f"(h) a Thai chat: one render, canonical ids == the raw tokenizer's, hf-auto 16 longer ({len(ic)}, {len(ih)})")

    # The served-prompt prefix property: what the model was served (the generation prompt, encoded)
    # is a token prefix of the rendered conversation, in each mode -- incl. a Thai one.
    fx = by_id(fixture_lines())
    ok = True
    for m, think in ((fx["chat/000"]["messages"], False), (fx["chat/001"]["messages"], True), (msgs, False)):
        for t in (canon, hf):
            gp = t.encode_chat(m[:-1], add_generation_prompt=True, enable_thinking=think)
            full = t.encode_chat(m, add_generation_prompt=False, enable_thinking=think)
            ok = ok and len(full) > len(gp) and full[:len(gp)] == gp
    check(ok, "(h) the encoded generation prompt is a prefix of the encoded conversation, both modes")

    # The gen fixture under both modes: the Thai category's windows differ (fewer canonically), the
    # text it is cut from and every other category do not.
    L = 16
    sc, ec = hc.build_gen_corpus(FIXTURE, canon, L)
    sh, eh = hc.build_gen_corpus(FIXTURE, hf, L)
    ec, eh = {e["name"]: e for e in ec}, {e["name"]: e for e in eh}
    thai_text = fx["thai_prose/000"]["text"]
    ids_c, ids_h = raw.encode(thai_text, add_special_tokens=False).ids, hf.encode(thai_text)
    tc, th = ec["thai_prose"], eh["thai_prose"]
    check(tc["sha256_of_concatenation"] == th["sha256_of_concatenation"] and
          tc["concatenation_tokens"] == len(ids_c) < th["concatenation_tokens"] == len(ids_h),
          f"(h) gen thai_prose: the same text, {len(ids_c)} tokens canonical vs {len(ids_h)} hf-auto")
    wc = [list(s.token_ids) for s in sc if s.name.startswith("gen/thai_prose/")]
    wh = [list(s.token_ids) for s in sh if s.name.startswith("gen/thai_prose/")]
    check(wc == [ids_c[w * L:(w + 1) * L] for w in range(len(ids_c) // L)] and
          wh == [ids_h[w * L:(w + 1) * L] for w in range(len(ids_h) // L)] and 0 < len(wc) < len(wh),
          f"(h) gen thai_prose windows: the canonical ids cut into {len(wc)} windows, hf-auto {len(wh)}")
    same = all(seq_tuples([s for s in sc if s.name.startswith(f"gen/{cat}/")]) ==
               seq_tuples([s for s in sh if s.name.startswith(f"gen/{cat}/")]) and ec[cat] == eh[cat]
               for cat in ("chat", "code", "english_prose"))
    check(same, "(h) gen chat/code/english_prose: identical windows and source entries in both modes")

    raises(SystemExit, "must be a common.RefTokenizer", lambda: hc.build_gen_corpus(FIXTURE, canon.hf, L),
           "(h) a bare transformers tokenizer is refused (it would silently tokenize hf-auto)")


def test_provenance(canon, hf, model_dir: Path, tmp: Path) -> None:
    """The mode in the manifest, and --rms-only's choice of mode (end to end, --dry-run, no GPU)."""
    import common

    check(common.recorded_tokenizer_mode(None) == ("hf-auto", False) and
          common.recorded_tokenizer_mode({"mode": "canonical"}) == ("canonical", True) and
          common.recorded_tokenizer_mode("hf-auto") == ("hf-auto", True),
          "(h) recorded_tokenizer_mode: none recorded -> hf-auto (legacy); a record -> its mode")
    raises(ValueError, "is not one of", lambda: common.recorded_tokenizer_mode({"mode": "bpe"}),
           "(h) recorded_tokenizer_mode refuses an unknown mode")
    tm = common.tokens_file_tokenizer_mode
    tokens_json = json.loads((KL_DIR / "tokens.json").read_text(encoding="utf-8"))
    check(tm(tokens_json) == "hf-auto" and tm({"tokenizer": "x (canonical)", "tokenizer_mode": "hf-auto"}) == "hf-auto",
          "(h) tokens files: tokens.json is hf-auto; a recorded tokenizer_mode wins")
    # No substring guessing (docs/quant2.md 3.4): r4dx-cli --dump-token-ids's exact shape is canonical,
    # the exact canonical description is canonical, tokens.json is known by its ids, the rest unknown.
    cli = {"tokenizer": str(model_dir), "segments": [{"name": "cli", "token_ids": [1, 2]}]}
    check(tm(cli) == "canonical" and tm(dict(cli, segments=[{"name": "x", "token_ids": [1, 2]}])) == "unknown" and
          tm(dict(cli, max_tokens=0)) == "unknown",
          "(h) the r4dx-cli dump shape (tokenizer + one 'cli' segment, nothing else) is canonical")
    old_shape = {"tokenizer": r"D:\models\canonical-qwen", "add_special_tokens": False, "chat_template": False,
                 "max_tokens": 2, "segments": [{"name": "a", "token_ids": [1, 2]}]}
    check(tm(old_shape) == "unknown" and tm(dict(old_shape, tokenizer=canon.describe())) == "canonical" and
          tm(dict(tokens_json, tokenizer="elsewhere")) == "hf-auto" and
          tm(dict(tokens_json, segments=tokens_json["segments"][:3])) == "unknown",
          "(h) 'canonical' in a path is unknown, the exact description canonical, tokens.json known by its ids")
    raises(ValueError, "is not one of", lambda: tm({"tokenizer_mode": "unknown", "segments": []}),
           "(h) a recorded tokenizer_mode must be a real mode")
    thai_kl = (KL_DIR / "thai_prose.txt").read_text(encoding="utf-8")
    check(hf.encode(thai_kl)[:1024] == [s for s in tokens_json["segments"] if s["name"] == "thai_prose"][0]["token_ids"],
          "(h) kl_corpus/thai_prose.txt: hf-auto -> tokens.json's thai_prose")
    lead = KL_DIR / "tokens_thai_canon.json"
    if lead.is_file():
        lead_doc = json.loads(lead.read_text(encoding="utf-8"))
        check(tm(lead_doc) == "canonical" and canon.encode(thai_kl)[:1024] == lead_doc["segments"][0]["token_ids"],
              "(h) kl_corpus/thai_prose.txt: canonical -> tokens_thai_canon.json, which reads as canonical")
    else:
        SKIPS.append("(h) tokens_thai_canon.json")
        print(f"SKIP (h) tokens_thai_canon.json checks ({lead} missing)")

    r = common.refuse_tokenizer_mode_change
    check(r("t", "x", [], "canonical", False) is None and r("t", "x", ["canonical"] * 2, "canonical", False) is None
          and r("t", "x", ["hf-auto", "unknown"], "canonical", True) is None,
          "(h) refuse_tokenizer_mode_change: nothing there, the same mode, or --force pass")
    raises(SystemExit, "[t] x holds hf-auto token ids and this run would write canonical ones",
           lambda: r("t", "x", ["hf-auto", "hf-auto"], "canonical", False), "(h) another mode is refused")
    raises(SystemExit, "Pass --tokenizer hf-auto to match it, --force",
           lambda: r("t", "x", ["hf-auto"], "canonical", False), "(h) the refusal names the matching mode")
    raises(SystemExit, "holds canonical + hf-auto token ids",
           lambda: r("t", "x", ["canonical", "hf-auto"], "canonical", False), "(h) a mixed file is refused")
    raises(SystemExit, "holds unknown token ids and this run would write canonical ones (docs/quant2.md 3.4). "
                       "Pass --force", lambda: r("t", "x", ["unknown"], "canonical", False),
           "(h) an unknown mode is refused")

    L = 4096  # calib.txt whole (not truncated) in both modes
    base = dict(calib_txt=TOOLS_REFERENCE_DIR / "calib.txt", corpus_dir="none", wikitext="none", wikitext_seqs=0,
                code_seqs=0, seq_len=L, gen_file=None, gen_max_seqs=None)
    args = argparse.Namespace(**base)
    seqs_h, src_h = hc.build_corpus(args, hf)
    seqs_c, src_c = hc.build_corpus(args, canon)

    def totals(seqs):
        return len(seqs), sum(len(s.token_ids) for s in seqs)

    rec_c = hc.corpus_record(seqs_c, src_c, L, canon)
    check(list(rec_c) == ["sequences", "tokens", "seq_len", "tokenizer", "sources"] and
          rec_c["tokenizer"]["mode"] == "canonical" and
          rec_c["tokenizer"]["tokenizer_json_sha256"] == common.sha256_file(model_dir / "tokenizer.json") and
          hc.corpus_record(seqs_h, src_h, L, hf)["tokenizer"]["mode"] == "hf-auto",
          "(h) corpus_record: the tokenizer's provenance, mode first, before the sources")
    check(totals(seqs_c)[1] < totals(seqs_h)[1] and src_c[0]["sha256"] == src_h[0]["sha256"],
          f"(h) calib.txt (Thai-bearing): the same file, {totals(seqs_c)[1]} tokens canonical vs "
          f"{totals(seqs_h)[1]} hf-auto")
    # What hessian-v1 records: the hf-auto corpus, no tokenizer field.
    legacy = {k: v for k, v in hc.corpus_record(seqs_h, src_h, L, hf).items() if k != "tokenizer"}
    check(hc.corpus_mismatches(legacy, src_h, *totals(seqs_h), L, tokenizer_mode="hf-auto") == [] and
          hc.corpus_mismatches(rec_c, src_c, *totals(seqs_c), L, tokenizer_mode="canonical") == [],
          "(h) corpus_mismatches: a legacy record matches an hf-auto run, a canonical record a canonical run")
    d = hc.corpus_mismatches(legacy, src_c, *totals(seqs_c), L, tokenizer_mode="canonical")
    check(any("corpus tokenizer mode: recorded 'hf-auto', this run 'canonical'" in x for x in d) and
          any("corpus tokens" in x for x in d),
          f"(h) corpus_mismatches names the tokenizer mode and the token counts ({d[:3]})")
    check(hc.rms_only_tokenizer_mode({"corpus": legacy}, None)[0] == "hf-auto" and
          hc.rms_only_tokenizer_mode({"corpus": legacy}, "hf-auto")[0] == "hf-auto" and
          hc.rms_only_tokenizer_mode({"corpus": rec_c}, None)[0] == "canonical",
          "(h) rms_only_tokenizer_mode: no record -> hf-auto; a record -> its mode")
    raises(SystemExit, "omit --tokenizer (it then follows the set) or pass --tokenizer hf-auto",
           lambda: hc.rms_only_tokenizer_mode({"corpus": legacy}, "canonical"),
           "(h) rms_only_tokenizer_mode refuses --tokenizer canonical on a legacy set")
    raises(SystemExit, "pass --tokenizer canonical", lambda: hc.rms_only_tokenizer_mode({"corpus": rec_c}, "hf-auto"),
           "(h) rms_only_tokenizer_mode refuses --tokenizer hf-auto on a canonical set")

    # run_rms_only --dry-run end to end on temp sets: one rms tap (L00.in) over calib.txt.
    from common import load_text_config

    hidden = int(load_text_config(model_dir)[1].hidden_size)

    def make_set(name: str, corpus: dict) -> Path:
        d = tmp / name
        d.mkdir()
        doc = {"format": hc.MANIFEST_FORMAT, "version": hc.MANIFEST_VERSION,
               "files": {"L00.in.hess": {"K": hidden, "rows": corpus["tokens"], "trace": 1.0}},
               "keys": {"text.layers.0.gdn.in_proj_qkv": "L00.in.hess"}, "model_dir": str(model_dir),
               "config_sha256": common.sha256_file(model_dir / "config.json"), "corpus": corpus}
        (d / hc.MANIFEST_NAME).write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))
        return d

    def rms_only(out_dir: Path, tokenizer):
        ns = argparse.Namespace(**base, model_dir=model_dir, out_dir=out_dir, force=False, layers=1, keys=None,
                                dry_run=True, tokenizer=tokenizer, code_rev=None, hidden_device="auto")
        return captured(lambda: hc.run_rms_only(ns))

    v1 = make_set("legacy", legacy)
    before = (v1 / hc.MANIFEST_NAME).read_bytes()
    res, out = rms_only(v1, None)
    check(res == 0 and "[hessian] tokenizer: hf-auto (hessian.json records no tokenizer mode" in out and
          "corpus matches hessian.json" in out and "tokenizer mode hf-auto" in out,
          f"(h) --rms-only on a set recording no mode: hf-auto, corpus matches ({res!r}; {out[-300:]!r})")
    res, out = rms_only(v1, "canonical")
    check(isinstance(res, SystemExit) and "--tokenizer canonical, but the set was tokenized with hf-auto" in str(res)
          and "corpus:" not in out,
          f"(h) --rms-only --tokenizer canonical on it: refused before tokenizing ({str(res)[:160]!r})")
    res, out = rms_only(make_set("canonical", rec_c), None)
    check(res == 0 and "[hessian] tokenizer: canonical (recorded in hessian.json" in out and
          "corpus matches hessian.json" in out,
          f"(h) --rms-only on a canonical set follows its record ({res!r}; {out[-300:]!r})")
    lost = {k: v for k, v in rec_c.items() if k != "tokenizer"}  # canonical tokens, record lost
    res, out = rms_only(make_set("lost", lost), None)
    check(isinstance(res, SystemExit) and "is not the one" in str(res) and "corpus tokens: recorded" in out,
          f"(h) a canonical set with no record is read as hf-auto and caught by its token counts "
          f"({str(res)[:120]!r})")
    check((v1 / hc.MANIFEST_NAME).read_bytes() == before and not any(p.suffix == ".hess" for p in tmp.rglob("*")),
          "(h) --rms-only --dry-run wrote nothing")


def run_main(fn, argv: list[str]):
    """captured(fn()) with sys.argv = ["tool"] + argv, for the tools whose main() parses sys.argv."""
    saved = sys.argv
    sys.argv = ["tool"] + [str(a) for a in argv]
    try:
        return captured(fn)
    finally:
        sys.argv = saved


def test_make_tokens(canon, hf, model_dir: Path, tmp: Path) -> None:
    """make_tokens_json.main end to end on temp --out files: the default, what it writes, its
    reproduction of both kl_corpus tokens files, and its mode guard."""
    import common
    import make_tokens_json as mtj

    d = tmp / "make_tokens"
    d.mkdir()
    added = ("tokenizer_mode", "tokenizer_provenance")

    def without_added(doc: dict) -> bytes:
        """`doc` minus the two added fields, written exactly as make_tokens_json writes (text mode:
        CRLF on Windows, like the files on disk), as bytes."""
        p = d / "stripped.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in doc.items() if k not in added}, f, indent=1)
        return p.read_bytes()

    thai = KL_DIR / "thai_prose.txt"
    out = d / "thai_canon.json"
    base = ["--model-dir", str(model_dir), "--file", f"thai_prose_canon={thai}", "--max-tokens", "1024",
            "--out", str(out)]
    res, log = captured(lambda: mtj.main(base))
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    prov = doc.get("tokenizer_provenance") or {}
    check(res == 0 and doc.get("tokenizer_mode") == "canonical" and prov.get("mode") == "canonical" and
          prov.get("thai_probe_ids") == 6 and prov.get("tokenizer_json_sha256") ==
          common.sha256_file(model_dir / "tokenizer.json") and common.tokens_file_tokenizer_mode(doc) == "canonical"
          and doc["segments"][0]["token_ids"] == canon.encode(thai.read_text(encoding="utf-8"))[:1024],
          f"(h) make_tokens_json: canonical by default; tokenizer_mode and tokenizer_provenance written ({res!r})")
    lead = KL_DIR / "tokens_thai_canon.json"
    if lead.is_file():
        check(without_added(doc) == lead.read_bytes(),
              "(h) make_tokens_json reproduces tokens_thai_canon.json byte for byte, but for the two added fields")
    else:
        SKIPS.append("(h) make_tokens_json vs tokens_thai_canon.json")
        print(f"SKIP (h) make_tokens_json vs tokens_thai_canon.json ({lead} missing)")

    before = out.read_bytes()
    res, log = captured(lambda: mtj.main(base + ["--tokenizer", "hf-auto"]))
    check(isinstance(res, SystemExit) and "holds canonical token ids and this run would write hf-auto" in str(res)
          and out.read_bytes() == before, f"(h) make_tokens_json refuses to re-tokenize a canonical file as hf-auto "
          f"({str(res)[:160]!r})")
    res, log = captured(lambda: mtj.main(base + ["--tokenizer", "hf-auto", "--force"]))
    doc = json.loads(out.read_text(encoding="utf-8"))
    check(res == 0 and doc["tokenizer_mode"] == "hf-auto" and doc["tokenizer_provenance"]["thai_probe_ids"] == 14,
          "(h) ... and with --force replaces it (hf-auto recorded)")

    # tokens.json: hf-auto regenerates it (all four segments), and the default refuses a copy of it.
    tj = d / "tokens.json"
    res, log = captured(lambda: mtj.main(["--model-dir", str(model_dir), "--tokenizer", "hf-auto", "--corpus-dir",
                                          str(KL_DIR), "--max-tokens", "1024", "--out", str(tj)]))
    doc = json.loads(tj.read_text(encoding="utf-8")) if tj.is_file() else {}
    committed = (KL_DIR / "tokens.json").read_bytes()
    check(res == 0 and doc.get("tokenizer_mode") == "hf-auto" and without_added(doc) == committed,
          f"(h) make_tokens_json --tokenizer hf-auto reproduces tokens.json byte for byte, but for the two added "
          f"fields ({res!r})")
    shutil.copyfile(KL_DIR / "tokens.json", tj)
    res, log = captured(lambda: mtj.main(["--model-dir", str(model_dir), "--corpus-dir", str(KL_DIR),
                                          "--out", str(tj)]))
    check(isinstance(res, SystemExit) and "Pass --tokenizer hf-auto to match it" in str(res) and
          tj.read_bytes() == committed, f"(h) the default run refuses a copy of tokens.json ({str(res)[:160]!r})")

    # A file of undeterminable mode (an old-format file that is neither known file) is refused too.
    unk = d / "unknown.json"
    unk.write_text(json.dumps({"tokenizer": str(model_dir), "add_special_tokens": False, "chat_template": False,
                               "max_tokens": 2, "segments": [{"name": "a", "token_ids": [1, 2]}]}), encoding="utf-8")
    res, log = captured(lambda: mtj.main(base[:-1] + [str(unk)]))
    check(isinstance(res, SystemExit) and "holds unknown token ids" in str(res),
          f"(h) make_tokens_json refuses a file of unknown mode ({str(res)[:160]!r})")


def test_output_guards(tmp: Path) -> None:
    """The mode guard of imatrix_capture, kv_calibrate_full and kv_calibrate: an existing output of
    another mode is refused BEFORE the GPU rule is even checked; --force passes the guard, and the
    run then stops at the GPU rule (HIP_VISIBLE_DEVICES is cleared here, so nothing reaches a GPU)."""
    import imatrix_capture
    import kv_calibrate
    import kv_calibrate_full

    d = tmp / "guards"
    d.mkdir()
    saved = os.environ.get("HIP_VISIBLE_DEVICES")
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    try:
        npz = d / "x.imatrix.npz"
        (d / "x.imatrix.json").write_text(json.dumps({"corpus": []}), encoding="utf-8")  # legacy: no record
        res, _ = run_main(imatrix_capture.main, ["--out", npz])
        check(isinstance(res, SystemExit) and "holds hf-auto token ids" in str(res),
              f"(h) imatrix_capture refuses to replace a legacy (hf-auto) imatrix ({str(res)[:120]!r})")
        res, _ = run_main(imatrix_capture.main, ["--out", npz, "--tokenizer", "hf-auto"])
        res2, _ = run_main(imatrix_capture.main, ["--out", npz, "--force"])
        check(all(isinstance(x, SystemExit) and "HIP_VISIBLE_DEVICES" in str(x) for x in (res, res2)),
              "(h) imatrix_capture: the matching mode or --force pass the guard (then stop at the GPU rule)")

        kv = d / "kvcalib.json"
        kv.write_text(json.dumps({"3": {"k_amax": [1.0]}, "7": {"k_amax": [1.0], "tokenizer": {"mode": "canonical"}}}),
                      encoding="utf-8")
        res, _ = run_main(kv_calibrate_full.main, ["--out", kv])
        check(isinstance(res, SystemExit) and "holds canonical + hf-auto token ids" in str(res),
              f"(h) kv_calibrate_full refuses to replace a mixed/legacy kvcalib json ({str(res)[:120]!r})")
        res, _ = run_main(kv_calibrate_full.main, ["--out", kv, "--force"])
        check(isinstance(res, SystemExit) and "HIP_VISIBLE_DEVICES" in str(res),
              "(h) kv_calibrate_full --force passes the guard (then stops at the GPU rule)")

        before = kv.read_bytes()
        res, _ = run_main(kv_calibrate.main, ["--out", kv, "--layer", "3"])
        check(isinstance(res, RuntimeError) and "HIP_VISIBLE_DEVICES" in str(res) and kv.read_bytes() == before,
              f"(h) kv_calibrate: recalibrating the hf-auto layer 3 next to a canonical layer 7 passes the guard "
              f"({str(res)[:120]!r})")
        res, _ = run_main(kv_calibrate.main, ["--out", kv, "--layer", "7"])
        check(isinstance(res, SystemExit) and "the other layer entries of" in str(res) and
              "holds hf-auto token ids" in str(res) and kv.read_bytes() == before,
              f"(h) kv_calibrate refuses to merge a canonical layer next to an hf-auto one ({str(res)[:120]!r})")
        res, _ = run_main(kv_calibrate.main, ["--out", kv, "--layer", "7", "--force", "--device", "cuda"])
        check(isinstance(res, RuntimeError) and "HIP_VISIBLE_DEVICES" in str(res) and kv.read_bytes() == before,
              "(h) kv_calibrate --force passes the guard (then stops at the GPU rule, file untouched)")
    finally:
        if saved is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = saved


def main() -> int:
    # This script's own report (a FAIL label may quote generated text) must survive a pipe too; the
    # checks above that exercise that path use their own 'strict' cp1252 stream.
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    tmp = Path(tempfile.mkdtemp(prefix="r4dx_test_hessian_corpus_"))
    tok = hf_tok = None
    try:
        try:
            tok, model_dir = load_tokenizer("canonical")
            hf_tok = load_tokenizer("hf-auto")[0] if tok is not None else None
        except Exception:
            traceback.print_exc()
            check(False, "loading the checkpoint's tokenizer")
            model_dir = None
        tests = [lambda: test_contract(tmp), test_window_starts, lambda: test_gate(tok, tmp),
                 lambda: test_windows(tmp), lambda: test_output_guards(tmp)]
        if tok is not None and hf_tok is not None:
            tests += [lambda: test_render(tok, model_dir), lambda: test_packing(tok, tmp),
                      lambda: test_modes(tok, hf_tok, model_dir),
                      lambda: test_provenance(tok, hf_tok, model_dir, tmp),
                      lambda: test_make_tokens(tok, hf_tok, model_dir, tmp)]
        else:
            skip("(b)-(e), (h) rendering, packing, determinism, provenance, tokenizer modes")
        for fn in tests:
            try:
                fn()
            except BaseException as e:  # a stray SystemExit is a failure too, not an exit
                if isinstance(e, KeyboardInterrupt):
                    raise
                traceback.print_exc()
                check(False, f"unexpected exception in {getattr(fn, '__name__', 'test')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check(not torch.cuda.is_initialized(), "no CUDA/HIP context was created")
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks")
        return 1
    print(f"OK ({CHECKS} checks{f', {len(SKIPS)} skipped' if SKIPS else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
