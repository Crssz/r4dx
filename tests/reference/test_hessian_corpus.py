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
    build_corpus' text sources."""

    chat_template = None

    def __call__(self, text, add_special_tokens=False, verbose=True):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(map(chr, ids))


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


def load_tokenizer():
    from common import DEFAULT_MODEL_DIR

    if not (DEFAULT_MODEL_DIR / "tokenizer.json").exists():
        return None, DEFAULT_MODEL_DIR
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_DIR)), DEFAULT_MODEL_DIR


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
    gp0 = tok.apply_chat_template(m[:-1], chat_template=tpl, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
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
    gp1 = tok.apply_chat_template(m[:-1], chat_template=tpl, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=True)
    check(gp1.endswith("<|im_start|>assistant\n<think>\n") and
          r1 == gp1 + m[3]["reasoning_content"] + "\n</think>\n\n" + m[3]["content"] + "<|im_end|>\n",
          "(b) thinking chat == the thinking generation prompt (which ends in <think>\\n) + the generated turn")

    # What preserve_thinking=false would do (neither the generator nor this script sets it): only the
    # earlier turn loses its thought; the final one keeps it. check_rendered_chat refuses that render,
    # and one that lost the final thought.
    np_ = tok.apply_chat_template(m, chat_template=tpl, tokenize=False, add_generation_prompt=False,
                                  enable_thinking=True, preserve_thinking=False)
    check(m[1]["reasoning_content"] not in np_ and np_.endswith(final),
          "(b) preserve_thinking=false: the earlier thought is dropped, the final turn's is kept")
    raises(SystemExit, "assistant turn 1/2", lambda: hc.check_rendered_chat(c1, np_),
           "(b) check_rendered_chat refuses a render that dropped an earlier thought")
    lost = r1[:len(r1) - len(final)] + f"<|im_start|>assistant\n{m[3]['content']}<|im_end|>\n"
    raises(SystemExit, "assistant turn 2/2", lambda: hc.check_rendered_chat(c1, lost),
           "(b) check_rendered_chat refuses a render that dropped the final thought")
    raises(SystemExit, "at the end of the conversation", lambda: hc.check_rendered_chat(c1, r1 + "x"),
           "(b) check_rendered_chat refuses anything after the final turn")

    # Tokens: exactly the template text's, one id per control token, no BOS, and what transformers'
    # own apply_chat_template(tokenize=True) gives.
    cid = {t: tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")}
    for sample, text in ((c0, r0), (c1, r1)):
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
        ref = tok.apply_chat_template(sample["messages"], chat_template=tpl, tokenize=True, return_dict=False,
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
        t = s["text"] if s["format"] == "raw" else tok.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=s["enable_thinking"])
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
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
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


def main() -> int:
    # This script's own report (a FAIL label may quote generated text) must survive a pipe too; the
    # checks above that exercise that path use their own 'strict' cp1252 stream.
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    tmp = Path(tempfile.mkdtemp(prefix="r4dx_test_hessian_corpus_"))
    tok = None
    try:
        try:
            tok, model_dir = load_tokenizer()
        except Exception:
            traceback.print_exc()
            check(False, "loading the checkpoint's tokenizer")
            model_dir = None
        tests = [lambda: test_contract(tmp), test_window_starts, lambda: test_gate(tok, tmp),
                 lambda: test_windows(tmp)]
        if tok is not None:
            tests += [lambda: test_render(tok, model_dir), lambda: test_packing(tok, tmp)]
        else:
            skip("(b)-(e) rendering, packing, determinism, provenance")
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
