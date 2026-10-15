"""tools/reference/lfm2/gen_samples_server.py  (L3-4: the LFM2.5 trellis calibration corpus, token-id JSONL)

Trellis calibration data for LFM2.5-24B-A2B comes from the model itself, generated through a RUNNING
r4dx-server (bf16, TP=2, `--dialect lfm2`; OpenAI API at http://127.0.0.1:8091/v1/chat/completions) over every
prompt of tools/quant2/corpus_v2_prompts.json (385 entries: thai_prose, english_prose, chat, code,
multilingual). One request at a time (the server is single-stream), non-streaming, with retry. The output is the
SAME token-id JSONL as tools/reference/gemma/gen_samples.py, which `hessian_capture.py --arch lfm2_moe
--gen-file` reads through gemma.capture.build_tokens_corpus (fields id, category, rejected, token_ids,
kl_token_overlap are the contract; the rest is provenance):

    id, category, format, enable_thinking (always true: LFM always thinks), prompt_enable_thinking,
    messages (prompt + generated assistant turns, "content" + "reasoning_content"), text (raw entries: the answer;
    chat: null), finish_reason, prompt_tokens, completion_tokens (summed server usage), seed, temperature, top_k,
    top_p, repetition_penalty, max_tokens (the budget actually sent), rejected, reject_reason,
    truncated_thought_turn, prompt_sha256, turn_seeds, turn_finish_reasons, turns [{prompt_len,
    prompt_tokens_server, completion_ids_len, completion_tokens_server, finish_reason}], token_ids, calib_kind
    ("chat_stream"), kl_token_overlap, base_id, pass, elapsed_s

How the token ids are obtained (the server does not return ids, and its answer is text split into
`reasoning_content` and `content`):

  1. prompt ids  = encode(chat_template(messages_so_far, add_generation_prompt=True)). The LFM template opens
     the thought itself (the generation prompt ends with `<|im_start|>assistant\\n<think>`, one added token), and
     emits `<|startoftext|>` (BOS) at the start; tokenizer = tokenizer.json of the model directory through
     transformers AutoTokenizer, add_special_tokens=False (common_lfm2.Lfm2RefTokenizer). usage.prompt_tokens of
     EVERY response is compared with len(prompt ids): on the first request of a run a mismatch aborts (exit 3: the
     server's prompt is not the one rendered here, so the ids would not be what the model saw); later turns that
     disagree are rejected ("prompt id mismatch").
  2. completion ids = encode(reasoning + "</think>" + content) + [<|im_end|>]  (the model's thought, the closing
     tag, the answer, end of turn; the same text the template renders for that assistant message, so
     prompt_ids + completion_ids == encode(chat_template(messages + [assistant]))[:-1 newline] - checked in
     --dry-run on every prompt and by the unit test). A turn cut by max_tokens (finish_reason "length") has no
     end-of-turn token; a turn cut inside its thought has no `</think>` either (what the model produced).
  3. token_ids = the FINAL turn's prompt ids + its completion ids (calib_kind "chat_stream"; earlier turns'
     thoughts are stripped by the template, exactly as at serving time).

Known, documented deviation from "the exact ids the model produced": the server's text split trims whitespace
(reasoning_content is stripped at both ends, blank lines right after `</think>` are dropped), so a thought's
leading/trailing whitespace and a "\\n\\n" the model wrote after `</think>` are not recoverable; the stream holds the
canonical template form (what the model sees when the same turn is replayed in a later request). BPE
re-tokenization of generated text can also differ from the sampled ids in rare places. Both are measured:
every turn records the rendered completion length against usage.completion_tokens (the manifest summarizes the
delta; usage may or may not count the end-of-turn token).

Raw-text entries: LFM2.5 always thinks (the template ignores enable_thinking), so raw prose / code entries are
generated and captured in chat format too (the thought precedes the answer in token_ids). `format` keeps the
prompt file's value; the rejection checks treat every entry as a chat turn (no 200-character minimum: a raw entry
whose thought exhausts max_tokens is a kept truncated thought, as in gen_corpus).

Sampling (user decision 2026-10-14): temperature 0.7, top_p 0.95, top_k 50 and repetition_penalty 1.05 (the
server's own LFM defaults, sent explicitly so the record is exact; --top-k / --repetition-penalty change them),
`seed` = sample_seed(sample id, turn) (depends on the sample id and turn only). max_tokens per turn = the prompt
entry's own budget capped at --max-tokens-cap (3072; thinking+answer share it); --min-max-tokens raises small
budgets (default 0 = off).

--passes N: N sampling passes over the prompt file; pass k >= 1 has sample id "<id>#<k>" (a different seed:
the seed hashes the sample id), pass 0 keeps the plain id. Everything else (prompt_sha256) is per prompt.

Rejection (the line is still written so a resume does not redo it): tools/quant2/gen_corpus.check_turn (empty
output, think tags aside, degenerate repetition), "kl overlap" by text (gen_corpus.kl_reason: 48 characters of
the kl_corpus *.txt files and of chat_lfm2.json's segment texts) or by ids (a 32-token run of tokens_lfm2*.json
or chat_lfm2.json, checked on the whole token_ids so the capture's own check can never fire), and "prompt id
mismatch". Resume: `--out` is appended, flushed and fsync'ed per sample; ids already in the file are skipped; a
changed prompt (prompt_sha256) or an unknown id stops the run; a torn last line is truncated.

Usage (one process; the server must be up with the lfm2 model and `--dialect lfm2`):

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\gen_samples_server.py --dry-run --passes 2
    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\gen_samples_server.py `
        --server http://127.0.0.1:8091 --out E:\\models\\r4dx\\lfm2\\corpus\\samples.jsonl --limit 3   # smoke
    ... same without --limit (add --passes 2 for the second pass); re-run the same command to resume.

`--dry-run` needs no server and no GPU (tokenizer + template + budget). `--merge a.jsonl b.jsonl --out x.jsonl
[--require-complete]` merges sample files (sharding is not useful for one single-stream server, but a merge of
a pass-1 file and a pass-2 file is). Exit codes: 0 ok, 1 transport failure, 2 repeated refusals / bad input,
3 prompt-id mismatch on the first request, 130 interrupted.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm2.common_lfm2 import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    IM_END_ID,
    KL_CORPUS_DIR,
    Lfm2RefTokenizer,
    sha256_bytes,
    sha256_file,
)

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "quant2" / "corpus_v2_prompts.json"
sys.path.insert(0, str(PROMPTS_PATH.parent))
import gen_corpus as gc  # noqa: E402  (tools/quant2: stdlib only)

SEED_DOMAIN = "r4dx-lfm2-corpus-v1"
KL_TOKEN_WINDOW = 32
KL_TOKEN_FILES = ("tokens_lfm2.json", "tokens_lfm2_long.json", "chat_lfm2.json")
KL_TEXT_JSON = "chat_lfm2.json"
MANIFEST_NAME = "gen_manifest.json"
DEFAULT_SERVER = "http://127.0.0.1:8091"
DEFAULT_OUT = Path(os.environ.get("R4DX_LFM2_OUT_DIR", r"E:\models\r4dx\lfm2")) / "corpus" / "samples.jsonl"
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05}
TARGET_TOKENS = 1_500_000
MAX_TOKENS_CAP = 3072
MAX_CONSECUTIVE_FAILURES = 3
CLOSE_TAG = "</think>"


class PromptIdMismatch(Exception):
    pass


def sample_seed(sample_id: str, turn: int = 0) -> int:
    d = hashlib.sha256(f"{SEED_DOMAIN}:{sample_id}#{turn}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") & 0x7FFFFFFF


# --------------------------------------------------------------------------------------------
# Jobs (prompt entries x passes)
# --------------------------------------------------------------------------------------------


def make_jobs(order: list[dict], passes: int) -> list[dict]:
    jobs = []
    for p in range(passes):
        for e in order:
            jobs.append({"sample_id": e["id"] if p == 0 else f"{e['id']}#{p}", "pass": p, "entry": e})
    return jobs


def budget_of(entry: dict, cap: int, floor: int) -> int:
    return min(max(entry["max_tokens"], floor), cap) if cap else max(entry["max_tokens"], floor)


# --------------------------------------------------------------------------------------------
# KL held-out corpus: token runs and chat texts
# --------------------------------------------------------------------------------------------


def kl_token_index(kl_dir: Path, window: int = KL_TOKEN_WINDOW) -> set:
    runs = set()
    for fn in KL_TOKEN_FILES:
        p = Path(kl_dir) / fn
        if not p.is_file():
            continue
        for seg in json.loads(p.read_text(encoding="utf-8"))["segments"]:
            ids = seg["token_ids"]
            for i in range(len(ids) - window + 1):
                runs.add(tuple(ids[i:i + window]))
    return runs


def kl_token_overlap(ids: list[int], runs: set, window: int = KL_TOKEN_WINDOW) -> int:
    if not runs:
        return 0
    return sum(tuple(ids[i:i + window]) in runs for i in range(len(ids) - window + 1))


def kl_text_shingles(kl_dir: Path) -> dict[str, str]:
    """gen_corpus' 48-char shingles of kl_corpus/*.txt plus those of chat_lfm2.json's segment `text`."""
    shingles, _ = gc.kl_shingle_index(kl_dir)
    p = Path(kl_dir) / KL_TEXT_JSON
    if p.is_file():
        for seg in json.loads(p.read_text(encoding="utf-8"))["segments"]:
            t = gc._collapse(seg.get("text") or "")
            for i in range(len(t) - gc.KL_WINDOW + 1):
                shingles.setdefault(t[i:i + gc.KL_WINDOW], KL_TEXT_JSON)
    return shingles


# --------------------------------------------------------------------------------------------
# Rendering: messages -> ids
# --------------------------------------------------------------------------------------------


def sendable(messages: list[dict]) -> list[dict]:
    """The messages as sent to the server and as rendered by the template (assistant turns carry their thought)."""
    out = []
    for m in messages:
        d = {"role": m["role"], "content": m["content"]}
        if m["role"] == "assistant" and m.get("reasoning_content") is not None:
            d["reasoning_content"] = m["reasoning_content"]
        out.append(d)
    return out


def prompt_ids_of(tok, messages: list[dict]) -> list[int]:
    return tok.encode_chat(sendable(messages), add_generation_prompt=True)


def completion_text(reasoning: str, content: str, finish: str | None) -> str:
    """The text the model produced after the generation prompt's open `<think>`: thought, close tag, answer.
    A turn cut inside its thought (length, no answer) has no close tag."""
    if finish == "length" and not content.strip():
        return reasoning
    return reasoning + CLOSE_TAG + content


def completion_ids_of(tok, reasoning: str, content: str, finish: str | None) -> list[int]:
    ids = tok.encode(completion_text(reasoning, content, finish))
    if finish == "stop":
        ids.append(IM_END_ID)
    return ids


def template_selfcheck(tok, messages: list[dict]) -> str | None:
    """Render `messages` + a synthetic answer with the full template and compare with prompt ids + completion ids.
    None when they agree, else the problem."""
    p_ids = prompt_ids_of(tok, messages)
    r, c = "Let me think about it.", "The answer."
    c_ids = completion_ids_of(tok, r, c, "stop")
    full = tok.render_chat(sendable(messages + [{"role": "assistant", "content": c, "reasoning_content": r}]))
    if not full.endswith("<|im_end|>\n"):
        return "full template render does not end with <|im_end|>\\n"
    f_ids = tok.encode(full[:-1])
    if f_ids != p_ids + c_ids:
        return "prompt ids + completion ids != ids of the full template render"
    if p_ids[0] != tok.bos_id or p_ids[1] == tok.bos_id:
        return "prompt does not start with exactly one BOS"
    return None


# --------------------------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------------------------


def request_body(messages: list[dict], max_tokens: int, seed: int, sampling: dict) -> dict:
    return {"messages": sendable(messages), "max_tokens": max_tokens, "seed": seed, "stream": False, **sampling}


def post(client: gc.Client, body: dict, retries: int, wait: float, log=print) -> dict:
    """POST /v1/chat/completions; gc.Client retries once itself, this adds `retries` rounds with growing waits."""
    for a in range(1, retries + 1):
        try:
            return client.request("POST", "/v1/chat/completions", body)
        except gc.TransportFailed:
            if a == retries:
                raise
            log(f"[gen_samples_server] transport failure, round {a}/{retries}; waiting {wait * a:g} s")
            time.sleep(wait * a)
    raise AssertionError("unreachable")


def generate_sample(job: dict, client: gc.Client, tok, sampling: dict, kl_shingles: dict, kl_runs: set,
                    cap: int, floor: int, retries: int, wait: float, state: dict, log=print) -> dict:
    entry = job["entry"]
    sid = job["sample_id"]
    max_tokens = budget_of(entry, cap, floor)
    messages = [dict(m) for m in entry["messages"]]
    followups = entry.get("followups", [])
    turns, seeds, finishes = [], [], []
    prompt_tokens = completion_tokens = 0
    reason, truncated_turn, content, p_ids, c_ids = None, None, "", [], []
    model = None
    t0 = time.monotonic()
    for turn in range(1 + len(followups)):
        if turn > 0:
            messages.append({"role": "user", "content": followups[turn - 1]})
        p_ids = prompt_ids_of(tok, messages)
        seed = sample_seed(sid, turn)
        resp = post(client, request_body(messages, max_tokens, seed, sampling), retries, wait, log)
        msg, finish = gc._parse_choice(resp)
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content")
        reasoning = rc if isinstance(rc, str) else ""
        usage = resp.get("usage") or {}
        p_srv = usage.get("prompt_tokens")
        c_srv = usage.get("completion_tokens")
        if p_srv is not None and int(p_srv) != len(p_ids):
            detail = (f"server prompt_tokens {p_srv} != {len(p_ids)} ids rendered here for {sid} turn {turn + 1}")
            if not state.get("verified") and state.get("verify", True):
                raise PromptIdMismatch(detail)
        else:
            detail = None
        state["verified"] = True
        messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
        c_ids = completion_ids_of(tok, reasoning, content, finish)
        prompt_tokens += int(p_srv or 0)
        completion_tokens += int(c_srv or 0)
        seeds.append(seed)
        finishes.append(finish)
        model = resp.get("model", model)
        t = resp.get("timings") or {}
        turns.append({"prompt_len": len(p_ids), "prompt_tokens_server": p_srv,
                      "completion_ids_len": len(c_ids), "completion_tokens_server": c_srv,
                      "finish_reason": finish, "timings": {k: t[k] for k in gc.TIMING_KEYS if k in t}})
        # every entry is judged as a chat turn: LFM always thinks (see the module docstring)
        reason = (gc.check_turn("chat", True, turn, content, reasoning, True, finish)
                  or gc.kl_reason(turn, content, reasoning, kl_shingles)
                  or (f"prompt id mismatch: {detail}" if detail else None))
        if reason:
            break
        if gc.is_truncated_thought(True, content, reasoning, finish):
            truncated_turn = turn + 1
            break
    token_ids = p_ids + c_ids
    overlap = kl_token_overlap(token_ids, kl_runs)
    if overlap and not reason:
        reason = f"kl overlap: {overlap} run(s) of {KL_TOKEN_WINDOW} ids shared with {', '.join(KL_TOKEN_FILES)}"
    return {
        "id": sid, "base_id": entry["id"], "pass": job["pass"], "category": entry["category"],
        "format": entry["format"], "enable_thinking": True, "prompt_enable_thinking": entry["enable_thinking"],
        "messages": messages, "text": content.strip() if entry["format"] == "raw" else None,
        "finish_reason": finishes[-1], "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "seed": seeds[0], **sampling, "max_tokens": max_tokens, "rejected": reason is not None,
        "reject_reason": reason, "truncated_thought_turn": truncated_turn, "prompt_sha256": gc.entry_sha256(entry),
        "turn_seeds": seeds, "turn_finish_reasons": finishes, "turns": turns, "token_ids": token_ids,
        "calib_kind": "chat_stream", "kl_token_overlap": overlap, "model": model,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


# --------------------------------------------------------------------------------------------
# File handling
# --------------------------------------------------------------------------------------------


def read_existing(path: Path, by_id: dict[str, dict], sampling: dict) -> dict[str, dict]:
    """id -> record of an existing samples file; a torn LAST line is truncated away, any other problem (malformed
    line, duplicate or unknown id, changed prompt, changed sampling) raises SystemExit."""
    recs: dict[str, dict] = {}
    if not path.is_file():
        return recs
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    tail_ok = raw.endswith(b"\n") or not raw
    keep = len(raw)
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        last = k == len(lines) - 1 or (k == len(lines) - 2 and not lines[-1].strip())
        try:
            r = json.loads(line.decode("utf-8"))
        except ValueError:
            if last and not tail_ok:
                keep = len(b"\n".join(lines[:k])) + (1 if k else 0)
                print(f"[gen_samples_server] truncating a torn last line of {path}")
                break
            raise SystemExit(f"[gen_samples_server] {path}: line {k + 1} is not valid JSON")
        sid = r.get("id")
        if sid not in by_id:
            raise SystemExit(f"[gen_samples_server] {path}: id {sid!r} is not in the prompt file / --passes")
        if sid in recs:
            raise SystemExit(f"[gen_samples_server] {path}: duplicate id {sid}")
        if r.get("prompt_sha256") != gc.entry_sha256(by_id[sid]["entry"]):
            raise SystemExit(f"[gen_samples_server] {path}: {sid} was generated from a different prompt "
                             "(prompt_sha256 differs); the prompt file changed")
        for key, val in sampling.items():
            if r.get(key) != val:
                raise SystemExit(f"[gen_samples_server] {path}: {sid} was sampled with {key}={r.get(key)!r}, this "
                                 f"run uses {val!r}; mixing samplers under one corpus is refused")
        recs[sid] = r
    if keep < len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return recs


def append_record(f, rec: dict) -> None:
    f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())


def merge(paths: list[Path], out: Path, prompts: Path | None, kl_dir: Path, passes: int,
          require_complete: bool) -> int:
    """Merge sample files into `out` (sorted by id). READ-ONLY on the inputs (a torn last line is skipped with a
    warning). Refuses an id in two files and, with `prompts`, a record from another prompt set; reports missing
    ids of --passes passes; --require-complete makes a missing id an error."""
    recs: dict[str, dict] = {}
    info = []
    for p in paths:
        raw = p.read_bytes()
        lines = raw.split(b"\n")
        last_k = max((k for k, ln in enumerate(lines) if ln.strip()), default=-1)
        n = torn = 0
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                r = json.loads(line.decode("utf-8"))
            except ValueError:
                if k == last_k and not raw.endswith(b"\n"):
                    torn += 1
                    print(f"[gen_samples_server] WARNING {p}: torn last line skipped (still being written?)")
                    continue
                raise SystemExit(f"[gen_samples_server] {p}: line {k + 1} is not valid JSON")
            if r["id"] in recs:
                raise SystemExit(f"duplicate id {r['id']} across input files")
            recs[r["id"]] = r
            n += 1
        info.append({"path": str(p), "sha256": sha256_bytes(raw), "samples": n, "torn_lines_skipped": torn})
    missing: list[str] = []
    if prompts is not None:
        entries, _ = gc.load_prompts(prompts, kl_dir)
        by_id = {j["sample_id"]: j for j in make_jobs(gc.processing_order(entries), passes)}
        for sid, r in recs.items():
            if sid not in by_id:
                raise SystemExit(f"[gen_samples_server] merge: id {sid!r} is not in the prompt file / --passes")
            if r.get("prompt_sha256") != gc.entry_sha256(by_id[sid]["entry"]):
                raise SystemExit(f"[gen_samples_server] merge: {sid} was generated from a different prompt")
        missing = sorted(set(by_id) - set(recs))
        print(f"[gen_samples_server] {len(recs)} of {len(by_id)} samples present; {len(missing)} missing")
        if missing and require_complete:
            raise SystemExit(f"[gen_samples_server] merge: --require-complete and {len(missing)} samples are "
                             f"missing (first: {missing[:5]})")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for sid in sorted(recs):
            f.write((json.dumps(recs[sid], ensure_ascii=False) + "\n").encode("utf-8"))
    s = summarize(recs)
    print(f"[gen_samples_server] merged {len(recs)} samples ({s['accepted']} accepted, "
          f"{s['calibration_tokens_accepted']:,} calibration tokens) -> {out}")
    mp = out.with_name(out.stem + ".merge.json")
    mp.write_text(json.dumps({"format": "r4dx-lfm2-corpus-merge", "version": 1, "out": str(out),
                              "samples_sha256": sha256_file(out), "inputs": info,
                              "complete": prompts is not None and not missing, "missing_ids": missing,
                              "summary": s}, indent=2), encoding="utf-8")
    print(f"[gen_samples_server] merge record {mp}")
    return 0


def summarize(recs: dict[str, dict]) -> dict:
    acc = [r for r in recs.values() if not r["rejected"]]
    by_cat: dict[str, dict] = {}
    for r in recs.values():
        c = by_cat.setdefault(r["category"], {"samples": 0, "accepted": 0, "tokens": 0, "truncated_thoughts": 0})
        c["samples"] += 1
        if not r["rejected"]:
            c["accepted"] += 1
            c["tokens"] += len(r["token_ids"])
        c["truncated_thoughts"] += r.get("truncated_thought_turn") is not None
    reasons: dict[str, int] = {}
    for r in recs.values():
        if r["rejected"]:
            k = gc.reject_kind(r["reject_reason"])
            reasons[k] = reasons.get(k, 0) + 1
    deltas = [t["completion_ids_len"] - int(t["completion_tokens_server"]) for r in recs.values()
              for t in r.get("turns", []) if t.get("completion_tokens_server") is not None]
    return {"samples": len(recs), "accepted": len(acc), "rejected": len(recs) - len(acc),
            "reject_reasons": reasons, "by_category": by_cat,
            "calibration_tokens_accepted": sum(len(r["token_ids"]) for r in acc),
            "completion_tokens_all": sum(r["completion_tokens"] for r in recs.values()),
            "finish_length": sum(1 for r in recs.values() if r["finish_reason"] == "length"),
            "kl_token_overlap_accepted": sum(r["kl_token_overlap"] for r in acc),
            "completion_ids_minus_server_tokens": (
                {"turns": len(deltas), "min": min(deltas), "max": max(deltas), "mean": round(sum(deltas) / len(deltas), 3),
                 "histogram": dict(sorted(collections.Counter(deltas).items())[:20])} if deltas else None)}


# --------------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------------


def dry_run(entries, order, tok, args) -> int:
    """No server, no GPU: render every prompt, self-check the template round trip, count the token budget."""
    cap, floor = args.max_tokens_cap, args.min_max_tokens
    bad, ptoks, ctx_over = [], [], []
    per_cat = collections.defaultdict(lambda: [0, 0, 0])  # samples, prompt tokens (final-turn estimate), budget
    for e in order:
        try:
            msgs = [dict(m) for m in e["messages"]]
            ids = prompt_ids_of(tok, msgs)
            ptoks.append(len(ids))
            why = template_selfcheck(tok, msgs)
            if why:
                bad.append(f"{e['id']}: {why}")
            if not tok.agrees(tok.render_chat(sendable(msgs), add_generation_prompt=True)):
                bad.append(f"{e['id']}: AutoTokenizer and tokenizers disagree on the rendered prompt")
            if not tok.render_chat(sendable(msgs), add_generation_prompt=True).endswith("<|im_start|>assistant\n<think>"):
                bad.append(f"{e['id']}: generation prompt does not end with the open <think>")
            mt = budget_of(e, cap, floor)
            fu = [tok.encode(f) for f in e.get("followups", [])]
            # worst case, every turn spends max_tokens: the last turn's prompt holds the earlier answers
            worst = len(ids) + sum(len(f) + 12 for f in fu) + mt * gc.turn_count(e)
            if worst > args.max_ctx:
                ctx_over.append(f"{e['id']}: worst case {worst} > --max-ctx {args.max_ctx}")
            c = per_cat[e["category"]]
            c[0] += 1
            c[1] += len(ids) + sum(len(f) + 12 for f in fu)
            c[2] += mt * gc.turn_count(e)
        except Exception as exc:  # report every entry, not the first
            bad.append(f"{e['id']}: {type(exc).__name__}: {exc}")
    gen = sum(c[2] for c in per_cat.values())
    prm = sum(c[1] for c in per_cat.values())
    print(f"[gen_samples_server] {len(entries)} prompts; max_tokens = entry budget capped at {cap}"
          f"{f', raised to >= {floor}' if floor else ''}; sampling {SAMPLING}")
    print(f"[gen_samples_server] prompt tokens: min {min(ptoks)} max {max(ptoks)} total {sum(ptoks):,}")
    for cat in gc.CATEGORIES:
        if cat in per_cat:
            c = per_cat[cat]
            print(f"    {cat:<14} {c[0]:>4} samples  generation budget {c[2]:>9,}  prompt-side {c[1]:>8,}")
    per_pass_max = gen + prm
    print(f"[gen_samples_server] one pass produces AT MOST {gen:,} generated tokens (every turn exhausts its "
          f"budget) + {prm:,} prompt tokens = {per_pass_max:,} stream tokens (token_ids of every sample)")
    print(f"[gen_samples_server] target {TARGET_TOKENS:,} tokens: passes needed if every turn hits its cap: "
          f"{-(-TARGET_TOKENS // per_pass_max)}")
    for u in (0.5, 0.7, 0.9):
        est = int(prm * 1.0 + gen * u)
        print(f"    at {int(u * 100)}% of the budget actually used: {est:,} tokens/pass -> "
              f"{-(-TARGET_TOKENS // max(est, 1))} pass(es) for the target")
    print(f"[gen_samples_server] --passes {args.passes}: at most {args.passes * per_pass_max:,} stream tokens "
          f"({'reaches' if args.passes * per_pass_max >= TARGET_TOKENS else 'CANNOT reach'} {TARGET_TOKENS:,} even if every budget is exhausted)")
    first = order[0]
    print(f"[gen_samples_server] first request {first['id']} seed {sample_seed(first['id'])}: "
          f"{tok.render_chat(sendable(first['messages']), add_generation_prompt=True)[:160]!r}")
    for c in ctx_over:
        print(f"[gen_samples_server] WARNING {c}")
    for b in bad:
        print(f"[gen_samples_server] PROBLEM {b}")
    if not bad:
        print(f"[gen_samples_server] template self-check ok on all {len(order)} prompts "
              f"(prompt ids + completion ids == ids of the full template render)")
    return 1 if bad else 0


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="tokenizer.json + chat_template.jinja")
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=None, help="default gen_manifest.json next to --out")
    ap.add_argument("--passes", type=int, default=1, help="sampling passes; pass k>=1 ids are '<id>#<k>'")
    ap.add_argument("--merge", nargs="+", type=Path, default=None, metavar="SAMPLES.jsonl")
    ap.add_argument("--require-complete", action="store_true", help="with --merge")
    ap.add_argument("--no-prompt-check", action="store_true", help="with --merge")
    ap.add_argument("--limit", type=int, default=0, help="generate at most N samples this run")
    ap.add_argument("--only-category", choices=gc.CATEGORIES, default=None)
    ap.add_argument("--max-tokens-cap", type=int, default=MAX_TOKENS_CAP)
    ap.add_argument("--min-max-tokens", type=int, default=0, help="raise smaller budgets to this (0 = off)")
    ap.add_argument("--max-ctx", type=int, default=8192, help="dry-run context warning threshold")
    ap.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    ap.add_argument("--top-p", type=float, default=SAMPLING["top_p"])
    ap.add_argument("--top-k", type=int, default=SAMPLING["top_k"])
    ap.add_argument("--repetition-penalty", type=float, default=SAMPLING["repetition_penalty"])
    ap.add_argument("--timeout", type=float, default=1800.0, help="per request, seconds")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-wait", type=float, default=5.0)
    ap.add_argument("--no-verify-prompt", action="store_true",
                    help="do not abort when usage.prompt_tokens differs from the rendered prompt on the first request")
    ap.add_argument("--kl-dir", type=Path, default=KL_CORPUS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, tok=None) -> int:
    args = parse_args(argv)
    for s in (sys.stdout, sys.stderr):  # Thai / multilingual text in log lines must not kill a cp1252 console
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if args.passes < 1:
        raise SystemExit("--passes must be >= 1")
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty}
    if args.merge:
        return merge(args.merge, args.out, None if args.no_prompt_check else args.prompts, args.kl_dir,
                     args.passes, args.require_complete)
    try:
        entries, prompts_sha = gc.load_prompts(args.prompts, args.kl_dir)
        kl_shingles = kl_text_shingles(args.kl_dir)
    except (gc.PromptFileError, gc.KlCorpusError, OSError) as e:
        print(f"[gen_samples_server] ERROR: {e}")
        return 2
    kl_runs = kl_token_index(args.kl_dir)
    tok = tok or Lfm2RefTokenizer(args.model_dir)
    order = gc.processing_order(entries)
    if args.only_category:
        order = [e for e in order if e["category"] == args.only_category]
    if args.dry_run:
        return dry_run(entries, order, tok, args)
    jobs = make_jobs(order, args.passes)
    by_id = {j["sample_id"]: j for j in make_jobs(gc.processing_order(entries), args.passes)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or out.with_name(MANIFEST_NAME)
    recs = read_existing(out, by_id, sampling)
    todo = [j for j in jobs if j["sample_id"] not in recs]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[gen_samples_server] {len(recs)} samples already in {out}; this run: {len(todo)} of {len(jobs)} "
          f"({args.passes} pass(es)), server {args.server}, sampling {sampling}", flush=True)
    log = print
    client = gc.Client(args.server, args.timeout, args.retry_wait, log)
    server_info = {}
    if todo:
        if not client.healthy():
            print(f"[gen_samples_server] ERROR: {args.server}/health does not answer ok")
            return 1
        for name, path in (("health", "/health"), ("models", "/v1/models")):
            try:
                server_info[name] = client._once("GET", path, None, 10.0)
            except Exception as e:  # noqa: BLE001 -- provenance only
                server_info[name] = f"unavailable: {e}"
    run = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "planned": len(todo),
           "generated": 0, "rejected": 0, "completion_tokens": 0}
    state = {"verified": False, "verify": not args.no_verify_prompt}
    t_run = time.monotonic()
    code, fails = 0, 0
    try:
        with open(out, "ab") as f:
            for k, job in enumerate(todo, 1):
                t = time.monotonic()
                try:
                    rec = generate_sample(job, client, tok, sampling, kl_shingles, kl_runs, args.max_tokens_cap,
                                          args.min_max_tokens, args.retries, args.retry_wait, state, log)
                except gc.TransportFailed as e:
                    print(f"[gen_samples_server] ABORT: {e}")
                    code = 1
                    break
                except (gc.RequestFailed, gc.ResponseInvalid) as e:
                    fails += 1
                    print(f"[gen_samples_server] {job['sample_id']} refused ({fails} in a row): {e}")
                    if fails >= MAX_CONSECUTIVE_FAILURES:
                        print("[gen_samples_server] ABORT: repeated refusals; the requests themselves are wrong")
                        code = 2
                        break
                    continue
                except PromptIdMismatch as e:
                    print(f"[gen_samples_server] ABORT (first request): {e}\n"
                          "  The server prompt is not the template render used here; the corpus ids would not be what "
                          "the model saw. Check --dialect lfm2 and the model directory (or --no-verify-prompt).")
                    code = 3
                    break
                fails = 0
                append_record(f, rec)
                recs[rec["id"]] = rec
                run["generated"] += 1
                run["rejected"] += rec["rejected"]
                run["completion_tokens"] += rec["completion_tokens"]
                rate = run["completion_tokens"] / max(time.monotonic() - t_run, 1e-9)
                print(f"[gen_samples_server] {k}/{len(todo)} {rec['id']:<20} {'REJECT' if rec['rejected'] else 'ok':<6} "
                      f"{rec['completion_tokens']:>5} tok {rec['finish_reason']:<6} {time.monotonic() - t:6.1f} s | "
                      f"run {run['completion_tokens']:,} tok {rate:.1f} tok/s"
                      + (f" | {rec['reject_reason']}" if rec["rejected"] else ""), flush=True)
    except KeyboardInterrupt:
        print("[gen_samples_server] interrupted; every finished sample is on disk")
        code = 130
    finally:
        run["ended"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        run["wall_s"] = round(time.monotonic() - t_run, 1)
        run["exit_code"] = code
        manifest = {
            "format": "r4dx-lfm2-corpus-gen", "version": 1, "server": args.server, "server_info": server_info,
            "model_dir": str(args.model_dir), "tokenizer": tok.provenance() if hasattr(tok, "provenance") else None,
            "prompts": str(args.prompts), "prompts_sha256": prompts_sha, "passes": args.passes,
            "sampling": sampling, "max_tokens_cap": args.max_tokens_cap, "min_max_tokens": args.min_max_tokens,
            "seed_domain": SEED_DOMAIN, "kl_token_window": KL_TOKEN_WINDOW, "kl_dir": str(args.kl_dir),
            "kl_token_runs": len(kl_runs), "kl_text_shingles": len(kl_shingles),
            "token_format": "chat_stream: final turn prompt ids (template, open <think>) + encode(reasoning + "
                            "</think> + content) + <|im_end|> when finish_reason is stop",
            "summary": summarize(recs), "samples_sha256": sha256_file(out) if out.is_file() else None,
            "target_tokens": TARGET_TOKENS,
        }
        prev = []
        if manifest_path.is_file():
            try:
                prev = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
            except ValueError:
                prev = []
        manifest["runs"] = prev + [run]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        s = manifest["summary"]
        print(f"[gen_samples_server] {s['samples']} samples on disk ({s['accepted']} accepted, {s['rejected']} rejected), "
              f"{s['calibration_tokens_accepted']:,} calibration tokens (target {TARGET_TOKENS:,}); manifest {manifest_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
