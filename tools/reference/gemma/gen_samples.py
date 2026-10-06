"""tools/reference/gemma/gen_samples.py  (M0-9: the Gemma calibration corpus, token-id JSONL)

Trellis calibration data for Gemma comes from Huihui-Gemma itself, generated through HF
`generate` on the bf16 checkpoint (docs/gemma4-plan.md 6.3 / 9.8; the r4dx engine is not needed): every
prompt of tools/quant2/corpus_v2_prompts.json (385 entries: thai_prose, english_prose, chat, code,
multilingual; thinking on/off as each entry says) is rendered through the Gemma chat template, sampled with
the checkpoint's own generation_config.json (temperature 1.0, top_k 64, top_p 0.95; EOS [1, 106, 50],
suppress [258883, 258882]) and appended to a JSONL, one object per line:

    id, category, format ("raw" | "chat"), enable_thinking, messages (prompt + generated assistant turns:
    "content", plus "reasoning_content" when thinking was on), text (raw: the answer; chat: null),
    finish_reason, prompt_tokens, completion_tokens, seed, temperature, top_k, top_p, max_tokens,
    rejected, reject_reason, truncated_thought_turn, prompt_sha256, turn_seeds, turn_finish_reasons,
    turns [{prompt_len, gen_ids, finish_reason}],
    token_ids        the calibration token stream (below), calib_kind, kl_token_overlap

`token_ids` are what the Hessian capture feeds the model:
  raw   `[BOS=2] + gen_ids` minus the trailing end-of-turn/EOS: the document exactly as the model
        tokenized it (calib_kind "raw_text"; thinking is off for every raw entry)
  chat  the final turn's rendered prompt (`<bos><|turn>user ...`, earlier turns' thoughts stripped by the
        template) + that turn's generated ids, end-of-turn included (calib_kind "chat_stream")
The ids are the model's own: generation never goes through text for the stream.

Rejection (the line is still written so a resume does not redo it): the checks of tools/quant2/gen_corpus.py
(`check_turn`: empty output, raw document < 200 chars, think tags, reasoning with thinking off, degenerate
repetition) and "kl overlap": text sharing 48 characters (gen_corpus.kl_reason) OR a 32-token run
(`kl_token_overlap`, against tokens_gemma.json / tokens_gemma_long.json) with the held-out KL corpus. A
thinking turn cut inside its thought is KEPT (truncated_thought_turn), as in gen_corpus.

Resume: `--out` is appended, flushed and fsync'ed per sample; ids already in the file are skipped; a
changed prompt (prompt_sha256) or an unknown id stops the run. Seeds depend on the sample id and turn only
(`sample_seed`), so shard layout / order / resume never change a sample.

Shards (two GPUs, no P2P needed): `--shard K/N` takes every N-th entry of the processing order; run one per
device and merge with `--merge a.jsonl b.jsonl --out samples.jsonl`:

    $env:HIP_VISIBLE_DEVICES='0'; $env:R4DX_REF_ALLOWED_DEVICES='0,1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\gen_samples.py --shard 0/2 --out <models root>\\r4dx\\huihui-gemma\\corpus\\samples.shard0.jsonl
    $env:HIP_VISIBLE_DEVICES='1'; $env:R4DX_REF_ALLOWED_DEVICES='0,1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\gen_samples.py --shard 1/2 --out <models root>\\r4dx\\huihui-gemma\\corpus\\samples.shard1.jsonl

(device 0 drives the desktop: only with the user's go-ahead for that run.) `--dry-run` needs no model and no GPU:
it validates the prompt file, renders every prompt with both thinking settings and prints the token budget.
`--tiny` runs the generator on a tiny random checkpoint on CPU (stub tokenizer, built-in prompts).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

if "--tiny" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma.common_gemma import (  # noqa: E402
    BOS_ID,
    DEFAULT_MODEL_DIR,
    EOS_IDS,
    GEMMA_OUT_DIR,
    KL_CORPUS_DIR,
    PROMPTS_PATH,
    load_generation_config,
    load_tokenizer,
    resolve_device,
    sampling_from_generation_config,
    sha256_bytes,
    sha256_file,
    split_model_output,
)
from gemma.ref import add_tiny_args, build_tiny_checkpoint  # noqa: E402

sys.path.insert(0, str(PROMPTS_PATH.parent))
import gen_corpus as gc  # noqa: E402  (tools/quant2: stdlib only; prompt validation, rejection, KL text check)

SEED_DOMAIN = "r4dx-gemma-corpus-v1"
KL_TOKEN_WINDOW = 32
MANIFEST_NAME = "gen_manifest.json"
KL_TOKEN_FILES = ("tokens_gemma.json", "tokens_gemma_long.json")


def sample_seed(sample_id: str, turn: int = 0) -> int:
    d = hashlib.sha256(f"{SEED_DOMAIN}:{sample_id}#{turn}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") & 0x7FFFFFFF


def kl_token_index(kl_dir: Path, window: int = KL_TOKEN_WINDOW) -> set:
    """Every `window`-token run of every KL segment (tokens_gemma*.json) as a tuple."""
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
    """How many `window`-token runs of `ids` occur in the KL segments."""
    if not runs:
        return 0
    return sum(tuple(ids[i:i + window]) in runs for i in range(len(ids) - window + 1))


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------


class HFGenerator:
    """HF `generate` on the bf16 checkpoint (one sequence at a time)."""

    def __init__(self, model_dir: Path, device, attn: str, sampling: dict, eos_ids, pad_id: int):
        import torch
        from transformers import GenerationConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        self.torch = torch
        self.device = device
        kw = {"dtype": torch.bfloat16, "attn_implementation": attn}
        if device.type == "cuda":
            kw["device_map"] = {"": 0}
        from gemma.common_gemma import guarded_from_pretrained

        self.model = guarded_from_pretrained(Gemma4UnifiedForConditionalGeneration, model_dir, **kw)
        if device.type != "cuda":
            self.model.to(device)
        self.model.eval()
        self.gen_cfg = GenerationConfig.from_pretrained(str(model_dir))
        self.sampling = sampling
        self.eos = list(eos_ids)
        self.pad = pad_id

    def generate(self, prompt_ids: list[int], max_new: int, seed: int) -> tuple[list[int], str]:
        torch = self.torch
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        cfg = copy.deepcopy(self.gen_cfg)  # one object (transformers rejects config + kwargs together)
        cfg.update(max_new_tokens=max_new, do_sample=True, temperature=self.sampling["temperature"],
                   top_k=self.sampling["top_k"], top_p=self.sampling["top_p"], eos_token_id=self.eos,
                   pad_token_id=self.pad, use_cache=True)
        with torch.no_grad():
            out = self.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), generation_config=cfg)
        gen = [int(t) for t in out[0, len(prompt_ids):].tolist()]
        finish = "stop" if gen and gen[-1] in self.eos else "length"
        return gen, finish


def request_kwargs(entry: dict) -> dict:
    return {"enable_thinking": bool(entry["enable_thinking"]), "add_generation_prompt": True}


def strip_end(ids: list[int], eos_ids) -> list[int]:
    while ids and ids[-1] in eos_ids:
        ids = ids[:-1]
    return ids


def generate_sample(entry: dict, gen: HFGenerator, tok, sampling: dict, eos_ids, kl_shingles: dict,
                    kl_runs: set, max_new_cap: int = 0) -> dict:
    thinking = entry["enable_thinking"]
    messages = [dict(m) for m in entry["messages"]]
    followups = entry.get("followups", [])
    turns, seeds, finishes = [], [], []
    prompt_tokens = completion_tokens = 0
    reason, truncated_turn, content, prompt_ids, gen_ids = None, None, "", [], []
    t0 = time.monotonic()
    for turn in range(1 + len(followups)):
        if turn > 0:
            messages.append({"role": "user", "content": followups[turn - 1]})
        # earlier assistant turns are sent as `content` only: the template drops past thoughts anyway
        sendable = [{"role": m["role"], "content": m["content"]} for m in messages]
        prompt_ids = tok.encode_chat(sendable, **request_kwargs(entry))
        seed = sample_seed(entry["id"], turn)
        max_new = min(entry["max_tokens"], max_new_cap) if max_new_cap else entry["max_tokens"]
        gen_ids, finish = gen.generate(prompt_ids, max_new, seed)
        parts = split_model_output(tok.decode(gen_ids))
        content = parts["content"]
        reasoning = parts["reasoning"]
        assistant = {"role": "assistant", "content": content}
        if thinking:
            reasoning = reasoning if reasoning is not None else ""
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
        prompt_tokens += len(prompt_ids)
        completion_tokens += len(gen_ids)
        seeds.append(seed)
        finishes.append(finish)
        turns.append({"prompt_len": len(prompt_ids), "gen_ids": gen_ids, "finish_reason": finish})
        reason = (gc.check_turn(entry["format"], thinking, turn, content, reasoning,
                                parts["reasoning"] is not None, finish)
                  or gc.kl_reason(turn, content, reasoning, kl_shingles))
        if not reason and kl_token_overlap(gen_ids, kl_runs):
            reason = f"kl overlap: {kl_token_overlap(gen_ids, kl_runs)} run(s) of {KL_TOKEN_WINDOW} ids shared with tokens_gemma*.json (turn {turn + 1})"
        if reason:
            break
        if gc.is_truncated_thought(thinking, content, reasoning, finish):
            truncated_turn = turn + 1
            break
    if entry["format"] == "raw":
        token_ids, kind = [BOS_ID] + strip_end(gen_ids, set(eos_ids)), "raw_text"
    else:
        token_ids, kind = prompt_ids + gen_ids, "chat_stream"
    return {
        "id": entry["id"], "category": entry["category"], "format": entry["format"], "enable_thinking": thinking,
        "messages": messages, "text": content.strip() if entry["format"] == "raw" else None,
        "finish_reason": finishes[-1], "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "seed": seeds[0], "temperature": sampling["temperature"], "top_k": sampling["top_k"],
        "top_p": sampling["top_p"], "max_tokens": entry["max_tokens"], "rejected": reason is not None,
        "reject_reason": reason, "truncated_thought_turn": truncated_turn,
        "prompt_sha256": gc.entry_sha256(entry), "turn_seeds": seeds, "turn_finish_reasons": finishes,
        "turns": turns, "token_ids": token_ids, "calib_kind": kind,
        "kl_token_overlap": kl_token_overlap(token_ids, kl_runs), "elapsed_s": round(time.monotonic() - t0, 3),
    }


# --------------------------------------------------------------------------------------------
# File handling
# --------------------------------------------------------------------------------------------


def read_existing(path: Path, by_id: dict) -> dict[str, dict]:
    """id -> record of an existing samples file; a torn LAST line is truncated away, any other
    problem (malformed line, duplicate or unknown id, changed prompt) raises SystemExit."""
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
                print(f"[gen_samples] truncating a torn last line of {path}")
                break
            raise SystemExit(f"[gen_samples] {path}: line {k + 1} is not valid JSON")
        sid = r.get("id")
        if sid not in by_id:
            raise SystemExit(f"[gen_samples] {path}: id {sid!r} is not in the prompt file")
        if sid in recs:
            raise SystemExit(f"[gen_samples] {path}: duplicate id {sid}")
        if r.get("prompt_sha256") != gc.entry_sha256(by_id[sid]):
            raise SystemExit(f"[gen_samples] {path}: {sid} was generated from a different prompt "
                             "(prompt_sha256 differs); the prompt file changed")
        recs[sid] = r
    if keep < len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return recs


def append_record(f, rec: dict) -> None:
    f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    f.flush()
    os.fsync(f.fileno())


def shard_of(order: list[dict], spec: str) -> list[dict]:
    k, n = (int(x) for x in spec.split("/"))
    if not (n >= 1 and 0 <= k < n):
        raise SystemExit(f"--shard {spec!r}: need K/N with 0 <= K < N")
    return [e for i, e in enumerate(order) if i % n == k]


def merge(paths: list[Path], out: Path, prompts: Path | None = None, kl_dir: Path | None = None,
          require_complete: bool = False) -> int:
    """Merge shard files into `out` (sorted by id). READ-ONLY on the shards, so it is safe while a shard is
    still being generated: a torn last line (a record mid-write) is skipped with a warning, never repaired.
    Checks every record's prompt_sha256 against the prompt file (a shard from another prompt set is
    refused), refuses an id present in two shards, and reports which prompts are still missing; with
    `require_complete` a missing prompt is an error (the Hessian corpus should be the whole set). Writes
    `<out stem>.merge.json` (shard sha256s, counts, summary) next to `out`."""
    recs: dict[str, dict] = {}
    shard_info = []
    for p in paths:
        raw = p.read_bytes()
        lines = raw.split(b"\n")
        last_k = max((k for k, ln in enumerate(lines) if ln.strip()), default=-1)
        torn = 0
        n = 0
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                r = json.loads(line.decode("utf-8"))
            except ValueError:
                if k == last_k and not raw.endswith(b"\n"):
                    torn += 1
                    print(f"[gen_samples] WARNING {p}: torn last line skipped (shard still being written?)")
                    continue
                raise SystemExit(f"[gen_samples] {p}: line {k + 1} is not valid JSON")
            if r["id"] in recs:
                raise SystemExit(f"duplicate id {r['id']} across shards")
            recs[r["id"]] = r
            n += 1
        shard_info.append({"path": str(p), "sha256": sha256_bytes(raw), "samples": n, "torn_lines_skipped": torn})
    missing: list[str] = []
    if prompts is not None:
        entries, _ = gc.load_prompts(prompts, kl_dir or KL_CORPUS_DIR)
        by_id = {e["id"]: e for e in entries}
        for sid, r in recs.items():
            if sid not in by_id:
                raise SystemExit(f"[gen_samples] merge: id {sid!r} is not in the prompt file {prompts}")
            if r.get("prompt_sha256") != gc.entry_sha256(by_id[sid]):
                raise SystemExit(f"[gen_samples] merge: {sid} was generated from a different prompt "
                                 "(prompt_sha256 differs)")
        missing = sorted(set(by_id) - set(recs))
        print(f"[gen_samples] {len(recs)} of {len(by_id)} prompts present; {len(missing)} missing")
        if missing and require_complete:
            raise SystemExit(f"[gen_samples] merge: --require-complete and {len(missing)} prompts have no "
                             f"sample yet (first: {missing[:5]}); is a shard still running?")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for sid in sorted(recs):
            f.write((json.dumps(recs[sid], ensure_ascii=False) + "\n").encode("utf-8"))
    acc = [r for r in recs.values() if not r["rejected"]]
    print(f"[gen_samples] merged {len(recs)} samples ({len(acc)} accepted, "
          f"{sum(len(r['token_ids']) for r in acc):,} calibration tokens) -> {out}")
    info = {"format": "r4dx-gemma-corpus-merge", "version": 1, "out": str(out), "samples_sha256": sha256_file(out),
            "shards": shard_info, "complete": not missing and prompts is not None,
            "missing_prompt_ids": missing, "summary": summarize(recs)}
    mp = out.with_name(out.stem + ".merge.json")
    mp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[gen_samples] merge record {mp}")
    return 0


def summarize(recs: dict[str, dict]) -> dict:
    acc = [r for r in recs.values() if not r["rejected"]]
    by_cat: dict[str, dict] = {}
    for r in recs.values():
        c = by_cat.setdefault(r["category"], {"samples": 0, "accepted": 0, "tokens": 0, "thinking": 0})
        c["samples"] += 1
        if not r["rejected"]:
            c["accepted"] += 1
            c["tokens"] += len(r["token_ids"])
        c["thinking"] += bool(r["enable_thinking"])
    reasons: dict[str, int] = {}
    for r in recs.values():
        if r["rejected"]:
            k = gc.reject_kind(r["reject_reason"])
            reasons[k] = reasons.get(k, 0) + 1
    return {"samples": len(recs), "accepted": len(acc), "rejected": len(recs) - len(acc),
            "reject_reasons": reasons, "by_category": by_cat,
            "calibration_tokens_accepted": sum(len(r["token_ids"]) for r in acc),
            "completion_tokens_all": sum(r["completion_tokens"] for r in recs.values()),
            "thinking_on_accepted": sum(1 for r in acc if r["enable_thinking"]),
            "kl_token_overlap_accepted": sum(r["kl_token_overlap"] for r in acc)}


TINY_PROMPTS = [
    {"id": "english_prose/001", "category": "english_prose", "format": "raw", "enable_thinking": False,
     "max_tokens": 300, "messages": [{"role": "user", "content": "Write about tides."}]},
    {"id": "code/001", "category": "code", "format": "raw", "enable_thinking": False, "max_tokens": 300,
     "messages": [{"role": "user", "content": "Write a sorting function."}]},
    {"id": "chat/001", "category": "chat", "format": "chat", "enable_thinking": True, "max_tokens": 200,
     "messages": [{"role": "user", "content": "Why is the sky blue?"}], "followups": ["And at sunset?"]},
    {"id": "chat/002", "category": "chat", "format": "chat", "enable_thinking": False, "max_tokens": 200,
     "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a prime."}]},
]


def dry_run(entries, order, tok, args) -> int:
    """No model, no GPU: render every prompt with its thinking setting and count tokens."""
    import collections

    n_think = sum(e["enable_thinking"] for e in entries)
    budget = sum(e["max_tokens"] * gc.turn_count(e) for e in entries)
    ptoks = []
    bad = []
    for e in entries:
        try:
            ids = tok.encode_chat([{"role": m["role"], "content": m["content"]} for m in e["messages"]],
                                  **request_kwargs(e))
            ptoks.append(len(ids))
            assert ids[0] == BOS_ID and ids[1] != BOS_ID, "exactly one BOS"
            if not tok.agrees(tok.render_chat([{"role": m["role"], "content": m["content"]} for m in e["messages"]],
                                              **request_kwargs(e))):
                bad.append(f"{e['id']}: AutoTokenizer and tokenizers disagree on the rendered prompt")
        except Exception as exc:  # report every entry, not the first
            bad.append(f"{e['id']}: {type(exc).__name__}: {exc}")
    cats = collections.Counter(e["category"] for e in entries)
    print(f"[gen_samples] {len(entries)} prompts {dict(cats)}; thinking on: {n_think}, off: {len(entries) - n_think}")
    print(f"[gen_samples] prompt tokens: min {min(ptoks)} max {max(ptoks)} total {sum(ptoks):,}; "
          f"generation budget (sum of turns x max_tokens): {budget:,} (the corpus target is ~0.6-0.9M)")
    first = order[0]
    ex = tok.render_chat([{"role": m["role"], "content": m["content"]} for m in first["messages"]],
                         **request_kwargs(first))
    print(f"[gen_samples] first request ({first['id']}, thinking {first['enable_thinking']}), seed "
          f"{sample_seed(first['id'])}: {ex[:200]!r}")
    if args.shard:
        print(f"[gen_samples] shard {args.shard}: {len(shard_of(order, args.shard))} entries")
    for b in bad:
        print(f"[gen_samples] PROBLEM {b}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    ap.add_argument("--out", type=Path, default=GEMMA_OUT_DIR / "corpus" / "samples.jsonl")
    ap.add_argument("--manifest", type=Path, default=None, help="default gen_manifest.json next to --out")
    ap.add_argument("--shard", default=None, help="K/N: every N-th entry of the processing order (one per GPU)")
    ap.add_argument("--merge", nargs="+", type=Path, default=None, metavar="SHARD.jsonl",
                    help="merge shard files into --out (sorted by id) and exit")
    ap.add_argument("--require-complete", action="store_true",
                    help="with --merge: fail unless every prompt of --prompts has a sample (all shards finished)")
    ap.add_argument("--no-prompt-check", action="store_true", help="with --merge: skip the prompt-file validation")
    ap.add_argument("--limit", type=int, default=0, help="generate at most N samples this run")
    ap.add_argument("--only-category", choices=gc.CATEGORIES, default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"], help="attention for generate()")
    ap.add_argument("--max-tokens-cap", type=int, default=0, help="clamp max_new_tokens of every turn (smoke tests; the records keep the entry's max_tokens)")
    ap.add_argument("--kl-dir", type=Path, default=KL_CORPUS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + render the prompts and print the token budget; no model, no GPU")
    add_tiny_args(ap)
    args = ap.parse_args(argv)

    if args.merge:
        return merge(args.merge, args.out, None if args.no_prompt_check else args.prompts, args.kl_dir,
                     args.require_complete)

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.device = "cpu"
        if args.out == ap.get_default("out"):
            args.out = Path(tmp.name) / "samples.jsonl"
        entries = [dict(e) for e in TINY_PROMPTS]
        for e in entries:
            problems = gc.validate_entry(e, e["id"])
            if problems:
                raise SystemExit("; ".join(problems))
        prompts_sha = sha256_bytes(json.dumps(entries, sort_keys=True).encode())
        kl_shingles, kl_runs = {}, set()
        tok = load_tokenizer(tiny_vocab=128)
    else:
        try:
            entries, prompts_sha = gc.load_prompts(args.prompts, args.kl_dir)
            kl_shingles, _ = gc.kl_shingle_index(args.kl_dir)
        except (gc.PromptFileError, gc.KlCorpusError, OSError) as e:
            print(f"[gen_samples] ERROR: {e}")
            return 2
        kl_runs = kl_token_index(args.kl_dir)
        tok = load_tokenizer(args.tokenizer_dir)
    by_id = {e["id"]: e for e in entries}
    order = gc.processing_order(entries)
    if args.only_category:
        order = [e for e in order if e["category"] == args.only_category]
    if args.dry_run:
        return dry_run(entries, order, tok, args)
    if args.shard:
        order = shard_of(order, args.shard)

    device = resolve_device(args.device)  # device rule (R4DX_REF_ALLOWED_DEVICES)
    gen_cfg = load_generation_config(args.model_dir)
    sampling = sampling_from_generation_config(gen_cfg)
    eos_ids = tuple(gen_cfg.get("eos_token_id", EOS_IDS)) if isinstance(gen_cfg.get("eos_token_id"), list) else EOS_IDS
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or out.with_name(MANIFEST_NAME)
    recs = read_existing(out, by_id)
    todo = [e for e in order if e["id"] not in recs]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"[gen_samples] {len(recs)} samples already in {out}; this run: {len(todo)} of {len(order)} "
          f"(shard {args.shard or 'all'}), sampling {sampling}, EOS {list(eos_ids)}", flush=True)

    gen = HFGenerator(args.model_dir, device, args.attn, sampling, eos_ids, pad_id=0)
    run = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "planned": len(todo),
           "generated": 0, "rejected": 0, "completion_tokens": 0, "shard": args.shard, "device": str(device),
           "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")}
    t_run = time.monotonic()
    code = 0
    try:
        with open(out, "ab") as f:
            for k, e in enumerate(todo, 1):
                t = time.monotonic()
                rec = generate_sample(e, gen, tok, sampling, eos_ids, kl_shingles, kl_runs, args.max_tokens_cap)
                append_record(f, rec)
                recs[rec["id"]] = rec
                run["generated"] += 1
                run["rejected"] += rec["rejected"]
                run["completion_tokens"] += rec["completion_tokens"]
                el = time.monotonic() - t
                rate = run["completion_tokens"] / max(time.monotonic() - t_run, 1e-9)
                print(f"[gen_samples] {k}/{len(todo)} {rec['id']:<18} {'REJECT' if rec['rejected'] else 'ok':<6} "
                      f"{rec['completion_tokens']:>5} tok {rec['finish_reason']:<6} {el:6.1f} s | run "
                      f"{run['completion_tokens']:,} tok {rate:.1f} tok/s"
                      + (f" | {rec['reject_reason']}" if rec["rejected"] else ""), flush=True)
    except KeyboardInterrupt:
        print("[gen_samples] interrupted; every finished sample is on disk")
        code = 130
    finally:
        run["ended"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        run["wall_s"] = round(time.monotonic() - t_run, 1)
        manifest = {
            "format": "r4dx-gemma-corpus-gen", "version": 1, "model_dir": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"), "prompts": str(args.prompts),
            "prompts_sha256": prompts_sha, "sampling": sampling, "eos_ids": list(eos_ids),
            "seed_domain": SEED_DOMAIN, "attn": args.attn, "kl_token_window": KL_TOKEN_WINDOW,
            "kl_dir": str(args.kl_dir), "kl_token_runs": len(kl_runs), "summary": summarize(recs),
            "samples_sha256": sha256_file(out) if out.is_file() else None, "run": run,
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
        print(f"[gen_samples] {s['samples']} samples on disk ({s['accepted']} accepted, {s['rejected']} rejected), "
              f"{s['calibration_tokens_accepted']:,} calibration tokens; manifest {manifest_path}")
    if tmp is not None:
        tmp.cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

