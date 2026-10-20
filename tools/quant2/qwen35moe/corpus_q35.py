"""tools/quant2/qwen35moe/corpus_q35.py -- the qwen3_5_moe (Ornith-1.5-122B) calibration corpus (docs/qwen35moe-plan.md M4).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\quant2\\qwen35moe\\corpus_q35.py build [--out ...]
    ... corpus_q35.py slice --src tokens_calib.json --tokens 100000 --out tokens_calib_100k.json
    ... corpus_q35.py random --tokens 20000 --vocab 4095 --out tiny_tokens.json      (tiny-model tests)

`build` = the hessian-v2 recipe of the other trellis tracks (calib.txt + kv_calib_corpus/, WikiText-2 windows, windows of this
repo's sources, self-generated corpus samples; 2048-token windows) tokenized with the CANONICAL Qwen3.5 pre-tokenizer of the
checkpoint's own tokenizer dir (common.RefTokenizer mode `canonical`, refuses non-NFC), unchanged, as the first segments, plus
(when present) a slice of the agent SFT parquet (`--sft`):

  * every conversation is rendered with the checkpoint's chat_template.jinja (add_generation_prompt=False), NFC normalized,
    tokenized canonically and capped at 4096 tokens;
  * ONE conversation per `trial_name`, trials in a seeded shuffle;
  * documents are taken until the corpus reaches --target-tokens (default 1,000,000), then packed first-fit into segments of
    <= 4096 tokens, documents separated by EOS (<|endoftext|> = 248044).

Each source is OPTIONAL and skipped, with a line in the meta file, when its input is not on disk (this machine, 2026-10-21, has
calib.txt + kv_calib_corpus + the repo code windows, but no WikiText-2, no corpus-v2 generated samples and no SFT parquet):
`--wikitext`, `--gen-file`, `--sft` name them; `--require-all` turns a missing one into an error. A corpus without the generated /
SFT part has no agent or thinking-mode traces: for the real run fetch the SFT parquet and, ideally, generate thinking-mode
samples with the Ornith checkpoint (plan 5.1: the DFlash2 drafter is judged on thinking traces).

Output `tokens_calib.json` in the shared segments format of make_tokens_json.py:
    {"tokenizer": ..., "tokenizer_mode": "canonical", "tokenizer_provenance": {...}, "add_special_tokens": false,
     "chat_template": false, "max_tokens": 4096, "eos_token_id": 248044,
     "segments": [{"name": "...", "token_ids": [...]}, ...]}
and `tokens_calib.meta.json` (per segment source + token count, the hv2 source records, the SFT draw, sha256s, skipped sources).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "reference"))

MODELS = Path(r"E:\models")
MODEL_DIR = MODELS / "r4dx" / "Ornith-1.5-122B"
TOKENIZER_DIR = MODEL_DIR
SFT_PARQUET = MODELS / "r4dx" / "q35moe" / "sft" / "qwen38-agent-sft-combined-100pct.parquet"
HV2_SAMPLES = MODELS / "r4dx" / "corpus-v2" / "samples.jsonl"
WIKITEXT = MODELS / "wikitext-2-raw" / "wiki.train.raw"
OUT_DIR = MODELS / "r4dx" / "q35moe"
EOS = 248044
SEG_MAX = 4096
SEED = 20261011
HV2_CODE_REV = "714955f"
HV2_SEQ_LEN = 2048


def sha256_ids(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids], separators=(",", ":")).encode()).hexdigest()


def build_hv2(tokenizer, wikitext: Path, gen_file: Path, skipped: list, require_all: bool):
    """The hessian-v2 corpus: (seqs [hessian_capture.Seq], sources). Wikitext / generated samples are skipped when absent."""
    import hessian_capture as hc

    wiki_ok, gen_ok = Path(wikitext).is_file(), Path(gen_file).is_file()
    for ok, what, path in ((wiki_ok, "wikitext", wikitext), (gen_ok, "generated samples", gen_file)):
        if not ok:
            if require_all:
                raise SystemExit(f"[corpus] {what} {path} does not exist (--require-all)")
            skipped.append({"source": what, "path": str(path), "reason": "not on disk"})
            print(f"[corpus] WARNING: {what} {path} not found: skipped", flush=True)
    args = argparse.Namespace(
        seq_len=HV2_SEQ_LEN, calib_txt=hc.DEFAULT_CALIB_TXT, corpus_dir=hc.DEFAULT_CORPUS_DIR,
        wikitext=str(wikitext) if wiki_ok else "none", wikitext_seqs=128 if wiki_ok else 0, code_seqs=48, code_rev=HV2_CODE_REV,
        gen_file=gen_file if gen_ok else None, gen_max_seqs=None)
    return hc.build_corpus(args, tokenizer)


def draw_sft(parquet: Path, seed: int):
    """One conversation per trial_name (seeded draw among its rows), trials in a seeded shuffle:
    [{trial, row, messages}]."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["conversations", "trial_name", "task", "episode"]).to_pylist()
    by_trial: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_trial.setdefault(r["trial_name"], []).append(i)
    rng = random.Random(seed)
    trials = sorted(by_trial)
    rng.shuffle(trials)
    out = []
    for t in trials:
        i = rng.choice(by_trial[t])
        out.append({"trial": t, "row": i, "task": rows[i]["task"], "episode": rows[i]["episode"],
                    "messages": [{"role": m["role"], "content": m["content"]} for m in rows[i]["conversations"]]})
    return out, len(rows), len(trials)


def render_sft(tokenizer, chat_template: str, sample: dict) -> list[int]:
    text = tokenizer.render_chat(sample["messages"], chat_template=chat_template, add_generation_prompt=False)
    text = unicodedata.normalize("NFC", text)
    return tokenizer.encode(text)[:SEG_MAX]


def pack(docs: list[tuple[str, list[int]]], eos: int = EOS, seg_max: int = SEG_MAX):
    """First-fit in document order into segments of <= seg_max tokens, documents EOS-separated.
    Returns [(name, ids, [doc names])]."""
    segs: list[list] = []
    for name, ids in docs:
        if len(ids) > seg_max:
            raise ValueError(f"{name}: {len(ids)} tokens > {seg_max}")
        for s in segs:
            if len(s[1]) + 1 + len(ids) <= seg_max:
                s[1].append(eos)
                s[1].extend(ids)
                s[2].append(name)
                break
        else:
            segs.append([None, list(ids), [name]])
    return [(f"sft/{k:03d}", ids, names) for k, (_, ids, names) in enumerate(segs)]


def write_tokens(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def cmd_build(a) -> int:
    from common import RefTokenizer, sha256_file

    tok = RefTokenizer(a.tokenizer_dir, "canonical")
    skipped: list[dict] = []
    seqs, sources = build_hv2(tok, a.wikitext, a.gen_file, skipped, a.require_all)
    hv2_tokens = sum(len(s.token_ids) for s in seqs)
    print(f"[corpus] hessian-v2: {len(seqs)} sequences, {hv2_tokens} tokens", flush=True)
    segments = [{"name": f"hv2/{s.name}", "token_ids": list(s.token_ids)} for s in seqs]
    meta_segs = [{"name": f"hv2/{s.name}", "source": "hessian-v2", "tokens": len(s.token_ids)} for s in seqs]

    need = max(0, a.target_tokens - hv2_tokens)
    template_path = a.tokenizer_dir / "chat_template.jinja"
    template = template_path.read_text(encoding="utf-8")
    docs, used, total, packed, sft_meta = [], [], 0, [], None
    if not a.sft.is_file():
        if a.require_all:
            raise SystemExit(f"[corpus] SFT parquet {a.sft} does not exist (--require-all)")
        skipped.append({"source": "agent SFT", "path": str(a.sft), "reason": "not on disk"})
        print(f"[corpus] WARNING: SFT parquet {a.sft} not found: skipped (corpus = {hv2_tokens} tokens, target {a.target_tokens})",
              flush=True)
    else:
        draw, n_rows, n_trials = draw_sft(a.sft, a.seed)
        for s in draw:
            if total >= need:
                break
            ids = render_sft(tok, template, s)
            docs.append((s["trial"], ids))
            used.append({"trial": s["trial"], "row": s["row"], "task": s["task"], "episode": s["episode"],
                         "tokens": len(ids), "sha256": sha256_ids(ids)})
            total += len(ids) + 1
        packed = pack(docs)
        sft_meta = {"parquet": str(a.sft), "sha256": sha256_file(a.sft), "rows": n_rows, "trials": n_trials,
                    "seed": a.seed, "documents": len(docs), "cap_tokens": SEG_MAX,
                    "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "used": used}
    for name, ids, names in packed:
        segments.append({"name": name, "token_ids": ids})
        meta_segs.append({"name": name, "source": "agent-sft", "tokens": len(ids), "documents": names})
    total_tokens = sum(len(s["token_ids"]) for s in segments)
    doc = {"tokenizer": tok.describe(), "tokenizer_mode": "canonical", "tokenizer_provenance": tok.provenance(),
           "add_special_tokens": False, "chat_template": False, "max_tokens": SEG_MAX, "eos_token_id": EOS,
           "segments": segments}
    write_tokens(a.out, doc)
    meta = {"out": str(a.out), "sha256": sha256_file(a.out), "segments": len(segments), "tokens": total_tokens,
            "target_tokens": a.target_tokens, "hv2_tokens": hv2_tokens, "sft_tokens": total, "sft": sft_meta,
            "skipped_sources": skipped, "hv2_sources": sources, "segment_list": meta_segs}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"[corpus] {a.out}: {len(segments)} segments, {total_tokens} tokens "
          f"(hv2 {hv2_tokens} + sft {total} in {len(packed)} segments from {len(docs)} trials; "
          f"{len(skipped)} source(s) skipped)", flush=True)
    return 0

def slice_segments(segs: list[dict], tokens: int) -> list[dict]:
    """An evenly spread subset of `segs` (in order) holding about `tokens` tokens: every source stays represented."""
    total = sum(len(s["token_ids"]) for s in segs)
    if tokens >= total:
        return list(segs)
    n = max(1, min(len(segs), round(tokens / (total / len(segs)))))
    idx = sorted({round(i * (len(segs) - 1) / max(n - 1, 1)) for i in range(n)})
    return [segs[i] for i in idx]


def cmd_slice(a) -> int:
    doc = json.loads(a.src.read_text(encoding="utf-8"))
    picked = slice_segments(doc["segments"], a.tokens)
    write_tokens(a.out, dict(doc, segments=picked))
    print(f"[corpus] slice {a.out}: {len(picked)} of {len(doc['segments'])} segments, "
          f"{sum(len(s['token_ids']) for s in picked)} tokens", flush=True)
    return 0


def cmd_random(a) -> int:
    """Random ids in [0, --vocab) with an EOS (--eos) between documents of ~--doc-len tokens, segments of --seg-len."""
    rng = random.Random(a.seed)
    segs, left, k = [], a.tokens, 0
    while left > 0:
        n = min(a.seg_len, left)
        ids = [rng.randrange(a.vocab) for _ in range(n)]
        for p in range(a.doc_len, n, a.doc_len + 1):
            ids[p] = a.eos
        segs.append({"name": f"rand/{k:03d}", "token_ids": ids})
        left -= n
        k += 1
    write_tokens(a.out, {"tokenizer": "random ids", "tokenizer_mode": "none", "add_special_tokens": False,
                         "chat_template": False, "max_tokens": a.seg_len, "eos_token_id": a.eos, "segments": segs})
    print(f"[corpus] random {a.out}: {len(segs)} segments, {a.tokens} tokens", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="hessian-v2 + SFT slice -> tokens_calib.json")
    p.add_argument("--out", type=Path, default=OUT_DIR / "tokens_calib.json")
    p.add_argument("--target-tokens", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--sft", type=Path, default=SFT_PARQUET, help="agent SFT parquet (optional: skipped when absent)")
    p.add_argument("--wikitext", type=Path, default=WIKITEXT, help="WikiText-2 train (optional: skipped when absent)")
    p.add_argument("--gen-file", type=Path, default=HV2_SAMPLES, help="self-generated samples.jsonl (optional)")
    p.add_argument("--require-all", action="store_true", help="a missing optional source is an error")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("slice", help="an evenly spread subset of segments")
    p.add_argument("--src", type=Path, default=OUT_DIR / "tokens_calib.json")
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_slice)
    p = sub.add_parser("random", help="random token ids (tiny-model tests)")
    p.add_argument("--tokens", type=int, default=20000)
    p.add_argument("--vocab", type=int, default=4095, help="ids drawn from [0, vocab); tiny: 4095 keeps 4095 = EOS out")
    p.add_argument("--eos", type=int, default=4095)
    p.add_argument("--seg-len", type=int, default=1000)
    p.add_argument("--doc-len", type=int, default=300)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_random)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
