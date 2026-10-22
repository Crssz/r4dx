"""tools/reference/kl_corpus/build_chat_q35.py

Builds the held-out KL corpus of the Ornith-1.5-122B (qwen3_5_moe) trellis gate, tokenized with the checkpoint's own tokenizer
(E:\\models\\r4dx\\Ornith-1.5-122B, `tokenizers` canonical mode):

  chat_q35moe.json        PRIMARY: the twelve conversations of build_chat_lfm2.py (english x3 + english thinking, thai x2 + thai
                          thinking, python x2, cpp, two tool conversations) rendered with the checkpoint's chat template.
                          Every assistant turn THINKS (generation prompt `<|im_start|>assistant\\n<think>\\n`); the scored span of a turn is
                          `{thought}\\n</think>\\n\\n{answer}<|im_end|>` (a tool turn: `...</think>\\n\\n<tool_call>...</tool_call><|im_end|>`).
                          Scoring is on those spans only (`score_mask`), the template scaffolding and user turns do not dilute the KL.
  tokens_q35moe.json      raw segments (english / python / cpp / thai text files of this directory), 1024 tokens each, no BOS (the
                          checkpoint adds none), reported next to the chat set.
  tokens_q35moe_long.json one long raw segment (all four text files, ~12k tokens) (the four text files joined), for the late-position (KV cache) rows.

Provenance: prompts, thoughts, tool results and answers were written for this corpus (original text, released with the repo); the
set is disjoint from the calibration prompts and the raw KL text files (word 6-gram / Thai shingle check from build_chat_lfm2.py).
Qwen's template drops the reasoning of turns before the last user message; here every assistant turn follows the last user message,
so each tool-turn keeps its thought.

Verification (asserted): the ids equal `AutoTokenizer.apply_chat_template(..., tokenize=True)` and the raw `tokenizers` encoding;
per-piece tokenization (unscored / scored pieces) equals the joint tokenization; the first assistant turn's prompt equals the
template's generation prompt; every segment ends with `<|im_end|>` (+ the template's trailing "\\n", not scored).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe -I tools\\reference\\kl_corpus\\build_chat_q35.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build_chat_lfm2 as L  # noqa: E402  (importing it has no side effects)
from common import RefTokenizer, sha256_bytes  # noqa: E402

MODEL_DIR = Path(r"E:\models\r4dx\Ornith-1.5-122B")
GEN_PROMPT_TAIL = "<|im_start|>assistant\n<think>\n"
IM_END = "<|im_end|>"


def token_ids_sha256(ids) -> str:
    return sha256_bytes(json.dumps([int(t) for t in ids], separators=(",", ":")).encode("utf-8"))


def template_messages(conv: dict) -> list[dict]:
    """The Qwen template reads `reasoning_content`; tool-call arguments stay dicts."""
    out = []
    for m in conv["messages"]:
        m = dict(m)
        if "thinking" in m:
            m["reasoning_content"] = m.pop("thinking")
        out.append(m)
    return out


def build_chat_segment(tok: RefTokenizer, conv: dict) -> dict:
    msgs, tools = template_messages(conv), conv["tools"]
    kw = {"tools": tools} if tools else {}
    text = tok.render_chat(msgs, **kw)
    ref_ids = tok.hf.apply_chat_template(msgs, tokenize=True, **kw)
    if not isinstance(ref_ids, list):
        ref_ids = list(ref_ids["input_ids"])
    spans = [(m.start(1), m.end()) for m in re.finditer(r"<\|im_start\|>assistant\n<think>\n(.*?<\|im_end\|>)", text, re.S)]
    n_asst = sum(1 for m in msgs if m["role"] == "assistant")
    assert len(spans) == n_asst, f"{conv['name']}: {len(spans)} assistant spans for {n_asst} assistant messages"
    pieces, cur = [], 0
    for a, b in spans:
        pieces += [(text[cur:a], 0), (text[a:b], 1)]
        cur = b
    pieces.append((text[cur:], 0))
    ids, mask = [], []
    for t, sc in pieces:
        if not t:
            continue
        p = tok.encode(t)
        ids += p
        mask += [sc] * len(p)
    joint = tok.encode(text)
    assert ids == joint, f"{conv['name']}: piecewise tokenization differs from the joint tokenization"
    assert joint == list(ref_ids), f"{conv['name']}: ids differ from apply_chat_template(tokenize=True)"
    first_asst = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    gen_prompt = tok.render_chat(msgs[:first_asst], add_generation_prompt=True, **kw)
    assert gen_prompt.endswith(GEN_PROMPT_TAIL) and text.startswith(gen_prompt), f"{conv['name']}: prompt != generation prompt"
    im_end_id = tok.encode(IM_END)[0]
    last_scored = max(i for i, m in enumerate(mask) if m)
    assert ids[last_scored] == im_end_id and mask[last_scored + 1:] in ([], [0])
    first_scored = mask.index(1)
    return {"name": conv["name"], "group": conv["group"], "tools": bool(tools), "n_assistant_turns": n_asst,
            "token_ids": ids, "score_mask": mask, "prompt_len": first_scored, "scored_tokens": sum(mask),
            "joint_tokenization_equals_concat": True, "text": text}


def main() -> int:
    tok = RefTokenizer(MODEL_DIR, "canonical")
    convs = L.conversations()
    problems = L.check_disjoint(convs)
    if problems:
        print("\n".join(problems))
        raise SystemExit("corpus is not disjoint from the calibration/raw corpora")
    segs = []
    for c in convs:
        s = build_chat_segment(tok, c)
        segs.append(s)
        print(f"{s['name']:15s} group={s['group']:8s} turns={s['n_assistant_turns']} prompt={s['prompt_len']:4d} "
              f"scored={s['scored_tokens']:4d} total={len(s['token_ids']):4d}")
    prov = tok.provenance()
    doc = {
        "format": "r4dx-kl-chat-v1", "tokenizer": tok.describe(), "tokenizer_arch": "qwen3_5_moe", "tokenizer_mode": "canonical",
        "tokenizer_provenance": prov, "add_bos": False, "chat_template": True,
        "score_mask_semantics":
            "score_mask[i]=1 for ids that are model-turn tokens: for every assistant turn, the thought, </think>, the answer or the "
            "tool call, and the closing <|im_end|> (the generation prompt ends after <think>\\n, which is NOT scored). Row i of a "
            "logprob file predicts token i+1, so row i is scored when score_mask[i+1]=1.",
        "provenance": "All user prompts, thoughts, tool results and answers were written for this corpus (original text; no "
                      "third-party material); the same twelve conversations as chat_lfm2.json re-rendered with the Ornith chat "
                      "template. Disjoint from tools/quant2/corpus_v2_prompts.json and kl_corpus/*.txt.",
        "gate_groups": "english, thai, python+cpp (reported as code), tool, and chat-ALL",
        "segments": segs,
    }
    out = HERE / "chat_q35moe.json"
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({sum(len(s['token_ids']) for s in segs)} tokens, {sum(s['scored_tokens'] for s in segs)} scored)")

    txt = {p.stem: p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.txt"))}
    raw_doc = {"tokenizer": tok.describe(), "tokenizer_mode": "canonical", "tokenizer_arch": "qwen3_5_moe", "tokenizer_provenance": prov,
               "add_special_tokens": False, "chat_template": False}

    def raw(sources, max_tokens):
        out_segs = []
        for name, text in sources:
            ids = tok.encode(text)[:max_tokens]
            out_segs.append({"name": name, "token_ids": [int(i) for i in ids]})
            print(f"{name:20s} {len(ids):6d} tokens (sha256 {token_ids_sha256(ids)[:16]}...)")
        return out_segs

    short = raw(sorted(txt.items()), 1024)
    (HERE / "tokens_q35moe.json").write_text(json.dumps({**raw_doc, "max_tokens": 1024, "segments": short}, indent=1), encoding="utf-8")
    long_text = "\n\n".join(txt[k] for k in ("english_prose", "python_source", "cpp_source", "thai_prose"))
    (HERE / "tokens_q35moe_long.json").write_text(
        json.dumps({**raw_doc, "max_tokens": 16384, "segments": raw([("mixed_long", long_text)], 16384)}, indent=1), encoding="utf-8")
    print("wrote tokens_q35moe.json, tokens_q35moe_long.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
