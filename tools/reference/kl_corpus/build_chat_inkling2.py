"""tools/reference/kl_corpus/build_chat_inkling2.py  (docs/inkling2-plan.md 7)

Builds the KL corpus of the Inkling-2-Mini (`inkling2_loop`) gate, tokenized with the checkpoint's own tokenizer and
rendered with its chat_template.jinja:

  chat_inkling2.json          PRIMARY: the SAME twelve conversations as chat_lfm2.json (english x3, thai x2, python x2,
                              cpp, the thinking ones, and the two tool conversations; prompts, thoughts, tool results and
                              answers are imported from build_chat_lfm2.py / build_chat_gemma.py), rendered with the
                              Inkling template (role tokens, `<|content_text|>`, the thinking-effort system message
                              `Thinking effort level: 0.9`, tools declared as one `tool_declare` system message).
                              Scoring is on model-turn tokens only (`score_mask`).
  tokens_inkling2.json        raw segments (english / python / cpp / thai text files), `<|begin_of_text|>` + ids,
                              truncated to 1024 tokens, reported only (no pass/fail).
  tokens_inkling2_long.json   one 6144-token raw segment (crosses the 1024-token sliding window / relative-bias extent
                              six times, and many 256-row prefill chunk boundaries).

An assistant turn is rendered exactly as the model emits it: the template has NO generation prompt (nothing is
appended for add_generation_prompt), so the scored span of a turn is
`<|message_model|><|content_thinking|>{thought}<|end_message|><|message_model|><|content_text|>{answer}<|end_message|>
<|content_model_end_sampling|>` (a tool turn: `...<|end_message|><|message_model|>NAME<|content_invoke_tool_json|>
{"name":"NAME","args":{...}}<|end_message|><|content_model_end_sampling|>`), starting at the model's own first
`<|message_model|>` header and ending with `<|content_model_end_sampling|>` (= eos, id 200006, inclusive).  Tool results
carry the tool name in their header (`<|message_tool|>get_weather<|content_text|>...`), the template's `message.name` rule.

Verification (asserted): the full ids equal `AutoTokenizer.apply_chat_template(..., tokenize=True)` AND the raw
`tokenizers` encoding; the per-piece tokenization (unscored / scored pieces cut at special tokens) equals the joint
tokenization; the prompt of the first assistant turn equals the template's `add_generation_prompt=True` rendering (which
is the same text); every segment ends with `<|content_model_end_sampling|>`; the first token is the template's first
special token; the corpus is word-6-gram disjoint from the calibration prompts and the raw KL text files.

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\reference\\kl_corpus\\build_chat_inkling2.py
"""

from __future__ import annotations

import copy
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

from inkling2.common_inkling2 import (  # noqa: E402
    BEGIN_OF_TEXT_ID,
    EOS_ID,
    MESSAGE_SYSTEM_ID,
    MESSAGE_USER_ID,
    Inkling2RefTokenizer,
    token_ids_sha256,
)

SPAN_RE = re.compile(r"<\|message_model\|>.*?<\|content_model_end_sampling\|>", re.S)


def template_messages(conv: dict) -> list[dict]:
    """The Inkling template reads `reasoning_content` (not `thinking`) and names a tool result by `message.name`
    (the function of the preceding assistant tool call); tool-call arguments stay parsed dicts (the template raises on
    strings)."""
    out, last_call = [], None
    for m in copy.deepcopy(conv["messages"]):
        if "thinking" in m:
            m["reasoning_content"] = m.pop("thinking")
        if m["role"] == "assistant" and m.get("tool_calls"):
            last_call = m["tool_calls"][0]["function"]["name"]
        if m["role"] == "tool" and last_call and "name" not in m:
            m["name"] = last_call
        out.append(m)
    return out


def build_chat_segment(tok: Inkling2RefTokenizer, conv: dict) -> dict:
    msgs, tools = template_messages(conv), conv["tools"]
    kw = {"tools": tools} if tools else {}
    text = tok.render_chat(msgs, **kw)
    ref_ids = tok.hf.apply_chat_template(msgs, tokenize=True, **kw)
    if not isinstance(ref_ids, list):
        ref_ids = list(ref_ids["input_ids"])
    spans = [(m.start(), m.end()) for m in SPAN_RE.finditer(text)]
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
    assert tok.raw.encode(text, add_special_tokens=False).ids == joint, "HF != tokenizers.json"
    assert ids[0] in (MESSAGE_SYSTEM_ID, MESSAGE_USER_ID) or tools, f"{conv['name']}: unexpected first token {ids[0]}"
    assert BEGIN_OF_TEXT_ID not in ids, f"{conv['name']}: a chat segment must not contain <|begin_of_text|>"
    # the first assistant turn's prompt == the template's generation prompt (the template adds nothing)
    first_asst = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    gen_prompt = tok.render_chat(msgs[:first_asst], add_generation_prompt=True, **kw)
    assert text[:spans[0][0]] == gen_prompt, f"{conv['name']}: prompt != generation prompt"
    last_scored = max(i for i, m in enumerate(mask) if m)
    assert ids[last_scored] == EOS_ID and last_scored == len(ids) - 1, f"{conv['name']}: must end with eos, scored"
    first_scored = mask.index(1)
    return {"name": conv["name"], "group": conv["group"], "tools": bool(tools), "n_assistant_turns": n_asst,
            "token_ids": ids, "score_mask": mask, "prompt_len": first_scored, "scored_tokens": sum(mask),
            "joint_tokenization_equals_concat": True, "text": text}


def raw_segments(tok: Inkling2RefTokenizer, sources: list[tuple[str, str]], max_tokens: int) -> list[dict]:
    segs = []
    for name, text in sources:
        ids = tok.encode(text)
        assert ids == tok.encode_raw(text), f"{name}: AutoTokenizer != tokenizers.json"
        assert not (ids and ids[0] == BEGIN_OF_TEXT_ID), f"{name}: text starts with <|begin_of_text|>"
        assert tok.is_nfc(text), f"{name}: text is not NFC"
        ids = tok.with_bos(ids)[:max_tokens]
        segs.append({"name": name, "token_ids": [int(i) for i in ids]})
        print(f"{name:20s} {len(ids):6d} tokens (sha256 {token_ids_sha256(ids)[:16]}...)")
    return segs


def main() -> int:
    tok = Inkling2RefTokenizer()
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
        "format": "r4dx-kl-chat-v1", "tokenizer": tok.describe(), "tokenizer_arch": "inkling2", "tokenizer_mode": "canonical",
        "tokenizer_provenance": prov, "add_bos": False, "chat_template": True,
        "score_mask_semantics":
            "score_mask[i]=1 for ids that are model-turn tokens: for every assistant turn, from the model's own first "
            "<|message_model|> header (the template has no generation prompt) through the thinking block, the answer "
            "or the tool-call block(s), up to and including <|content_model_end_sampling|>. Row i of a logprob file "
            "predicts token i+1, so row i is scored when score_mask[i+1]=1.",
        "provenance": "The same twelve conversations as chat_lfm2.json (prompts, thoughts, tool results and answers were "
                      "written for this corpus family: original text, no third-party material), re-rendered with the "
                      "Inkling chat template (thinking effort 0.9, tool results named by their function). Disjoint "
                      "from tools/quant2/corpus_v2_prompts.json and kl_corpus/*.txt (word 6-gram / Thai 24-char "
                      "shingle check of build_chat_lfm2.py).",
        "gate_groups": "english, thai, python+cpp (reported as code), tool, and chat-ALL",
        "segments": segs,
    }
    out = HERE / "chat_inkling2.json"
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({sum(len(s['token_ids']) for s in segs)} tokens, {sum(s['scored_tokens'] for s in segs)} scored)")

    txt = {p.stem: p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.txt"))}
    raw_doc = {"tokenizer": tok.describe(), "tokenizer_mode": "canonical", "tokenizer_arch": "inkling2",
               "tokenizer_provenance": prov, "autotokenizer_matches_tokenizers_json": True,
               "add_special_tokens": False, "add_bos": True, "bos_token_id": BEGIN_OF_TEXT_ID, "chat_template": False}
    short = raw_segments(tok, sorted(txt.items()), 1024)
    (HERE / "tokens_inkling2.json").write_text(json.dumps({**raw_doc, "max_tokens": 1024, "segments": short}, indent=1),
                                               encoding="utf-8")
    long_text = "\n\n".join(txt[k] for k in ("english_prose", "python_source", "cpp_source", "thai_prose"))
    long_segs = raw_segments(tok, [("mixed_long", long_text)], 6144)
    assert len(long_segs[0]["token_ids"]) >= 5500, "the long raw segment must be ~6k tokens"
    (HERE / "tokens_inkling2_long.json").write_text(
        json.dumps({**raw_doc, "max_tokens": 6144, "segments": long_segs}, indent=1), encoding="utf-8")
    print("wrote tokens_inkling2.json, tokens_inkling2_long.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
