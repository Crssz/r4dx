"""tools/reference/kl_corpus/build_chat_lfm2.py  (docs/lfm2-plan.md L0-7, 6.4)

Builds the KL corpus of the LFM2.5-24B-A2B gate:

  chat_lfm2.json          PRIMARY: the same ten conversations as chat_gemma.json (prompts and answers imported
                          from build_chat_gemma.py) rendered with LFM's chat_template.jinja, plus two tool
                          conversations (group `tool`).  Scoring is on model-turn tokens only (`score_mask`).
  tokens_lfm2.json        raw segments (english/thai/python/cpp prose and code), [BOS=124894] + ids, truncated to
                          1024 tokens, reported only (no pass/fail).
  tokens_lfm2_long.json   one ~6k-token raw segment (many 256-row prefill chunk boundaries: conv state carry,
                          attention over several chunks).

The model ALWAYS thinks (the generation prompt ends in `<|im_start|>assistant\\n<think>`), so every scored span
is `{thought}</think>{answer}<|im_end|>` (tool turns: `{thought}</think><|tool_call_start|>[...]<|tool_call_end|>
<|im_end|>`), i.e. exactly what the template re-renders for an assistant message after the last user turn.  The
BOS is the template's own `<|startoftext|>` literal (the post-processor adds none).  Thoughts are short
author-written ones (original text; the two Gemma thinking samples keep theirs); a model-generated set is a later,
non-gating extra (plan open question 5, decided: author-written for the gate).

Verification (asserted): the full ids equal `AutoTokenizer.apply_chat_template(..., tokenize=True)` AND the
raw `tokenizers` encoding; the per-piece tokenization (unscored / scored pieces cut at the special tokens) equals
the joint tokenization; the prompt of the first assistant turn equals the template's generation prompt;
BOS appears exactly once, at index 0; every segment ends with `<|im_end|>` (+ the template's trailing "\\n"
is NOT scored); the corpus is word-6-gram disjoint from the calibration prompts and the raw KL text files.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\kl_corpus\\build_chat_lfm2.py
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

import build_chat_gemma as G  # noqa: E402  (importing it has no side effects)

from lfm2.common_lfm2 import BOS_ID, IM_END_ID, Lfm2RefTokenizer, token_ids_sha256  # noqa: E402

# Short author-written thoughts for the eight Gemma samples that carry none (the two thinking ones keep theirs).
THOUGHTS = {
    "en_bread": "The question is why dough rests. There are really three rests: after mixing, so the flour hydrates "
                "and gluten forms by itself; the long rise, where yeast makes gas and flavor; and the rest after "
                "shaping, which relaxes the gluten. I will go through them in that order and end with the finger dent "
                "test, because it is the practical thing a baker can use.",
    "en_email": "A short polite decline with an alternative. Keep it to a few sentences: thank them, say the week is "
                "full without drama, offer to send notes beforehand, propose another slot, offer to take the action "
                "items. Use a plain subject line and a friendly sign-off.",
    "en_tides": "Plain language, so no tidal-force equations. The Moon's pull differs across the Earth, which gives two "
                "bulges. The Sun adds a smaller effect, which explains spring and neap tides. I should mention that "
                "the cycle is a bit over 24 hours because the Moon moves, and that local geography changes the size "
                "a lot.",
    "th_rain": "The user asks in Thai how rain forms. Describe the water cycle in order: evaporation, condensation into "
               "clouds, droplets growing until they fall, then a note about Thai monsoon rain and afternoon storms, "
               "and finally the water returning to the sea. Write the answer in Thai, in short paragraphs.",
    "th_plants": "A beginner growing plants in a condo, writing in Thai. Useful advice: match plants to light, do not "
                 "overwater, drainage holes, loose soil, little fertilizer, and how to read the leaves. A numbered "
                 "list is easy to follow. Answer in Thai.",
    "py_intervals": "Merge overlapping intervals: sort by start, sweep once, extend the last interval with max of the "
                    "ends. Show the function with a docstring and three example calls, then explain why sorting makes "
                    "one pass enough, why max is needed, what happens for touching intervals, and the complexity.",
    "cpp_queue": "A minimal thread-safe blocking queue in C++17, built from std::mutex plus a wait/notify pair. It needs push, "
                 "a blocking pop that returns an optional, and close so consumers can stop. Show the template class, "
                 "then say how a consumer loops and what to add for back-pressure.",
    "py_decorator": "Explain decorators: a function that takes a function and returns a wrapper, and the at-syntax is "
                    "just assignment. A practical example is a timing decorator with functools.wraps and try/finally. "
                    "Then three details worth remembering and a note on decorators with arguments.",
}

TOOL_SAMPLES = [
    {
        "name": "tool_weather", "group": "tool",
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}],
        "turns": [
            {"role": "user", "content": "What is the weather like in Chiang Mai right now? Please use celsius."},
            {"role": "assistant", "content": "",
             "thinking": "The user wants the current weather in Chiang Mai in celsius. I have a weather tool, so I "
                         "should call it with city Chiang Mai and unit celsius instead of guessing.",
             "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Chiang Mai", "unit": "celsius"}}}]},
            {"role": "tool", "content": "{\"temperature\": 29, \"condition\": \"light rain\", \"humidity\": 86}"},
            {"role": "assistant",
             "thinking": "The tool returned 29 degrees with light rain and 86 percent humidity. I will report that in "
                         "one or two sentences and suggest taking an umbrella, since it is raining.",
             "content": "It is 29 degrees Celsius in Chiang Mai right now, with light rain and high humidity (86%). "
                        "An umbrella would be a good idea if you are heading out."},
        ],
    },
    {
        "name": "tool_files", "group": "tool",
        "tools": [
            {"type": "function", "function": {
                "name": "search_files", "description": "Search project files for a text pattern.",
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"}, "dirs": {"type": "array", "items": {"type": "string"}}},
                    "required": ["pattern"]}}},
            {"type": "function", "function": {
                "name": "read_file", "description": "Read a text file and return its contents.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                               "required": ["path"]}}}],
        "turns": [
            {"role": "system", "content": "You are a careful coding assistant. Use tools when they help."},
            {"role": "user", "content": "Find where the retry limit is set in the src and config folders and show me the first 20 lines of that file."},
            {"role": "assistant", "content": "",
             "thinking": "First I need to find the retry limit, so I search for the text 'retry_limit' in the src and "
                         "config folders. I can already ask for the likely settings file in the same turn, but the "
                         "path depends on the search result, so I will only search now.",
             "tool_calls": [{"function": {"name": "search_files", "arguments": {"pattern": "retry_limit", "dirs": ["src", "config"]}}}]},
            {"role": "tool", "content": "config/settings.toml:12: retry_limit = 5\nsrc/net/client.py:48: if attempts > cfg.retry_limit:"},
            {"role": "assistant", "content": "",
             "thinking": "The setting lives in config/settings.toml at line 12, and the client only reads it. The user "
                         "asked for the first 20 lines of that file, so I read it with max_lines 20.",
             "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "config/settings.toml", "max_lines": 20}}}]},
            {"role": "tool", "content": "[network]\ntimeout_s = 30\nretry_limit = 5\nbackoff = \"exponential\"\n"},
            {"role": "assistant",
             "thinking": "The file is short. The retry limit is 5 under the network section, with a 30 second timeout and "
                         "exponential backoff. I will quote the relevant lines and mention that src/net/client.py "
                         "reads the value.",
             "content": "The retry limit is set in `config/settings.toml`, in the `[network]` section:\n\n```toml\n"
                        "[network]\ntimeout_s = 30\nretry_limit = 5\nbackoff = \"exponential\"\n```\n\n"
                        "`src/net/client.py` (line 48) reads it and stops retrying once `attempts` exceeds `retry_limit`."},
        ],
    },
]

IM_ASSISTANT = "<|im_start|>assistant\n<think>"


def check_disjoint(samples: list[dict]) -> list[str]:
    """Word 6-gram (Thai: 24-char shingle) overlap of every sample text with the calibration prompts and the raw KL
    text files, as build_chat_gemma.check_disjoint."""
    other = []
    doc = json.loads((HERE.parents[1] / "quant2" / "corpus_v2_prompts.json").read_text(encoding="utf-8"))
    other.append(("corpus_v2_prompts.json", json.dumps(doc["samples"], ensure_ascii=False)))
    for p in sorted(HERE.glob("*.txt")):
        other.append((p.name, p.read_text(encoding="utf-8")))
    problems = []
    for s in samples:
        text = s["text_for_check"]
        for oname, otext in other:
            shared = G.ngrams(text) & G.ngrams(otext)
            if shared:
                problems.append(f"{s['name']} shares {len(shared)} word 6-grams with {oname}: {sorted(shared)[:2]}")
            tt, ot = re.sub(r"\s+", "", text), re.sub(r"\s+", "", otext)
            if any("\u0e00" <= ch <= "\u0e7f" for ch in tt):
                hits = [sh for sh in {tt[i:i + 24] for i in range(0, len(tt) - 23)} if sh in ot]
                if hits:
                    problems.append(f"{s['name']} shares {len(hits)} 24-char shingles with {oname}: {hits[:1]}")
    return problems


def conversations() -> list[dict]:
    out = []
    for g in G.SAMPLES:
        thought = g["thought"] or THOUGHTS[g["name"]]
        msgs = [{"role": "user", "content": g["user"]},
                {"role": "assistant", "content": g["answer"], "thinking": thought}]
        out.append({"name": g["name"], "group": g["group"], "tools": None, "messages": msgs,
                    "text_for_check": " ".join([g["user"], thought, g["answer"]])})
    for t in TOOL_SAMPLES:
        check = " ".join(str(m.get("content", "")) + " " + str(m.get("thinking", "")) for m in t["turns"])
        out.append({"name": t["name"], "group": t["group"], "tools": t["tools"], "messages": t["turns"],
                    "text_for_check": check})
    return out


def build_chat_segment(tok: Lfm2RefTokenizer, conv: dict) -> dict:
    msgs, tools = conv["messages"], conv["tools"]
    kw = {"tools": tools} if tools else {}
    text = tok.render_chat(msgs, **kw)
    ref_ids = tok.hf.apply_chat_template(msgs, tokenize=True, **kw)
    if not isinstance(ref_ids, list):
        ref_ids = list(ref_ids["input_ids"])
    # Scored char spans: after `<|im_start|>assistant\n<think>` through `<|im_end|>` (inclusive).
    spans = [(m.start(1), m.end()) for m in re.finditer(r"<\|im_start\|>assistant\n<think>(.*?<\|im_end\|>)", text, re.S)]
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
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1, f"{conv['name']}: BOS rule"
    # the first assistant turn's prompt == the template's generation prompt
    first_user_end = text.index("<|im_start|>assistant\n<think>") + len("<|im_start|>assistant\n<think>")
    gen_prompt = tok.render_chat(msgs[: next(i for i, m in enumerate(msgs) if m["role"] == "assistant")],
                                 add_generation_prompt=True, **kw)
    assert text[:first_user_end] == gen_prompt, f"{conv['name']}: prompt != generation prompt"
    # the last scored token is <|im_end|>; the template's trailing "\n" is not scored
    last_scored = max(i for i, m in enumerate(mask) if m)
    assert ids[last_scored] == IM_END_ID and mask[last_scored + 1:] in ([], [0])
    first_scored = mask.index(1)
    return {"name": conv["name"], "group": conv["group"], "tools": bool(tools), "n_assistant_turns": n_asst,
            "token_ids": ids, "score_mask": mask, "prompt_len": first_scored, "scored_tokens": sum(mask),
            "joint_tokenization_equals_concat": True, "text": text}


def raw_segments(tok: Lfm2RefTokenizer, sources: list[tuple[str, str]], max_tokens: int) -> list[dict]:
    segs = []
    for name, text in sources:
        ids = tok.encode(text)
        assert ids == tok.encode_raw(text), f"{name}: AutoTokenizer != tokenizers.json"
        assert not (ids and ids[0] == BOS_ID), f"{name}: text starts with BOS"
        assert tok.is_nfc(text), f"{name}: text is not NFC"
        ids = tok.with_bos(ids)[:max_tokens]
        segs.append({"name": name, "token_ids": [int(i) for i in ids]})
        print(f"{name:20s} {len(ids):6d} tokens (sha256 {token_ids_sha256(ids)[:16]}...)")
    return segs


def main() -> int:
    tok = Lfm2RefTokenizer()
    convs = conversations()
    problems = check_disjoint(convs)
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
        "format": "r4dx-kl-chat-v1", "tokenizer": tok.describe(), "tokenizer_arch": "lfm2", "tokenizer_mode": "canonical",
        "tokenizer_provenance": prov, "add_bos": True, "bos_token_id": BOS_ID, "chat_template": True,
        "score_mask_semantics":
            "score_mask[i]=1 for ids that are model-turn tokens: for every assistant turn, the thought, </think>, the "
            "answer or the Pythonic tool call, and the closing <|im_end|> (the generation prompt ends after <think>, "
            "which is NOT scored). Row i of a logprob file predicts token i+1, so row i is scored when score_mask[i+1]=1.",
        "provenance": "All user prompts, thoughts, tool results and answers were written for this corpus (original "
                      "text; no third-party material). The ten conversations are those of chat_gemma.json (the "
                      "thoughts of the eight non-thinking ones are new short author-written ones). Disjoint from "
                      "tools/quant2/corpus_v2_prompts.json and kl_corpus/*.txt (word 6-gram / Thai 24-char shingle "
                      "check in build_chat_lfm2.py).",
        "gate_groups": "english, thai, python+cpp (reported as code), tool, and chat-ALL",
        "segments": segs,
    }
    out = HERE / "chat_lfm2.json"
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({sum(len(s['token_ids']) for s in segs)} tokens, {sum(s['scored_tokens'] for s in segs)} scored)")

    txt = {p.stem: p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.txt"))}
    raw_doc = {"tokenizer": tok.describe(), "tokenizer_mode": "canonical", "tokenizer_arch": "lfm2",
               "tokenizer_provenance": prov, "autotokenizer_matches_tokenizers_json": True,
               "add_special_tokens": False, "add_bos": True, "bos_token_id": BOS_ID, "chat_template": False}
    short = raw_segments(tok, sorted(txt.items()), 1024)
    (HERE / "tokens_lfm2.json").write_text(json.dumps({**raw_doc, "max_tokens": 1024, "segments": short}, indent=1),
                                           encoding="utf-8")
    long_text = "\n\n".join(txt[k] for k in ("english_prose", "python_source", "cpp_source", "thai_prose"))
    long_segs = raw_segments(tok, [("mixed_long", long_text)], 6144)
    (HERE / "tokens_lfm2_long.json").write_text(
        json.dumps({**raw_doc, "max_tokens": 6144, "segments": long_segs}, indent=1), encoding="utf-8")
    print("wrote tokens_lfm2.json, tokens_lfm2_long.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
