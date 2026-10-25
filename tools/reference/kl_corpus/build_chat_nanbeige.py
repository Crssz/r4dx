"""tools/reference/kl_corpus/build_chat_nanbeige.py  (docs/nanbeige-plan.md N0-7)

Builds the held-out KL corpus of the Nanbeige4.3-33B w8a16 gate, tokenized with the checkpoint's own tokenizer
(E:\\models\\Nanbeige\\Nanbeige4.3-33B-tok, `tokenizers` canonical mode):

  chat_nanbeige.json        PRIMARY: the twelve conversations of build_chat_lfm2.py (english x3 + english thinking, thai x2 +
                            thai thinking, python x2, cpp, two tool conversations) plus two Chinese conversations written for
                            this corpus (group zh: Nanbeige is a Chinese-first model), rendered with the checkpoint's chat
                            template (thinking on, preserve_thinking left undefined: every turn keeps its reasoning). Every
                            assistant turn is `<|im_start|>assistant\\n<think>\\n{thought}\\n</think>\\n\\n{answer}<|im_end|>` (a tool
                            turn: `...</think>\\n\\n<tool_call>...</tool_call><|im_end|>`); the scored span of a turn starts after
                            the generation prompt `<|im_start|>assistant\\n<think>\\n`. Scoring is on those spans only (`score_mask`).
  tokens_nanbeige.json      raw segments (english / python / cpp / thai text files of this directory), 1024 tokens each, each
                            starting with the BOS <|im_start|> (the tokenizer's post-processor adds it to raw text), reported
                            next to the chat set.
  tokens_nanbeige_long.json one long raw segment (all four text files, ~12k tokens, BOS first), for the late-position rows.

Provenance: prompts, thoughts, tool results and answers were written for this corpus (original text, released with the repo);
the set is disjoint from the calibration prompts and the raw KL text files (word 6-gram / Thai shingle check from
build_chat_lfm2.py; the Chinese conversations are checked with 12-character shingles).

Verification (asserted): the ids equal `AutoTokenizer.apply_chat_template(..., tokenize=True)` and the raw `tokenizers`
encoding; the score mask comes from the joint encoding's character offsets and no token crosses a scored span boundary
(encoding the pieces separately is NOT equivalent here: Metaspace prepend_scheme "first" would give each piece a leading
U+2581); the first assistant turn's prompt equals the template's generation prompt; every segment's last scored token is
`<|im_end|>`.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\kl_corpus\\build_chat_nanbeige.py
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

MODEL_DIR = Path(os.environ.get("R4DX_NANBEIGE_TOKENIZER_DIR", r"E:\models\Nanbeige\Nanbeige4.3-33B-tok"))
GEN_PROMPT_TAIL = "<|im_start|>assistant\n<think>\n"
IM_END = "<|im_end|>"
BOS_ID = 166100  # <|im_start|>: the post-processor's BOS for raw text

# Two Chinese conversations written for this corpus (original text).
ZH_SAMPLES = [
    {
        "name": "zh_tea", "group": "zh",
        "user": "\u6211\u60f3\u5728\u5bb6\u91cc\u6ce1\u4e00\u58f6\u597d\u559d\u7684\u7eff\u8336\uff0c\u6c34\u6e29\u548c\u65f6\u95f4\u5e94\u8be5\u600e\u4e48\u638c\u63e1\uff1f",
        "thought": "\u7528\u6237\u95ee\u5728\u5bb6\u6ce1\u7eff\u8336\u7684\u6c34\u6e29\u548c\u65f6\u95f4\u3002\u7eff\u8336\u5ae9\uff0c\u6c34\u592a\u70eb\u4f1a\u53d1\u82e6\uff0c"
                   "\u4e00\u822c\u7528\u516b\u5341\u5ea6\u5de6\u53f3\u7684\u6c34\u3002\u7b2c\u4e00\u6ce1\u4e00\u5206\u949f\u5de6\u53f3\uff0c\u540e\u9762\u6bcf\u6ce1"
                   "\u7a0d\u5fae\u5ef6\u957f\u3002\u8fd8\u53ef\u4ee5\u63d0\u4e00\u4e0b\u8336\u53f6\u7528\u91cf\u548c\u7528\u73bb\u7483\u676f\u3002",
        "answer": "\u6ce1\u7eff\u8336\u53ef\u4ee5\u6309\u8fd9\u51e0\u70b9\u6765\uff1a\n\n1. \u6c34\u6e29\uff1a\u7528\u70e7\u5f00\u540e\u653e\u51c9\u5230\u516b\u5341\u5ea6"
                  "\u5de6\u53f3\u7684\u6c34\uff0c\u592a\u70eb\u4f1a\u628a\u5ae9\u8336\u70eb\u719f\uff0c\u5473\u9053\u53d1\u82e6\u3002\n"
                  "2. \u7528\u91cf\uff1a\u5927\u7ea6\u4e09\u514b\u8336\u53f6\u914d\u4e00\u767e\u4e94\u5341\u6beb\u5347\u6c34\u3002\n"
                  "3. \u65f6\u95f4\uff1a\u7b2c\u4e00\u6ce1\u4e00\u5206\u949f\u5de6\u53f3\u5c31\u51fa\u6c64\uff0c\u4e4b\u540e\u6bcf\u6ce1\u591a\u52a0"
                  "\u4e8c\u5341\u79d2\uff0c\u4e00\u822c\u80fd\u6ce1\u4e09\u5230\u56db\u6b21\u3002\n\n\u7528\u900f\u660e\u7684\u73bb\u7483\u676f\u8fd8\u80fd"
                  "\u770b\u5230\u8336\u53f6\u5728\u6c34\u91cc\u6162\u6162\u8212\u5c55\u3002",
    },
    {
        "name": "zh_bike", "group": "zh",
        "user": "\u81ea\u884c\u8f66\u7684\u94fe\u6761\u603b\u662f\u54cd\uff0c\u6211\u8be5\u600e\u4e48\u5904\u7406\uff1f",
        "thought": "\u94fe\u6761\u54cd\u5e38\u89c1\u539f\u56e0\uff1a\u7f3a\u6cb9\u3001\u592a\u810f\u3001\u53d8\u901f\u6ca1\u8c03\u597d\u6216\u94fe\u6761\u78e8\u635f\u3002"
                   "\u5148\u6e05\u6d01\u518d\u4e0a\u6cb9\uff0c\u5982\u679c\u8fd8\u54cd\u5c31\u68c0\u67e5\u53d8\u901f\u5668\uff0c\u6700\u540e\u91cf\u4e00\u4e0b"
                   "\u94fe\u6761\u662f\u5426\u62c9\u957f\u3002\u6309\u987a\u5e8f\u5199\u6210\u51e0\u6b65\u3002",
        "answer": "\u53ef\u4ee5\u6309\u987a\u5e8f\u6392\u67e5\uff1a\n\n\u7b2c\u4e00\u6b65\uff0c\u7528\u65e7\u7259\u5237\u548c\u6e05\u6d01\u5242\u628a\u94fe\u6761\u4e0a\u7684"
                  "\u6ce5\u6c99\u5237\u5e72\u51c0\uff0c\u64e6\u5e72\u3002\n\u7b2c\u4e8c\u6b65\uff0c\u5728\u6bcf\u4e00\u8282\u94fe\u6761\u4e0a\u6ef4\u5c11\u91cf"
                  "\u94fe\u6761\u6cb9\uff0c\u8f6c\u52a8\u811a\u8e0f\u51e0\u5708\uff0c\u518d\u628a\u591a\u4f59\u7684\u6cb9\u64e6\u6389\u3002\n"
                  "\u7b2c\u4e09\u6b65\uff0c\u5982\u679c\u6362\u6321\u65f6\u6709\u6469\u64e6\u58f0\uff0c\u53ef\u80fd\u662f\u53d8\u901f\u5668\u504f\u4e86\uff0c"
                  "\u53ef\u4ee5\u5fae\u8c03\u8c03\u8282\u87ba\u4e1d\u6216\u53bb\u8f66\u5e97\u3002\n\u7b2c\u56db\u6b65\uff0c\u94fe\u6761\u7528\u4e45\u4e86\u4f1a"
                  "\u62c9\u957f\uff0c\u7528\u94fe\u6761\u5c3a\u91cf\u4e00\u4e0b\uff0c\u78e8\u635f\u592a\u5927\u5c31\u8be5\u6362\u65b0\u7684\u3002",
    },
]


def token_ids_sha256(ids) -> str:
    return sha256_bytes(json.dumps([int(t) for t in ids], separators=(",", ":")).encode("utf-8"))


def zh_conversations() -> list[dict]:
    out = []
    for s in ZH_SAMPLES:
        msgs = [{"role": "user", "content": s["user"]},
                {"role": "assistant", "content": s["answer"], "thinking": s["thought"]}]
        out.append({"name": s["name"], "group": s["group"], "tools": None, "messages": msgs,
                    "text_for_check": " ".join([s["user"], s["thought"], s["answer"]])})
    return out


def check_disjoint_zh(convs: list[dict]) -> list[str]:
    """12-character shingle overlap of the Chinese samples with the calibration prompts and the raw KL text files."""
    other = [json.dumps(json.loads((HERE.parents[1] / "quant2" / "corpus_v2_prompts.json").read_text(encoding="utf-8"))["samples"],
                        ensure_ascii=False)]
    other += [p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.txt"))]
    joined = re.sub(r"\s+", "", "".join(other))
    problems = []
    for c in convs:
        t = re.sub(r"\s+", "", c["text_for_check"])
        hits = [t[i:i + 12] for i in range(0, max(0, len(t) - 11)) if t[i:i + 12] in joined]
        if hits:
            problems.append(f"{c['name']} shares {len(hits)} 12-char shingles: {hits[:1]}")
    return problems


def template_messages(conv: dict) -> list[dict]:
    """The Nanbeige template reads `reasoning_content`; tool-call arguments stay dicts (it iterates them with |items)."""
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
    # The mask comes from the JOINT encoding's character offsets, not from encoding the pieces separately: Metaspace
    # (prepend_scheme "first") prepends U+2581 to whatever piece is encoded first, so a scored piece encoded alone would
    # gain a token the conversation does not have. Every token must lie wholly inside or outside a scored span.
    enc = tok._raw.encode(text, add_special_tokens=False)
    joint = list(enc.ids)
    assert joint == tok.encode(text)
    mask = []
    for (s0, s1) in enc.offsets:
        inside = [a <= s0 and s1 <= b for a, b in spans]
        crosses = [(s0 < a < s1) or (s0 < b < s1) for a, b in spans]
        assert not any(crosses), f"{conv['name']}: a token at chars [{s0},{s1}) crosses a scored span boundary"
        mask.append(1 if any(inside) and s1 > s0 else 0)
    ids = joint
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
            "mask_from_joint_offsets": True, "text": text}


def main() -> int:
    tok = RefTokenizer(MODEL_DIR, "canonical")
    convs = L.conversations()
    zh = zh_conversations()
    problems = L.check_disjoint(convs) + check_disjoint_zh(zh)
    if problems:
        print("\n".join(problems))
        raise SystemExit("corpus is not disjoint from the calibration/raw corpora")
    segs = []
    for c in convs + zh:
        s = build_chat_segment(tok, c)
        segs.append(s)
        print(f"{s['name']:15s} group={s['group']:8s} turns={s['n_assistant_turns']} prompt={s['prompt_len']:4d} "
              f"scored={s['scored_tokens']:4d} total={len(s['token_ids']):4d}")
    prov = tok.provenance()
    doc = {
        "format": "r4dx-kl-chat-v1", "tokenizer": tok.describe(), "tokenizer_arch": "llama", "tokenizer_mode": "canonical",
        "tokenizer_provenance": prov, "add_bos": False, "chat_template": True,
        "score_mask_semantics":
            "score_mask[i]=1 for ids that are model-turn tokens: for every assistant turn, the thought, </think>, the answer or the "
            "tool call, and the closing <|im_end|> (the generation prompt ends after <think>\\n, which is NOT scored). Row i of a "
            "logprob file predicts token i+1, so row i is scored when score_mask[i+1]=1.",
        "provenance": "All user prompts, thoughts, tool results and answers were written for this corpus (original text; no "
                      "third-party material); the twelve conversations of chat_lfm2.json plus two Chinese ones (zh_tea, zh_bike), "
                      "rendered with the Nanbeige4.3 chat template. Disjoint from tools/quant2/corpus_v2_prompts.json and "
                      "kl_corpus/*.txt.",
        "gate_groups": "english, thai, python+cpp (reported as code), tool, zh, and chat-ALL",
        "segments": segs,
    }
    out = HERE / "chat_nanbeige.json"
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({sum(len(s['token_ids']) for s in segs)} tokens, {sum(s['scored_tokens'] for s in segs)} scored)")

    txt = {p.stem: p.read_text(encoding="utf-8") for p in sorted(HERE.glob("*.txt"))}
    raw_doc = {"tokenizer": tok.describe(), "tokenizer_mode": "canonical", "tokenizer_arch": "llama", "tokenizer_provenance": prov,
               "add_special_tokens": False, "bos_prepended": BOS_ID, "chat_template": False}

    def raw(sources, max_tokens):
        out_segs = []
        for name, text in sources:
            ids = [BOS_ID] + tok.encode(text)[:max_tokens - 1]
            out_segs.append({"name": name, "token_ids": [int(i) for i in ids]})
            print(f"{name:20s} {len(ids):6d} tokens (sha256 {token_ids_sha256(ids)[:16]}...)")
        return out_segs

    short = raw(sorted(txt.items()), 1024)
    (HERE / "tokens_nanbeige.json").write_text(json.dumps({**raw_doc, "max_tokens": 1024, "segments": short}, indent=1), encoding="utf-8")
    long_text = "\n\n".join(txt[k] for k in ("english_prose", "python_source", "cpp_source", "thai_prose"))
    (HERE / "tokens_nanbeige_long.json").write_text(
        json.dumps({**raw_doc, "max_tokens": 16384, "segments": raw([("mixed_long", long_text)], 16384)}, indent=1), encoding="utf-8")
    print("wrote tokens_nanbeige.json, tokens_nanbeige_long.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
