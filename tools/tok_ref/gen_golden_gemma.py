#!/usr/bin/env python
"""Generates tests/tokenizer/golden_gemma.json, the reference corpus r4dx's C++ SPM-style tokenizer
backend and Gemma chat-template rendering are checked against (CTest 'tokenizer_golden_gemma',
tests/tokenizer/golden_test.cpp). Also carries tool-call parser vectors for the server's Gemma
parser (docs/gemma4-plan.md M1-13); the C++ golden test ignores those.

CPU only, no GPU touched:

    python tools\\tok_ref\\gen_golden_gemma.py [--model-dir DIR] [--out PATH]

The tokenizer dir (default D:\\models\\Huihui-gemma-4-12B-it-abliterated-tok, or the
R4DX_GEMMA_TOKENIZER_DIR environment variable) holds google/gemma-4-12B-it's tokenizer.json,
tokenizer_config.json, chat_template.jinja and generation_config.json (the Huihui repo's
tokenizer.json is byte-identical; its chat_template.jinja is older, see the plan).

Ground truth, and the one trap this script exists to defend against (docs/gemma4-plan.md 5.1):
the raw Rust `tokenizers.Tokenizer.from_file(tokenizer.json)` and `transformers.AutoTokenizer`
(the runtime GemmaTokenizer class, which can add its own overrides -- Qwen had exactly this
trap) are BOTH run for every encode/decode case and the script ABORTS if they ever disagree, so a
disagreement can never be silently baked into the golden file.

Case kinds written ("cases"):
  encode  : text, parse_special, ids, decoded (decode(ids, skip_special_tokens=True))
  decode  : ids -> text. `skip_special_tokens`, and `keep_special` (apply the header's
            keep_special_on_decode list on top of skip_special_tokens=True). `stream` says whether
            the incremental StreamDecoder must reproduce `expected` token by token (false only for
            malformed byte-fallback runs, where HF replaces a whole run and a streaming decoder
            necessarily replaces byte by byte).
  chat    : messages / tools / add_generation_prompt / extra_context, the rendered `prompt`
            (apply_chat_template(tokenize=False)) and its `ids` (encode, special tokens parsed).
  chat_error : a render that must raise (the C++ render must throw too).
Parser vectors ("parser_cases"): text -> `tok.parse_response(text)` result, or the error HF raises.

Header fields (checked by the C++ test): vocab_size, bos/eos/pad ids, eos_token_ids,
keep_special_on_decode, apply_polyfills (false for Gemma; see ChatTemplateOptions).
"""
import argparse
import json
import os
import random
import sys

DEFAULT_DIR = os.environ.get("R4DX_GEMMA_TOKENIZER_DIR", r"D:\models\Huihui-gemma-4-12B-it-abliterated-tok")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, "tests", "tokenizer", "golden_gemma.json")

# Special added tokens the server needs to see in decoded text (reasoning / tool-call markers).
KEEP_SPECIAL = ["<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", '<|"|>']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    model_dir = args.model_dir

    from tokenizers import Tokenizer as RawTokenizer
    from transformers import AutoTokenizer

    raw = RawTokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    auto = AutoTokenizer.from_pretrained(model_dir)
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    added = {a["content"]: a["id"] for a in tj["added_tokens"]}
    special_ids = {a["id"] for a in tj["added_tokens"] if a["special"]}
    keep_ids = {added[t] for t in KEEP_SPECIAL}

    cases = []
    parser_cases = []
    mismatches = []

    def ids_both(name, text, parse_special):
        # r4dx parse_special=True <-> raw encode_special_tokens=False <-> HF split_special_tokens=False.
        raw.encode_special_tokens = not parse_special
        r = raw.encode(text, add_special_tokens=False).ids
        a = auto.encode(text, add_special_tokens=False, split_special_tokens=not parse_special)
        if r != a:
            mismatches.append((name, text[:80], r[:20], a[:20]))
        return r

    def add_encode(name, text, parse_special=False):
        ids = ids_both(name, text, parse_special)
        dec = raw.decode(ids, skip_special_tokens=True)
        dec_auto = auto.decode(ids, skip_special_tokens=True)
        if dec != dec_auto:
            mismatches.append((name + " (decode)", dec[:80], dec_auto[:80], None))
        cases.append({"kind": "encode", "name": name, "text": text, "parse_special": parse_special,
                      "ids": ids, "decoded": dec})

    def add_decode(name, ids, skip_special_tokens, keep_special=False, stream=True):
        if keep_special:
            use = [i for i in ids if i not in special_ids or i in keep_ids]
            expected = raw.decode(use, skip_special_tokens=False)
            expected_auto = auto.decode(use, skip_special_tokens=False)
        else:
            expected = raw.decode(ids, skip_special_tokens=skip_special_tokens)
            expected_auto = auto.decode(ids, skip_special_tokens=skip_special_tokens)
        if expected != expected_auto:
            mismatches.append((name, expected[:80], expected_auto[:80], None))
        cases.append({"kind": "decode", "name": name, "ids": ids,
                      "skip_special_tokens": skip_special_tokens, "keep_special": keep_special,
                      "stream": stream, "expected": expected})

    def add_chat(name, messages, tools=None, add_generation_prompt=True, **extra):
        kwargs = dict(add_generation_prompt=add_generation_prompt, tokenize=False)
        if tools is not None:
            kwargs["tools"] = tools
        kwargs.update(extra)
        prompt = auto.apply_chat_template(messages, **kwargs)
        ids = ids_both(name, prompt, True)
        cases.append({"kind": "chat", "name": name, "messages": messages, "tools": tools,
                      "add_generation_prompt": add_generation_prompt,
                      "extra_context": dict(extra), "prompt": prompt, "ids": ids})

    def add_chat_error(name, messages, tools=None, **extra):
        kwargs = dict(add_generation_prompt=True, tokenize=False)
        if tools is not None:
            kwargs["tools"] = tools
        kwargs.update(extra)
        try:
            auto.apply_chat_template(messages, **kwargs)
        except Exception as e:  # noqa: BLE001 - any raise is the expected behaviour
            cases.append({"kind": "chat_error", "name": name, "messages": messages, "tools": tools,
                          "add_generation_prompt": True, "extra_context": dict(extra),
                          "hf_error": str(e)[:200]})
            return
        raise SystemExit(f"chat_error case {name} did not raise in HF")

    # ---- English ---------------------------------------------------------------------------
    english = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I'm not sure this'll work, but we'll see.",
        "She said, \"I can't believe it's already Friday!\"",
        "r4dx targets a single AMD Radeon AI PRO R9700 on Windows 11.",
        "   leading and trailing spaces   ",
        " ",
        "  ",
        "CamelCaseIdentifierExample snake_case_identifier SCREAMING_SNAKE kebab-case-id",
        "A sentence that ends without punctuation",
        "Multiple   spaces   between   words",
        "One.Two.Three.Four",
        "email@example.com and https://example.com/path?query=1&x=2",
        "hi",
        "",
        "a",
        "\u00a0non-breaking\u00a0spaces\u00a0",
        "x" + " " * 40 + "y",
        " " * 70,
    ]
    for i, t in enumerate(english):
        add_encode(f"english_{i}", t)

    # ---- Code ------------------------------------------------------------------------------
    code = [
        "def bpe_merge(word: str) -> list[int]:\n    return [ord(c) for c in word]\n",
        "function add(a, b) {\n  return a + b;\n}\n",
        "#include <vector>\nstd::vector<int> v = {1, 2, 3};\n",
        "for (int i = 0; i < 10; ++i) { total += i * i; }",
        "if x is None:\n\tpass\nelse:\n\treturn x + 1",
        "SELECT * FROM tokens WHERE id > 248044 ORDER BY id;",
        "class Foo(Bar, metaclass=ABCMeta):\n    __slots__ = ('a', 'b')\n",
        "printf(\"%d + %d = %d\\n\", a, b, a + b);",
        "git commit -m \"fix: handle empty merges list\" --no-verify",
        "\n\n    def f(self):\n\n\n        pass\n\n",
    ]
    for i, t in enumerate(code):
        add_encode(f"code_{i}", t)

    # ---- JSON ------------------------------------------------------------------------------
    json_blobs = [
        '{"role": "user", "content": "hi"}',
        '{"a": 1, "b": [1, 2, 3], "c": {"nested": true, "x": null}}',
        '[]',
        '{"unicode": "caf\u00e9 \u2014 \u4f60\u597d"}',
        '{"escaped": "line1\\nline2\\ttabbed"}',
        '{"num": -1.5e10, "big": 123456789012345}',
        json.dumps({"k%d" % i: {"v": [i, str(i)]} for i in range(12)}, indent=2),
    ]
    for i, t in enumerate(json_blobs):
        add_encode(f"json_{i}", t)

    # ---- Thai ------------------------------------------------------------------------------
    thai = [
        "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01",
        "\u0e1c\u0e21\u0e0a\u0e37\u0e48\u0e2d\u0e19\u0e23\u0e30\u0e27\u0e34\u0e17",
        "\u0e27\u0e31\u0e19\u0e19\u0e35\u0e49\u0e2d\u0e32\u0e01\u0e32\u0e28\u0e14\u0e35\u0e21\u0e32\u0e01",
        "\u0e01\u0e23\u0e38\u0e07\u0e40\u0e17\u0e1e\u0e21\u0e2b\u0e32\u0e19\u0e04\u0e23",
        "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13\u0e21\u0e32\u0e01\u0e04\u0e23\u0e31\u0e1a",
        "\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02 \u0e51\u0e52\u0e53\u0e54\u0e55",
        "Mixed \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22 and English \u0e43\u0e19\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04\u0e40\u0e14\u0e35\u0e22\u0e27",
    ]
    for i, t in enumerate(thai):
        add_encode(f"thai_{i}", t)

    # ---- Chinese / Japanese / Korean / Arabic / Hindi / Cyrillic / Greek --------------------
    chinese = [
        "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
        "\u6211\u7231\u4eba\u5de5\u667a\u80fd\u548c\u673a\u5668\u5b66\u4e60",
        "\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u53e5\u5b50\uff0c\u5305\u542b\u6807\u70b9\u7b26\u53f7\u3001\u6570\u5b57123\u3002",
        "\u7b80\u4f53\u4e2d\u6587\u548c\u7e41\u9ad4\u4e2d\u6587\u90fd\u9700\u8981\u652f\u6301\u3002",
    ]
    for i, t in enumerate(chinese):
        add_encode(f"chinese_{i}", t)
    japanese = [
        "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\uff01",
        "\u30ab\u30bf\u30ab\u30ca\u3068\u3072\u3089\u304c\u306a\u3068\u6f22\u5b57\u306e\u6df7\u5728\u6587\u3067\u3059\u3002",
        "\u6570\u5b57\uff11\uff12\uff13\u3068\u82f1\u6570\u5b57123\u306e\u6bd4\u8f03\u3002",
    ]
    for i, t in enumerate(japanese):
        add_encode(f"japanese_{i}", t)
    other_scripts = [
        "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4",                                  # Korean
        "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",      # Arabic
        "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",      # Hindi
        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0422\u0435\u0441\u0442.",  # Russian
        "\u0393\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5 \u03ba\u03cc\u03c3\u03bc\u03b5",    # Greek
        "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",                              # Hebrew
    ]
    for i, t in enumerate(other_scripts):
        add_encode(f"script_{i}", t)

    # ---- Emoji -----------------------------------------------------------------------------
    emoji = [
        "Hello \U0001F600 world \U0001F30D!",
        "\U0001F600\U0001F601\U0001F602\U0001F923\U0001F60A",
        "Family: \U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",
        "Flags: \U0001F1F9\U0001F1ED \U0001F1FA\U0001F1F8 \U0001F1EF\U0001F1F5",
        "Skin tone: \U0001F44B\U0001F3FD",
        "Mixed \u4f60\u597d \U0001F600 \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \U0001F30F emoji test",
    ]
    for i, t in enumerate(emoji):
        add_encode(f"emoji_{i}", t)

    # ---- Whitespace: newline / space / tab runs, \r\n --------------------------------------
    for n in list(range(1, 71)) + [99, 100, 127, 128, 200, 300]:
        add_encode(f"newlines_{n}", "\n" * n)
    for n in (1, 2, 3, 30, 31, 32, 33, 62, 63, 64, 100):
        add_encode(f"nl_between_{n}", "a" + "\n" * n + "b")
        add_encode(f"nl_lead_{n}", "\n" * n + " indented")
        add_encode(f"nl_trail_{n}", "text " + "\n" * n)
    for n in (1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64, 70):
        add_encode(f"spaces_{n}", " " * n)
        add_encode(f"spaces_between_{n}", "a" + " " * n + "b")
    for n in (1, 2, 3, 8, 31, 32, 33):
        add_encode(f"tabs_{n}", "\t" * n)
    for n in (1, 2, 5, 31, 32):
        add_encode(f"mixed_nl_space_{n}", ("\n" + " " * 3) * n)
    whitespace = [
        "line1\nline2\nline3",
        "line1\r\nline2\r\nline3",
        "\r\n\r\n",
        "\r",
        "tab\there\tand\tthere",
        "trailing spaces on a line   \nand another line",
        "mix\t \t of \n\t tabs \t\tand\nnewlines",
        "    four-space indent\n        eight-space indent",
        "a\n\nb\n\n\nc",
        "trailing newline\n",
        "\u2028\u2029 unicode line separators \u2028",
        "\x0b\x0c form feed and vertical tab",
    ]
    for i, t in enumerate(whitespace):
        add_encode(f"whitespace_{i}", t)

    # ---- Byte fallback: codepoints absent from the vocab, controls, odd Unicode --------------
    fallback = [
        "\x00", "\x01\x02\x03", "\x1b[31mred\x1b[0m", "\x7f",
        "\ue000", "\ue000\ue001\ue002",                      # private use
        "\U0001FAE0 melting face", "\U0001FAE8",              # newer emoji
        "\U00010000\U0010FFFF",                               # first / last supplementary
        "\ufdd0\ufdef\uffff",                                 # noncharacters
        "\ufeff BOM in text", "\u200b\u200c\u200d\u2060",     # zero widths
        "\u0e01\u0e34\u0e49\u0e49\u0e49\u0e49",              # stacked Thai marks
        "\u4e00\u9fa5\u3400\u4dbf\U00020000\U0002A6DF",       # CJK extension ranges
        "\ud7a3\ud7b0\ud7c6",                                 # Hangul Jamo ext
        "\u10ffff"[:1] + "x",
        "\u0300combining at start", "e\u0301 decomposed e-acute",
        "A\u030a vs \u00c5", "\u1e9b\u0323", "\ufb01 ligature",
    ]
    for i, t in enumerate(fallback):
        add_encode(f"fallback_{i}", t)

    # Non-NFC input: Gemma's normalizer is only Replace(" " -> U+2581), so decomposed text must
    # tokenize as given (the Qwen tokenizer's NFC known gap does not exist here).
    nfd = [
        "caf\u0065\u0301", "\u0061\u0308\u0302", "\u0041\u030a ngstr\u00f6m",
        "\u0e01\u0e33", "\u1100\u1161\u11a8",  # Hangul jamo
    ]
    for i, t in enumerate(nfd):
        add_encode(f"nonnfc_{i}", t)

    # ---- Long words / numbers ------------------------------------------------------------------
    long_words = [
        "supercalifragilisticexpialidocious",
        "a" * 200,
        "x" * 64 + "y" * 64 + "z" * 64,
        "".join(chr(ord('a') + (i % 26)) for i in range(300)),
        "ThisIsAVeryLongCamelCaseIdentifierThatKeepsGoingAndGoingAndGoing",
        "0", "42", "-17", "3.14159265358979", "1234567890123456789",
        "The year 2026 has 365 days and 8760 hours.",
    ]
    for i, t in enumerate(long_words):
        add_encode(f"longword_{i}", t)

    # ---- A 20 KB single line (the whole line is one BPE word; guards the id-keyed merge loop) --
    rng = random.Random(20260930)
    words = ["the", "model", "tensor", "attention", "layer", "r4dx", "GPU", "cache", "token", "merge",
             "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35", "\u4f60\u597d", "caf\u00e9", "x=1;", "(a+b)", "{\"k\":1}"]
    line = " ".join(rng.choice(words) for _ in range(4000))
    line = line[:20480]
    add_encode("line_20kb_words", line)
    add_encode("line_20kb_one_char", "ab" * 10240)
    add_encode("line_5kb_nospace", "".join(rng.choice("abcdefghij") for _ in range(5000)))
    add_encode("multi_line_text", "\n".join(" ".join(rng.choice(words) for _ in range(rng.randint(0, 20)))
                                            for _ in range(200)))

    # ---- Seeded random fuzz ----------------------------------------------------------------------
    alphabet = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") +
                [" "] * 12 + ["\n"] * 5 + ["\t", "\r", ".", ",", "!", "?", "'", '"', "-", "_", "/", "(", ")", "{", "}"] +
                list("\u00e9\u00fc\u00f1\u4f60\u597d\u0e01\u0e34\u0e49\u0e32\U0001F600\u2581\u2018\u2019"))
    for i in range(150):
        n = rng.randint(1, 160)
        add_encode(f"fuzz_{i}", "".join(rng.choice(alphabet) for _ in range(n)))
    for i in range(20):  # fuzz with special-token surface text spliced in
        parts = []
        for _ in range(rng.randint(2, 6)):
            parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12))))
            parts.append(rng.choice(list(added)))
        t = "".join(parts)
        add_encode(f"fuzz_special_notparsed_{i}", t, parse_special=False)
        add_encode(f"fuzz_special_parsed_{i}", t, parse_special=True)

    # ---- Special tokens embedded as text (parse_special gating) -------------------------------------
    special_texts = [
        "<bos>hello", "<eos>", "<pad><unk><mask>",
        "<|turn>user\nhi<turn|>\n<|turn>model\n",
        "<|channel>thought\nreasoning<channel|>answer",
        '<|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|>',
        "<|tool>declaration:f{}<tool|>", "<|tool_response>response:f{v:1}<tool_response|>",
        "<|think|>", "<|image><|image|><|image|><image|>", "<|audio><|audio|><audio|>", "<|video|>",
        "<turn|><turn|>", "<bos><bos>", "text<|turn>text<|turn>",
        "<unused0> <start_of_turn> not special", "<|channel", "<channel|", "<<turn|>>",
    ]
    for i, t in enumerate(special_texts):
        add_encode(f"special_notparsed_{i}", t, parse_special=False)
        add_encode(f"special_parsed_{i}", t, parse_special=True)

    # ---- Decode cases ---------------------------------------------------------------------------------
    byte_id = {b: raw.token_to_id("<0x%02X>" % b) for b in range(256)}
    mixed = raw.encode('<|channel>thought\nok<channel|>Hi <|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|> <turn|>',
                       add_special_tokens=False).ids
    raw.encode_special_tokens = False
    mixed = raw.encode('<|channel>thought\nok<channel|>Hi <|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|> <turn|><eos>',
                       add_special_tokens=False).ids
    add_decode("decode_mixed_skip", mixed, True)
    add_decode("decode_mixed_keep", mixed, True, keep_special=True)
    add_decode("decode_mixed_noskip", mixed, False)
    add_decode("decode_empty", [], True)
    euro = [byte_id[0xE2], byte_id[0x82], byte_id[0xAC]]
    add_decode("decode_bytes_valid", euro, True)
    add_decode("decode_bytes_valid_in_text", mixed[:3] + euro + mixed[3:6], True)
    add_decode("decode_bytes_two_valid_runs", euro + [raw.token_to_id("a")] + euro, True)
    add_decode("decode_bytes_emoji", [byte_id[b] for b in "\U0001F600".encode("utf-8")], True)
    add_decode("decode_bytes_truncated", euro[:2], True, stream=False)
    add_decode("decode_bytes_lone_lead", euro[:1], True, stream=False)
    add_decode("decode_bytes_lone_cont", [byte_id[0x82]], True, stream=False)
    add_decode("decode_bytes_invalid_ff", [byte_id[0xFF], byte_id[0x41]], True, stream=False)
    add_decode("decode_bytes_overlong", [byte_id[0xC0], byte_id[0x80]], True, stream=False)
    add_decode("decode_bytes_surrogate", [byte_id[0xED], byte_id[0xA0], byte_id[0x80]], True, stream=False)
    add_decode("decode_bytes_split_by_text", euro[:2] + [raw.token_to_id("a")] + euro[2:], True, stream=False)
    add_decode("decode_bytes_split_by_skipped_special", euro[:2] + [added["<|turn>"]] + euro[2:], True, stream=False)
    add_decode("decode_bytes_split_by_kept_special", euro[:2] + [added["<|channel>"]] + euro[2:], True,
               keep_special=True, stream=False)
    add_decode("decode_space_marks", [raw.token_to_id("\u2581"), raw.token_to_id("\u2581\u2581"),
                                       raw.token_to_id("a"), raw.token_to_id("\u2581the")], True)
    add_decode("decode_newline_runs", [raw.token_to_id("\n" * k) for k in (1, 2, 31)], True)

    # ---- Chat template renders ----------------------------------------------------------------------------
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "Zone": {"type": "string", "description": "mixed-case key sorts after 'city' under Jinja dictsort"},
                    "alpha": {"type": "integer"},
                    "Beta": {"type": "boolean", "nullable": True},
                },
                "required": ["city"],
            },
        },
    }
    nested_tool = {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "location": {
                        "type": "object",
                        "description": "Where.",
                        "properties": {
                            "room": {"type": "string"},
                            "Floor": {"type": "integer"},
                            "geo": {"type": "object", "properties": {"lat": {"type": "number"}, "lng": {"type": "number"}},
                                    "required": ["lat", "lng"]},
                        },
                        "required": ["room"],
                    },
                    "reminders": {"type": "array", "items": {"type": "object", "properties": {"minutes": {"type": "integer"}},
                                                              "required": ["minutes"]}},
                },
                "required": ["title", "attendees"],
            },
            "response": {"type": "object", "description": "The created event."},
        },
    }
    simple_tool = {
        "type": "function",
        "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}},
    }

    add_chat("chat_user_only", [{"role": "user", "content": "What is the capital of Thailand?"}])
    add_chat("chat_user_only_think", [{"role": "user", "content": "What is the capital of Thailand?"}],
             enable_thinking=True)
    add_chat("chat_system_user", [
        {"role": "system", "content": "  You are a concise, helpful assistant.  "},
        {"role": "user", "content": "  What is the capital of Thailand?\n"},
    ])
    add_chat("chat_system_user_think", [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "What is the capital of Thailand?"},
    ], enable_thinking=True)
    add_chat("chat_developer_role", [
        {"role": "developer", "content": "Always answer in Thai."},
        {"role": "user", "content": "Hello"},
    ])
    add_chat("chat_system_content_parts", [
        {"role": "system", "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": "Part two."}]},
        {"role": "user", "content": [{"type": "text", "text": "Hi there"}]},
    ])
    add_chat("chat_multiturn", [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "Tell me a short joke."},
        {"role": "assistant", "content": "Why did the GPU cross the road? To get to the other thread."},
        {"role": "user", "content": "Good one. Now explain it."},
    ])
    add_chat("chat_multiturn_think", [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "Tell me a short joke."},
    ], enable_thinking=True)
    add_chat("chat_no_generation_prompt", [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hi!"},
    ], add_generation_prompt=False)
    add_chat("chat_assistant_last_continue", [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hi!"},
        {"role": "assistant", "content": "And also hello."},
    ], add_generation_prompt=False)
    add_chat("chat_empty_assistant", [
        {"role": "user", "content": "Say nothing."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Again."},
    ])
    add_chat("chat_assistant_with_thinking_in_content", [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<|channel>thought\nadd them<channel|>It is 4."},
        {"role": "user", "content": "3+3?"},
    ])
    add_chat("chat_unicode_content", [
        {"role": "user", "content": "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \U0001F600 \u4f60\u597d"},
    ])

    # reasoning_content / preserve_thinking
    reasoning_msgs = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "It's 4.", "reasoning_content": "Two plus two is four."},
        {"role": "user", "content": "What about 3+3?"},
    ]
    add_chat("chat_reasoning_default", reasoning_msgs, enable_thinking=True)
    add_chat("chat_reasoning_preserve_true", reasoning_msgs, enable_thinking=True, preserve_thinking=True)
    add_chat("chat_reasoning_preserve_false", reasoning_msgs, enable_thinking=True, preserve_thinking=False)
    add_chat("chat_reasoning_after_last_user", [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "It's 4.", "reasoning_content": "Two plus two is four."},
    ], add_generation_prompt=False, enable_thinking=True)
    add_chat("chat_reasoning_key_reasoning", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "reasoning": "greet back"},
    ], add_generation_prompt=False)

    # tools
    add_chat("chat_with_tool", [{"role": "user", "content": "What's the weather in Bangkok right now?"}],
             tools=[weather_tool])
    add_chat("chat_with_tool_think", [{"role": "user", "content": "What's the weather in Bangkok right now?"}],
             tools=[weather_tool], enable_thinking=True)
    add_chat("chat_with_tool_and_system", [
        {"role": "system", "content": "Use tools when helpful."},
        {"role": "user", "content": "Weather in Bangkok?"},
    ], tools=[weather_tool])
    add_chat("chat_nested_tool_schema", [{"role": "user", "content": "Book a meeting."}], tools=[nested_tool])
    add_chat("chat_two_tools", [{"role": "user", "content": "Hi"}], tools=[weather_tool, simple_tool])

    tc_weather = {"id": "call_1", "type": "function",
                  "function": {"name": "get_weather", "arguments": {"city": "Bangkok", "units": "celsius"}}}
    add_chat("chat_tool_call_and_response", [
        {"role": "user", "content": "What's the weather in Bangkok right now?"},
        {"role": "assistant", "content": "", "tool_calls": [tc_weather]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 33, \"condition\": \"partly cloudy\"}"},
    ], tools=[weather_tool])
    add_chat("chat_tool_call_and_response_think", [
        {"role": "user", "content": "What's the weather in Bangkok right now?"},
        {"role": "assistant", "content": "", "tool_calls": [tc_weather]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 33}"},
    ], tools=[weather_tool], enable_thinking=True)
    add_chat("chat_tool_call_then_answer", [
        {"role": "user", "content": "What's the weather in Bangkok right now?"},
        {"role": "assistant", "content": "", "tool_calls": [tc_weather]},
        {"role": "tool", "tool_call_id": "call_1", "content": "33C and cloudy"},
        {"role": "assistant", "content": "It is 33C and partly cloudy in Bangkok."},
        {"role": "user", "content": "Thanks!"},
    ], tools=[weather_tool])
    add_chat("chat_tool_call_with_reasoning_preserved", [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool.",
         "tool_calls": [tc_weather]},
        {"role": "tool", "tool_call_id": "call_1", "content": "33C"},
    ], tools=[weather_tool], enable_thinking=True, preserve_thinking=True)
    add_chat("chat_parallel_tool_calls", [
        {"role": "user", "content": "Weather in Bangkok and Tokyo?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Bangkok"}}},
            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "33C"},
        {"role": "tool", "tool_call_id": "c2", "content": "18C"},
    ], tools=[weather_tool])
    add_chat("chat_tool_response_out_of_order", [
        {"role": "user", "content": "Weather in Bangkok and Tokyo?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Bangkok"}}},
            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}},
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": "18C"},
        {"role": "tool", "tool_call_id": "c1", "content": "33C"},
    ], tools=[weather_tool])
    add_chat("chat_tool_call_argument_types", [
        {"role": "user", "content": "Do it."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {
            "name": "create_event",
            "arguments": {"title": "Standup", "attendees": ["a", "b"], "n": 3, "pi": 3.5, "ok": True, "none": None,
                          "Zed": "z", "apple": "a",
                          "location": {"room": "R1", "Floor": 2, "geo": {"lng": 100.5, "lat": 13.7}},
                          "reminders": [{"minutes": 5}, {"minutes": 10}], "quote": 'say "hi"'}}}]},
        {"role": "tool", "tool_call_id": "x", "content": "created"},
    ], tools=[nested_tool])
    add_chat("chat_tool_call_no_args", [
        {"role": "user", "content": "Ping."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "p", "type": "function", "function": {"name": "ping", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "p", "content": "pong"},
    ], tools=[simple_tool])
    add_chat("chat_tool_response_mapping", [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [tc_weather]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "33C"}]},
    ], tools=[weather_tool])
    add_chat("chat_tool_responses_legacy", [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [tc_weather],
         "tool_responses": [{"name": "get_weather", "response": {"temp_c": 33, "Cond": "cloudy", "ok": True}}]},
    ], tools=[weather_tool])
    add_chat("chat_tool_message_without_call", [
        {"role": "user", "content": "Weather?"},
        {"role": "tool", "tool_call_id": "zz", "content": "orphan"},
    ])
    add_chat_error("chat_error_string_arguments", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
    ])

    # multimodal content parts
    add_chat("chat_image_part", [{"role": "user", "content": [
        {"type": "text", "text": "Describe this:"}, {"type": "image"}]}])
    add_chat("chat_image_url_part", [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "What is it?"}]}])
    add_chat("chat_audio_part", [{"role": "user", "content": [
        {"type": "audio"}, {"type": "text", "text": "Transcribe."}]}])
    add_chat("chat_image_and_audio", [{"role": "user", "content": [
        {"type": "image"}, {"type": "audio"}, {"type": "text", "text": "Both?"}]}])

    # ---- Parser vectors (for the server's Gemma tool-call parser, M1-13) -------------------------------------------------
    parse_inputs = [
        "Hello there",
        "<|channel>thought\nx<channel|>The answer",
        "<|channel>thought\nI should check.<channel|><|tool_call>call:get_weather{city:<|\"|>Bangkok<|\"|>}<tool_call|>",
        '<|tool_call>call:get_weather{city:<|"|>Bangkok<|"|>,units:<|"|>celsius<|"|>}<tool_call|>',
        '<|tool_call>call:f{a:1,b:2.5,c:true,d:false,e:null}<tool_call|>',
        '<|tool_call>call:f{list:[1,2,3],obj:{a:<|"|>x<|"|>,b:[<|"|>y<|"|>]}}<tool_call|>',
        '<|tool_call>call:f{}<tool_call|>',
        '<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:<|"|>z<|"|>}<tool_call|>',
        'Sure, calling it.<|tool_call>call:f{x:1}<tool_call|>',
        '<|tool_call>call:f{s:<|"|>has, comma and } brace<|"|>}<tool_call|>',
        '<|tool_call>call:f{s:<|"|><|"|>}<tool_call|>',
        '<|tool_call>call:f{n:-3,m:1e3}<tool_call|>',
        '<|tool_call>call:f{a:bare}<tool_call|>',
        '<|tool_call>call:f{a:1<tool_call|>',
        '<|tool_call>call:f{a:1}',
        '<|tool_call>not a call<tool_call|>',
        '<|tool_call>call:f{a:1}<tool_call|><turn|>',
        'The answer is 4.<turn|>',
        '<|channel>thought\nunterminated thinking',
    ]
    for text in parse_inputs:
        entry = {"text": text}
        try:
            entry["expected"] = auto.parse_response(text)
        except Exception as e:  # noqa: BLE001 - HF raising is itself the recorded behaviour
            entry["hf_error"] = str(e).splitlines()[0][:200]
        parser_cases.append(entry)

    if mismatches:
        print("ABORT: raw tokenizers and AutoTokenizer disagree on %d case(s):" % len(mismatches), file=sys.stderr)
        for m in mismatches[:20]:
            print("  ", m, file=sys.stderr)
        return 1

    out = {
        "generated_by": "tools/tok_ref/gen_golden_gemma.py",
        "model_dir": model_dir,
        "tokenizer_class": type(auto).__name__,
        "kind": "spm_byte_fallback",
        "vocab_size": len(auto),
        "bos_token_id": auto.bos_token_id,
        "eos_token_id": auto.eos_token_id,
        "pad_token_id": auto.pad_token_id,
        "eos_token_ids": json.load(open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8"))["eos_token_id"],
        "keep_special_on_decode": KEEP_SPECIAL,
        "apply_polyfills": False,
        "bos_on_raw_prompt": bool(auto("hi").input_ids[:1] == [auto.bos_token_id]),
        "cases": cases,
        "parser_cases": parser_cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {args.out}: {kinds}, {len(parser_cases)} parser cases; raw == AutoTokenizer on every case")
    return 0


if __name__ == "__main__":
    sys.exit(main())
