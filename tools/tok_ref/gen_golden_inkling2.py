#!/usr/bin/env python
"""Generates tests/tokenizer/golden_inkling2.json, the reference corpus r4dx's C++ tokenizer is checked against for
OpenUncensored/Inkling-2-Mini-Uncensored (CTest 'tokenizer_golden_inkling2', tests/tokenizer/golden_test.cpp).

CPU only. Run with the reference venv, -I:

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\tok_ref\\gen_golden_inkling2.py

Reads <R4DX_INKLING2_TOKENIZER_DIR | E:\\models\\OpenUncensored\\Inkling-2-Mini-Uncensored> (tokenizer.json,
tokenizer_config.json, generation_config.json; read-only).

Ground truth: the raw Rust backend (tokenizers.Tokenizer.from_file(tokenizer.json)); the script asserts that
AutoTokenizer (PreTrainedTokenizerFast) produces identical ids for every encode case, then writes the raw ids.

Facts this checks: byte-level BPE with ignore_merges, no normalizer (non-NFC text is NOT normalized), the o200k-style
pre-tokenizer regex (digits grouped by 3, Lu/Lt/Lm/Lo/M | Ll/Lm/Lo/M letter split, case-insensitive contraction
suffixes, `[\\r\\n/]*` after a symbol run), no automatic BOS, 60 special added tokens 199998..200057 (recognised only
with parse_special).

The chat template is covered by gen_chat_golden_inkling2.py. Case kinds: "encode", "decode" (golden_test.cpp).
"""
import json
import os
import random
import sys

MODEL_DIR = os.environ.get("R4DX_INKLING2_TOKENIZER_DIR", r"E:\models\OpenUncensored\Inkling-2-Mini-Uncensored")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "tests", "tokenizer", "golden_inkling2.json")

KEEP = ["<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>"]


def main():
    from tokenizers import Tokenizer as RawTokenizer
    from transformers import AutoTokenizer

    raw = RawTokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cases = []

    def raw_ids(text, parse_special):
        # raw.encode_special_tokens is the OPPOSITE polarity of r4dx's parse_special.
        raw.encode_special_tokens = not parse_special
        return raw.encode(text, add_special_tokens=False).ids

    def add_encode(name, text, parse_special=False):
        ids = raw_ids(text, parse_special)
        auto = tok.encode(text, add_special_tokens=False, split_special_tokens=not parse_special)
        assert ids == auto, f"raw vs AutoTokenizer mismatch for {name}: {ids[:20]} vs {auto[:20]}"
        raw.encode_special_tokens = not parse_special
        cases.append({"kind": "encode", "name": name, "text": text, "parse_special": parse_special,
                      "ids": ids, "decoded": raw.decode(ids, skip_special_tokens=True)})

    groups = {
        "english": [
            "Hello, world!", "The quick brown fox jumps over the lazy dog.", "x", " ", "", "A", "a", "I",
            "She said, \"I can't believe it's already Friday!\"", "   leading and trailing spaces   ",
            "CamelCaseIdentifierExample", "snake_case_identifier_example", "HTTPServerError and XMLHttpRequest",
            "email@example.com and https://example.com/path?query=1&x=2", "Multiple   spaces   between   words",
            "ALL CAPS SHOUTING TEXT", "Title Case Words Here", "iPhone eBay McDonald's O'Brien d'Artagnan",
            "path/to/file.txt, a/b/c, //comment, ///", "dash-separated-words and under_scores", "end.\n\nNext paragraph.",
        ],
        "contractions": [
            "I'm sure it's fine, you're right, we've won, they'll see, he'd go, don't, can't, shouldn't've",
            "I'M SURE IT'S FINE, YOU'RE RIGHT, WE'VE WON, THEY'LL SEE, HE'D GO, DON'T, CAN'T",
            "I'm SURE It's FINE yOu'Re RiGhT we'Ve wOn tHeY'lL hE'D gO DoN't cAn'T",
            "x's x'S X's X'S y't Y'T z're Z're z'RE Z'RE w've W'VE q'm Q'M u'll U'LL U'lL v'd V'D",
            "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D", "rock'n'roll 'tis 'twas ''quoted'' '' ' '",
            "it\u2019s curly apostrophe \u2018quoted\u2019", "it's\nit's\tit's  it's", "ABC's xyz's \u00c9COLE's caf\u00e9's",
            "hello'ss hello'sx hello'lll hello'rex hello're'd", "\u017f's 's\u017f it'\u017f it'S",
        ],
        "code": [
            "def bpe_merge(word: str) -> list[int]:\n    return [ord(c) for c in word]\n",
            "function add(a, b) {\n  return a + b;\n}\n", "#include <vector>\nstd::vector<int> v = {1, 2, 3};\n",
            "for (int i = 0; i < 10; ++i) { total += i * i; }", "if x is None:\n\tpass\nelse:\n\treturn x + 1",
            "SELECT * FROM tokens WHERE id > 124892 ORDER BY id;", "let x: Option<u32> = None;\nmatch x { Some(v) => v, None => 0 }",
            "printf(\"%d + %d = %d\\n\", a, b, a + b);", "git commit -m \"fix: empty merges\" --no-verify",
            "// comment with /slashes/ and /* blocks */ and ///doc\nint x = a / b;",
            "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\nimport os, sys", "<div class=\"a\">&nbsp;</div>\n<!-- c -->",
            "a->b.c::d[e](f){g}<h>~i!j@k#l$m%n^o&p*q", "http://a.b/c//d///e/\n/\n//", "x = '''doc''' + \"\"\"d\"\"\"",
        ],
        "json": [
            '{"role": "user", "content": "hi"}', '{"a": 1, "b": [1, 2, 3], "c": {"nested": true, "x": null}}',
            '[]', '{"unicode": "caf\u00e9 \u2014 \u4f60\u597d"}', '{"escaped": "line1\\nline2\\ttabbed"}',
            '{"num": -1.5e10, "big": 123456789012345}', '{"name":"get_weather","args":{"city":"Bangkok","n":3}}',
        ],
        "thai": [
            "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01",
            "\u0e1c\u0e21\u0e0a\u0e37\u0e48\u0e2d\u0e19\u0e23\u0e30\u0e27\u0e34\u0e17",
            "\u0e27\u0e31\u0e19\u0e19\u0e35\u0e49\u0e2d\u0e32\u0e01\u0e32\u0e28\u0e14\u0e35\u0e21\u0e32\u0e01",
            "\u0e01\u0e23\u0e38\u0e07\u0e40\u0e17\u0e1e\u0e21\u0e2b\u0e32\u0e19\u0e04\u0e23",
            "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13\u0e21\u0e32\u0e01\u0e04\u0e23\u0e31\u0e1a",
            "\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02 \u0e51\u0e52\u0e53\u0e54\u0e55 123456",
            "Mixed \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22 and English \u0e43\u0e19\u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04",
        ],
        "cjk": [
            "\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u6211\u7231\u4eba\u5de5\u667a\u80fd\u548c\u673a\u5668\u5b66\u4e60123",
            "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\uff01", "\ud55c\uad6d\uc5b4\ub97c \uacf5\ubd80\ud569\ub2c8\ub2e4",
            "\u30ab\u30bf\u30ab\u30ca\u3068\u3072\u3089\u304c\u306a\u3068\u6f22\u5b57", "\u4e2d\u6587English\u6df7\u5408text\u7684\u53e5\u5b50",
            "\ud841\udd0b\ud842\udf9f".encode("utf-16", "surrogatepass").decode("utf-16") if False else "\U00020bb7\U00020b9f \u3400\u4dbf",
        ],
        "arabic": [
            "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
            "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627",
            "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645 \u0661\u0662\u0663\u0664\u0665\u0666\u0667",
            "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
        ],
        "hindi": [
            "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
            "\u092d\u093e\u0930\u0924 \u090f\u0915 \u0935\u093f\u0936\u093e\u0932 \u0926\u0947\u0936 \u0939\u0948\u0964 \u0967\u0968\u0969\u096a",
            "\u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f \u0924\u094d\u0930\u093f\u0915\u094b\u0923 \u0936\u094d\u0930\u0940",
            "\u09ac\u09be\u0982\u09b2\u09be \u09ad\u09be\u09b7\u09be", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bae\u0bca\u0bb4\u0bbf",
        ],
        "latin_marks": [
            "\u00c5ngstr\u00f6m caf\u00e9 na\u00efve", "e\u0301 (decomposed e + U+0301) vs \u00e9 (precomposed)",
            "A\u030a vs \u00c5; n\u0303 vs \u00f1", "\u01c5 \u01c8 \u01cb titlecase letters", "\u02b0\u02b1 modifier \u02bb\u02bc letters",
            "\u00aa \u00ba ordinals", "\u0130stanbul I\u0307 \u0131", "\u1e9e STRASSE stra\u00dfe", "\u03a3\u03af\u03c3\u03c5\u03c6\u03bf\u03c2 \u0394\u03c0\u03cc",
            "\u041c\u043e\u0441\u043a\u0432\u0430 \u0420\u043e\u0441\u0441\u0438\u044f", "\u0410\u0411\u0412\u0413 \u0430\u0431\u0432\u0433",
            "\u0301\u0301 lone marks \u0301a", "a\u0301\u0301\u0301b", "\u0301", " \u0301x", "-\u0301x", "ABC\u0301def",
        ],
        "emoji": [
            "Hello \U0001F600 world \U0001F30D!", "\U0001F600\U0001F601\U0001F602\U0001F923\U0001F60A",
            "Family: \U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",
            "Flags: \U0001F1F9\U0001F1ED \U0001F1FA\U0001F1F8", "Skin tone: \U0001F44B\U0001F3FD",
            "Mixed \u4f60\u597d \U0001F600 \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \U0001F30F emoji test",
            "\u2764\ufe0f \u2705 \u2b50 \u00a9\u00ae\u2122 \u2603\ufe0f", "ZWJ\u200dmid\u200bzwsp\u2060wj\ufeffbom",
            "\U0001F468\u200D\U0001F4BB code \U0001F9D1\u200D\U0001F680 space",
        ],
        "whitespace": [
            "line1\nline2\nline3", "line1\r\nline2\r\nline3", "tab\there\tand\tthere",
            "trailing spaces on a line   \nand another line", "\n\n\nmultiple blank lines above\n\n\n",
            "mix\t \t of \n\t tabs \t\tand\nnewlines", "    four-space indent\n        eight-space indent",
            "a\n\nb\n\n\nc", "no collapsing:      here", "trailing newline\n", "\r", "\r\r\n\n \n ", "a \n b",
            "word  \n  \n  word", "\u00a0nbsp\u00a0\u2003em space\u2028line sep\u2029para sep",
            "a\u0085b nel", "\x0b\x0c vt ff \x1f us", "x \r\n y \r\n\r\n z", "tab\t\t\t", "\t\t\tlead", "end   ", "   ",
            "word.   \n\n   next", "a  ", "a \t", "\u3000ideographic\u3000space\u3000",
        ] + [" " * k for k in (1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 64, 100)]
          + ["\n" * k for k in (1, 2, 3, 5, 9, 17, 33)]
          + ["\r\n" * k for k in (1, 2, 3, 6)]
          + ["\t" * k for k in (1, 2, 4, 9)]
          + ["a" + " " * k + "b" for k in (1, 2, 3, 4, 5, 9)]
          + ["a" + "\n" * k + "b" for k in (1, 2, 3, 4)]
          + ["a" + " \n" * k + "b" for k in (1, 2, 3)]
          + ["a" + " " * k for k in (1, 2, 3, 8)] + ["a" + "\n" * k for k in (1, 2, 5)],
        "symbols": [
            "!!!", "...", "a.b.c", "?!", " !", "  !", "!\n", "!\n\n", "!/", "!//\n/", "// text", "/ path", " /usr/bin",
            "-- sql", "* * *", "=== heading ===\n", "~~~ code\n", "!@#$%^&*()_+-=[]{}|;':\",./<>?", " .", "a .b", "a  .b",
        ],
        "longword": [
            "supercalifragilisticexpialidocious", "a" * 200, "x" * 64 + "y" * 64 + "z" * 64, "A" * 100,
            "ThisIsAVeryLongCamelCaseIdentifierThatKeepsGoingAndGoingAndGoing", "ABCDEFGHIJabcdefghij", "aBcDeFgHiJ" * 10,
            "THISISALLCAPSWORDWITHNOLOWERCASE", "ALLCAPSthenlower", "lowerthenALLCAPS",
        ],
        "numbers": [
            "0", "42", "-17", "3.14159265358979", "1e10", "-2.5e-3", "0x1F 0b1010 0o17",
            "The year 2026 has 365 days and 8760 hours.",
            "\u0661\u0662\u0663\u0664\u0665 arabic-indic", "\uff11\uff12\uff13\uff14 fullwidth", "x\u00b2 + y\u00b3 = \u00bd",
            "\u2160\u2161\u2162 roman \u2460\u2461 circled \u0e51\u0e52\u0e53\u0e54", "1,000,000 and 1.000.000 and 10 000",
            "2026-11-22T10:30:00Z", "v1.2.3-rc4+build567", "$1,234.56 \u20ac99 \u00a5100",
        ] + ["1234567890123456789"[:k] for k in range(1, 20)] + ["a" + "7" * k + "b" for k in range(1, 10)]
          + [" " + "5" * k for k in range(1, 8)] + ["-" + "5" * k for k in range(1, 8)],
    }
    for gname, texts in groups.items():
        for i, t in enumerate(texts):
            add_encode(f"{gname}_{i}", t)

    rng = random.Random(20261122)
    twenty_kb = " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9)))
                         for _ in range(4000))[:20000]
    add_encode("long_line_20kb", twenty_kb)
    mixed_kb = "".join(rng.choice(["Hello", " world", ",", ".\n", "  ", "123", "\u4f60\u597d", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
                                    "\U0001F600", "'s", "CamelCase", "\t", "/", "_x", "ABC", "\r\n"]) for _ in range(5000))
    add_encode("mixed_long_line", mixed_kb)

    # ---- special / added tokens as text, both parse_special values ----------------------------------
    specials = [
        "<|message_user|><|content_text|>hi<|end_message|>", "<|message_model|><|content_thinking|>t<|end_message|>",
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{}}<|end_message|>",
        "<|content_model_end_sampling|>", "<|endoftext|>", "<|begin_of_text|>hello", "<|unused|><|unused_200007|>",
        "<|content_invoke_tool_text|>", "<|content_image|><|unused_200054|>", "<|content_xml|>tool_declare",
        "before <|end_message|> after", "<|message_user", "<|content_text|", "<<|end_message|>>", "<|end_message|><|end_message|>",
        "x<|message_tool|>y<|message_system|>z", "<|UNUSED_200007|> <|Message_User|>",
    ]
    for i, t in enumerate(specials):
        add_encode(f"special_notparsed_{i}", t, parse_special=False)
        add_encode(f"special_parsed_{i}", t, parse_special=True)

    # ---- decode cases (skip_special_tokens on/off, kept dialect specials, streaming) ----------------
    ids_of = {t: raw.token_to_id(t) for t in KEEP + ["<|content_model_end_sampling|>", "<|endoftext|>"]}
    for t, i in ids_of.items():
        assert i is not None, t
    keep_ids = {ids_of[t] for t in KEEP}

    def add_decode(name, ids, skip, keep):
        if keep:
            filtered = [i for i in ids if not raw.id_to_token(i).startswith("<|") or i in keep_ids]
            expected = raw.decode(filtered, skip_special_tokens=False)
        else:
            expected = raw.decode(ids, skip_special_tokens=skip)
            assert expected == tok.decode(ids, skip_special_tokens=skip), f"decode mismatch {name}"
        cases.append({"kind": "decode", "name": name, "ids": ids, "expected": expected,
                      "skip_special_tokens": skip, "keep_special": keep, "stream": True})

    mm, think, text, tool, end = (ids_of["<|message_model|>"], ids_of["<|content_thinking|>"], ids_of["<|content_text|>"],
                                  ids_of["<|content_invoke_tool_json|>"], ids_of["<|end_message|>"])
    eos = ids_of["<|content_model_end_sampling|>"]
    mixed = ([mm, think] + raw_ids("let me think \u0e2a\u0e27\u0e31", False) + [end, mm, text] + raw_ids("Hi \U0001F600!", False)
             + [end, mm] + raw_ids("get_weather", False) + [tool] + raw_ids('{"name":"get_weather","args":{"city":"BKK"}}', False)
             + [end, eos])
    add_decode("decode_mixed_skip", mixed, True, False)
    add_decode("decode_mixed_noskip", mixed, False, False)
    add_decode("decode_mixed_keep_dialect", mixed, True, True)
    add_decode("decode_emoji", raw_ids("a \U0001F600\U0001F468\u200D\U0001F469 b", False), True, False)
    add_decode("decode_thai", raw_ids(groups["thai"][3], False), True, False)
    add_decode("decode_cjk_ext_b", raw_ids("\U00020bb7\U00020b9f", False), True, False)
    add_decode("decode_empty", [], True, False)
    add_decode("decode_only_specials_skip", [mm, text, end, eos], True, False)
    add_decode("decode_only_specials_keep", [mm, text, end, eos], True, True)

    out = {
        "generated_by": "tools/tok_ref/gen_golden_inkling2.py",
        "model_dir": MODEL_DIR,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": len(tok),
        "bos_token_id": None,
        "eos_token_id": eos,
        "eos_token_ids": [eos],
        "pad_token_id": ids_of["<|endoftext|>"],
        "keep_special_on_decode": KEEP,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {OUT_PATH}: {kinds}, vocab_size={out['vocab_size']}, eos={eos}, pad={out['pad_token_id']}, "
          f"size={os.path.getsize(OUT_PATH)} bytes")


if __name__ == "__main__":
    sys.exit(main())
