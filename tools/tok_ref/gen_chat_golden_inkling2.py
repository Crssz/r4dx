#!/usr/bin/env python
"""Generates tests/tokenizer/chat_inkling2_golden.json: HF `apply_chat_template` renderings of Inkling-2-Mini's
chat_template.jinja, which r4dx's minja-based ChatTemplate must reproduce byte for byte (CTest 'tokenizer_chat_inkling2',
tests/tokenizer/golden_test.cpp; ids are the raw `tokenizers` encoding of the prompt with specials parsed).

CPU only. Run with the reference venv, -I:

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\tok_ref\\gen_chat_golden_inkling2.py

Reads <R4DX_INKLING2_TOKENIZER_DIR | E:\\models\\OpenUncensored\\Inkling-2-Mini-Uncensored> (read-only).

The template has NO generation prompt (add_generation_prompt changes nothing): the prompt ends after the last message
(plus the "Thinking effort level" system message, which goes before the first non-system message, or at the end when
there is none) and the model emits its own `<|message_model|>` header. Tool-call `arguments` must be a parsed mapping
(a JSON string raises). Case kinds: "chat", "chat_error" (render must throw).
"""
import json
import os
import sys

MODEL_DIR = os.environ.get("R4DX_INKLING2_TOKENIZER_DIR", r"E:\models\OpenUncensored\Inkling-2-Mini-Uncensored")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "tests", "tokenizer", "chat_inkling2_golden.json")


def main():
    from tokenizers import Tokenizer as RawTokenizer
    from transformers import AutoTokenizer

    raw = RawTokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cases = []

    def add_chat(name, messages, tools=None, add_generation_prompt=False, **extra):
        kwargs = dict(add_generation_prompt=add_generation_prompt, tokenize=False)
        if tools is not None:
            kwargs["tools"] = tools
        kwargs.update(extra)
        prompt = tok.apply_chat_template(messages, **kwargs)
        raw.encode_special_tokens = False
        ids = raw.encode(prompt, add_special_tokens=False).ids
        auto = tok.encode(prompt, add_special_tokens=False)
        assert ids == auto, f"chat ids raw vs auto mismatch for {name}"
        cases.append({"kind": "chat", "name": name, "messages": messages, "tools": tools,
                      "add_generation_prompt": add_generation_prompt, "extra_context": dict(extra),
                      "prompt": prompt, "ids": ids})

    def add_chat_error(name, messages, tools=None, add_generation_prompt=False, **extra):
        kwargs = dict(add_generation_prompt=add_generation_prompt, tokenize=False)
        if tools is not None:
            kwargs["tools"] = tools
        kwargs.update(extra)
        try:
            tok.apply_chat_template(messages, **kwargs)
        except Exception as e:  # noqa: BLE001
            cases.append({"kind": "chat_error", "name": name, "messages": messages, "tools": tools,
                          "add_generation_prompt": add_generation_prompt, "extra_context": dict(extra),
                          "hf_error": str(e)[:200]})
            return
        raise AssertionError(f"{name}: HF did not raise")

    U = lambda t: {"role": "user", "content": t}  # noqa: E731
    A = lambda t, **k: {"role": "assistant", "content": t, **k}  # noqa: E731
    S = lambda t: {"role": "system", "content": t}  # noqa: E731

    weather = {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
    search = {"type": "function", "function": {
        "name": "search", "description": "Search the web (\u4f60\u597d, caf\u00e9, \U0001F600).",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"},
                       "Filters": {"type": "array", "items": {"type": "string"}},
                       "opts": {"type": "object", "properties": {"limit": {"type": "integer"}, "Z": {"type": "number"}}}}}}}
    bare_tool = {"name": "bare", "parameters": {"type": "object", "properties": {}}}
    no_desc_tool = {"type": "function", "function": {"name": "nodesc"}}

    # ---- basics, effort placement ----
    add_chat("system_user", [S("You are concise."), U("Hi")])
    add_chat("user_only", [U("What is the capital of Thailand?")])
    add_chat("user_only_gen_prompt_true", [U("Hi")], add_generation_prompt=True)
    add_chat("system_only_effort_at_end", [S("Only a system prompt.")])
    add_chat("two_system_then_user", [S("a"), S("b"), U("c")])
    add_chat("user_then_system", [U("u"), S("late system")])
    add_chat("assistant_first", [A("hello")])
    add_chat("empty_user", [U("")])
    add_chat("unicode", [S("\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \U0001F600 <|end_message|> \"q\" \\ \n"), U("\u4f60\u597d\uff0c\u4e16\u754c caf\u00e9 \u0645\u0631\u062d\u0628\u0627")])
    add_chat("multi_turn", [S("s"), U("Hi there!"), A("Hello! How can I help?"), U("Tell me a joke."),
                            A("Why did the GPU cross the road?"), U("Explain it.")])
    add_chat("multi_turn_reasoning", [
        U("2+2?"), A("4", reasoning_content="two plus two is four"), U("3+3?"),
        A("6", reasoning_content="three plus three\nis six"), U("thanks")])
    add_chat("assistant_reasoning_only", [U("q"), A("", reasoning_content="thinking only"), U("q2")])
    add_chat("assistant_reasoning_none_content", [U("q"), A(None, reasoning_content="t"), U("q2")])
    add_chat("assistant_empty_no_reasoning", [U("q"), A(""), U("q2")])
    add_chat("assistant_named", [U("q"), A("answer", name="inkling"), U("q2")])

    # ---- reasoning_effort ----
    for label, eff in [("high", "high"), ("none", "none"), ("minimal", "minimal"), ("low", "low"), ("medium", "medium"),
                       ("xhigh", "xhigh"), ("max", "max"), ("padded_upper_ws", "  high "), ("f0", 0.0), ("i0", 0),
                       ("f05", 0.5), ("f099", 0.99), ("f0123", 0.123), ("f07", 0.7), ("f001", 0.01), ("f0_9", 0.9),
                       ("s_numeric_text_err", "0.5")]:
        if label == "s_numeric_text_err":
            add_chat_error(f"effort_{label}", [U("x")], reasoning_effort=eff)
        else:
            add_chat(f"effort_{label}", [S("s"), U("x")], reasoning_effort=eff)
    add_chat("effort_none_value_is_default", [U("x")], reasoning_effort=None)
    add_chat("effort_with_only_system", [S("s")], reasoning_effort="low")
    add_chat_error("effort_bogus", [U("x")], reasoning_effort="bogus")
    add_chat_error("effort_too_high_float", [U("x")], reasoning_effort=1.0)
    add_chat_error("effort_too_high_int", [U("x")], reasoning_effort=1)
    add_chat_error("effort_negative", [U("x")], reasoning_effort=-0.1)
    add_chat_error("effort_just_over", [U("x")], reasoning_effort=0.991)

    # ---- tools ----
    add_chat("tools", [U("Weather in Bangkok?")], tools=[weather])
    add_chat("tools_two_system", [S("Be brief."), U("Weather in Bangkok?")], tools=[weather, search])
    add_chat("tools_bare_and_nodesc", [U("x")], tools=[bare_tool, no_desc_tool])
    add_chat("tools_empty_list", [U("x")], tools=[])
    add_chat("tools_only_system", [S("")], tools=[search])

    # ---- assistant tool calls ----
    tc = lambda name, args, **k: {"id": k.get("id", "call_1"), "type": "function", "function": {"name": name, "arguments": args}}  # noqa: E731
    add_chat("toolcall_basic", [U("Weather?"), A("", tool_calls=[tc("get_weather", {"city": "Bangkok", "units": "celsius"})]),
                                {"role": "tool", "name": "get_weather", "content": "{\"temp_c\": 33}"}], tools=[weather])
    add_chat("toolcall_with_text_and_reasoning", [
        U("Weather?"), A("Let me check.", reasoning_content="need the tool", tool_calls=[tc("get_weather", {"city": "Chiang Mai"})]),
        {"role": "tool", "name": "get_weather", "content": "sunny"}, A("It is sunny."), U("ok")], tools=[weather])
    add_chat("toolcall_nested_args", [U("go"), A("", tool_calls=[tc("search", {
        "q": "plain", "Filters": ["a", "b", {"k": [1, 2.5, None, True, False]}], "opts": {"Z": 1.5, "limit": 3, "a": {"x": {"y": []}}},
        "s_quote": "it's \"q\"", "s_back": "a\\b", "s_nl": "l1\nl2\r\tx", "s_uni": "caf\u00e9 \u0e44\u0e17\u0e22 \U0001F600",
        "i": 7, "neg": -3, "f": 1.5, "f_exp": 1e-7, "f_big": 1e16, "f_int": 3.0, "t": True, "fa": False, "n": None,
        "empty_lst": [], "empty_dct": {}, "B_upper": 1, "a_lower": 2, "_under": 3, "10": "numeric-key", "9": "nine",
        "\u00e9clair": "non-ascii key", "ZZ": {"b": 1, "B": 2, "a": 3}})])], tools=[search])
    add_chat("toolcall_multi_and_empty_args", [
        U("go"), A("", tool_calls=[tc("a", {}, id="c1"), tc("b", {"x": 1}, id="c2"), tc("c", {"y": "z"}, id="c3")]),
        {"role": "tool", "tool_call_id": "c1", "content": "r1"}, {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "tool", "tool_call_id": "c3", "content": "r3"}])
    add_chat("toolcall_missing_arguments", [U("go"), A("", tool_calls=[{"id": "c", "type": "function", "function": {"name": "a"}}])])
    add_chat("toolcall_arguments_none", [U("go"), A("", tool_calls=[tc("a", None)])])
    add_chat("toolcall_named_assistant_match", [U("go"), A("", name="a", tool_calls=[tc("a", {"k": "v"})])])
    add_chat("toolcall_with_text_only_content", [U("go"), A("calling", tool_calls=[tc("a", {"k": "v"})])])
    add_chat("tool_result_by_tool_call_id_unmatched", [
        U("go"), A("", tool_calls=[tc("a", {}, id="c1")]), {"role": "tool", "tool_call_id": "zzz", "content": "r"}])
    add_chat("tool_result_no_name_no_id", [U("go"), {"role": "tool", "content": "r"}])
    add_chat("tool_result_name_wins_over_id", [
        U("go"), A("", tool_calls=[tc("a", {}, id="c1")]), {"role": "tool", "name": "explicit", "tool_call_id": "c1", "content": "r"}])
    add_chat("tool_result_without_id_in_call", [
        U("go"), A("", tool_calls=[{"type": "function", "function": {"name": "a", "arguments": {}}}]),
        {"role": "tool", "tool_call_id": "c1", "content": "r"}])
    add_chat("tool_result_json_content", [U("go"), {"role": "tool", "name": "a", "content": "{\"a\": [1, 2]}"}])
    add_chat_error("toolcall_string_arguments", [U("go"), A("", tool_calls=[tc("a", "{\"x\": 1}")])])
    add_chat_error("toolcall_list_arguments", [U("go"), A("", tool_calls=[tc("a", [1, 2])])])
    add_chat_error("toolcall_name_mismatch", [U("go"), A("", name="other", tool_calls=[tc("a", {})])])
    add_chat_error("toolcall_missing_name", [U("go"), A("", tool_calls=[{"id": "c", "type": "function", "function": {"arguments": {}}}])])

    # ---- content parts / images ----
    add_chat("parts_text", [S("sys"), {"role": "user", "content": [{"type": "text", "text": "look: "}, "tail", {"text": "no type"}, {"type": "input_text", "text": "it"}]}])
    add_chat("part_image_user", [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image"}]}])
    add_chat("part_image_url_user", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "x"}]}])
    add_chat("part_image_only", [{"role": "user", "content": [{"type": "input_image"}]}])
    add_chat("part_image_tool", [U("go"), {"role": "tool", "name": "shot", "content": [{"type": "image"}]}])
    add_chat("part_empty_list_content", [{"role": "user", "content": []}])
    add_chat("part_non_string_text", [{"role": "user", "content": [{"type": "text", "text": 5}]}])
    add_chat("assistant_parts", [U("q"), A([{"type": "text", "text": "a1"}, {"type": "text", "text": "a2"}]), U("q2")])
    add_chat("assistant_parts_with_reasoning_and_call", [
        U("q"), A([{"type": "text", "text": "part"}], reasoning_content="r", tool_calls=[tc("a", {"k": 1})])])
    add_chat_error("part_image_assistant", [U("q"), A([{"type": "image"}])])
    add_chat_error("part_image_system", [{"role": "system", "content": [{"type": "image"}]}, U("x")])
    add_chat_error("part_audio", [{"role": "user", "content": [{"type": "audio"}]}])
    add_chat_error("part_unknown_type", [{"role": "user", "content": [{"type": "video"}]}])
    add_chat_error("unknown_role", [U("q"), {"role": "ipython", "content": "out"}])

    out = {
        "generated_by": "tools/tok_ref/gen_chat_golden_inkling2.py",
        "model_dir": MODEL_DIR,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": len(tok),
        "eos_token_id": 200006,
        "eos_token_ids": [200006],
        "apply_polyfills": False,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"wrote {OUT_PATH}: {kinds}, size={os.path.getsize(OUT_PATH)} bytes")


if __name__ == "__main__":
    sys.exit(main())
