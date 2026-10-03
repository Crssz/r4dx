# Vendored third-party dependencies

Header-only. Fetched with `Invoke-WebRequest`, one file each, LICENSE alongside. Pinned by tag
where the upstream cuts one, else by commit SHA of the default branch at fetch time
(2026-09-19).

| Library | Path | Upstream | Version / commit |
|---|---|---|---|
| nlohmann/json | `third_party/nlohmann/json.hpp` | github.com/nlohmann/json | `v3.11.3` |
| cpp-httplib | `third_party/httplib/httplib.h` | github.com/yhirose/cpp-httplib | `v0.18.3` |
| minja | `third_party/minja/minja.hpp`, `chat-template.hpp` | github.com/google/minja | `021c2293c187789ef13d56c6cfd89c9b134fd80f` |
| stb_image | `third_party/stb/stb_image.h` | github.com/nothings/stb | `2c980bb59875b0d32144a71867fbdebb2f77cd20` |

libr4d (the GPU kernels) is not header-only: it is vendored as plain sources under
`third_party/libr4d` (tree of upstream libr4d commit `f47a8bce908b062c7af77f8dbd18b760c1d199e6`,
branch `linear`, merged in with its history), then cut down to the units r4dx uses (its unused GEMMs,
the all-reduce files, the pybind module, its Python / build scripts and its README are not carried;
the credit lives in `NOTICE`). It has no licence file upstream; see `NOTICE`.

To refresh a dependency: re-download the file(s) at a new tag/commit, update this table, and
re-run the build + tests.

## Local patches

- **minja.hpp, `021c2293c` + 1 line**: added an `is undefined` test to the `is`/`is not` operator
  (`BinaryOpExpr::do_evaluate`'s `eval()` lambda, next to the existing `"defined"` case) --
  upstream at this commit implements `is defined` (`!l.is_null()`) but has no complementary
  `is undefined`, and Qwen3.8-27B's own `chat_template.jinja` (`C:\AI\models\Qwen3.8-27B\chat_template.jinja`,
  line 46: `{%- if enable_thinking is undefined or enable_thinking is true %}`) uses `is undefined`
  directly, so every `ChatTemplate::render()` call threw `"Unknown type for 'is' operator:
  undefined"` without this patch. The added line is `if (name == "undefined") return l.is_null();`
  -- the natural complement of the existing `"defined"` line, since minja represents an undefined
  Jinja variable as a null `Value`. On a future re-vendor from upstream, re-apply this one line if
  upstream still hasn't picked it up (worth upstreaming to google/minja).

- **minja.hpp, Gemma 4 patches (3 local changes, all needed by google/gemma-4-12B-it's
  `chat_template.jinja`; covered by `tests/tokenizer/test_minja_patches.cpp` and the
  `tokenizer_golden_gemma` chat cases):**
  1. *Adjacent string literals concatenate* (`Parser::parseConstant`): after a string literal, further
     quote-started literals (whitespace/newlines between) are appended, as in Python/Jinja. The
     template's `raise_exception("..." "..." "...")` (lines 258-262) made `ChatTemplate::from_source`
     throw at parse time without it.
  2. *`dictsort` is case-insensitive* (the `dictsort` global): Jinja's default is
     `case_sensitive=False`; upstream minja sorted keys byte-wise, so mixed-case tool-parameter names
     (`Zone` vs `city`) rendered in a different order than HF. ASCII lowercase compare, stable sort.
  3. *An empty dict is falsy* (`Value::to_bool`): upstream returned true for every object, so
     `{%- if params['properties'] -%}` on `{}` emitted `properties:{}` where HF omits it.
  The Qwen3.8-27B goldens (`tokenizer_golden`) are unchanged by all three. On a re-vendor, re-apply
  them if upstream still lacks them (each is a few lines).

  Related, not a minja patch: Gemma's template needs `ChatTemplateOptions{.apply_polyfills = false}`
  (src/tokenizer/chat_template.h). minja's caps probe reports `supports_tool_calls=false` for it and
  the default polyfills would rewrite assistant `tool_calls` into JSON content.
