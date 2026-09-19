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
