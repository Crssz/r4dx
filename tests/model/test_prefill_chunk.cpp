// test_prefill_chunk: pure CPU unit test of src/model/prefill_chunk.h -- the R4DX_PREFILL_CHUNK parser (256
// by default, "0" / "64" the kill switch), the decision of when a Model runs 256-row prefill super-chunks
// (docs/prefill.md, docs/trellis-m256.md) and the chunk grid a Prefill call walks (tail lengths). No HIP,
// no container. Every case the wide path does not serve falls back to 64-row chunks with a reason, never
// an error.
#include <cstdio>
#include <cstring>
#include <vector>

#include "prefill_chunk.h"

using namespace r4dx::model;

namespace {
int g_fail = 0;
#define CHECK(cond, ...)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);       \
      std::printf(__VA_ARGS__);                              \
      std::printf("\n");                                     \
      ++g_fail;                                              \
    }                                                        \
  } while (0)

std::vector<int> Grid(long long n, int wide) {
  std::vector<int> rows(static_cast<size_t>(n / 64 + 8));
  const int count = PrefillChunkGrid(n, wide, rows.data(), static_cast<int>(rows.size()));
  rows.resize(static_cast<size_t>(count));
  return rows;
}
long long Sum(const std::vector<int>& v) {
  long long s = 0;
  for (int x : v) s += x;
  return s;
}
}  // namespace

int main() {
  // ---- parser (the warning for an unrecognized value goes to stderr; the value falls back to 64) ----
  CHECK(ParsePrefillChunk(nullptr) == 256, "unset -> 256 (the default)");
  CHECK(ParsePrefillChunk("") == 256, "empty -> 256 (the default)");
  CHECK(ParsePrefillChunk("256") == 256, "256 -> 256");
  CHECK(ParsePrefillChunk("64") == 64, "64 -> 64 (kill switch)");
  CHECK(ParsePrefillChunk("0") == 64, "0 -> 64 (kill switch)");
  CHECK(ParsePrefillChunk("128") == 64, "128 (unsupported) -> 64");
  CHECK(ParsePrefillChunk("256 ") == 64, "'256 ' (trailing blank) -> 64");
  CHECK(ParsePrefillChunk("on") == 64, "on -> 64");
  CHECK(ParsePrefillChunk("off") == 64, "off (not a spelling of the kill switch) -> 64 with a warning");
  CHECK(ParsePrefillChunk("abc") == 64, "abc -> 64");

  // ---- decision ----
  PrefillChunkInputs ok;  // a plain TP=1 text-only Model whose buffers are wide
  const char* why = "unset";
  CHECK(DecidePrefillChunk(ok, &why) == 256 && why == nullptr, "default request, plain Model -> 256");

  PrefillChunkInputs off = ok;
  off.requested = 64;
  CHECK(DecidePrefillChunk(off, &why) == 64 && why == nullptr, "kill switch -> 64, no reason (a choice)");
  PrefillChunkInputs off_tp = off;
  off_tp.tensor_parallel = true;
  off_tp.mtp = true;
  CHECK(DecidePrefillChunk(off_tp, &why) == 64 && why == nullptr,
        "kill switch wins over every other input, silently");

  struct Case {
    const char* name;
    void (*set)(PrefillChunkInputs&);
    const char* reason_part;
  };
  const Case cases[] = {
      {"--tp 2", [](PrefillChunkInputs& i) { i.tensor_parallel = true; }, "tensor parallel"},
      {"--mtp", [](PrefillChunkInputs& i) { i.mtp = true; }, "MTP"},
      {"--dflash", [](PrefillChunkInputs& i) { i.dflash = true; }, "DFlash"},
      {"feature capture", [](PrefillChunkInputs& i) { i.dflash_capture = true; }, "feature capture"},
      {"image spliced", [](PrefillChunkInputs& i) { i.mrope_active = true; }, "image"},
      {"quant2 container", [](PrefillChunkInputs& i) { i.rotated_container = true; }, "quant2"},
      {"chunk callback", [](PrefillChunkInputs& i) { i.on_chunk_captured = true; }, "callback"},
      {"buffers not wide", [](PrefillChunkInputs& i) { i.buffers_wide = false; }, "64 rows at load"},
  };
  for (const Case& c : cases) {
    PrefillChunkInputs in = ok;
    c.set(in);
    why = nullptr;
    const int rows = DecidePrefillChunk(in, &why);
    CHECK(rows == 64, "%s: must fall back to 64, got %d", c.name, rows);
    CHECK(why != nullptr && std::strstr(why, c.reason_part) != nullptr, "%s: reason '%s' should mention '%s'",
          c.name, why ? why : "(none)", c.reason_part);
    CHECK(DecidePrefillChunk(in) == 64, "%s: also without a reason out-parameter", c.name);
  }
  // several at once (the TP rank of a --dflash run): still 64, with a reason
  {
    PrefillChunkInputs in = ok;
    in.tensor_parallel = true;
    in.dflash = true;
    why = nullptr;
    CHECK(DecidePrefillChunk(in, &why) == 64 && why != nullptr, "TP + dflash -> 64 with a reason");
  }

  // ---- the chunk grid of one Prefill call: anchored at the call start ----
  // (the lengths of the docs/prefill.md tail tests: 1, 63, 64, 65, 255, 256, 257, 511, 8145)
  {
    struct G {
      long long n;
      std::vector<int> wide;
    };
    const G grids[] = {
        {1, {1}},
        {63, {63}},
        {64, {64}},
        {65, {64, 1}},
        {255, {64, 64, 64, 63}},
        {256, {256}},
        {257, {256, 1}},
        {319, {256, 63}},
        {320, {256, 64}},
        {511, {256, 64, 64, 64, 63}},
        {512, {256, 256}},
    };
    for (const G& g : grids) {
      const std::vector<int> got = Grid(g.n, 256);
      CHECK(got == g.wide, "grid of %lld tokens at 256: %zu chunks, first %d", g.n, got.size(),
            got.empty() ? 0 : got[0]);
      CHECK(Sum(got) == g.n, "grid of %lld tokens covers %lld", g.n, Sum(got));
      // The 64-row grid: ceil(n / 64) chunks, every one full but the last.
      const std::vector<int> narrow = Grid(g.n, 64);
      CHECK(static_cast<long long>(narrow.size()) == (g.n + 63) / 64 && Sum(narrow) == g.n,
            "64-row grid of %lld tokens", g.n);
      for (size_t i = 0; i + 1 < narrow.size(); ++i) CHECK(narrow[i] == 64, "64-row grid: chunk %zu full", i);
    }
    // 8145 tokens (the 8k prompt of the TTFT runs): 31 super-chunks and 209 rows = 3 full chunks + 17.
    const std::vector<int> g8k = Grid(8145, 256);
    CHECK(g8k.size() == 31 + 4 && Sum(g8k) == 8145, "8145 tokens: %zu chunks", g8k.size());
    CHECK(g8k[30] == 256 && g8k[31] == 64 && g8k[32] == 64 && g8k[33] == 64 && g8k[34] == 17,
          "8145 tokens: 31 x 256, then 64, 64, 64, 17");
    // Two calls (a prefix split at 1001, as the KL harness's --prefix-split-at): each anchors its own grid.
    const std::vector<int> a = Grid(1001, 256), b = Grid(8145 - 1001, 256);
    CHECK(a.size() == 3 + 4 && a[2] == 256 && a[3] == 64 && a[6] == 41, "1001 tokens: 3 x 256, 64 x 3, 41");
    CHECK(Sum(a) + Sum(b) == 8145, "split calls cover the prompt");
    // A tail is never wide, and no chunk is ever larger than 256 or empty.
    for (long long n = 1; n <= 1100; ++n) {
      const std::vector<int> got = Grid(n, 256);
      CHECK(Sum(got) == n, "grid of %lld covers it", n);
      for (size_t i = 0; i < got.size(); ++i) {
        CHECK(got[i] == 256 || (got[i] >= 1 && got[i] <= 64), "grid of %lld: chunk %zu is %d", n, i, got[i]);
        if (got[i] == 256) CHECK(i * 256 + 256 <= static_cast<size_t>(n), "wide chunk only while >= 256 remain");
      }
    }
  }

  if (g_fail != 0) {
    std::printf("test_prefill_chunk: %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("test_prefill_chunk: PASS\n");
  return 0;
}
