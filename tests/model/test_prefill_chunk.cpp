// test_prefill_chunk: pure CPU unit test of src/model/prefill_chunk.h -- the R4DX_PREFILL_CHUNK parser and
// the decision of when a prompt Prefill call may run 256-row super-chunks (docs/trellis-m256.md). No HIP,
// no container. The default and every unparsable value stay on today's 64-row chunks; every case the
// 256-row path was not built for (tensor parallel, MTP, DFlash, a feature capture, an image, a quant2
// container, a per-chunk callback, buffers not sized for it) falls back to 64 with a reason, never an
// error. Includes the --tp 2 fallback the GPU validation could not run (TP = 2 is not exercised there).
#include <cstdio>
#include <cstring>

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
}  // namespace

int main() {
  // ---- parser (the warning for an unrecognized value goes to stderr; the value falls back to 64) ----
  CHECK(ParsePrefillChunk(nullptr) == 64, "unset -> 64");
  CHECK(ParsePrefillChunk("") == 64, "empty -> 64");
  CHECK(ParsePrefillChunk("64") == 64, "64 -> 64");
  CHECK(ParsePrefillChunk("256") == 256, "256 -> 256");
  CHECK(ParsePrefillChunk("128") == 64, "128 (unsupported) -> 64");
  CHECK(ParsePrefillChunk("0") == 64, "0 -> 64");
  CHECK(ParsePrefillChunk("256 ") == 64, "'256 ' (trailing blank) -> 64");
  CHECK(ParsePrefillChunk("on") == 64, "on -> 64");
  CHECK(ParsePrefillChunk("abc") == 64, "abc -> 64");

  // ---- decision ----
  PrefillChunkInputs ok;
  ok.requested = 256;
  ok.buffers_wide = true;
  const char* why = "unset";
  CHECK(DecidePrefillChunk(ok, &why) == 256 && why == nullptr, "TP=1, no drafter, text only, buffers wide -> 256");

  PrefillChunkInputs def = ok;
  def.requested = 64;
  CHECK(DecidePrefillChunk(def, &why) == 64 && why == nullptr, "not requested -> 64, no reason");
  PrefillChunkInputs def_tp = def;
  def_tp.tensor_parallel = true;
  CHECK(DecidePrefillChunk(def_tp, &why) == 64 && why == nullptr, "not requested at TP=2 -> 64, silently");

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
      {"buffers not wide", [](PrefillChunkInputs& i) { i.buffers_wide = false; }, "not sized"},
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
  // several at once (the TP rank of a --dflash run): still 64, the first reason
  {
    PrefillChunkInputs in = ok;
    in.tensor_parallel = true;
    in.dflash = true;
    why = nullptr;
    CHECK(DecidePrefillChunk(in, &why) == 64 && why != nullptr, "TP + dflash -> 64 with a reason");
  }

  if (g_fail != 0) {
    std::printf("test_prefill_chunk: %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("test_prefill_chunk: PASS\n");
  return 0;
}
