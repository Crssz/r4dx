// tests/model/test_decode_legacy.cpp -- pure CPU. The host-side contracts of the decode-t1 pass
// (docs/perf.md): R4DX_DECODE_LEGACY's parser (core/decode_legacy.hpp) and the one-copy step-metadata
// layout (src/model/step_meta.h) that Model::StageStepMeta ships with a single async H2D.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "r4dx/core/decode_legacy.hpp"
#include "step_meta.h"

using r4dx::core::DecodeItem;
using r4dx::core::ParseDecodeLegacy;

namespace {

int g_fail = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}
constexpr unsigned A = static_cast<unsigned>(DecodeItem::kArgmax);
constexpr unsigned B = static_cast<unsigned>(DecodeItem::kAb);
constexpr unsigned C = static_cast<unsigned>(DecodeItem::kAttn);
constexpr unsigned D = static_cast<unsigned>(DecodeItem::kHost);

}  // namespace

int main() {
  // ---- R4DX_DECODE_LEGACY ----
  Check(ParseDecodeLegacy(nullptr) == 0, "unset: nothing legacy");
  Check(ParseDecodeLegacy("") == 0, "empty: nothing legacy");
  Check(ParseDecodeLegacy("argmax") == A, "argmax");
  Check(ParseDecodeLegacy("ab") == B, "ab");
  Check(ParseDecodeLegacy("attn") == C, "attn");
  Check(ParseDecodeLegacy("host") == D, "host");
  Check(ParseDecodeLegacy("argmax,ab,attn,host") == (A | B | C | D), "the documented four");
  Check(ParseDecodeLegacy("ab,host") == (B | D), "two items");
  Check(ParseDecodeLegacy("all") == r4dx::core::kDecodeItemAll, "all");
  Check(r4dx::core::kDecodeItemAll == (A | B | C | D), "all is every item");
  Check(ParseDecodeLegacy("ATTN; Host argmax") == (C | D | A), "case, semicolons and spaces");
  Check(ParseDecodeLegacy(",,ab,,") == B, "empty tokens");
  Check(ParseDecodeLegacy("bogus", /*warn=*/false) == 0, "an unknown token is ignored");
  Check(ParseDecodeLegacy("bogus,ab", /*warn=*/false) == B, "an unknown token does not stop the rest");
  Check(ParseDecodeLegacy("abx", /*warn=*/false) == 0, "no prefix matching");
  Check(ParseDecodeLegacy("argmax,argmax") == A, "a repeated item");

  // ---- step metadata layout ----
  using namespace r4dx::model;
  static_assert(kStepMetaIds == 0 && kStepMetaPos == 64 && kStepMetaSeq == 128, "section offsets");
  static_assert((kStepMetaPos * sizeof(int32_t)) % 256 == 0 && (kStepMetaSeq * sizeof(int32_t)) % 256 == 0,
                "sections start on a 256-byte boundary");
  static_assert((kStepMetaInts * sizeof(int32_t)) % 16 == 0 && kStepMetaInts > kStepMetaSeq, "one 16-byte padded copy");
  {
    std::vector<int32_t> h(static_cast<size_t>(kStepMetaInts), -7);
    FillStepMeta(h.data(), {11}, 1234);
    Check(h[kStepMetaIds] == 11 && h[kStepMetaPos] == 1234 && h[kStepMetaSeq] == 1235, "T = 1 decode step");
    Check(h[kStepMetaIds + 1] == -7 && h[kStepMetaPos + 1] == -7, "T = 1 leaves the other rows alone");
  }
  {
    std::vector<int32_t> h(static_cast<size_t>(kStepMetaInts), 0);
    std::vector<int32_t> ids(8);
    for (int i = 0; i < 8; ++i) ids[static_cast<size_t>(i)] = 100 + i;
    FillStepMeta(h.data(), ids, 40000);
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && h[kStepMetaIds + i] == 100 + i && h[kStepMetaPos + i] == 40000 + i;
    Check(ok, "an 8-row verify window: id t, position pos + t");
    Check(h[kStepMetaSeq] == 40008, "seqused_k = pos + T");
  }
  {
    std::vector<int32_t> h(static_cast<size_t>(kStepMetaInts), 0);
    FillStepMeta(h.data(), std::vector<int32_t>(64, 5), 0);
    Check(h[kStepMetaPos + 63] == 63 && h[kStepMetaSeq] == 64, "a 64-row chunk");
    bool threw = false;
    try {
      FillStepMeta(h.data(), std::vector<int32_t>(65, 5), 0);
    } catch (const std::exception&) {
      threw = true;
    }
    Check(threw, "65 rows refused");
    threw = false;
    try {
      FillStepMeta(h.data(), {}, 0);
    } catch (const std::exception&) {
      threw = true;
    }
    Check(threw, "0 rows refused");
  }

  if (g_fail != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_decode_legacy: OK\n");
  return 0;
}
