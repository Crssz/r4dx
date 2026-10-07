// test_prefill_int8_cpu: host-only checks of R4DX_PREFILL_INT8's decision and planning code (docs/int8-prefill.md
// "Production path"); no HIP call, no device, no container.
//   * src/model/prefill_int8.h: the parser (unset / empty / 1 / on = on -- the default --, 0 / off = off, anything else =
//     the default with a warning; 1 / on are the only EXPLICIT requests), the ModelOptions::prefill_int8 resolution, and the decision table of DecidePrefillInt8 -- off is not a
//     fallback (no reason), and every refusal names its reason in the order of the checks;
//   * src/model/linear.cpp: PlanTrellisI8 for the seven linear classes of the 27B at both rates (the part boundary of
//     mlp.gate_up included), its refusals (KB, parts, whole 128-blocks, the part boundary, a shape without a table
//     row, R4DX_M256_SHAPES), and the tuning table itself: no duplicate key, every row legal under the kernel's own
//     check (r4d_gemm_trellis_nt_i8_check) at its shape, every class of the model has a row at KB 4 and KB 5;
//   * ScopedTrellisI8: off by default, nests, restores;
//   * R4DX_PREFILL_INT8_FUSEDQ: the parser and the producer / consumer decision (TrellisI8Takes, TrellisI8FusedQ);
//   * R4DX_PREFILL_INT8_SCALES: the parser and the option, the coarse plan and its own tuning table (TrellisI8Coarse).
// Links r4dx_model_linear (src/model/linear.cpp) and runs with HIP_VISIBLE_DEVICES=-1 (linking the library loads the
// HIP runtime, which must see no device here); the libr4d check is host code.
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "linear.h"
#include "prefill_int8.h"
#include "r4d.h"

namespace {

int g_fail = 0;
int g_checks = 0;
#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                      \
      std::printf(__VA_ARGS__);                                             \
      std::printf("\n");                                                    \
      ++g_fail;                                                             \
    }                                                                       \
  } while (0)

using namespace r4dx::model;

struct Cls {
  const char* name;
  int64_t N, K;
  int parts;
  int64_t part_n0;
};
const Cls kClasses[] = {
    {"mlp.gate_up", 34816, 5120, 2, 17408}, {"mlp.down", 5120, 17408, 1, 0},
    {"gdn.in_proj_qkv", 10240, 5120, 1, 0}, {"gdn.in_proj_z", 6144, 5120, 1, 0},
    {"gdn.out_proj/attn.o", 5120, 6144, 1, 0}, {"attn.qg", 12288, 5120, 1, 0},
    {"attn.k/attn.v", 1024, 5120, 1, 0},
};

void TestParser() {
  // int8 prefill is ON by default (2026-10-07); "0" / "off" is the kill switch
  CHECK(ParsePrefillInt8(nullptr) == kPrefillInt8On, "unset is on (the default)");
  CHECK(ParsePrefillInt8("") == kPrefillInt8On, "empty is on (the default)");
  CHECK(ParsePrefillInt8("0") == kPrefillInt8Off, "0 is off");
  CHECK(ParsePrefillInt8("off") == kPrefillInt8Off, "off is off");
  CHECK(ParsePrefillInt8("1") == kPrefillInt8On, "1 is on");
  CHECK(ParsePrefillInt8("on") == kPrefillInt8On, "on is on");
  std::fflush(stderr);
  CHECK(ParsePrefillInt8("yes") == kPrefillInt8On, "an unrecognized value keeps the default, on (with a warning on stderr)");
  CHECK(ParsePrefillInt8("2") == kPrefillInt8On, "2 is not recognized: the default, on");
  CHECK(ParsePrefillInt8("ON") == kPrefillInt8On, "the spellings are case sensitive: ON is not recognized: the default, on");
  CHECK(!ParsePrefillInt8Explicit(nullptr) && !ParsePrefillInt8Explicit("") && !ParsePrefillInt8Explicit("0") &&
            !ParsePrefillInt8Explicit("off") && !ParsePrefillInt8Explicit("yes"),
        "only 1 / on are an explicit request: unset, empty, off and unreadable are not");
  CHECK(ParsePrefillInt8Explicit("1") && ParsePrefillInt8Explicit("on"), "1 and on are explicit");
  CHECK(ResolvePrefillInt8Explicit(1) && !ResolvePrefillInt8Explicit(0), "option 1 is explicit, option 0 is not");
  CHECK(ResolvePrefillInt8Explicit(-1) == PrefillInt8RequestExplicit(), "option -1 follows the environment's explicitness");
  CHECK(ValidPrefillInt8Option(-1) && ValidPrefillInt8Option(0) && ValidPrefillInt8Option(1), "options -1, 0, 1 are valid");
  CHECK(!ValidPrefillInt8Option(2) && !ValidPrefillInt8Option(-2), "options 2 and -2 are not");
  CHECK(ResolvePrefillInt8Request(0) == kPrefillInt8Off, "option 0 forces off whatever the environment says");
  CHECK(ResolvePrefillInt8Request(1) == kPrefillInt8On, "option 1 forces on whatever the environment says");
  // option -1 follows the process environment (PrefillInt8Request caches it; the test environment does not set it)
  CHECK(ResolvePrefillInt8Request(-1) == PrefillInt8Request(), "option -1 follows R4DX_PREFILL_INT8");
}

PrefillInt8Inputs Good() {
  PrefillInt8Inputs in;
  in.requested = kPrefillInt8On;
  in.wide = true;
  in.tp_world = 1;
  in.has_trellis = true;
  in.rotated_container = false;
  return in;
}

void TestDecision() {
  const char* why = "unset";
  CHECK(DecidePrefillInt8(Good(), &why) && why == nullptr, "a wide TP = 1 trellis Model with the switch on uses it, no reason");
  CHECK(DecidePrefillInt8(Good()), "(why is optional)");
  {
    PrefillInt8Inputs in = Good();
    in.requested = kPrefillInt8Off;
    why = "unset";
    CHECK(!DecidePrefillInt8(in, &why) && why == nullptr, "off is a choice, not a fallback: no reason");
  }
  {
    PrefillInt8Inputs in = Good();
    in.wide = false;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("super-chunks") != std::string::npos,
          "a 64-row Model is refused with its reason");
  }
  {
    PrefillInt8Inputs in = Good();
    in.tp_world = 2;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("tensor parallel") != std::string::npos,
          "TP = 2 is refused (v1 keeps the f16 kernel)");
  }
  {
    PrefillInt8Inputs in = Good();
    in.has_trellis = false;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("trellis") != std::string::npos,
          "a container without trellis linears is refused");
  }
  {
    PrefillInt8Inputs in = Good();
    in.fakeq_active = true;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("FAKEQ") != std::string::npos,
          "R4DX_FAKEQ_ACT / R4DX_FAKEQ_W set: the default yields, with its reason");
  }
  {
    PrefillInt8Inputs in = Good();
    in.rotated_container = true;
    in.wide = false;   // (a rotated container is never wide; the rotation is the more specific reason)
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("quant2") != std::string::npos,
          "a rotated container names quant2, not the chunk size");
  }
  {
    PrefillInt8Inputs in = Good();
    in.tables_built = 0;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("int8 plan") != std::string::npos,
          "no linear with a table: refused after the load-time pass");
    in.tables_built = 12;
    CHECK(DecidePrefillInt8(in), "some tables built: used");
    in.tables_built = -1;
    CHECK(DecidePrefillInt8(in), "tables not built yet (the first decision): not a reason");
  }
  {
    // the order of the checks: rotated before wide before TP before trellis
    PrefillInt8Inputs in = Good();
    in.wide = false;
    in.tp_world = 2;
    in.has_trellis = false;
    why = nullptr;
    (void)DecidePrefillInt8(in, &why);
    CHECK(why != nullptr && std::string(why).find("super-chunks") != std::string::npos, "wide is checked before TP");
    in.wide = true;
    why = nullptr;
    (void)DecidePrefillInt8(in, &why);
    CHECK(why != nullptr && std::string(why).find("tensor parallel") != std::string::npos, "TP is checked before trellis");
  }
}

void TestPlans() {
  for (int kb : {4, 5}) {
    for (const Cls& c : kClasses) {
      const TrellisI8Plan p = PlanTrellisI8(c.N, c.K, kb, c.parts, c.part_n0);
      if (c.N == 10240 && c.K == 5120) {
        // excluded by the R4DX_M256_SHAPES filter main() sets (every other class passes it)
        CHECK(!p.ok && p.why.rfind("shape excluded", 0) == 0, "%s KB%d: the R4DX_M256_SHAPES filter must exclude it: %s", c.name, kb,
              p.why.c_str());
        continue;
      }
      CHECK(p.ok, "%s KB%d: no int8 plan: %s", c.name, kb, p.why.c_str());
      if (!p.ok) continue;
      const int64_t n_split = c.parts > 1 ? c.part_n0 : c.N;
      CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(c.K), static_cast<int>(c.N), static_cast<int>(n_split), kb, p.skw,
                                         p.skg) == nullptr,
            "%s KB%d: the plan's (skw %d, skg %d) is not legal for the kernel", c.name, kb, p.skw, p.skg);
      std::printf("  %-22s KB%d plan skw %d skg %d\n", c.name, kb, p.skw, p.skg);
    }
  }
  // refusals, each with its reason
  const auto why_of = [](int64_t N, int64_t K, int kb, int parts, int64_t n0) { return PlanTrellisI8(N, K, kb, parts, n0).why; };
  CHECK(why_of(5120, 17408, 6, 1, 0).find("KB") != std::string::npos, "KB 6 is refused");
  CHECK(why_of(5120, 17408, 4, 3, 0).find("parts") != std::string::npos, "3 parts are refused");
  CHECK(why_of(5120 + 64, 17408, 4, 1, 0).find("128") != std::string::npos, "N not a multiple of 128 is refused");
  CHECK(why_of(5120, 17408 + 64, 4, 1, 0).find("128") != std::string::npos, "K not a multiple of 128 is refused");
  CHECK(why_of(0, 5120, 4, 1, 0).find("range") != std::string::npos, "an empty shape is refused");
  CHECK(why_of(34816, 5120, 4, 2, 17408 + 64).find("boundary") != std::string::npos, "a part boundary inside a 128-block is refused");
  CHECK(why_of(34816, 5120, 4, 2, 0).find("boundary") != std::string::npos, "a part boundary of 0 is refused");
  CHECK(why_of(34816, 5120, 4, 2, 34816).find("boundary") != std::string::npos, "a part boundary of N is refused");
  CHECK(why_of(4096, 3840, 4, 1, 0).find("no int8 tuning row") != std::string::npos, "a shape without a table row (Gemma q_sliding) is refused");
  CHECK(why_of(1024, 5120, 5, 1, 0).empty(), "attn.k KB5 has a plan (empty reason)");
  CHECK(PlanTrellisI8(1024, 5120, 5, 1, 0).ok, "attn.k KB5 is ok");
  // a one-part linear ignores part_n0
  CHECK(PlanTrellisI8(5120, 17408, 4, 1, 12345).ok, "a one-part linear ignores part_n0");
}

void TestTable() {
  size_t n = 0;
  const TrellisI8Row* rows = TrellisI8Rows(&n);
  CHECK(rows != nullptr && n == 14, "the table has 14 rows (7 classes x 2 rates), got %zu", n);
  std::set<std::tuple<int64_t, int64_t, int>> keys;
  for (size_t i = 0; i < n; ++i) {
    const TrellisI8Row& r = rows[i];
    CHECK(keys.emplace(r.N, r.K, r.rate).second, "duplicate row for (N %lld, K %lld, KB %d)", static_cast<long long>(r.N),
          static_cast<long long>(r.K), r.rate);
    CHECK(r.rate == 4 || r.rate == 5, "row %zu: rate %d", i, r.rate);
    CHECK(r.skw == 2 || r.skw == 4 || r.skw == 8, "row %zu: skw %d", i, r.skw);
    CHECK(r.skg == 1 || r.skg == 2 || r.skg == 4 || r.skg == 8, "row %zu: skg %d", i, r.skg);
    // legal at the shape under the kernel's own rules, as a one-part launch and (every class has an 8-wide boundary) a split one
    CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(r.K), static_cast<int>(r.N), static_cast<int>(r.N), r.rate, r.skw, r.skg) == nullptr,
          "row %zu (N %lld, K %lld, KB %d): (skw %d, skg %d) is not legal", i, static_cast<long long>(r.N), static_cast<long long>(r.K),
          r.rate, r.skw, r.skg);
    // the LDS rule: 8 KiB per slice
    CHECK(8192 * r.skw <= 64 * 1024, "row %zu: LDS", i);
  }
  for (const Cls& c : kClasses)
    for (int kb : {4, 5}) CHECK(keys.count({c.N, c.K, kb}) == 1, "%s KB%d has no table row", c.name, kb);
}

void TestScope() {
  CHECK(!TrellisI8Active(), "off by default");
  {
    ScopedTrellisI8 a(true);
    CHECK(TrellisI8Active(), "on inside a scope");
    {
      ScopedTrellisI8 b(false);
      CHECK(!TrellisI8Active(), "an inner off scope turns it off");
    }
    CHECK(TrellisI8Active(), "and restores it on exit");
  }
  CHECK(!TrellisI8Active(), "off again after the scope");
}

// R4DX_PREFILL_INT8_FUSEDQ: the parser, and the one decision producers and ApplyLinear share. The positive case (a
// linear WITH its scale table inside both scopes) needs device memory for the table: tests/model/test_prefill_int8 has it.
void TestFusedQ() {
  CHECK(ParsePrefillInt8FusedQ(nullptr) && ParsePrefillInt8FusedQ("") && ParsePrefillInt8FusedQ("1") &&
            ParsePrefillInt8FusedQ("on"),
        "unset, empty, 1, on: the fused quantizer is on (the default)");
  CHECK(!ParsePrefillInt8FusedQ("0") && !ParsePrefillInt8FusedQ("off"), "0 and off are the kill switch");
  std::fflush(stderr);
  CHECK(ParsePrefillInt8FusedQ("yes") && ParsePrefillInt8FusedQ("ON") && ParsePrefillInt8FusedQ("2"),
        "an unrecognized value keeps the default, on (with a warning on stderr)");
  {
    // follows the environment the test runs under: a run with the kill switch set, to see it honored, must not fail here
    const char* st = std::getenv("R4DX_TRELLIS_A_STATS");
    const bool want = ParsePrefillInt8FusedQ(std::getenv("R4DX_PREFILL_INT8_FUSEDQ")) && !(st != nullptr && st[0] != '\0');
    CHECK(TrellisI8FusedQEnabled() == want, "TrellisI8FusedQEnabled() follows R4DX_PREFILL_INT8_FUSEDQ (on when unset), off under R4DX_TRELLIS_A_STATS");
  }
  QuantLinear w;
  w.layout = Layout::kTrellis;
  w.N = 5120;
  w.K = 6144;
  w.trellis_bits = 4;
  w.trellis_parts = 1;
  CHECK(!TrellisI8Takes(w, 256) && !TrellisI8FusedQ(w, 256), "outside the scopes nothing is taken or fused");
  {
    ScopedTrellisM256 m(true);
    ScopedTrellisI8 i(true);
    CHECK(!TrellisI8Takes(w, 256), "inside both scopes a linear without a scale table does not take int8 (ApplyLinear throws for it)");
    CHECK(!TrellisI8FusedQ(w, 256), "... so its producer does not fuse either: the consumer's decision is the producer's");
    CHECK(!TrellisI8Takes(w, 64) && !TrellisI8Takes(w, 255), "any M but 256 never takes int8");
  }
  w.layout = Layout::kBf16;
  {
    ScopedTrellisM256 m(true);
    ScopedTrellisI8 i(true);
    CHECK(!TrellisI8Takes(w, 256), "a non-trellis linear never does");
  }
}

// R4DX_PREFILL_INT8_SCALES=coarse (docs/int8-prefill.md "Coarse scales"): the parser and the option, the coarse plan and its
// table (same classes, same legality, its own rows), and the per-linear decision (a linear is coarse when it holds the
// column table; the positive case needs device memory: tests/model/test_prefill_int8 has it).
void TestScales() {
  CHECK(ParsePrefillInt8Scales(nullptr) == kPrefillInt8ScalesBlk128 && ParsePrefillInt8Scales("") == kPrefillInt8ScalesBlk128 &&
            ParsePrefillInt8Scales("blk128") == kPrefillInt8ScalesBlk128,
        "unset, empty, blk128: the per-128 scales (the default)");
  CHECK(ParsePrefillInt8Scales("coarse") == kPrefillInt8ScalesCoarse, "coarse is coarse");
  std::fflush(stderr);
  CHECK(ParsePrefillInt8Scales("Coarse") == kPrefillInt8ScalesBlk128 && ParsePrefillInt8Scales("1") == kPrefillInt8ScalesBlk128 &&
            ParsePrefillInt8Scales("row") == kPrefillInt8ScalesBlk128,
        "an unrecognized value keeps the default, blk128 (with a warning on stderr; the spelling is case sensitive)");
  CHECK(ValidPrefillInt8ScalesOption(-1) && ValidPrefillInt8ScalesOption(0) && ValidPrefillInt8ScalesOption(1) &&
            !ValidPrefillInt8ScalesOption(2) && !ValidPrefillInt8ScalesOption(-2),
        "options -1, 0, 1 are valid, 2 and -2 are not");
  CHECK(ResolvePrefillInt8Scales(0) == kPrefillInt8ScalesBlk128 && ResolvePrefillInt8Scales(1) == kPrefillInt8ScalesCoarse,
        "option 0 forces blk128, option 1 coarse, whatever the environment says");
  CHECK(ResolvePrefillInt8Scales(-1) == PrefillInt8ScalesRequest(), "option -1 follows R4DX_PREFILL_INT8_SCALES");
  CHECK(std::string(PrefillInt8ScalesName(kPrefillInt8ScalesCoarse)) == "coarse" &&
            std::string(PrefillInt8ScalesName(kPrefillInt8ScalesBlk128)) == "blk128", "the names");
  QuantLinear w;
  w.layout = Layout::kTrellis;
  CHECK(!TrellisI8Coarse(w), "a linear with no coarse table is not coarse");
}

void TestCoarsePlansAndTable() {
  for (int kb : {4, 5}) {
    for (const Cls& c : kClasses) {
      const TrellisI8Plan p = PlanTrellisI8(c.N, c.K, kb, c.parts, c.part_n0, /*coarse=*/true);
      if (c.N == 10240 && c.K == 5120) {
        CHECK(!p.ok && p.why.rfind("shape excluded", 0) == 0, "coarse %s KB%d: the R4DX_M256_SHAPES filter excludes it too: %s", c.name, kb,
              p.why.c_str());
        continue;
      }
      CHECK(p.ok, "coarse %s KB%d: no int8 plan: %s", c.name, kb, p.why.c_str());
      if (!p.ok) continue;
      const int64_t n_split = c.parts > 1 ? c.part_n0 : c.N;
      CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(c.K), static_cast<int>(c.N), static_cast<int>(n_split), kb, p.skw, p.skg) == nullptr,
            "coarse %s KB%d: the plan's (skw %d, skg %d) is not legal for the kernel", c.name, kb, p.skw, p.skg);
    }
  }
  const auto why_of = [](int64_t N, int64_t K, int kb, int parts, int64_t n0) { return PlanTrellisI8(N, K, kb, parts, n0, true).why; };
  CHECK(why_of(5120, 17408, 6, 1, 0).find("KB") != std::string::npos, "coarse: KB 6 is refused");
  CHECK(why_of(34816, 5120, 4, 2, 17408 + 64).find("boundary") != std::string::npos, "coarse: a part boundary inside a 128-block is refused");
  CHECK(why_of(4096, 3840, 4, 1, 0).find("no int8 tuning row") != std::string::npos, "coarse: a shape without a table row is refused");
  size_t n = 0, n_blk = 0;
  const TrellisI8Row* rows = TrellisI8Rows(&n, /*coarse=*/true);
  const TrellisI8Row* blk = TrellisI8Rows(&n_blk);
  CHECK(rows != nullptr && rows != blk && n == 14, "the coarse table is its own, 14 rows (7 classes x 2 rates), got %zu", n);
  std::set<std::tuple<int64_t, int64_t, int>> keys;
  for (size_t i = 0; i < n; ++i) {
    const TrellisI8Row& r = rows[i];
    CHECK(keys.emplace(r.N, r.K, r.rate).second, "coarse: duplicate row for (N %lld, K %lld, KB %d)", static_cast<long long>(r.N),
          static_cast<long long>(r.K), r.rate);
    CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(r.K), static_cast<int>(r.N), static_cast<int>(r.N), r.rate, r.skw, r.skg) == nullptr,
          "coarse row %zu (N %lld, K %lld, KB %d): (skw %d, skg %d) is not legal", i, static_cast<long long>(r.N), static_cast<long long>(r.K), r.rate,
          r.skw, r.skg);
  }
  for (const Cls& c : kClasses)
    for (int kb : {4, 5}) CHECK(keys.count({c.N, c.K, kb}) == 1, "coarse: %s KB%d has no table row", c.name, kb);
  // the same (N, K, KB) key set as the per-128 table
  std::set<std::tuple<int64_t, int64_t, int>> keys_blk;
  for (size_t i = 0; i < n_blk; ++i) keys_blk.emplace(blk[i].N, blk[i].K, blk[i].rate);
  CHECK(keys == keys_blk, "the coarse table covers exactly the classes the per-128 table does");
}

}  // namespace

int main() {
  // The R4DX_M256_SHAPES filter is read once, at the first plan call: set it before any (it excludes the 10240 x 5120 class only (4096 x 3840, a Gemma shape with no table row, is let through to reach the row lookup), which TestPlans expects to be refused for that reason).
#ifdef _WIN32
  _putenv_s("R4DX_M256_SHAPES", "34816x5120,5120x17408,6144x5120,5120x6144,12288x5120,1024x5120,4096x3840");
#else
  setenv("R4DX_M256_SHAPES", "34816x5120,5120x17408,6144x5120,5120x6144,12288x5120,1024x5120,4096x3840", 1);
#endif
  TestParser();
  TestDecision();
  TestPlans();
  TestTable();
  TestScope();
  TestFusedQ();
  TestScales();
  TestCoarsePlansAndTable();
  if (g_fail != 0) {
    std::printf("test_prefill_int8_cpu: %d of %d checks FAILED\n", g_fail, g_checks);
    return 1;
  }
  std::printf("test_prefill_int8_cpu: PASS (%d checks)\n", g_checks);
  return 0;
}
