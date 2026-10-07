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
//   * R4DX_PREFILL_INT8_SCALES: the parser and the option, the coarse plan and its own tuning table (TrellisI8Coarse);
//   * TP = 2 (docs/int8-prefill.md "Tensor parallel"): R4DX_PREFILL_INT8_TP2's parser / option, the decision table's TP rows, the
//     rank shapes of every trellis class DERIVED from the real sharding rules (tp/tp_shard.h RuleFor / RankRows / RankCols on the
//     27B's config) and checked against the expected list, every range whole 128-blocks (no K block straddles a shard), at least one
//     kernel-legal (skw, skg) per shard class and rate, the shard plan keyed by tp_shard (a shard reads the TP = 2 tables only, even
//     for a shape equal to a TP = 1 class's), the TP = 2 tables themselves (placeholder skipped, legality, keys), and what the
//     f16 M = 256 plan says at those shapes (informational).
// Links r4dx_model_linear (src/model/linear.cpp) and r4dx_tp_shard and runs with HIP_VISIBLE_DEVICES=-1 (linking the library
// loads the HIP runtime, which must see no device here); the libr4d check is host code.
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "linear.h"
#include "model_config.h"
#include "prefill_int8.h"
#include "r4d.h"
#include "tp/tp_shard.h"

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
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("tensor parallel") != std::string::npos &&
              std::string(why).find("R4DX_PREFILL_INT8_TP2") != std::string::npos,
          "TP = 2 is refused unless R4DX_PREFILL_INT8_TP2 asks (the reason names the switch)");
    in.tp2_enabled = true;
    why = "unset";
    CHECK(DecidePrefillInt8(in, &why) && why == nullptr, "TP = 2 with the switch on is used, no reason");
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
    in.tp2_enabled = true;   // with the TP switch on the TP reason is gone and the next one shows
    why = nullptr;
    (void)DecidePrefillInt8(in, &why);
    CHECK(why != nullptr && std::string(why).find("trellis linears") != std::string::npos, "TP switch on: trellis is the next reason");
  }
  {
    // after the load-time pass: no shard linear has a row (the TP = 2 tables are empty until the GPU run) -> the TP-specific reason
    PrefillInt8Inputs in = Good();
    in.tp_world = 2;
    in.tp2_enabled = true;
    in.tables_built = 0;
    why = nullptr;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("rank-shard") != std::string::npos &&
              std::string(why).find("--tp 2") != std::string::npos,
          "TP = 2 with no shard row: refused after the pass, the reason names the TP tables and the bench command");
    in.tables_built = 16;
    CHECK(DecidePrefillInt8(in), "TP = 2 with some shard rows: used (shard shapes without a row stay f16)");
    in.tp_world = 1;
    why = nullptr;
    in.tables_built = 0;
    CHECK(!DecidePrefillInt8(in, &why) && why != nullptr && std::string(why).find("rank-shard") == std::string::npos,
          "TP = 1 with no row keeps its own reason");
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

// ---- TP = 2 (docs/int8-prefill.md "Tensor parallel") ---------------------------------------------------------------------

void TestTp2Switch() {
  CHECK(!ParsePrefillInt8Tp2(nullptr) && !ParsePrefillInt8Tp2("") && !ParsePrefillInt8Tp2("0") && !ParsePrefillInt8Tp2("off"),
        "unset, empty, 0, off: int8 on TP shards is off (the default)");
  CHECK(ParsePrefillInt8Tp2("1") && ParsePrefillInt8Tp2("on"), "1 and on ask for it");
  std::fflush(stderr);
  CHECK(!ParsePrefillInt8Tp2("yes") && !ParsePrefillInt8Tp2("ON") && !ParsePrefillInt8Tp2("2"),
        "an unrecognized value keeps the default, off (with a warning on stderr; the spelling is case sensitive)");
  CHECK(ValidPrefillInt8Tp2Option(-1) && ValidPrefillInt8Tp2Option(0) && ValidPrefillInt8Tp2Option(1) && !ValidPrefillInt8Tp2Option(2) &&
            !ValidPrefillInt8Tp2Option(-2),
        "options -1, 0, 1 are valid, 2 and -2 are not");
  CHECK(!ResolvePrefillInt8Tp2(0) && ResolvePrefillInt8Tp2(1), "option 0 forces off, 1 on, whatever the environment says");
  CHECK(ResolvePrefillInt8Tp2(-1) == PrefillInt8Tp2Request(), "option -1 follows R4DX_PREFILL_INT8_TP2");
  CHECK(!PrefillInt8Tp2Request() || std::getenv("R4DX_PREFILL_INT8_TP2") != nullptr, "the request is off unless the environment sets it");
  QuantLinear w;
  CHECK(!w.trellis_tp_shard, "a linear is not a TP shard by default (TP = 1 loads and replicated linears never are)");
}

// The 27B's config (tests/model/test_tp_shard.cpp's RealConfig): the real RuleFor / RankRows / RankCols run on it.
r4dx::model::ModelConfig RealConfig() {
  r4dx::model::ModelConfig c;
  c.hidden_size = 5120;
  c.num_hidden_layers = 64;
  for (int i = 0; i < 64; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 24;
  c.num_key_value_heads = 4;
  c.head_dim = 256;
  c.intermediate_size = 17408;
  c.linear_num_key_heads = 16;
  c.linear_num_value_heads = 48;
  c.linear_key_head_dim = 128;
  c.linear_value_head_dim = 128;
  c.vocab_size = 248320;
  return c;
}

struct GlobalCls {
  const char* name;
  const char* base;       // the container base name RuleFor takes
  int64_t N, K;           // the global linear
  int parts;
  int64_t gate_rows;      // two-part linears: the first part's global rows
  int64_t rank_N, rank_K, rank_n0;   // EXPECTED rank shape (docs/tp.md 4.2) and part boundary (one part: 0)
};
const GlobalCls kGlobals[] = {
    {"mlp.gate_up", "text.layers.0.mlp.gate_up", 34816, 5120, 2, 17408, 17408, 5120, 8704},
    {"mlp.down", "text.layers.0.mlp.down", 5120, 17408, 1, 0, 5120, 8704, 0},
    {"gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_qkv", 10240, 5120, 1, 0, 5120, 5120, 0},
    {"gdn.in_proj_z", "text.layers.0.gdn.in_proj_z", 6144, 5120, 1, 0, 3072, 5120, 0},
    {"gdn.out_proj", "text.layers.0.gdn.out_proj", 5120, 6144, 1, 0, 5120, 3072, 0},
    {"attn.qg", "text.layers.3.attn.qg", 12288, 5120, 1, 0, 6144, 5120, 0},
    {"attn.k", "text.layers.3.attn.k", 1024, 5120, 1, 0, 512, 5120, 0},
    {"attn.v", "text.layers.3.attn.v", 1024, 5120, 1, 0, 512, 5120, 0},
    {"attn.o", "text.layers.3.attn.o", 5120, 6144, 1, 0, 5120, 3072, 0},
};

// The distinct rank shapes (the bench's kShapesTp2): gdn.out_proj and attn.o, attn.k and attn.v share one.
struct ShardShape {
  const char* name;
  int64_t N, K;
  int parts;
  int64_t n0;
};
const ShardShape kShards[] = {
    {"mlp.gate_up", 17408, 5120, 2, 8704}, {"mlp.down", 5120, 8704, 1, 0},       {"gdn.in_proj_qkv", 5120, 5120, 1, 0},
    {"gdn.in_proj_z", 3072, 5120, 1, 0},   {"gdn.out_proj/attn.o", 5120, 3072, 1, 0}, {"attn.qg", 6144, 5120, 1, 0},
    {"attn.k/attn.v", 512, 5120, 1, 0},
};

bool Has128(int64_t v) { return v > 0 && v % 128 == 0; }

// The rank shape of every trellis linear class, from the real sharding rules, against docs/tp.md 4.2's table; and the alignment
// the int8 kernel needs of it: N, K and the part boundary whole 128-blocks, every K shard boundary a whole 128-block (so no
// activation / weight scale block straddles two ranks and the per-(row, 128 k) and per-(column, 128 k) scales of a shard are the
// TP = 1 scales of its blocks).
void TestShardShapes() {
  const r4dx::model::ModelConfig cfg = RealConfig();
  namespace tp = r4dx::model::tp;
  for (const GlobalCls& g : kGlobals) {
    const tp::ShardRule rule = tp::RuleFor(g.base, cfg);
    CHECK(rule.split == tp::Split::kRows || rule.split == tp::Split::kCols, "%s: a trellis linear is column- or row-parallel", g.name);
    for (int r = 0; r < 2; ++r) {
      int64_t N = g.N, K = g.K, n0 = 0;
      if (rule.split == tp::Split::kRows) {
        const std::vector<tp::Range> rows = tp::RankRows(rule, 2, r);
        N = 0;
        for (const tp::Range& x : rows) {
          N += x.count;
          CHECK(Has128(x.begin) || x.begin == 0, "%s rank %d: row range begin %lld is not a whole 128-block", g.name, r, static_cast<long long>(x.begin));
          CHECK(Has128(x.count), "%s rank %d: row range count %lld is not a whole 128-block", g.name, r, static_cast<long long>(x.count));
        }
        n0 = g.parts > 1 ? rows[0].count : 0;   // ShardLoader: the rank's rows inside part 0
        CHECK(g.parts == 1 || rows.size() == 2, "%s: two segments for two parts", g.name);
      } else {
        const tp::Range c = tp::RankCols(rule, 2, r);
        K = c.count;
        CHECK(c.begin % 128 == 0 && Has128(c.count), "%s rank %d: K range [%lld, +%lld) is not whole 128-blocks (a scale block would straddle the shard)",
              g.name, r, static_cast<long long>(c.begin), static_cast<long long>(c.count));
        CHECK(c.begin == r * c.count, "%s rank %d: K shard starts at r * K / 2", g.name, r);
      }
      CHECK(N == g.rank_N && K == g.rank_K && n0 == g.rank_n0, "%s rank %d: shape %lld x %lld (part %lld), expected %lld x %lld (part %lld)", g.name, r,
            static_cast<long long>(N), static_cast<long long>(K), static_cast<long long>(n0), static_cast<long long>(g.rank_N),
            static_cast<long long>(g.rank_K), static_cast<long long>(g.rank_n0));
      CHECK(Has128(N) && Has128(K) && (g.parts == 1 || Has128(n0)), "%s rank %d: N, K, part boundary must be whole 128-blocks", g.name, r);
      // both ranks have the same shape (every plan, table row and scale table is then rank-independent)
    }
  }
  // the distinct shapes are kShards
  std::set<std::tuple<int64_t, int64_t, int, int64_t>> from_rules, listed;
  for (const GlobalCls& g : kGlobals) from_rules.emplace(g.rank_N, g.rank_K, g.parts, g.rank_n0);
  for (const ShardShape& s : kShards) listed.emplace(s.N, s.K, s.parts, s.n0);
  CHECK(from_rules == listed, "the seven distinct rank shapes are the ones the tables and the bench list");
  CHECK(sizeof(kShards) / sizeof(kShards[0]) == 7, "seven shard classes");
}

// Every rank shape has at least one (skw, skg) the kernel accepts at both rates: the table (and the bench's sweep) can always get a row.
// K / 128 must divide by skw * skg: 68 blocks (mlp.down, K 8704) allow only skw * skg of 2 or 4.
void TestShardLegality() {
  for (const ShardShape& s : kShards) {
    std::string legal;
    int n_legal = 0;
    for (int kb : {4, 5})
      for (int skw : {2, 4, 8})
        for (int skg : {1, 2, 4, 8}) {
          const int n_split = static_cast<int>(s.parts > 1 ? s.n0 : s.N);
          if (r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(s.K), static_cast<int>(s.N), n_split, kb, skw, skg) == nullptr) {
            if (kb == 4) {
              legal += " (" + std::to_string(skw) + "," + std::to_string(skg) + ")";
              ++n_legal;
            }
          }
        }
    CHECK(n_legal > 0, "%s %lld x %lld: no legal (skw, skg)", s.name, static_cast<long long>(s.N), static_cast<long long>(s.K));
    std::printf("  TP2 %-22s %5lld x %-5lld K/128 %3lld  legal (skw,skg):%s\n", s.name, static_cast<long long>(s.N), static_cast<long long>(s.K),
                static_cast<long long>(s.K / 128), legal.c_str());
    if (s.K == 8704) {
      CHECK(n_legal == 3 && r4d_gemm_trellis_nt_i8_check(256, 8704, 5120, 5120, 4, 2, 1) == nullptr &&
                r4d_gemm_trellis_nt_i8_check(256, 8704, 5120, 5120, 4, 2, 2) == nullptr &&
                r4d_gemm_trellis_nt_i8_check(256, 8704, 5120, 5120, 4, 4, 1) == nullptr &&
                r4d_gemm_trellis_nt_i8_check(256, 8704, 5120, 5120, 4, 4, 2) != nullptr &&
                r4d_gemm_trellis_nt_i8_check(256, 8704, 5120, 5120, 4, 8, 1) != nullptr,
            "mlp.down at K 8704 (68 blocks): only (2,1) (2,2) (4,1) are legal");
    }
  }
}

bool HasRow(const TrellisI8Row* rows, size_t n, int64_t N, int64_t K, int kb) {
  for (size_t i = 0; i < n; ++i)
    if (rows[i].N == N && rows[i].K == K && rows[i].rate == kb) return true;
  return false;
}

// PlanTrellisI8 with tp_shard: a shard reads the TP = 2 tables only; a TP = 1 linear never reads them; until the GPU run has filled
// the tables (today: empty) every shard is refused with the TP reason and runs f16, never a silent or a borrowed row.
void TestShardPlans() {
  for (bool coarse : {false, true}) {
    size_t n_tp = 0, n_main = 0;
    const TrellisI8Row* tp_rows = TrellisI8Rows(&n_tp, coarse, /*tp_shard=*/true);
    const TrellisI8Row* main_rows = TrellisI8Rows(&n_main, coarse, /*tp_shard=*/false);
    CHECK(tp_rows != nullptr && main_rows != nullptr && tp_rows != main_rows && n_tp >= 1, "%s: the shard table is its own and never zero-sized", coarse ? "coarse" : "blk128");
    for (int kb : {4, 5}) {
      for (const ShardShape& s : kShards) {
        const TrellisI8Plan shard = PlanTrellisI8(s.N, s.K, kb, s.parts, s.n0, coarse, /*tp_shard=*/true);
        const TrellisI8Plan tp1 = PlanTrellisI8(s.N, s.K, kb, s.parts, s.n0, coarse, /*tp_shard=*/false);
        const bool in_tp = HasRow(tp_rows, n_tp, s.N, s.K, kb), in_main = HasRow(main_rows, n_main, s.N, s.K, kb);
        CHECK(shard.ok == in_tp, "%s %s KB%d: the shard plan is ok iff the TP = 2 table has the row (ok %d, row %d): %s", coarse ? "coarse" : "blk128",
              s.name, kb, shard.ok, in_tp, shard.why.c_str());
        CHECK(tp1.ok == in_main, "%s %s KB%d: a TP = 1 linear of that shape is ok iff the TP = 1 table has the row (ok %d, row %d): %s",
              coarse ? "coarse" : "blk128", s.name, kb, tp1.ok, in_main, tp1.why.c_str());
        if (!shard.ok) {
          CHECK(shard.why.find("no int8 tuning row for this TP rank-shard") != std::string::npos && shard.why.find("--tp 2") != std::string::npos,
                "%s %s KB%d: the refusal says it is the TP table and how to fill it: %s", coarse ? "coarse" : "blk128", s.name, kb, shard.why.c_str());
        } else {
          const int n_split = static_cast<int>(s.parts > 1 ? s.n0 : s.N);
          CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(s.K), static_cast<int>(s.N), n_split, kb, shard.skw, shard.skg) == nullptr,
                "%s %s KB%d: the shard plan's (skw %d, skg %d) is not legal", coarse ? "coarse" : "blk128", s.name, kb, shard.skw, shard.skg);
        }
        if (s.N == 6144 && s.K == 5120 && !in_tp) {
          // the rank's attn.qg is gdn.in_proj_z's shape: the TP = 1 row exists, and the shard must not borrow it
          CHECK(tp1.ok && !shard.ok, "%s attn.qg shard KB%d: gdn.in_proj_z's TP = 1 row exists and is NOT used for the shard", coarse ? "coarse" : "blk128", kb);
        }
      }
    }
    // the shard table: placeholder skipped, no duplicate, legal, only shard shapes, rates 4 / 5
    std::set<std::tuple<int64_t, int64_t, int>> keys;
    std::set<std::pair<int64_t, int64_t>> shapes;
    for (const ShardShape& s : kShards) shapes.emplace(s.N, s.K);
    size_t real = 0;
    for (size_t i = 0; i < n_tp; ++i) {
      const TrellisI8Row& r = tp_rows[i];
      if (r.N == 0) {
        CHECK(r.K == 0 && r.rate == 0 && r.skw == 0 && r.skg == 0, "the placeholder is all zeros");
        continue;
      }
      ++real;
      CHECK(keys.emplace(r.N, r.K, r.rate).second, "TP table: duplicate row for (N %lld, K %lld, KB %d)", static_cast<long long>(r.N), static_cast<long long>(r.K), r.rate);
      CHECK(shapes.count({r.N, r.K}) == 1, "TP table row (N %lld, K %lld) is not a rank shard shape", static_cast<long long>(r.N), static_cast<long long>(r.K));
      CHECK(r.rate == 4 || r.rate == 5, "TP table row %zu: rate %d", i, r.rate);
      CHECK(r4d_gemm_trellis_nt_i8_check(256, static_cast<int>(r.K), static_cast<int>(r.N), static_cast<int>(r.N), r.rate, r.skw, r.skg) == nullptr,
            "TP table row %zu (N %lld, K %lld, KB %d): (skw %d, skg %d) is not legal", i, static_cast<long long>(r.N), static_cast<long long>(r.K), r.rate, r.skw, r.skg);
    }
    std::printf("  TP2 %s table: %zu real row(s) of 14 (a shape without one runs f16)\n", coarse ? "coarse" : "blk128", real);
  }
  // a refusal that is not about the table keeps its own reason for a shard too
  CHECK(PlanTrellisI8(5120, 8704, 6, 1, 0, false, true).why.find("KB") != std::string::npos, "shard: KB 6 is refused for its KB");
  CHECK(PlanTrellisI8(5120 + 64, 8704, 4, 1, 0, false, true).why.find("128") != std::string::npos, "shard: N not a multiple of 128 is refused");
  // the f16 side the shard falls back to / is benchmarked against: the M = 256 plan at the rank shapes, on a TP rank's thread (informational)
  SetTp2TuningForThisThread(true);
  for (int kb : {4, 5})
    for (const ShardShape& s : kShards) {
      const TrellisM256Plan p = PlanTrellisM256(s.N, s.K, kb, s.parts, s.n0);
      std::printf("  TP2 f16 M=256 plan %-22s KB%d: %s (SK %d SKG %d SKW %d)%s%s\n", s.name, kb, p.ok ? "ok" : "NONE", p.SK, p.SKG, p.SKW,
                  p.ok ? "" : " why: ", p.ok ? "" : p.why.c_str());
    }
  SetTp2TuningForThisThread(false);
}

}  // namespace

int main() {
  // The R4DX_M256_SHAPES filter is read once, at the first plan call: set it before any (it excludes the 10240 x 5120 class only (4096 x 3840, a Gemma shape with no table row, is let through to reach the row lookup), which TestPlans expects to be refused for that reason).
#ifdef _WIN32
  // (plus the TP = 2 rank shapes, 17408x5120 5120x8704 5120x5120 3072x5120 5120x3072 512x5120; 6144x5120 is both a TP = 1 and a rank shape)
  _putenv_s("R4DX_M256_SHAPES",
            "34816x5120,5120x17408,6144x5120,5120x6144,12288x5120,1024x5120,4096x3840,17408x5120,5120x8704,5120x5120,3072x5120,5120x3072,512x5120");
#else
  setenv("R4DX_M256_SHAPES",
         "34816x5120,5120x17408,6144x5120,5120x6144,12288x5120,1024x5120,4096x3840,17408x5120,5120x8704,5120x5120,3072x5120,5120x3072,512x5120", 1);
#endif
  TestParser();
  TestDecision();
  TestPlans();
  TestTable();
  TestScope();
  TestFusedQ();
  TestScales();
  TestCoarsePlansAndTable();
  TestTp2Switch();
  TestShardShapes();
  TestShardLegality();
  TestShardPlans();
  if (g_fail != 0) {
    std::printf("test_prefill_int8_cpu: %d of %d checks FAILED\n", g_fail, g_checks);
    return 1;
  }
  std::printf("test_prefill_int8_cpu: PASS (%d checks)\n", g_checks);
  return 0;
}
