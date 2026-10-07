// test_fake_quant_act_cpu: pure CPU unit test of src/model/fake_quant_act.h -- the R4DX_FAKEQ_ACT parser and the
// fp32 reference of the int8 round trip (docs/int8-prefill.md) that the GPU kernel r4dx_fake_quant_act_f16 is
// held to (tests/kernels/test_fake_quant_act, GPU) -- and of src/model/fake_quant_w.h, the weight side
// (R4DX_FAKEQ_W: parser, the per-(column, k group) scale and rounding reference, the scale-table layout) that
// libr4d's `_wq` trellis kernels are held to (tests/kernels/test_fake_quant_w, GPU). No HIP, no container.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fake_quant_act.h"
#include "fake_quant_w.h"

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

int Parse(const char* e, bool* ok = nullptr) {
  int mode = -1;
  const bool r = ParseFakeQuantAct(e, &mode);
  if (ok != nullptr) *ok = r;
  return mode;
}

// Bit-exact comparison (so -0 and +0 differ, a NaN equals itself).
bool SameBits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
}  // namespace

int main() {
  // ---- parser ----
  bool ok = false;
  CHECK(Parse(nullptr, &ok) == kFakeQuantOff && ok, "unset -> off");
  CHECK(Parse("", &ok) == kFakeQuantOff && ok, "empty -> off");
  CHECK(Parse("off", &ok) == kFakeQuantOff && ok, "off -> off");
  CHECK(Parse("row", &ok) == kFakeQuantRow && ok, "row");
  CHECK(Parse("blk128", &ok) == kFakeQuantBlk128 && ok, "blk128");
  CHECK(Parse("blk32", &ok) == kFakeQuantBlk32 && ok, "blk32");
  for (const char* bad : {"ROW", "Row", "blk64", "blk", "1", "on", "row ", " row", "int8", "0"}) {
    Parse(bad, &ok);
    CHECK(!ok, "'%s' must be rejected (an experiment must not silently run unquantized)", bad);
  }
  {
    int mode = 77;
    ParseFakeQuantAct("nope", &mode);
    CHECK(mode == 77, "a rejected value leaves *mode alone");
  }
  CHECK(kFakeQuantOff == 0 && kFakeQuantRow == 1 && kFakeQuantBlk128 == 2 && kFakeQuantBlk32 == 3,
        "mode values are r4dx_fake_quant_act_f16's `mode` argument");
  CHECK(FakeQuantActGroup(kFakeQuantRow) == 0 && FakeQuantActGroup(kFakeQuantBlk128) == 128 &&
            FakeQuantActGroup(kFakeQuantBlk32) == 32,
        "scale group widths");
  CHECK(std::strcmp(FakeQuantActName(kFakeQuantBlk128), "blk128") == 0 &&
            std::strcmp(FakeQuantActName(kFakeQuantOff), "off") == 0,
        "names round-trip the spellings");
  for (int m : {kFakeQuantOff, kFakeQuantRow, kFakeQuantBlk128, kFakeQuantBlk32}) {
    int back = -1;
    CHECK(ParseFakeQuantAct(FakeQuantActName(m), &back) && back == m, "name of mode %d parses back", m);
  }

  // ---- reference: hand values ----
  {
    // amax 127 -> s = 1 exactly: q = rint(x) with ties to even (-63.5 -> -64, 0.5 -> 0, 1.5 -> 2, 2.5 -> 2,
    // -2.5 -> -2), a plain value rounds to nearest (126.4 -> 126).
    const float x[8] = {127.f, -63.5f, 0.5f, 0.f, 1.5f, 2.5f, -2.5f, 126.4f};
    const float want[8] = {127.f, -64.f, 0.f, 0.f, 2.f, 2.f, -2.f, 126.f};
    float v[8], s = -1.f;
    FakeQuantGroupRef(x, 8, v, &s);
    CHECK(s == 1.0f, "scale of an amax-127 group is 1, got %g", s);
    for (int i = 0; i < 8; ++i) CHECK(v[i] == want[i], "s=1 element %d: %g, want %g", i, v[i], want[i]);
  }
  {
    // amax 254 -> s = 2 exactly (254 / 127): x / s = 127, 50, -25.5 (tie -> -26, even), 1.5 (tie -> 2).
    const float x[4] = {254.f, 100.f, -51.f, 3.f};
    const float want[4] = {254.f, 100.f, -52.f, 4.f};
    float v[4], s = -1.f;
    FakeQuantGroupRef(x, 4, v, &s);
    CHECK(s == 2.0f, "scale %g, want 2", s);
    for (int i = 0; i < 4; ++i) CHECK(v[i] == want[i], "s=2 element %d: %g, want %g", i, v[i], want[i]);
  }
  {
    // The sign is symmetric (no zero point): negating the group negates the result bit for bit.
    const float x[6] = {5.f, -3.f, 0.7f, 2.2f, -4.9f, 0.01f};
    float nx[6], v[6], nv[6];
    for (int i = 0; i < 6; ++i) nx[i] = -x[i];
    FakeQuantGroupRef(x, 6, v);
    FakeQuantGroupRef(nx, 6, nv);
    for (int i = 0; i < 6; ++i) CHECK(SameBits(nv[i], -v[i]), "symmetry at %d: %g vs %g", i, nv[i], v[i]);
  }
  {
    // A group of zeros (and of -0) gives +0; scale 0.
    const float x[4] = {0.f, -0.f, 0.f, -0.f};
    float v[4], s = -1.f;
    FakeQuantGroupRef(x, 4, v, &s);
    CHECK(s == 0.0f, "zero group scale");
    for (int i = 0; i < 4; ++i) CHECK(SameBits(v[i], 0.0f), "zero group element %d is +0", i);
  }
  {
    // A tiny element next to a huge one is flushed (s ~ 1): the int8 resolution limit.
    const float x[3] = {127.f, 0.49f, -0.49f};
    float v[3];
    FakeQuantGroupRef(x, 3, v);
    CHECK(v[0] == 127.f && v[1] == 0.f && v[2] == 0.f, "elements below s/2 flush to zero");
  }
  {
    // Non-finite: the group is left exactly as it is.
    const float inf = INFINITY;
    const float x[4] = {inf, 1.f, -2.f, 0.3f};
    float v[4];
    FakeQuantGroupRef(x, 4, v);
    for (int i = 0; i < 4; ++i) CHECK(SameBits(v[i], x[i]), "non-finite group passes through (element %d)", i);
    const float y[3] = {NAN, 1.f, 5.f};
    float w[3];
    FakeQuantGroupRef(y, 3, w);
    CHECK(std::isnan(w[0]) && w[1] == 1.f && w[2] == 5.f, "a NaN group passes through unquantized");
  }

  // ---- reference: scale groups of a row ----
  {
    const int K = 256;
    std::vector<float> x(K), v(K);
    for (int i = 0; i < 128; ++i) x[i] = (i == 0) ? 127.f : 0.4f;      // block 0: amax 127
    for (int i = 128; i < K; ++i) x[i] = (i == 128) ? 1.27f : 0.4f;    // block 1: amax 1.27
    FakeQuantRowRef(x.data(), K, kFakeQuantRow, v.data());
    // one scale for the row (s = 1): every 0.4 flushes
    CHECK(v[0] == 127.f && v[1] == 0.f && v[200] == 0.f, "row: block 1's 0.4s flush under the row's scale");
    FakeQuantRowRef(x.data(), K, kFakeQuantBlk128, v.data());
    const float s1 = 1.27f / 127.f;
    CHECK(v[1] == 0.f, "blk128: block 0 (s = 1) still flushes 0.4");
    CHECK(std::fabs(v[200] - 0.4f) <= 0.5f * s1 * 1.0001f && v[200] != 0.f,
          "blk128: block 1 has its own scale, 0.4 survives (got %g)", v[200]);
    CHECK(std::fabs(v[128] - 1.27f) <= 1e-6f, "blk128: block 1's own max reproduces (got %g)", v[128]);
    // blk32: four sub-blocks of block 0; give sub-block 1 (cols 32..63) a small amax
    for (int i = 32; i < 64; ++i) x[i] = 0.4f * (i == 40 ? 2.f : 1.f);  // amax 0.8
    FakeQuantRowRef(x.data(), K, kFakeQuantBlk32, v.data());
    CHECK(v[1] == 0.f, "blk32: sub-block 0 (amax 127) flushes 0.4");
    CHECK(std::fabs(v[33] - 0.4f) <= 0.5f * (0.8f / 127.f) * 1.0001f && v[33] != 0.f,
          "blk32: sub-block 1 has its own scale (got %g)", v[33]);
    CHECK(std::fabs(v[40] - 0.8f) <= 1e-6f, "blk32: sub-block 1's own max reproduces (got %g)", v[40]);
    FakeQuantRowRef(x.data(), K, kFakeQuantBlk128, v.data());
    CHECK(v[33] == 0.f, "blk128: the same element flushes under the 128-block's scale");
    // off is the identity
    FakeQuantRowRef(x.data(), K, kFakeQuantOff, v.data());
    for (int i = 0; i < K; ++i) CHECK(SameBits(v[i], x[i]), "off is the identity (element %d)", i);
  }

  // ---- reference: properties on pseudo-random rows ----
  {
    uint32_t st = 12345u;
    const auto rnd = [&st] {  // uniform in [-1, 1)
      st = st * 1664525u + 1013904223u;
      return static_cast<float>(st >> 8) / static_cast<float>(1u << 23) - 1.0f;
    };
    const int K = 512;
    std::vector<float> x(K), v(K);
    int bad_bound = 0, bad_grid = 0, bad_max = 0, groups = 0;
    for (int trial = 0; trial < 200; ++trial) {
      const float mag = std::exp2(4.0f * rnd());
      for (int i = 0; i < K; ++i) x[i] = rnd() * mag * (i % 61 == 0 ? 8.f : 1.f);
      for (int mode : {kFakeQuantRow, kFakeQuantBlk128, kFakeQuantBlk32}) {
        FakeQuantRowRef(x.data(), K, mode, v.data());
        const int g = FakeQuantActGroup(mode) == 0 ? K : FakeQuantActGroup(mode);
        for (int c = 0; c < K; c += g) {
          float amax = 0.f;
          for (int i = c; i < c + g; ++i) amax = std::fmax(amax, std::fabs(x[i]));
          const float s = amax / 127.f;
          ++groups;
          float vmax = 0.f;
          for (int i = c; i < c + g; ++i) {
            if (std::fabs(v[i] - x[i]) > 0.5f * s * 1.0001f) ++bad_bound;  // the int8 round-off bound
            const float q = v[i] / s;
            if (std::fabs(q - std::rint(q)) > 1e-3f || std::fabs(q) > 127.0001f) ++bad_grid;  // on the grid
            vmax = std::fmax(vmax, std::fabs(v[i]));
          }
          if (std::fabs(vmax - amax) > 1e-5f * amax) ++bad_max;  // the group's max maps to +-127 s
        }
      }
    }
    CHECK(bad_bound == 0, "%d elements farther than s / 2 from x (of %d groups)", bad_bound, groups);
    CHECK(bad_grid == 0, "%d results off the int8 grid", bad_grid);
    CHECK(bad_max == 0, "%d groups whose max did not survive", bad_max);
  }

  // ======== R4DX_FAKEQ_W: fake_quant_w.h ========
  // ---- parser ----
  {
    const auto parse_w = [](const char* e, bool* ok) {
      int mode = -1;
      *ok = ParseFakeQuantW(e, &mode);
      return mode;
    };
    bool okw = false;
    CHECK(parse_w(nullptr, &okw) == kFakeQuantWOff && okw, "W: unset -> off");
    CHECK(parse_w("", &okw) == kFakeQuantWOff && okw, "W: empty -> off");
    CHECK(parse_w("off", &okw) == kFakeQuantWOff && okw, "W: off -> off");
    CHECK(parse_w("col128", &okw) == kFakeQuantWCol128 && okw, "W: col128");
    CHECK(parse_w("col32", &okw) == kFakeQuantWCol32 && okw, "W: col32");
    for (const char* bad :
         {"COL128", "Col32", "col64", "col", "blk128", "row", "128", "1", "on", "int8", "col128 ", " col32"}) {
      parse_w(bad, &okw);
      CHECK(!okw, "W: '%s' must be rejected (an experiment must not silently run unquantized)", bad);
    }
    int keep = 77;
    ParseFakeQuantW("nope", &keep);
    CHECK(keep == 77, "W: a rejected value leaves *mode alone");
    CHECK(kFakeQuantWOff == 0 && kFakeQuantWCol128 == 1 && kFakeQuantWCol32 == 2, "W: mode values");
    CHECK(FakeQuantWGroup(kFakeQuantWCol128) == 128 && FakeQuantWGroup(kFakeQuantWCol32) == 32 &&
              FakeQuantWGroup(kFakeQuantWOff) == 0,
          "W: scale group widths");
    // the kernels' `gsh` is log2 of k-tiles of 16: 16 << gsh is the group
    CHECK((16 << FakeQuantWGroupShift(kFakeQuantWCol128)) == 128 &&
              (16 << FakeQuantWGroupShift(kFakeQuantWCol32)) == 32,
          "W: group shift matches the group width");
    for (int m : {kFakeQuantWOff, kFakeQuantWCol128, kFakeQuantWCol32}) {
      int back = -1;
      CHECK(ParseFakeQuantW(FakeQuantWName(m), &back) && back == m, "W: name of mode %d parses back", m);
    }
    // the activation and weight spellings are disjoint where they could be confused
    int dummy = 0;
    CHECK(!ParseFakeQuantW("blk128", &dummy) && !ParseFakeQuantAct("col128", &dummy),
          "W: R4DX_FAKEQ_W does not take R4DX_FAKEQ_ACT's spellings and vice versa");
  }

  // ---- scale ----
  {
    const float a[4] = {127.f, -3.f, 0.f, 50.f};
    CHECK(FakeQuantWScaleRef(a, 4) == 1.0f, "W: amax 127 -> s = 1");
    const float b[3] = {-254.f, 1.f, 3.f};
    CHECK(FakeQuantWScaleRef(b, 3) == 2.0f, "W: amax 254 (negative max) -> s = 2");
    const float z[4] = {0.f, -0.f, 0.f, 0.f};
    CHECK(FakeQuantWScaleRef(z, 4) == 1.0f, "W: an all-zero group gets s = 1 (so 1 / s is finite)");
    const float one[1] = {0.3f};
    CHECK(FakeQuantWScaleRef(one, 1) == 0.3f / 127.0f, "W: s = amax / 127, one fp32 division");
  }

  // ---- rounding, by hand (s = 1: q = rint(w), half to even; clamp at +-127) ----
  {
    const float in[10] = {0.5f, 1.5f, 2.5f, -2.5f, -0.5f, 126.4f, 127.f, -127.f, 300.f, -300.f};
    const float want[10] = {0.f, 2.f, 2.f, -2.f, -0.f, 126.f, 127.f, -127.f, 127.f, -127.f};
    for (int i = 0; i < 10; ++i) {
      const float got = FakeQuantWRoundRef(in[i], 1.0f);
      CHECK(SameBits(got, want[i]), "W: round(%g, s=1) = %g, want %g", in[i], got, want[i]);
    }
    // s = 2: grid 2 q; 3 / 2 = 1.5 ties to 2 -> 4, 5 / 2 = 2.5 ties to 2 -> 4, -7 / 2 = -3.5 -> -4 -> -8
    CHECK(FakeQuantWRoundRef(3.f, 2.f) == 4.f && FakeQuantWRoundRef(5.f, 2.f) == 4.f &&
              FakeQuantWRoundRef(-7.f, 2.f) == -8.f,
          "W: ties to even on a scale of 2");
    // symmetric: negating the input negates the output bit for bit, for scales that are not powers of two too
    uint32_t st = 777u;
    const auto rnd = [&st] {  // uniform in [-1, 1)
      st = st * 1664525u + 1013904223u;
      return static_cast<float>(st >> 8) / static_cast<float>(1u << 23) - 1.0f;
    };
    int asym = 0;
    for (int i = 0; i < 20000; ++i) {
      const float s = std::exp2(6.0f * rnd()) / 127.0f * (1.0f + 0.1f * rnd());
      const float w = rnd() * 130.0f * s;
      if (!SameBits(FakeQuantWRoundRef(-w, s), -FakeQuantWRoundRef(w, s))) ++asym;
    }
    CHECK(asym == 0, "W: %d asymmetric roundings (no zero point)", asym);
  }

  // ---- rounding, properties over pseudo-random columns, both group widths ----
  {
    uint32_t st = 4242u;
    const auto rnd = [&st] {
      st = st * 1664525u + 1013904223u;
      return static_cast<float>(st >> 8) / static_cast<float>(1u << 23) - 1.0f;
    };
    int bad_bound = 0, bad_grid = 0, bad_max = 0, groups = 0;
    for (int n : {32, 128}) {
      std::vector<float> w(static_cast<size_t>(n));
      for (int trial = 0; trial < 2000; ++trial) {
        const float mag = std::exp2(3.0f * rnd());
        float amax = 0.f;
        for (int i = 0; i < n; ++i) {
          w[static_cast<size_t>(i)] = rnd() * mag * (i % 29 == 0 ? 3.f : 1.f);
          amax = std::fmax(amax, std::fabs(w[static_cast<size_t>(i)]));
        }
        const float s = FakeQuantWScaleRef(w.data(), n);
        ++groups;
        float vmax = 0.f;
        for (int i = 0; i < n; ++i) {
          const float v = FakeQuantWRoundRef(w[static_cast<size_t>(i)], s);
          if (std::fabs(v - w[static_cast<size_t>(i)]) > 0.5f * s * 1.0001f) ++bad_bound;  // the int8 round-off bound
          const float q = v / s;
          if (std::fabs(q - std::rint(q)) > 1e-3f || std::fabs(q) > 127.0001f) ++bad_grid;
          vmax = std::fmax(vmax, std::fabs(v));
        }
        if (std::fabs(vmax - amax) > 1e-5f * amax) ++bad_max;  // the column's max maps to +-127 s
      }
    }
    CHECK(bad_bound == 0, "W: %d elements farther than s / 2 from w (of %d groups)", bad_bound, groups);
    CHECK(bad_grid == 0, "W: %d results off the int8 grid", bad_grid);
    CHECK(bad_max == 0, "W: %d groups whose max did not survive", bad_max);
  }

  // ---- table layout and apply: Q[K][N], K = 256, N = 3 ----
  {
    const int K = 256, N = 3;
    std::vector<float> Q(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) {
      Q[static_cast<size_t>(k) * N + 0] = (k == 5) ? 127.f : 0.4f;                          // col 0: group 0 amax 127
      Q[static_cast<size_t>(k) * N + 1] = (k == 200) ? -2.54f : (k == 40 ? 0.4f : 0.01f);   // col 1
      Q[static_cast<size_t>(k) * N + 2] = 0.f;                                              // col 2: all zero
    }
    const std::vector<float> t128 = FakeQuantWTableRef(Q.data(), K, N, 128);
    CHECK(t128.size() == 2u * N, "W: col128 table is [K/128][N], got %zu floats", t128.size());
    CHECK(t128[0 * N + 0] == 1.0f, "W: col128 [g0][c0] = 127/127");
    CHECK(t128[1 * N + 0] == 0.4f / 127.0f, "W: col128 [g1][c0] = 0.4/127");
    CHECK(t128[0 * N + 1] == 0.4f / 127.0f, "W: col128 [g0][c1]: the 0.4 at k = 40 is the max of its group");
    CHECK(t128[1 * N + 1] == 2.54f / 127.0f, "W: col128 [g1][c1]: k = 200 is in group 1");
    CHECK(t128[0 * N + 2] == 1.0f && t128[1 * N + 2] == 1.0f, "W: an all-zero column gets 1.0f");
    const std::vector<float> t32 = FakeQuantWTableRef(Q.data(), K, N, 32);
    CHECK(t32.size() == 8u * N, "W: col32 table is [K/32][N], got %zu floats", t32.size());
    CHECK(t32[1 * N + 1] == 0.4f / 127.0f && t32[0 * N + 1] == 0.01f / 127.0f,
          "W: col32: k = 40 is in group 1, group 0 of column 1 has amax 0.01");
    CHECK(t32[6 * N + 1] == 2.54f / 127.0f, "W: col32: k = 200 is in group 6");
    // apply: column 0 under col128: group 0's 0.4s flush (s = 1), group 1's survive (own scale 0.4 / 127)
    const std::vector<float> q128 = FakeQuantWApplyRef(Q.data(), K, N, 128, t128);
    CHECK(q128[5 * N + 0] == 127.f && q128[7 * N + 0] == 0.f, "W: col128 column 0, group 0: max kept, 0.4 flushed");
    CHECK(std::fabs(q128[200 * N + 0] - 0.4f) <= 0.5f * (0.4f / 127.0f) * 1.0001f && q128[200 * N + 0] != 0.f,
          "W: col128 column 0, group 1 has its own scale (got %g)", q128[200 * N + 0]);
    // col32: k = 40 of column 0 is in group 1 (0.4 only), so it keeps its value where col128 flushed it
    const std::vector<float> q32 = FakeQuantWApplyRef(Q.data(), K, N, 32, t32);
    CHECK(q32[6 * N + 0] == 0.f && std::fabs(q32[40 * N + 0] - 0.4f) <= 0.5f * (0.4f / 127.0f) * 1.0001f &&
              q32[40 * N + 0] != 0.f,
          "W: col32: column 0's 0.4 at k = 40 has its own group (got %g), k = 6 shares group 0 with the 127",
          q32[40 * N + 0]);
    CHECK(q128[40 * N + 0] == 0.f, "W: col128 flushes that same element");
    for (int k = 0; k < K; ++k) CHECK(q32[static_cast<size_t>(k) * N + 2] == 0.f, "W: the zero column stays zero");
  }

  if (g_fail != 0) {
    std::printf("test_fake_quant_act_cpu: %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("test_fake_quant_act_cpu: PASS\n");
  return 0;
}
