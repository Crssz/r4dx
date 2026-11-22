// mxfp8 (docs/inkling2-plan.md 4.1): the e4m3 codec, the E8M0 scale rule, EncodeGroup, the converter's packers
// (quant_mxfp8.hpp) and the LayoutSet plumbing (linear_layouts.hpp). CPU only, no data files.
//
//  * all 256 e4m3 codes decode to an independently computed value; every non-NaN code encodes back to itself;
//  * round-to-nearest-even at every midpoint between adjacent codes (subnormal -> normal and the top included),
//    saturation at +-448, NaN -> 0, the sign of zero;
//  * ScaleExponent on amax 0 / tiny / huge / exact powers / one ulp either side, and the "no element clipped, minimal
//    exponent" invariant on random amax;
//  * EncodeGroup: zero groups, exact values, the error bound on Gaussian groups, no clipping;
//  * the packers: every (row, k) code lands at the byte the layout documents (hand-derived index), ws is one byte per
//    (row, group) in w8a16's ws element order, DequantMxfp8 inverts both;
//  * the raw-copy rule for checkpoint MXFP8 groups: encode(decode(x)) == x exactly when the group's largest code
//    magnitude exceeds 224 -- and not otherwise, which is why the converter copies checkpoint groups RAW.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/mxfp8_format.hpp"
#include "r4dx_convert/quant_mxfp8.hpp"

namespace mx = r4dx_convert::mxfp8;
using r4dx_convert::LayoutSet;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
}

// Independent e4m3fn decode (OCP spec text): sign, 4 exponent bits (bias 7), 3 mantissa bits; subnormal e == 0.
double RefDecode(int c) {
  const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
  double v = e == 0 ? m / 8.0 * std::pow(2.0, -6) : (1.0 + m / 8.0) * std::pow(2.0, e - 7);
  return s ? -v : v;
}

bool IsNan(int c) { return (c & 0x7F) == 0x7F; }

void TestE4m3Codec() {
  std::printf("---- e4m3 codec ----\n");
  bool decode_ok = true, roundtrip_ok = true, nan_ok = true;
  for (int c = 0; c < 256; ++c) {
    const float d = mx::DecodeE4m3(static_cast<uint8_t>(c));
    if (IsNan(c)) {
      nan_ok = nan_ok && std::isnan(d);
      continue;
    }
    decode_ok = decode_ok && (static_cast<double>(d) == RefDecode(c)) && (std::signbit(d) == ((c & 0x80) != 0));
    roundtrip_ok = roundtrip_ok && mx::EncodeE4m3(d) == c;
  }
  Check(decode_ok, "all 254 non-NaN codes decode to the OCP value (sign of zero included)");
  Check(nan_ok, "0x7F and 0xFF decode to NaN");
  Check(roundtrip_ok, "encode(decode(c)) == c for every non-NaN code, subnormals and +-0 included");
  Check(mx::DecodeE4m3(0x01) == std::ldexp(1.0f, -9) && mx::DecodeE4m3(0x07) == 7.0f * std::ldexp(1.0f, -9) &&
            mx::DecodeE4m3(0x08) == std::ldexp(1.0f, -6) && mx::DecodeE4m3(0x38) == 1.0f && mx::DecodeE4m3(0x7E) == 448.0f &&
            mx::DecodeE4m3(0xFE) == -448.0f,
        "spot values: 0x01 = 2^-9, 0x07 = 7 * 2^-9, 0x08 = 2^-6, 0x38 = 1, 0x7E = 448");
}

void TestRounding() {
  std::printf("---- e4m3 rounding ----\n");
  // Midpoints between adjacent non-NaN positive codes: ties go to the EVEN code (RNE), one float either side of the
  // midpoint goes to the nearer code. Codes are monotonic in value, so "even code" is "even mantissa LSB".
  bool ties_ok = true, near_ok = true;
  for (int c = 0; c < 0x7E; ++c) {
    const double lo = RefDecode(c), hi = RefDecode(c + 1), mid = 0.5 * (lo + hi);
    const float fm = static_cast<float>(mid);
    if (static_cast<double>(fm) != mid) continue;  // not exactly representable (never happens for e4m3 pairs)
    const uint8_t want = (c % 2 == 0) ? c : c + 1;
    ties_ok = ties_ok && mx::EncodeE4m3(fm) == want && mx::EncodeE4m3(-fm) == (want | 0x80);
    near_ok = near_ok && mx::EncodeE4m3(std::nextafterf(fm, 1e30f)) == c + 1 &&
              mx::EncodeE4m3(std::nextafterf(fm, 0.0f)) == c;
  }
  Check(ties_ok, "every midpoint between adjacent codes rounds to the even code (both signs)");
  Check(near_ok, "one float above / below every midpoint rounds to the nearer code");
  Check(mx::EncodeE4m3(std::ldexp(7.5f, -9)) == 0x08, "subnormal -> normal tie 7.5 * 2^-9 goes to 0x08 (even)");
  Check(mx::EncodeE4m3(std::ldexp(1.0f, -10)) == 0x00 && mx::EncodeE4m3(-std::ldexp(1.0f, -10)) == 0x80,
        "2^-10 (tie between 0 and the first subnormal) goes to zero; -2^-10 to -0 (0x80)");
  Check(mx::EncodeE4m3(std::nextafterf(std::ldexp(1.0f, -10), 1.0f)) == 0x01, "just above 2^-10 is the first subnormal");
  Check(mx::EncodeE4m3(432.0f) == 0x7E, "tie 432 between 416 and 448 goes to 448 (even)");
  bool sat_ok = true;
  for (float x : {448.0f, 449.0f, 464.0f, 480.0f, 1.0e30f, FLT_MAX, INFINITY}) sat_ok = sat_ok && mx::EncodeE4m3(x) == 0x7E && mx::EncodeE4m3(-x) == 0xFE;
  Check(sat_ok, "everything >= 448 (and inf) saturates to +-0x7E, never a NaN code");
  Check(mx::EncodeE4m3(NAN) == 0, "a NaN input maps to 0 (the caller's bug, never a NaN code)");
  Check(mx::EncodeE4m3(0.0f) == 0x00 && mx::EncodeE4m3(-0.0f) == 0x80, "+0 -> 0x00, -0 -> 0x80");
  // Random values: the result is the nearest representable code (brute force over the 126 positive codes).
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> u(-600.0, 600.0);
  std::normal_distribution<double> g(0.0, 0.01);
  bool nearest_ok = true;
  for (int n = 0; n < 20000 && nearest_ok; ++n) {
    const float x = static_cast<float>(n % 2 ? u(rng) : g(rng));
    double best = 1e300;
    for (int c = 0; c < 0x7F; ++c) best = std::min(best, std::fabs(RefDecode(c) - std::fabs(static_cast<double>(x))));
    const double got = std::fabs(std::fabs(static_cast<double>(x)) - std::fabs(RefDecode(mx::EncodeE4m3(x) & 0x7F)));
    nearest_ok = std::fabs(static_cast<double>(x)) >= 448.0 ? true : got == best;
  }
  Check(nearest_ok, "20000 random values encode to a nearest code");
}

void TestScale() {
  std::printf("---- E8M0 scale rule ----\n");
  Check(mx::ScaleExponent(0.0f) == -127 && mx::ScaleExponent(-1.0f) == -127 && mx::ScaleExponent(NAN) == -127,
        "amax 0 (and negative / NaN) -> -127");
  Check(mx::ScaleExponent(448.0f) == 0 && mx::ScaleExponent(std::nextafterf(448.0f, 1e9f)) == 1 &&
            mx::ScaleExponent(std::nextafterf(448.0f, 0.0f)) == 0,
        "amax 448 -> 0, one ulp above -> 1, one ulp below -> 0");
  Check(mx::ScaleExponent(224.0f) == -1 && mx::ScaleExponent(std::nextafterf(224.0f, 1e9f)) == 0 && mx::ScaleExponent(1.0f) == -8 &&
            mx::ScaleExponent(448.0f * 256.0f) == 8,
        "amax 224 -> -1, one ulp above -> 0, amax 1 -> -8 (448 * 2^-8 = 1.75 >= 1 > 0.875), 448 * 2^8 -> 8");
  Check(mx::ScaleExponent(1.0e-38f) == -127 && mx::ScaleExponent(FLT_MIN) == -127 && mx::ScaleExponent(FLT_TRUE_MIN) == -127,
        "tiny amax (1e-38, FLT_MIN, the smallest subnormal) clamps to -127");
  Check(mx::ScaleExponent(static_cast<float>(448.0 * std::ldexp(1.0, -127))) == -127,
        "amax 448 * 2^-127 -> exactly -127");
  Check(mx::ScaleExponent(FLT_MAX) == 120, "amax FLT_MAX -> 120 (448 * 2^120 = 5.95e38 >= 3.4e38 > 448 * 2^119)");
  bool inv_ok = true, bytes_ok = true;
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> ue(-40.0, 40.0);
  for (int n = 0; n < 20000; ++n) {
    const float amax = static_cast<float>(std::exp2(ue(rng)));
    if (!(amax > 0.0f) || std::isinf(amax)) continue;
    const int e = mx::ScaleExponent(amax);
    inv_ok = inv_ok && std::ldexp(448.0, e) >= amax && (e == -127 || std::ldexp(448.0, e - 1) < amax);
  }
  for (int e = -127; e <= 127; ++e)
    bytes_ok = bytes_ok && mx::ScaleByte(e) == e + 127 && mx::ScaleValue(mx::ScaleByte(e)) == std::ldexp(1.0f, e);
  Check(inv_ok, "20000 random amax: 448 * 2^e >= amax and e is minimal (or clamped at -127)");
  Check(bytes_ok, "ScaleByte / ScaleValue round-trip every exponent -127..127 (byte 0 = 2^-127 subnormal, 254 = 2^127)");
}

void TestEncodeGroup() {
  std::printf("---- EncodeGroup ----\n");
  uint8_t codes[32];
  std::vector<float> zeros(32, 0.0f);
  uint8_t sb = mx::EncodeGroup(zeros.data(), 32, codes);
  bool all_zero = true;
  for (uint8_t c : codes) all_zero = all_zero && c == 0;
  Check(sb == 0 && all_zero, "an all-zero group -> scale byte 0 (2^-127), codes 0");

  // Values that are exactly code * 2^e decode back exactly.
  std::vector<float> v(32);
  for (int i = 0; i < 32; ++i) v[static_cast<size_t>(i)] = static_cast<float>(RefDecode(0x30 + (i % 16)) * std::ldexp(1.0, -5)) * (i % 3 == 0 ? -1.0f : 1.0f);
  sb = mx::EncodeGroup(v.data(), 32, codes);
  bool exact = true;
  for (int i = 0; i < 32; ++i) exact = exact && mx::DecodeE4m3(codes[i]) * mx::ScaleValue(sb) == v[static_cast<size_t>(i)];
  Check(exact, "values that are exact e4m3 * 2^e re-encode exactly");

  // Gaussian groups: no element clipped (amax lands in (224, 448] codes), relative error <= 2^-4 + subnormal floor.
  std::mt19937_64 rng(21);
  std::normal_distribution<double> nd(0.0, 0.3);
  bool clip_ok = true, err_ok = true;
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<float> g(32);
    const double mag = std::exp2(static_cast<int>(trial % 30) - 20);
    for (auto& x : g) x = static_cast<float>(nd(rng) * mag);
    sb = mx::EncodeGroup(g.data(), 32, codes);
    const double scale = mx::ScaleValue(sb);
    float amax = 0, dmax = 0;
    for (int i = 0; i < 32; ++i) {
      amax = std::max(amax, std::fabs(g[static_cast<size_t>(i)]));
      const double d = mx::DecodeE4m3(codes[i]) * scale;
      dmax = std::max(dmax, std::fabs(static_cast<float>(d)));
      const double bound = std::fabs(g[static_cast<size_t>(i)]) * 0.0625 + scale * std::ldexp(1.0, -10) * 1.0000001;
      err_ok = err_ok && std::fabs(d - g[static_cast<size_t>(i)]) <= bound;
    }
    clip_ok = clip_ok && amax <= 448.0 * scale && (sb == 0 || dmax >= 0.5 * 448.0 * scale * 0.5);
  }
  Check(clip_ok, "2000 Gaussian groups: amax <= 448 * scale (nothing clipped), largest decoded code >= 112 * scale");
  Check(err_ok, "2000 Gaussian groups: |decoded - v| <= 2^-4 |v| + 2^-10 * scale");
  // A short group (n < 32): the tail is not touched.
  uint8_t tail[32];
  std::memset(tail, 0xA5, sizeof tail);
  std::vector<float> few = {1.0f, -2.0f, 3.0f};
  mx::EncodeGroup(few.data(), 3, tail);
  Check(tail[3] == 0xA5 && tail[31] == 0xA5, "EncodeGroup(n = 3) writes exactly 3 codes");
}

// Index of the byte holding code(row, k): derived by hand from mxfp8_format.hpp / quant_int8.hpp's layout text
// (16 rows x 32 K per 512-byte block; byte i of dword j of lane l: row 16t + (l & 15), k = 32kb + 16 (j >> 1) + 8 (j & 1) + 4 (l >> 4) + i).
size_t WqIndex(int row, int k, int K) {
  const int t = row / 16, r = row % 16, kb = k / 32, kk = k % 32;
  const int j = (kk / 16) * 2 + (kk % 16) / 8;
  const int l = r + 16 * ((kk % 8) / 4);
  const int i = kk % 4;
  return ((static_cast<size_t>(t) * (K / 32) + kb) * 32 + l) * 16 + j * 4 + i;
}

void TestPackers() {
  std::printf("---- packers ----\n");
  const int N = 48, K = 96;
  std::mt19937_64 rng(77);
  std::normal_distribution<double> nd(0.0, 0.2);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& x : w) x = static_cast<float>(nd(rng));
  w[5] = 0.0f;
  for (int k = 0; k < 32; ++k) w[static_cast<size_t>(7) * K + 32 + k] = 0.0f;  // an all-zero group inside a row
  std::vector<uint8_t> codes, scales;
  r4dx_convert::QuantizeMxfp8(w.data(), N, K, 3, codes, scales);
  Check(codes.size() == static_cast<size_t>(N) * K && scales.size() == static_cast<size_t>(N) * (K / 32), "QuantizeMxfp8: sizes");
  bool same = true;
  for (int r = 0; r < N && same; ++r)
    for (int g = 0; g < K / 32; ++g) {
      uint8_t c[32];
      const uint8_t sb = mx::EncodeGroup(w.data() + static_cast<size_t>(r) * K + g * 32, 32, c);
      same = same && sb == scales[static_cast<size_t>(r) * (K / 32) + g] &&
             std::memcmp(c, codes.data() + static_cast<size_t>(r) * K + g * 32, 32) == 0;
    }
  Check(same, "QuantizeMxfp8 == EncodeGroup on every (row, group) (any thread count)");
  std::vector<uint8_t> codes1, scales1;
  r4dx_convert::QuantizeMxfp8(w.data(), N, K, 1, codes1, scales1);
  Check(codes1 == codes && scales1 == scales, "QuantizeMxfp8 is thread-count independent");

  const std::vector<uint8_t> wq = r4dx_convert::PackMxfp8Codes(codes, N, K, 2);
  const std::vector<uint8_t> ws = r4dx_convert::PackMxfp8Scales(scales, N, K);
  bool wq_ok = wq.size() == codes.size();
  for (int r = 0; r < N && wq_ok; ++r)
    for (int k = 0; k < K; ++k) wq_ok = wq_ok && wq[WqIndex(r, k, K)] == codes[static_cast<size_t>(r) * K + k];
  Check(wq_ok, "wq: every code (row, k) is at the byte the documented fragment layout says");
  bool ws_ok = ws.size() == scales.size();
  const int gpr = K / 32;
  for (int t = 0; t < N / 16 && ws_ok; ++t)
    for (int g = 0; g < gpr; ++g)
      for (int r = 0; r < 16; ++r) ws_ok = ws_ok && ws[(static_cast<size_t>(t) * gpr + g) * 16 + r] == scales[static_cast<size_t>(16 * t + r) * gpr + g];
  Check(ws_ok, "ws: one byte per (row, group) at (tile * groups + group) * 16 + row-in-tile (w8a16's ws element order)");
  Check(wq == r4dx_convert::PackW8Bytes(codes, N, K, 1), "wq is exactly w8a16's PackW8Bytes of the codes (the permutation is reused)");

  const std::vector<float> back = r4dx_convert::DequantMxfp8(wq, ws, N, K);
  bool deq_ok = back.size() == w.size();
  for (int r = 0; r < N && deq_ok; ++r)
    for (int k = 0; k < K; ++k)
      deq_ok = deq_ok && back[static_cast<size_t>(r) * K + k] ==
                             mx::DecodeE4m3(codes[static_cast<size_t>(r) * K + k]) * mx::ScaleValue(scales[static_cast<size_t>(r) * gpr + k / 32]);
  Check(deq_ok, "DequantMxfp8(pack(q)) == decode(q): the lane-walking reference inverts both packers");

  Check(Throws([&] { r4dx_convert::PackMxfp8Scales(scales, 40, K); }, "N"), "PackMxfp8Scales refuses N % 16 != 0");
  Check(Throws([&] { std::vector<uint8_t> c, s; r4dx_convert::QuantizeMxfp8(w.data(), N, 90, 1, c, s); }, "K"),
        "QuantizeMxfp8 refuses K % 32 != 0");
  std::vector<float> bad = w;
  bad[3] = INFINITY;
  Check(Throws([&] { std::vector<uint8_t> c, s; r4dx_convert::QuantizeMxfp8(bad.data(), N, K, 2, c, s, "t"); }, "non-finite"),
        "QuantizeMxfp8 refuses a non-finite weight");
  std::vector<uint8_t> nanw = wq;
  nanw[100] = 0x7F;
  Check(Throws([&] { r4dx_convert::DequantMxfp8(nanw, ws, N, K); }, "NaN"), "DequantMxfp8 refuses a NaN code");
}

void TestRawCopyRule() {
  std::printf("---- checkpoint MXFP8 groups: re-encode vs raw copy ----\n");
  // Random valid groups (arbitrary scale bytes, codes with the NaN codes excluded). encode(decode(x)) must equal x iff
  // the group's largest code magnitude exceeds 224 (then the ceil rule picks the stored exponent again); otherwise the
  // converter's raw copy is what keeps the group bit-exact.
  std::mt19937_64 rng(99);
  int equal = 0, differ = 0;
  bool rule_ok = true;
  for (int trial = 0; trial < 20000; ++trial) {
    uint8_t x[32];
    for (auto& c : x) {
      c = static_cast<uint8_t>(rng());
      if (IsNan(c)) c &= 0x80;
    }
    if (trial % 4 == 0)  // force a large max code in a quarter of the groups
      x[rng() % 32] = static_cast<uint8_t>(0x77 + rng() % 8) | static_cast<uint8_t>(rng() & 0x80);
    const uint8_t sbyte = static_cast<uint8_t>(1 + rng() % 200);  // a scale that keeps decode finite
    float d[32];
    double mag = 0.0;
    for (int i = 0; i < 32; ++i) {
      d[i] = mx::DecodeE4m3(x[i]) * mx::ScaleValue(sbyte);
      mag = std::max(mag, std::fabs(RefDecode(x[i])));
    }
    uint8_t y[32];
    const uint8_t sb2 = mx::EncodeGroup(d, 32, y);
    const bool same = sb2 == sbyte && std::memcmp(x, y, 32) == 0;
    equal += same;
    differ += !same;
    // A sign-of-zero difference (x holds -0, y holds the zero of the re-encoded sign) is covered by memcmp too.
    if (mag > 224.0) rule_ok = rule_ok && same;
    else if (mag > 0.0) rule_ok = rule_ok && !same;  // max code <= 224: the ceil rule picks a smaller exponent
  }
  Check(rule_ok, "re-encode reproduces a group iff its largest code magnitude is > 224 (" + std::to_string(equal) + " equal, " +
                     std::to_string(differ) + " different of 20000)");
  Check(differ > 0 && equal > 0, "both sides of the rule occur, so the raw copy is load-bearing: the converter copies checkpoint MXFP8 groups raw");
}

void TestLayoutPlumbing() {
  std::printf("---- LayoutSet / plan / ids ----\n");
  LayoutSet ls;
  ls.bf16 = false;
  ls.mxfp8 = true;
  Check(r4dx_convert::LayoutSetId(ls) == "mxfp8.g32", "LayoutSetId token is mxfp8.g32");
  LayoutSet back;
  Check(r4dx_convert::ParseLayoutSetId("mxfp8.g32", &back) && back.mxfp8 && !back.bf16 && !back.w8a16 && !back.w4a16,
        "ParseLayoutSetId(mxfp8.g32)");
  Check(r4dx_convert::ParseLayoutSetId("bf16+mxfp8.g32", &back) && back.mxfp8 && back.bf16, "ParseLayoutSetId(bf16+mxfp8.g32)");
  Check(!r4dx_convert::ParseLayoutSetId("mxfp8.g32+bf16", &back) && !r4dx_convert::ParseLayoutSetId("mxfp8.g64", &back) &&
            !r4dx_convert::ParseLayoutSetId("mxfp8.g32+mxfp8.g32", &back) && !r4dx_convert::ParseLayoutSetId("trellis.k4+mxfp8.g32", &back),
        "ParseLayoutSetId is canonical-only (order, group, duplicates, trellis exclusive)");
  const auto names = r4dx_convert::LinearLayoutTensorNames("a.b", ls);
  Check(names == std::vector<std::string>({"a.b.mxfp8.wq", "a.b.mxfp8.ws"}), "tensor names <base>.mxfp8.wq / .ws");
  Check(r4dx_convert::LinearLayoutBytes(32, 64, ls) == 32u * 64 + 32u * 64 / 32, "LinearLayoutBytes = N*K + N*K/32");
  r4dx_convert::ContainerWriter w;
  r4dx_convert::PlanLinearLayouts(w, "a.b", 32, 64, ls);
  Check(w.PlannedTensorCount() == 2 && w.PlannedName(0) == "a.b.mxfp8.wq" && w.PlannedBytes(0) == 2048 &&
            w.PlannedShape(0) == std::vector<int64_t>({2048}) && w.PlannedName(1) == "a.b.mxfp8.ws" && w.PlannedBytes(1) == 64 &&
            w.PlannedShape(1) == std::vector<int64_t>({64}),
        "PlanLinearLayouts: wq u8 {N*K}, ws u8 {N*K/32}");
  r4dx_convert::ContainerWriter w2;
  Check(Throws([&] { r4dx_convert::PlanLinearLayouts(w2, "x", 40, 64, ls); }, "N") && Throws([&] { r4dx_convert::PlanLinearLayouts(w2, "x", 32, 48, ls); }, "K"),
        "PlanLinearLayouts refuses N % 16 and K % 32");
  Check(mx::kWqSuffix == std::string(".mxfp8.wq") && mx::kWsSuffix == std::string(".mxfp8.ws") && mx::kMetaBlockKey == std::string("mxfp8"),
        "format constants");
}

}  // namespace

int main() {
  TestE4m3Codec();
  TestRounding();
  TestScale();
  TestEncodeGroup();
  TestPackers();
  TestRawCopyRule();
  TestLayoutPlumbing();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all mxfp8 format checks passed\n");
  return 0;
}
