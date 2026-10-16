// tests/kernels/test_w8a16_ref_cpu.cpp -- pure CPU test of w8a16_ref.hpp (docs/w8a16.md 2-4; no HIP).
//   * conversions: f16 RNE / RTZ hand values (saturation, subnormals), bf16 RNE, the kernel's magic-exponent
//     unpack (0x64bb - 1152 == b - 128 for every byte);
//   * quantizer: a hand group (scale 1, half-away rounding, clamp), a dead group, amax overflow throws;
//   * layout: hand byte / scale positions of the 3.2 map, and pack -> unpack round trips;
//   * GEMM: hand cases with exact answers (incl. the RTZ saturation 1e5 -> 65504 and an f16 subnormal), the
//     reference against the fp64 naive GEMM under the design's bound at G 64 / 128 and every legal SK
//     (including uneven group counts: K = 384 at SK = 3, K = 1536 = 12 groups, K = 11776 = 92 groups), and
//     SK-independence when every partial sum is exact.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "w8a16_ref.hpp"

using namespace w8a16_ref;

namespace {

int g_pass = 0, g_fail = 0;
void Check(bool c, const std::string& what) {
  std::printf("%-96s %s\n", what.c_str(), c ? "ok" : "FAIL");
  (c ? g_pass : g_fail)++;
}

std::vector<float> Gauss(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}
std::vector<uint16_t> Bf16Vec(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = FloatToBf16Rne(v[i]);
  return o;
}

void TestConversions() {
  Check(FloatToF16Rne(1.0f) == 0x3c00 && FloatToF16Rtz(1.0f) == 0x3c00, "f16: 1.0 -> 0x3c00");
  Check(FloatToF16Rne(-2.0f) == 0xc000, "f16: -2.0 -> 0xc000");
  Check(FloatToF16Rne(65504.0f) == 0x7bff && FloatToF16Rtz(65504.0f) == 0x7bff, "f16: 65504 -> 0x7bff");
  Check(FloatToF16Rne(65520.0f) == 0x7c00, "f16 RNE: 65520 -> inf");
  Check(FloatToF16Rtz(65520.0f) == 0x7bff && FloatToF16Rtz(1e5f) == 0x7bff && FloatToF16Rtz(-1e5f) == 0xfbff,
        "f16 RTZ: 65520, +-1e5 saturate to +-65504");
  Check(FloatToF16Rne(1.0f + 1.0f / 2048) == 0x3c00 && FloatToF16Rne(1.0f + 3.0f / 2048) == 0x3c02,
        "f16 RNE: ties to even");
  Check(FloatToF16Rtz(1.0f + 1.9f / 1024) == 0x3c01, "f16 RTZ: 1+1.9ulp -> 1+1ulp");
  Check(FloatToF16Rne(std::ldexp(1.0f, -24)) == 0x0001 && FloatToF16Rne(std::ldexp(1.0f, -14)) == 0x0400,
        "f16: 2^-24 -> 0x0001, 2^-14 -> 0x0400");
  Check(FloatToF16Rtz(1e-9f) == 0x0000 && FloatToF16Rtz(-1e-9f) == 0x8000, "f16 RTZ: +-1e-9 -> +-0");
  Check(FloatToF16Rtz(1e-6f) == 0x0010, "f16 RTZ: 1e-6 -> 16 * 2^-24 (subnormal)");
  Check(FloatToF16Rne(4096.0f) == 0x6c00, "f16: 4096 -> 0x6c00");
  bool rt = true;
  for (uint32_t h = 0; h < 0x7c00; ++h) {
    rt &= FloatToF16Rne(F16ToFloat(static_cast<uint16_t>(h))) == h;
    rt &= FloatToF16Rtz(F16ToFloat(static_cast<uint16_t>(h))) == h;
  }
  Check(rt, "f16: every finite positive f16 round-trips through float (RNE and RTZ)");
  bool bfexact = true;
  for (uint32_t b = 0x3880; b <= 0x477f; ++b) {  // bf16 in [2^-14, 65280]: f16-exact
    const float f = Bf16ToFloat(static_cast<uint16_t>(b));
    bfexact &= F16ToFloat(FloatToF16Rtz(f)) == f;
  }
  Check(bfexact, "bf16 -> f16 RTZ exact for every bf16 in [2^-14, 65280]");
  Check(FloatToBf16Rne(1.0f + 1.0f / 256) == 0x3f80 && FloatToBf16Rne(1.0f + 3.0f / 256) == 0x3f82,
        "bf16 RNE: ties to even");
  Check(FloatToBf16Rne(65504.0f) == 0x4780, "bf16 RNE: 65504 -> 65536");
  bool magic = true;
  for (int b = 0; b < 256; ++b) {
    const float v = F16ToFloat(static_cast<uint16_t>(0x6400 | b));
    magic &= v == 1024.0f + b;
    // v_pk_add_f16(v, -1152): the f16 sum is exact (|b - 128| <= 128), so float arithmetic models it
    magic &= F16ToFloat(FloatToF16Rne(v + F16ToFloat(0xe480))) == static_cast<float>(b - 128);
  }
  Check(magic && F16ToFloat(0xe480) == -1152.0f, "unpack: f16 0x64bb == 1024 + b, + (-1152 = 0xe480) == b - 128");
}

void TestQuantizer() {
  const int N = 16, K = 128, G = 128;
  std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
  // row 0: amax 127 -> s = 1, s16 = f16(4096) = 0x6c00
  w[0] = 127.0f;
  w[1] = 2.5f;
  w[2] = -2.5f;
  w[3] = 0.49f;
  w[4] = -126.5f;
  // row 1: dead (amax 1e-25 -> biased scale flushes to 0)
  for (int k = 0; k < K; ++k) w[K + k] = (k & 1 ? -1e-25f : 1e-25f);
  // row 2: constant group
  for (int k = 0; k < K; ++k) w[2 * K + k] = 0.75f;
  Quantized q = Quantize(w, N, K, G);
  Check(q.s16[0] == 0x6c00, "quant: amax 127 -> stored scale f16(4096) = 0x6c00");
  Check(q.q[0] == 127 && q.q[1] == 3 && q.q[2] == -3 && q.q[3] == 0 && q.q[4] == -127,
        "quant: codes 127, 2.5->3, -2.5->-3, 0.49->0, -126.5->-127 (half away from zero)");
  Check(q.s16[1] == 0 && q.zero_groups >= 1, "quant: amax 1e-25 is a zero group (scale 0)");
  bool dead0 = true;
  for (int k = 0; k < K; ++k) dead0 &= q.q[K + k] == 0;
  Check(dead0, "quant: zero group codes are 0 (byte 128)");
  bool cst = true;
  for (int k = 0; k < K; ++k) cst &= q.q[2 * K + k] == 127;
  Check(cst, "quant: constant group -> all codes 127");
  // all-zero rows 3..15 are zero groups too
  Check(q.zero_groups == 14, "quant: 14 zero groups (row 1 + 13 all-zero rows)");
  // overflow: amax 2100 > 2031 -> s*4096 > 65504
  std::vector<float> big(static_cast<size_t>(N) * K, 0.0f);
  big[5] = 2100.0f;
  bool threw = false;
  try {
    (void)Quantize(big, N, K, G);
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw, "quant: amax 2100 (biased scale overflows f16) throws");
  // the stored-scale quantization: codes are against float(s16) * 2^-12, |q| <= 127 always
  std::mt19937 rng(7);
  auto wr = Gauss(rng, static_cast<size_t>(64) * 256, 0.02f);
  Quantized qr = Quantize(wr, 64, 256, 64);
  bool inrange = true;
  for (int8_t c : qr.q) inrange &= c >= -127 && c <= 127;
  Check(inrange, "quant: random codes within -127..127");
  // relative Frobenius error of RTN g64 on Gaussian data (docs/w8a16.md 7.1 bound 0.012)
  double num = 0, den = 0;
  for (int n = 0; n < 64; ++n)
    for (int k = 0; k < 256; ++k) {
      const double d = qr.q[n * 256 + k] * F16ToFloat(qr.s16[n * 4 + k / 64]) / 4096.0;
      num += (d - wr[n * 256 + k]) * (d - wr[n * 256 + k]);
      den += static_cast<double>(wr[n * 256 + k]) * wr[n * 256 + k];
    }
  Check(std::sqrt(num / den) < 0.012, "quant: rel. Frobenius error < 0.012 (g64, Gaussian): " +
                                          std::to_string(std::sqrt(num / den)));
}

void TestLayout() {
  // hand positions: (row 17, k 37) at K = 64: t 1, r 1, kb 1, kk 5 -> j 0, lane 17, byte 1
  //   ((((1*2 + 1)*32 + 17)*4 + 0)*4) + 1 = 1809
  {
    const int N = 32, K = 64;
    std::vector<int8_t> q(static_cast<size_t>(N) * K, 0);
    q[17 * K + 37] = 77;
    auto wq = PackWq(q, N, K);
    Check(wq[1809] == 77 + 128, "layout: code(17, 37) at K=64 is byte 1809");
    int nonzero = 0;
    for (uint8_t b : wq) nonzero += b != 128;
    Check(nonzero == 1, "layout: exactly one non-128 byte");
    // (row 3, k 30): t 0, r 3, kb 0, kk 30 -> j = 2*1 + 1 = 3, lane = 3 + 16*((30%8)/4 = 1) = 19, i = 2
    //   (((0*32 + 19)*4 + 3)*4) + 2 = 318
    std::vector<int8_t> q2(static_cast<size_t>(N) * K, 0);
    q2[3 * K + 30] = -5;
    Check(PackWq(q2, N, K)[318] == 123, "layout: code(3, 30) at K=64 is byte 318");
  }
  {
    // scale (row 17, group 1) at K = 256, G = 128: ((1*2 + 1)*16 + 1) = 49
    const int N = 32, K = 256, G = 128;
    std::vector<uint16_t> s(static_cast<size_t>(N) * 2, 0);
    s[17 * 2 + 1] = 0x1234;
    Check(PackWs(s, N, K, G)[49] == 0x1234, "layout: scale(17, g1) at K=256 G=128 is u16 49");
  }
  std::mt19937 rng(11);
  bool rt = true;
  const int shapes[][3] = {{16, 64, 64}, {48, 128, 64}, {32, 384, 128}, {64, 1536, 128}, {16, 11776, 128}};
  for (const auto& sh : shapes) {
    const int N = sh[0], K = sh[1], G = sh[2];
    std::uniform_int_distribution<int> d(-127, 127);
    std::vector<int8_t> q(static_cast<size_t>(N) * K);
    for (auto& c : q) c = static_cast<int8_t>(d(rng));
    std::vector<uint16_t> s(static_cast<size_t>(N) * (K / G));
    for (auto& x : s) x = static_cast<uint16_t>(rng() & 0x7bff);
    const Unpacked u = UnpackRows(PackWq(q, N, K), PackWs(s, N, K, G), N, K, G);
    for (size_t i = 0; i < q.size(); ++i) rt &= u.q[i] == q[i];
    rt &= u.s16 == s;
  }
  Check(rt, "layout: pack -> kernel-order unpack round trips (5 shapes, K 64..11776)");
}

// Build a packed weight directly from codes and stored scale bits.
Unpacked FromCodes(const std::vector<int8_t>& q, const std::vector<uint16_t>& s16, int N, int K, int G) {
  return UnpackRows(PackWq(q, N, K), PackWs(s16, N, K, G), N, K, G);
}

void TestGemmHand() {
  {
    // A = 1, code(n, k) = n - 8, scale 1 (s16 = 4096): out[n] = 128 * (n - 8), exact
    const int M = 1, N = 16, K = 128, G = 128;
    std::vector<int8_t> q(static_cast<size_t>(N) * K);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) q[n * K + k] = static_cast<int8_t>(n - 8);
    std::vector<uint16_t> s(N, 0x6c00);
    std::vector<uint16_t> a(K, FloatToBf16Rne(1.0f));
    const auto out = Gemm(a, FromCodes(q, s, N, K, G), M, K, G, 1);
    bool ok = true;
    for (int n = 0; n < N; ++n) ok &= Bf16ToFloat(out[n]) == 128.0f * (n - 8);
    Check(ok, "gemm hand: A=1, q=n-8, scale 1 -> 128*(n-8)");
  }
  {
    // saturation and subnormals: column n picks A[n] (q = 1 at k = n), scale 1
    const int M = 1, N = 16, K = 64, G = 64;
    std::vector<int8_t> q(static_cast<size_t>(N) * K, 0);
    for (int n = 0; n < N; ++n) q[n * K + n] = 1;
    std::vector<uint16_t> s(N, 0x6c00);
    std::vector<uint16_t> a(K, 0);
    a[0] = FloatToBf16Rne(1e5f);
    a[1] = FloatToBf16Rne(-1e5f);
    a[2] = FloatToBf16Rne(1e-6f);
    a[3] = FloatToBf16Rne(1e-9f);
    a[4] = FloatToBf16Rne(3.0e38f);  // huge finite bf16 -> 65504
    a[5] = FloatToBf16Rne(65504.0f);  // bf16 65536 -> RTZ 65504
    a[6] = FloatToBf16Rne(-0.0f);
    const auto out = Gemm(a, FromCodes(q, s, N, K, G), M, K, G, 1);
    Check(out[0] == FloatToBf16Rne(65504.0f) && Bf16ToFloat(out[0]) == 65536.0f,
          "gemm hand: A=1e5 -> f16 65504 (RTZ saturation) -> bf16 65536, finite");
    Check(out[1] == FloatToBf16Rne(-65504.0f), "gemm hand: A=-1e5 -> -65504");
    const float sub = F16ToFloat(FloatToF16Rtz(Bf16ToFloat(FloatToBf16Rne(1e-6f))));
    Check(sub == 16.0f * std::ldexp(1.0f, -24) && out[2] == FloatToBf16Rne(sub),
          "gemm hand: A=1e-6 -> f16 subnormal 16*2^-24, exact through the GEMM");
    Check(Bf16ToFloat(out[3]) == 0.0f, "gemm hand: A=1e-9 -> f16 0");
    Check(out[4] == FloatToBf16Rne(65504.0f) && out[5] == FloatToBf16Rne(65504.0f),
          "gemm hand: A=3e38 and A=bf16(65504)=65536 saturate to 65504");
    Check(Bf16ToFloat(out[6]) == 0.0f, "gemm hand: A=-0 -> 0");
  }
  {
    // the bias is undone exactly: s16 = 1 (2^-24 biased -> 2^-36 real), A = 2^10, q = 64 over one k
    const int M = 1, N = 16, K = 64, G = 64;
    std::vector<int8_t> q(static_cast<size_t>(N) * K, 0);
    q[0] = 64;
    std::vector<uint16_t> s(N, 0x0001);
    std::vector<uint16_t> a(K, FloatToBf16Rne(1024.0f));
    const auto out = Gemm(a, FromCodes(q, s, N, K, G), M, K, G, 1);
    Check(Bf16ToFloat(out[0]) == std::ldexp(1.0f, 16 - 36), "gemm hand: subnormal scale 2^-24 (biased) -> 2^-20 exact");
  }
  {
    // zero-scale group contributes 0 even with nonzero codes
    const int M = 2, N = 16, K = 256, G = 128;
    std::vector<int8_t> q(static_cast<size_t>(N) * K, 5);
    std::vector<uint16_t> s(static_cast<size_t>(N) * 2, 0x6c00);
    for (int n = 0; n < N; ++n) s[n * 2 + 1] = 0;
    std::vector<uint16_t> a(static_cast<size_t>(M) * K, FloatToBf16Rne(1.0f));
    const auto out = Gemm(a, FromCodes(q, s, N, K, G), M, K, G, 2);
    Check(Bf16ToFloat(out[0]) == 640.0f && Bf16ToFloat(out[N + 15]) == 640.0f,
          "gemm hand: a zero scale group contributes 0 (SK 2)");
  }
}

void TestGemmRandom() {
  std::mt19937 rng(1234);
  struct Case {
    int N, K, G;
  };
  const Case cases[] = {{32, 128, 64},  {32, 128, 128}, {48, 384, 128}, {32, 384, 64}, {64, 1536, 128},
                        {32, 2048, 64}, {16, 11776, 128}, {32, 768, 128}};
  for (const auto& c : cases) {
    const int M = 17;
    // weights: Gaussian, a few rows scaled tiny (scales near the bottom of the biased range) and one dead row
    auto w = Gauss(rng, static_cast<size_t>(c.N) * c.K, 0.02f);
    for (int k = 0; k < c.K; ++k) {
      w[static_cast<size_t>(1) * c.K + k] *= 1e-3f;
      w[static_cast<size_t>(2) * c.K + k] = 1e-25f;
    }
    const Quantized qz = Quantize(w, c.N, c.K, c.G);
    const Unpacked u = FromCodes(qz.q, qz.s16, c.N, c.K, c.G);
    // activations: normal, with a row of large values (up to 6e4) and a row of tiny ones
    auto af = Gauss(rng, static_cast<size_t>(M) * c.K, 1.0f);
    for (int k = 0; k < c.K; ++k) {
      af[static_cast<size_t>(3) * c.K + k] *= 2e4f;
      af[static_cast<size_t>(4) * c.K + k] *= 1e-5f;
    }
    af[static_cast<size_t>(3) * c.K] = 6e4f;
    const auto a = Bf16Vec(af);
    const NaiveOut nv = GemmNaive(a, u, M, c.K, c.G);
    const int groups = c.K / c.G;
    for (int SK = 1; SK <= 8; ++SK) {
      if (groups % SK) continue;
      const auto out = Gemm(a, u, M, c.K, c.G, SK);
      size_t bad = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < c.N; ++n) {
          const size_t i = static_cast<size_t>(m) * c.N + n;
          bad += !WithinBound(Bf16ToFloat(out[i]), nv.c[i], nv.a_norm[m], nv.w_norm[n]);
        }
      Check(bad == 0, "gemm vs fp64 naive within 2 ulp + 1e-6|a||w|: N " + std::to_string(c.N) + " K " +
                          std::to_string(c.K) + " G " + std::to_string(c.G) + " SK " + std::to_string(SK) +
                          " (groups " + std::to_string(groups) + ")");
    }
    // a row subset of the packed weight (UnpackRows) gives the same values as the full computation
    const std::vector<int> cols = {0, 5, c.N - 1};
    const auto full = Gemm(a, u, M, c.K, c.G, 1);
    const Unpacked us = UnpackRows(PackWq(qz.q, c.N, c.K), PackWs(qz.s16, c.N, c.K, c.G), c.N, c.K, c.G, cols);
    const auto sub = Gemm(a, us, M, c.K, c.G, 1);
    bool same = true;
    for (int m = 0; m < M; ++m)
      for (size_t ci = 0; ci < cols.size(); ++ci)
        same &= full[static_cast<size_t>(m) * c.N + cols[ci]] == sub[m * cols.size() + ci];
    Check(same, "gemm: UnpackRows subset == full (N " + std::to_string(c.N) + " K " + std::to_string(c.K) + ")");
  }
}

void TestGemmExactSkIndependent() {
  // small integer A and codes, scale 1: every partial sum is an exact integer, so all SK agree bit for bit and
  // equal the naive result exactly
  std::mt19937 rng(99);
  const int M = 8, N = 32, K = 1536, G = 128;
  std::uniform_int_distribution<int> da(-3, 3), dq(-127, 127);
  std::vector<int8_t> q(static_cast<size_t>(N) * K);
  for (auto& c : q) c = static_cast<int8_t>(dq(rng));
  std::vector<uint16_t> s(static_cast<size_t>(N) * (K / G), 0x6c00);
  std::vector<uint16_t> a(static_cast<size_t>(M) * K);
  for (auto& x : a) x = FloatToBf16Rne(static_cast<float>(da(rng)));
  const Unpacked u = FromCodes(q, s, N, K, G);
  const auto r1 = Gemm(a, u, M, K, G, 1);
  bool same = true;
  for (int SK : {2, 3, 4, 6}) same &= Gemm(a, u, M, K, G, SK) == r1;
  const NaiveOut nv = GemmNaive(a, u, M, K, G);
  bool exact = true;
  for (size_t i = 0; i < r1.size(); ++i) exact &= Bf16ToFloat(r1[i]) == Bf16Round(static_cast<float>(nv.c[i]));
  Check(same, "gemm: integer data -> SK 1/2/3/4/6 bit-identical (K 1536, 12 groups)");
  Check(exact, "gemm: integer data -> equals bf16(naive) exactly");
  bool threw = false;
  try {
    (void)Gemm(a, u, M, K, G, 5);  // 12 groups % 5
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw, "gemm: K % (SK*G) != 0 throws");
}

}  // namespace

int main() {
  TestConversions();
  TestQuantizer();
  TestLayout();
  TestGemmHand();
  TestGemmRandom();
  TestGemmExactSkIndependent();
  std::printf("test_w8a16_ref_cpu: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
