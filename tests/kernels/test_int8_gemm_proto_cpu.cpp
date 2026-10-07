// tests/kernels/test_int8_gemm_proto_cpu.cpp -- pure CPU (no HIP call, no device). The CPU side of the int8
// prefill GEMM prototype (docs/int8-gemm-proto.md, tests/kernels/int8_gemm_proto_ref.h):
//   * the fragment position map and the A8 / W8 layouts are bijections and agree with the trellis decode's own
//     lane map (k = 8 (e >> 2) + 4 h + (e & 3), the column of element e of fragment F on lane L);
//   * the software f16 conversions (RNE, from double) round-trip every f16 and match the library's;
//   * the weight quantizer: v_pk_fma_f16(w, rs, 1536) + low byte (QuantWTrick) is exactly rint(w * rs)
//     (QuantW), ties included, over all f16 w and a spread of rs; the table's s_eff = 1 / rs is idempotent
//     (RsHalf(s_eff) == rs for every rs the table can hold); how far it is from the fp32 rule of
//     R4DX_FAKEQ_W (a few LSB flips, the same error RMS);
//   * a software-WMMA emulation of the kernel's whole fragment-level chain (A8 / W8 loads, the 16 x 16 x 16
//     iu8 tile, the per-128-K rescale into fp32, the accumulator row / column map) on a small shape, against
//     the exact-integer reference RefGemmInt8, fine and coarse scales;
//   * the trellis weights end to end on the CPU: random pair-grid words (KB 4 and 5), trellis_ref's decode,
//     the scale table and the int8 matrix, every |q| <= 127 and every group's amax hits +-127.
// The device bench's --selftest repeats the last three against the GPU kernels.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "int8_gemm_proto_ref.h"
#include "trellis_ref.hpp"

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-96s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

void TestMaps() {
  bool inv = true, same = true;
  for (int lane = 0; lane < 32; ++lane)
    for (int e = 0; e < 8; ++e) {
      int h, ee;
      i8p::K16ToFrag(i8p::FragK16(lane, e), h, ee);
      inv = inv && h == (lane >> 4) && ee == e;
      // the decode's documented map (r4d_trellis_dq.h): k = 8 (e >> 2) + 4 (L >> 4) + (e & 3)
      same = same && i8p::FragK16(lane, e) == 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
    }
  Check(inv, "K16ToFrag inverts FragK16 (all 32 lanes x 8 elements)");
  Check(same, "FragK16 is the trellis decode's k map");
  // every (lane parity, e) pair covers k16 = 0..15 exactly once per half
  bool cover = true;
  int all[16] = {};
  for (int h = 0; h < 2; ++h)
    for (int e = 0; e < 8; ++e) ++all[i8p::FragK16(16 * h, e)];
  for (int k = 0; k < 16; ++k) cover = cover && all[k] == 1;
  Check(cover, "the 16 (half, element) positions cover k16 = 0..15 once");

  const int K = 384, N = 96;
  std::vector<int> hit_a((size_t)i8p::kM * K, 0), hit_w((size_t)N * K, 0);
  for (int r = 0; r < i8p::kM; ++r)
    for (int k = 0; k < K; ++k) ++hit_a[i8p::A8Offset(r, k, K)];
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) ++hit_w[i8p::W8Offset(n, k, K)];
  bool ba = true, bw = true;
  for (int v : hit_a) ba = ba && v == 1;
  for (int v : hit_w) bw = bw && v == 1;
  Check(ba, "A8Offset is a bijection of the 256 x 384 plain matrix onto 98304 bytes");
  Check(bw, "W8Offset is a bijection of the 96 x 384 plain matrix onto 36864 bytes");
  // W8's columns are the decode's: lane L, fragment F, element e is column FragCol, k = 16 kt + FragK16
  bool cols = true;
  for (int pair = 0; pair < 3; ++pair)
    for (int lane = 0; lane < 32; ++lane)
      for (int F = 0; F < 2; ++F)
        for (int e = 0; e < 8; ++e) {
          const int n = i8p::FragCol(pair, lane, F), k = 16 * 5 + i8p::FragK16(lane, e);
          cols = cols && i8p::W8Offset(n, k, K) == i8p::W8BlockOffset(pair, 5, K / 16) + (size_t)lane * 16 + F * 8 + e;
        }
  Check(cols, "W8Offset(FragCol, FragK16) lands on lane * 16 + F * 8 + e of the block");
  bool acols = true;   // the accumulator fragment f of lane L is the column of fragment F = f, a pair's 32 columns once
  for (int pair = 0; pair < 2; ++pair) {
    int seen[32] = {};
    for (int lane = 0; lane < 16; ++lane)
      for (int f = 0; f < 2; ++f) ++seen[i8p::FragCol(pair, lane, f) - 32 * pair];
    for (int lane = 16; lane < 32; ++lane) {   // lanes 16..31 repeat lanes 0..15's columns
      for (int f = 0; f < 2; ++f)
        acols = acols && i8p::FragCol(pair, lane, f) == i8p::FragCol(pair, lane - 16, f);
    }
    for (int j = 0; j < 32; ++j) acols = acols && seen[j] == 1;
  }
  Check(acols, "the two accumulator fragments of a pair are its 32 columns, lanes l and l + 16 share a column");
}

void TestF16() {
  bool rt = true, lib = true;
  for (uint32_t h = 0; h < 65536; ++h) {
    const uint16_t hh = (uint16_t)h;
    if (((hh >> 10) & 0x1F) == 31) continue;   // inf / NaN
    const float f = i8p::F16ToF32(hh);
    rt = rt && i8p::F32ToF16(f) == hh;
    lib = lib && f == r4dx::core::F16ToFloat(hh);
  }
  Check(rt, "F32ToF16(F16ToF32(h)) == h for every finite f16");
  Check(lib, "F16ToF32 equals r4dx::core::F16ToFloat for every finite f16");
  std::mt19937_64 g(7);
  std::uniform_real_distribution<float> u(-70000.f, 70000.f);
  bool conv = true;
  for (int i = 0; i < 2000000; ++i) {
    const float f = u(g) * (float)std::ldexp(1.0, -(int)(g() % 40));
    const uint16_t a = i8p::F32ToF16(f), b = r4dx::core::FloatToF16(f);
    const bool nanb = ((b >> 10) & 0x1F) == 31 && (b & 0x3FF);
    if (a != b && !nanb) { conv = false; break; }
  }
  Check(conv, "F32ToF16 equals r4dx::core::FloatToF16 on 2M random floats (exponents 2^-40 .. 2^16)");
}

void TestQuantizer() {
  // exhaustive over w (every finite f16 of magnitude <= 8, the decode's range is about +-4) x a spread of rs
  bool eq = true;
  long long n = 0;
  int bad_w = -1, bad_rs = -1;
  std::vector<uint16_t> rss;
  for (uint32_t h = 0x2C00; h <= 0x7BFF; h += 83) rss.push_back((uint16_t)h);   // 2^-4 .. 65504
  rss.push_back(0x3C00); rss.push_back(0x4000); rss.push_back(0x5800); rss.push_back(0x57F0);
  for (uint16_t rs : rss) {
    const float rsf = i8p::F16ToF32(rs);
    for (uint32_t h = 0; h < 65536; ++h) {
      const uint16_t w = (uint16_t)h;
      if (((w >> 10) & 0x1F) == 31) continue;
      const float wf = i8p::F16ToF32(w);
      // the quantizer's contract: |w * rs| <= 127.5 (the group's amax maps to 127 + the f16 rounding of rs)
      if (std::fabs((double)wf * (double)rsf) > 127.5) continue;
      ++n;
      if (i8p::QuantW(w, rs) != i8p::QuantWTrick(w, rs)) { eq = false; bad_w = w; bad_rs = rs; goto done; }
    }
  }
done:
  Check(eq, "QuantWTrick (fma + 1536, low byte) == rint(w * rs) for every f16 w with |w rs| <= 127.5, ~330 rs");
  if (!eq) std::printf("    first mismatch w=0x%04X rs=0x%04X\n", bad_w, bad_rs);
  std::printf("    (%lld (w, rs) pairs)\n", n);

  // ties: w * rs exactly k + 1/2 must round to the even integer
  bool ties = true;
  for (int k = -127; k < 127; ++k) {
    const double y = k + 0.5;   // w = y, rs = 1
    const uint16_t w = i8p::F32ToF16((float)y);
    const int want = (k % 2 == 0) ? k : k + 1;
    ties = ties && i8p::QuantWTrick(w, 0x3C00) == want && i8p::QuantW(w, 0x3C00) == want;
  }
  Check(ties, "ties round to even (k + 1/2 at rs = 1, k = -127 .. 126)");

  // idempotence of the table: s_eff = 1 / rs  ->  RsHalf(s_eff) == rs, and the device's reciprocal (within a
  // few f32 ulps of the IEEE one) cannot move it: test with +-4 ulps of 1 / s_eff
  bool idem = true, ulp = true;
  for (uint32_t h = 0x2C00; h <= 0x7BFF; ++h) {
    if (h > 0x7BFF) break;
    const uint16_t rs = (uint16_t)h;
    if (i8p::F16ToF32(rs) > 60000.f) continue;
    const float seff = 1.0f / i8p::F16ToF32(rs);
    idem = idem && i8p::RsHalf(seff) == rs;
    const float r = 1.0f / seff;
    for (int d = -4; d <= 4; ++d) {
      uint32_t b;
      std::memcpy(&b, &r, 4);
      b += d;
      float rd;
      std::memcpy(&rd, &b, 4);
      ulp = ulp && i8p::F32ToF16(std::fmin(rd, 60000.f)) == rs;
    }
  }
  Check(idem, "RsHalf(1 / rs) == rs for every f16 rs in [2^-4, 60000]");
  Check(ulp, "f16(rcp(s_eff)) == rs even with the reciprocal off by +-4 f32 ulps");

  // against the fp32 rule: Gaussian weights in 128-groups (the decoded Q is about N(0,1)), s = amax / 127
  std::mt19937_64 g(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  long long total = 0, flips = 0;
  double e_trick = 0, e_fp32 = 0, ref2 = 0;
  for (int grp = 0; grp < 4000; ++grp) {
    uint16_t q[128];
    float amax = 0.f;
    for (int k = 0; k < 128; ++k) {
      q[k] = i8p::F32ToF16(nd(g));
      amax = std::fmax(amax, std::fabs(i8p::F16ToF32(q[k])));
    }
    const float s0 = i8p::WScale0(amax);
    const uint16_t rs = i8p::RsHalf(s0);
    const float seff = 1.0f / i8p::F16ToF32(rs);
    for (int k = 0; k < 128; ++k) {
      const float w = i8p::F16ToF32(q[k]);
      const int a = i8p::QuantWTrick(q[k], rs), b = i8p::QuantWFp32(q[k], s0);
      ++total;
      flips += a != b;
      e_trick += (double)(a * seff - w) * (a * seff - w);
      e_fp32 += (double)(b * s0 - w) * (b * s0 - w);
      ref2 += (double)w * w;
    }
  }
  std::printf("    quantizer vs the fp32 rule (R4DX_FAKEQ_W): %.3f%% of values differ by 1 LSB; rel RMS error %.5f (f16 trick) vs %.5f (fp32)\n",
              100.0 * flips / total, std::sqrt(e_trick / ref2), std::sqrt(e_fp32 / ref2));
  Check(flips < total / 20 && std::fabs(std::sqrt(e_trick / ref2) / std::sqrt(e_fp32 / ref2) - 1.0) < 0.01,
        "f16-trick quantizer: < 5% LSB flips against the fp32 rule, error RMS within 1% of it");

  // int32 accumulator and its fp32 conversion are exact for one 128-K block
  Check(128LL * 127 * 127 < (1LL << 24), "a 128-K int8 dot (<= 2,064,512) converts to fp32 exactly");
}

// the kernel's fragment-level chain on a CPU, in the arithmetic order of i8g_kernel for each RESC mode, with its
// K slicing: SKW x SKG slices of ktw = KT / (SKW SKG) k-tiles, slice (y, ks) at kt0 = (y SKW + ks) ktw, each slice
// run as one wave does (nkb = ktw / 8 blocks of 8 k-tiles, kb = kt0 / 8 + kbi), the slices' partial tiles added in
// slice order (the kernel's LDS reduction and ws y-sum):
//   0  per 128 K: acc += float(int32) * (sa[kb][row] * sw[kb][col])
//   1  one int32 accumulator over the whole slice, then float(int32) * (sa[0][row] * sw[0][col])
//   2  per 128 K: acc += float(int32) * sw[kb][col]; at the end acc * sa[0][row]
//   3  per 128 K: acc += float(int32) * sa[kb][row]; at the end acc * sw[0][col]
void EmuKernel(const std::vector<int8_t>& a8, const std::vector<float>& sa, const std::vector<int8_t>& w8,
               const std::vector<float>& sw, int N, int K, int resc, int skw, int skg, std::vector<double>& C) {
  const int KT = K / 16;
  const int ktw = KT / (skw * skg), nkb = ktw / 8;
  C.assign((size_t)i8p::kM * N, 0.0);
  for (int pair = 0; pair < N / 32; ++pair)
    for (int rg = 0; rg < 4; ++rg)
      for (int i = 0; i < 4; ++i)
        for (int f = 0; f < 2; ++f)
          for (int slice = 0; slice < skw * skg; ++slice) {
            const int kt0 = slice * ktw, kbg0 = kt0 / 8;
            float accf[32][8] = {};
            int32_t acci_all[32][8] = {};
            for (int kbi = 0; kbi < nkb; ++kbi) {
              const int kb = kbg0 + kbi;
              int32_t acci[32][8] = {};
              for (int u = 0; u < 8; ++u) {
                const int kt = kt0 + kbi * 8 + u;
                int8_t af[32][8], bf[32][8];
                for (int L = 0; L < 32; ++L)
                  for (int e = 0; e < 8; ++e) {
                    af[L][e] = a8[i8p::A8FragOffset(rg, kt, i, L, KT) + e];
                    bf[L][e] = w8[i8p::W8BlockOffset(pair, kt, KT) + (size_t)L * 16 + f * 8 + e];
                  }
                i8p::EmuWmmaI8(af, bf, resc == 1 ? acci_all : acci);
              }
              if (resc != 1)
                for (int L = 0; L < 32; ++L)
                  for (int e = 0; e < 8; ++e) {
                    const int row = i8p::AccRow(rg, i, L, e), col = i8p::FragCol(pair, L, f);
                    const float sav = sa[(size_t)kb * i8p::kM + row], swv = sw[(size_t)kb * N + col];
                    const float t = (float)acci[L][e];
                    accf[L][e] = resc == 0 ? std::fmaf(t, sav * swv, accf[L][e]) : resc == 2 ? std::fmaf(t, swv, accf[L][e]) : std::fmaf(t, sav, accf[L][e]);
                  }
            }
            for (int L = 0; L < 32; ++L)
              for (int e = 0; e < 8; ++e) {
                const int row = i8p::AccRow(rg, i, L, e), col = i8p::FragCol(pair, L, f);
                float v = accf[L][e];
                if (resc == 1) v = (float)acci_all[L][e] * (sa[row] * sw[col]);   // kb 0 of the whole K
                else if (resc == 2) v = v * sa[row];
                else if (resc == 3) v = v * sw[col];
                C[(size_t)row * N + col] += (double)v;
              }
          }
}

void TestEmulatedGemm() {
  const int N = 96, K = 1024, M = i8p::kM, NKB = K / 128;
  std::mt19937_64 g(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> X((size_t)M * K);
  for (float& x : X) x = i8p::F16ToF32(i8p::F32ToF16(nd(g) * 0.7f));
  std::vector<int8_t> Ap;
  std::vector<float> sa;
  i8p::QuantizeActRef(X.data(), K, sa, Ap);
  std::vector<uint16_t> Q((size_t)K * N);
  for (uint16_t& q : Q) q = i8p::F32ToF16(nd(g) * 1.3f);
  std::vector<int8_t> Wp;
  std::vector<float> sw;
  i8p::QuantizeWeightsRef(Q.data(), K, N, sw, Wp);
  const std::vector<int8_t> a8 = i8p::PackA8(Ap, K), w8 = i8p::PackW8(Wp, K, N);
  const int mask_of[4] = {0, 3, 1, 2};
  const int cfgs[5][2] = {{2, 1}, {2, 2}, {4, 1}, {8, 1}, {2, 4}};
  for (int resc = 0; resc < 4; ++resc) {
    std::vector<double> ref((size_t)M * N);
    i8p::RefGemmInt8(Ap.data(), sa.data(), Wp.data(), sw.data(), M, N, K, mask_of[resc], ref.data());
    double rms = 0, worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) rms += ref[i] * ref[i];
    rms = std::sqrt(rms / ref.size());
    for (const auto& cfg : cfgs) {
      std::vector<double> got;
      EmuKernel(a8, sa, w8, sw, N, K, resc, cfg[0], cfg[1], got);
      for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - got[i]));
    }
    std::printf("    RESC %d (scale mask %d): max |emulated - exact| over (skw, skg) in {2x1, 2x2, 4x1, 8x1, 2x4}: %.3e, rms of C %.3f (NKB %d)\n", resc, mask_of[resc], worst, rms, NKB);
    char what[160];
    std::snprintf(what, sizeof what, "emulated kernel chain incl. the K slicing == exact reference, RESC %d (%s)", resc,
                  resc == 0 ? "per-128 both" : resc == 1 ? "both coarse, one rescale" : resc == 2 ? "activation coarse" : "weight coarse");
    Check(worst < 1e-5 * rms, what);
  }  // and the quantized product is the f16 product up to the quantization error (a sanity bound, not a gate)
  std::vector<double> exact((size_t)M * N, 0.0), q((size_t)M * N);
  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += (double)X[(size_t)r * K + k] * (double)i8p::F16ToF32(Q[(size_t)k * N + n]);
      exact[(size_t)r * N + n] = s;
    }
  i8p::RefGemmInt8(Ap.data(), sa.data(), Wp.data(), sw.data(), M, N, K, 0, q.data());
  double d2 = 0, e2 = 0;
  for (size_t i = 0; i < q.size(); ++i) { d2 += (q[i] - exact[i]) * (q[i] - exact[i]); e2 += exact[i] * exact[i]; }
  std::printf("    int8 (per-128 both sides) vs exact f16 x f16 product: rel RMS error %.5f\n", std::sqrt(d2 / e2));
  Check(std::sqrt(d2 / e2) < 0.02, "w8a8 rel RMS error of the product < 2% (Gaussian data)");
}

void TestTrellisChain() {
  for (int KB = 4; KB <= 5; ++KB) {
    const int K = 256, N = 64;
    std::mt19937 g(100 + KB);
    std::vector<uint32_t> words((size_t)K / 16 * (N / 16) * 8 * KB);
    for (uint32_t& w : words) w = g();
    const std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
    const std::vector<uint16_t> Q = trellis_ref::DecodePairGrid(grid, K, N, KB);
    std::vector<int8_t> Wp;
    std::vector<float> sw;
    i8p::QuantizeWeightsRef(Q.data(), K, N, sw, Wp);
    bool range = true, hit = true;
    for (int8_t v : Wp) range = range && v >= -127;
    for (int kb = 0; kb < K / 128; ++kb)
      for (int n = 0; n < N; ++n) {
        int mx = 0;
        for (int k = kb * 128; k < kb * 128 + 128; ++k) mx = std::max(mx, std::abs((int)Wp[(size_t)n * K + k]));
        hit = hit && mx >= 126;   // the amax element is 127 (126 when rs rounds down by a hair)
      }
    std::vector<int8_t> w8 = i8p::PackW8(Wp, K, N);
    // spot check: the block layout read the way the kernel reads it gives back the plain matrix
    bool back = true;
    for (int pair = 0; pair < N / 32; ++pair)
      for (int kt = 0; kt < K / 16; ++kt)
        for (int lane = 0; lane < 32; ++lane)
          for (int F = 0; F < 2; ++F)
            for (int e = 0; e < 8; ++e)
              back = back && w8[i8p::W8BlockOffset(pair, kt, K / 16) + (size_t)lane * 16 + F * 8 + e] ==
                                 Wp[(size_t)i8p::FragCol(pair, lane, F) * K + 16 * kt + i8p::FragK16(lane, e)];
    char what[128];
    std::snprintf(what, sizeof what, "KB %d: trellis decode -> scale table -> int8 W: |q| <= 127, group amax -> 126..127", KB);
    Check(range && hit, what);
    std::snprintf(what, sizeof what, "KB %d: the W8 block read as the kernel reads it is the plain matrix", KB);
    Check(back, what);
  }
}

// The COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse): QuantizeWeightsColRef (one scale per column over the whole K) and
// QuantizeActRowRef (one per row), their relation to the per-128 quantizers, the kernel chain with them (EmuKernel RESC 1 is the
// production RESC 4's math: sa [256], sw [N]), the int32 range, and the accuracy price on Gaussian data.
void TestCoarse() {
  const int N = 96, K = 1024, M = i8p::kM;
  std::mt19937_64 g(31);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> X((size_t)M * K);
  for (float& x : X) x = i8p::F16ToF32(i8p::F32ToF16(nd(g) * 0.7f));
  for (int k = 0; k < K; ++k) X[(size_t)9 * K + k] = 0.f;                                       // an all-zero row (s = 1)
  std::vector<uint16_t> Q((size_t)K * N);
  for (uint16_t& q : Q) q = i8p::F32ToF16(nd(g) * 1.3f);
  for (int k = 0; k < K; ++k) Q[(size_t)k * N + 5] = 0;                                         // an all-zero column
  std::vector<int8_t> Apc, Wpc, Ap, Wp;
  std::vector<float> sac, swc, sa, sw;
  i8p::QuantizeActRowRef(X.data(), K, sac, Apc);
  i8p::QuantizeWeightsColRef(Q.data(), K, N, swc, Wpc);
  i8p::QuantizeActRef(X.data(), K, sa, Ap);
  i8p::QuantizeWeightsRef(Q.data(), K, N, sw, Wp);
  // range, amax -> 127, the all-zero rule
  bool range = true, hit = true;
  for (int8_t v : Apc) range = range && v >= -127;
  for (int8_t v : Wpc) range = range && v >= -127;
  for (int r = 0; r < M; ++r) {
    int mx = 0;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::abs((int)Apc[(size_t)r * K + k]));
    hit = hit && (r == 9 ? mx == 0 : mx == 127);
  }
  for (int n = 0; n < N; ++n) {
    int mx = 0;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::abs((int)Wpc[(size_t)n * K + k]));
    hit = hit && (n == 5 ? mx == 0 : mx >= 126);
  }
  Check(range && hit && sac[9] == 1.0f && swc[5] == 1.0f, "coarse quantizers: |q| <= 127, every row / column amax -> 127 (126 when rs rounds down), all-zero row / column -> scale 1, zeros");
  // the coarse scale of a column is the largest per-128 one (monotone rule), the row scale is the largest per-128 one too
  bool mono = true;
  for (int n = 0; n < N; ++n)
    for (int kb = 0; kb < K / 128; ++kb) mono = mono && swc[n] >= sw[(size_t)kb * N + n];
  for (int r = 0; r < M; ++r)
    for (int kb = 0; kb < K / 128; ++kb) mono = mono && sac[r] >= sa[(size_t)kb * M + r];
  Check(mono, "the coarse scale of a row / column is at least every per-128 scale in it (it is their max)");
  // one 128-block: coarse == per-128 exactly
  {
    std::vector<float> X1((size_t)M * 128);
    for (int r = 0; r < M; ++r) std::copy(X.begin() + (size_t)r * K, X.begin() + (size_t)r * K + 128, X1.begin() + (size_t)r * 128);
    std::vector<uint16_t> Q1((size_t)128 * N);
    std::copy(Q.begin(), Q.begin() + (size_t)128 * N, Q1.begin());
    std::vector<int8_t> a_c, a_b, w_c, w_b;
    std::vector<float> s_ac, s_ab, s_wc, s_wb;
    i8p::QuantizeActRowRef(X1.data(), 128, s_ac, a_c);
    i8p::QuantizeActRef(X1.data(), 128, s_ab, a_b);
    i8p::QuantizeWeightsColRef(Q1.data(), 128, N, s_wc, w_c);
    i8p::QuantizeWeightsRef(Q1.data(), 128, N, s_wb, w_b);
    Check(a_c == a_b && s_ac == s_ab && w_c == w_b && s_wc == s_wb, "K = 128 (one block): the coarse quantizers are the per-128 ones, bit for bit");
  }
  // the kernel chain (EmuKernel RESC 1 on sa [256] / sw [N]) against the exact integer reference, every K split
  {
    const std::vector<int8_t> a8 = i8p::PackA8(Apc, K), w8 = i8p::PackW8(Wpc, K, N);
    std::vector<double> ref((size_t)M * N);
    i8p::RefGemmInt8(Apc.data(), sac.data(), Wpc.data(), swc.data(), M, N, K, 3, ref.data());
    double rms = 0, worst = 0;
    for (double v : ref) rms += v * v;
    rms = std::sqrt(rms / ref.size());
    const int cfgs[5][2] = {{2, 1}, {2, 2}, {4, 1}, {8, 1}, {2, 4}};
    for (const auto& cfg : cfgs) {
      std::vector<double> got;
      EmuKernel(a8, sac, w8, swc, N, K, 1, cfg[0], cfg[1], got);
      for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - got[i]));
    }
    std::printf("    coarse chain (RESC 4 math): max |emulated - exact| over (skw, skg) in {2x1, 2x2, 4x1, 8x1, 2x4}: %.3e, rms of C %.3f\n", worst, rms);
    Check(worst < 1e-5 * rms, "coarse chain (A8 / W8 loads, whole-K int32 accumulate per slice, one rescale) == exact reference");
    // the accuracy price: coarse vs per-128 vs the f16 product, Gaussian data
    std::vector<double> exact((size_t)M * N, 0.0), blk((size_t)M * N);
    for (int r = 0; r < M; ++r)
      for (int n = 0; n < N; ++n) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)X[(size_t)r * K + k] * (double)i8p::F16ToF32(Q[(size_t)k * N + n]);
        exact[(size_t)r * N + n] = s;
      }
    i8p::RefGemmInt8(Ap.data(), sa.data(), Wp.data(), sw.data(), M, N, K, 0, blk.data());
    double dc = 0, db = 0, e2 = 0;
    for (size_t i = 0; i < exact.size(); ++i) {
      dc += (ref[i] - exact[i]) * (ref[i] - exact[i]);
      db += (blk[i] - exact[i]) * (blk[i] - exact[i]);
      e2 += exact[i] * exact[i];
    }
    std::printf("    rel RMS error of the product vs exact f16 x f16 (Gaussian): coarse %.5f, per-128 %.5f\n", std::sqrt(dc / e2), std::sqrt(db / e2));
    Check(std::sqrt(dc / e2) < 0.03 && std::sqrt(dc / e2) < 1.6 * std::sqrt(db / e2), "coarse w8a8 rel RMS error < 3% and within 1.6x of the per-128 one (Gaussian data)");
  }
  // int32 range: the longest slice any legal configuration has is K / 2 (skw 2, skg 1) of the longest K, 17408
  Check((17408LL / 2) * 127 * 127 < (1LL << 31) && 17408LL * 127 * 127 < (1LL << 31), "int32 accumulators cannot overflow: K 17408 x 127 x 127 = 2.8e8 < 2^31 (a slice is at most half of it)");
}

}  // namespace

int main() {
  TestMaps();
  TestF16();
  TestQuantizer();
  TestEmulatedGemm();
  TestCoarse();
  TestTrellisChain();
  std::printf("%s\n", g_ok ? "ALL OK" : "FAILED");
  return g_ok ? 0 : 1;
}

