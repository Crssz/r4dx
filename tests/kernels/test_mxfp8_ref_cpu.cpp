// test_mxfp8_ref_cpu -- the CPU half of the mxfp8 kernel contract (tests/kernels/mxfp8_ref.hpp against the fixed
// codec src/convert/include/r4dx_convert/mxfp8_format.hpp). Pure CPU, no HIP:
//   * the kernel's in-register decode (perm into the high byte, shift / mask, * 2^8 in f16), emulated bit for bit, is
//     exactly DecodeE4m3 for all 254 non-NaN codes, and lands on a NORMAL f16 for every nonzero code;
//   * the E8M0 value 2^(b - 127) at bytes 0 (the f32 subnormal 2^-127), 127, 254;
//   * PackWq / PackWs / UnpackRows: the kernel's forward index map, round trip on random data, plus the layout facts the
//     loader and the TP slicer rely on (a 16-row tile is one contiguous 16*K byte run; a (tile, group) block of wq is
//     512 bytes, of ws 16 bytes);
//   * Gemm (kernel order) against GemmNaive (fp64): bit-exact on the exact class (A in {0, +-1, +-2}, one group per
//     split so no f32 rounding of the fma chain can occur ... checked against bf16(naive)), and within the bound on
//     general data; SK changes only the last bits;
//   * EncodeGroup -> decode round trip: relative error <= 2^-4 of the group's amax / 448 * 2^e grid (no clipping).
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "mxfp8_ref.hpp"

namespace fmt = r4dx_convert::mxfp8;

namespace {
int g_fail = 0, g_checks = 0;
void Check(bool ok, const std::string& what) {
  ++g_checks;
  std::printf("%-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}
}  // namespace

int main() {
  // ---- decode ----
  {
    bool same = true, normal = true;
    int bad = -1;
    for (int b = 0; b < 256; ++b) {
      if (b == 0x7F || b == 0xFF) continue;
      const float want = fmt::DecodeE4m3(static_cast<uint8_t>(b));
      const float got = mxfp8_ref::DeviceDecode(static_cast<uint8_t>(b));
      if (!(got == want && std::signbit(got) == std::signbit(want))) {
        same = false;
        if (bad < 0) bad = b;
      }
      const uint16_t h = mxfp8_ref::DeviceDecodeF16Bits(static_cast<uint8_t>(b));
      if ((b & 0x7F) != 0 && ((h >> 10) & 0x1F) == 0) normal = false;
    }
    Check(same, "device decode == DecodeE4m3 for all 254 non-NaN codes (sign of -0 kept)" +
                    (bad < 0 ? std::string() : " (first bad " + std::to_string(bad) + ")"));
    Check(normal, "every nonzero e4m3 value decodes to a NORMAL f16");
    Check(mxfp8_ref::DeviceDecode(0x7E) == 448.0f && mxfp8_ref::DeviceDecode(0xFE) == -448.0f &&
              mxfp8_ref::DeviceDecode(0x01) == std::ldexp(1.0f, -9) && mxfp8_ref::DeviceDecode(0x08) == std::ldexp(1.0f, -6),
          "decode: +-448, 2^-9 (smallest subnormal), 2^-6 (smallest normal)");
    Check(mxfp8_ref::ScaleValue(0) == std::ldexp(1.0f, -127) && mxfp8_ref::ScaleValue(0) > 0.0f &&
              mxfp8_ref::ScaleValue(127) == 1.0f && mxfp8_ref::ScaleValue(254) == std::ldexp(1.0f, 127),
          "E8M0: bytes 0 / 127 / 254 -> 2^-127 (subnormal, nonzero) / 1 / 2^127");
  }
  // ---- packing ----
  std::mt19937 rng(7);
  {
    const int N = 48, K = 96, ng = K / 32;
    std::vector<uint8_t> code(static_cast<size_t>(N) * K), scale(static_cast<size_t>(N) * ng);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& c : code) c = static_cast<uint8_t>(d(rng));
    for (auto& s : scale) s = static_cast<uint8_t>(d(rng));
    const auto wq = mxfp8_ref::PackWq(code, N, K);
    const auto ws = mxfp8_ref::PackWs(scale, N, K);
    const auto u = mxfp8_ref::UnpackRows(wq, ws, N, K);
    Check(u.code == code && u.scale == scale, "pack -> unpack (the kernel's lane walk) round trip, codes and scales");
    // tile t is bytes [16 t K, 16 (t + 1) K) of wq; block (t, kb) bytes [(t K/32 + kb) 512, +512)
    bool tile = true;
    for (int t = 0; t < N / 16; ++t)
      for (int l = 0; l < 32; ++l)
        for (int kb = 0; kb < K / 32; ++kb)
          for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) {
              const size_t off = w8a16_ref::WqByteOffset(K, t, kb, l, j, i);
              tile = tile && off / (16 * static_cast<size_t>(K)) == static_cast<size_t>(t) &&
                     off / 512 == static_cast<size_t>(t) * (K / 32) + kb;
            }
    Check(tile, "wq: a row tile is one 16*K run, a (tile, 32-K block) is 512 bytes");
    bool wsb = true;
    for (int t = 0; t < N / 16; ++t)
      for (int g = 0; g < ng; ++g)
        for (int r = 0; r < 16; ++r) wsb = wsb && mxfp8_ref::WsIndex(K, t, g, r) / 16 == static_cast<size_t>(t) * ng + g;
    Check(wsb, "ws: a (tile, group) is 16 consecutive bytes, tile-major");
  }
  // ---- the GEMM reference ----
  {
    const int N = 32, K = 256, M = 5;
    std::vector<uint8_t> code(static_cast<size_t>(N) * K), scale(static_cast<size_t>(N) * (K / 32));
    std::uniform_int_distribution<int> d(0, 255), ds(118, 130), de(-2, 2);
    for (auto& c : code) {
      int v;
      do v = d(rng); while (v == 0x7F || v == 0xFF);
      c = static_cast<uint8_t>(v);
    }
    for (auto& s : scale) s = static_cast<uint8_t>(ds(rng));
    const auto wq = mxfp8_ref::PackWq(code, N, K);
    const auto ws = mxfp8_ref::PackWs(scale, N, K);
    const auto u = mxfp8_ref::UnpackRows(wq, ws, N, K);
    std::vector<uint16_t> a(static_cast<size_t>(M) * K);
    std::normal_distribution<float> dn(0.0f, 1.0f);
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k)
        a[static_cast<size_t>(m) * K + k] = w8a16_ref::FloatToBf16Rne(m < 2 ? static_cast<float>(de(rng)) : dn(rng));
    const auto nv = mxfp8_ref::GemmNaive(a, u, M, K);
    bool bound = true;
    for (int SK : {1, 2, 4, 8}) {
      const auto c = mxfp8_ref::Gemm(a, u, M, K, SK);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          const size_t i = static_cast<size_t>(m) * N + n;
          bound = bound && mxfp8_ref::WithinBound(w8a16_ref::Bf16ToFloat(c[i]), nv.c[i], nv.a_norm[m], nv.w_norm[n]);
        }
    }
    Check(bound, "Gemm (kernel order, SK 1/2/4/8) within the bound of the fp64 GEMM");
    // one group of exact-class data at scale 2^0: every partial sum exact, the fma chain exact -> bf16(naive)
    const int K1 = 32;
    std::vector<uint8_t> c1(static_cast<size_t>(N) * K1), s1(N, 127);
    for (size_t i = 0; i < c1.size(); ++i) c1[i] = code[i];
    const auto u1 = mxfp8_ref::UnpackRows(mxfp8_ref::PackWq(c1, N, K1), mxfp8_ref::PackWs(s1, N, K1), N, K1);
    std::vector<uint16_t> a1(static_cast<size_t>(2) * K1);
    for (auto& x : a1) x = w8a16_ref::FloatToBf16Rne(static_cast<float>(de(rng)));
    const auto g1 = mxfp8_ref::Gemm(a1, u1, 2, K1, 1);
    const auto n1 = mxfp8_ref::GemmNaive(a1, u1, 2, K1);
    bool exact = true;
    for (size_t i = 0; i < g1.size(); ++i) {
      const uint16_t want = w8a16_ref::FloatToBf16Rne(static_cast<float>(n1.c[i]));
      exact = exact && (g1[i] == want || (want == 0x8000 && g1[i] == 0));
    }
    Check(exact, "exact class, one group at scale 1: Gemm == bf16(fp64 GEMM) bit for bit");
  }
  // ---- the codec the converter writes ----
  {
    std::normal_distribution<float> dn(0.0f, 1.0f);
    std::uniform_real_distribution<float> dlog(-30.0f, 30.0f);
    bool ok = true, noclip = true;
    for (int trial = 0; trial < 2000; ++trial) {
      float v[32];
      const float sc = std::exp2(dlog(rng));
      for (float& x : v) x = dn(rng) * sc;
      uint8_t codes[32];
      const uint8_t s = fmt::EncodeGroup(v, 32, codes);
      const float step = fmt::ScaleValue(s);
      float amax = 0.0f;
      for (float x : v) amax = std::fmax(amax, std::fabs(x));
      noclip = noclip && 448.0f * step >= amax;
      for (int i = 0; i < 32; ++i) {
        const float back = fmt::DecodeE4m3(codes[i]) * step;
        const float err = std::fabs(back - v[i]);
        // RNE to 3 mantissa bits: <= 2^-4 relative for normals, <= half the subnormal step 2^-10 * step otherwise
        ok = ok && err <= std::fmax(std::fabs(v[i]) * 0.0625f, std::ldexp(step, -10)) * 1.0001f;
      }
    }
    Check(noclip, "EncodeGroup: 448 * 2^e >= amax (no element clipped), 2000 groups over 2^-30 .. 2^30");
    Check(ok, "EncodeGroup -> DecodeE4m3 * 2^e: error <= 2^-4 relative (or half a subnormal step)");
  }
  std::printf("test_mxfp8_ref_cpu: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
