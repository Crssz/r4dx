// qwen4_exp decode fusions (docs/qwen4exp.md "Decode performance"): every fused kernel against the unfused chain it replaces, BITWISE.
// Random weights and activations at the real shapes (hc 4 x hidden 2560, lowrank 320), several magnitudes (bf16 rounding points
// matter most where silu / sigmoid saturate), several weight sets. No fixtures: the unfused chain IS the reference (the same
// kernels the production prefill path and R4DX_Q4_FUSE_*=0 run).
//   * R4DX_Q4_FUSE_HC: HyperConn::Prep (one row) fused vs R4DX_Q4_FUSE_HC=0, both with and without inject weights (the layer's
//     mixer and the final mixer), x_out and inj_out compared bit for bit.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <exception>
#include <memory>
#include <random>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/fuse_flags.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_conn.h"
#include "r4dx/model/qwen4exp/qsa.h"

using namespace r4dx;
using namespace r4dx::model::qwen4exp;
using r4dx::model::QuantLinear;

namespace {

bool g_ok = true;

std::vector<uint16_t> RandBf16(size_t n, uint32_t seed, float amp) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-amp, amp);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = core::FloatToBf16(d(g));
  return v;
}

core::DeviceBuffer<uint16_t> Up(const std::vector<uint16_t>& v) {
  core::DeviceBuffer<uint16_t> b(v.size());
  b.CopyFromHost(v.data(), v.size());
  return b;
}

void SetFuse(const char* name, bool on) {
#ifdef _WIN32
  _putenv_s(name, on ? "1" : "0");
#else
  setenv(name, on ? "1" : "0", 1);
#endif
}

void FuseHc() {
  constexpr int kHc = 4, kH = 2560, kL = 320;
  constexpr int64_t kWide = int64_t{kHc} * kH;
  core::Stream stream;
  core::Arena arena(size_t{8} << 20);
  int cases = 0, bad = 0;
  for (int with_inject = 1; with_inject >= 0; --with_inject) {
    for (int set = 0; set < 3; ++set) {
      auto norm_w = Up(RandBf16(kWide, 11 + set, 0.2f));
      QuantLinear down = MakeBf16Linear(RandBf16(size_t{kL} * kWide, 21 + set, 0.03f).data(), kL, kWide);
      QuantLinear up = MakeBf16Linear(RandBf16(size_t{kWide} * kL, 31 + set, 0.08f).data(), kWide, kL);
      QuantLinear inject;
      HyperConnWeights w{norm_w.data(), &down, &up, nullptr};
      if (with_inject != 0) {
        inject = MakeBf16Linear(RandBf16(size_t{kHc} * kWide, 41 + set, 0.03f).data(), kHc, kWide);
        w.inject = &inject;
      }
      HyperConn hc(HyperConnConfig{}, w);
      for (int mag = 0; mag < 5; ++mag) {
        const float amp = mag == 0 ? 1.0f : mag == 1 ? 8.0f : mag == 2 ? 0.05f : mag == 3 ? 100.0f : 3000.0f;
        auto h = Up(RandBf16(kWide, 100 + 7 * set + mag, amp));
        std::vector<uint16_t> x_ref, x_got, i_ref, i_got;
        for (int fused = 0; fused < 2; ++fused) {
          SetFuse("R4DX_Q4_FUSE_HC", fused != 0);
          core::DeviceBuffer<uint16_t> x(kH), inj(kHc);
          arena.Reset();
          hc.Prep(stream.get(), arena, h.data(), 1, x.data(), with_inject != 0 ? inj.data() : nullptr);
          R4DX_HIP_CHECK(hipStreamSynchronize(stream.get()));
          (fused != 0 ? x_got : x_ref) = x.CopyToHost();
          if (with_inject != 0) (fused != 0 ? i_got : i_ref) = inj.CopyToHost();
        }
        ++cases;
        const bool same = x_ref == x_got && i_ref == i_got;
        if (!same) {
          ++bad;
          size_t nx = 0;
          for (size_t i = 0; i < x_ref.size(); ++i) nx += x_ref[i] != x_got[i];
          std::printf("  MISMATCH inject=%d set=%d amp=%g: x differs in %zu / %d, inj %s\n", with_inject, set, amp, nx, kH,
                      i_ref == i_got ? "equal" : "differs");
        }
      }
    }
  }
  SetFuse("R4DX_Q4_FUSE_HC", true);
  std::printf("  %-52s %d / %d cases bit-identical\n", "HyperConn::Prep, R4DX_Q4_FUSE_HC=1 vs =0 (rows = 1)", cases - bad, cases);
  if (bad != 0) g_ok = false;
}

void FuseQsa() {
  // QsaLayerCore::Forward (one decode row) with R4DX_Q4_FUSE_QSA=1 (the twelve query heads of a kv head over four workgroups) vs =0
  // (one workgroup per kv head): attn_out bit for bit, at contexts of one tile, several tiles, and past the dense / sparse switch.
  core::Stream stream;
  using r4dx::model::attention::PagedKvCache;
  QsaDims d;
  d.max_ctx = 8192;
  const int qw = d.heads_q * d.head_dim, kvw = d.heads_kv * d.head_dim, iw = (d.index_heads + 1) * d.index_dim;
  auto qg = Up(RandBf16(size_t{256} * 2 * qw, 1, 2.0f));
  auto k = Up(RandBf16(size_t{256} * kvw, 2, 2.0f));
  auto v = Up(RandBf16(size_t{256} * kvw, 3, 2.0f));
  auto ix = Up(RandBf16(size_t{256} * iw, 4, 2.0f));
  auto qn = Up(RandBf16(256, 5, 0.2f)), kn = Up(RandBf16(256, 6, 0.2f)), iqn = Up(RandBf16(128, 7, 0.2f)), ikn = Up(RandBf16(128, 8, 0.2f));
  core::DeviceBuffer<float> kd(2), vd(2);
  const float kdv[2] = {0.05f, 0.07f}, vdv[2] = {0.03f, 0.04f};
  kd.CopyFromHost(kdv, 2);
  vd.CopyFromHost(vdv, 2);
  QsaWeights w{qn.data(), kn.data(), iqn.data(), ikn.data(), kd.data(), vd.data()};
  core::DeviceBuffer<int32_t> pos(256);
  core::Arena arena(size_t{256} << 20);
  int cases = 0, bad = 0;
  for (const int ctx : {1, 2, 17, 243, 256, 257, 700, 1025, 2048, 2200, 4500, 8000}) {
    QsaLayerCore core_l(d);
    PagedKvCache kv(d.heads_kv, d.head_dim, 16, d.max_ctx);
    R4DX_HIP_CHECK(hipDeviceSynchronize());  // the constructors' null-stream memsets are not ordered against `stream`
    const int t0 = ctx - 1;
    for (int first = 0; first < t0; first += 256) {  // the state of the first ctx - 1 positions (chunked prefill, unaffected by the switch)
      const int rows = std::min(256, t0 - first);
      std::vector<int32_t> p(256);
      for (int i = 0; i < rows; ++i) p[static_cast<size_t>(i)] = first + i;
      pos.CopyFromHost(p.data(), 256);
      arena.Reset();
      core_l.Forward(stream.get(), arena, QsaInputs{qg.data(), k.data(), v.data(), ix.data()}, w, kv, first, rows, pos.data());
      R4DX_HIP_CHECK(hipStreamSynchronize(stream.get()));  // `pos` is rewritten below (a null-stream copy is not ordered against `stream`)
    }
    const int32_t last = t0;
    pos.CopyFromHost(&last, 1);
    std::vector<uint16_t> out[2];
    for (int fused = 0; fused < 2; ++fused) {
      SetFuse("R4DX_Q4_FUSE_QSA", fused != 0);
      arena.Reset();
      uint16_t* o = core_l.Forward(stream.get(), arena, QsaInputs{qg.data(), k.data(), v.data(), ix.data()}, w, kv, t0, 1, pos.data());
      R4DX_HIP_CHECK(hipStreamSynchronize(stream.get()));
      out[fused].resize(static_cast<size_t>(qw));
      R4DX_HIP_CHECK(hipMemcpy(out[fused].data(), o, out[fused].size() * 2, hipMemcpyDeviceToHost));
    }
    SetFuse("R4DX_Q4_FUSE_QSA", true);
    ++cases;
    if (out[0] != out[1]) {
      ++bad;
      std::printf("  MISMATCH QSA decode at ctx %d\n", ctx);
    }
  }
  std::printf("  %-52s %d / %d contexts bit-identical\n", "QsaLayerCore::Forward, R4DX_Q4_FUSE_QSA=1 vs =0 (T = 1)", cases - bad, cases);
  if (bad != 0) g_ok = false;
}

}  // namespace

int main() {
  try {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
      std::fprintf(stderr, "[SKIP] no HIP device\n");
      return 77;
    }
    FuseHc();
    FuseQsa();
    std::printf(g_ok ? "test_gpu_fuse_decode: PASS\n" : "test_gpu_fuse_decode: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_fuse_decode: exception: %s\n", e.what());
    return 1;
  }
}
