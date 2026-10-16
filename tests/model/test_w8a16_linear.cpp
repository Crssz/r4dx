// tests/model/test_w8a16_linear.cpp -- docs/w8a16.md 7.3 `model_w8a16_linear`: ApplyLinear on w8a16 QuantLinears vs the CPU
// reference, at the REAL LFM2.5-24B-A2B shapes, through the model's own dispatch (PickTuning's w8a16 fallback, the 64-row
// chunking of ApplyLinear, the kernel's launch).
//
//   attn.q / attn.o / conv.out_proj [2048, 2048]   attn.k / attn.v  [512, 2048]     conv.in_proj    [6144, 2048]
//   dense mlp.gate_up               [23552, 2048]  dense mlp.down   [2048, 11776]   lm_head         [128000, 2048]
//   expert gate_up                  [3072, 2048]   expert down      [2048, 1536]
//
// Weights: Gaussian rows (with a few dead rows of amax ~1e-25, which become zero groups), quantized with the converter's
// own QuantizeInt8Symmetric + PackW8Bytes + PackW8a16Scales (quant_int8.hpp); the reference is DequantW8a16 (written
// from the kernel's indexing) x the bf16 activations, in fp64. Per shape, group (128; and 64 for the expert shapes) and M in
// {1, 8, 16, 17, 64, 65, 256} (65 and 256 cross the 64-row chunking; the head and the dense gate_up run M <= 65 only, to keep
// the fp64 reference short):
//   - every element within 4 bf16 ulp(|ref|) + 1e-5 * ||a_row|| * ||w_row|| of the reference (the kernel test's bound,
//     docs/w8a16.md 7.2, doubled for the model level);
//   - row identity: every row of the M = 8 and M = 16 calls equals the M = 1 call on that row (the bits a verify window and
//     LFM2's per-slot / sorted MoE paths rely on);
//   - a second identical call gives identical bytes;
//   - the output is finite.
// Plus the loader: the tiny w8a16 container's linears loaded through container_util::LoadQuantLinearWithFallback (the
// path every model family takes) match DequantW8a16 of the same on-disk bytes at M 1 and 65, and a w8a16 request on a
// linear that has only a bf16 form (the per-linear fallback of a MIXED container) loads it as bf16 and counts it.
//
// GPU test (one device; NOT RUN by its author). The loader section SKIPs (not the test) when the tiny container is absent.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "container_load_util.h"
#include "linear.h"
#include "lfm2_container_info.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/models_root.h"
#include "r4dx_convert/quant_int8.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "test_common.h"

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::FloatToBf16;
using r4dx::model::ApplyLinear;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

int g_checks = 0, g_failures = 0;
void Check(bool ok, const std::string& what) {
  ++g_checks;
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

double Bf16Ulp(double v) {
  const double a = std::fabs(v);
  if (a < 1e-30) return 1e-30;
  return std::ldexp(1.0, std::ilogb(a) - 7);
}

struct Shape {
  const char* name;
  int64_t N, K;
  int64_t max_m;  // the largest M run (the fp64 reference of the widest shapes stays short)
  bool g64;       // also run group 64
};

std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-6.f, 3.f), od(10.f, 40.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  std::vector<float> row(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

struct Host {
  std::vector<float> wdq;  // DequantW8a16 [N, K]
  std::vector<double> wnorm;  // ||w_row||
};

QuantLinear Upload(const std::vector<uint8_t>& wq, const std::vector<uint16_t>& ws, int64_t N, int64_t K, int group) {
  QuantLinear q;
  q.layout = Layout::kW8a16;
  q.N = N;
  q.K = K;
  q.w8a16_group = group;
  q.w8_wq = DeviceBuffer<uint8_t>(wq.size());
  q.w8_wq.CopyFromHost(wq);
  q.w8_ws = DeviceBuffer<uint16_t>(ws.size());
  q.w8_ws.CopyFromHost(ws);
  return q;
}

QuantLinear MakeLinear(const Shape& s, int group, std::mt19937_64& rng, Host* h) {
  const int N = static_cast<int>(s.N), K = static_cast<int>(s.K);
  std::vector<float> w(static_cast<size_t>(s.N * s.K));
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-12.f, -6.f);
  for (int64_t n = 0; n < s.N; ++n) {
    // About 1 % dead rows (amax ~1e-25: zero groups, docs/w8a16.md 2), the rest at a per-row scale 2^-12 .. 2^-6.
    const bool dead = (rng() % 97) == 0;
    const float sc = dead ? 1e-25f : std::exp2(ud(rng));
    for (int64_t k = 0; k < s.K; ++k) w[static_cast<size_t>(n * s.K + k)] = nd(rng) * sc;
  }
  std::vector<uint8_t> q8;
  std::vector<uint16_t> s16;
  r4dx_convert::QuantizeInt8Symmetric(w.data(), N, K, group, 8, q8, s16);
  const std::vector<uint8_t> wq = r4dx_convert::PackW8Bytes(q8, N, K, 8);
  const std::vector<uint16_t> ws = r4dx_convert::PackW8a16Scales(s16, N, K, group);
  h->wdq = r4dx_convert::DequantW8a16(wq, ws, N, K, group);
  h->wnorm.assign(static_cast<size_t>(s.N), 0.0);
  for (int64_t n = 0; n < s.N; ++n) {
    double ss = 0;
    for (int64_t k = 0; k < s.K; ++k) ss += static_cast<double>(h->wdq[static_cast<size_t>(n * s.K + k)]) * h->wdq[static_cast<size_t>(n * s.K + k)];
    h->wnorm[static_cast<size_t>(n)] = std::sqrt(ss);
  }
  return Upload(wq, ws, s.N, s.K, group);
}

std::vector<uint16_t> Run(hipStream_t st, r4dx::core::Arena& arena, const QuantLinear& q, const std::vector<uint16_t>& x,
                          int64_t M) {
  DeviceBuffer<uint16_t> xd(x.size()), yd(static_cast<size_t>(M * q.N));
  xd.CopyFromHost(x);
  std::vector<uint16_t> sent(yd.size(), 0x7FC1u);
  yd.CopyFromHost(sent);
  arena.Reset();
  ApplyLinear(st, arena, q, xd.data(), yd.data(), M);
  R4DX_HIP_CHECK(hipStreamSynchronize(st));
  return yd.CopyToHost();
}

// Elements beyond the bound; `worst` = the largest err / bound.
size_t CompareRef(const std::vector<uint16_t>& y, const std::vector<uint16_t>& x, const std::vector<float>& wdq,
                  const std::vector<double>& wnorm, int64_t M, int64_t N, int64_t K, double* worst, bool* finite) {
  size_t bad = 0;
  std::vector<double> a(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    double an = 0;
    for (int64_t k = 0; k < K; ++k) {
      a[static_cast<size_t>(k)] = Bf16ToFloat(x[static_cast<size_t>(m * K + k)]);
      an += a[static_cast<size_t>(k)] * a[static_cast<size_t>(k)];
    }
    an = std::sqrt(an);
    for (int64_t n = 0; n < N; ++n) {
      const float* wr = wdq.data() + n * K;
      double ref = 0;
      for (int64_t k = 0; k < K; ++k) ref += a[static_cast<size_t>(k)] * wr[k];
      const double got = Bf16ToFloat(y[static_cast<size_t>(m * N + n)]);
      if (!std::isfinite(got)) *finite = false;
      const double lim = 4.0 * Bf16Ulp(ref) + 1e-5 * an * wnorm[static_cast<size_t>(n)] + 1e-30;
      const double err = std::fabs(got - ref);
      if (!(err <= lim)) ++bad;
      *worst = std::max(*worst, err / lim);
    }
  }
  return bad;
}

void SyntheticShapes(hipStream_t st, r4dx::core::Arena& arena) {
  const Shape shapes[] = {
      {"expert gate_up", 3072, 2048, 256, true}, {"expert down", 2048, 1536, 256, true},
      {"attn.q/o, conv.out_proj", 2048, 2048, 256, false}, {"attn.k/v", 512, 2048, 256, false},
      {"conv.in_proj", 6144, 2048, 65, false},   {"dense mlp.down", 2048, 11776, 65, false},
      {"dense mlp.gate_up", 23552, 2048, 17, false}, {"lm_head", 128000, 2048, 1, false},
  };
  std::mt19937_64 rng(0x5EED8A16ull);
  double worst_all = 0;
  for (const Shape& s : shapes) {
    for (int group : {128, 64}) {
      if (group == 64 && !s.g64) continue;
      Host h;
      const QuantLinear q = MakeLinear(s, group, rng, &h);
      const std::string tag = std::string(s.name) + " [" + std::to_string(s.N) + ", " + std::to_string(s.K) + "] g" +
                              std::to_string(group);
      for (int64_t M : {int64_t{1}, int64_t{8}, int64_t{16}, int64_t{17}, int64_t{64}, int64_t{65}, int64_t{256}}) {
        if (M > s.max_m) continue;
        const r4dx::model::LinearTuning t = r4dx::model::PickTuning(Layout::kW8a16, s.N, s.K, std::min<int64_t>(M, 64), group);
        const std::vector<uint16_t> x = RandomX(rng, M, s.K);
        const std::vector<uint16_t> y = Run(st, arena, q, x, M);
        double worst = 0;
        bool finite = true;
        const size_t bad = CompareRef(y, x, h.wdq, h.wnorm, M, s.N, s.K, &worst, &finite);
        worst_all = std::max(worst_all, worst);
        Check(bad == 0 && finite, tag + " M=" + std::to_string(M) + " (WV " + std::to_string(t.WV) + " SK " +
                                      std::to_string(t.SK) + " MB " + std::to_string(t.MB) + " NPW " + std::to_string(t.NPW) +
                                      " NT " + std::to_string(t.NT) + "): " + std::to_string(bad) +
                                      " element(s) beyond the bound (worst " + std::to_string(worst) + " of it)" +
                                      (finite ? "" : ", NON-FINITE output"));
        if (M == 8 || M == 16) {
          bool same = true;
          for (int64_t m = 0; m < M; ++m) {
            const std::vector<uint16_t> row(x.begin() + m * s.K, x.begin() + (m + 1) * s.K);
            const std::vector<uint16_t> y1 = Run(st, arena, q, row, 1);
            same = same && std::memcmp(y1.data(), y.data() + m * s.N, static_cast<size_t>(s.N) * 2) == 0;
          }
          Check(same, tag + " M=" + std::to_string(M) + ": every row equals the M = 1 call on it (row identity)");
          Check(Run(st, arena, q, x, M) == y, tag + " M=" + std::to_string(M) + ": a second call gives identical bytes");
        }
      }
    }
  }
  std::printf("synthetic shapes: worst element %.3f of the bound\n", worst_all);
}

void LoaderSection(hipStream_t st, r4dx::core::Arena& arena) {
  namespace cu = r4dx::model::container_util;
  const char* env = std::getenv("R4DX_LFM2_TINY_W8A16_DIR");
  const std::string dir = env != nullptr && *env != '\0' ? std::string(env) : r4dx::ModelsPath("r4dx/lfm2/tiny_w8a16");
  const std::string path = dir + "/tiny_w8a16.r4dx";
  if (!r4dx_test::FileExists(path)) {
    std::printf("SKIP: %s absent -- the loader section is not run\n", path.c_str());
    return;
  }
  const r4dx::model::Lfm2ContainerInfo info = r4dx::model::InspectLfm2Container(path);
  cu::LinearLoadMeta meta;
  meta.path = path;
  meta.w4a16 = cu::CheckQuantGroups(info.metadata, path, Layout::kW8a16, Layout::kW8a16, Layout::kBf16);
  meta.w8a16 = cu::CheckW8a16Choice(info.metadata, path, Layout::kW8a16, Layout::kW8a16, Layout::kBf16, 1);
  Check(meta.w8a16.has_value() && meta.w8a16->group == 128, "CheckW8a16Choice on the tiny container: group 128");
  const r4dx_convert::SafetensorsReader r(r4dx_convert::Utf8ToWide(path));
  const int64_t H = info.config.hidden_size;
  struct L {
    std::string base;
    int64_t N, K;
  };
  const std::vector<L> lins = {
      {r4dx::model::Lfm2ExpertBase(2, 3, "gate_up"), info.config.ExpertGateUpRows(), H},
      {r4dx::model::Lfm2ExpertBase(3, 5, "down"), H, info.config.moe_intermediate_size},
      {"text.layers.0.conv.in_proj", info.config.ConvInProjRows(), H},
      {"text.layers.1.mlp.down", H, info.config.intermediate_size},
      {"lm_head", info.config.vocab_size, H},
  };
  std::mt19937_64 rng(0xA16ull);
  for (const L& l : lins) {
    int fallbacks = 0;
    const QuantLinear q = cu::LoadQuantLinearWithFallback(r, meta, l.base, Layout::kW8a16, l.N, l.K, &fallbacks);
    Check(q.layout == Layout::kW8a16 && fallbacks == 0 && q.w8a16_group == 128 &&
              q.w8_wq.size() == static_cast<size_t>(l.N * l.K) && q.w8_ws.size() == static_cast<size_t>(l.N * l.K / 128),
          l.base + ": loaded as w8a16 (no fallback), buffers sized N*K and N*K/128");
    const auto* wq = reinterpret_cast<const uint8_t*>(r.Data(l.base + ".w8a16.wq"));
    const auto* ws = reinterpret_cast<const uint16_t*>(r.Data(l.base + ".w8a16.ws"));
    Host h;
    h.wdq = r4dx_convert::DequantW8a16(wq, ws, static_cast<int>(l.N), static_cast<int>(l.K), 128);
    h.wnorm.assign(static_cast<size_t>(l.N), 0.0);
    for (int64_t n = 0; n < l.N; ++n) {
      double ss = 0;
      for (int64_t k = 0; k < l.K; ++k) ss += static_cast<double>(h.wdq[static_cast<size_t>(n * l.K + k)]) * h.wdq[static_cast<size_t>(n * l.K + k)];
      h.wnorm[static_cast<size_t>(n)] = std::sqrt(ss);
    }
    for (int64_t M : {int64_t{1}, int64_t{65}}) {
      const std::vector<uint16_t> x = RandomX(rng, M, l.K);
      const std::vector<uint16_t> y = Run(st, arena, q, x, M);
      double worst = 0;
      bool finite = true;
      const size_t bad = CompareRef(y, x, h.wdq, h.wnorm, M, l.N, l.K, &worst, &finite);
      Check(bad == 0 && finite, l.base + " (from the container) M=" + std::to_string(M) + ": " + std::to_string(bad) +
                                    " element(s) beyond the bound (worst " + std::to_string(worst) + ")");
    }
  }
  // The per-linear fallback of a MIXED container (bf16 non-expert + w8a16 experts, docs/w8a16.md 5.2 fallback A): the tiny
  // container is all-w8a16, so a w8a16 request is made for a tensor that exists only in a bf16 form here -- the
  // bare text.embed_tokens viewed as a [vocab, hidden] linear -- which must load as bf16 and be counted (tier 3, the bare
  // form; the .bf16.w tier is the same code path one test earlier) and apply as a bf16 GEMM.
  {
    int fallbacks = 0;
    const QuantLinear q = cu::LoadQuantLinearWithFallback(r, meta, "text.embed_tokens", Layout::kW8a16,
                                                          info.config.vocab_size, H, &fallbacks);
    Check(q.layout == Layout::kBf16 && fallbacks == 1, "a w8a16 request on a bf16-only tensor falls back to bf16, counted once");
    const std::vector<uint16_t> x = RandomX(rng, 3, H);
    const std::vector<uint16_t> y = Run(st, arena, q, x, 3);
    const auto* wb = reinterpret_cast<const uint16_t*>(r.Data("text.embed_tokens"));
    double worst = 0;
    size_t bad = 0;
    for (int64_t m = 0; m < 3; ++m) {
      for (int64_t n = 0; n < info.config.vocab_size; ++n) {
        double ref = 0, mag = 0;
        for (int64_t k = 0; k < H; ++k) {
          const double p = static_cast<double>(Bf16ToFloat(x[static_cast<size_t>(m * H + k)])) * Bf16ToFloat(wb[n * H + k]);
          ref += p;
          mag += std::fabs(p);
        }
        const double err = std::fabs(Bf16ToFloat(y[static_cast<size_t>(m * info.config.vocab_size + n)]) - ref);
        const double lim = 4.0 * Bf16Ulp(ref) + 1e-5 * mag + 1e-30;
        if (!(err <= lim)) ++bad;
        worst = std::max(worst, err / lim);
      }
    }
    Check(bad == 0, "the fallback-loaded bf16 linear applies as a bf16 GEMM (" + std::to_string(bad) + " beyond, worst " +
                        std::to_string(worst) + ")");
  }
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_w8a16_linear", []() -> int {
    r4dx::core::Stream stream;
    r4dx::core::Arena arena(64ull * 1024 * 1024);
    SyntheticShapes(stream.get(), arena);
    LoaderSection(stream.get(), arena);
    std::printf("%d check(s), %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  });
}
