// qwen4_exp gated residual (HyperConn) and the fused embedding gather + 4x repeat, against the HF modules in
// tools/reference/qwen4exp/small_fixtures.py (Qwen4ExpTextGatedResidual at the real 4 x 2560 dims, bf16 on the CPU).
// Fixtures: env Q4EXP_SMALL_FIXTURES (default E:\models\r4dx\q4exp-fixtures\SmallKernels); exit 77 when missing.
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_conn.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"
#include "small_fixture.h"

using namespace q4test;
using namespace r4dx;
using namespace r4dx::model::qwen4exp;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

bool g_ok = true;

double Rms(const uint16_t* p, int64_t n) {
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += double(F(p[i])) * F(p[i]);
  return std::sqrt(s / double(n));
}

// ulps: how many bf16 ulps (relative 2^-7 each, worst case) a single element may be off; atol_rms: absolute slack as a
// fraction of the reference rms (covers elements near zero whose own ulp is tiny but whose inputs carried rounding).
void Check(const std::string& name, const std::vector<uint16_t>& got, const Tensor& ref, double ulps,
           double atol_rms) {
  const int64_t n = ref.count();
  const ErrorStats s = CompareBf16(got.data(), ref.bf16(), n, atol_rms * Rms(ref.bf16(), n), ulps * 0.0078125);
  if (!Report(name, s)) g_ok = false;
}

std::vector<uint16_t> Dl(const core::DeviceBuffer<uint16_t>& b) { return Download(b); }

void Case(Fixtures& fx, const std::string& name, bool with_inject) {
  const std::string p = "hc." + name + ".";
  const int rows = static_cast<int>(fx.Param("hc.rows"));
  const int64_t wide = 4 * 2560;
  std::printf("[%s] rows=%d\n", name.c_str(), rows);

  auto norm_w = Upload(fx.Get(p + "norm_w"));
  QuantLinear down = MakeBf16Linear(fx.Get(p + "down_w").bf16(), 320, wide);
  QuantLinear up = MakeBf16Linear(fx.Get(p + "up_w").bf16(), wide, 320);
  QuantLinear inject;
  HyperConnWeights w{norm_w.data(), &down, &up, nullptr};
  if (with_inject) {
    inject = MakeBf16Linear(fx.Get(p + "inject_w").bf16(), 4, wide);
    w.inject = &inject;
  }
  HyperConnConfig cfg;
  cfg.eps = static_cast<float>(fx.Param("hc.eps"));
  HyperConn hc(cfg, w);

  core::Stream stream;
  core::Arena arena(hc.ArenaBytes(rows));
  auto h = Upload(fx.Get(p + "h"));

  // the grouped norm kernel on its own: HF hc_norm output, one fp32 pass with one rounding -> within one ulp.
  {
    core::DeviceBuffer<uint16_t> n(static_cast<size_t>(rows * wide));
    if (q4x_grouped_rmsnorm_bf16(h.data(), norm_w.data(), n.data(), rows, 4, 2560, cfg.eps, stream.get()) != 0) {
      throw std::runtime_error("q4x_grouped_rmsnorm_bf16 failed");
    }
    Check("hc_norm (kernel)", Dl(n), fx.Get(p + "n"), 1.0, 0.0);
  }

  core::DeviceBuffer<uint16_t> x(static_cast<size_t>(rows * 2560));
  core::DeviceBuffer<uint16_t> inj(static_cast<size_t>(rows * 4));
  arena.Reset();
  if (with_inject) {
    hc.Prep(stream.get(), arena, h.data(), rows, x.data(), inj.data());
  } else {
    hc.Mix(stream.get(), arena, h.data(), rows, x.data());
  }
  Check("x = mean(m * n)", Dl(x), fx.Get(p + "x"), 2.0, 0.01);
  if (!with_inject) return;
  Check("inj = 2*sigmoid(inject(n)/4)", Dl(inj), fx.Get(p + "inj"), 2.0, 0.0);

  // inject kernel alone, fed the HF injection weights: exact (one product rounding + one add rounding)
  {
    auto inj_ref = Upload(fx.Get(p + "inj"));
    auto block_out = Upload(fx.Get(p + "block_out"));
    core::DeviceBuffer<uint16_t> h_new(static_cast<size_t>(rows * wide));
    hc.Inject(stream.get(), h.data(), block_out.data(), inj_ref.data(), rows, h_new.data());
    Check("h_new (inject kernel, HF inj)", Dl(h_new), fx.Get(p + "h_new"), 0.0, 0.0);
  }

  // the whole block in one call, in place: Prep -> callback -> Inject. The callback hands back HF's block output.
  {
    auto block_out = Upload(fx.Get(p + "block_out"));
    bool called = false;
    arena.Reset();
    hc.ApplyBlock(stream.get(), arena, h.data(), rows, [&](const uint16_t* xin, uint16_t* out) {
      called = true;
      // x is the block input: it must equal the standalone Prep's x bit for bit
      core::DeviceBuffer<uint16_t> copy(static_cast<size_t>(rows * 2560));
      R4DX_HIP_CHECK(hipMemcpyAsync(copy.data(), xin, copy.bytes(), hipMemcpyDeviceToDevice, stream.get()));
      const std::vector<uint16_t> a = Dl(copy), b = Dl(x);
      if (std::memcmp(a.data(), b.data(), a.size() * 2) != 0) {
        std::printf("  ApplyBlock input differs from Prep's x\n");
        g_ok = false;
      }
      R4DX_HIP_CHECK(hipMemcpyAsync(out, block_out.data(), block_out.bytes(), hipMemcpyDeviceToDevice, stream.get()));
    });
    if (!called) {
      std::printf("  ApplyBlock never called the block\n");
      g_ok = false;
    }
    Check("h_new (ApplyBlock, in place)", Dl(h), fx.Get(p + "h_new"), 2.0, 0.01);
  }
}

void Embed(Fixtures& fx) {
  std::printf("[embed]\n");
  const Tensor& table = fx.Get("embed.table");
  const Tensor& ids = fx.Get("embed.ids");
  const int64_t rows = ids.count(), vocab = table.shape[0];
  core::Stream stream;
  auto d_table = Upload(table);
  auto d_ids = UploadI32(ids.i32(), static_cast<size_t>(rows));
  core::DeviceBuffer<uint16_t> out(static_cast<size_t>(rows * 4 * 2560));
  EmbedRepeat(stream.get(), d_table.data(), vocab, d_ids.data(), rows, 2560, 4, out.data());
  Check("embed gather x4 repeat (exact)", Dl(out), fx.Get("embed.out"), 0.0, 0.0);

  // out-of-range ids (negative, == vocab) produce NaN rows, valid ones in the same call are untouched
  const int32_t bad_ids[3] = {-1, 7, static_cast<int32_t>(vocab)};
  auto d_bad = UploadI32(bad_ids, 3);
  core::DeviceBuffer<uint16_t> out2(static_cast<size_t>(3 * 4 * 2560));
  EmbedRepeat(stream.get(), d_table.data(), vocab, d_bad.data(), 3, 2560, 4, out2.data());
  const std::vector<uint16_t> o = Dl(out2);
  bool nan_rows = true;
  for (int64_t i = 0; i < 4 * 2560; ++i) {
    nan_rows &= std::isnan(F(o[static_cast<size_t>(i)])) && std::isnan(F(o[static_cast<size_t>(2 * 4 * 2560 + i)]));
    nan_rows &= o[static_cast<size_t>(4 * 2560 + i)] == table.bf16()[7 * 2560 + i % 2560];
  }
  std::printf("  %-34s %s\n", "out-of-range ids -> NaN rows", nan_rows ? "OK" : "FAIL");
  if (!nan_rows) g_ok = false;
}

}  // namespace

int main() {
  try {
    Fixtures fx;
    if (!fx.Open()) {
      std::fprintf(stderr, "[SKIP] fixtures not found in %s (run tools/reference/qwen4exp/small_fixtures.py)\n",
                   Fixtures::Dir().c_str());
      return kSkip;
    }
    Case(fx, "mix", true);
    Case(fx, "final", false);
    Embed(fx);
    std::printf(g_ok ? "test_gpu_hc: PASS\n" : "test_gpu_hc: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_hc: exception: %s\n", e.what());
    return 1;
  }
}
