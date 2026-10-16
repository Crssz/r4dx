// qwen4_exp PLE layer (PleLayer: key / value projections, gate, norms, dilated depthwise conv with persistent state)
// against HF Qwen4ExpTextPLELayer (real dims, small n-gram vocabulary, bf16 on the CPU) from
// tools/reference/qwen4exp/small_fixtures.py. The n-gram rows (`ple.emb`) come from HF's own embedding lookup, so this
// covers everything after the table read. Streaming: 1 + 6 + 1 (+ 11) token chunks must equal the one-shot run bit for
// bit; a conv-state snapshot restored later reproduces the continuation bit for bit; Reset() returns to the empty state.
// Exit 77 when the fixtures are missing.
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <exception>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/ple_layer.h"
#include "small_fixture.h"

using namespace q4test;
using namespace r4dx;
using namespace r4dx::model::qwen4exp;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

bool g_ok = true;

void Expect(bool cond, const char* what) {
  std::printf("  %-62s %s\n", what, cond ? "OK" : "FAIL");
  if (!cond) g_ok = false;
}

double Rms(const uint16_t* p, int64_t n) {
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += double(F(p[i])) * F(p[i]);
  return std::sqrt(s / double(n));
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
    const int T = static_cast<int>(fx.Param("ple.T"));
    const int64_t C = 4 * 2560;
    PleConfig cfg;
    cfg.eps = static_cast<float>(fx.Param("ple.eps"));

    QuantLinear key_proj = MakeBf16Linear(fx.Get("ple.key_proj_w").bf16(), C, 2560);
    QuantLinear value_proj = MakeBf16Linear(fx.Get("ple.value_proj_w").bf16(), 2560, 2560);
    auto norm_key = Upload(fx.Get("ple.norm_key_w"));
    auto norm_query = Upload(fx.Get("ple.norm_query_w"));
    auto norm_conv = Upload(fx.Get("ple.norm_conv_w"));
    auto conv1d = Upload(fx.Get("ple.conv1d_w"));
    PleWeights w{&key_proj, &value_proj, norm_key.data(), norm_query.data(), norm_conv.data(), conv1d.data()};
    PleLayer ple(cfg, w);

    core::Stream stream;
    core::Arena arena(ple.ArenaBytes(T));
    auto emb = Upload(fx.Get("ple.emb"));
    const Tensor& h_in = fx.Get("ple.h");
    const Tensor& h_ref = fx.Get("ple.h_new");
    std::printf("[ple] T=%d channels=%lld state elems=%zu\n", T, static_cast<long long>(C), ple.StateElems());

    // run rows [t0, t0 + n) of the sequence into `h` (a device copy of the whole input), advancing the layer
    auto run = [&](core::DeviceBuffer<uint16_t>& h, int t0, int n) {
      arena.Reset();
      ple.Forward(stream.get(), arena, h.data() + int64_t(t0) * C, emb.data() + int64_t(t0) * 2560, n);
    };
    auto fresh = [&] { return Upload(h_in); };

    // 1. one shot vs HF
    ple.Reset(stream.get());
    auto h_whole = fresh();
    run(h_whole, 0, T);
    const std::vector<uint16_t> whole = Download(h_whole);
    const double rms = Rms(h_ref.bf16(), h_ref.count());
    const ErrorStats s = CompareBf16(whole.data(), h_ref.bf16(), h_ref.count(), 0.02 * rms, 0.02);
    if (!Report("h + PLE out, one shot vs HF", s)) g_ok = false;

    // 1b. a 2^-8 smaller residual: h + out is then dominated by out, so this compares the PLE output itself (HF
    // and ours differ only by the fp32 summation order of the GEMMs / norms feeding bf16 roundings)
    {
      ple.Reset(stream.get());
      auto h_small = Upload(fx.Get("ple.h_small"));
      run(h_small, 0, T);
      const Tensor& ref_small = fx.Get("ple.h_small_new");
      const double rms_small = Rms(ref_small.bf16(), ref_small.count());
      const ErrorStats ss = CompareBf16(Download(h_small).data(), ref_small.bf16(), ref_small.count(), 0.02 * rms_small, 0.02);
      if (!Report("h/256 + PLE out vs HF (out visible)", ss)) g_ok = false;
    }

    // 2. streaming: 1 + 6 + 1 + 11 equals the one-shot run bit for bit (first 8 rows are the 1 + 6 + 1 prefix)
    ple.Reset(stream.get());
    auto h_chunked = fresh();
    int t0 = 0;
    for (int n : {1, 6, 1, 11}) {
      run(h_chunked, t0, n);
      t0 += n;
    }
    Expect(Download(h_chunked) == whole, "chunks 1+6+1+11 == one shot (bit exact)");

    ple.Reset(stream.get());
    auto h_prefix = fresh();
    t0 = 0;
    for (int n : {1, 6, 1}) {
      run(h_prefix, t0, n);
      t0 += n;
    }
    {
      const std::vector<uint16_t> pre = Download(h_prefix);
      Expect(std::memcmp(pre.data(), whole.data(), size_t(8) * C * 2) == 0, "chunks 1+6+1 == first 8 rows of one shot");
    }

    // 3. snapshot after 8 rows, finish, then restore the snapshot and redo the tail
    core::DeviceBuffer<uint16_t> snapshot(ple.StateElems());
    ple.SaveState(stream.get(), snapshot.data());
    auto h_tail = fresh();
    run(h_tail, 8, T - 8);
    const std::vector<uint16_t> tail1 = Download(h_tail);
    ple.LoadState(stream.get(), snapshot.data());
    auto h_tail2 = fresh();
    run(h_tail2, 8, T - 8);
    Expect(Download(h_tail2) == tail1, "snapshot restored after 8 rows reproduces the tail (bit exact)");
    Expect(std::memcmp(tail1.data() + size_t(8) * C, whole.data() + size_t(8) * C, size_t(T - 8) * C * 2) == 0,
           "tail after 8 rows == one shot rows 8..T");

    // 4. the state is really used: continuing without Reset changes the first rows; Reset brings them back
    auto h_again = fresh();
    run(h_again, 0, T);  // state here is the one left after the whole sequence
    Expect(Download(h_again) != whole, "state persists across calls (a second pass without Reset differs)");
    ple.Reset(stream.get());
    auto h_reset = fresh();
    run(h_reset, 0, T);
    Expect(Download(h_reset) == whole, "Reset() restores the empty-history output (bit exact)");

    std::printf(g_ok ? "test_gpu_ple: PASS\n" : "test_gpu_ple: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_ple: exception: %s\n", e.what());
    return 1;
  }
}
