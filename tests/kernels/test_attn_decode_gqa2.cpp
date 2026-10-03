// tests/kernels/test_attn_decode_gqa2.cpp -- r4d_attn_decode_h256_gqa2_fp8kv, libr4d's sliding-window
// split-KV decode / verify (docs/gemma4-plan.md 3.3, task M1-32), against the stage-0 reference kernel on
// the SAME fp8 cache bytes (tests/kernels/gemma_attn_fixture.hpp). Gemma 4's sliding geometry: 16 q / 8
// KV heads, head_dim 256, scale 1.0, window 1024 on the 1536-token ring.
//   * q_len 1 (decode), 2, 8, 16 (verify windows) and 32 (the most the 64-row decode band holds at gqa
//     2), at contexts that wrap the ring (1537, 3001, 3500, 5000) and that do not (700);
//   * small windows (100, 17) and a window longer than the context;
//   * every split count: the law's own (splits = 0), 1, 4, 16, and the segment count the law picks is
//     reported;
//   * INFORMATIONAL, not gated: how far a verify window's row t is from the q_len = 1 launch at that
//     row's own position (ctx' = ctx - q_len + t + 1). With a window the split geometry depends on the
//     earliest row's window start, so unlike the unwindowed kernel the two are not bit-identical by
//     construction; the numbers say how close they are (the M1b / DFlash identity argument needs them).
//   * argument refusals (-4 rows, -5 scratch, -6 window).
// GPU test (ctest sets HIP_VISIBLE_DEVICES=1).
#include "gemma_attn_fixture.hpp"

using namespace gemma_attn;

namespace {
int g_fail = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-100s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_fail;
}

// Launches the windowed decode with a caller-owned scratch of the size the library asks for.
int Launch(R4DArgsW* a, DeviceBuffer<uint8_t>* scratch) {
  const int64_t bytes = r4d_attn_decode_h256_gqa2_scratch_bytes(a);
  if (static_cast<int64_t>(scratch->size()) < bytes) scratch->Resize(static_cast<size_t>(bytes));
  a->scratch = scratch->data();
  const int rc = r4d_attn_decode_h256_gqa2_fp8kv(a, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return rc;
}

void Run(const Shape& s, int splits, std::mt19937& rng, bool identity_report) {
  Fixture f(s, rng);
  const std::vector<uint16_t> ref = f.Reference();
  DeviceBuffer<uint8_t> scratch;
  R4DArgsW a = f.Args(f.OutK(), f.Q(), s.q_len, s.ctx, f.Seqused());
  a.splits = splits;
  f.ZeroOutK();
  const int rc = Launch(&a, &scratch);
  char name[160];
  std::snprintf(name, sizeof name, "decode ctx %d q_len %d W %d splits %d", s.ctx, s.q_len, s.window, splits);
  if (rc != 0) { Check(false, std::string(name) + ": returned " + std::to_string(rc)); return; }
  const std::vector<uint16_t> got = f.OutKHost();
  const Err e = Compare(got, ref, 0, s.q_len);
  std::printf("    norm_rel %.3e max_abs %.3e (max |ref| %.3e); scratch %lld B\n", e.norm_rel, e.max_abs,
              e.max_ref, static_cast<long long>(r4d_attn_decode_h256_gqa2_scratch_bytes(&a)));
  Check(Within(e), std::string(name) + ": matches the reference kernel");

  if (!identity_report || s.q_len < 2) return;
  // Row t alone at its own position: q_len = 1, ctx' = ctx - q_len + t + 1, the query of row t.
  double worst = 0.0;
  int differing_rows = 0;
  for (int t = 0; t < s.q_len; ++t) {
    DeviceBuffer<int32_t> sq(1);
    sq.CopyFromHost(std::vector<int32_t>{s.ctx - s.q_len + t + 1});
    DeviceBuffer<uint16_t> out1(static_cast<size_t>(kQHeads) * kHeadDim);
    out1.Zero();
    R4DArgsW one = f.Args(out1.data(), f.Q() + static_cast<size_t>(t) * kQHeads * kHeadDim, 1,
                          s.ctx - s.q_len + t + 1, sq.data());
    one.splits = splits;
    if (Launch(&one, &scratch) != 0) { Check(false, std::string(name) + ": single-row launch failed"); return; }
    const std::vector<uint16_t> o1 = out1.CopyToHost();
    int diff = 0;
    for (int i = 0; i < kQHeads * kHeadDim; ++i) {
      const double d = std::abs(static_cast<double>(Bf16ToFloat(o1[i])) -
                                Bf16ToFloat(got[static_cast<size_t>(t) * kQHeads * kHeadDim + i]));
      worst = std::max(worst, d);
      diff += d != 0.0;
    }
    differing_rows += diff != 0;
  }
  std::printf("    [info] verify row vs its own q_len=1 launch: %d / %d rows differ, max |d| %.3e\n",
              differing_rows, s.q_len, worst);
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(73);

  // q_len x context sweep with the law's own split count.
  for (int q : {1, 2, 8, 16, 32}) {
    for (int ctx : {1537, 3001, 3500, 5000, 700}) Run({ctx, q, 1024, true, false}, 0, rng, q == 8 || q == 16);
  }
  // Explicit split counts.
  for (int splits : {1, 4, 16}) {
    Run({3500, 1, 1024, true, false}, splits, rng, false);
    Run({3500, 8, 1024, true, false}, splits, rng, false);
  }
  // Windows around the tile, a window longer than the context, a contiguous cache.
  for (int w : {17, 100}) Run({2500, 4, w, true, false}, 0, rng, false);
  Run({300, 4, 1024, true, false}, 0, rng, false);
  Run({1500, 1, 1024, false, false}, 0, rng, false);
  Run({1500, 8, 200, false, false}, 0, rng, false);

  // Refusals.
  {
    Shape s{300, 20, 1024, true, false};
    Fixture f(s, rng);
    DeviceBuffer<uint8_t> scratch(1 << 20);
    R4DArgsW a = f.Args(f.OutK(), f.Q(), s.q_len, s.ctx, f.Seqused());
    a.scratch = scratch.data();
    R4DArgsW b = a; b.window = 0;
    Check(r4d_attn_decode_h256_gqa2_fp8kv(&b, nullptr) == -6, "window 0 refused (-6)");
    b = a; b.q_len = 33;
    Check(r4d_attn_decode_h256_gqa2_fp8kv(&b, nullptr) == -4, "q_len 33 (66 rows) refused (-4)");
    b = a; b.scratch = nullptr;
    Check(r4d_attn_decode_h256_gqa2_fp8kv(&b, nullptr) == -5, "missing scratch refused (-5)");
    b = a; b.head_dim = 128;
    Check(r4d_attn_decode_h256_gqa2_fp8kv(&b, nullptr) == -1, "head_dim 128 refused (-1)");
  }

  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail ? 1 : 0;
}
