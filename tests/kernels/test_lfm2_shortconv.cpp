// tests/kernels/test_lfm2_shortconv.cpp -- r4dx_lfm2_shortconv_bf16 (docs/lfm2-plan.md 3.3, task L1-7) against
// lfm2_ref.hpp's ShortConv, bit for bit (outputs AND the updated state):
//   * T = 1, 2, 3, 15, 16, 17, 63, 64, 255, 256 at H = 2048 (hidden) and 1024 (a TP=2 rank), odd H 100 / 130,
//     zero state and a random state, strided bcx rows (stride > 3H);
//   * chunk equivalence on the device: 600 rows as one launch == 256 + 256 + 88 == 600 launches of T = 1
//     (output bytes and final state bytes), and all of them == the CPU reference;
//   * decode after prefill: a 300-row prefill, then 8 single-token steps, vs the reference over 308 rows;
//   * the state micro-kernel path (T > 16, two launches) and the fused one-launch path (T <= 16) agree;
//   * nothing is written outside out [T, H] and state [2, H] (canaries); precondition throws.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt); WRITTEN, NOT YET RUN. Built with /EHc-.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_test_util.hpp"
#include "r4dx/kernels/lfm2_kernels.h"

using namespace r4dx::core;
using namespace lfm2_test;

namespace {

constexpr uint16_t kCanary = 0x7F31;  // a bf16 NaN pattern no kernel result equals

struct Run {
  std::vector<float> out, state;
};

// Feed `n` rows of bcx (stride) starting at row r0 as one launch, state in/out on the device.
void Launch(const DeviceBuffer<uint16_t>& bcx_d, int r0, int n, int64_t stride, DeviceBuffer<uint16_t>& st_d,
            const DeviceBuffer<uint16_t>& w_d, DeviceBuffer<uint16_t>& out_d, int H) {
  r4dx_lfm2_shortconv_bf16(P(bcx_d.data() + static_cast<size_t>(r0) * stride), P(st_d.data()), P(w_d.data()),
                           P(out_d.data() + static_cast<size_t>(r0) * H), n, H, stride, 0);
}

Run OnDevice(const std::vector<float>& bcx, const std::vector<float>& w, const std::vector<float>& init, int T,
             int H, int64_t stride, const std::vector<int>& chunks) {
  DeviceBuffer<uint16_t> bcx_d(bcx.size()), w_d(w.size()), st_d(init.size()), out_d(static_cast<size_t>(T) * H);
  bcx_d.CopyFromHost(ToBits(bcx));
  w_d.CopyFromHost(ToBits(w));
  st_d.CopyFromHost(ToBits(init));
  int r0 = 0;
  for (int n : chunks) {
    Launch(bcx_d, r0, n, stride, st_d, w_d, out_d, H);
    r0 += n;
  }
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return {FromBits(out_d.CopyToHost()), FromBits(st_d.CopyToHost())};
}

void TestSize(std::mt19937& rng, int T, int H, bool zero_state, int64_t stride_extra) {
  const int64_t stride = 3 * static_cast<int64_t>(H) + stride_extra;
  const auto bcx = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  const auto w = RandBf16(rng, static_cast<size_t>(H) * 3, 1.0f);
  const std::vector<float> init = zero_state ? std::vector<float>(2 * static_cast<size_t>(H), 0.0f)
                                             : RandBf16(rng, 2 * static_cast<size_t>(H), 2.0f);
  std::vector<float> st = init;
  const auto want = lfm2_ref::ShortConv(bcx, &st, w, T, H, stride);
  const Run got = OnDevice(bcx, w, init, T, H, stride, {T});
  const Diff d = Compare(got.out, want, 0.0f);
  const Diff ds = Compare(got.state, st, 0.0f);
  CheckF(d.differ == 0 && ds.differ == 0,
         "shortconv T=" + std::to_string(T) + " H=" + std::to_string(H) + (zero_state ? " zero state" : " random state") +
             " stride+" + std::to_string(stride_extra) + ": output and state == reference (bitwise)");
}

void TestChunks(std::mt19937& rng) {
  const int H = 2048, T = 600;
  const int64_t stride = 3 * static_cast<int64_t>(H);
  const auto bcx = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  const auto w = RandBf16(rng, static_cast<size_t>(H) * 3, 1.0f);
  const std::vector<float> zero(2 * static_cast<size_t>(H), 0.0f);
  std::vector<float> st = zero;
  const auto want = lfm2_ref::ShortConv(bcx, &st, w, T, H, stride);

  const Run one = OnDevice(bcx, w, zero, T, H, stride, {T});
  const Run three = OnDevice(bcx, w, zero, T, H, stride, {256, 256, 88});
  const Run ones = OnDevice(bcx, w, zero, T, H, stride, std::vector<int>(T, 1));
  Check(one.out == want && one.state == st, "shortconv 600 rows, one launch == reference (bitwise)");
  Check(three.out == one.out && three.state == one.state, "shortconv 256 + 256 + 88 == one pass (output and state, bitwise)");
  Check(ones.out == one.out && ones.state == one.state, "shortconv 600 x T=1 == one pass (output and state, bitwise)");
  const Run odd = OnDevice(bcx, w, zero, T, H, stride, {1, 15, 16, 17, 2, 3, 63, 64, 255, 1, 163});
  Check(odd.out == one.out && odd.state == one.state, "shortconv uneven split around the 16-row tile == one pass (bitwise)");

  // decode after prefill: 300-row prefill then 8 single-token steps
  const Run dec = OnDevice(bcx, w, zero, 308, H, stride, [] {
    std::vector<int> c = {300};
    for (int i = 0; i < 8; ++i) c.push_back(1);
    return c;
  }());
  std::vector<float> st308 = zero;
  const auto want308 = lfm2_ref::ShortConv(std::vector<float>(bcx.begin(), bcx.begin() + 308 * stride), &st308, w, 308, H, stride);
  Check(dec.out == want308 && dec.state == st308, "shortconv decode after a 300-row prefill (8 x T=1) == reference (bitwise)");
}

void TestCanaries(std::mt19937& rng) {
  const int H = 130, T = 40;
  const int64_t stride = 3 * H;
  const auto bcx = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  const auto w = RandBf16(rng, static_cast<size_t>(H) * 3, 1.0f);
  const size_t pad = 64;
  std::vector<uint16_t> out_h(static_cast<size_t>(T) * H + 2 * pad, kCanary), st_h(2 * static_cast<size_t>(H) + 2 * pad, kCanary);
  const auto st0 = RandBf16(rng, 2 * static_cast<size_t>(H), 1.0f);
  const auto st0_bits = ToBits(st0);
  std::copy(st0_bits.begin(), st0_bits.end(), st_h.begin() + pad);
  DeviceBuffer<uint16_t> bcx_d(bcx.size()), w_d(w.size()), st_d(st_h.size()), out_d(out_h.size());
  bcx_d.CopyFromHost(ToBits(bcx));
  w_d.CopyFromHost(ToBits(w));
  st_d.CopyFromHost(st_h);
  out_d.CopyFromHost(out_h);
  r4dx_lfm2_shortconv_bf16(P(bcx_d.data()), P(st_d.data() + pad), P(w_d.data()), P(out_d.data() + pad), T, H, stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto out_g = out_d.CopyToHost();
  const auto st_g = st_d.CopyToHost();
  bool intact = true;
  for (size_t i = 0; i < pad; ++i) {
    intact &= out_g[i] == kCanary && out_g[out_g.size() - 1 - i] == kCanary;
    intact &= st_g[i] == kCanary && st_g[st_g.size() - 1 - i] == kCanary;
  }
  Check(intact, "shortconv writes nothing outside out [T, H] and state [2, H] (odd H = 130, T = 40)");
}

void TestThrows() {
  DeviceBuffer<uint16_t> a(64);
  bool threw = false;
  try {
    r4dx_lfm2_shortconv_bf16(P(a.data()), P(a.data()), P(a.data()), P(a.data()), 1, 8, 16, 0);  // stride < 3H
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "shortconv: bcx_row_stride < 3H throws");
  threw = false;
  try {
    r4dx_lfm2_shortconv_bf16(P(a.data()), P(a.data()), P(a.data()), P(a.data()), 1, 0, 0, 0);  // H = 0
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Check(threw, "shortconv: H = 0 throws");
  r4dx_lfm2_shortconv_bf16(0, 0, 0, 0, 0, 8, 24, 0);  // T = 0 returns without touching anything
  Check(true, "shortconv: T = 0 is a no-op");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(2026);
  for (int H : {2048, 1024}) {
    for (int T : {1, 2, 3, 15, 16, 17, 63, 64, 255, 256}) {
      TestSize(rng, T, H, true, 0);
      TestSize(rng, T, H, false, 0);
    }
  }
  for (int T : {1, 2, 17, 256}) TestSize(rng, T, 100, false, 0);  // odd widths: partial wave / block
  for (int T : {1, 5, 64}) TestSize(rng, T, 130, false, 16);
  TestSize(rng, 256, 2048, false, 64);  // strided rows
  TestChunks(rng);
  TestCanaries(rng);
  TestThrows();
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
