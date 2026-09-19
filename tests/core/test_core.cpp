// tests/core/test_core.cpp -- covers src/core: dtype round-trips, DeviceBuffer/PinnedBuffer/
// Stream/Arena RAII, TensorView, and the r4d.hpp wrapper's geometry queries + error-to-exception
// path. Runs on HIP device 1 (HIP_VISIBLE_DEVICES=1, set by tests/run_tests.ps1, so device index
// 0 here IS physical device 1, same convention as tests/smoke_r4d.cpp).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tensor.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;

void Check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    g_failures++;
  } else {
    std::printf("PASS: %s\n", what);
  }
}

void TestDtypeRoundTrips() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  double bf16_max_rel = 0.0, f16_max_rel = 0.0, fp8_max_rel = 0.0;
  for (int i = 0; i < 10000; ++i) {
    float f = dist(rng);
    float bf = Bf16ToFloat(FloatToBf16(f));
    float hf = F16ToFloat(FloatToF16(f));
    double rel_bf = std::abs(static_cast<double>(bf) - f) / std::max(1e-6, std::abs((double)f));
    double rel_hf = std::abs(static_cast<double>(hf) - f) / std::max(1e-6, std::abs((double)f));
    bf16_max_rel = std::max(bf16_max_rel, rel_bf);
    f16_max_rel = std::max(f16_max_rel, rel_hf);
  }
  // fp8 e4m3 activation range in r4dx is post-per-row-scale, i.e. values in roughly [-448, 448]
  // with 3 mantissa bits -- worst-case rounding error is 1/16 of the value (half an ULP at the
  // coarsest exponent step), so this loop scales into that operating range.
  std::uniform_real_distribution<float> fp8_dist(-448.0f, 448.0f);
  for (int i = 0; i < 10000; ++i) {
    float f = fp8_dist(rng);
    if (std::abs(f) < 0.02f) continue;  // subnormal range, dominated by absolute not relative err
    float g = Fp8E4M3ToFloat(FloatToFp8E4M3(f));
    double rel = std::abs(static_cast<double>(g) - f) / std::abs((double)f);
    fp8_max_rel = std::max(fp8_max_rel, rel);
  }
  std::printf("  bf16 max rel err=%.4e  f16 max rel err=%.4e  fp8e4m3 max rel err=%.4e\n",
              bf16_max_rel, f16_max_rel, fp8_max_rel);
  // bf16 has 8 significant bits (1 implicit + 7 explicit); round-to-nearest's worst-case
  // relative error is 2^-8 (half an ULP at the coarsest step within a binade), with a small
  // margin for the sampled points landing near that worst case.
  Check(bf16_max_rel < 1.1 / 256.0, "bf16 round-trip within 8-bit-mantissa tolerance");
  Check(f16_max_rel < 0.6 / 1024.0, "f16 round-trip within 10-bit-mantissa tolerance");
  Check(fp8_max_rel < 0.5 / 8.0 + 1e-6, "fp8e4m3 round-trip within 3-bit-mantissa tolerance");

  // Known exact values.
  Check(Bf16ToFloat(FloatToBf16(0.0f)) == 0.0f, "bf16(0) == 0");
  Check(Bf16ToFloat(FloatToBf16(1.0f)) == 1.0f, "bf16(1) == 1 (exact in bf16)");
  Check(Fp8E4M3ToFloat(FloatToFp8E4M3(1.0f)) == 1.0f, "fp8e4m3(1) == 1 (exact)");
  Check(Fp8E4M3ToFloat(FloatToFp8E4M3(500.0f)) == 448.0f, "fp8e4m3 saturates at 448");
}

void TestTensorView() {
  std::vector<float> storage(2 * 3 * 4, 0.0f);
  TensorView t = TensorView::Contiguous(storage.data(), Dtype::kF32, {2, 3, 4});
  Check(t.ndim == 3, "TensorView ndim");
  Check(t.ElemCount() == 24, "TensorView element count");
  Check(t.strides[0] == 12 && t.strides[1] == 4 && t.strides[2] == 1,
        "TensorView row-major strides");
  Check(t.IsContiguous(), "TensorView reports contiguous");
  Check(t.ByteSize() == 24 * 4, "TensorView byte size");
}

void TestArena() {
  Arena arena(1024);
  float* a = arena.Alloc<float>(10);
  int32_t* b = arena.Alloc<int32_t>(5);
  Check(reinterpret_cast<uint8_t*>(b) >= reinterpret_cast<uint8_t*>(a) + 10 * sizeof(float),
        "Arena bump-allocates without overlap");
  arena.Reset();
  Check(arena.used_bytes() == 0, "Arena::Reset reclaims everything");
  bool threw = false;
  try {
    arena.Alloc<float>(1'000'000);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  Check(threw, "Arena throws bad_alloc when exhausted");
}

void TestDeviceAndPinnedBuffers() {
  std::vector<float> host_in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  DeviceBuffer<float> dbuf(host_in.size());
  dbuf.CopyFromHost(host_in);
  std::vector<float> host_out = dbuf.CopyToHost();
  Check(host_out == host_in, "DeviceBuffer round-trips a host vector");

  PinnedBuffer<uint16_t> pinned(4);
  for (size_t i = 0; i < pinned.size(); ++i) pinned[i] = static_cast<uint16_t>(i * 7);
  bool ok = true;
  for (size_t i = 0; i < pinned.size(); ++i) ok &= (pinned[i] == static_cast<uint16_t>(i * 7));
  Check(ok, "PinnedBuffer read/write");

  Stream stream;
  DeviceBuffer<float> dbuf2(host_in.size());
  dbuf2.CopyFromHostAsync(host_in.data(), host_in.size(), stream);
  stream.Synchronize();
  std::vector<float> host_out2 = dbuf2.CopyToHost();
  Check(host_out2 == host_in, "DeviceBuffer async H2D via r4dx::core::Stream");
}

void TestR4dGeometryAndErrors() {
  r4d::AttnDims ad = r4d::GetAttnDims();
  std::printf("  r4d attn dims: head_dim=%d gqa=%d block_size=%d max_decode_rows=%d\n",
              ad.head_dim, ad.gqa, ad.block_size, ad.max_decode_rows);
  Check(ad.head_dim == 256 && ad.gqa == 6 && ad.block_size == 16, "r4d::GetAttnDims() geometry");

  r4d::GdnDims gd = r4d::GetGdnDims();
  std::printf("  r4d gdn dims: head_k=%d head_v=%d chunk=%d\n", gd.head_k, gd.head_v, gd.chunk);
  Check(gd.head_k == 128 && gd.head_v == 128 && gd.chunk == 64, "r4d::GetGdnDims() geometry");

  Check(r4d::KernelCount() > 0, "r4d::KernelCount() reports a non-empty registry");

  // Negative path: an attn decode call with a head_dim this build was not compiled for must
  // surface as r4dx::core::R4dError, not a silent wrong answer or a crash.
  R4DArgs bad{};
  bad.num_seqs = 1;
  bad.q_len = 1;
  bad.q_heads = 4;
  bad.kv_heads = 4;
  bad.head_dim = 128;  // wrong: this build only serves head_dim 256
  bad.block_size = 16;
  bad.max_blocks = 1;
  bool threw = false;
  try {
    r4d::AttnDecodeFp8Kv(bad, nullptr);
  } catch (const R4dError& e) {
    threw = true;
    std::printf("  got expected R4dError: %s\n", e.what());
  }
  Check(threw, "r4d::AttnDecodeFp8Kv on an unsupported shape throws R4dError");
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));  // HIP_VISIBLE_DEVICES=1 -> local index 0 is physical device 1

  std::printf("-- dtype round-trips --\n");
  TestDtypeRoundTrips();
  std::printf("-- TensorView --\n");
  TestTensorView();
  std::printf("-- Arena --\n");
  TestArena();
  std::printf("-- DeviceBuffer / PinnedBuffer / Stream --\n");
  TestDeviceAndPinnedBuffers();
  std::printf("-- r4d.hpp geometry + error wrapping --\n");
  TestR4dGeometryAndErrors();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
