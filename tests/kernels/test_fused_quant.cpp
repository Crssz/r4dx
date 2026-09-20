// tests/kernels/test_fused_quant.cpp -- docs/r9700.md R2/P2 byte-diff harness, written BEFORE any
// call site was wired to use the fused epilogues (task step 2, "the byte-diff harness comes FIRST,
// before any wiring").
//
// For each producer (r4dx_rmsnorm_bf16, r4dx_residual_rmsnorm_bf16, r4dx_silu_mul_bf16) x each
// epilogue (r4dx_epilogue_f16 / _fp8_e4m3_row / _int8_fraga8) x M in {1,2,4,16,64} x K in
// {5120,6144,17408} (the task's own grid, and this model's three real hidden/intermediate sizes):
// runs the producer TWICE on the SAME seeded random bf16 input --
//   "old" path:   producer(epilogue=none) -> the existing STANDALONE quant kernel
//                 (r4dx_model_cast_bf16_to_f16 / r4dx_quant_act_fp8e4m3_row /
//                 third_party/libr4d's r4d_quant_act_i8) applied to that producer's bf16 output.
//   "fused" path: producer(epilogue=X) -- the new in-kernel epilogue.
// and asserts the two paths' bytes (and, for fp8/int8, their per-row scales) are BYTE-IDENTICAL,
// plus that the producer's own plain bf16 output is unaffected by requesting an epilogue. Per the
// task's explicit instruction, any difference is a hard failure -- this test never loosens to a
// tolerance for the quantized outputs (the plain bf16 output already has its own tolerance-checked
// coverage in test_rmsnorm.cpp/test_kernel_bandwidth.cpp; this test only compares old-vs-fused,
// which is exact-integer-byte comparable by construction since both paths run the identical
// reduction+quantize algorithm on bit-identical bf16 inputs -- see kernels.h's r4dx_epilogue doc).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/model_kernels.h"
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

int g_failures = 0;
// Real check count (review finding, 2026-09-20): main()'s final summary used to hardcode "1" in
// the PASS case regardless of how many Check() calls actually ran (135 across this grid, not 1),
// which understates the harness to a future reader as near-empty. Counted here, alongside
// g_failures, so the summary always reports the true total.
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

std::vector<uint16_t> RandomBf16(std::mt19937& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(dist(rng));
  return v;
}

// Runs the existing STANDALONE quant kernel/entry point on a device bf16 buffer, matching exactly
// what a pre-R2/P2 ApplyLinear call site would have launched.
void OldQuantF16(const DeviceBuffer<uint16_t>& in, int64_t n, DeviceBuffer<uint16_t>* out) {
  r4dx_model_cast_bf16_to_f16(reinterpret_cast<int64_t>(in.data()),
                               reinterpret_cast<int64_t>(out->data()), n, 0);
}
void OldQuantFp8(const DeviceBuffer<uint16_t>& in, int M, int K, DeviceBuffer<uint8_t>* out,
                  DeviceBuffer<float>* scale) {
  r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(in.data()),
                              reinterpret_cast<int64_t>(out->data()),
                              reinterpret_cast<int64_t>(scale->data()), M, K, 0);
}
void OldQuantI8(const DeviceBuffer<uint16_t>& in, int M, int K, DeviceBuffer<int8_t>* out,
                 DeviceBuffer<float>* scale) {
  r4d_quant_act_i8(reinterpret_cast<int64_t>(in.data()), reinterpret_cast<int64_t>(out->data()),
                    reinterpret_cast<int64_t>(scale->data()), M, K, 0);
}

// Compares one (producer, epilogue) combination at a given (M,K). `RunProducer` runs the producer
// under test with the given epilogue mode and epilogue output buffers (nullptr epilogue buffers
// when epilogue==none); it must also return the producer's own plain bf16 output for the
// bf16-unaffected-by-epilogue check.
template <typename ProducerFn>
void CheckOne(const std::string& label, int epilogue, int M, int K, ProducerFn run_producer) {
  // Baseline: epilogue=none.
  std::vector<uint16_t> base_bf16 = run_producer(r4dx_epilogue_none, nullptr, nullptr);

  // "Old" path: apply the existing standalone quant kernel to the baseline bf16 output.
  DeviceBuffer<uint16_t> base_d(base_bf16.size());
  base_d.CopyFromHost(base_bf16);

  std::vector<uint8_t> old_bytes;
  std::vector<float> old_scale;
  if (epilogue == r4dx_epilogue_f16) {
    DeviceBuffer<uint16_t> out_d(base_bf16.size());
    OldQuantF16(base_d, static_cast<int64_t>(M) * K, &out_d);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> h = out_d.CopyToHost();
    old_bytes.resize(h.size() * 2);
    std::memcpy(old_bytes.data(), h.data(), old_bytes.size());
  } else if (epilogue == r4dx_epilogue_fp8_e4m3_row) {
    DeviceBuffer<uint8_t> out_d(static_cast<size_t>(M) * K);
    DeviceBuffer<float> scale_d(static_cast<size_t>(M));
    OldQuantFp8(base_d, M, K, &out_d, &scale_d);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    old_bytes = out_d.CopyToHost();
    old_scale = scale_d.CopyToHost();
  } else if (epilogue == r4dx_epilogue_int8_fraga8) {
    DeviceBuffer<int8_t> out_d(static_cast<size_t>(M) * K);
    DeviceBuffer<float> scale_d(static_cast<size_t>(M));
    OldQuantI8(base_d, M, K, &out_d, &scale_d);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::vector<int8_t> h = out_d.CopyToHost();
    old_bytes.resize(h.size());
    std::memcpy(old_bytes.data(), h.data(), h.size());
    old_scale = scale_d.CopyToHost();
  }

  // "Fused" path: producer's own epilogue.
  int elem_size = (epilogue == r4dx_epilogue_f16) ? 2 : 1;
  DeviceBuffer<uint8_t> fused_out_d(static_cast<size_t>(M) * K * elem_size);
  DeviceBuffer<float> fused_scale_d(static_cast<size_t>(M));
  std::vector<uint16_t> fused_bf16 =
      run_producer(epilogue, fused_out_d.data(), fused_scale_d.data());
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  // The plain bf16 output must be bit-identical whether or not an epilogue was requested.
  Check(fused_bf16 == base_bf16, label + ": bf16 output changed when epilogue was requested");

  std::vector<uint8_t> fused_bytes = fused_out_d.CopyToHost();
  Check(fused_bytes == old_bytes, label + ": epilogue bytes differ from old (standalone-kernel) path");

  if (epilogue != r4dx_epilogue_f16) {
    std::vector<float> fused_scale = fused_scale_d.CopyToHost();
    bool scale_ok = (fused_scale.size() == old_scale.size());
    if (scale_ok) {
      for (size_t i = 0; i < fused_scale.size(); ++i) {
        // Exact float compare: both paths compute scale = fmaxf(absmax,1e-8f)/divisor from the
        // SAME bf16 row via the SAME reduction -- must be bit-identical, not merely close.
        if (std::memcmp(&fused_scale[i], &old_scale[i], sizeof(float)) != 0) {
          scale_ok = false;
          break;
        }
      }
    }
    Check(scale_ok, label + ": epilogue scale differs from old (standalone-kernel) path");
  }

  if (g_failures == 0) {
    std::printf("  ok: %s (M=%d,K=%d)\n", label.c_str(), M, K);
  }
}

// ArenaAlignmentInvariant -- Milestone 4 follow-up (docs/status.md's R2/P2 incident, root-caused
// this pass): the ORIGINAL 135-check grid above never caught the real bug because every buffer it
// allocates is its own isolated DeviceBuffer (each a fresh hipMalloc, which happens to come back
// >=256-byte aligned) -- it never exercises r4dx::core::Arena, the shared bump allocator every real
// model call site (GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward) actually carves its
// epilogue scratch from. The real bug was an Arena::Alloc gap: it aligned each call's START but not
// its END, so a `T`-float per-row scale scratch (T = chunk row count; not a multiple of 4 at decode
// T=1 or MTP verify T=2..4) left the bump offset a few bytes short of 16 bytes, and every
// allocation AFTER it in the same layer inherited that drift -- exactly the allocation sequence
// GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward use (an epilogue's uint8 data buffer,
// explicitly 16-aligned; its float[T] scale scratch, default-aligned; then the layer's own next
// default-aligned buffer). This reproduces that exact sequence directly, host-side, for every T in
// 1..64 (decode, MTP verify, and every prefill chunk width), asserting every subsequent pointer is
// 16-byte aligned -- independent of any GPU kernel, so it fails immediately on a regression to the
// pre-fix Arena::Alloc rather than needing a real-hardware generated-text divergence to notice it.
void CheckArenaAlignmentInvariant() {
  r4dx::core::Arena arena(4 << 20);  // 4 MiB, plenty for this synthetic sequence (max ~1 MiB used).
  for (int64_t T = 1; T <= 64; ++T) {
    arena.Reset();
    // Mirrors gdn_layer.cpp/attention_layer.hpp/mlp.cpp's real sequence for a hidden=5120 block:
    // (1) the epilogue's quantized-activation data buffer, explicitly 16-aligned (int8/fp8 element,
    //     T*hidden bytes -- always a multiple of 16 since hidden=5120 is);
    // (2) the epilogue's per-row fp32 scale scratch, T floats, DEFAULT-aligned (the buffer whose
    //     size is NOT always a multiple of 16 -- this is the one that mattered);
    // (3) the layer's own next default-aligned buffer (e.g. mixed_qkv, a uint16_t buffer) -- this
    //     is the one a downstream libr4d kernel reads with an unchecked wide load.
    constexpr int64_t kHidden = 5120;
    uint8_t* data_buf = arena.Alloc<uint8_t>(static_cast<size_t>(T * kHidden), /*align_bytes=*/16);
    float* scale_buf = arena.Alloc<float>(static_cast<size_t>(T));
    uint16_t* next_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * kHidden));
    // A second round in the same "layer" (mirrors Mlp::Forward's gate_up-epilogue THEN
    // down-epilogue, two fused scale scratches per layer, not just one) to catch cumulative drift
    // across more than one odd-sized allocation before Reset().
    float* scale_buf2 = arena.Alloc<float>(static_cast<size_t>(T));
    uint16_t* next_buf2 = arena.Alloc<uint16_t>(static_cast<size_t>(3 * T));

    auto aligned16 = [](const void* p) {
      return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
    };
    Check(aligned16(data_buf), "ArenaAlignmentInvariant: data_buf not 16-aligned, T=" + std::to_string(T));
    Check(aligned16(next_buf), "ArenaAlignmentInvariant: next_buf not 16-aligned (post scale[T]), T=" + std::to_string(T));
    Check(aligned16(next_buf2), "ArenaAlignmentInvariant: next_buf2 not 16-aligned (post 2nd scale[T]), T=" + std::to_string(T));
    (void)scale_buf;
    (void)scale_buf2;
  }
  if (g_failures == 0) {
    std::printf("  ok: ArenaAlignmentInvariant (T=1..64)\n");
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(1234);
  const float eps = 1e-6f;

  CheckArenaAlignmentInvariant();

  const int Ms[] = {1, 2, 4, 16, 64};
  const int64_t Ks[] = {5120, 6144, 17408};
  const int epilogues[] = {r4dx_epilogue_f16, r4dx_epilogue_fp8_e4m3_row,
                            r4dx_epilogue_int8_fraga8};
  const char* epilogue_names[] = {"", "f16", "fp8_e4m3_row", "int8_fraga8"};

  for (int M : Ms) {
    for (int64_t K : Ks) {
      std::vector<uint16_t> x_h = RandomBf16(rng, static_cast<size_t>(M) * K, -3.0f, 3.0f);
      std::vector<uint16_t> w_h = RandomBf16(rng, static_cast<size_t>(K), -0.2f, 0.2f);
      std::vector<uint16_t> resid_h = RandomBf16(rng, static_cast<size_t>(M) * K, -3.0f, 3.0f);
      DeviceBuffer<uint16_t> x_d(x_h.size()), w_d(w_h.size()), resid_d(resid_h.size());
      x_d.CopyFromHost(x_h);
      w_d.CopyFromHost(w_h);
      resid_d.CopyFromHost(resid_h);

      for (int epilogue : epilogues) {
        const char* en = epilogue_names[epilogue];

        // ---- rmsnorm ----
        CheckOne(std::string("rmsnorm/") + en, epilogue, M, static_cast<int>(K),
                 [&](int ep, void* eo, float* es) {
                   DeviceBuffer<uint16_t> out_d(x_h.size());
                   r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x_d.data()),
                                      reinterpret_cast<int64_t>(w_d.data()),
                                      reinterpret_cast<int64_t>(out_d.data()), M, K, eps, 0, ep,
                                      reinterpret_cast<int64_t>(eo),
                                      reinterpret_cast<int64_t>(es));
                   R4DX_HIP_CHECK(hipDeviceSynchronize());
                   return out_d.CopyToHost();
                 });

        // ---- residual_rmsnorm (epilogue applies to out_normed only) ----
        CheckOne(std::string("residual_rmsnorm/") + en, epilogue, M, static_cast<int>(K),
                 [&](int ep, void* eo, float* es) {
                   DeviceBuffer<uint16_t> out_resid_d(x_h.size()), out_norm_d(x_h.size());
                   r4dx_residual_rmsnorm_bf16(
                       reinterpret_cast<int64_t>(x_d.data()),
                       reinterpret_cast<int64_t>(resid_d.data()),
                       reinterpret_cast<int64_t>(w_d.data()),
                       reinterpret_cast<int64_t>(out_resid_d.data()),
                       reinterpret_cast<int64_t>(out_norm_d.data()), M, K, eps, 0, ep,
                       reinterpret_cast<int64_t>(eo), reinterpret_cast<int64_t>(es));
                   R4DX_HIP_CHECK(hipDeviceSynchronize());
                   return out_norm_d.CopyToHost();
                 });

        // ---- silu_mul (gate_up: [M, 2K], contiguous) ----
        std::vector<uint16_t> gate_up_h = RandomBf16(rng, static_cast<size_t>(M) * 2 * K, -3.0f, 3.0f);
        DeviceBuffer<uint16_t> gate_up_d(gate_up_h.size());
        gate_up_d.CopyFromHost(gate_up_h);
        CheckOne(std::string("silu_mul/") + en, epilogue, M, static_cast<int>(K),
                 [&](int ep, void* eo, float* es) {
                   DeviceBuffer<uint16_t> out_d(static_cast<size_t>(M) * K);
                   r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up_d.data()),
                                       reinterpret_cast<int64_t>(out_d.data()), M, K,
                                       /*in_row_stride=*/2 * K, 0, ep,
                                       reinterpret_cast<int64_t>(eo),
                                       reinterpret_cast<int64_t>(es));
                   R4DX_HIP_CHECK(hipDeviceSynchronize());
                   return out_d.CopyToHost();
                 });
      }
    }
  }

  std::printf(g_failures == 0 ? "PASS (%d checks)\n" : "FAIL (%d/%d checks failing)\n",
              g_failures == 0 ? g_checks : g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
