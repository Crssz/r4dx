// tests/kernels/test_kv_write.cpp -- r4dx_kv_write_paged_fp8_hnd against a CPU reference that
// reads back the same slots via the exact R4DArgs.kv addressing (docs/architecture.md "fp8 KV
// paging"): (num_blocks, kv_heads, block_size, 2*head_dim), K at [0,head_dim), V at
// [head_dim,2*head_dim), block_id = slot/block_size, offset = slot % block_size.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int T = 40, kv_heads = 4, head_dim = 256, block_size = 16;
  const int num_blocks = 4;  // 4*16 = 64 total slots, a bijective permutation of which covers T=40
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * head_dim;
  const int64_t kv_block_stride = static_cast<int64_t>(kv_heads) * kv_head_stride;

  std::mt19937 rng(29);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  std::uniform_real_distribution<float> descale_dist(0.5f, 2.0f);

  std::vector<uint16_t> k_h(static_cast<size_t>(T) * kv_heads * head_dim);
  std::vector<uint16_t> v_h(static_cast<size_t>(T) * kv_heads * head_dim);
  for (auto& v : k_h) v = FloatToBf16(dist(rng));
  for (auto& v : v_h) v = FloatToBf16(dist(rng));

  std::vector<float> k_descale(kv_heads), v_descale(kv_heads);
  for (auto& d : k_descale) d = descale_dist(rng);
  for (auto& d : v_descale) d = descale_dist(rng);

  // Scatter tokens across blocks/offsets rather than a trivial contiguous mapping, to exercise
  // the block_id/offset split -- a single affine permutation over the flat slot space (37 is
  // coprime with num_blocks*block_size=64, so this is a bijection over [0,64) and, since T=40 <
  // 64, collision-free) rather than two independently-periodic mod sequences, which would
  // silently alias distinct tokens onto the same slot.
  std::vector<int32_t> slot_mapping(T);
  const int total_slots = num_blocks * block_size;
  for (int t = 0; t < T; ++t) {
    slot_mapping[t] = (t * 37 + 5) % total_slots;
  }

  DeviceBuffer<uint16_t> k_d(k_h.size()), v_d(v_h.size());
  DeviceBuffer<int32_t> slot_d(T);
  DeviceBuffer<float> kd_d(kv_heads), vd_d(kv_heads);
  DeviceBuffer<uint8_t> cache_d(static_cast<size_t>(num_blocks) * kv_heads * block_size * 2 *
                                 head_dim);
  k_d.CopyFromHost(k_h);
  v_d.CopyFromHost(v_h);
  slot_d.CopyFromHost(slot_mapping);
  kd_d.CopyFromHost(k_descale);
  vd_d.CopyFromHost(v_descale);
  cache_d.Zero();

  r4dx_kv_write_paged_fp8_hnd(reinterpret_cast<int64_t>(k_d.data()),
                               reinterpret_cast<int64_t>(v_d.data()),
                               reinterpret_cast<int64_t>(slot_d.data()),
                               reinterpret_cast<int64_t>(kd_d.data()),
                               reinterpret_cast<int64_t>(vd_d.data()),
                               reinterpret_cast<int64_t>(cache_d.data()), T, kv_heads, head_dim,
                               block_size, kv_block_stride, kv_head_stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint8_t> cache_h = cache_d.CopyToHost();

  // fp8 e4m3 has 3 mantissa bits, so round-to-nearest's worst-case RELATIVE error is ~1/16 of a
  // value near the top of its binade -- but for a value close to zero the quantization step is
  // large relative to the value itself, which is an expected property of a floating-point
  // format, not a kv_write bug. The max-relative check below only looks at values with enough
  // magnitude for "relative" to be a meaningful worst case; the norm check covers the whole
  // tensor including the near-zero entries.
  double max_rel = 0.0;
  double norm_num = 0.0, norm_den = 0.0;
  for (int t = 0; t < T; ++t) {
    int slot = slot_mapping[t];
    int block_id = slot / block_size;
    int offset = slot % block_size;
    for (int h = 0; h < kv_heads; ++h) {
      int64_t base = static_cast<int64_t>(block_id) * kv_block_stride +
                      static_cast<int64_t>(h) * kv_head_stride +
                      static_cast<int64_t>(offset) * 2 * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        float k_ref = Bf16ToFloat(k_h[(static_cast<size_t>(t) * kv_heads + h) * head_dim + d]);
        float v_ref = Bf16ToFloat(v_h[(static_cast<size_t>(t) * kv_heads + h) * head_dim + d]);
        float k_dequant = Fp8E4M3ToFloat(cache_h[base + d]) * k_descale[h];
        float v_dequant = Fp8E4M3ToFloat(cache_h[base + head_dim + d]) * v_descale[h];
        if (std::abs(k_ref) > 0.2f) {
          max_rel = std::max(max_rel, static_cast<double>(std::abs(k_dequant - k_ref) / std::abs(k_ref)));
        }
        if (std::abs(v_ref) > 0.2f) {
          max_rel = std::max(max_rel, static_cast<double>(std::abs(v_dequant - v_ref) / std::abs(v_ref)));
        }
        norm_num += static_cast<double>(k_dequant - k_ref) * (k_dequant - k_ref) +
                    static_cast<double>(v_dequant - v_ref) * (v_dequant - v_ref);
        norm_den += static_cast<double>(k_ref) * k_ref + static_cast<double>(v_ref) * v_ref;
      }
    }
  }
  double norm_rel = std::sqrt(norm_num) / std::max(1e-9, std::sqrt(norm_den));
  std::printf("kv_write_paged_fp8_hnd dequant max rel err (|ref|>0.2)=%.4e, norm rel err=%.4e\n",
              max_rel, norm_rel);
  // fp8 e4m3 round-to-nearest's theoretical worst-case relative error is 1/16 = 0.0625 (half an
  // ULP at the coarsest exponent step); 0.07 leaves a small margin for sampled points landing
  // near that worst case while still actually gating the rounding mode (the previous 0.15 bound
  // was ~2.4x looser than the theoretical worst case, so a half-ulp-biased rounding bug would
  // still have passed).
  bool ok = max_rel < 0.07 && norm_rel < 0.1;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
