// tests/model/attention/test_mrope_attn_layer.cpp -- one real full-attention decoder layer driven
// by 3-AXIS mrope position ids, against tools/reference/mrope_layer_golden.py's capture of the
// SAME layer with the SAME real weights (docs/vision.md "Text-side splicing").
//
// This is the layer-level numeric check the vision milestone's splicing stage needs and that
// tests/model/attention/test_attn_layer.cpp structurally cannot provide: that golden's position
// ids collapse (t,h,w) to one sequential index, so it passes identically whether the rope kernel
// selects a per-bin position stream or ignores the h/w rows entirely. Here the golden's rows come
// from the real `Qwen3_5Model.get_rope_index` for a prompt with a 10x16-patch image in it, so the
// three streams genuinely differ across the image's 40 merged tokens and the whole tail of the
// prompt ropes at `sequence index + delta` rather than at its own index.
//
// A full bf16 27B reference does not fit on this card, so one layer is the largest piece of the
// real stack that can be compared numerically at all; the end-to-end behaviour check is
// tests/vision/tool_vision_chat (docs/vision.md).
//
// TOLERANCE: identical reasoning, and identical value, to test_attn_layer.cpp's own -- this
// component always writes through the fp8 e4m3 paged KV cache while the golden ran bf16 K/V, so
// the bound is that repo-standard 2e-2 bf16_matmul_rel_err, not a rope-specific number. A rope
// that read the wrong position stream is not a 2e-2 effect: the h and w rows of this golden differ
// from the t row by up to 14 positions on the image rows and the whole post-image tail sits 32
// positions away from its sequence index, both of which move the rotation by radians, not ulps.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "../test_common.h"  // r4dx_test::FileExists / SkipMissing (exit 77 = CTest SKIPPED)

#ifndef R4DX_GOLDEN_MROPE_PATH
#define R4DX_GOLDEN_MROPE_PATH "tools/reference/golden_out/mrope_layer_003.safetensors"
#endif
#ifndef R4DX_BF16_CONTAINER_PATH
#define R4DX_BF16_CONTAINER_PATH "D:/models/r4dx/qwen38-27b-l4-bf16.r4dx"
#endif

using namespace r4dx::core;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;
using r4dx::model::attention::AttentionLayer;
using r4dx::model::attention::AttnConfig;
using r4dx::model::attention::AttnWeights;
using r4dx::model::attention::PagedKvCache;
using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

namespace {

// docs/container-format.md: a container bf16 tensor is stored as U8 with a trailing byte-pair
// axis, so ElemCount() is BYTES. A golden written by tools/reference/** carries a real BF16 dtype.
std::vector<uint16_t> ReadContainerBf16(const SafetensorsReader& r, const std::string& name,
                                         int64_t expected_elems) {
  const int64_t elems = r.Meta(name).ElemCount() / 2;
  if (elems != expected_elems) {
    std::fprintf(stderr, "ReadContainerBf16(%s): expected %lld elements, container has %lld\n",
                 name.c_str(), static_cast<long long>(expected_elems),
                 static_cast<long long>(elems));
    std::exit(1);
  }
  const uint16_t* p = reinterpret_cast<const uint16_t*>(r.Data(name));
  return std::vector<uint16_t>(p, p + elems);
}

std::vector<uint16_t> ReadGoldenBf16(const SafetensorsReader& r, const std::string& name,
                                      int64_t expected_elems) {
  const auto& meta = r.Meta(name);
  if (meta.dtype != "BF16" || meta.ElemCount() != expected_elems) {
    std::fprintf(stderr, "ReadGoldenBf16(%s): expected BF16[%lld], golden has %s[%lld]\n",
                 name.c_str(), static_cast<long long>(expected_elems), meta.dtype.c_str(),
                 static_cast<long long>(meta.ElemCount()));
    std::exit(1);
  }
  const uint16_t* p = reinterpret_cast<const uint16_t*>(r.Data(name));
  return std::vector<uint16_t>(p, p + expected_elems);
}

std::vector<int64_t> ReadGoldenI64(const SafetensorsReader& r, const std::string& name) {
  const auto& meta = r.Meta(name);
  if (meta.dtype != "I64") {
    std::fprintf(stderr, "ReadGoldenI64(%s): expected I64, golden has %s\n", name.c_str(),
                 meta.dtype.c_str());
    std::exit(1);
  }
  const int64_t* p = reinterpret_cast<const int64_t*>(r.Data(name));
  return std::vector<int64_t>(p, p + meta.ElemCount());
}

std::vector<float> ReadFp32Flat(const SafetensorsReader& r, const std::string& name,
                                 int64_t expected_elems) {
  const float* p = reinterpret_cast<const float*>(r.Data(name));
  return std::vector<float>(p, p + expected_elems);
}

template <typename T>
DeviceBuffer<T> Upload(const std::vector<T>& h) {
  DeviceBuffer<T> d(h.size());
  d.CopyFromHost(h);
  return d;
}

double NormRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = Bf16ToFloat(got[i]), r = Bf16ToFloat(ref[i]);
    num += (g - r) * (g - r);
    den += r * r;
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

int Run() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  SafetensorsReader golden(Utf8ToWide(R4DX_GOLDEN_MROPE_PATH));
  SafetensorsReader container(Utf8ToWide(R4DX_BF16_CONTAINER_PATH));

  AttnConfig cfg;  // defaults match Qwen3.8-27B's text_config, incl. mrope_section [11,11,10]
  const int hidden = cfg.hidden, H = cfg.num_heads, Hkv = cfg.kv_heads, D = cfg.head_dim;

  // ---- golden shapes, read from the golden itself rather than restated here -------------------
  const std::vector<int64_t> pos_rows = ReadGoldenI64(golden, "mrope_position_ids");
  const int64_t total_len = static_cast<int64_t>(pos_rows.size()) / 3;
  const int64_t prefill_len = static_cast<int64_t>(
      golden.Meta("prefill_hidden_states").ElemCount() / hidden);
  const int64_t decode_len = total_len - prefill_len;
  const std::vector<int64_t> mm_ids_i64 = [&] {
    const auto& meta = golden.Meta("mm_token_type_ids");
    const int32_t* p = reinterpret_cast<const int32_t*>(golden.Data("mm_token_type_ids"));
    return std::vector<int64_t>(p, p + meta.ElemCount());
  }();

  // The property the whole stage rests on, asserted against the reference's own rows rather than
  // recomputed: on this golden the three axes genuinely diverge, and the post-image tail does not
  // rope at its own sequence index. Without this a "passing" tolerance below would prove nothing.
  int64_t rows_with_distinct_axes = 0, rows_off_sequence = 0;
  for (int64_t p = 0; p < total_len; ++p) {
    const int64_t t = pos_rows[p], h = pos_rows[total_len + p], w = pos_rows[2 * total_len + p];
    if (t != h || h != w) ++rows_with_distinct_axes;
    if (t != p) ++rows_off_sequence;
  }
  std::printf("[mrope_attn_layer] total=%lld prefill=%lld decode=%lld; %lld rows have distinct "
              "(t,h,w), %lld rows rope off their sequence index\n",
              static_cast<long long>(total_len), static_cast<long long>(prefill_len),
              static_cast<long long>(decode_len),
              static_cast<long long>(rows_with_distinct_axes),
              static_cast<long long>(rows_off_sequence));
  bool ok = rows_with_distinct_axes > 0 && rows_off_sequence > 0;
  if (!ok) {
    std::fprintf(stderr, "golden does not actually exercise 3-axis mrope -- regenerate it\n");
    return 1;
  }
  int64_t image_rows = 0;
  for (int64_t v : mm_ids_i64) {
    if (v == 1) ++image_rows;
  }
  std::printf("[mrope_attn_layer] prompt carries %lld image-placeholder rows\n",
              static_cast<long long>(image_rows));

  // ---- layer 3's bf16 weights ------------------------------------------------------------------
  auto input_layernorm_d =
      Upload(ReadContainerBf16(container, "text.layers.3.input_layernorm", hidden));
  auto q_norm_d = Upload(ReadContainerBf16(container, "text.layers.3.attn.q_norm", D));
  auto k_norm_d = Upload(ReadContainerBf16(container, "text.layers.3.attn.k_norm", D));
  auto k_descale_d = Upload(ReadFp32Flat(container, "text.layers.3.attn.k_descale", Hkv));
  auto v_descale_d = Upload(ReadFp32Flat(container, "text.layers.3.attn.v_descale", Hkv));

  auto make_bf16_linear = [&](const std::string& name, int64_t N, int64_t K) {
    QuantLinear ql;
    ql.layout = Layout::kBf16;
    ql.N = N;
    ql.K = K;
    ql.bf16_w = Upload(ReadContainerBf16(container, name, N * K));
    return ql;
  };
  QuantLinear qg_ql = make_bf16_linear("text.layers.3.attn.qg.bf16.w",
                                        static_cast<int64_t>(2) * H * D, hidden);
  QuantLinear k_ql = make_bf16_linear("text.layers.3.attn.k.bf16.w",
                                       static_cast<int64_t>(Hkv) * D, hidden);
  QuantLinear v_ql = make_bf16_linear("text.layers.3.attn.v.bf16.w",
                                       static_cast<int64_t>(Hkv) * D, hidden);
  QuantLinear o_ql = make_bf16_linear("text.layers.3.attn.o.bf16.w", hidden,
                                       static_cast<int64_t>(H) * D);

  AttnWeights w{};
  w.input_layernorm = input_layernorm_d.data();
  w.qg = &qg_ql;
  w.k = &k_ql;
  w.v = &v_ql;
  w.o = &o_ql;
  w.q_norm = q_norm_d.data();
  w.k_norm = k_norm_d.data();
  w.k_descale = k_descale_d.data();
  w.v_descale = v_descale_d.data();

  // ---- run prefill then a continuing decode, both with the golden's own 3-axis rows ------------
  const r4d::AttnDims dims = r4d::GetAttnDims();
  AttentionLayer layer(cfg);
  PagedKvCache kv(Hkv, D, dims.block_size, /*max_context_tokens=*/128);
  Arena arena(64ull << 20);

  // `positions` is the KV slot mapping and stays the plain sequence index; `rope_pos3` is the
  // [3, T] compact slice of the golden's rows -- the split this stage exists to introduce.
  auto slice_rope3 = [&](int64_t lo, int64_t n) {
    std::vector<int32_t> out(static_cast<size_t>(3 * n));
    for (int axis = 0; axis < 3; ++axis) {
      for (int64_t t = 0; t < n; ++t) {
        out[static_cast<size_t>(axis * n + t)] =
            static_cast<int32_t>(pos_rows[static_cast<size_t>(axis * total_len + lo + t)]);
      }
    }
    return out;
  };
  auto slot_positions = [&](int64_t lo, int64_t n) {
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t) out[static_cast<size_t>(t)] = static_cast<int32_t>(lo + t);
    return out;
  };
  auto seqused = [&](int64_t lo, int64_t n) {
    return std::vector<int32_t>{static_cast<int32_t>(lo + n)};
  };

  struct Stage {
    const char* name;
    int64_t lo, n;
  };
  const Stage stages[2] = {{"prefill", 0, prefill_len}, {"decode", prefill_len, decode_len}};
  constexpr double kTol = 2e-2;  // the repo's bf16_matmul_rel_err band -- see the file comment
  double prefill_rel = 0.0;
  for (const Stage& st : stages) {
    const auto in_h = ReadGoldenBf16(golden, std::string(st.name) + "_hidden_states",
                                      st.n * hidden);
    const auto expect_h = ReadGoldenBf16(golden, std::string(st.name) + "_attention_output",
                                          st.n * hidden);
    auto in_d = Upload(in_h);
    DeviceBuffer<uint16_t> out_d(static_cast<size_t>(st.n) * hidden);
    auto pos_d = Upload(slot_positions(st.lo, st.n));
    auto seq_d = Upload(seqused(st.lo, st.n));
    auto rope_d = Upload(slice_rope3(st.lo, st.n));

    // Everything between `stream` and `rope_pos3` is the default text-path argument list; the one
    // non-default value is the trailing rope_pos3 (AttentionLayer::Forward's doc comment).
    layer.Forward(arena, in_d.data(), out_d.data(), w, kv, static_cast<int>(st.n),
                  static_cast<int>(st.lo), pos_d.data(), seq_d.data(), nullptr,
                  /*x_normed_in=*/nullptr, /*next_norm_weight=*/nullptr, /*x_normed_out=*/nullptr,
                  /*prof=*/nullptr, /*x_normed_pre_epilogue=*/0, /*x_normed_pre_data=*/nullptr,
                  /*x_normed_pre_scale=*/nullptr, /*next_epilogue=*/0,
                  /*next_epilogue_out=*/nullptr, /*next_epilogue_scale=*/nullptr, rope_d.data());
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    arena.Reset();

    // Forward() fuses the residual add, so the comparison target is residual + attention_output,
    // exactly as test_attn_layer.cpp's own ExpectedResidualSum builds it.
    std::vector<uint16_t> expect(expect_h.size());
    for (size_t i = 0; i < expect.size(); ++i) {
      expect[i] = FloatToBf16(Bf16ToFloat(in_h[i]) + Bf16ToFloat(expect_h[i]));
    }
    const double rel = NormRelErr(out_d.CopyToHost(), expect);
    if (st.lo == 0) prefill_rel = rel;
    const bool stage_ok = rel < kTol;
    ok = ok && stage_ok;
    std::printf("[mrope_attn_layer] %-8s T=%2lld start_pos=%2lld norm rel err=%.4e (%s)\n",
                st.name, static_cast<long long>(st.n), static_cast<long long>(st.lo), rel,
                stage_ok ? "PASS" : "FAIL");
  }

  // ---- negative control: the SAME call with the plain sequence index as the rope position ------
  // A tolerance test only means something if the wrong answer actually fails it. Re-running the
  // prefill with rope_pos3 = nullptr (i.e. the pre-vision text path, roping at the KV slot index)
  // must land outside the band above -- if it did not, this test could not distinguish a correct
  // 3-axis rope from no mrope at all.
  //
  // The bound is stated two ways, because the single-number version would be a guess: the wrong
  // rope must exceed BOTH 2x the absolute tolerance AND 3x whatever the correct run just
  // measured. Measured on HIP device 1 against this golden: correct 1.22e-2, wrong 5.48e-2, i.e.
  // 4.5x apart. That ratio is what it is because roping at the sequence index is not a small
  // perturbation of roping at `index + delta` -- but it is also not unbounded, since the first
  // seven tokens of this prompt DO sit at their own sequence index and rope identically either
  // way, and the rotation is confined to 64 of each head's 256 dims.
  {
    const auto in_h = ReadGoldenBf16(golden, "prefill_hidden_states", prefill_len * hidden);
    const auto expect_h = ReadGoldenBf16(golden, "prefill_attention_output", prefill_len * hidden);
    PagedKvCache kv2(Hkv, D, dims.block_size, /*max_context_tokens=*/128);
    auto in_d = Upload(in_h);
    DeviceBuffer<uint16_t> out_d(static_cast<size_t>(prefill_len) * hidden);
    auto pos_d = Upload(slot_positions(0, prefill_len));
    auto seq_d = Upload(seqused(0, prefill_len));
    layer.Forward(arena, in_d.data(), out_d.data(), w, kv2, static_cast<int>(prefill_len), 0,
                  pos_d.data(), seq_d.data(), nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    arena.Reset();
    std::vector<uint16_t> expect(expect_h.size());
    for (size_t i = 0; i < expect.size(); ++i) {
      expect[i] = FloatToBf16(Bf16ToFloat(in_h[i]) + Bf16ToFloat(expect_h[i]));
    }
    const double rel = NormRelErr(out_d.CopyToHost(), expect);
    const double floor_abs = 2.0 * kTol, floor_rel = 3.0 * prefill_rel;
    const bool control_ok = rel > floor_abs && rel > floor_rel;
    ok = ok && control_ok;
    std::printf("[mrope_attn_layer] control (rope at the KV slot index, no mrope): norm rel "
                "err=%.4e -- must exceed %.1e and 3x the correct run's %.4e (%s)\n",
                rel, floor_abs, prefill_rel, control_ok ? "PASS" : "FAIL");
  }

  std::printf(ok ? "test_mrope_attn_layer: OK\n" : "test_mrope_attn_layer: FAILED\n");
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  if (!r4dx_test::FileExists(R4DX_GOLDEN_MROPE_PATH)) {
    return r4dx_test::SkipMissing(R4DX_GOLDEN_MROPE_PATH);
  }
  if (!r4dx_test::FileExists(R4DX_BF16_CONTAINER_PATH)) {
    return r4dx_test::SkipMissing(R4DX_BF16_CONTAINER_PATH);
  }
  try {
    return Run();
  } catch (const std::exception& e) {
    // Same reason test_attn_layer.cpp wraps its body: an exception escaping main() on Windows
    // becomes __fastfail (0xC0000409), which CTest reports as a crash rather than a failure.
    std::fprintf(stderr, "test_mrope_attn_layer: unexpected exception: %s\n", e.what());
    return 1;
  }
}
