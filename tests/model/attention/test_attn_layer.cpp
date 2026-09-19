// tests/model/attention/test_attn_layer.cpp -- r4dx::model::attention::AttentionLayer against
// tools/reference/layer_golden.py's layer 3 (full-attention) golden: real Qwen3.8-27B weights
// (D:\models\r4dx\qwen38-27b-l4-bf16.r4dx, bf16 layout, layers 0-3) and real transformers
// activations (tools/reference/golden_out/layer_003_full_attention.safetensors). Runs a T=64
// prefill (fresh KV cache) then a T=4 decode continuing the same cache (matching the golden's own
// "decode reuses the prefill Cache" note), and diffs AttentionLayer::Forward's residual-added
// output (residual + attn_out, since this component fuses the residual add per the task brief)
// against golden hidden_states + attention_output for each stage.
//
// TOLERANCE: this component's r4d attention call always writes through the fp8 e4m3 paged KV
// cache (docs/architecture.md's only wired attention path), while the golden ran real bf16
// attention with no KV quantization at all -- so this diff measures BOTH the bf16 GEMM/rmsnorm/
// rope chain AND the fp8 KV round-trip in one number, against this container's still-placeholder
// (1.0, docs/status.md "Known gaps") k_descale/v_descale rather than a calibrated table. Measured
// on real Qwen3.8-27B layer-3 weights + real transformers activations (2026-09-19, HIP device 1):
// prefill (T=64, fresh cache) norm rel err = 1.84e-2; decode (T=4, start_pos=64, continued cache)
// = 1.54e-2 -- both under this repo's standard bf16_matmul_rel_err bound (2e-2,
// tools/reference/golden_out/manifest.json's tolerance table), so the fp8-KV-vs-bf16-KV gap turns
// out small enough at this container's activation magnitudes that the looser 5e-2 band the task
// brief allows ("if the golden used bf16 KV, allow up to 5e-2 and document") was not needed --
// gated at the stricter 2e-2 here since that is what actually holds.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "quant_linear.h"
#include "r4dx/core/r4d.hpp"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

#ifndef R4DX_GOLDEN_ATTN_PATH
#define R4DX_GOLDEN_ATTN_PATH "tools/reference/golden_out/layer_003_full_attention.safetensors"
#endif
#ifndef R4DX_BF16_CONTAINER_PATH
#define R4DX_BF16_CONTAINER_PATH "D:/models/r4dx/qwen38-27b-l4-bf16.r4dx"
#endif

using namespace r4dx::core;
using r4dx::model::Layout;
using r4dx::model::LayoutName;
using r4dx::model::QuantLinear;
using r4dx::model::attention::AttentionLayer;
using r4dx::model::attention::AttnConfig;
using r4dx::model::attention::AttnWeights;
using r4dx::model::attention::PagedKvCache;
using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

namespace {

// Reads an r4dx CONTAINER bf16 tensor (docs/container-format.md: stored as U8 with a trailing
// [...,2] byte-pair axis, so meta.ElemCount() is BYTES, not elements) as a flat uint16_t vector.
std::vector<uint16_t> ReadContainerBf16(const SafetensorsReader& r, const std::string& name,
                                         int64_t expected_elems) {
  const auto& meta = r.Meta(name);
  const int64_t elems = meta.ElemCount() / 2;  // last axis is the raw byte-pair, not a real dim
  if (elems != expected_elems) {
    std::fprintf(stderr, "ReadContainerBf16(%s): expected %lld elements, container has %lld\n",
                 name.c_str(), static_cast<long long>(expected_elems), static_cast<long long>(elems));
    std::exit(1);
  }
  const uint16_t* p = reinterpret_cast<const uint16_t*>(r.Data(name));
  return std::vector<uint16_t>(p, p + elems);
}

// Reads a GOLDEN tensor (tools/reference/layer_golden.py writes plain safetensors with a real
// "BF16" dtype -- meta.ElemCount() is already the element count, unlike the container's U8-with-
// trailing-axis encoding above) as a flat uint16_t vector.
std::vector<uint16_t> ReadGoldenBf16(const SafetensorsReader& r, const std::string& name,
                                      int64_t expected_elems) {
  const auto& meta = r.Meta(name);
  if (meta.dtype != "BF16") {
    std::fprintf(stderr, "ReadGoldenBf16(%s): expected dtype BF16, golden has %s\n", name.c_str(),
                 meta.dtype.c_str());
    std::exit(1);
  }
  const int64_t elems = meta.ElemCount();
  if (elems != expected_elems) {
    std::fprintf(stderr, "ReadGoldenBf16(%s): expected %lld elements, golden has %lld\n", name.c_str(),
                 static_cast<long long>(expected_elems), static_cast<long long>(elems));
    std::exit(1);
  }
  const uint16_t* p = reinterpret_cast<const uint16_t*>(r.Data(name));
  return std::vector<uint16_t>(p, p + elems);
}

std::vector<float> ReadFp32Flat(const SafetensorsReader& r, const std::string& name,
                                 int64_t expected_elems) {
  const auto& meta = r.Meta(name);
  const int64_t bytes = meta.ElemCount();
  if (bytes != expected_elems * 4) {
    std::fprintf(stderr, "ReadFp32Flat(%s): expected %lld bytes, container has %lld\n", name.c_str(),
                 static_cast<long long>(expected_elems * 4), static_cast<long long>(bytes));
    std::exit(1);
  }
  const float* p = reinterpret_cast<const float*>(r.Data(name));
  return std::vector<float>(p, p + expected_elems);
}

DeviceBuffer<uint16_t> UploadBf16(const std::vector<uint16_t>& h) {
  DeviceBuffer<uint16_t> d(h.size());
  d.CopyFromHost(h);
  return d;
}

DeviceBuffer<float> UploadFp32(const std::vector<float>& h) {
  DeviceBuffer<float> d(h.size());
  d.CopyFromHost(h);
  return d;
}

// positions[t] = start_pos+t doubles as both RoPE position ids and KV slot_mapping (contiguous
// block table => slot==pos, see AttentionLayer::Forward's doc comment).
DeviceBuffer<int32_t> UploadPositions(int start_pos, int T) {
  std::vector<int32_t> h(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) h[static_cast<size_t>(t)] = start_pos + t;
  DeviceBuffer<int32_t> d(h.size());
  d.CopyFromHost(h);
  return d;
}

DeviceBuffer<int32_t> UploadSequsedK(int start_pos, int T) {
  const int32_t v = start_pos + T;
  DeviceBuffer<int32_t> d(1);
  d.CopyFromHost(&v, 1);
  return d;
}

// Norm-relative L2 error over the whole tensor -- the style tests/kernels/test_rope.cpp and
// tests/kernels/test_kv_write.cpp use for a diff that mixes a bf16 GEMM chain with a lossy fp8
// KV round-trip (a per-element max-relative metric would spike on the (rare) near-zero reference
// entries fp8 rounding is least accurate on, without that reflecting the chain's real accuracy).
double NormRelErr(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g = Bf16ToFloat(got[i]);
    double r = Bf16ToFloat(ref[i]);
    num += (g - r) * (g - r);
    den += r * r;
  }
  return std::sqrt(num) / std::max(1e-9, std::sqrt(den));
}

// expected[i] = bf16(fp32(residual[i]) + fp32(attn_out[i])) -- what r4dx_residual_add_bf16 itself
// computes, applied here to two GOLDEN tensors (no new numerics introduced) to build the
// "residual after this layer's attention block" ground truth architecture.md's outer loop would
// carry into post_attention_layernorm. No golden tensor for this sum exists directly (the golden
// captures pre-residual `attention_output`, matching architecture.md's `x = attn_layer(x)` before
// its own separate `x = residual + x`) -- this component fuses the residual add into
// AttentionLayer::Forward per the task brief, so this is the correct comparison target.
std::vector<uint16_t> ExpectedResidualSum(const std::vector<uint16_t>& residual,
                                           const std::vector<uint16_t>& attn_out) {
  std::vector<uint16_t> out(residual.size());
  for (size_t i = 0; i < residual.size(); ++i) {
    out[i] = FloatToBf16(Bf16ToFloat(residual[i]) + Bf16ToFloat(attn_out[i]));
  }
  return out;
}

void ReportQuantizedLayouts(const SafetensorsReader& container) {
  const char* bases[] = {"text.layers.3.attn.qg", "text.layers.3.attn.o", "text.layers.3.attn.k",
                          "text.layers.3.attn.v"};
  const char* layouts[] = {"bf16.w", "mxfp4.wq", "w4a16.wq", "w4a8.wq"};
  std::printf("quantized layouts present in %s (informational; the quantized layouts below are "
              "now actually dispatched through AttentionLayer/ApplyLinear and rel-err REPORTED, "
              "not just gated bf16-tight -- see main()'s quantized pass):\n",
              R4DX_BF16_CONTAINER_PATH);
  for (const char* base : bases) {
    for (const char* layout : layouts) {
      const std::string name = std::string(base) + "." + layout;
      std::printf("  %-32s %s\n", name.c_str(), container.Has(name) ? "present" : "MISSING");
    }
  }
}

// Small local mirror of container.cpp's (anonymous-namespace, not exported) LoadQuantLinear --
// this test intentionally does not link all of r4dx_model (only r4dx_model_attention, which pulls
// in r4dx_model_linear for QuantLinear/ApplyLinear -- see this directory's CMakeLists.txt comment),
// so it cannot call Container::Load's private loader directly. Returns std::nullopt if any of this
// layout's tensors are missing from `r` (e.g. a container that only carries bf16).
std::optional<QuantLinear> TryLoadQuantLinear(const SafetensorsReader& r, const std::string& base,
                                               Layout layout, int64_t N, int64_t K) {
  auto upload_u8 = [&](const std::string& name) {
    DeviceBuffer<uint8_t> d(static_cast<size_t>(r.Meta(name).ElemCount()));
    d.CopyFromHost(reinterpret_cast<const uint8_t*>(r.Data(name)), d.size());
    return d;
  };
  auto upload_i8 = [&](const std::string& name) {
    DeviceBuffer<int8_t> d(static_cast<size_t>(r.Meta(name).ElemCount()));
    d.CopyFromHost(reinterpret_cast<const int8_t*>(r.Data(name)), d.size());
    return d;
  };
  auto upload_u32 = [&](const std::string& name) {
    DeviceBuffer<uint32_t> d(static_cast<size_t>(r.Meta(name).ElemCount()));
    d.CopyFromHost(reinterpret_cast<const uint32_t*>(r.Data(name)), d.size());
    return d;
  };

  QuantLinear q;
  q.layout = layout;
  q.N = N;
  q.K = K;
  switch (layout) {
    case Layout::kW4a16: {
      const std::string wq = base + ".w4a16.wq", wsz = base + ".w4a16.wsz";
      if (!r.Has(wq) || !r.Has(wsz)) return std::nullopt;
      q.wq = upload_u8(wq);
      q.w4a16_wsz = upload_u32(wsz);
      break;
    }
    case Layout::kW4a8: {
      const std::string wq = base + ".w4a8.wq", ws = base + ".w4a8.ws";
      if (!r.Has(wq) || !r.Has(ws)) return std::nullopt;
      q.wq = upload_u8(wq);
      q.w4a8_ws = upload_u32(ws);
      break;
    }
    case Layout::kMxfp4: {
      const std::string wq = base + ".mxfp4.wq", ws = base + ".mxfp4.ws", wref = base + ".mxfp4.wref";
      if (!r.Has(wq) || !r.Has(ws) || !r.Has(wref)) return std::nullopt;
      q.mxfp4_wq = upload_u8(wq);
      q.mxfp4_ws = upload_u8(ws);
      q.mxfp4_wref = upload_i8(wref);
      break;
    }
    case Layout::kBf16:
      return std::nullopt;  // caller already has the bf16 path
  }
  return q;
}

// Runs AttentionLayer's prefill+decode pair (same golden activations as the bf16 pass above) with
// qg/o loaded in `layout` instead, and gates PASS/FAIL against the same bf16-golden target at a
// deliberately loose bound (see kQuantTol below) -- quantization error on qg/o alone, mixed with
// everything else in the layer staying bf16-precision, is expected to land in the same
// ~6.5e-2..8.5e-2 ballpark docs/perf.md already documents for GDN/MLP/lm_head's own quantized
// linears (this component's own "bf16 layout tight; quantized layouts reported" scope, task
// brief). Review finding, 2026-09-19: this function used to only PRINT the rel-err and return
// void, so a regression in the w4a16/w4a8/mxfp4 dispatch through ApplyLinear would still print
// PASS -- returns bool now, folded into main()'s own `ok`.
bool RunQuantizedLayoutSmoke(const SafetensorsReader& container, const AttnConfig& cfg,
                              const AttnWeights& bf16_w, const std::vector<uint16_t>& prefill_hidden_h,
                              const std::vector<uint16_t>& prefill_expected,
                              const std::vector<uint16_t>& decode_hidden_h,
                              const std::vector<uint16_t>& decode_expected, int T_prefill,
                              int T_decode) {
  // Deliberately loose but real bound -- roughly 2x the worst measured baseline (w4a16 prefill
  // 7.1698e-02 / decode 6.5744e-02, w4a8 8.4888e-02 / 7.5489e-02, mxfp4 8.2541e-02 / 7.3365e-02, all
  // measured on HIP device 1 against the real bf16 container during this pass) so today's numbers
  // cannot silently drift without tripping this test, without being tight enough to false-positive
  // on ordinary run-to-run quantization noise.
  constexpr double kQuantTol = 1.5e-1;
  bool ok = true;
  const int hidden = cfg.hidden, H = cfg.num_heads, Hkv = cfg.kv_heads, D = cfg.head_dim;
  const r4d::AttnDims dims = r4d::GetAttnDims();
  for (Layout layout : {Layout::kW4a16, Layout::kW4a8, Layout::kMxfp4}) {
    auto qg_ql = TryLoadQuantLinear(container, "text.layers.3.attn.qg", layout,
                                     static_cast<int64_t>(2) * H * D, hidden);
    auto o_ql = TryLoadQuantLinear(container, "text.layers.3.attn.o", layout, hidden,
                                    static_cast<int64_t>(H) * D);
    if (!qg_ql || !o_ql) {
      std::printf("quantized layout %s: SKIPPED (tensors not present in %s)\n", LayoutName(layout),
                  R4DX_BF16_CONTAINER_PATH);
      continue;
    }
    // k/v (R1, docs/r9700.md): optional -- a container converted before this pass carries qg/o in
    // every quantized layout but k/v only bf16, so k_ql/v_ql legitimately come back nullopt on an
    // old fixture; this loop still exercises qg/o's own quantized dispatch in that case (unchanged
    // behavior), it just cannot also report k/v's accuracy until the fixture is regenerated.
    auto k_ql = TryLoadQuantLinear(container, "text.layers.3.attn.k", layout,
                                    static_cast<int64_t>(Hkv) * D, hidden);
    auto v_ql = TryLoadQuantLinear(container, "text.layers.3.attn.v", layout,
                                    static_cast<int64_t>(Hkv) * D, hidden);

    AttnWeights w = bf16_w;
    w.qg = &*qg_ql;
    w.o = &*o_ql;
    if (k_ql) w.k = &*k_ql;
    if (v_ql) w.v = &*v_ql;
    std::printf("  (k/v at %-6s: %s)\n", LayoutName(layout),
                (k_ql && v_ql) ? "quantized" : "bf16 fallback (container predates R1)");

    AttentionLayer layer(cfg);
    PagedKvCache kv(Hkv, D, dims.block_size, /*max_context_tokens=*/128);
    Arena arena(64ull << 20);

    auto prefill_in_d = UploadBf16(prefill_hidden_h);
    DeviceBuffer<uint16_t> prefill_out_d(static_cast<size_t>(T_prefill) * hidden);
    auto prefill_pos_d = UploadPositions(0, T_prefill);
    auto prefill_seqused_d = UploadSequsedK(0, T_prefill);
    layer.Forward(arena, prefill_in_d.data(), prefill_out_d.data(), w, kv, T_prefill, 0,
                  prefill_pos_d.data(), prefill_seqused_d.data(), nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    arena.Reset();

    auto decode_in_d = UploadBf16(decode_hidden_h);
    DeviceBuffer<uint16_t> decode_out_d(static_cast<size_t>(T_decode) * hidden);
    auto decode_pos_d = UploadPositions(T_prefill, T_decode);
    auto decode_seqused_d = UploadSequsedK(T_prefill, T_decode);
    layer.Forward(arena, decode_in_d.data(), decode_out_d.data(), w, kv, T_decode, T_prefill,
                  decode_pos_d.data(), decode_seqused_d.data(), nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    arena.Reset();

    const double prefill_rel = NormRelErr(prefill_out_d.CopyToHost(), prefill_expected);
    const double decode_rel = NormRelErr(decode_out_d.CopyToHost(), decode_expected);
    const bool layout_ok = prefill_rel < kQuantTol && decode_rel < kQuantTol;
    ok = ok && layout_ok;
    std::printf("quantized layout %-6s: prefill norm rel err=%.4e, decode norm rel err=%.4e (%s)\n",
                LayoutName(layout), prefill_rel, decode_rel, layout_ok ? "PASS" : "FAIL");
  }
  return ok;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  SafetensorsReader golden(Utf8ToWide(R4DX_GOLDEN_ATTN_PATH));
  SafetensorsReader container(Utf8ToWide(R4DX_BF16_CONTAINER_PATH));

  ReportQuantizedLayouts(container);

  AttnConfig cfg;  // defaults match Qwen3.8-27B's text_config (see types.hpp)
  const int hidden = cfg.hidden, H = cfg.num_heads, Hkv = cfg.kv_heads, D = cfg.head_dim;

  // ---- load layer 3's bf16 weights from the container -----------------------------------------
  auto input_layernorm_h = ReadContainerBf16(container, "text.layers.3.input_layernorm", hidden);
  auto qg_w_h = ReadContainerBf16(container, "text.layers.3.attn.qg.bf16.w",
                                   static_cast<int64_t>(2) * H * D * hidden);
  // attn.k/v (R1, docs/r9700.md): now go through the same `.bf16.w`-suffixed multi-layout naming
  // as qg/o (container.cpp's LoadQuantLinearWithFallback also accepts the old bare form for a
  // pre-R1 container; this test reads directly via SafetensorsReader, not that fallback, so it
  // must know which form the fixture container it's pointed at was actually built with).
  auto k_w_h = ReadContainerBf16(container, "text.layers.3.attn.k.bf16.w",
                                  static_cast<int64_t>(Hkv) * D * hidden);
  auto v_w_h = ReadContainerBf16(container, "text.layers.3.attn.v.bf16.w",
                                  static_cast<int64_t>(Hkv) * D * hidden);
  auto o_w_h = ReadContainerBf16(container, "text.layers.3.attn.o.bf16.w",
                                  static_cast<int64_t>(hidden) * H * D);
  auto q_norm_h = ReadContainerBf16(container, "text.layers.3.attn.q_norm", D);
  auto k_norm_h = ReadContainerBf16(container, "text.layers.3.attn.k_norm", D);
  auto k_descale_h = ReadFp32Flat(container, "text.layers.3.attn.k_descale", Hkv);
  auto v_descale_h = ReadFp32Flat(container, "text.layers.3.attn.v_descale", Hkv);

  auto input_layernorm_d = UploadBf16(input_layernorm_h);
  auto qg_w_d = UploadBf16(qg_w_h);
  auto k_w_d = UploadBf16(k_w_h);
  auto v_w_d = UploadBf16(v_w_h);
  auto o_w_d = UploadBf16(o_w_h);
  auto q_norm_d = UploadBf16(q_norm_h);
  auto k_norm_d = UploadBf16(k_norm_h);
  auto k_descale_d = UploadFp32(k_descale_h);
  auto v_descale_d = UploadFp32(v_descale_h);

  QuantLinear qg_ql;
  qg_ql.layout = Layout::kBf16;
  qg_ql.N = static_cast<int64_t>(2) * H * D;
  qg_ql.K = hidden;
  qg_ql.bf16_w = std::move(qg_w_d);

  QuantLinear k_ql;
  k_ql.layout = Layout::kBf16;
  k_ql.N = static_cast<int64_t>(Hkv) * D;
  k_ql.K = hidden;
  k_ql.bf16_w = std::move(k_w_d);

  QuantLinear v_ql;
  v_ql.layout = Layout::kBf16;
  v_ql.N = static_cast<int64_t>(Hkv) * D;
  v_ql.K = hidden;
  v_ql.bf16_w = std::move(v_w_d);

  QuantLinear o_ql;
  o_ql.layout = Layout::kBf16;
  o_ql.N = hidden;
  o_ql.K = static_cast<int64_t>(H) * D;
  o_ql.bf16_w = std::move(o_w_d);

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

  // ---- golden activations -----------------------------------------------------------------------
  const int T_prefill = 64, T_decode = 4;
  auto prefill_hidden_h =
      ReadGoldenBf16(golden, "prefill_hidden_states", static_cast<int64_t>(T_prefill) * hidden);
  auto prefill_attn_out_h =
      ReadGoldenBf16(golden, "prefill_attention_output", static_cast<int64_t>(T_prefill) * hidden);
  auto decode_hidden_h =
      ReadGoldenBf16(golden, "decode_hidden_states", static_cast<int64_t>(T_decode) * hidden);
  auto decode_attn_out_h =
      ReadGoldenBf16(golden, "decode_attention_output", static_cast<int64_t>(T_decode) * hidden);

  // ---- run the layer -------------------------------------------------------------------------
  r4d::AttnDims dims = r4d::GetAttnDims();  // block_size (16), sanity-checked against head_dim/gqa
  if (dims.head_dim != D || dims.gqa != cfg.Gqa()) {
    std::fprintf(stderr, "r4d_attn_dims mismatch: head_dim=%d gqa=%d, expected %d/%d\n", dims.head_dim,
                 dims.gqa, D, cfg.Gqa());
    return 1;
  }

  AttentionLayer layer(cfg);
  PagedKvCache kv(Hkv, D, dims.block_size, /*max_context_tokens=*/128);
  r4dx::core::Arena arena(64ull << 20);

  auto prefill_in_d = UploadBf16(prefill_hidden_h);
  DeviceBuffer<uint16_t> prefill_out_d(static_cast<size_t>(T_prefill) * hidden);
  auto prefill_pos_d = UploadPositions(/*start_pos=*/0, T_prefill);
  auto prefill_seqused_d = UploadSequsedK(/*start_pos=*/0, T_prefill);
  layer.Forward(arena, prefill_in_d.data(), prefill_out_d.data(), w, kv, T_prefill,
                /*start_pos=*/0, prefill_pos_d.data(), prefill_seqused_d.data(),
                /*stream=*/nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  arena.Reset();
  std::vector<uint16_t> prefill_out_h = prefill_out_d.CopyToHost();

  auto decode_in_d = UploadBf16(decode_hidden_h);
  DeviceBuffer<uint16_t> decode_out_d(static_cast<size_t>(T_decode) * hidden);
  auto decode_pos_d = UploadPositions(/*start_pos=*/T_prefill, T_decode);
  auto decode_seqused_d = UploadSequsedK(/*start_pos=*/T_prefill, T_decode);
  layer.Forward(arena, decode_in_d.data(), decode_out_d.data(), w, kv, T_decode,
                /*start_pos=*/T_prefill, decode_pos_d.data(), decode_seqused_d.data(),
                /*stream=*/nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  arena.Reset();
  std::vector<uint16_t> decode_out_h = decode_out_d.CopyToHost();

  // ---- compare ------------------------------------------------------------------------------
  std::vector<uint16_t> prefill_expected = ExpectedResidualSum(prefill_hidden_h, prefill_attn_out_h);
  std::vector<uint16_t> decode_expected = ExpectedResidualSum(decode_hidden_h, decode_attn_out_h);

  double prefill_rel = NormRelErr(prefill_out_h, prefill_expected);
  double decode_rel = NormRelErr(decode_out_h, decode_expected);

  std::printf("attention_layer prefill (T=%d, fresh fp8 KV cache) norm rel err=%.4e\n", T_prefill,
              prefill_rel);
  std::printf("attention_layer decode (T=%d, start_pos=%d, continued fp8 KV cache) norm rel err=%.4e\n",
              T_decode, T_prefill, decode_rel);

  // Repo-standard bf16_matmul_rel_err bound (tools/reference/golden_out/manifest.json); see this
  // file's header comment for why the looser 5e-2 fp8-KV allowance the task brief offers was not
  // needed in practice.
  const double kTol = 2e-2;
  bool ok = prefill_rel < kTol && decode_rel < kTol;
  std::printf(ok ? "PASS\n" : "FAIL\n");

  // Quantized qg/o layouts (decode-perf pass, 2026-09-19): now gated too (review finding,
  // 2026-09-19) -- see RunQuantizedLayoutSmoke's own comment and this component's "bf16 layout
  // tight; quantized layouts reported" scope.
  const bool quant_ok = RunQuantizedLayoutSmoke(container, cfg, w, prefill_hidden_h,
                                                 prefill_expected, decode_hidden_h, decode_expected,
                                                 T_prefill, T_decode);
  ok = ok && quant_ok;

  return ok ? 0 : 1;
}
