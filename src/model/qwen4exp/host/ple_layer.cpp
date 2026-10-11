#include "r4dx/model/qwen4exp/ple_layer.h"

#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"
#include "r4dx/model/qwen4exp/ple_kernels.h"

namespace r4dx::model::qwen4exp {

namespace {

constexpr const char* kCtx = "qwen4exp::PleLayer";

[[noreturn]] void Fail(const std::string& msg) { qwen4exp::Fail(kCtx, msg); }
void Check(int code, const char* what) { qwen4exp::Check(kCtx, code, what); }

}  // namespace

PleLayer::PleLayer(const PleConfig& cfg, const PleWeights& w) : cfg_(cfg), w_(w) {
  if (cfg.hc < 1 || cfg.hidden < 1 || cfg.embed_dim < 1) Fail("non-positive dimension in the config");
  if (cfg.conv_kernel != 4 || cfg.conv_dilation != 3) {
    Fail("only conv_kernel 4 / dilation 3 are instantiated, got " + std::to_string(cfg.conv_kernel) + " / " +
         std::to_string(cfg.conv_dilation));
  }
  const int64_t channels = static_cast<int64_t>(cfg.hc) * cfg.hidden;
  CheckLinear(kCtx, w.key_proj, "key_proj", channels, cfg.embed_dim);
  CheckLinear(kCtx, w.value_proj, "value_proj", cfg.hidden, cfg.embed_dim);
  if (w.norm_key == nullptr || w.norm_query == nullptr || w.norm_conv == nullptr || w.conv1d == nullptr) {
    Fail("missing norm / conv weight");
  }
  state_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(channels) *
                                        static_cast<size_t>((cfg.conv_kernel - 1) * cfg.conv_dilation));
  // hipMemset runs on the null stream, which a non-blocking stream does not wait for: finish it before returning.
  state_.Zero();
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));
}

size_t PleLayer::ArenaBytes(int64_t rows) const {
  const size_t r = static_cast<size_t>(rows), channels = static_cast<size_t>(cfg_.hc) * cfg_.hidden;
  return 2 * Padded(r * channels * 2) +                                  // key_proj out, gated value
         Padded(r * static_cast<size_t>(cfg_.hidden) * 2) +             // value_proj out
         Padded(r * channels * 2);                                      // norm_conv out
}

void PleLayer::Reset(hipStream_t stream) {
  R4DX_HIP_CHECK(hipMemsetAsync(state_.data(), 0, state_.bytes(), stream));
}

void PleLayer::SaveState(hipStream_t stream, uint16_t* dst) const {
  R4DX_HIP_CHECK(hipMemcpyAsync(dst, state_.data(), state_.bytes(), hipMemcpyDeviceToDevice, stream));
}

void PleLayer::LoadState(hipStream_t stream, const uint16_t* src) {
  R4DX_HIP_CHECK(hipMemcpyAsync(state_.data(), src, state_.bytes(), hipMemcpyDeviceToDevice, stream));
}

void PleLayer::Forward(hipStream_t stream, core::Arena& arena, uint16_t* h, const uint16_t* emb, int64_t rows) {
  if (rows <= 0) return;
  const int64_t channels = static_cast<int64_t>(cfg_.hc) * cfg_.hidden;
  uint16_t* kp = arena.Alloc<uint16_t>(static_cast<size_t>(rows * channels));
  uint16_t* gv = arena.Alloc<uint16_t>(static_cast<size_t>(rows * channels));
  uint16_t* value = arena.Alloc<uint16_t>(static_cast<size_t>(rows * cfg_.hidden));
  uint16_t* gvn = arena.Alloc<uint16_t>(static_cast<size_t>(rows * channels));

  ApplyBf16Linear(stream, arena, *w_.key_proj, emb, kp, rows);
  ApplyBf16Linear(stream, arena, *w_.value_proj, emb, value, rows);
  Check(q4x_ple_gate_value_bf16(kp, h, value, w_.norm_key, w_.norm_query, gv, rows, cfg_.hc, cfg_.hidden, cfg_.eps,
                                stream),
        "gate / value");
  Check(q4x_grouped_rmsnorm_bf16(gv, w_.norm_conv, gvn, rows, cfg_.hc, cfg_.hidden, cfg_.eps, stream), "norm_conv");
  Check(q4x_ple_conv_bf16(gvn, gv, w_.conv1d, state_.data(), h, rows, static_cast<int>(channels), cfg_.conv_kernel,
                          cfg_.conv_dilation, stream),
        "conv");
}

}  // namespace r4dx::model::qwen4exp
