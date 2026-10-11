#include "r4dx/model/qwen4exp/hc_conn.h"

#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"

namespace r4dx::model::qwen4exp {

namespace {

constexpr const char* kCtx = "qwen4exp::HyperConn";

[[noreturn]] void Fail(const std::string& msg) { qwen4exp::Fail(kCtx, msg); }
void Check(int code, const char* what) { qwen4exp::Check(kCtx, code, what); }

}  // namespace

HyperConn::HyperConn(const HyperConnConfig& cfg, const HyperConnWeights& w) : cfg_(cfg), w_(w) {
  if (cfg.hc < 1 || cfg.hidden < 1 || cfg.lowrank < 1) Fail("non-positive dimension in the config");
  if (w.norm == nullptr) Fail("missing norm weight");
  const int64_t wide = static_cast<int64_t>(cfg.hc) * cfg.hidden;
  CheckLinear(kCtx, w.down, "down", cfg.lowrank, wide);
  CheckLinear(kCtx, w.up, "up", wide, cfg.lowrank);
  if (w.inject != nullptr) CheckLinear(kCtx, w.inject, "inject", cfg.hc, wide);
}

size_t HyperConn::ArenaBytes(int64_t rows) const {
  const size_t r = static_cast<size_t>(rows), wide = static_cast<size_t>(cfg_.hc) * cfg_.hidden;
  return Padded(r * wide * 2) +                              // n
         Padded(r * static_cast<size_t>(cfg_.lowrank) * 2) +  // silu(down(n) / hc)
         Padded(r * wide * 2) +                              // up(...)
         Padded(r * static_cast<size_t>(cfg_.hc) * 2) +      // inj of ApplyBlock
         Padded(r * static_cast<size_t>(cfg_.hidden) * 2) +   // x of ApplyBlock
         Padded(r * static_cast<size_t>(cfg_.hidden) * 2);    // block_out of ApplyBlock
}

void HyperConn::Prep(hipStream_t stream, core::Arena& arena, const uint16_t* h, int64_t rows, uint16_t* x_out,
                     uint16_t* inj_out) const {
  if (rows <= 0) return;
  if ((w_.inject != nullptr) != (inj_out != nullptr)) {
    Fail(w_.inject != nullptr ? "inj_out is required (this block has inject weights)"
                              : "inj_out given but this block has no inject weights");
  }
  const int64_t wide = static_cast<int64_t>(cfg_.hc) * cfg_.hidden;
  uint16_t* n = arena.Alloc<uint16_t>(static_cast<size_t>(rows * wide));
  uint16_t* mid = arena.Alloc<uint16_t>(static_cast<size_t>(rows * cfg_.lowrank));
  uint16_t* up = arena.Alloc<uint16_t>(static_cast<size_t>(rows * wide));

  Check(q4x_grouped_rmsnorm_bf16(h, w_.norm, n, rows, cfg_.hc, cfg_.hidden, cfg_.eps, stream), "hc norm");
  ApplyBf16Linear(stream, arena, *w_.down, n, mid, rows);
  Check(q4x_hc_silu_scaled_bf16(mid, mid, rows * cfg_.lowrank, 1.0f / static_cast<float>(cfg_.hc), stream),
        "hc silu");
  ApplyBf16Linear(stream, arena, *w_.up, mid, up, rows);
  Check(q4x_hc_mix_bf16(n, up, x_out, rows, cfg_.hc, cfg_.hidden, stream), "hc mix");
  if (w_.inject != nullptr) {
    ApplyBf16Linear(stream, arena, *w_.inject, n, inj_out, rows);
    Check(q4x_hc_inject_gate_bf16(inj_out, inj_out, rows * cfg_.hc, cfg_.hc, stream), "hc inject gate");
  }
}

void HyperConn::Inject(hipStream_t stream, const uint16_t* h, const uint16_t* block_out, const uint16_t* inj,
                       int64_t rows, uint16_t* h_out) const {
  Check(q4x_hc_inject_bf16(h, block_out, inj, h_out, rows, cfg_.hc, cfg_.hidden, stream), "hc inject");
}

void HyperConn::Mix(hipStream_t stream, core::Arena& arena, const uint16_t* h, int64_t rows,
                    uint16_t* x_out) const {
  if (w_.inject != nullptr) Fail("Mix is the final mixer; this block has inject weights (use Prep / ApplyBlock)");
  Prep(stream, arena, h, rows, x_out, nullptr);
}

void HyperConn::ApplyBlock(hipStream_t stream, core::Arena& arena, uint16_t* h, int64_t rows,
                           const BlockFn& block) const {
  if (w_.inject == nullptr) Fail("ApplyBlock needs inject weights");
  if (rows <= 0) return;
  uint16_t* x = arena.Alloc<uint16_t>(static_cast<size_t>(rows * cfg_.hidden));
  uint16_t* inj = arena.Alloc<uint16_t>(static_cast<size_t>(rows * cfg_.hc));
  uint16_t* block_out = arena.Alloc<uint16_t>(static_cast<size_t>(rows * cfg_.hidden));
  Prep(stream, arena, h, rows, x, inj);
  block(x, block_out);
  Inject(stream, h, block_out, inj, rows, h);
}

void EmbedRepeat(hipStream_t stream, const uint16_t* table, int64_t vocab, const int32_t* ids, int64_t rows,
                 int hidden, int hc, uint16_t* out) {
  Check(q4x_embed_gather_repeat_bf16(table, ids, out, rows, vocab, hidden, hc, stream), "embed gather");
}

}  // namespace r4dx::model::qwen4exp
