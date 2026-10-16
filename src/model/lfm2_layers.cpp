#include "lfm2_layers.h"

#include <stdexcept>
#include <string>

#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/lfm2_kernels.h"

namespace r4dx::model {

namespace {
inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }
}  // namespace

// ---- ShortConv mixer ------------------------------------------------------------------------------------------------

void Lfm2ConvLayer::Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out,
                            uint16_t* state, int64_t T, SpanAccumulator* prof) const {
  const int64_t C = cfg_.conv_dim;  // the rank's channels
  const int64_t s = P(stream);
  if (w_.in_proj.N != 3 * C || w_.out_proj.K != C) {
    throw std::logic_error("Lfm2ConvLayer: in_proj / out_proj shapes do not match conv_dim " + std::to_string(C));
  }
  uint16_t* bcx = arena.Alloc<uint16_t>(static_cast<size_t>(T * 3 * C), 16);
  uint16_t* y = arena.Alloc<uint16_t>(static_cast<size_t>(T * C), 16);
  ProfiledCall(prof, stream, "gemm:conv.in_proj", [&] { ApplyLinear(stream, arena, w_.in_proj, x_normed, bcx, T); });
  ProfiledCall(prof, stream, "conv.shortconv", [&] {
    r4dx_lfm2_shortconv_bf16(P(bcx), P(state), P(w_.weight.data()), P(y), static_cast<int>(T), static_cast<int>(C), 3 * C, s);
  });
  ProfiledCall(prof, stream, "gemm:conv.out_proj", [&] { ApplyLinear(stream, arena, w_.out_proj, y, out, T); });
  if (comm_ != nullptr) {
    ProfiledCall(prof, stream, "tp.allreduce", [&] { comm_->AllReduceSumBf16Rows(out, T, cfg_.hidden_size, stream); });
  }
}

// ---- dense SwiGLU (layers < num_dense_layers) -----------------------------------------------------------------------

void Lfm2DenseMlp::Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out, int64_t T,
                           SpanAccumulator* prof) const {
  const int64_t I = cfg_.intermediate_size;  // the rank's
  const int64_t s = P(stream);
  uint16_t* gu = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * I), 16);
  uint16_t* act = arena.Alloc<uint16_t>(static_cast<size_t>(T * I), 16);
  ProfiledCall(prof, stream, "gemm:mlp.gate_up", [&] { ApplyLinear(stream, arena, w_.gate_up, x_normed, gu, T); });
  ProfiledCall(prof, stream, "mlp.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), T, I, 2 * I, s); });
  ProfiledCall(prof, stream, "gemm:mlp.down", [&] { ApplyLinear(stream, arena, w_.down, act, out, T); });
  if (comm_ != nullptr) {
    ProfiledCall(prof, stream, "tp.allreduce", [&] { comm_->AllReduceSumBf16Rows(out, T, cfg_.hidden_size, stream); });
  }
}

// ---- MoE ------------------------------------------------------------------------------------------------------------

void Lfm2MoeFfn::Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out, int64_t T,
                         SpanAccumulator* prof, Lfm2MoePath path, Lfm2MoeDebug* dbg,
                         const std::vector<int32_t>* forced_ids, const std::vector<float>* forced_w) const {
  const int64_t H = cfg_.hidden_size;
  const int64_t E = cfg_.num_experts;
  const int64_t K = cfg_.num_experts_per_tok;
  const int64_t I = cfg_.moe_intermediate_size;  // the rank's
  const int64_t Pn = T * K;                      // (token, slot) pairs
  const int64_t s = P(stream);
  if (host_ == nullptr || static_cast<int64_t>(host_->ids.size()) < Pn ||
      static_cast<int64_t>(host_->offsets.size()) < E + 1) {
    throw std::logic_error("Lfm2MoeFfn: the pinned readback buffers are missing or too small");
  }
  if (static_cast<int64_t>(w_.gate_up.size()) != E || static_cast<int64_t>(w_.down.size()) != E) {
    throw std::logic_error("Lfm2MoeFfn: expert table size != num_experts");
  }

  // ---- router (fp32, replicated: every rank computes identical ids / weights from identical bits) ----
  int32_t* ids = arena.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
  float* wts = arena.Alloc<float>(static_cast<size_t>(Pn), 16);
  ProfiledCall(prof, stream, "moe.router", [&] {
    r4dx_lfm2_router_f32(P(x_normed), P(w_.router.data()), P(w_.expert_bias.data()), P(ids), P(wts), static_cast<int>(T),
                         static_cast<int>(H), static_cast<int>(E), static_cast<int>(K),
                         static_cast<float>(cfg_.routed_scaling_factor), H, s);
  });
  if (forced_ids != nullptr || forced_w != nullptr) {
    if (forced_ids == nullptr || forced_w == nullptr || static_cast<int64_t>(forced_ids->size()) != Pn ||
        static_cast<int64_t>(forced_w->size()) != Pn) {
      throw std::invalid_argument("Lfm2MoeFfn: forced ids and weights must both be [T, top_k]");
    }
    for (int32_t e : *forced_ids) {
      if (e < 0 || e >= E) throw std::invalid_argument("Lfm2MoeFfn: forced expert id out of range");
    }
    R4DX_HIP_CHECK(hipMemcpyAsync(ids, forced_ids->data(), static_cast<size_t>(Pn) * sizeof(int32_t), hipMemcpyHostToDevice,
                                  stream));
    R4DX_HIP_CHECK(hipMemcpyAsync(wts, forced_w->data(), static_cast<size_t>(Pn) * sizeof(float), hipMemcpyHostToDevice,
                                  stream));
    R4DX_HIP_CHECK(hipStreamSynchronize(stream));  // pageable sources
  }

  const bool per_slot = path == Lfm2MoePath::kPerSlot || (path == Lfm2MoePath::kAuto && T == 1);
  uint16_t* gu = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * I), 16);  // an expert holds at most T rows
  uint16_t* act = arena.Alloc<uint16_t>(static_cast<size_t>(T * I), 16);
  uint16_t* ys = arena.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);      // one output row per pair

  if (per_slot) {
    // ---- decode path: read the T * top_k ids back (one sync), then M = 1 expert GEMMs per pair, slot layout ----
    ProfiledCall(prof, stream, "moe.readback", [&] {
      R4DX_HIP_CHECK(hipMemcpyAsync(host_->ids.data(), ids, static_cast<size_t>(Pn) * sizeof(int32_t),
                                    hipMemcpyDeviceToHost, stream));
      R4DX_HIP_CHECK(hipStreamSynchronize(stream));
    });
    // The experts' GEMM scratch (trellis: input transform + split-group partials) is given back after each pair: every
    // launch is on `stream`, so the next pair's GEMMs only overwrite it after this pair's have read it.
    const size_t mark = arena.Mark();
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t j = 0; j < K; ++j) {
        arena.Rewind(mark);
        const int32_t e = host_->ids.data()[t * K + j];
        if (e < 0 || e >= E) throw std::runtime_error("Lfm2MoeFfn: router returned expert id " + std::to_string(e));
        const size_t ue = static_cast<size_t>(e);
        ProfiledCall(prof, stream, "gemm:moe.gate_up", [&] {
          ApplyLinear(stream, arena, w_.gate_up[ue], x_normed + t * H, gu, 1);
        });
        ProfiledCall(prof, stream, "moe.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), 1, I, 2 * I, s); });
        ProfiledCall(prof, stream, "gemm:moe.down", [&] {
          ApplyLinear(stream, arena, w_.down[ue], act, ys + (t * K + j) * H, 1);
        });
      }
    }
    ProfiledCall(prof, stream, "moe.combine", [&] {
      r4dx_lfm2_moe_combine_bf16(P(ys), 0, P(wts), P(out), static_cast<int>(T), static_cast<int>(K), static_cast<int>(H), s);
    });
  } else {
    // ---- sorted path: deterministic counting sort, one readback of the offsets, per-expert GEMMs over its rows ----
    int32_t* counts = arena.Alloc<int32_t>(static_cast<size_t>(E), 16);
    int32_t* offsets = arena.Alloc<int32_t>(static_cast<size_t>(E + 1), 16);
    int32_t* sorted = arena.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
    int32_t* pos_of_pair = arena.Alloc<int32_t>(static_cast<size_t>(Pn), 16);
    uint16_t* xs = arena.Alloc<uint16_t>(static_cast<size_t>(Pn * H), 16);
    ProfiledCall(prof, stream, "moe.sort", [&] {
      r4dx_lfm2_moe_sort(P(ids), static_cast<int>(T), static_cast<int>(K), static_cast<int>(E), P(counts), P(offsets),
                         P(sorted), P(pos_of_pair), s);
    });
    ProfiledCall(prof, stream, "moe.gather", [&] {
      r4dx_lfm2_moe_gather_bf16(P(x_normed), P(sorted), P(xs), static_cast<int>(Pn), static_cast<int>(K),
                                static_cast<int>(H), H, P(offsets + E), s);
    });
    ProfiledCall(prof, stream, "moe.readback", [&] {
      R4DX_HIP_CHECK(hipMemcpyAsync(host_->offsets.data(), offsets, static_cast<size_t>(E + 1) * sizeof(int32_t),
                                    hipMemcpyDeviceToHost, stream));
      R4DX_HIP_CHECK(hipStreamSynchronize(stream));
    });
    const int32_t* off = host_->offsets.data();
    if (off[0] != 0 || off[E] != Pn) {
      throw std::runtime_error("Lfm2MoeFfn: the expert sort covers " + std::to_string(off[E]) + " of " +
                               std::to_string(Pn) + " pairs (a router id out of range?)");
    }
    // One expert's GEMM scratch at a time (see the per-slot path): the layer's arena peak does not grow with E.
    const size_t mark = arena.Mark();
    for (int64_t e = 0; e < E; ++e) {
      const int64_t r0 = off[e], c = off[e + 1] - off[e];
      if (c == 0) continue;
      arena.Rewind(mark);
      if (c < 0 || c > T) throw std::runtime_error("Lfm2MoeFfn: bad expert count " + std::to_string(c));
      const size_t ue = static_cast<size_t>(e);
      ProfiledCall(prof, stream, "gemm:moe.gate_up", [&] { ApplyLinear(stream, arena, w_.gate_up[ue], xs + r0 * H, gu, c); });
      ProfiledCall(prof, stream, "moe.silu_mul", [&] { r4dx_silu_mul_bf16(P(gu), P(act), c, I, 2 * I, s); });
      ProfiledCall(prof, stream, "gemm:moe.down", [&] { ApplyLinear(stream, arena, w_.down[ue], act, ys + r0 * H, c); });
    }
    ProfiledCall(prof, stream, "moe.combine", [&] {
      r4dx_lfm2_moe_combine_bf16(P(ys), P(pos_of_pair), P(wts), P(out), static_cast<int>(T), static_cast<int>(K),
                                 static_cast<int>(H), s);
    });
  }
  if (dbg != nullptr) {
    R4DX_HIP_CHECK(hipStreamSynchronize(stream));
    dbg->ids.resize(static_cast<size_t>(Pn));
    dbg->w.resize(static_cast<size_t>(Pn));
    R4DX_HIP_CHECK(hipMemcpy(dbg->ids.data(), ids, static_cast<size_t>(Pn) * sizeof(int32_t), hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(dbg->w.data(), wts, static_cast<size_t>(Pn) * sizeof(float), hipMemcpyDeviceToHost));
  }
  if (comm_ != nullptr) {
    ProfiledCall(prof, stream, "tp.allreduce", [&] { comm_->AllReduceSumBf16Rows(out, T, H, stream); });
  }
}

}  // namespace r4dx::model
