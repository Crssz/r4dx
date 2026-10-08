// Model::RunChunk's PP-emulate composition (docs/pp-prefill.md): the chunk's prologue, layers [0, split) as
// "stage A", the inter-stage carry through pinned host staging, layers [split, N) as "stage B", the epilogue.
// Everything here is Model member code (it reads the persistent buffers and the KV caches directly) but nothing
// of it runs unless Model::SetPpEmulate switched the mode on.
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chunk_run.h"
#include "model.h"
#include "pp_plan.h"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// The poison byte: 0xFF is NaN in bf16, f16 and fp32 and in fp8 e4m3, and -1 as int8 -- every consumer of the
// stage-boundary buffers turns a byte that was not carried into something that cannot compare equal.
constexpr int kPoison = 0xFF;

}  // namespace

int64_t Model::PpEmulateRequest(int option) {
  if (option > 0) return option;
  if (option == 0) return 0;
  std::string env;
#ifdef _MSC_VER
  char* buf = nullptr;
  size_t len = 0;
  if (_dupenv_s(&buf, &len, "R4DX_PP_EMULATE") == 0 && buf != nullptr) env = buf;
  std::free(buf);
#else
  if (const char* e = std::getenv("R4DX_PP_EMULATE")) env = e;
#endif
  const int64_t split = pp::ParseEmulateSplit(env.c_str());
  if (split < 0) {
    throw std::invalid_argument("Model::Load: R4DX_PP_EMULATE='" + env + "' is not 0 / off or a positive layer index");
  }
  return split;
}

void Model::SetPpEmulate(const PpEmulateConfig& config) {
  if (config.split < 0) throw std::invalid_argument("Model::SetPpEmulate: split must be >= 0");
  if (config.poison_arena < 0 || config.poison_arena > 2) {
    throw std::invalid_argument("Model::SetPpEmulate: poison_arena must be 0, 1 or 2");
  }
  if (config.split > 0) {
    if (pp_role_ != PpRole::kNone) {
      throw std::invalid_argument("Model::SetPpEmulate: not on a Model that is a stage of the real pipeline (PpAttach)");
    }
    if (comm_ != nullptr) {
      throw std::invalid_argument("Model::SetPpEmulate: not on a tensor-parallel rank (the pipeline replaces TP)");
    }
    if (container_.HasRotation()) {
      throw std::invalid_argument(
          "Model::SetPpEmulate: not on a rotated (quant2) container: the stack-entry and stack-exit rotations "
          "straddle the stages (docs/pp-prefill.md 4)");
    }
    if (!pp::ValidSplit(config.split, container_.NumLoadedLayers())) {
      throw std::invalid_argument("Model::SetPpEmulate: split " + std::to_string(config.split) +
                                  " must leave at least one of the " +
                                  std::to_string(container_.NumLoadedLayers()) + " layers on each side");
    }
  }
  stream_.Synchronize();  // the device is idle between calls; this makes a mid-call misuse harmless
  pp_emulate_ = config;
}

void Model::PoisonArena() {
  arena_.Reset();
  const size_t cap = arena_.capacity_bytes();
  uint8_t* p = arena_.Alloc<uint8_t>(cap);
  R4DX_HIP_CHECK(hipMemsetAsync(p, kPoison, cap, stream_.get()));
  arena_.Reset();
}

std::vector<float> Model::RunChunkPpEmulated(ChunkRun& r, int64_t split) {
  const bool timing = pp_emulate_.timing;
  PpChunkTimes t;
  t.pos = pos_;
  t.rows = r.T;
  r.timing = timing;
  r.poison_arena_layers = pp_emulate_.poison_arena >= 2 && !timing;
  const auto t_start = Clock::now();

  // Stage A: the prologue (ids, embedding gather, scopes, ...) and layers [0, split).
  ChunkPrologue(r);
  RunLayerRange(r, 0, split);
  stream_.Synchronize();  // a stage's chunk ends with the device idle (RunChunk's invariant)
  t.a_ms = Ms(t_start, Clock::now());

  // The boundary, through host memory.
  PpEmulateHop(r, split, &t);

  // Stage B: layers [split, N), then the epilogue (MTP priming, lm_head, DFlash injection, pos_ += T).
  const auto t_b = Clock::now();
  RunLayerRange(r, split, r.num_layers);
  if (timing) {
    stream_.Synchronize();
    t.b_ms = Ms(t_b, Clock::now());
  }
  const auto t_e = Clock::now();
  std::vector<float> logits = ChunkEpilogue(r);
  ++pp_emulated_chunks_;
  if (timing) {
    t.epilogue_ms = Ms(t_e, Clock::now());
    t.mtp_ms = r.mtp_ms;
    t.inject_ms = r.inject_ms;
    pp_times_.push_back(t);
  }
  return logits;
}

// What the two-GPU pipeline sends from stage A to stage B for one chunk, on one device (docs/pp-prefill.md 1.3,
// 3.1, 4): the residual stream and the fused-norm pair (buf_normed_, buf_normed_pre_) -- three buffers of T x
// hidden bf16-sized rows --, the DFlash feature columns stage A's target layers captured (a contiguous prefix of
// the capture's columns: the target layers are ascending), and the KV blocks stage A's attention layers wrote
// (whole blocks of one layer are one contiguous byte range). Exported to host, the device-side destinations
// poisoned, then imported: `cur` always lands in buf_a_ (as the pipeline's stage B will put it), whichever
// ping-pong buffer stage A's attention-layer parity left it in. The GDN state needs no hop here: it is one
// Model; its compact export / import is the pipeline's, at the end of a Prefill call.
void Model::PpEmulateHop(ChunkRun& r, int64_t split, PpChunkTimes* times) {
  const ModelConfig& cfg = container_.Config();
  const int64_t T = r.T;
  const int64_t hidden = r.hidden;
  const bool timing = pp_emulate_.timing;
  const PpEmulateConfig::Fault fault = pp_emulate_.fault;
  if (r.normed_in == nullptr) {
    throw std::logic_error("Model::PpEmulateHop: the layer before the split carries no fused norm");
  }
  const size_t carry_bytes = static_cast<size_t>(T * hidden) * sizeof(uint16_t);

  // ---- the host staging plan -------------------------------------------------------------------------------
  size_t total = 0;
  const auto take = [&total](size_t bytes) {
    const size_t off = total;
    total += (bytes + 255) & ~static_cast<size_t>(255);
    return off;
  };
  constexpr size_t kSlack = 64;  // kShiftCurOneByte reads one byte past the residual stream's block
  const size_t o_cur = take(carry_bytes + kSlack);
  const size_t o_norm = take(carry_bytes);
  const size_t o_pre = take(carry_bytes);
  const int64_t dfl_cols = static_cast<int64_t>(dflash_target_layers_.size());
  int64_t dfl_here = 0;  // captured columns that belong to stage A: target layers < split
  if (r.dflash_capture_active) {
    for (const int64_t l : dflash_target_layers_) {
      if (l < split) ++dfl_here;
    }
  }
  const size_t dfl_pitch = static_cast<size_t>(dfl_cols * hidden) * sizeof(uint16_t);
  const size_t dfl_width = static_cast<size_t>(dfl_here * hidden) * sizeof(uint16_t);
  const size_t o_dfl = take(dfl_width * static_cast<size_t>(T));
  struct KvPiece {
    attention::PagedKvCache* kv;
    size_t dev_off, bytes, host_off;
  };
  std::vector<KvPiece> kvs;
  size_t kv_bytes = 0;
  for (int64_t i = 0; i < split; ++i) {
    if (cfg.IsGdnLayer(i)) continue;
    attention::PagedKvCache& kv = *kv_caches_[static_cast<size_t>(i)];
    const pp::KvBlockRange br = pp::KvBlocksTouched(pos_, T, kv.BlockSize());
    const size_t stride = static_cast<size_t>(kv.KvBlockStride());
    const size_t bytes = static_cast<size_t>(br.Blocks()) * stride;
    kvs.push_back({&kv, static_cast<size_t>(br.block0) * stride, bytes, take(bytes)});
    kv_bytes += bytes;
  }
  stream_.Synchronize();  // stage A is done (the caller synchronized; this keeps the hop self-contained)
  if (pp_host_.bytes() < total) pp_host_ = core::PinnedBuffer<uint8_t>(total, hipHostMallocPortable);
  uint8_t* const host = pp_host_.data();
  const hipStream_t s = stream_.get();

  // ---- stage A's side: export ------------------------------------------------------------------------------
  const auto t0 = Clock::now();
  R4DX_HIP_CHECK(hipMemcpyAsync(host + o_cur, r.cur, carry_bytes, hipMemcpyDeviceToHost, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(host + o_norm, buf_normed_.data(), carry_bytes, hipMemcpyDeviceToHost, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(host + o_pre, buf_normed_pre_.data(), carry_bytes, hipMemcpyDeviceToHost, s));
  if (dfl_here > 0) {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(host + o_dfl, dfl_width, dflash_features_dev_.data(), dfl_pitch, dfl_width,
                                    static_cast<size_t>(T), hipMemcpyDeviceToHost, s));
  }
  for (const KvPiece& k : kvs) {
    R4DX_HIP_CHECK(hipMemcpyAsync(host + k.host_off, k.kv->Data() + k.dev_off, k.bytes, hipMemcpyDeviceToHost, s));
  }
  stream_.Synchronize();  // the host owns the stage's output now
  const auto t1 = Clock::now();

  // ---- stage B's side: what it must not already have, then the import ---------------------------------------
  // (Timing runs skip the poison fill so the hop is timed as a copy, not as a copy and a memset.)
  if (!timing) {
    R4DX_HIP_CHECK(hipMemsetAsync(buf_a_.data(), kPoison, carry_bytes, s));
    R4DX_HIP_CHECK(hipMemsetAsync(buf_b_.data(), kPoison, carry_bytes, s));
    R4DX_HIP_CHECK(hipMemsetAsync(buf_normed_.data(), kPoison, carry_bytes, s));
    R4DX_HIP_CHECK(hipMemsetAsync(buf_normed_pre_.data(), kPoison, carry_bytes, s));
    if (dfl_here > 0) {
      // The negative control that drops these columns zero-fills them instead: a 0xFF (NaN) feature column makes
      // the drafter's logits NaN, r4dx_topk16_f32 then returns its (-inf, INT32_MAX) sentinel ids, and the host
      // selector walk indexes its codebooks with INT32_MAX (an access violation, no exception). Zeros are finite,
      // so the drafter runs, and the ring it leaves is not the monolithic run's -- which is what the control needs.
      const int fill = fault == PpEmulateConfig::Fault::kSkipDflash ? 0 : kPoison;
      R4DX_HIP_CHECK(hipMemset2DAsync(dflash_features_dev_.data(), dfl_pitch, fill, dfl_width,
                                      static_cast<size_t>(T), s));
    }
    for (const KvPiece& k : kvs) R4DX_HIP_CHECK(hipMemsetAsync(k.kv->Data() + k.dev_off, kPoison, k.bytes, s));
    if (pp_emulate_.poison_arena >= 1) PoisonArena();
  }
  const size_t cur_shift = fault == PpEmulateConfig::Fault::kShiftCurOneByte ? 1 : 0;
  R4DX_HIP_CHECK(hipMemcpyAsync(buf_a_.data(), host + o_cur + cur_shift, carry_bytes, hipMemcpyHostToDevice, s));
  if (fault != PpEmulateConfig::Fault::kSkipNormed) {
    R4DX_HIP_CHECK(hipMemcpyAsync(buf_normed_.data(), host + o_norm, carry_bytes, hipMemcpyHostToDevice, s));
    R4DX_HIP_CHECK(hipMemcpyAsync(buf_normed_pre_.data(), host + o_pre, carry_bytes, hipMemcpyHostToDevice, s));
  }
  if (dfl_here > 0 && fault != PpEmulateConfig::Fault::kSkipDflash) {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dflash_features_dev_.data(), dfl_pitch, host + o_dfl, dfl_width, dfl_width,
                                    static_cast<size_t>(T), hipMemcpyHostToDevice, s));
  }
  if (fault != PpEmulateConfig::Fault::kSkipKv) {
    for (const KvPiece& k : kvs) {
      R4DX_HIP_CHECK(hipMemcpyAsync(k.kv->Data() + k.dev_off, host + k.host_off, k.bytes, hipMemcpyHostToDevice, s));
    }
  }
  r.cur = buf_a_.data();
  r.other = buf_b_.data();
  stream_.Synchronize();
  const auto t2 = Clock::now();
  if (times != nullptr) {
    times->d2h_ms = Ms(t0, t1);
    times->h2d_ms = Ms(t1, t2);
    times->hop_bytes = static_cast<int64_t>(3 * carry_bytes + dfl_width * static_cast<size_t>(T) + kv_bytes);
  }
}

}  // namespace r4dx::model
