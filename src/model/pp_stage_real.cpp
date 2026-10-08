// Model's side of the REAL two-GPU pipeline-parallel prefill (docs/pp-prefill.md, Phase 2): the two chunk
// compositions (stage A: prologue + layers [0, split) + export into a channel slot; stage B: prologue + import from a
// slot + layers [split, N) + epilogue), and the state hand-off primitives PpModel (pp_model.cpp) drives -- the GDN live
// state in its compact wire form (with the conv-line pitch conversion between the stages' different window sizes), KV
// row ranges, the live-state digest, the sync state. Everything here is Model member code; nothing runs unless
// PpAttach made this Model a stage and PpSetActive engaged it.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "chunk_run.h"
#include "model.h"
#include "pp_plan.h"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

namespace {

uint64_t FnvUpdate(uint64_t h, const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}
constexpr uint64_t kFnvInit = 1469598103934665603ull;

}  // namespace

// ---- attachment ---------------------------------------------------------------------------------------------------

void Model::PpAttach(PpStageSetup setup) {
  if (setup.role == PpRole::kNone) {
    PpDetach();
    return;
  }
  if (comm_ != nullptr) {
    throw std::invalid_argument("Model::PpAttach: not on a tensor-parallel rank (the pipeline replaces TP)");
  }
  if (container_.HasRotation()) {
    throw std::invalid_argument(
        "Model::PpAttach: not on a rotated (quant2) container: the stack-entry and -exit rotations straddle the stages "
        "(docs/pp-prefill.md 4)");
  }
  if (pp_emulate_.split > 0) throw std::invalid_argument("Model::PpAttach: the PP-emulate mode is on");
  if (probe_ != nullptr) {
    throw std::invalid_argument("Model::PpAttach: not with R4DX_CLOCK_PROBE / R4DX_PROFILE_LINEARS (one probe, one stream)");
  }
  if (setup.channel == nullptr) throw std::invalid_argument("Model::PpAttach: no channel");
  if (setup.timeout_ms <= 0) throw std::invalid_argument("Model::PpAttach: timeout_ms must be positive");
  if (!pp::ValidSplit(setup.split, container_.NumLoadedLayers())) {
    throw std::invalid_argument("Model::PpAttach: split " + std::to_string(setup.split) + " must leave at least one of the " +
                                std::to_string(container_.NumLoadedLayers()) + " loaded layers on each side");
  }
  stream_.Synchronize();
  pp_setup_ = std::move(setup);
  pp_role_ = pp_setup_.role;
  pp_active_ = pp_role_ == PpRole::kStageA;
  if (pp_role_ == PpRole::kStageB && !pp_aux_stream_) pp_aux_stream_.emplace();
}

void Model::PpDetach() {
  stream_.Synchronize();
  if (pp_aux_stream_) pp_aux_stream_->Synchronize();
  pp_role_ = PpRole::kNone;
  pp_setup_ = PpStageSetup{};
  pp_active_ = false;
  pp_bounded_ = false;
}

void Model::PpSetActive(bool on) {
  if (pp_role_ == PpRole::kNone) {
    if (on) throw std::logic_error("Model::PpSetActive: not attached as a pipeline stage");
    return;
  }
  if (pp_role_ == PpRole::kStageA && !on) throw std::logic_error("Model::PpSetActive: stage A is always active");
  stream_.Synchronize();
  pp_active_ = on;
}

void Model::PpEnableBounding(int submit_layers, int max_inflight) {
  if (pp_role_ != PpRole::kStageA) throw std::logic_error("Model::PpEnableBounding: stage A only");
  if (submit_layers < 0 || submit_layers > 64 || max_inflight < 0 || max_inflight > 64) {
    throw std::invalid_argument("Model::PpEnableBounding: submit_layers and max_inflight must be in [0, 64]");
  }
  stream_.Synchronize();
  if (submit_layers == 0) {
    submit_ = tp::SubmitBounder();
    submit_layers_ = 0;
    pp_bounded_ = false;
    return;
  }
  submit_ = tp::SubmitBounder(max_inflight);
  submit_layers_ = submit_layers;
  pp_bounded_ = true;
}

void Model::PpSetSyncState(const PpSyncState& s) {
  if (pp_role_ != PpRole::kStageA) throw std::logic_error("Model::PpSetSyncState: stage A only");
  stream_.Synchronize();
  pos_ = s.pos;
  started_ = s.started;
  mrope_active_ = s.mrope_active;
  mrope_delta_ = s.mrope_delta;
  at_prefill_end_ = false;
}

// ---- geometry -------------------------------------------------------------------------------------------------------

std::vector<int64_t> Model::PpAttnLayers(int64_t split) const {
  std::vector<int64_t> out;
  const ModelConfig& cfg = container_.Config();
  for (int64_t i = 0; i < split && i < static_cast<int64_t>(kv_caches_.size()); ++i) {
    if (!cfg.IsGdnLayer(i) && kv_caches_[static_cast<size_t>(i)]) out.push_back(i);
  }
  return out;
}

int64_t Model::PpKvBlockSize() const {
  for (const auto& kv : kv_caches_) {
    if (kv) return kv->BlockSize();
  }
  throw std::logic_error("Model::PpKvBlockSize: no attention layer is loaded");
}

size_t Model::PpKvBlockStrideBytes() const {
  for (const auto& kv : kv_caches_) {
    if (kv) return static_cast<size_t>(kv->KvBlockStride());
  }
  throw std::logic_error("Model::PpKvBlockStrideBytes: no attention layer is loaded");
}

namespace {

// The wire form of one GDN layer's live state, from its manager's own dimensions.
pp::GdnWire WireOf(const GdnStateManager& g) {
  return pp::MakeGdnWire(g.NumHeads(), g.ValueDim(), g.KeyDim(), g.ConvDim(), g.ConvHistory() + 1);
}

// Moves the live conv history ([conv_dim][conv_width - 1] bf16, the compact wire form) between the host and the live line
// ([conv_dim][StateLenMax()], history at offset 0). A line whose pitch is exactly the history length (stage A, and any
// Model that does not speculate: state_len_max = conv_width - 2 + max_decode_window = conv_width - 1) IS the wire form: one
// contiguous copy. A speculating Model's longer lines need the 2D copy (row pitch StateLenMax(), 2 x 3 bytes per row).
void CopyConvHistory(GdnStateManager& g, uint8_t* host, bool to_host, hipStream_t s) {
  const size_t hist = static_cast<size_t>(g.ConvHistory()) * sizeof(uint16_t);
  const size_t rows = static_cast<size_t>(g.ConvDim());
  const size_t pitch = static_cast<size_t>(g.StateLenMax()) * sizeof(uint16_t);
  uint16_t* line = g.ConvLinePtr(0);
  if (pitch == hist) {
    R4DX_HIP_CHECK(hipMemcpyAsync(to_host ? static_cast<void*>(host) : static_cast<void*>(line),
                                   to_host ? static_cast<const void*>(line) : static_cast<const void*>(host), hist * rows,
                                   to_host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice, s));
  } else if (to_host) {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(host, hist, line, pitch, hist, rows, hipMemcpyDeviceToHost, s));
  } else {
    // The history lands at offset 0 of the live line (where a prefill / the first step of a fresh seed reads it); the
    // line's other entries (this stage's longer window) are not part of the live state and keep what they hold.
    R4DX_HIP_CHECK(hipMemcpy2DAsync(line, pitch, host, hist, hist, rows, hipMemcpyHostToDevice, s));
  }
}

}  // namespace

// ---- the GDN live state -----------------------------------------------------------------------------------------------

size_t Model::PpGdnWireBytes(int64_t split) const {
  size_t n = 0;
  for (int64_t i = 0; i < split && i < static_cast<int64_t>(gdn_states_.size()); ++i) {
    if (gdn_states_[static_cast<size_t>(i)]) n += WireOf(*gdn_states_[static_cast<size_t>(i)]).PerLayer();
  }
  return n;
}

void Model::PpExportGdn(uint8_t* host, int64_t split) {
  stream_.Synchronize();  // the device is idle between calls; this makes a mid-call misuse harmless
  size_t off = 0;
  const hipStream_t s = stream_.get();
  for (int64_t i = 0; i < split && i < static_cast<int64_t>(gdn_states_.size()); ++i) {
    if (!gdn_states_[static_cast<size_t>(i)]) continue;
    GdnStateManager& g = *gdn_states_[static_cast<size_t>(i)];
    const pp::GdnWire w = WireOf(g);
    R4DX_HIP_CHECK(hipMemcpyAsync(host + off, g.RecurrentSlotPtr(g.SlotForSeq(0)), w.recurrent_bytes,
                                   hipMemcpyDeviceToHost, s));
    CopyConvHistory(g, host + off + w.recurrent_bytes, /*to_host=*/true, s);
    off += w.PerLayer();
  }
  stream_.Synchronize();
}

namespace {

// Enqueues the compact -> live-layout copies of layers [0, split) on `s`.
void EnqueueGdnImport(std::vector<std::optional<GdnStateManager>>& states, const uint8_t* host, int64_t split,
                      hipStream_t s) {
  size_t off = 0;
  for (int64_t i = 0; i < split && i < static_cast<int64_t>(states.size()); ++i) {
    if (!states[static_cast<size_t>(i)]) continue;
    GdnStateManager& g = *states[static_cast<size_t>(i)];
    const pp::GdnWire w = WireOf(g);
    R4DX_HIP_CHECK(hipMemcpyAsync(g.RecurrentSlotPtr(g.SlotForSeq(0)), host + off, w.recurrent_bytes,
                                   hipMemcpyHostToDevice, s));
    // (const_cast: the helper takes one signature for both directions; the import only reads `host`.)
    CopyConvHistory(g, const_cast<uint8_t*>(host) + off + w.recurrent_bytes, /*to_host=*/false, s);
    off += w.PerLayer();
  }
}

}  // namespace

void Model::PpImportGdn(const uint8_t* host, int64_t split) {
  stream_.Synchronize();
  EnqueueGdnImport(gdn_states_, host, split, stream_.get());
  stream_.Synchronize();
}

void Model::PpImportGdnAsync(const uint8_t* host, int64_t split) {
  if (!pp_aux_stream_) throw std::logic_error("Model::PpImportGdnAsync: stage B only");
  EnqueueGdnImport(gdn_states_, host, split, pp_aux_stream_->get());
}

void Model::PpImportFence() {
  if (pp_aux_stream_) pp_aux_stream_->Synchronize();
}

// ---- KV rows --------------------------------------------------------------------------------------------------------

size_t Model::PpKvWireBytes(int64_t split, int64_t row0, int64_t row1) const {
  if (row1 <= row0) return 0;
  size_t n = 0;
  const std::vector<int64_t> layers = PpAttnLayers(split);
  if (layers.empty()) return 0;
  int64_t b0 = 0, blocks = 0;
  pp::KvBlockSpan(row0, row1 - row0, PpKvBlockSize(), &b0, &blocks);
  return layers.size() * static_cast<size_t>(blocks) * PpKvBlockStrideBytes();
}

void Model::PpExportKvRows(uint8_t* host, int64_t split, int64_t row0, int64_t row1) {
  stream_.Synchronize();
  if (row1 <= row0) return;
  int64_t b0 = 0, blocks = 0;
  pp::KvBlockSpan(row0, row1 - row0, PpKvBlockSize(), &b0, &blocks);
  const size_t stride = PpKvBlockStrideBytes();
  size_t off = 0;
  for (const int64_t i : PpAttnLayers(split)) {
    attention::PagedKvCache& kv = *kv_caches_[static_cast<size_t>(i)];
    R4DX_HIP_CHECK(hipMemcpyAsync(host + off, kv.Data() + static_cast<size_t>(b0) * stride,
                                   static_cast<size_t>(blocks) * stride, hipMemcpyDeviceToHost, stream_.get()));
    off += static_cast<size_t>(blocks) * stride;
  }
  stream_.Synchronize();
}

void Model::PpImportKvRows(const uint8_t* host, int64_t split, int64_t row0, int64_t row1) {
  stream_.Synchronize();
  if (row1 <= row0) return;
  int64_t b0 = 0, blocks = 0;
  pp::KvBlockSpan(row0, row1 - row0, PpKvBlockSize(), &b0, &blocks);
  const size_t stride = PpKvBlockStrideBytes();
  size_t off = 0;
  for (const int64_t i : PpAttnLayers(split)) {
    attention::PagedKvCache& kv = *kv_caches_[static_cast<size_t>(i)];
    R4DX_HIP_CHECK(hipMemcpyAsync(kv.Data() + static_cast<size_t>(b0) * stride, host + off,
                                   static_cast<size_t>(blocks) * stride, hipMemcpyHostToDevice, stream_.get()));
    off += static_cast<size_t>(blocks) * stride;
  }
  stream_.Synchronize();
}

// ---- the live-state digest ------------------------------------------------------------------------------------------

std::vector<std::pair<std::string, uint64_t>> Model::PpLiveDigest(int64_t split) {
  stream_.Synchronize();
  std::vector<std::pair<std::string, uint64_t>> out;
  out.emplace_back("pos", static_cast<uint64_t>(pos_));
  const ModelConfig& cfg = container_.Config();
  for (int64_t i = 0; i < split && i < static_cast<int64_t>(gdn_states_.size()); ++i) {
    const size_t idx = static_cast<size_t>(i);
    if (gdn_states_[idx]) {
      GdnStateManager& g = *gdn_states_[idx];
      const pp::GdnWire w = WireOf(g);
      std::vector<uint8_t> host(w.recurrent_bytes + w.conv_bytes);
      R4DX_HIP_CHECK(hipMemcpy(host.data(), g.RecurrentSlotPtr(g.SlotForSeq(0)), w.recurrent_bytes,
                                hipMemcpyDeviceToHost));
      CopyConvHistory(g, host.data() + w.recurrent_bytes, /*to_host=*/true, stream_.get());
      stream_.Synchronize();
      out.emplace_back("gdn." + std::to_string(i), FnvUpdate(kFnvInit, host.data(), host.size()));
    } else if (!cfg.IsGdnLayer(i) && kv_caches_[idx]) {
      attention::PagedKvCache& kv = *kv_caches_[idx];
      const int64_t bs = kv.BlockSize();
      const size_t stride = static_cast<size_t>(kv.KvBlockStride());
      const int64_t full = pos_ / bs;
      const int64_t rem = pos_ % bs;
      uint64_t h = kFnvInit;
      constexpr int64_t kGroup = 256;  // blocks per host copy (8 MiB at 32 KiB per block)
      std::vector<uint8_t> host(static_cast<size_t>(kGroup) * stride);
      for (int64_t b = 0; b < full; b += kGroup) {
        const int64_t n = std::min<int64_t>(kGroup, full - b);
        R4DX_HIP_CHECK(hipMemcpy(host.data(), kv.Data() + static_cast<size_t>(b) * stride, static_cast<size_t>(n) * stride,
                                  hipMemcpyDeviceToHost));
        h = FnvUpdate(h, host.data(), static_cast<size_t>(n) * stride);
      }
      if (rem > 0) {
        R4DX_HIP_CHECK(hipMemcpy(host.data(), kv.Data() + static_cast<size_t>(full) * stride, stride, hipMemcpyDeviceToHost));
        const size_t head_stride = static_cast<size_t>(kv.KvHeadStride());
        const size_t row_bytes = static_cast<size_t>(2 * kv.HeadDim());
        for (int hd = 0; hd < kv.KvHeads(); ++hd) {
          h = FnvUpdate(h, host.data() + static_cast<size_t>(hd) * head_stride, static_cast<size_t>(rem) * row_bytes);
        }
      }
      out.emplace_back("kv." + std::to_string(i), h);
    }
  }
  return out;
}

// ---- the chunk compositions -------------------------------------------------------------------------------------------

std::vector<float> Model::RunChunkPpStageA(ChunkRun& r) {
  const int64_t split = pp_setup_.split;
  const int64_t T = r.T;
  const int64_t hidden = r.hidden;
  const std::chrono::milliseconds timeout(pp_setup_.timeout_ms);
  // Layers [0, split): the last one's Mlp fuses layer `split`'s input rmsnorm (stage A holds that layer's weight, and
  // has_next_layer is `i + 1 < NumLoadedLayers()`), so `normed_in` leaves this range exactly as in the monolithic run.
  ChunkPrologue(r);
  RunLayerRange(r, 0, split);
  stream_.Synchronize();  // a stage's chunk ends with the device idle (RunChunk's invariant)
  if (r.normed_in == nullptr) {
    throw std::logic_error("Model::RunChunkPpStageA: the layer before the split carries no fused norm");
  }
  r.fakeq_scope->End();
  r.fakeqw_scope->End();

  int64_t dfl_here = 0;  // captured columns of this chunk (stage A's whole target list: all of it is below the split)
  if (r.dflash_capture_active) {
    for (const int64_t l : dflash_target_layers_) {
      if (l < split) ++dfl_here;
    }
  }
  const pp::SlotLayout lay = pp::MakeSlotLayout(pos_, T, hidden, dfl_here, PpAttnLayers(split), PpKvBlockSize(),
                                                PpKvBlockStrideBytes());
  pp::StageChannel::Slot* slot = pp_setup_.channel->AcquireFree(timeout);
  if (lay.total > slot->capacity) {
    throw std::logic_error("Model::RunChunkPpStageA: a chunk's payload (" + std::to_string(lay.total) +
                           " bytes) does not fit a channel slot (" + std::to_string(slot->capacity) + ")");
  }
  uint8_t* const host = slot->data;
  const hipStream_t s = stream_.get();
  R4DX_HIP_CHECK(hipMemcpyAsync(host + lay.off_cur, r.cur, lay.carry_bytes, hipMemcpyDeviceToHost, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(host + lay.off_norm, buf_normed_.data(), lay.carry_bytes, hipMemcpyDeviceToHost, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(host + lay.off_pre, buf_normed_pre_.data(), lay.carry_bytes, hipMemcpyDeviceToHost, s));
  if (dfl_here > 0) {
    const size_t pitch = static_cast<size_t>(dflash_target_layers_.size() * static_cast<size_t>(hidden)) * sizeof(uint16_t);
    const size_t width = static_cast<size_t>(dfl_here * hidden) * sizeof(uint16_t);
    R4DX_HIP_CHECK(hipMemcpy2DAsync(host + lay.off_dfl, width, dflash_features_dev_.data(), pitch, width,
                                     static_cast<size_t>(T), hipMemcpyDeviceToHost, s));
  }
  for (const pp::KvPiece& k : lay.kv) {
    attention::PagedKvCache& kv = *kv_caches_[static_cast<size_t>(k.layer)];
    R4DX_HIP_CHECK(hipMemcpyAsync(host + k.host_off, kv.Data() + k.dev_off, k.bytes, hipMemcpyDeviceToHost, s));
  }
  stream_.Synchronize();  // the host owns the stage's output now
  slot->hdr.pos = pos_;
  slot->hdr.rows = T;
  slot->hdr.dfl_cols = dfl_here;
  slot->hdr.last = r.want_logits;  // Prefill passes want_logits on the call's last chunk only
  slot->hdr.bytes = lay.total;
  pp_setup_.channel->Publish(slot);

  pos_ += T;
  started_ = true;
  ++pp_staged_chunks_;
  if (pp_setup_.on_chunk) pp_setup_.on_chunk();
  return {};
}

std::vector<float> Model::RunChunkPpStageB(ChunkRun& r) {
  const int64_t split = pp_setup_.split;
  const int64_t T = r.T;
  const int64_t hidden = r.hidden;
  const std::chrono::milliseconds timeout(pp_setup_.timeout_ms);
  // The prologue first: it overlaps stage A's chunk (it uploads this chunk's positions and gathers an embedding that
  // the import then overwrites -- the cost of keeping the prologue a single piece of code for both stages).
  ChunkPrologue(r);
  pp::StageChannel::Slot* slot = pp_setup_.channel->AcquireFull(timeout);

  int64_t dfl_here = 0;
  if (r.dflash_capture_active) {
    for (const int64_t l : dflash_target_layers_) {
      if (l < split) ++dfl_here;
    }
  }
  const pp::SlotHeader& h = slot->hdr;
  if (h.pos != pos_ || h.rows != T || h.last != r.want_logits || h.dfl_cols != dfl_here) {
    throw std::runtime_error("Model::RunChunkPpStageB: the stages disagree on the chunk: stage A sent position " +
                             std::to_string(h.pos) + ", " + std::to_string(h.rows) + " rows, last " +
                             std::to_string(h.last) + ", " + std::to_string(h.dfl_cols) + " DFlash columns; stage B is at " +
                             std::to_string(pos_) + ", " + std::to_string(T) + " rows, last " +
                             std::to_string(r.want_logits) + ", " + std::to_string(dfl_here) + " columns");
  }
  const pp::SlotLayout lay = pp::MakeSlotLayout(pos_, T, hidden, dfl_here, PpAttnLayers(split), PpKvBlockSize(),
                                                PpKvBlockStrideBytes());
  if (lay.total != h.bytes) {
    throw std::runtime_error("Model::RunChunkPpStageB: the stages disagree on the payload size (" + std::to_string(h.bytes) +
                             " vs " + std::to_string(lay.total) + " bytes)");
  }
  if (r.want_logits && pp_setup_.before_last_chunk) pp_setup_.before_last_chunk();

  const uint8_t* const host = slot->data;
  const hipStream_t s = stream_.get();
  // `cur` always lands in buf_a_ (whichever ping-pong buffer stage A's attention-layer parity left it in).
  R4DX_HIP_CHECK(hipMemcpyAsync(buf_a_.data(), host + lay.off_cur, lay.carry_bytes, hipMemcpyHostToDevice, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(buf_normed_.data(), host + lay.off_norm, lay.carry_bytes, hipMemcpyHostToDevice, s));
  R4DX_HIP_CHECK(hipMemcpyAsync(buf_normed_pre_.data(), host + lay.off_pre, lay.carry_bytes, hipMemcpyHostToDevice, s));
  if (dfl_here > 0) {
    const size_t pitch = static_cast<size_t>(dflash_target_layers_.size() * static_cast<size_t>(hidden)) * sizeof(uint16_t);
    const size_t width = static_cast<size_t>(dfl_here * hidden) * sizeof(uint16_t);
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dflash_features_dev_.data(), pitch, host + lay.off_dfl, width, width,
                                     static_cast<size_t>(T), hipMemcpyHostToDevice, s));
  }
  for (const pp::KvPiece& k : lay.kv) {
    attention::PagedKvCache& kv = *kv_caches_[static_cast<size_t>(k.layer)];
    R4DX_HIP_CHECK(hipMemcpyAsync(kv.Data() + k.dev_off, host + k.host_off, k.bytes, hipMemcpyHostToDevice, s));
  }
  r.cur = buf_a_.data();
  r.other = buf_b_.data();
  r.normed_in = buf_normed_.data();
  r.normed_in_epilogue = body_epilogue_;

  RunLayerRange(r, split, r.num_layers);
  std::vector<float> logits = ChunkEpilogue(r);  // ends with the stream synchronize: every copy above has landed
  pp_setup_.channel->Release(slot);
  ++pp_staged_chunks_;
  if (pp_setup_.on_chunk) pp_setup_.on_chunk();
  return logits;
}

}  // namespace r4dx::model
