// Model's side of the hybrid serving mode's P2 (docs/pp-tp2-hybrid.md 3, 4, 6, 9): the reshard executor (the live state of a pipeline
// stage <-> a TP rank, by the copy ops of reshard_plan.h), the DFlash tail (stage-side capture, rank-side injection) and the test
// hooks that make the byte-identity gate G-H1 checkable (live-state digest, host export / import). Everything here is Model member
// code that nothing calls unless a hybrid caller does; the TP=1, --pp 2, --tp 2 and emulate paths do not reach it.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_tail.h"
#include "dflash_tail_plan.h"
#include "model.h"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

// ---- the free DFlash injection loop ---------------------------------------------------------------------------------------------

void InjectFeatureRowsFromHost(DflashDraft& draft, core::Stream& stream, core::Arena& arena, uint16_t* staging_dev, int64_t slice_rows,
                               const uint16_t* host, int64_t rows, int64_t start_pos, const int32_t* rope_t) {
  if (rows <= 0) return;
  if (host == nullptr || staging_dev == nullptr) throw std::invalid_argument("InjectFeatureRowsFromHost: null feature rows or staging buffer");
  const int64_t cols = draft.FeatureCols();
  for (const hybrid::TailSlice& s : hybrid::TailSlices(rows, slice_rows)) {
    R4DX_HIP_CHECK(hipMemcpyAsync(staging_dev, host + s.row0 * cols, static_cast<size_t>(s.rows * cols) * sizeof(uint16_t),
                                   hipMemcpyHostToDevice, stream.get()));
    draft.InjectFeatures(stream, arena, staging_dev, s.rows, start_pos + s.row0, rope_t != nullptr ? rope_t + s.row0 : nullptr);
    // As RunChunk: the arena is clean before the next slice, and the drafter's one pinned position array is rewritten by the next
    // slice, so the device must be done with this one first.
    arena.Reset();
    stream.Synchronize();
  }
}

// ---- the reshard executor -------------------------------------------------------------------------------------------------------

void Model::ReshardInit() {
  size_t need = 0;
  for (const auto& g : gdn_states_) {
    if (g) need = std::max(need, static_cast<size_t>(g->ConvDim() * g->ConvHistory()) * sizeof(uint16_t));
  }
  if (reshard_scratch_.bytes() >= need) return;
  stream_.Synchronize();
  reshard_scratch_ = core::DeviceBuffer<uint8_t>(need);
}

int64_t Model::ReshardConvPitch() const {
  for (const auto& g : gdn_states_) {
    if (g) return g->StateLenMax();
  }
  return container_.Config().linear_conv_kernel_dim - 1;
}

void Model::ReshardCollapse() { CollapseSpeculativeWindow(); }

void Model::CheckGdnLiveInPlace(const char* who) const {
  // A speculative round leaves the live recurrent state in window slot n - 1 / behind a pending write-once prefix, and the live conv
  // history at offset n - 1 of its line: the plan's offsets address window 0 / offset 0 (the layout a prefill reads and writes).
  if ((mtp_num_accepted_valid_ && mtp_last_committed_ > 1) || (gdn_write_once_ && gdn_book_.Pending())) {
    throw std::logic_error(std::string(who) + ": a speculative round left the live GDN state in a window slot / behind a pending prefix; "
                                              "call ReshardCollapse() first");
  }
}

uint8_t* Model::ReshardLocate(const hybrid::CopyOp& op, const hybrid::OpSlice& slice, hybrid::Side side, const char* who, hybrid::Rect* rect) {
  using hybrid::StateKind;
  const ModelConfig& cfg = container_.Config();
  const std::string w = who;
  // Which holder of the op this Model is.
  if (side == hybrid::Side::kRank) {
    if (cfg.tp_world != 2 || cfg.tp_rank != op.rank) {
      throw std::logic_error(w + ": this Model is not TP rank " + std::to_string(op.rank) + " of 2 (world " + std::to_string(cfg.tp_world) + ", rank " +
                             std::to_string(cfg.tp_rank) + ")");
    }
    if (stage_role_ != stage::Role::kOff) throw std::logic_error(w + ": a pipeline stage is not a TP rank");
  } else {
    if (cfg.tp_world != 1) throw std::logic_error(w + ": the full-head side is held by a TP=1 Model or a stage, not by a TP rank");
    if (stage_role_ != stage::Role::kOff) {
      const int stage_index = stage_role_ == stage::Role::kFront ? hybrid::kStageA : hybrid::kStageB;
      if (op.stage != stage_index) {
        throw std::logic_error(w + ": the op's full-head side is stage " + std::to_string(op.stage) + ", this Model is " + stage::RoleName(stage_role_));
      }
    }
  }
  const hybrid::Rect q = hybrid::RectOf(op, slice, side);
  const hybrid::Run2D& run = op.runs.at(slice.run);
  const size_t li = op.layer >= 0 ? static_cast<size_t>(op.layer) : 0;
  uint8_t* base = nullptr;
  uint64_t bytes = 0, pitch = 0;  // pitch: the buffer's own row pitch for this kind (0: no constraint)
  switch (op.kind) {
    case StateKind::kKv: {
      if (op.layer < 0 || li >= kv_caches_.size() || !kv_caches_[li]) {
        throw std::logic_error(w + ": this Model has no KV cache for layer " + std::to_string(op.layer));
      }
      attention::PagedKvCache& kv = *kv_caches_[li];
      base = kv.Data();
      pitch = static_cast<uint64_t>(kv.KvBlockStride());
      bytes = static_cast<uint64_t>(kv.MaxBlocks()) * pitch;
      break;
    }
    case StateKind::kMtpKv: {
      if (!mtp_) throw std::logic_error(w + ": this Model has no MTP head");
      attention::PagedKvCache& kv = mtp_->KvCache();
      base = kv.Data();
      pitch = static_cast<uint64_t>(kv.KvBlockStride());
      bytes = static_cast<uint64_t>(kv.MaxBlocks()) * pitch;
      break;
    }
    case StateKind::kGdnRecurrent: {
      if (op.layer < 0 || li >= gdn_states_.size() || !gdn_states_[li]) {
        throw std::logic_error(w + ": this Model has no GDN state for layer " + std::to_string(op.layer));
      }
      CheckGdnLiveInPlace(who);
      GdnStateManager& g = *gdn_states_[li];
      base = reinterpret_cast<uint8_t*>(g.RecurrentSlotPtr(g.SlotForSeq(0)));
      bytes = static_cast<uint64_t>(g.RecurrentSlotStride()) * sizeof(float);
      break;
    }
    case StateKind::kGdnConv: {
      if (op.layer < 0 || li >= gdn_states_.size() || !gdn_states_[li]) {
        throw std::logic_error(w + ": this Model has no GDN state for layer " + std::to_string(op.layer));
      }
      CheckGdnLiveInPlace(who);
      GdnStateManager& g = *gdn_states_[li];
      base = reinterpret_cast<uint8_t*>(g.ConvLinePtr(0));
      pitch = static_cast<uint64_t>(g.StateLenMax()) * sizeof(uint16_t);
      bytes = static_cast<uint64_t>(g.ConvDim()) * pitch;
      if (static_cast<uint64_t>(g.ConvHistory()) * sizeof(uint16_t) != run.width) {
        throw std::logic_error(w + ": the op moves " + std::to_string(run.width) + " bytes per conv channel, this Model's live history is " +
                               std::to_string(g.ConvHistory() * 2));
      }
      break;
    }
  }
  if (pitch != 0 && run.height > 1 && q.pitch != pitch) {
    throw std::logic_error(w + ": layer " + std::to_string(op.layer) + " kind " + std::to_string(static_cast<int>(op.kind)) + " has a row pitch of " +
                           std::to_string(pitch) + " bytes here, the plan assumes " + std::to_string(q.pitch) +
                           " (a plan built for another geometry: cache size, conv pitch)");
  }
  if (q.rows > 1 && q.width > q.pitch) throw std::logic_error(w + ": rows wider than their pitch");
  if (q.rows > 0 && q.End() > bytes) {
    throw std::out_of_range(w + ": the slice reaches byte " + std::to_string(q.End()) + " of a " + std::to_string(bytes) + "-byte buffer (layer " +
                            std::to_string(op.layer) + ")");
  }
  *rect = q;
  return base + q.off;
}

void Model::ReshardExport(const hybrid::CopyOp& op, const hybrid::OpSlice& slice, hybrid::Side side, uint8_t* host) {
  hybrid::Rect q;
  uint8_t* src = ReshardLocate(op, slice, side, "Model::ReshardExport", &q);
  if (q.rows == 0) return;
  if (host == nullptr) throw std::invalid_argument("Model::ReshardExport: no host buffer");
  const hipStream_t s = stream_.get();
  if (op.kind == hybrid::StateKind::kGdnConv && q.rows > 1 && q.pitch != q.width) {
    // The live entries of a pitched conv line are 6-byte rows: compact them on the device first, then one contiguous D2H.
    if (reshard_scratch_.bytes() < q.Bytes()) throw std::logic_error("Model::ReshardExport: the conv scratch is not allocated (ReshardInit)");
    R4DX_HIP_CHECK(hipMemcpy2DAsync(reshard_scratch_.data(), q.width, src, q.pitch, q.width, q.rows, hipMemcpyDeviceToDevice, s));
    R4DX_HIP_CHECK(hipMemcpyAsync(host, reshard_scratch_.data(), q.Bytes(), hipMemcpyDeviceToHost, s));
  } else if (q.rows == 1 || q.pitch == q.width) {
    R4DX_HIP_CHECK(hipMemcpyAsync(host, src, q.Bytes(), hipMemcpyDeviceToHost, s));
  } else {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(host, q.width, src, q.pitch, q.width, q.rows, hipMemcpyDeviceToHost, s));
  }
}

void Model::ReshardImport(const hybrid::CopyOp& op, const hybrid::OpSlice& slice, hybrid::Side side, const uint8_t* host) {
  hybrid::Rect q;
  uint8_t* dst = ReshardLocate(op, slice, side, "Model::ReshardImport", &q);
  if (q.rows == 0) return;
  if (host == nullptr) throw std::invalid_argument("Model::ReshardImport: no host buffer");
  const hipStream_t s = stream_.get();
  if (op.kind == hybrid::StateKind::kGdnConv && q.rows > 1 && q.pitch != q.width) {
    if (reshard_scratch_.bytes() < q.Bytes()) throw std::logic_error("Model::ReshardImport: the conv scratch is not allocated (ReshardInit)");
    R4DX_HIP_CHECK(hipMemcpyAsync(reshard_scratch_.data(), host, q.Bytes(), hipMemcpyHostToDevice, s));
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dst, q.pitch, reshard_scratch_.data(), q.width, q.width, q.rows, hipMemcpyDeviceToDevice, s));
  } else if (q.rows == 1 || q.pitch == q.width) {
    R4DX_HIP_CHECK(hipMemcpyAsync(dst, host, q.Bytes(), hipMemcpyHostToDevice, s));
  } else {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dst, q.pitch, host, q.width, q.width, q.rows, hipMemcpyHostToDevice, s));
  }
}

void Model::ReshardCopyLocal(Model& stage, Model& rank, const hybrid::CopyOp& op, const hybrid::OpSlice& slice, hybrid::Dir dir) {
  if (!op.local) throw std::logic_error("Model::ReshardCopyLocal: the op crosses cards (go through the host: ReshardExport / ReshardImport)");
  const bool scatter = dir == hybrid::Dir::kStageToRank;
  Model& src = scatter ? stage : rank;
  Model& dst = scatter ? rank : stage;
  hybrid::Rect qs, qd;
  uint8_t* sp = src.ReshardLocate(op, slice, scatter ? hybrid::Side::kFull : hybrid::Side::kRank, "Model::ReshardCopyLocal (source)", &qs);
  uint8_t* dp = dst.ReshardLocate(op, slice, scatter ? hybrid::Side::kRank : hybrid::Side::kFull, "Model::ReshardCopyLocal (destination)", &qd);
  if (qs.width != qd.width || qs.rows != qd.rows) throw std::logic_error("Model::ReshardCopyLocal: the two sides of the op disagree on the slice's shape");
  if (qs.rows == 0) return;
  const hipStream_t s = dst.stream_.get();
  if (qs.rows == 1) {
    R4DX_HIP_CHECK(hipMemcpyAsync(dp, sp, qs.width, hipMemcpyDeviceToDevice, s));
  } else {
    R4DX_HIP_CHECK(hipMemcpy2DAsync(dp, qd.pitch, sp, qs.pitch, qs.width, qs.rows, hipMemcpyDeviceToDevice, s));
  }
}

// ---- the DFlash tail ------------------------------------------------------------------------------------------------------------

void Model::TpInjectDflashTail(const uint16_t* features, int64_t rows, int64_t start_pos, const int32_t* rope_t) {
  if (stage_role_ != stage::Role::kOff || pp_role_ != PpRole::kNone) {
    throw std::logic_error("Model::TpInjectDflashTail: a rank Model injects the tail, a pipeline stage does not");
  }
  if (!dflash_.has_value()) throw std::logic_error("Model::TpInjectDflashTail: this Model has no DFlash drafter");
  if (features == nullptr) throw std::invalid_argument("Model::TpInjectDflashTail: no feature rows");
  if (max_chunk_ != hybrid::kDflashSliceRows) {  // DflashTailStart's grid and the drafter's max_inject_rows are 64 rows
    throw std::logic_error("Model::TpInjectDflashTail: the tail's slice grid is " + std::to_string(hybrid::kDflashSliceRows) + " rows but max_chunk_ is " +
                           std::to_string(max_chunk_));
  }
  CheckTuningScope();  // the drafter's fc / k_proj / v_proj GEMMs pick their tuning table by the thread's flag
  const int64_t window = dflash_->Config().attention.sliding_window;
  if (const std::string why = hybrid::CheckTailInject(dflash_->InjectedCount(), pos_, start_pos, rows, window); !why.empty()) {
    throw std::logic_error("Model::TpInjectDflashTail: " + why);
  }
  if (dflash_features_dev_.size() < static_cast<size_t>(max_chunk_ * DflashFeatureCols())) {
    throw std::logic_error("Model::TpInjectDflashTail: the feature capture buffer is not allocated");
  }
  stream_.Synchronize();
  InjectFeatureRowsFromHost(*dflash_, stream_, arena_, dflash_features_dev_.data(), max_chunk_, features, rows, start_pos, rope_t);
  SetDflashFeatureRows(0);  // the staging reused the capture buffer: nothing in it is a captured chunk
}

struct Model::DflashTailState {
  hybrid::TailCover cover;
  uint16_t* host = nullptr;
  int64_t cols = 0;
  DflashTailState(int64_t start, int64_t end, uint16_t* h, int64_t c) : cover(start, end), host(h), cols(c) {}
};

void Model::StageArmDflashTail(uint16_t* host, int64_t capacity_rows, int64_t start_pos, int64_t end_pos) {
  if (dflash_tail_) throw std::logic_error("Model::StageArmDflashTail: already armed");
  if (!DflashFeatureCaptureAttached()) throw std::logic_error("Model::StageArmDflashTail: needs AttachDflashFeatureCapture");
  if (dflash_observer_) throw std::logic_error("Model::StageArmDflashTail: the capture-observer slot is taken");
  if (host == nullptr || start_pos < 0 || end_pos <= start_pos || end_pos - start_pos > capacity_rows) {
    throw std::invalid_argument("Model::StageArmDflashTail: needs a buffer and 0 <= start < end with end - start <= capacity_rows");
  }
  stream_.Synchronize();
  auto st = std::make_shared<DflashTailState>(start_pos, end_pos, host, DflashFeatureCols());
  const hipStream_t s = stream_.get();  // not `this`: the observer must survive a Model move
  dflash_observer_ = [st, s](const uint16_t* features, int64_t rows, int64_t slice_start) {
    const hybrid::TailCover::Take t = st->cover.Offer(slice_start, rows);
    if (t.rows == 0) return;
    // Asynchronous into pinned memory on the Model's stream; the next chunk's capture overwrites `features` behind it in stream order.
    R4DX_HIP_CHECK(hipMemcpyAsync(st->host + t.dst_row0 * st->cols, features + t.src_row0 * st->cols,
                                   static_cast<size_t>(t.rows * st->cols) * sizeof(uint16_t), hipMemcpyDeviceToHost, s));
  };
  dflash_tail_ = std::move(st);
}

int64_t Model::StageDisarmDflashTail() {
  if (!dflash_tail_) throw std::logic_error("Model::StageDisarmDflashTail: not armed");
  const std::shared_ptr<DflashTailState> st = std::move(dflash_tail_);
  dflash_tail_.reset();
  ClearDflashCaptureObserver();
  stream_.Synchronize();
  if (!st->cover.Complete()) {
    throw std::runtime_error("Model::StageDisarmDflashTail: captured " + std::to_string(st->cover.Taken()) + " of the " +
                             std::to_string(st->cover.End() - st->cover.Start()) + " tail rows (the call did not reach position " +
                             std::to_string(st->cover.End()) + ")");
  }
  return st->cover.Taken();
}

// ---- test hooks -----------------------------------------------------------------------------------------------------------------

void Model::LiveGdnPlace(GdnStateManager& gs, int32_t* slot, int64_t* conv_off) const {
  *slot = gs.SlotForSeq(0);
  *conv_off = 0;
  if (mtp_num_accepted_valid_ && mtp_last_committed_ > 1) {
    if (!gs.WriteOnce()) *slot = gs.WindowSlot(0, static_cast<int32_t>(mtp_last_committed_ - 1));
    *conv_off = mtp_last_committed_ - 1;
  }
}

#ifdef R4DX_TP_TESTING
namespace {

// A KV cache's rows [0, rows) as canonical records, read in groups of blocks.
uint64_t DigestKvCache(attention::PagedKvCache& kv, int64_t rows) {
  hybrid::KvDigester d(kv.KvHeads(), kv.BlockSize(), 2 * kv.HeadDim(), rows);
  const int64_t need = d.BlocksNeeded();
  const size_t stride = static_cast<size_t>(kv.KvBlockStride());
  constexpr int64_t kGroup = 256;  // blocks per host copy (8 MiB at 32 KiB per block)
  std::vector<uint8_t> host(static_cast<size_t>(std::min<int64_t>(kGroup, std::max<int64_t>(need, 1))) * stride);
  for (int64_t b = 0; b < need; b += kGroup) {
    const int64_t n = std::min(kGroup, need - b);
    R4DX_HIP_CHECK(hipMemcpy(host.data(), kv.Data() + static_cast<size_t>(b) * stride, static_cast<size_t>(n) * stride, hipMemcpyDeviceToHost));
    d.Feed(host.data(), b, n);
  }
  return d.Finish();
}

// A KV cache's first ceil(rows / block) blocks as a canonical host image.
std::vector<uint8_t> ExportKvCache(attention::PagedKvCache& kv, int64_t rows) {
  const int64_t need = rows > 0 ? (rows + kv.BlockSize() - 1) / kv.BlockSize() : 0;
  std::vector<uint8_t> out(static_cast<size_t>(need) * static_cast<size_t>(kv.KvBlockStride()));
  if (need > 0) R4DX_HIP_CHECK(hipMemcpy(out.data(), kv.Data(), out.size(), hipMemcpyDeviceToHost));
  hybrid::CanonicalizeKv(&out, kv.KvHeads(), kv.BlockSize(), 2 * kv.HeadDim(), rows);
  return out;
}

void ImportKvCache(attention::PagedKvCache& kv, const std::vector<uint8_t>& image, int64_t rows, const char* what) {
  const int64_t need = rows > 0 ? (rows + kv.BlockSize() - 1) / kv.BlockSize() : 0;
  const size_t stride = static_cast<size_t>(kv.KvBlockStride());
  if (image.size() != static_cast<size_t>(need) * stride || need > kv.MaxBlocks()) {
    throw std::invalid_argument(std::string("Model::DebugImportLiveState: the ") + what + " image holds " + std::to_string(image.size()) +
                                " bytes, " + std::to_string(need) + " blocks of " + std::to_string(stride) + " were expected (cache of " +
                                std::to_string(kv.MaxBlocks()) + " blocks)");
  }
  if (need > 0) R4DX_HIP_CHECK(hipMemcpy(kv.Data(), image.data(), image.size(), hipMemcpyHostToDevice));
}

}  // namespace

hybrid::DigestList Model::DebugLiveStateDigest() {
  FlushGdnPending();  // a pending write-once prefix is applied to B first (invisible to the continuing run)
  stream_.Synchronize();
  hybrid::DigestList out;
  out.emplace_back("pos", static_cast<uint64_t>(pos_));
  for (size_t i = 0; i < gdn_states_.size(); ++i) {
    if (kv_caches_[i]) out.emplace_back(hybrid::DigestName("kv", static_cast<int64_t>(i)), DigestKvCache(*kv_caches_[i], pos_));
    if (!gdn_states_[i]) continue;
    GdnStateManager& gs = *gdn_states_[i];
    int32_t slot = 0;
    int64_t conv_off = 0;
    LiveGdnPlace(gs, &slot, &conv_off);
    std::vector<uint8_t> rec(static_cast<size_t>(gs.RecurrentSlotStride()) * sizeof(float));
    R4DX_HIP_CHECK(hipMemcpy(rec.data(), gs.RecurrentSlotPtr(slot), rec.size(), hipMemcpyDeviceToHost));
    out.emplace_back(hybrid::DigestName("gdn.rec", static_cast<int64_t>(i)), hybrid::DigestBytes(rec.data(), rec.size()));
    const size_t hist = static_cast<size_t>(gs.ConvHistory()) * sizeof(uint16_t);
    std::vector<uint16_t> conv(static_cast<size_t>(gs.ConvDim() * gs.ConvHistory()));
    R4DX_HIP_CHECK(hipMemcpy2D(conv.data(), hist, gs.ConvLinePtr(0) + conv_off, static_cast<size_t>(gs.StateLenMax()) * sizeof(uint16_t), hist,
                                static_cast<size_t>(gs.ConvDim()), hipMemcpyDeviceToHost));
    out.emplace_back(hybrid::DigestName("gdn.conv", static_cast<int64_t>(i)), hybrid::DigestBytes(conv.data(), conv.size() * sizeof(uint16_t)));
  }
  if (mtp_ && hybrid::MtpLiveRows(pos_) > 0) out.emplace_back("mtp.kv", DigestKvCache(mtp_->KvCache(), hybrid::MtpLiveRows(pos_)));
  if (dflash_.has_value()) {
    const Dflash2Config& dc = dflash_->Config();
    int64_t lo = 0, hi = 0;
    hybrid::DflashWindow(dflash_->InjectedCount(), dflash_->ValidFrom(), dc.attention.sliding_window, &lo, &hi);
    const int64_t row_elems = dc.attention.head_count_kv * dc.attention.key_length;
    for (int64_t l = 0; l < dc.block_count; ++l) {
      const std::vector<uint16_t> k = dflash_->DebugStoreK(stream_, l, lo, hi - lo);
      const std::vector<uint16_t> v = dflash_->DebugStoreV(stream_, l, lo, hi - lo);
      out.emplace_back(hybrid::DigestName("dflash.k", l), hybrid::DigestDflashRows(k.data(), hi - lo, row_elems, lo));
      out.emplace_back(hybrid::DigestName("dflash.v", l), hybrid::DigestDflashRows(v.data(), hi - lo, row_elems, lo));
    }
    out.emplace_back("dflash.injected", static_cast<uint64_t>(dflash_->InjectedCount()));
    out.emplace_back("dflash.lo", static_cast<uint64_t>(lo));
  }
  return out;
}

hybrid::LiveState Model::DebugExportLiveState() {
  if (pos_ <= 0) throw std::logic_error("Model::DebugExportLiveState: nothing has been fed (pos 0)");
  FlushGdnPending();
  stream_.Synchronize();
  hybrid::LiveState st;
  st.scalars = StageGetSyncState();
  for (size_t i = 0; i < gdn_states_.size(); ++i) {
    if (kv_caches_[i]) st.image.kv[static_cast<int64_t>(i)] = ExportKvCache(*kv_caches_[i], pos_);
    if (!gdn_states_[i]) continue;
    GdnStateManager& gs = *gdn_states_[i];
    int32_t slot = 0;
    int64_t conv_off = 0;
    LiveGdnPlace(gs, &slot, &conv_off);
    std::vector<uint8_t>& rec = st.image.gdn_rec[static_cast<int64_t>(i)];
    rec.resize(static_cast<size_t>(gs.RecurrentSlotStride()) * sizeof(float));
    R4DX_HIP_CHECK(hipMemcpy(rec.data(), gs.RecurrentSlotPtr(slot), rec.size(), hipMemcpyDeviceToHost));
    const size_t hist = static_cast<size_t>(gs.ConvHistory()) * sizeof(uint16_t);
    std::vector<uint16_t>& conv = st.image.gdn_conv[static_cast<int64_t>(i)];
    conv.resize(static_cast<size_t>(gs.ConvDim() * gs.ConvHistory()));
    R4DX_HIP_CHECK(hipMemcpy2D(conv.data(), hist, gs.ConvLinePtr(0) + conv_off, static_cast<size_t>(gs.StateLenMax()) * sizeof(uint16_t), hist,
                                static_cast<size_t>(gs.ConvDim()), hipMemcpyDeviceToHost));
  }
  if (mtp_ && hybrid::MtpLiveRows(pos_) > 0) st.image.mtp_kv = ExportKvCache(mtp_->KvCache(), hybrid::MtpLiveRows(pos_));
  return st;
}

void Model::DebugImportLiveState(const hybrid::LiveState& state) {
  if (stage_role_ != stage::Role::kOff) throw std::logic_error("Model::DebugImportLiveState: a pipeline stage is loaded by the reshard, not by an image");
  const int64_t pos = state.scalars.pos;
  if (pos <= 0) throw std::invalid_argument("Model::DebugImportLiveState: the state's position must be positive");
  // Validate the whole image against this Model's geometry before anything is written.
  const hybrid::LiveImage& img = state.image;
  size_t n_kv = 0, n_gdn = 0;
  for (size_t i = 0; i < gdn_states_.size(); ++i) {
    const int64_t layer = static_cast<int64_t>(i);
    if (kv_caches_[i]) {
      ++n_kv;
      if (img.kv.find(layer) == img.kv.end()) throw std::invalid_argument("Model::DebugImportLiveState: no KV image for layer " + std::to_string(layer));
    }
    if (gdn_states_[i]) {
      ++n_gdn;
      const GdnStateManager& gs = *gdn_states_[i];
      const auto rec = img.gdn_rec.find(layer);
      const auto conv = img.gdn_conv.find(layer);
      if (rec == img.gdn_rec.end() || rec->second.size() != static_cast<size_t>(gs.RecurrentSlotStride()) * sizeof(float)) {
        throw std::invalid_argument("Model::DebugImportLiveState: the recurrent image of layer " + std::to_string(layer) + " is missing or the wrong size");
      }
      if (conv == img.gdn_conv.end() || conv->second.size() != static_cast<size_t>(gs.ConvDim() * gs.ConvHistory())) {
        throw std::invalid_argument("Model::DebugImportLiveState: the conv image of layer " + std::to_string(layer) + " is missing or the wrong size");
      }
    }
  }
  if (img.kv.size() != n_kv || img.gdn_rec.size() != n_gdn || img.gdn_conv.size() != n_gdn) {
    throw std::invalid_argument("Model::DebugImportLiveState: the image carries layers this Model does not hold");
  }
  const bool want_mtp = mtp_ && hybrid::MtpLiveRows(pos) > 0;
  if (want_mtp != !img.mtp_kv.empty()) {
    throw std::invalid_argument("Model::DebugImportLiveState: the MTP-KV image is " + std::string(img.mtp_kv.empty() ? "missing" : "present") +
                                ", this Model " + (want_mtp ? "needs it" : "does not"));
  }
  // The scalars first: TpAdoptPrefill collapses the speculative bookkeeping, so the live GDN state is at window 0 / offset 0.
  TpAdoptPrefill(state.scalars);
  for (size_t i = 0; i < gdn_states_.size(); ++i) {
    const int64_t layer = static_cast<int64_t>(i);
    if (kv_caches_[i]) ImportKvCache(*kv_caches_[i], img.kv.at(layer), pos, "KV");
    if (!gdn_states_[i]) continue;
    GdnStateManager& gs = *gdn_states_[i];
    const std::vector<uint8_t>& rec = img.gdn_rec.at(layer);
    R4DX_HIP_CHECK(hipMemcpy(gs.RecurrentSlotPtr(gs.SlotForSeq(0)), rec.data(), rec.size(), hipMemcpyHostToDevice));
    const std::vector<uint16_t>& conv = img.gdn_conv.at(layer);
    const size_t hist = static_cast<size_t>(gs.ConvHistory()) * sizeof(uint16_t);
    R4DX_HIP_CHECK(hipMemcpy2D(gs.ConvLinePtr(0), static_cast<size_t>(gs.StateLenMax()) * sizeof(uint16_t), conv.data(), hist, hist,
                                static_cast<size_t>(gs.ConvDim()), hipMemcpyHostToDevice));
  }
  if (want_mtp) ImportKvCache(mtp_->KvCache(), img.mtp_kv, hybrid::MtpLiveRows(pos), "MTP KV");
  stream_.Synchronize();
}

void Model::DebugExportFullState(const std::string& path) {
  if (comm_ != nullptr) throw std::logic_error("Model::DebugExportFullState: a TP=1 Model's full-head state (a rank's image is DebugExportLiveState)");
  hybrid::WriteLiveState(path, DebugExportLiveState());
}

void Model::DebugImportFullState(const std::string& path) {
  hybrid::LiveState full = hybrid::ReadLiveState(path);
  if (comm_ == nullptr) {
    DebugImportLiveState(full);
    return;
  }
  const hybrid::StateGeometry g = hybrid::StateGeometry::FromRules(container_.GlobalConfig(), PpKvBlockSize());
  hybrid::LiveState mine;
  mine.scalars = std::move(full.scalars);
  mine.image = hybrid::ReshardRef(g, full.image, container_.Config().tp_rank);
  DebugImportLiveState(mine);
}
#endif  // R4DX_TP_TESTING

}  // namespace r4dx::model
