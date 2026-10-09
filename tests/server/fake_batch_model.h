// tests/server/fake_batch_model.h -- a CPU TextModel with batch slots, shared by test_engine_batch.cpp (the whole engine) and test_batch_port.cpp
// (BatchPort + BatchExecutor alone). Its "logits" are a pure function of every token a sequence has been fed -- in the single-sequence state
// and in a batch slot alike -- so a request's text depends on its own tokens only, and a test can compare a batched run against a one-at-a-time
// run token for token.
//
// All state is guarded by one mutex only so that a WRONG caller (a second thread) is reported rather than racing: every entry point records
// the calling thread (`threads`). `interleavings` counts Reset()/Prefill() calls that arrive between another request's Prefill and its
// BatchImport -- the PRIMARY lock discipline of BatchPort. No HIP call.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "model_config.h"
#include "model_types.h"
#include "text_model.h"

namespace fake_batch {

using r4dx::model::BatchDecodeRow;
using r4dx::model::ImageRows;
using r4dx::model::ImageSpan;
using r4dx::model::StepProfile;
using r4dx::model::VramReport;

constexpr int64_t kVocab = 262144;

// A TextModel with `slots` batch slots. All state is guarded by one mutex only so that a WRONG caller (a second thread) is reported rather than
// racing: every entry point records the calling thread.
class FakeBatchModel final : public r4dx::model::TextModel {
 public:
  FakeBatchModel(int slots, int64_t slot_ctx) : slots_(slots), slot_ctx_(slot_ctx), slot_fed_(static_cast<size_t>(slots)) {}

  // ---- what the test reads --------------------------------------------------------------------------------------------------
  std::mutex mu;
  std::set<std::thread::id> threads;      // every thread that called into the model
  int max_rows = 0;                       // the widest DecodeBatch call
  int64_t batch_calls = 0, batch_rows = 0;
  int imports = 0, releases = 0, interleavings = 0;
  std::atomic<int> fail_batch_after{-1};  // the n-th DecodeBatch call from now throws (-1: never)
  std::atomic<bool> fail_import{false};   // the next BatchImport throws (a prompt that does not fit, say)
  std::atomic<int> decode_delay_ms{0};    // each DecodeBatch call takes at least this long (a step that is visibly "on the device")
  std::vector<std::vector<BatchDecodeRow>> calls;  // every DecodeBatch call's rows, as received (guarded by mu)

  int ActiveSlots() {
    std::lock_guard<std::mutex> lock(mu);
    int n = 0;
    for (const auto& a : slot_active_) n += a ? 1 : 0;
    return n;
  }

  // ---- host-only ---------------------------------------------------------------------------------------------------------------
  const r4dx::model::ModelConfig& Config() const override { return cfg_; }
  const std::string& ModelId() const override { return id_; }
  int64_t ImageTokenId() const override { return 248056; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int64_t SampledFallbackRows() const override { return 0; }
  int64_t PositionCount() const override { return static_cast<int64_t>(fed_.size()); }
  int64_t NumLoadedLayers() const override { return 4; }
  int TpWorld() const override { return 1; }
  std::vector<VramReport> Vram() const override { return {}; }
  void SetDflashInjectionEnabled(bool) override {}
  void SaveCheckpoint() override { throw std::logic_error("no checkpoint in batch mode"); }
  void RestoreCheckpoint() override { throw std::logic_error("no checkpoint in batch mode"); }
  void EncodeImages(const float*, int64_t, const std::vector<r4dx::vision::GridThw>&, ImageRows*,
                    r4dx::vision::VisionEncodeStats*) override {
    throw std::logic_error("FakeBatchModel: no vision");
  }

  // ---- the single-sequence state -------------------------------------------------------------------------------------------
  void Reset() override {
    Note();
    std::lock_guard<std::mutex> lock(mu);
    if (prefilled_ && !imported_) ++interleavings;  // another request's prompt was prefilled and not yet imported
    fed_.clear();
    prefilled_ = imported_ = false;
  }
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override {
    Note();
    std::lock_guard<std::mutex> lock(mu);
    if (prefilled_ && !imported_) ++interleavings;
    fed_.insert(fed_.end(), token_ids.begin(), token_ids.end());
    prefilled_ = true;
    imported_ = false;
    return Row(fed_);
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& t, const std::vector<ImageSpan>&) override { return Prefill(t); }
  std::vector<float> DecodeStep(int32_t) override { throw std::logic_error("FakeBatchModel: DecodeStep unused"); }
  int32_t DecodeStepGreedy(int32_t token_id) override {  // the one-request-at-a-time reference engine
    Note();
    std::lock_guard<std::mutex> lock(mu);
    fed_.push_back(token_id);
    return Next(fed_);
  }
  int32_t DecodeStepSampled(int32_t, const r4dx::kernels::SampleParams&, std::mt19937_64&) override {
    throw std::logic_error("FakeBatchModel: greedy requests only");
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t, int64_t) override { throw std::logic_error("no MTP"); }
  std::vector<int32_t> DecodeStepMtpSampled(int32_t, int64_t, const r4dx::kernels::SampleParams&, std::mt19937_64&) override {
    throw std::logic_error("no MTP");
  }
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) override { throw std::logic_error("no DFlash"); }
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const r4dx::kernels::SampleParams&,
                                               std::mt19937_64&, int64_t*) override {
    throw std::logic_error("no DFlash");
  }
  StepProfile DecodeStepProfiled(int32_t) override { throw std::logic_error("no profiling"); }
  StepProfile PrefillProfiled(const std::vector<int32_t>&) override { throw std::logic_error("no profiling"); }

  // ---- batched decode ------------------------------------------------------------------------------------------------------
  int BatchSlots() const override { return slots_; }
  int64_t BatchSlotCtx() const override { return slot_ctx_; }
  void BatchImport(int slot) override {
    Note();
    std::lock_guard<std::mutex> lock(mu);
    if (slot < 0 || slot >= slots_) throw std::invalid_argument("slot");
    if (fail_import.exchange(false)) throw std::runtime_error("FakeBatchModel: injected import failure");
    if (!prefilled_) throw std::logic_error("FakeBatchModel: BatchImport before a prefill");
    if (static_cast<int64_t>(fed_.size()) >= slot_ctx_) throw std::runtime_error("FakeBatchModel: the prompt fills the slot");
    slot_fed_[static_cast<size_t>(slot)] = fed_;
    if (slot_active_.size() < static_cast<size_t>(slots_)) slot_active_.resize(static_cast<size_t>(slots_), false);
    slot_active_[static_cast<size_t>(slot)] = true;
    imported_ = true;
    ++imports;
  }
  void BatchRelease(int slot) override {
    Note();
    std::lock_guard<std::mutex> lock(mu);
    slot_active_[static_cast<size_t>(slot)] = false;
    slot_fed_[static_cast<size_t>(slot)].clear();
    ++releases;
  }
  std::vector<int32_t> DecodeBatch(const std::vector<BatchDecodeRow>& rows) override {
    Note();
    if (decode_delay_ms.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(decode_delay_ms.load()));
    std::lock_guard<std::mutex> lock(mu);
    calls.push_back(rows);
    ++batch_calls;
    batch_rows += static_cast<int64_t>(rows.size());
    max_rows = std::max(max_rows, static_cast<int>(rows.size()));
    int n = fail_batch_after.load();
    if (n == 0) {
      fail_batch_after = -1;
      for (const BatchDecodeRow& r : rows) slot_active_[static_cast<size_t>(r.slot)] = false;  // Model::DecodeBatch releases the rows' slots
      throw std::runtime_error("injected batch failure");
    }
    if (n > 0) fail_batch_after = n - 1;
    std::vector<int32_t> out;
    for (const BatchDecodeRow& r : rows) {
      if (r.slot < 0 || r.slot >= slots_ || !slot_active_[static_cast<size_t>(r.slot)]) throw std::logic_error("FakeBatchModel: inactive slot");
      auto& fed = slot_fed_[static_cast<size_t>(r.slot)];
      fed.push_back(r.token);
      out.push_back(Next(fed));
    }
    return out;
  }

 private:
  void Note() {
    std::lock_guard<std::mutex> lock(mu);
    threads.insert(std::this_thread::get_id());
  }
  static int32_t Next(const std::vector<int32_t>& fed) {  // FNV-1a over everything fed, into ordinary (non-special, non-EOS) token ids
    uint64_t h = 1469598103934665603ull;
    for (int32_t t : fed) h = (h ^ static_cast<uint32_t>(t)) * 1099511628211ull;
    return static_cast<int32_t>(1000 + h % 20000);
  }
  static std::vector<float> Row(const std::vector<int32_t>& fed) {
    std::vector<float> row(static_cast<size_t>(kVocab), 0.0f);
    row[static_cast<size_t>(Next(fed))] = 1.0f;
    return row;
  }

  const int slots_;
  const int64_t slot_ctx_;
  r4dx::model::ModelConfig cfg_;
  std::string id_ = "fake/batch";
  std::vector<int32_t> fed_;
  bool prefilled_ = false, imported_ = false;
  std::vector<std::vector<int32_t>> slot_fed_;
  std::vector<bool> slot_active_ = std::vector<bool>(16, false);
};

}  // namespace fake_batch
