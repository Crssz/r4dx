// r4dx::server::BatchPort -- the TextModel one request sees in `--batch N` serving (docs/batch-decode.md 7). Engine::RunRequest is
// unchanged apart from calling this instead of the real model: the port forwards every call to the executor thread
// (batch_executor.h), which owns the model, and turns the decode calls into rows of a batched step.
//
// What a request does, and what the port makes of it:
//
//   Reset()                         -> takes the PRIMARY lock (one request at a time owns the model's single-sequence state), then Reset
//                                      on the executor. A request waiting for the lock is a request waiting for another's prefill.
//   EncodeImages()                  -> a job (the vision tower does not touch the sequence state).
//   Prefill() / PrefillMultimodal() -> ONE job: the real prefill (hybrid PP-2 pipeline included -- the TextModel decides), then
//                                      BatchImport(slot) copies the state it built into this request's slot; the primary lock is released
//                                      and the slot joins the decoding set.
//   DecodeStepGreedy / Sampled /    -> one row of the next batched step (BatchExecutor::StepAsync), blocking until the step returns this
//   DecodeStepGreedyOverlap            sequence's token. The Overlap form posts the row, runs the caller's host work while the step is on
//                                      the device, then waits (an exception from the callback is rethrown after the step has finished).
//   destruction                     -> releases the slot, tells the executor the sequence has left, drops the primary lock if a failed
//                                      request still held it.
//
// Not available in batch mode, by name: the full-logits DecodeStep, the speculative rounds (MTP / DFlash2 -- `--batch` refuses them at
// startup) and the prompt checkpoint (the Engine turns it off), because the single-sequence state is a scratch pad here, not a
// conversation.
//
// Each request owns one port, created on its own thread; `slot` is the request thread's index (Engine::BatchWorkerLoop), so no two ports
// ever share a slot.
#pragma once

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_executor.h"
#include "text_model.h"

namespace r4dx::server {

class BatchPort final : public r4dx::model::TextModel {
 public:
  using Row = BatchExecutor::Row;

  BatchPort(r4dx::model::TextModel& inner, BatchExecutor& exec, std::mutex& primary_mu, int slot)
      : m_(inner), exec_(exec), primary_(primary_mu, std::defer_lock), slot_(slot) {}
  ~BatchPort() override { Finish(); }
  BatchPort(const BatchPort&) = delete;
  BatchPort& operator=(const BatchPort&) = delete;

  // ---- identity / capabilities: cached values of the inner model, read-only after load -----------------------------------
  const r4dx::model::ModelConfig& Config() const override { return m_.Config(); }
  const std::string& ModelId() const override { return m_.ModelId(); }
  int64_t ImageTokenId() const override { return m_.ImageTokenId(); }
  int64_t VisionMergeSize() const override { return m_.VisionMergeSize(); }
  bool HasVision() const override { return m_.HasVision(); }
  int64_t ImageBoiTokenId() const override { return m_.ImageBoiTokenId(); }
  int64_t ImageEoiTokenId() const override { return m_.ImageEoiTokenId(); }
  bool MtpEnabled() const override { return false; }  // batch mode refuses speculation at startup
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int TpWorld() const override { return m_.TpWorld(); }
  int64_t NumLoadedLayers() const override { return m_.NumLoadedLayers(); }
  int BatchSlots() const override { return m_.BatchSlots(); }
  int64_t BatchSlotCtx() const override { return m_.BatchSlotCtx(); }

  // ---- state that moves with the model: read on the executor thread --------------------------------------------------------
  int64_t SampledFallbackRows() const override { return exec_.Run([this] { return m_.SampledFallbackRows(); }); }
  int64_t PositionCount() const override { return exec_.Run([this] { return m_.PositionCount(); }); }
  std::vector<r4dx::model::VramReport> Vram() const override { return exec_.Run([this] { return m_.Vram(); }); }

  // ---- sequence state ----------------------------------------------------------------------------------------------------
  void Reset() override {
    if (!primary_.owns_lock()) primary_.lock();  // until this request's prefill has been imported into its slot
    exec_.Run([this] { m_.Reset(); });
  }
  void SetDflashInjectionEnabled(bool enabled) override {
    exec_.Run([this, enabled] { m_.SetDflashInjectionEnabled(enabled); });
  }
  void SaveCheckpoint() override { Unsupported("SaveCheckpoint (the prompt checkpoint is off in batch mode)"); }
  void RestoreCheckpoint() override { Unsupported("RestoreCheckpoint (the prompt checkpoint is off in batch mode)"); }

  void EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<r4dx::vision::GridThw>& grids,
                    r4dx::model::ImageRows* out, r4dx::vision::VisionEncodeStats* stats = nullptr) override {
    exec_.Run([&] { m_.EncodeImages(pixel_values, total_patches, grids, out, stats); });
  }

  // ---- forward ---------------------------------------------------------------------------------------------------------------
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override {
    return PrefillAndImport([&] { return m_.Prefill(token_ids); });
  }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<r4dx::model::ImageSpan>& images) override {
    return PrefillAndImport([&] { return m_.PrefillMultimodal(token_ids, images); });
  }
  std::vector<float> DecodeStep(int32_t) override { Unsupported("DecodeStep (full logits)"); return {}; }
  int32_t DecodeStepGreedy(int32_t token_id) override { return exec_.Step(MakeRow(token_id, nullptr, nullptr)); }
  int32_t DecodeStepGreedyOverlap(int32_t token_id, const std::function<void()>& while_busy) override {
    std::future<int32_t> next = exec_.StepAsync(MakeRow(token_id, nullptr, nullptr));
    std::exception_ptr callback_error;
    try {
      while_busy();  // the caller's host work (decode, stop-string scan, streaming) while the batched step runs
    } catch (...) {
      callback_error = std::current_exception();
    }
    const int32_t token = next.get();  // wait for the step either way: the slot has advanced
    if (callback_error) std::rethrow_exception(callback_error);
    return token;
  }
  int32_t DecodeStepSampled(int32_t token_id, const r4dx::kernels::SampleParams& params, std::mt19937_64& rng) override {
    return exec_.Step(MakeRow(token_id, &params, &rng));
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t, int64_t) override { Unsupported("DecodeStepMtpGreedy"); return {}; }
  std::vector<int32_t> DecodeStepMtpSampled(int32_t, int64_t, const r4dx::kernels::SampleParams&, std::mt19937_64&) override {
    Unsupported("DecodeStepMtpSampled");
    return {};
  }
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t* = nullptr) override {
    Unsupported("DecodeStepDflashGreedy");
    return {};
  }
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const r4dx::kernels::SampleParams&,
                                               std::mt19937_64&, int64_t* = nullptr) override {
    Unsupported("DecodeStepDflashSampled");
    return {};
  }
  r4dx::model::StepProfile DecodeStepProfiled(int32_t) override { Unsupported("DecodeStepProfiled"); return {}; }
  r4dx::model::StepProfile PrefillProfiled(const std::vector<int32_t>&) override { Unsupported("PrefillProfiled"); return {}; }

  // Releases the slot and the primary lock; idempotent, never throws (it runs from the destructor of a request that may have failed
  // because the executor itself went away).
  void Finish() noexcept {
    try {
      if (joined_) {
        exec_.Leave(slot_);
        exec_.Run([this] { m_.BatchRelease(slot_); });
      }
    } catch (...) {
    }
    joined_ = false;
    if (primary_.owns_lock()) primary_.unlock();
  }

 private:
  [[noreturn]] static void Unsupported(const char* what) {
    throw std::logic_error(std::string("batched serving does not support ") + what);
  }
  Row MakeRow(int32_t token, const r4dx::kernels::SampleParams* params, std::mt19937_64* rng) const {
    if (!joined_) throw std::logic_error("BatchPort: a decode step before any prefill (the slot holds no sequence)");
    Row r;
    r.slot = slot_;
    r.token = token;
    if (params != nullptr) r.params = *params;
    else r.params.temperature = 0.0f;  // greedy
    r.rng = rng;
    return r;
  }
  // The real prefill and the import of its state into this slot, as ONE job: nothing else runs on the model in between. Afterwards the
  // primary lock goes back and the slot counts as decoding.
  template <class F>
  std::vector<float> PrefillAndImport(F&& prefill) {
    if (!primary_.owns_lock()) primary_.lock();  // a caller that prefills without a Reset() first (not RunRequest) still gets the lock
    std::vector<float> logits = exec_.Run([&] {
      std::vector<float> l = prefill();
      m_.BatchImport(slot_);
      return l;
    });
    joined_ = true;
    exec_.Join(slot_);
    primary_.unlock();
    return logits;
  }

  r4dx::model::TextModel& m_;
  BatchExecutor& exec_;
  std::unique_lock<std::mutex> primary_;
  const int slot_;
  bool joined_ = false;  // the slot holds this request's sequence (a prefill was imported)
};

}  // namespace r4dx::server
