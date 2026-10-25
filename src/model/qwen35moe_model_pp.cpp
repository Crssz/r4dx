// Qwen35MoeModel::PrefillPipelined -- the chunks of a prompt flow through the two stages of the layer split as a pipeline
// (docs/pp-prefill.md, docs/qwen35moe-plan.md section 8 step c). While the second stage runs chunk i, the first stage already runs chunk
// i + 1, so both cards work at once instead of taking turns. Nothing about the numerics changes: every stage still sees the chunks in order
// (its GDN state and KV cache are its own), and the residual that crosses between the cards is copied through a small ring of pinned host
// buffers exactly as the sequential path copies it through handoff_host_.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "qwen35moe_model.h"

namespace r4dx::model {

namespace {
using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
}  // namespace

bool Qwen35MoeModel::CanPipelinePrefill(size_t chunks) const {
  // (an image in the prompt, or a conversation that has seen one, needs RunChunk's embedding splice and explicit 3-axis rope rows per chunk)
  if (chunks < 2 || stages_.size() != 2 || dflash_.has_value() || capture_active_ || mrope_active_ || !mrope_block_images_.empty()) return false;
  const char* e = std::getenv("R4DX_Q35_PP");
  return !(e != nullptr && e[0] == '0');
}

std::vector<float> Qwen35MoeModel::PrefillPipelined(const std::vector<int32_t>& token_ids, const std::vector<int64_t>& plan) {
  constexpr int kSlots = 2;
  const int64_t hidden = cfg_.hidden_size;
  if (static_cast<int>(pp_ring_.size()) < kSlots) {
    pp_ring_.clear();
    for (int i = 0; i < kSlots; ++i) {
      pp_ring_.push_back(std::make_unique<core::PinnedBuffer<uint16_t>>(static_cast<size_t>(chunk_rows_ * hidden)));
    }
  }

  struct Job {
    int slot;
    int64_t rows, first_pos;
    bool last;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Job> queue;
  bool busy[kSlots] = {false, false};
  bool stop = false, failed = false;
  std::exception_ptr error;
  std::vector<float> logits;
  double stage1_ms = 0, head_ms = 0, stage0_ms = 0, handoff_ms = 0, embed_ms = 0;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kLogits, 0.0f};

  std::thread worker([&] {
    try {
      Qwen35MoeStage& st = *stages_[1];
      for (;;) {
        Job j;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return stop || !queue.empty(); });
          if (queue.empty()) return;
          j = queue.front();
          queue.pop_front();
        }
        Qwen35MoeChunk chunk;
        chunk.rows = j.rows;
        chunk.first_pos = j.first_pos;
        chunk.is_prefill = true;
        chunk.gdn_prefill_recurrent = gdn_recurrent_;
        st.LoadResidual(pp_ring_[static_cast<size_t>(j.slot)]->data(), j.rows);
        st.RunLayers(chunk);
        stage1_ms += st.LastRunMs();
        if (j.last) {
          Qwen35MoeHeadResult r = st.Head(j.rows, req);
          head_ms += st.LastHeadMs();
          logits = std::move(r.logits);
        }
        {
          std::lock_guard<std::mutex> lk(mu);
          busy[j.slot] = false;
        }
        cv.notify_all();
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lk(mu);
        error = std::current_exception();
        failed = true;
      }
      cv.notify_all();
    }
  });
  // Whatever happens below, the worker is stopped and joined before the locals it points into go away.
  struct Joiner {
    std::thread& t;
    std::mutex& mu;
    std::condition_variable& cv;
    bool& stop;
    ~Joiner() {
      {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
      }
      cv.notify_all();
      if (t.joinable()) t.join();
    }
  } joiner{worker, mu, cv, stop};

  const uint16_t* table = containers_.front()->EmbedTokensHost();
  int64_t off = 0, p = pos_;
  for (size_t i = 0; i < plan.size(); ++i) {
    const int64_t rows = plan[i];
    {
      std::lock_guard<std::mutex> lk(mu);
      if (failed) break;
    }
    const auto t_embed = Clock::now();
    for (int64_t t = 0; t < rows; ++t) {
      const int32_t id = token_ids[static_cast<size_t>(off + t)];
      if (id < 0 || id >= cfg_.vocab_size) {
        throw std::out_of_range("Qwen35MoeModel: token id " + std::to_string(id) + " is outside [0, " + std::to_string(cfg_.vocab_size) + ")");
      }
      std::copy_n(table + static_cast<int64_t>(id) * hidden, hidden, embed_rows_host_.data() + t * hidden);
    }
    Qwen35MoeStage& st0 = *stages_[0];
    st0.LoadEmbedding(embed_rows_host_.data(), rows);
    embed_ms += MsSince(t_embed);
    Qwen35MoeChunk chunk;
    chunk.rows = rows;
    chunk.first_pos = p;
    chunk.is_prefill = true;
    chunk.gdn_prefill_recurrent = gdn_recurrent_;
    st0.RunLayers(chunk);
    stage0_ms += st0.LastRunMs();
    const int slot = static_cast<int>(i % kSlots);
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&] { return !busy[slot] || failed; });
      if (failed) break;
    }
    const auto t_store = Clock::now();
    st0.StoreResidual(pp_ring_[static_cast<size_t>(slot)]->data(), rows);
    handoff_ms += MsSince(t_store);
    {
      std::lock_guard<std::mutex> lk(mu);
      busy[slot] = true;
      queue.push_back(Job{slot, rows, p, i + 1 == plan.size()});
    }
    cv.notify_all();
    off += rows;
    p += rows;
  }
  // wait for the second stage to drain, then stop it
  {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return failed || (queue.empty() && !busy[0] && !busy[1]); });
  }
  {
    std::lock_guard<std::mutex> lk(mu);
    stop = true;
  }
  cv.notify_all();
  worker.join();
  if (error) std::rethrow_exception(error);
  pos_ = p;
  last_.embed_ms += embed_ms;
  last_.stage_ms[0] += stage0_ms;
  last_.stage_ms[1] += stage1_ms;
  last_.handoff_ms += handoff_ms;
  last_.head_ms += head_ms;
  last_.rows = plan.back();
  last_.chunks += static_cast<int64_t>(plan.size());
  return logits;
}

}  // namespace r4dx::model
