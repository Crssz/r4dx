#include "qwen35moe_model.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

[[noreturn]] void Fail(const std::string& msg) { throw std::runtime_error("Qwen35MoeModel: " + msg); }

// Model::SummaryInvTemperature's rule (model.cpp): the device summary's measured tolerance band stops at ~0.01.
constexpr float kMinSummaryTemperature = 0.01f;
float SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;
  if (params.temperature < kMinSummaryTemperature) return 0.0f;
  const float inv = 1.0f / params.temperature;
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

std::string Lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// "a,b" -> {a, b}
std::vector<int> ParseDevices(const std::string& s, const char* what) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t used = 0;
    int v = -1;
    try {
      v = std::stoi(tok, &used);
    } catch (const std::exception&) {
      used = 0;
    }
    if (used == 0 || used != tok.size() || v < 0) {
      throw std::invalid_argument(std::string(what) + "='" + s + "': '" + tok + "' is not a device ordinal");
    }
    out.push_back(v);
  }
  if (out.size() != 2) {
    throw std::invalid_argument(std::string(what) + "='" + s + "': want two device ordinals \"a,b\" (stage 0, stage 1)");
  }
  return out;
}

// The borrowed pointers of one resident layer (the container outlives the stage that uses them).
Qwen35MoeLayerRefs LayerRefs(const Qwen35MoeContainer& c, int64_t i) {
  const qwen35moe::LayerWeights& lw = c.Layer(i);
  Qwen35MoeLayerRefs r;
  r.index = i;
  r.full_attention = lw.full_attention;
  r.input_layernorm = lw.input_layernorm.data();
  r.post_attention_layernorm = lw.post_attention_layernorm.data();
  if (!lw.full_attention) {
    if (!lw.gdn) Fail("layer " + std::to_string(i) + " has no GDN weights");
    const qwen4exp::GdnWeights& g = *lw.gdn;
    r.gdn.in_proj_qkv = &g.in_proj_qkv;
    r.gdn.in_proj_z = &g.in_proj_z;
    r.gdn.in_proj_a = g.in_proj_a.data();
    r.gdn.in_proj_b = g.in_proj_b.data();
    r.gdn.conv1d_weight = g.conv1d_weight.data();
    r.gdn.A_log = g.A_log.data();
    r.gdn.dt_bias = g.dt_bias.data();
    r.gdn.norm_weight = g.norm_weight.data();
    r.gdn.out_proj = &g.out_proj;
  } else {
    if (!lw.attn) Fail("layer " + std::to_string(i) + " has no attention weights");
    const qwen35moe::AttnWeights& a = *lw.attn;
    r.attn.qg = &a.qg;
    r.attn.k = &a.k;
    r.attn.v = &a.v;
    r.attn.o = &a.o;
    r.attn.q_norm = a.q_norm.data();
    r.attn.k_norm = a.k_norm.data();
    r.attn.k_descale = a.k_descale.data();
    r.attn.v_descale = a.v_descale.data();
  }
  const qwen4exp::MoeWeights& mw = lw.moe;
  r.moe.router = mw.router.data();
  r.moe.shared_gate_proj = &mw.shared.gate;
  r.moe.shared_up_proj = &mw.shared.up;
  r.moe.shared_down_proj = &mw.shared.down;
  r.moe.shared_gate = mw.shared_gate.data();
  r.moe.expert_gate_up.reserve(mw.experts.size());
  r.moe.expert_down.reserve(mw.experts.size());
  for (const qwen4exp::ExpertLinears& e : mw.experts) {
    r.moe.expert_gate_up.push_back(&e.gate_up);
    r.moe.expert_down.push_back(&e.down);
  }
  if (mw.gate_up_store.layout == Layout::kTrellis && mw.down_store.layout == Layout::kTrellis) {
    Qwen35MoeTrellisPtrs& tp = r.moe.trellis;
    tp.gate_up_w = mw.gate_up_store.trellis_w.data();
    tp.gate_up_suh = mw.gate_up_store.trellis_suh.data();
    tp.gate_up_svh = mw.gate_up_store.trellis_svh.data();
    tp.gate_up_bits = mw.gate_up_store.bits;
    tp.down_w = mw.down_store.trellis_w.data();
    tp.down_suh = mw.down_store.trellis_suh.data();
    tp.down_svh = mw.down_store.trellis_svh.data();
    tp.down_bits = mw.down_store.bits;
  }
  return r;
}

}  // namespace

std::vector<int64_t> Qwen35MoeModel::PlanChunks(int64_t rows) {
  std::vector<int64_t> out;
  while (rows >= kChunkRows) {
    out.push_back(kChunkRows);
    rows -= kChunkRows;
  }
  if (rows >= 64) {
    out.push_back(rows / 64 * 64);
    rows -= rows / 64 * 64;
  }
  if (rows > 0) out.push_back(rows);
  return out;
}

// ---- load -----------------------------------------------------------------------------------------------------------

std::unique_ptr<Qwen35MoeModel> Qwen35MoeModel::Load(const Qwen35MoeModelOptions& options) {
  std::unique_ptr<Qwen35MoeModel> mp(new Qwen35MoeModel());
  Qwen35MoeModel& m = *mp;
  m.opt_ = options;
  m.prompt_checkpoint_ = options.prompt_checkpoint;

  const Qwen35MoeContainerInfo info =
      Qwen35MoeContainer::Inspect(options.container_path, options.layout, options.lm_head_layout);
  m.cfg_ = info.config;
  m.generic_config_ = info.model_config;
  m.model_id_ = info.model_id;
  const Qwen35MoeConfig& cfg = m.cfg_;
  const int64_t L = info.layers_present;  // == cfg.num_hidden_layers unless the container is a prefix (r4dx-convert --layers N)

  if (options.max_ctx < 1) throw std::invalid_argument("Qwen35MoeModel::Load: max_ctx must be >= 1");
  if (options.max_ctx > cfg.max_position_embeddings) {
    throw std::invalid_argument("Qwen35MoeModel::Load: max_ctx " + std::to_string(options.max_ctx) +
                                " exceeds the checkpoint's max_position_embeddings " +
                                std::to_string(cfg.max_position_embeddings));
  }
  if (options.max_ctx > std::numeric_limits<int32_t>::max() / 2) {
    throw std::invalid_argument("Qwen35MoeModel::Load: max_ctx is too large for the 32-bit position arrays");
  }

  // ---- devices and the split -----------------------------------------------------------------------------------
  int visible = 0;
  R4DX_HIP_CHECK(hipGetDeviceCount(&visible));
  if (visible < 1) Fail("no HIP device");
  std::vector<int> devices = options.devices;
  if (devices.empty()) {
    if (const char* e = std::getenv("R4DX_Q35_DEVICES"); e != nullptr && *e != '\0') {
      devices = ParseDevices(e, "R4DX_Q35_DEVICES");
    } else {
      devices = visible >= 2 ? std::vector<int>{0, 1} : std::vector<int>{0, 0};
    }
  }
  if (devices.size() != 2) throw std::invalid_argument("Qwen35MoeModel::Load: devices must name two ordinals");
  for (int d : devices) {
    if (d < 0 || d >= visible) {
      throw std::invalid_argument("Qwen35MoeModel::Load: device ordinal " + std::to_string(d) + " is not visible (" +
                                  std::to_string(visible) + " HIP device(s))");
    }
  }
  int64_t split = options.split;
  if (split == 0) {
    if (const char* e = std::getenv("R4DX_Q35_SPLIT"); e != nullptr && *e != '\0') {
      char* end = nullptr;
      const long long v = std::strtoll(e, &end, 10);
      if (end == e || *end != '\0' || v < 0) {
        throw std::invalid_argument(std::string("R4DX_Q35_SPLIT='") + e + "' is not a layer index");
      }
      split = v;
    }
  }
  const bool two_stage = devices[0] != devices[1] || split > 0;
  if (two_stage) {
    if (L < 2) Fail("a two-stage split needs at least two layers");
    if (split == 0) {
      split = ChooseSplit(info.PerLayerDeviceBytes(), /*head_bytes=*/0, info.TailDeviceBytes(), options.weight_cap_bytes).k;
    }
    if (split < 1 || split > L - 1) {
      throw std::invalid_argument("Qwen35MoeModel::Load: split " + std::to_string(split) + " is outside [1, " +
                                  std::to_string(L - 1) + "]");
    }
  } else {
    split = L;
  }
  m.split_ = split;
  m.gdn_recurrent_ = options.gdn_prefill_recurrent;
  if (const char* e = std::getenv("R4DX_Q35_GDN_PREFILL"); e != nullptr && *e != '\0') {
    const std::string v = Lower(e);
    if (v == "recurrent") {
      m.gdn_recurrent_ = true;
    } else if (v != "chunked") {
      throw std::invalid_argument(std::string("R4DX_Q35_GDN_PREFILL='") + e + "' (want chunked|recurrent)");
    }
  }
  if (m.gdn_recurrent_) {
    std::fprintf(stderr, "[r4dx::model::Qwen35MoeModel] GDN prefill through the recurrent decode kernels (reference path, slow)\n");
  }

  // ---- containers and stages ------------------------------------------------------------------------------------
  struct Range {
    int device;
    int64_t first, last;
    bool embed, head;
  };
  std::vector<Range> ranges;
  if (two_stage) {
    ranges.push_back({devices[0], 0, split, true, false});
    ranges.push_back({devices[1], split, L, false, true});
  } else {
    ranges.push_back({devices[0], 0, L, true, true});
  }
  for (const Range& r : ranges) {
    Qwen35MoeLoadOptions lo;
    lo.layout = options.layout;
    lo.lm_head_layout = options.lm_head_layout;
    lo.device = r.device;
    lo.layer_first = r.first;
    lo.layer_last = r.last;
    lo.load_embed = r.embed;
    lo.load_head = r.head;
    m.containers_.push_back(std::make_unique<Qwen35MoeContainer>(Qwen35MoeContainer::Load(options.container_path, lo)));
  }
  for (size_t s = 0; s < ranges.size(); ++s) {
    const Range& r = ranges[s];
    const Qwen35MoeContainer& c = *m.containers_[s];
    std::vector<Qwen35MoeLayerRefs> layers;
    for (int64_t i = r.first; i < r.last; ++i) layers.push_back(LayerRefs(c, i));
    std::optional<Qwen35MoeHeadRefs> head;
    if (r.head) {
      Qwen35MoeHeadRefs h;
      h.final_norm = c.FinalNorm().data();
      h.lm_head = &c.LmHead();
      head = h;
    }
    Qwen35MoeStageOptions so;
    so.device = r.device;
    so.config = cfg;
    so.max_ctx = options.max_ctx;
    so.max_rows = kChunkRows;
    so.prompt_checkpoint = options.prompt_checkpoint;
    so.grouped_decode = options.grouped_decode;
    m.stages_.push_back(std::make_unique<Qwen35MoeStage>(so, std::move(layers), std::move(head)));
    {
      Qwen35MoeDeviceScope dev(r.device);
      m.containers_[s]->ZeroTrellisTickets(m.stages_[s]->Stream());
      R4DX_HIP_CHECK(hipStreamSynchronize(m.stages_[s]->Stream()));
    }
    std::fprintf(stderr, "[r4dx::model::Qwen35MoeModel] stage %zu: layers [%lld, %lld) on device %d, %.3f GiB weights%s%s\n", s,
                 static_cast<long long>(r.first), static_cast<long long>(r.last), r.device,
                 static_cast<double>(c.DeviceBytes()) / (1024.0 * 1024.0 * 1024.0), r.embed ? ", host embedding" : "",
                 r.head ? ", final norm + lm_head" : "");
  }

  // ---- host buffers -----------------------------------------------------------------------------------------------
  const size_t R = static_cast<size_t>(kChunkRows);
  m.embed_rows_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(cfg.hidden_size));
  if (m.stages_.size() > 1) m.handoff_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(cfg.hidden_size));
  return mp;
}

Qwen35MoeModel::~Qwen35MoeModel() = default;

// ---- one chunk -------------------------------------------------------------------------------------------------------

void Qwen35MoeModel::RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen35MoeHeadRequest* head,
                              HeadOut* out, const IsolatedLayer* iso) {
  if (rows < 1 || rows > kChunkRows || (rows > 64 && rows % 64 != 0)) {
    Fail("a chunk is 1..64 rows or a multiple of 64 up to " + std::to_string(kChunkRows) + " (got " + std::to_string(rows) + ")");
  }
  if (pos_ + rows > opt_.max_ctx) {
    Fail("position " + std::to_string(pos_ + rows) + " exceeds max_ctx " + std::to_string(opt_.max_ctx));
  }
  const auto t_start = Clock::now();
  const int64_t hidden = cfg_.hidden_size;

  // The embedding rows (host gather; a token outside the vocabulary throws here, before any state moves). A layer-isolated
  // chunk starts from the caller's residual and needs no embedding.
  if (iso == nullptr) {
    const uint16_t* table = containers_.front()->EmbedTokensHost();
    for (int64_t t = 0; t < rows; ++t) {
      const int32_t id = tokens[t];
      if (id < 0 || id >= cfg_.vocab_size) {
        throw std::out_of_range("Qwen35MoeModel: token id " + std::to_string(id) + " is outside [0, " +
                                std::to_string(cfg_.vocab_size) + ")");
      }
      std::copy_n(table + static_cast<int64_t>(id) * hidden, hidden, embed_rows_host_.data() + t * hidden);
    }
  }
  last_.embed_ms += MsSince(t_start);

  Qwen35MoeChunk chunk;
  chunk.rows = rows;
  chunk.first_pos = pos_;
  chunk.is_prefill = is_prefill;
  chunk.gdn_prefill_recurrent = gdn_recurrent_;
  if (iso != nullptr) chunk.only_layer = iso->layer;

  if (iso == nullptr) {
    const auto t_load = Clock::now();
    stages_.front()->LoadEmbedding(embed_rows_host_.data(), rows);
    last_.embed_ms += MsSince(t_load);
  }
  for (size_t s = 0; s < stages_.size(); ++s) {
    Qwen35MoeStage& st = *stages_[s];
    if (iso != nullptr) {
      if (iso->layer < st.FirstLayer() || iso->layer >= st.EndLayer()) continue;
      st.LoadResidual(iso->residual_in, rows);
      st.RunLayers(chunk);
      last_.stage_ms[std::min<size_t>(s, 1)] += st.LastRunMs();
      continue;
    }
    if (s > 0) {
      const auto t0 = Clock::now();
      st.LoadResidual(handoff_host_.data(), rows);
      last_.handoff_ms += MsSince(t0);
    }
    st.RunLayers(chunk);
    last_.stage_ms[std::min<size_t>(s, 1)] += st.LastRunMs();
    if (s + 1 < stages_.size()) {
      const auto t0 = Clock::now();
      st.StoreResidual(handoff_host_.data(), rows);
      last_.handoff_ms += MsSince(t0);
    }
  }

  if (head != nullptr) {
    Qwen35MoeStage& last = *stages_.back();
    Qwen35MoeHeadResult r = last.Head(rows, *head);
    last_.head_ms += last.LastHeadMs();
    if (out != nullptr) {
      out->logits = std::move(r.logits);
      out->token = r.token;
      out->summary = r.summary;
    }
  }
  pos_ += rows;
  last_.rows = rows;
  ++last_.chunks;
}

void Qwen35MoeModel::RunLayerIsolated(const std::vector<int32_t>& token_ids, int64_t layer, const uint16_t* residual_in) {
  if (token_ids.empty()) throw std::runtime_error("Qwen35MoeModel::RunLayerIsolated: token_ids is empty");
  if (residual_in == nullptr) throw std::invalid_argument("Qwen35MoeModel::RunLayerIsolated: residual_in is null");
  if (layer < 0 || layer >= cfg_.num_hidden_layers) {
    throw std::out_of_range("Qwen35MoeModel::RunLayerIsolated: layer " + std::to_string(layer) + " is outside [0, " +
                            std::to_string(cfg_.num_hidden_layers) + ")");
  }
  Reset();
  last_ = Timings{};
  const int64_t hidden = cfg_.hidden_size;
  int64_t off = 0;
  for (int64_t n : PlanChunks(static_cast<int64_t>(token_ids.size()))) {
    const IsolatedLayer iso{layer, residual_in + off * hidden};
    RunChunk(token_ids.data() + off, n, /*is_prefill=*/true, nullptr, nullptr, &iso);
    off += n;
  }
}

// ---- public forward ---------------------------------------------------------------------------------------------------

std::vector<float> Qwen35MoeModel::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::runtime_error("Qwen35MoeModel::Prefill: token_ids is empty");
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kLogits, 0.0f};
  const std::vector<int64_t> plan = PlanChunks(static_cast<int64_t>(token_ids.size()));
  int64_t off = 0;
  for (size_t i = 0; i < plan.size(); ++i) {
    const bool last_chunk = i + 1 == plan.size();
    RunChunk(token_ids.data() + off, plan[i], /*is_prefill=*/true, (last_chunk || capture_active_) ? &req : nullptr,
             last_chunk ? &out : nullptr);
    off += plan[i];
  }
  at_prefill_end_ = true;
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

std::vector<float> Qwen35MoeModel::DecodeStep(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kLogits, 0.0f};
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

int32_t Qwen35MoeModel::DecodeStepGreedy(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kArgmax, 0.0f};
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  last_.total_ms = MsSince(t0);
  return out.token;
}

int32_t Qwen35MoeModel::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
  // Model::DecodeStepSampled's contract: a greedy request keeps the greedy path and draws nothing; a degenerate
  // temperature takes the full-logits path; otherwise ONE draw, the device row summary, and the row-granular fallback.
  if (params.temperature <= 0.0f) return DecodeStepGreedy(token_id);
  const float inv_t = SummaryInvTemperature(params);
  const int64_t vocab = cfg_.vocab_size;
  if (inv_t == 0.0f) {
    const std::vector<float> logits = DecodeStep(token_id);
    return kernels::SampleCanonical(logits.data(), vocab, params, kernels::DrawUniform01(rng));
  }
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kSummary, inv_t};
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  last_.total_ms = MsSince(t0);
  const double u = kernels::DrawUniform01(rng);
  const kernels::SummarySampleResult r = kernels::SampleFromSummary(out.summary, params, u);
  if (r.resolved) return r.token;
  ++sampled_fallback_rows_;
  const std::vector<float> row = stages_.back()->LastLogitsRow();  // this step's own fp32 row, no second lm_head pass
  return kernels::SampleCanonical(row.data(), vocab, params, u);
}

// ---- state ------------------------------------------------------------------------------------------------------------

void Qwen35MoeModel::Reset() {
  for (size_t s = 0; s < stages_.size(); ++s) {
    Qwen35MoeDeviceScope dev(stages_[s]->Device());
    // a trellis GEMM that did not complete can leave its split-group tickets non-zero (container.cpp's Reset note)
    containers_[s]->ZeroTrellisTickets(stages_[s]->Stream());
    stages_[s]->Reset();
  }
  pos_ = 0;
  at_prefill_end_ = false;
  ckpt_pos_ = -1;
  last_ = Timings{};
}

void Qwen35MoeModel::SaveCheckpoint() {
  if (!prompt_checkpoint_) {
    throw std::runtime_error("Qwen35MoeModel::SaveCheckpoint: this model was loaded without ModelOptions::prompt_checkpoint");
  }
  if (pos_ == 0) throw std::runtime_error("Qwen35MoeModel::SaveCheckpoint: nothing has been fed yet");
  if (!at_prefill_end_) {
    throw std::runtime_error("Qwen35MoeModel::SaveCheckpoint: a decode step has run since the last prefill; checkpoint "
                             "right after Prefill");
  }
  for (auto& st : stages_) st->SaveCheckpoint();
  ckpt_pos_ = pos_;
}

void Qwen35MoeModel::RestoreCheckpoint() {
  if (!prompt_checkpoint_) {
    throw std::runtime_error("Qwen35MoeModel::RestoreCheckpoint: this model was loaded without ModelOptions::prompt_checkpoint");
  }
  if (ckpt_pos_ < 0) throw std::runtime_error("Qwen35MoeModel::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  for (auto& st : stages_) st->RestoreCheckpoint(ckpt_pos_);
  pos_ = ckpt_pos_;
  at_prefill_end_ = true;  // the restored state IS a post-prefill state
}

void Qwen35MoeModel::SetCapture(Qwen35MoeCaptureFn fn) {
  capture_active_ = static_cast<bool>(fn);
  for (auto& st : stages_) st->SetCapture(fn);
}

void Qwen35MoeModel::SetGroupedDecode(bool on, bool with_capture) {
  for (auto& st : stages_) st->SetGroupedDecode(on, with_capture);
}

int64_t Qwen35MoeModel::GroupedDecodeLayers() const {
  int64_t n = 0;
  for (const auto& st : stages_) n += st->GroupedDecodeLayers();
  return n;
}

std::vector<VramReport> Qwen35MoeModel::Vram() const {
  std::vector<VramReport> out;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  for (size_t s = 0; s < stages_.size(); ++s) {
    VramReport r;
    r.rank = static_cast<int>(s);
    r.device = stages_[s]->Device();
    Qwen35MoeDeviceScope dev(r.device);
    size_t free_b = 0, total_b = 0;
    R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    r.used_gib = static_cast<double>(total_b - free_b) / kGiB;
    r.free_gib = static_cast<double>(free_b) / kGiB;
    r.total_gib = static_cast<double>(total_b) / kGiB;
    r.buffers_gib = static_cast<double>(core::DeviceBufferBytes(r.device)) / kGiB;
    out.push_back(r);
  }
  return out;
}

}  // namespace r4dx::model
