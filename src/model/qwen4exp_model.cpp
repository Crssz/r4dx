#include "qwen4exp_model.h"

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

[[noreturn]] void Fail(const std::string& msg) { throw std::runtime_error("Qwen4ExpModel: " + msg); }

// Model::SummaryInvTemperature's rule (model.cpp): the device summary's measured tolerance band stops at ~0.01.
constexpr float kMinSummaryTemperature = 0.01f;
float SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;
  if (params.temperature < kMinSummaryTemperature) return 0.0f;
  const float inv = 1.0f / params.temperature;
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

std::string HexLower(const std::array<uint8_t, 32>& d) {
  static const char* kHex = "0123456789abcdef";
  std::string s;
  for (uint8_t b : d) {
    s.push_back(kHex[b >> 4]);
    s.push_back(kHex[b & 15]);
  }
  return s;
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

Qwen4ExpHcRefs HcRefs(const qwen4exp::HcWeights& w) {
  Qwen4ExpHcRefs r;
  r.norm = w.norm.data();
  r.down = w.down.data();
  r.up = w.up.data();
  r.inject = w.inject.empty() ? nullptr : w.inject.data();
  return r;
}

// The borrowed pointers of one resident layer (the container outlives the stage that uses them).
Qwen4ExpLayerRefs LayerRefs(const Qwen4ExpContainer& c, int64_t i) {
  const qwen4exp::LayerWeights& lw = c.Layer(i);
  Qwen4ExpLayerRefs r;
  r.index = i;
  r.qsa = lw.qsa;
  r.hc_attn = HcRefs(lw.hc_attn);
  r.hc_mlp = HcRefs(lw.hc_mlp);
  if (!lw.qsa) {
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
    const qwen4exp::QsaWeights& a = *lw.attn;
    r.attn.qg = &a.qg;
    r.attn.k = &a.k;
    r.attn.v = &a.v;
    r.attn.o = &a.o;
    r.attn.q_norm = a.q_norm.data();
    r.attn.k_norm = a.k_norm.data();
    r.attn.indexer_qk = a.indexer_qk.data();
    r.attn.indexer_q_norm = a.indexer_q_norm.data();
    r.attn.indexer_k_norm = a.indexer_k_norm.data();
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
    r.moe.trellis_ptrs.gate_up_w = mw.gate_up_store.trellis_w.data();
    r.moe.trellis_ptrs.gate_up_suh = mw.gate_up_store.trellis_suh.data();
    r.moe.trellis_ptrs.gate_up_svh = mw.gate_up_store.trellis_svh.data();
    r.moe.trellis_ptrs.gate_up_bits = mw.gate_up_store.bits;
    r.moe.trellis_ptrs.down_w = mw.down_store.trellis_w.data();
    r.moe.trellis_ptrs.down_suh = mw.down_store.trellis_suh.data();
    r.moe.trellis_ptrs.down_svh = mw.down_store.trellis_svh.data();
    r.moe.trellis_ptrs.down_bits = mw.down_store.bits;
  }
  if (lw.ple) {
    const qwen4exp::PleWeights& p = *lw.ple;
    Qwen4ExpPleRefs pr;
    pr.key_proj = p.key_proj.data();
    pr.value_proj = p.value_proj.data();
    pr.norm_key = p.norm_key.data();
    pr.norm_query = p.norm_query.data();
    pr.norm_conv = p.norm_conv.data();
    pr.conv1d = p.conv1d.data();
    r.ple = pr;
  }
  return r;
}

}  // namespace

// ---- load -----------------------------------------------------------------------------------------------------------

std::unique_ptr<Qwen4ExpModel> Qwen4ExpModel::Load(const Qwen4ExpModelOptions& options) {
  std::unique_ptr<Qwen4ExpModel> mp(new Qwen4ExpModel());
  Qwen4ExpModel& m = *mp;
  m.opt_ = options;
  m.prompt_checkpoint_ = options.prompt_checkpoint;

  const Qwen4ExpContainerInfo info =
      Qwen4ExpContainer::Inspect(options.container_path, options.layout, options.lm_head_layout);
  m.cfg_ = info.config;
  m.generic_config_ = info.model_config;
  m.model_id_ = info.model_id;
  const Qwen4ExpConfig& cfg = m.cfg_;
  const int64_t L = info.layers_present;  // == cfg.num_hidden_layers unless the container is a prefix (r4dx-convert --layers N)

  if (options.max_ctx < 1) throw std::invalid_argument("Qwen4ExpModel::Load: max_ctx must be >= 1");
  if (options.max_ctx > cfg.max_position_embeddings) {
    throw std::invalid_argument("Qwen4ExpModel::Load: max_ctx " + std::to_string(options.max_ctx) +
                                " exceeds the checkpoint's max_position_embeddings " +
                                std::to_string(cfg.max_position_embeddings));
  }
  if (options.max_ctx > std::numeric_limits<int32_t>::max() / 2) {
    throw std::invalid_argument("Qwen4ExpModel::Load: max_ctx is too large for the 32-bit position arrays");
  }

  // ---- devices and the split -----------------------------------------------------------------------------------
  int visible = 0;
  R4DX_HIP_CHECK(hipGetDeviceCount(&visible));
  if (visible < 1) Fail("no HIP device");
  std::vector<int> devices = options.devices;
  if (devices.empty()) {
    if (const char* e = std::getenv("R4DX_Q4_DEVICES"); e != nullptr && *e != '\0') {
      devices = ParseDevices(e, "R4DX_Q4_DEVICES");
    } else {
      devices = visible >= 2 ? std::vector<int>{0, 1} : std::vector<int>{0, 0};
    }
  }
  if (devices.size() != 2) throw std::invalid_argument("Qwen4ExpModel::Load: devices must name two ordinals");
  for (int d : devices) {
    if (d < 0 || d >= visible) {
      throw std::invalid_argument("Qwen4ExpModel::Load: device ordinal " + std::to_string(d) + " is not visible (" +
                                  std::to_string(visible) + " HIP device(s))");
    }
  }
  int64_t split = options.split;
  if (split == 0) {
    if (const char* e = std::getenv("R4DX_Q4_SPLIT"); e != nullptr && *e != '\0') {
      char* end = nullptr;
      const long long v = std::strtoll(e, &end, 10);
      if (end == e || *end != '\0' || v < 0) {
        throw std::invalid_argument(std::string("R4DX_Q4_SPLIT='") + e + "' is not a layer index");
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
      throw std::invalid_argument("Qwen4ExpModel::Load: split " + std::to_string(split) + " is outside [1, " +
                                  std::to_string(L - 1) + "]");
    }
  } else {
    split = L;
  }
  m.split_ = split;
  m.gdn_recurrent_ = options.gdn_prefill_recurrent;
  if (const char* e = std::getenv("R4DX_Q4_GDN_PREFILL"); e != nullptr && *e != '\0') {
    const std::string v = Lower(e);
    if (v == "recurrent") {
      m.gdn_recurrent_ = true;
    } else if (v != "chunked") {
      throw std::invalid_argument(std::string("R4DX_Q4_GDN_PREFILL='") + e + "' (want chunked|recurrent)");
    }
  }
  if (m.gdn_recurrent_) {
    std::fprintf(stderr, "[r4dx::model::Qwen4ExpModel] GDN prefill through the recurrent decode kernels (reference path, slow)\n");
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
    Qwen4ExpLoadOptions lo;
    lo.layout = options.layout;
    lo.lm_head_layout = options.lm_head_layout;
    lo.device = r.device;
    lo.layer_first = r.first;
    lo.layer_last = r.last;
    lo.load_embed = r.embed;
    lo.load_head = r.head;
    m.containers_.push_back(std::make_unique<Qwen4ExpContainer>(Qwen4ExpContainer::Load(options.container_path, lo)));
  }
  for (size_t s = 0; s < ranges.size(); ++s) {
    const Range& r = ranges[s];
    const Qwen4ExpContainer& c = *m.containers_[s];
    std::vector<Qwen4ExpLayerRefs> layers;
    for (int64_t i = r.first; i < r.last; ++i) layers.push_back(LayerRefs(c, i));
    std::optional<Qwen4ExpHeadRefs> head;
    if (r.head) {
      Qwen4ExpHeadRefs h;
      h.hc_final = HcRefs(c.HcFinal());
      h.lm_head = &c.LmHead();
      head = h;
    }
    Qwen4ExpStageOptions so;
    so.device = r.device;
    so.config = cfg;
    so.max_ctx = options.max_ctx;
    so.max_rows = kChunkRows;
    so.prompt_checkpoint = options.prompt_checkpoint;
    m.stages_.push_back(std::make_unique<Qwen4ExpStage>(so, std::move(layers), std::move(head)));
    {
      Qwen4ExpDeviceScope dev(r.device);
      m.containers_[s]->ZeroTrellisTickets(m.stages_[s]->Stream());
      R4DX_HIP_CHECK(hipStreamSynchronize(m.stages_[s]->Stream()));
    }
    std::fprintf(stderr, "[r4dx::model::Qwen4ExpModel] stage %zu: layers [%lld, %lld) on device %d, %.3f GiB weights%s%s\n", s,
                 static_cast<long long>(r.first), static_cast<long long>(r.last), r.device,
                 static_cast<double>(c.DeviceBytes()) / (1024.0 * 1024.0 * 1024.0), r.embed ? ", host embedding" : "",
                 r.head ? ", final mixer + lm_head" : "");
  }

  // ---- host buffers -----------------------------------------------------------------------------------------------
  const size_t R = static_cast<size_t>(kChunkRows);
  m.embed_rows_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(cfg.hidden_size));
  if (m.stages_.size() > 1) {
    m.handoff_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(cfg.hc_count * cfg.hidden_size));
  }
  // ---- the n-gram table -------------------------------------------------------------------------------------------
  m.ple_enabled_ = !cfg.ple_layer_indices.empty();
  if (m.ple_enabled_) {
    if (!info.ngram) {
      Fail("the container has PLE layers but carries no __metadata__.ngram_sidecar record (" + options.container_path + ")");
    }
    const Qwen4ExpNgramRef& ref = *info.ngram;
    // Open reads the 4 KiB header through the OS file cache and closes it again; the unbuffered reader starts after.
    qwen4exp::NgramSidecar sidecar = qwen4exp::NgramSidecar::Open(ref.resolved_path);
    const qwen4exp::NgramSidecarHeader& h = sidecar.header();
    if (HexLower(h.body_sha256) != Lower(ref.sha256)) {
      Fail("n-gram sidecar " + ref.resolved_path + " has body sha256 " + HexLower(h.body_sha256) +
           " but the container records " + ref.sha256 + " (a different table)");
    }
    if (static_cast<int64_t>(h.rows) != ref.rows) {
      Fail("n-gram sidecar " + ref.resolved_path + " has " + std::to_string(h.rows) + " rows, the container records " +
           std::to_string(ref.rows));
    }
    if (ref.row_bytes != 0 && static_cast<int64_t>(h.row_bytes) != ref.row_bytes) {
      Fail("n-gram sidecar " + ref.resolved_path + " has row_bytes " + std::to_string(h.row_bytes) +
           ", the container records " + std::to_string(ref.row_bytes));
    }
    if (static_cast<int64_t>(h.heads) * h.head_dim != cfg.ple_embed_dim) {
      Fail("n-gram sidecar rows are " + std::to_string(h.heads * h.head_dim) + " wide, the config's ple_embed_dim is " +
           std::to_string(cfg.ple_embed_dim));
    }
    if (h.eos_token_id != cfg.eos_token_id) {
      Fail("n-gram sidecar eos id " + std::to_string(h.eos_token_id) + " differs from the config's " +
           std::to_string(cfg.eos_token_id));
    }
    m.ngram_header_ = h;
    m.ngram_ids_ = std::make_unique<qwen4exp::NgramIds>(h);
    m.ngram_reader_ = std::make_unique<qwen4exp::NgramReader>(sidecar);
    m.ngram_ids_host_ = core::PinnedBuffer<int64_t>(R * qwen4exp::NgramIds::kHeads);
    m.ngram_rows_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(cfg.ple_embed_dim));
  }
  return mp;
}

Qwen4ExpModel::~Qwen4ExpModel() {
  DrainNgram();  // reads in flight write into the pinned buffers below
}

// ---- n-gram plumbing --------------------------------------------------------------------------------------------------

const uint16_t* Qwen4ExpModel::WaitNgramRows() {
  if (!ngram_reader_) Fail("a PLE layer asked for n-gram rows but the model has no n-gram reader");
  if (ngram_pending_) {
    const auto t0 = Clock::now();
    const qwen4exp::NgramReader::Handle h = *ngram_pending_;
    ngram_pending_.reset();  // Wait consumes the handle, also when it throws
    ngram_reader_->Wait(h);
    ngram_wait_ms_ += MsSince(t0);
  }
  return ngram_rows_host_.data();
}

void Qwen4ExpModel::DrainNgram() noexcept {
  if (!ngram_pending_ || !ngram_reader_) return;
  const qwen4exp::NgramReader::Handle h = *ngram_pending_;
  ngram_pending_.reset();
  try {
    ngram_reader_->Wait(h);
  } catch (...) {
  }
}

// ---- one chunk -------------------------------------------------------------------------------------------------------

void Qwen4ExpModel::RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen4ExpHeadRequest* head,
                             HeadOut* out, const IsolatedLayer* iso) {
  if (rows < 1 || rows > kChunkRows) Fail("a chunk is 1.." + std::to_string(kChunkRows) + " rows");
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
        throw std::out_of_range("Qwen4ExpModel: token id " + std::to_string(id) + " is outside [0, " +
                                std::to_string(cfg_.vocab_size) + ")");
      }
      std::copy_n(table + static_cast<int64_t>(id) * hidden, hidden, embed_rows_host_.data() + t * hidden);
    }
  }

  // The n-gram rows of the chunk's tokens: ids on the host, reads submitted now, awaited at the PLE layer.
  DrainNgram();
  ngram_wait_ms_ = 0;
  if (ple_enabled_) {
    const qwen4exp::NgramIds::State snapshot = ngram_ids_->Snapshot();
    try {
      ngram_ids_->Generate(tokens, static_cast<size_t>(rows), ngram_ids_host_.data());
      ngram_pending_ = ngram_reader_->Submit(ngram_ids_host_.data(), static_cast<size_t>(rows) * qwen4exp::NgramIds::kHeads,
                                             ngram_rows_host_.data());
    } catch (...) {
      ngram_ids_->Restore(snapshot);
      throw;
    }
  }

  last_.embed_ms += MsSince(t_start);

  Qwen4ExpChunk chunk;
  chunk.rows = rows;
  chunk.first_pos = pos_;
  chunk.is_prefill = is_prefill;
  chunk.gdn_prefill_recurrent = gdn_recurrent_;
  chunk.ple_rows = [this]() { return WaitNgramRows(); };
  if (iso != nullptr) chunk.only_layer = iso->layer;

  // Both stages are driven in sequence on this thread; each Qwen4ExpStage method sets its own device scope. Stage s > 0
  // takes the residual stage s-1 left in pinned host memory.
  if (iso == nullptr) stages_.front()->LoadEmbedding(embed_rows_host_.data(), rows);
  for (size_t s = 0; s < stages_.size(); ++s) {
    Qwen4ExpStage& st = *stages_[s];
    if (iso != nullptr) {
      if (iso->layer < st.FirstLayer() || iso->layer >= st.EndLayer()) continue;
      st.LoadResidual(iso->residual_in, rows);
      st.RunLayers(chunk);
      last_.stage_ms[std::min<size_t>(s, 1)] += st.LastRunMs();
      last_.ple_wait_ms += st.LastPleWaitMs();
      continue;
    }
    if (s > 0) {
      const auto t0 = Clock::now();
      st.LoadResidual(handoff_host_.data(), rows);
      last_.handoff_ms += MsSince(t0);
    }
    st.RunLayers(chunk);
    last_.stage_ms[std::min<size_t>(s, 1)] += st.LastRunMs();
    last_.ple_wait_ms += st.LastPleWaitMs();
    if (s + 1 < stages_.size()) {
      const auto t0 = Clock::now();
      st.StoreResidual(handoff_host_.data(), rows);
      last_.handoff_ms += MsSince(t0);
    }
  }
  // a stage without a PLE layer in range never consumed the rows: still settle the reads before the buffers are reused
  DrainNgram();

  if (head != nullptr) {
    Qwen4ExpStage& last = *stages_.back();
    Qwen4ExpHeadResult r = last.Head(rows, *head);
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

void Qwen4ExpModel::RunLayerIsolated(const std::vector<int32_t>& token_ids, int64_t layer, const uint16_t* residual_in) {
  if (token_ids.empty()) throw std::runtime_error("Qwen4ExpModel::RunLayerIsolated: token_ids is empty");
  if (residual_in == nullptr) throw std::invalid_argument("Qwen4ExpModel::RunLayerIsolated: residual_in is null");
  if (layer < 0 || layer >= cfg_.num_hidden_layers) {
    throw std::out_of_range("Qwen4ExpModel::RunLayerIsolated: layer " + std::to_string(layer) + " is outside [0, " +
                            std::to_string(cfg_.num_hidden_layers) + ")");
  }
  Reset();
  last_ = Timings{};
  const int64_t wide = cfg_.hc_count * cfg_.hidden_size;
  for (size_t off = 0; off < token_ids.size();) {
    const int64_t n = std::min<int64_t>(kChunkRows, static_cast<int64_t>(token_ids.size() - off));
    const IsolatedLayer iso{layer, residual_in + static_cast<int64_t>(off) * wide};
    RunChunk(token_ids.data() + off, n, /*is_prefill=*/true, nullptr, nullptr, &iso);
    off += static_cast<size_t>(n);
  }
}

// ---- public forward ---------------------------------------------------------------------------------------------------

std::vector<float> Qwen4ExpModel::Prefill(const std::vector<int32_t>& token_ids) {
  if (token_ids.empty()) throw std::runtime_error("Qwen4ExpModel::Prefill: token_ids is empty");
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen4ExpHeadRequest req{Qwen4ExpHeadRequest::Kind::kLogits, 0.0f};
  for (size_t off = 0; off < token_ids.size();) {
    const int64_t n = std::min<int64_t>(kChunkRows, static_cast<int64_t>(token_ids.size() - off));
    const bool last_chunk = off + static_cast<size_t>(n) == token_ids.size();
    RunChunk(token_ids.data() + off, n, /*is_prefill=*/true, (last_chunk || capture_active_) ? &req : nullptr,
             last_chunk ? &out : nullptr);
    off += static_cast<size_t>(n);
  }
  at_prefill_end_ = true;
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

std::vector<float> Qwen4ExpModel::DecodeStep(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen4ExpHeadRequest req{Qwen4ExpHeadRequest::Kind::kLogits, 0.0f};
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

int32_t Qwen4ExpModel::DecodeStepGreedy(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen4ExpHeadRequest req{Qwen4ExpHeadRequest::Kind::kArgmax, 0.0f};
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  last_.total_ms = MsSince(t0);
  return out.token;
}

int32_t Qwen4ExpModel::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) {
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
  const Qwen4ExpHeadRequest req{Qwen4ExpHeadRequest::Kind::kSummary, inv_t};
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

void Qwen4ExpModel::Reset() {
  DrainNgram();
  for (size_t s = 0; s < stages_.size(); ++s) {
    Qwen4ExpDeviceScope dev(stages_[s]->Device());
    // a trellis GEMM that did not complete can leave its split-group tickets non-zero (container.cpp's Reset note)
    containers_[s]->ZeroTrellisTickets(stages_[s]->Stream());
    stages_[s]->Reset();
  }
  if (ngram_ids_) ngram_ids_->Reset();
  pos_ = 0;
  at_prefill_end_ = false;
  ckpt_pos_ = -1;
  last_ = Timings{};
}

void Qwen4ExpModel::SaveCheckpoint() {
  if (!prompt_checkpoint_) {
    throw std::runtime_error("Qwen4ExpModel::SaveCheckpoint: this model was loaded without ModelOptions::prompt_checkpoint");
  }
  if (pos_ == 0) throw std::runtime_error("Qwen4ExpModel::SaveCheckpoint: nothing has been fed yet");
  if (!at_prefill_end_) {
    throw std::runtime_error("Qwen4ExpModel::SaveCheckpoint: a decode step has run since the last prefill; checkpoint "
                             "right after Prefill");
  }
  DrainNgram();
  for (auto& st : stages_) st->SaveCheckpoint();
  if (ngram_ids_) ckpt_ngram_ = ngram_ids_->Snapshot();
  ckpt_pos_ = pos_;
}

void Qwen4ExpModel::RestoreCheckpoint() {
  if (!prompt_checkpoint_) {
    throw std::runtime_error("Qwen4ExpModel::RestoreCheckpoint: this model was loaded without ModelOptions::prompt_checkpoint");
  }
  if (ckpt_pos_ < 0) throw std::runtime_error("Qwen4ExpModel::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  DrainNgram();
  for (auto& st : stages_) st->RestoreCheckpoint(ckpt_pos_);
  if (ngram_ids_) ngram_ids_->Restore(ckpt_ngram_);
  pos_ = ckpt_pos_;
  at_prefill_end_ = true;  // the restored state IS a post-prefill state
}

void Qwen4ExpModel::SetCapture(Qwen4ExpCaptureFn fn) {
  capture_active_ = static_cast<bool>(fn);
  for (auto& st : stages_) st->SetCapture(fn);
}

std::vector<VramReport> Qwen4ExpModel::Vram() const {
  std::vector<VramReport> out;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  for (size_t s = 0; s < stages_.size(); ++s) {
    VramReport r;
    r.rank = static_cast<int>(s);
    r.device = stages_[s]->Device();
    Qwen4ExpDeviceScope dev(r.device);
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
