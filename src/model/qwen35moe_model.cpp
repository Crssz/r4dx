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

#include <memory>
#include <optional>

#include "position_ids.h"  // src/vision: BuildMropePositionIds (docs/vision.md)
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/embedding.hpp"
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
  const bool tp = options.tp_world > 1;
  if (tp) {
    if (options.tp_world != 2 || options.tp_rank < 0 || options.tp_rank >= options.tp_world || options.tp_comm == nullptr ||
        options.tp_comm->World() != options.tp_world || options.tp_comm->Rank() != options.tp_rank || options.tp_device < 0) {
      throw std::invalid_argument("Qwen35MoeModel::Load: tp_world 2 needs tp_rank in [0, 2), a TpComm endpoint of the same world and rank and a tp_device");
    }
  } else if (options.tp_comm != nullptr || options.tp_rank != 0 || options.tp_device >= 0 || options.shared_embed_host) {
    throw std::invalid_argument("Qwen35MoeModel::Load: tp_comm / tp_rank / tp_device / shared_embed_host are tensor-parallel options (tp_world > 1 only)");
  }
  m.global_cfg_ = info.config;
  m.cfg_ = tp ? Qwen35MoeConfig::Shard(info.config, options.tp_world, options.tp_rank) : info.config;
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
  if (tp) {
    if (tp && options.tp_device >= visible) {
      throw std::invalid_argument("Qwen35MoeModel::Load: tp_device " + std::to_string(options.tp_device) + " is not visible");
    }
    devices = {options.tp_device, options.tp_device};
  } else if (devices.empty()) {
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
  const bool two_stage = !tp && (devices[0] != devices[1] || split > 0);  // a TP rank is always ONE stage (no hand-off)
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

  // ---- speculation: the verify window, the drafter's contract with this container ----------------------------------
  int64_t window = std::max<int64_t>(options.spec_window, 1);
  std::optional<Dflash2Config> dc;
  std::vector<int64_t> taps;  // global layer indices (layer INPUTS) the drafter reads
  if (!options.dflash_container.empty()) {
    if (options.dflash_draft_k < 1) {
      throw std::invalid_argument("Qwen35MoeModel::Load: dflash_container is set but dflash_draft_k < 1");
    }
    const DflashDraftWeights peek = DflashDraftWeights::Open(options.dflash_container);  // CPU: metadata + directory only
    dc = peek.Config();
    if (dc->layout.empty()) Fail("the dflash container has no layout in its __metadata__.dflash2 block");
    if (dc->hidden_size != cfg.hidden_size) {
      Fail("the dflash container's hidden_size " + std::to_string(dc->hidden_size) + " != the target's " +
           std::to_string(cfg.hidden_size));
    }
    if (options.dflash_small_vocab ? dc->vocab_size < cfg.vocab_size : dc->vocab_size != cfg.vocab_size) {
      Fail("the dflash container's vocab_size " + std::to_string(dc->vocab_size) + " != the target's " +
           std::to_string(cfg.vocab_size) + " (it shares the target's embedding and lm_head)");
    }
    if (options.dflash_draft_k > dc->block_size - 1) {
      throw std::invalid_argument("Qwen35MoeModel::Load: dflash_draft_k " + std::to_string(options.dflash_draft_k) +
                                  " exceeds the drafter's block_size - 1 = " + std::to_string(dc->block_size - 1));
    }
    taps = options.dflash_target_layers.empty() ? dc->target_layers : options.dflash_target_layers;
    if (taps.size() != dc->target_layers.size()) Fail("dflash_target_layers must name as many layers as the drafter reads");
    for (int64_t l : taps) {
      if (l < 0 || l >= L) {
        Fail("the dflash drafter reads the input of target layer " + std::to_string(l) + ", but this container has layers [0, " +
             std::to_string(L) + ")");
      }
    }
    window = std::max<int64_t>(window, options.dflash_draft_k + 1);
  }
  m.opt_.spec_window = window;

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
    lo.tp_world = options.tp_world;
    lo.tp_rank = options.tp_rank;
    lo.shared_embed_host = options.shared_embed_host;
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
    so.tp_comm = options.tp_comm;
    so.tp_submit_layers = options.tp_submit_layers;
    so.tp_max_inflight_units = options.tp_max_inflight_units;
    so.max_ctx = options.max_ctx;
    so.max_rows = kChunkRows;
    so.prompt_checkpoint = options.prompt_checkpoint;
    so.grouped_decode = options.grouped_decode;
    so.spec_window = window;
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

  // ---- the vision tower (docs/qwen35moe-vision.md) --------------------------------------------------------------------
  if (options.vision > 0 && tp) {
    throw std::invalid_argument("Qwen35MoeModel::Load: --vision on is not supported under --tp 2 yet (use the layer split, or --vision off)");
  }
  if (options.vision != 0 && !tp) {
    const bool has = Qwen35MoeVision::ContainerHasVision(options.container_path);
    if (options.vision > 0 && !has) {
      throw std::invalid_argument("Qwen35MoeModel::Load: --vision on, but " + options.container_path +
                                  " carries no vision.* tensors (convert with r4dx-convert --vision on)");
    }
    if (has) {
      int vdev = options.vision_device;
      if (vdev < 0) {
        if (const char* e = std::getenv("R4DX_Q35_VISION_DEVICE"); e != nullptr && *e != '\0') {
          vdev = std::atoi(e);
        } else {
          vdev = m.stages_.back()->Device();
        }
      }
      if (vdev < 0 || vdev >= visible) throw std::invalid_argument("Qwen35MoeModel::Load: vision device " + std::to_string(vdev) + " is not visible");
      m.vision_ = Qwen35MoeVision::Load(options.container_path, vdev, cfg.hidden_size);
      std::fprintf(stderr, "[r4dx::model::Qwen35MoeModel] vision tower on device %d, %.3f GiB weights (image token %lld, merge %lld)\n", vdev,
                   static_cast<double>(m.vision_->WeightBytes()) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<long long>(m.vision_->ImageTokenId()), static_cast<long long>(m.vision_->MergeSize()));
    }
  }

  // ---- the DFlash2 drafter -----------------------------------------------------------------------------------------
  if (dc) {
    const int64_t hidden = cfg.hidden_size;
    m.feat_cols_ = static_cast<int64_t>(taps.size()) * hidden;
    m.feat_host_ = core::PinnedBuffer<uint16_t>(R * static_cast<size_t>(m.feat_cols_));
    for (size_t s = 0; s < m.stages_.size(); ++s) {
      std::vector<Qwen35MoeFeatureTap> st_taps;
      for (size_t c = 0; c < taps.size(); ++c) {
        if (taps[c] >= m.stages_[s]->FirstLayer() && taps[c] < m.stages_[s]->EndLayer()) {
          st_taps.push_back({taps[c], static_cast<int64_t>(c)});
        }
      }
      m.stages_[s]->SetFeatureCapture(std::move(st_taps), m.feat_cols_);
      m.stages_[s]->SetFeatureDest(m.feat_host_.data());
    }
    Qwen35MoeDeviceScope dev(m.stages_.back()->Device());
    m.dflash_stream_ = std::make_unique<core::Stream>();
    m.dflash_arena_.Reserve(size_t{128} << 20);
    m.dflash_feat_dev_ = core::DeviceBuffer<uint16_t>(R * static_cast<size_t>(m.feat_cols_));
    DflashDraftOptions d;
    d.container_path = options.dflash_container;
    d.layout = LayoutFromName(dc->layout);
    d.max_inject_rows = 64;
    d.lm_head_vocab = cfg.VocabShardSize();  // the TARGET's lm_head rows: the whole vocabulary at TP=1, this rank's slice under TP
    if (tp) {
      // docs/tp.md 8.2: the drafter's per-rank top-16s are merged over the ranks (H7); the codebooks and token ids stay global
      d.vocab_offset = cfg.VocabShardBegin();
      d.global_vocab = cfg.vocab_size;
      d.comm = options.tp_comm;
      d.shared_codebooks = options.dflash_codebooks;
    }
    if (options.dflash_small_vocab) d.mask_token_id_override = cfg.vocab_size - 1;
    m.dflash_ = DflashDraft::Load(d);
    m.dflash_k_ = options.dflash_draft_k;
    m.dflash_stream_->Synchronize();
    std::fprintf(stderr, "[r4dx::model::Qwen35MoeModel] DFlash2 drafter loaded: %s (layout %s, block %lld, k %lld, target layer inputs [",
                 options.dflash_container.c_str(), dc->layout.c_str(), static_cast<long long>(dc->block_size),
                 static_cast<long long>(options.dflash_draft_k));
    for (size_t i = 0; i < taps.size(); ++i) std::fprintf(stderr, "%s%lld", i ? "," : "", static_cast<long long>(taps[i]));
    std::fprintf(stderr, "], verify window %lld)\n", static_cast<long long>(window));
  }
  return mp;
}

Qwen35MoeModel::~Qwen35MoeModel() {
  const char* e = std::getenv("R4DX_Q35_STATS");
  if (e != nullptr && e[0] == '1') {
    const std::string line = RoundStatsLine();
    if (!line.empty()) std::fprintf(stderr, "[r4dx::model::Qwen35MoeModel] %s\n", line.c_str());
  }
}

// ---- one chunk -------------------------------------------------------------------------------------------------------

void Qwen35MoeModel::RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen35MoeHeadRequest* head,
                              HeadOut* out, const IsolatedLayer* iso, std::vector<int32_t>* verify_out) {
  if (rows < 1 || rows > kChunkRows || (rows > 64 && rows % 64 != 0)) {
    Fail("a chunk is 1..64 rows or a multiple of 64 up to " + std::to_string(kChunkRows) + " (got " + std::to_string(rows) + ")");
  }
  if (verify_out != nullptr && (is_prefill || head != nullptr || iso != nullptr || rows > opt_.spec_window)) {
    Fail("a verify window is 1.." + std::to_string(opt_.spec_window) + " rows of a decode-kind chunk with no head request");
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
    if (!mrope_block_images_.empty()) SpliceImageRows(pos_, rows);  // PrefillMultimodal: image rows replace the placeholder embeddings
  }
  last_.embed_ms += MsSince(t_start);

  Qwen35MoeChunk chunk;
  chunk.rows = rows;
  chunk.first_pos = pos_;
  chunk.is_prefill = is_prefill;
  chunk.verify = verify_out != nullptr;
  chunk.gdn_prefill_recurrent = gdn_recurrent_;
  if (iso != nullptr) chunk.only_layer = iso->layer;
  if (mrope_active_ && iso == nullptr) {  // a conversation that has seen an image: explicit 3-axis rope rows for every chunk
    RopeRowsHost(pos_, rows, &rope_rows_scratch_);
    chunk.rope_pos3 = rope_rows_scratch_.data();
  }

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
  if (verify_out != nullptr) {
    Qwen35MoeStage& last = *stages_.back();
    *verify_out = last.HeadRows(rows);
    last_.head_ms += last.LastHeadMs();
    last_.rows = rows;
    ++last_.chunks;
    return;  // nothing is committed: pos_ moves in CommitVerifiedWindow
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
  // The drafter only ever attends to the last `sliding_window` positions, so the earlier rows of a long prompt are never injected
  // (a gap, DflashDraft::InjectFeatures -- exact, the window masks those keys for every later query).
  const int64_t inject_from =
      dflash_ ? pos_ + static_cast<int64_t>(token_ids.size()) - dflash_->Config().attention.sliding_window : 0;
  for (size_t i = 0; i < plan.size(); ++i) {
    const bool last_chunk = i + 1 == plan.size();
    const int64_t p0 = pos_;
    RunChunk(token_ids.data() + off, plan[i], /*is_prefill=*/true, (last_chunk || capture_active_) ? &req : nullptr,
             last_chunk ? &out : nullptr);
    InjectCaptured(plan[i], p0, inject_from);
    off += plan[i];
  }
  at_prefill_end_ = true;
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

// ---- vision (docs/qwen35moe-vision.md) --------------------------------------------------------------------------------------

int64_t Qwen35MoeModel::ImageTokenId() const { return vision_ ? vision_->ImageTokenId() : -1; }
int64_t Qwen35MoeModel::VisionMergeSize() const { return vision_ ? vision_->MergeSize() : 2; }

void Qwen35MoeModel::EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                                  std::vector<uint16_t>* rows, vision::VisionEncodeStats* stats) {
  if (!vision_) {
    throw std::runtime_error("Qwen35MoeModel::EncodeImages: this model has no vision tower (the container carries no vision.* tensors, "
                             "or it was loaded with --vision off, or under --tp 2)");
  }
  vision_->Encode(pixel_values, total_patches, grids, rows, stats);
}

// Model::RopePositionsHost's rule: inside the block the PrefillMultimodal call feeds, the block's own explicit rows; everywhere else a
// token at sequence index s ropes at s + mrope_delta_ on all three axes.
void Qwen35MoeModel::RopeRowsHost(int64_t start, int64_t rows, std::vector<int32_t>* out3) const {
  out3->assign(static_cast<size_t>(3 * rows), 0);
  const int64_t n_block = static_cast<int64_t>(mrope_block_.size() / 3);
  for (int64_t t = 0; t < rows; ++t) {
    const int64_t s = start + t;
    const int64_t in_block = s - mrope_block_base_;
    if (n_block > 0 && in_block >= 0 && in_block < n_block) {
      for (int axis = 0; axis < 3; ++axis) {
        (*out3)[static_cast<size_t>(axis * rows + t)] = mrope_block_[static_cast<size_t>(axis * n_block + in_block)];
      }
    } else {
      const int32_t p = static_cast<int32_t>(s + mrope_delta_);
      (*out3)[static_cast<size_t>(t)] = p;
      (*out3)[static_cast<size_t>(rows + t)] = p;
      (*out3)[static_cast<size_t>(2 * rows + t)] = p;
    }
  }
}

void Qwen35MoeModel::SpliceImageRows(int64_t start, int64_t rows) {
  const int64_t hidden = cfg_.hidden_size;
  for (const ImageSpan& sp : mrope_block_images_) {
    const int64_t s0 = mrope_block_base_ + sp.offset;  // absolute sequence index of the span's first row
    const int64_t lo = std::max(s0, start);
    const int64_t hi = std::min(s0 + sp.tokens, start + rows);
    if (lo >= hi) continue;
    std::copy_n(sp.embeds + (lo - s0) * hidden, (hi - lo) * hidden, embed_rows_host_.data() + (lo - start) * hidden);
  }
}

std::vector<float> Qwen35MoeModel::PrefillMultimodal(const std::vector<int32_t>& token_ids, const std::vector<ImageSpan>& images) {
  if (token_ids.empty()) throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: token_ids is empty");
  // No images: a plain Prefill, which already ropes through the conversation's mrope delta once an image has been seen.
  if (images.empty()) return Prefill(token_ids);
  if (!HasVision()) {
    throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: image spans supplied but this model has no vision tower loaded "
                             "(--vision off, --tp 2, or a container with no vision.* tensors)");
  }
  const int64_t seq_len = static_cast<int64_t>(token_ids.size());
  const int merge_size = static_cast<int>(VisionMergeSize());
  const int32_t image_token_id = static_cast<int32_t>(ImageTokenId());
  // The mm_token_type_ids the position walk needs come from the SPANS (they are the contract: grid + rows); the tokens under each span
  // must really be the placeholder id, which catches a caller whose offsets drifted from its own rendered prompt.
  std::vector<uint8_t> mm_ids(static_cast<size_t>(seq_len), vision::kMmTokenTypeText);
  std::vector<vision::GridThw> grids;
  int64_t prev_end = 0;
  for (const ImageSpan& sp : images) {
    if (sp.offset < prev_end || sp.tokens <= 0 || sp.offset + sp.tokens > seq_len) {
      throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: image spans must be sorted, non-overlapping and inside token_ids; got offset=" +
                               std::to_string(sp.offset) + " tokens=" + std::to_string(sp.tokens) + " against a " +
                               std::to_string(seq_len) + "-token prompt");
    }
    if (sp.embeds == nullptr) {
      throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: image span at offset " + std::to_string(sp.offset) + " has no embeddings");
    }
    if (!sp.embeds_on_host) {
      throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: image rows must be HOST rows (the embedding gather is a host gather)");
    }
    if (sp.grid.MergedTokenCount(merge_size) != sp.tokens) {
      throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: image span at offset " + std::to_string(sp.offset) + " claims " +
                               std::to_string(sp.tokens) + " tokens but its grid merges to " +
                               std::to_string(sp.grid.MergedTokenCount(merge_size)));
    }
    for (int64_t i = sp.offset; i < sp.offset + sp.tokens; ++i) {
      if (token_ids[static_cast<size_t>(i)] != image_token_id) {
        throw std::runtime_error("Qwen35MoeModel::PrefillMultimodal: token at index " + std::to_string(i) +
                                 " is inside an image span but is not the image placeholder id " + std::to_string(image_token_id) +
                                 " (got " + std::to_string(token_ids[static_cast<size_t>(i)]) + ")");
      }
      mm_ids[static_cast<size_t>(i)] = vision::kMmTokenTypeImage;
    }
    grids.push_back(sp.grid);
    prev_end = sp.offset + sp.tokens;
  }
  if (pos_ + seq_len > opt_.max_ctx) {
    Fail("position " + std::to_string(pos_ + seq_len) + " exceeds max_ctx " + std::to_string(opt_.max_ctx));
  }
  // This block's 3-axis rows, and the delta every later token ropes at.
  const vision::MropePositions mp =
      vision::BuildMropePositionIds(mm_ids, grids, merge_size, /*seq_start=*/pos_, /*mrope_start=*/pos_ + mrope_delta_);
  mrope_block_ = mp.position_ids;
  mrope_block_base_ = pos_;
  mrope_block_images_ = images;
  mrope_active_ = true;
  // Cleared on EVERY exit: a stale block table would rope a later call's tokens from this call's rows.
  struct BlockGuard {
    Qwen35MoeModel* m;
    ~BlockGuard() {
      m->mrope_block_.clear();
      m->mrope_block_images_.clear();
      m->mrope_block_base_ = 0;
    }
  } guard{this};
  std::vector<float> logits = Prefill(token_ids);
  // Only now: until the last chunk landed the block's explicit rows were what every chunk read.
  mrope_delta_ = mp.mrope_position_delta;
  if (dflash_) dflash_->SetRopeDelta(mrope_delta_);  // the drafter ropes its blocks on the temporal axis at index + delta
  return logits;
}

std::vector<float> Qwen35MoeModel::DecodeStep(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kLogits, 0.0f};
  const int64_t p0 = pos_;
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  InjectCaptured(1, p0);
  last_.total_ms = MsSince(t0);
  return std::move(out.logits);
}

int32_t Qwen35MoeModel::DecodeStepGreedy(int32_t token_id) {
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  HeadOut out;
  const Qwen35MoeHeadRequest req{Qwen35MoeHeadRequest::Kind::kArgmax, 0.0f};
  const int64_t p0 = pos_;
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  InjectCaptured(1, p0);
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
  const int64_t p0 = pos_;
  RunChunk(&token_id, 1, /*is_prefill=*/false, &req, &out);
  InjectCaptured(1, p0);
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
  verified_rows_ = 0;
  mrope_active_ = false;
  mrope_delta_ = 0;
  mrope_block_.clear();
  mrope_block_images_.clear();
  mrope_block_base_ = 0;
  if (dflash_) {
    dflash_->Reset();  // host-only counters: the ring's bytes are overwritten before they are read again
    dflash_->SetRopeDelta(0);
  }
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
  ckpt_mrope_active_ = mrope_active_;
  ckpt_mrope_delta_ = mrope_delta_;
}

void Qwen35MoeModel::RestoreCheckpoint() {
  if (!prompt_checkpoint_) {
    throw std::runtime_error("Qwen35MoeModel::RestoreCheckpoint: this model was loaded without ModelOptions::prompt_checkpoint");
  }
  if (ckpt_pos_ < 0) throw std::runtime_error("Qwen35MoeModel::RestoreCheckpoint: no checkpoint since Load()/Reset()");
  for (auto& st : stages_) st->RestoreCheckpoint(ckpt_pos_);
  pos_ = ckpt_pos_;
  mrope_active_ = ckpt_mrope_active_;
  mrope_delta_ = ckpt_mrope_delta_;
  if (dflash_) dflash_->SetRopeDelta(mrope_delta_);
  at_prefill_end_ = true;  // the restored state IS a post-prefill state
  verified_rows_ = 0;
  if (dflash_ && dflash_->InjectedCount() > pos_) dflash_->Rewind(pos_);
}

// ---- speculative decoding ---------------------------------------------------------------------------------------------

std::vector<int32_t> Qwen35MoeModel::VerifyWindow(const std::vector<int32_t>& candidates) {
  if (opt_.spec_window <= 1) Fail("VerifyWindow: the model was loaded without a speculative window (Qwen35MoeModelOptions::spec_window)");
  const int64_t rows = static_cast<int64_t>(candidates.size());
  if (rows < 1 || rows > opt_.spec_window) {
    Fail("VerifyWindow: " + std::to_string(rows) + " candidates, the window is 1.." + std::to_string(opt_.spec_window));
  }
  const auto t0 = Clock::now();
  last_ = Timings{};
  at_prefill_end_ = false;
  std::vector<int32_t> argmax;
  RunChunk(candidates.data(), rows, /*is_prefill=*/false, nullptr, nullptr, nullptr, &argmax);
  verified_rows_ = rows;
  last_.total_ms = MsSince(t0);
  return argmax;
}

std::vector<float> Qwen35MoeModel::LastVerifyLogits() const {
  if (verified_rows_ < 1) Fail("LastVerifyLogits: no uncommitted VerifyWindow");
  return stages_.back()->LastHeadRowsLogits(verified_rows_);
}

void Qwen35MoeModel::CommitVerifiedWindow(int64_t n) {
  if (verified_rows_ < 1 || n < 1 || n > verified_rows_) {
    Fail("CommitVerifiedWindow: n = " + std::to_string(n) + " is outside [1, " + std::to_string(verified_rows_) +
         "] (the rows of the last VerifyWindow)");
  }
  for (auto& st : stages_) st->CommitVerified(n);
  pos_ += n;
  verified_rows_ = 0;
  at_prefill_end_ = false;
}

int64_t Qwen35MoeModel::DflashInjectedCount() const {
  if (!dflash_) Fail("DflashInjectedCount: no drafter is loaded");
  return dflash_->InjectedCount();
}

void Qwen35MoeModel::SetDflashInjectionEnabled(bool enabled) {
  if (enabled && !dflash_) Fail("SetDflashInjectionEnabled(true): no drafter is loaded");
  dflash_injection_ = enabled;
  // No drafter feeding: the stages stop capturing too (8 small D2H copies per step that a plain decode step can do without).
  if (dflash_) {
    for (auto& st : stages_) st->SetFeatureDest(enabled ? feat_host_.data() : nullptr);
  }
}

void Qwen35MoeModel::InjectCaptured(int64_t rows, int64_t start_pos, int64_t min_pos) {
  if (!dflash_ || !dflash_injection_ || rows <= 0) return;
  const int64_t skip = std::clamp<int64_t>(min_pos - start_pos, 0, rows);
  if (skip >= rows) return;
  Qwen35MoeDeviceScope dev(stages_.back()->Device());
  core::Stream& s = *dflash_stream_;
  // <= 64 rows per call (DflashDraftOptions::max_inject_rows); each slice is waited for because InjectFeatures stages its rope
  // positions in ONE pinned array.
  for (int64_t r0 = skip; r0 < rows; r0 += 64) {
    const int64_t n = std::min<int64_t>(64, rows - r0);
    R4DX_HIP_CHECK(hipMemcpyAsync(dflash_feat_dev_.data(), feat_host_.data() + r0 * feat_cols_,
                                  static_cast<size_t>(n * feat_cols_) * sizeof(uint16_t), hipMemcpyHostToDevice, s.get()));
    dflash_->InjectFeatures(s, dflash_arena_, dflash_feat_dev_.data(), n, start_pos + r0);
    s.Synchronize();
  }
  dflash_arena_.Reset();
}

std::vector<int32_t> Qwen35MoeModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                            int64_t* walk_len_out) {
  if (!dflash_) Fail("DecodeStepDflashGreedy: no drafter is loaded (Qwen35MoeModelOptions::dflash_container)");
  if (k < 0 || k > dflash_k_) Fail("DecodeStepDflashGreedy: k must be in [0, " + std::to_string(dflash_k_) + "]");
  if (!dflash_injection_) {
    Fail("DecodeStepDflashGreedy: drafter injection is disabled (SetDflashInjectionEnabled(false)); its ring frontier lags the position");
  }
  if (dflash_->InjectedCount() != pos_) {
    Fail("DecodeStepDflashGreedy: drafter frontier (" + std::to_string(dflash_->InjectedCount()) + ") != position (" +
         std::to_string(pos_) + ")");
  }
  if (pos_ + 1 > opt_.max_ctx) Fail("DecodeStepDflashGreedy: position " + std::to_string(pos_ + 1) + " exceeds max_ctx");
  k = std::min<int64_t>(k, opt_.max_ctx - pos_ - 1);  // the window is k + 1 rows; k = 0 still verifies the anchor
  const auto t0 = Clock::now();

  DflashDraftResult draft;
  {
    Qwen35MoeDeviceScope dev(stages_.back()->Device());
    const uint16_t* table = containers_.front()->EmbedTokensHost();
    const int64_t hidden = cfg_.hidden_size, vocab = cfg_.vocab_size;
    auto staging = std::make_shared<std::vector<uint16_t>>();
    const DflashEmbeddingProvider embed = [table, hidden, vocab, staging](core::Stream& stream, const int32_t* ids_host,
                                                                          const int32_t*, int64_t n, uint16_t* out_dev) {
      staging->resize(static_cast<size_t>(n * hidden));
      const std::vector<int32_t> ids(ids_host, ids_host + n);
      kernels::EmbeddingGatherHost(table, vocab, hidden, ids, staging->data());
      R4DX_HIP_CHECK(hipMemcpyAsync(out_dev, staging->data(), staging->size() * sizeof(uint16_t), hipMemcpyHostToDevice,
                                    stream.get()));
    };
    const DflashLmHeadProvider lm_head = MakeLmHeadProviderFromLinear(&containers_.back()->LmHead());
    draft = dflash_->DraftRound(*dflash_stream_, dflash_arena_, token_id, k, p_min, n_min, embed, lm_head);
    dflash_arena_.Reset();
  }
  last_.draft_ms = MsSince(t0);
  if (draft_override_) {
    draft.tokens = draft_override_(token_id, k);
    if (static_cast<int64_t>(draft.tokens.size()) > k) draft.tokens.resize(static_cast<size_t>(k));
    draft.walk_len = static_cast<int64_t>(draft.tokens.size());
  }
  if (walk_len_out != nullptr) *walk_len_out = draft.walk_len;
  last_drafts_ = draft.tokens;

  std::vector<int32_t> cands;
  cands.reserve(draft.tokens.size() + 1);
  cands.push_back(token_id);
  cands.insert(cands.end(), draft.tokens.begin(), draft.tokens.end());
  const double draft_ms = last_.draft_ms;
  const std::vector<int32_t> argmax = VerifyWindow(cands);
  last_.draft_ms = draft_ms;
  round_stats_.verify_ms += last_.total_ms;
  round_stats_.stage_ms[0] += last_.stage_ms[0];
  round_stats_.stage_ms[1] += last_.stage_ms[1];
  round_stats_.handoff_ms += last_.handoff_ms;
  round_stats_.head_ms += last_.head_ms;
  round_stats_.draft_ms += draft_ms;

  // Greedy acceptance: draft i (candidates[i + 1]) is confirmed iff the target's argmax after candidates[i] equals it.
  int64_t accepted = 0;
  while (accepted < static_cast<int64_t>(draft.tokens.size()) &&
         argmax[static_cast<size_t>(accepted)] == draft.tokens[static_cast<size_t>(accepted)]) {
    ++accepted;
  }
  const int64_t committed = accepted + 1;  // the anchor + the accepted drafts
  std::vector<int32_t> result(argmax.begin(), argmax.begin() + committed);  // accepted drafts, then the bonus token

  // The committed rows' own captured features (rows 0 .. committed-1 of the window) go into the drafter's ring at its frontier,
  // THEN the model commits. The rejected rows' features are never injected.
  const auto t1 = Clock::now();
  InjectCaptured(committed, dflash_->InjectedCount());
  last_.inject_ms = MsSince(t1);
  CommitVerifiedWindow(committed);
  last_.total_ms = MsSince(t0);
  ++round_stats_.rounds;
  round_stats_.tokens += committed;
  round_stats_.inject_ms += last_.inject_ms;
  round_stats_.total_ms += last_.total_ms;
  return result;
}

std::string Qwen35MoeModel::RoundStatsLine() const {
  const RoundStats& r = round_stats_;
  if (r.rounds == 0) return "";
  const double n = static_cast<double>(r.rounds);
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "dflash rounds: %lld, %.2f tokens/round; per round %.1f ms = draft %.1f + verify %.1f (stage0 %.1f, stage1 %.1f, handoff %.1f, head %.1f) + inject %.1f",
                static_cast<long long>(r.rounds), static_cast<double>(r.tokens) / n, r.total_ms / n, r.draft_ms / n, r.verify_ms / n,
                r.stage_ms[0] / n, r.stage_ms[1] / n, r.handoff_ms / n, r.head_ms / n, r.inject_ms / n);
  return buf;
}

void Qwen35MoeModel::TpWarmup() {
  if (!IsTp()) throw std::logic_error("Qwen35MoeModel::TpWarmup: only a tensor-parallel rank (tp_world > 1) warms up");
  if (opt_.max_ctx < WarmupPositions()) {
    throw std::invalid_argument("Qwen35MoeModel::TpWarmup: max_ctx " + std::to_string(opt_.max_ctx) + " < " + std::to_string(WarmupPositions()) +
                                " warm-up positions");
  }
  const auto ids = [&](int64_t n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = static_cast<int32_t>((i * 7919 + 13) % cfg_.vocab_size);
    return v;
  };
  Reset();
  (void)Prefill(ids(kChunkRows));  // the M = 256 GEMMs, the sliced all-reduces, every expert shape of the host-routed MoE
  Reset();
  (void)Prefill(ids(64));
  (void)DecodeStepGreedy(0);
  kernels::SampleParams sp;  // temperature 1: the summary path and its merge
  std::mt19937_64 rng(0x7e57);
  (void)DecodeStepSampled(1, sp, rng);
  (void)DecodeStep(2);  // the full-row gather
  if (DflashEnabled()) {
    // the drafter's first round, the M-row verify kernels and the speculative-window GDN path, so none of their first-use launches lands in a request
    Reset();
    (void)Prefill(ids(64));
    for (int round = 0; round < 3; ++round) (void)DecodeStepDflashGreedy(round, dflash_k_, 0.0f, 0);
  }
  Reset();
}

void Qwen35MoeModel::EnableStepProfile(bool on) {
  for (auto& st : stages_) st->EnableStepProfile(on);
}

Qwen35MoeModel::StepGpu Qwen35MoeModel::LastStepGpu() const {
  StepGpu g;
  for (size_t s = 0; s < stages_.size(); ++s) g.layers_ms[std::min<size_t>(s, 1)] += stages_[s]->LastLayersGpuMs();
  g.head_ms = stages_.back()->LastHeadGpuMs();
  return g;
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
