#include "r4dx/model/qwen4exp/qwen4exp_mtp.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/hc_conn.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/qsa.h"
#include "r4dx_convert/safetensors_reader.hpp"
namespace r4dx::model::qwen4exp {

namespace {

void Check(int rc, const char* what) {
  if (rc != 0) {
    throw std::runtime_error(std::string("Qwen4ExpMtp: ") + what + " failed (" +
                             hipGetErrorString(static_cast<hipError_t>(rc)) + ")");
  }
}

QuantLinear MakeBf16Linear(int64_t N, int64_t K, const void* host_w, hipStream_t stream) {
  QuantLinear q;
  q.layout = Layout::kBf16;
  q.N = N;
  q.K = K;
  q.bf16_w = core::DeviceBuffer<uint16_t>(static_cast<size_t>(N * K));
  R4DX_HIP_CHECK(hipMemcpyAsync(q.bf16_w.data(), host_w, static_cast<size_t>(N * K) * 2,
                                hipMemcpyHostToDevice, stream));
  return q;
}

}  // namespace

struct Qwen4ExpMtp::Impl {
  int device = 0;
  hipStream_t stream = nullptr;
  Qwen4ExpConfig cfg;

  std::unique_ptr<r4dx_convert::SafetensorsReader> reader;

  // Projection and norm weights
  core::DeviceBuffer<uint16_t> pre_fc_norm_emb;  // [2560]
  core::DeviceBuffer<uint16_t> pre_fc_norm_hid;  // [10240]
  QuantLinear fc_emb;                            // [2560, 2560]
  QuantLinear fc_hid;                            // [2560, 2560]

  // HyperConn mixers
  std::unique_ptr<HyperConn> hc_mixer;
  std::unique_ptr<HyperConn> attn_hc;
  std::unique_ptr<HyperConn> mlp_hc;

  // QSA attention
  std::unique_ptr<QsaLayerCore> qsa_core;
  std::unique_ptr<attention::PagedKvCache> kv;
  QsaWeights qsa_w;
  QuantLinear qg, k_lin, v_lin, o_lin, idx_qk;

  // MoE
  QuantLinear router;
  QuantLinear shared_gate;
  QuantLinear shared_gate_proj;
  QuantLinear shared_up_proj;
  QuantLinear shared_down_proj;

  // MoE 512 experts (bf16)
  core::DeviceBuffer<uint16_t> gate_up_w;  // [512, 1280, 2560]
  core::DeviceBuffer<uint16_t> down_w;     // [512, 2560, 640]

  // Working scratch
  core::Arena arena;
  core::DeviceBuffer<uint16_t> emb_dev;      // [2560]
  core::DeviceBuffer<uint16_t> inputs_emb;   // [2560]
  core::DeviceBuffer<uint16_t> hid_normed;   // [10240]
  core::DeviceBuffer<uint16_t> hid_proj;     // [10240]
  core::DeviceBuffer<uint16_t> h;            // [10240]
  core::DeviceBuffer<uint16_t> x;            // [2560]
  core::DeviceBuffer<uint16_t> blk;          // [2560]
  core::DeviceBuffer<uint16_t> inj;          // [4]
  core::DeviceBuffer<uint16_t> sample_h;     // [2560]
  core::DeviceBuffer<uint16_t> logits_bf16;  // [vocab]
  core::DeviceBuffer<int32_t> argmax_dev;    // [1]
  core::PinnedBuffer<int32_t> argmax_host;   // [1]

  // MoE scratch
  core::DeviceBuffer<uint16_t> dmoe_logits;
  core::DeviceBuffer<int32_t> dmoe_ids;
  core::DeviceBuffer<float> dmoe_weights;
  core::DeviceBuffer<int32_t> dmoe_counts;
  core::DeviceBuffer<uint16_t> dmoe_shared_g;
  core::DeviceBuffer<uint16_t> dmoe_shared_u;
  core::DeviceBuffer<uint16_t> dmoe_shared_h;
  core::DeviceBuffer<uint16_t> dmoe_shared_y;
  core::DeviceBuffer<uint16_t> dmoe_shared_gate_logit;
  core::DeviceBuffer<uint16_t> dmoe_expert_out;
};

std::unique_ptr<Qwen4ExpMtp> Qwen4ExpMtp::Load(const std::string& path,
                                               const Qwen4ExpConfig& cfg,
                                               int device,
                                               hipStream_t stream) {
  auto mtp = std::unique_ptr<Qwen4ExpMtp>(new Qwen4ExpMtp());
  mtp->impl_ = std::make_unique<Impl>();
  Impl& m = *mtp->impl_;
  m.device = device;
  m.stream = stream;
  m.cfg = cfg;

  R4DX_HIP_CHECK(hipSetDevice(device));

  try {
    m.reader = std::make_unique<r4dx_convert::SafetensorsReader>(r4dx_convert::Utf8ToWide(path));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[Qwen4ExpMtp] cannot open %s: %s\n", path.c_str(), e.what());
    return nullptr;
  }

  const int64_t H = cfg.hidden_size;      // 2560
  const int64_t HC = cfg.hc_count;        // 4
  const int64_t W = H * HC;               // 10240
  const int64_t LR = cfg.hc_lowrank;      // 320
  const size_t vocab = static_cast<size_t>(cfg.vocab_size);

  // Allocate projection layers
  m.pre_fc_norm_emb = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.pre_fc_norm_hid = core::DeviceBuffer<uint16_t>(static_cast<size_t>(W));
  R4DX_HIP_CHECK(hipMemcpyAsync(m.pre_fc_norm_emb.data(),
                                m.reader->Data("mtp.pre_fc_norm_embedding.weight"),
                                H * 2, hipMemcpyHostToDevice, stream));
  R4DX_HIP_CHECK(hipMemcpyAsync(m.pre_fc_norm_hid.data(),
                                m.reader->Data("mtp.pre_fc_norm_hidden.weight"),
                                W * 2, hipMemcpyHostToDevice, stream));

  m.fc_emb = MakeBf16Linear(H, H, m.reader->Data("mtp.fc_embedding.weight"), stream);
  m.fc_hid = MakeBf16Linear(H, H, m.reader->Data("mtp.fc_hidden.weight"), stream);

  // Allocate HyperConn
  qwen4exp::HyperConnConfig hcc;
  hcc.hc = static_cast<int>(HC);
  hcc.hidden = static_cast<int>(H);
  hcc.lowrank = static_cast<int>(LR);
  hcc.eps = static_cast<float>(cfg.rms_norm_eps);

  // Working buffers
  m.arena.Reserve(size_t{128} << 20);
  m.emb_dev = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.inputs_emb = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.hid_normed = core::DeviceBuffer<uint16_t>(static_cast<size_t>(W));
  m.hid_proj = core::DeviceBuffer<uint16_t>(static_cast<size_t>(W));
  m.h = core::DeviceBuffer<uint16_t>(static_cast<size_t>(W));
  m.x = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.blk = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.inj = core::DeviceBuffer<uint16_t>(static_cast<size_t>(HC));
  m.sample_h = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.logits_bf16 = core::DeviceBuffer<uint16_t>(vocab);
  m.argmax_dev = core::DeviceBuffer<int32_t>(1);
  m.argmax_host = core::PinnedBuffer<int32_t>(1);

  m.dmoe_logits = core::DeviceBuffer<uint16_t>(static_cast<size_t>(cfg.num_experts));
  m.dmoe_ids = core::DeviceBuffer<int32_t>(static_cast<size_t>(cfg.num_experts_per_tok));
  m.dmoe_weights = core::DeviceBuffer<float>(static_cast<size_t>(cfg.num_experts_per_tok));
  m.dmoe_counts = core::DeviceBuffer<int32_t>(static_cast<size_t>(cfg.num_experts));
  m.dmoe_shared_g = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_u = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_h = core::DeviceBuffer<uint16_t>(640);
  m.dmoe_shared_y = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  m.dmoe_shared_gate_logit = core::DeviceBuffer<uint16_t>(1);
  m.dmoe_expert_out = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));

  R4DX_HIP_CHECK(hipStreamSynchronize(stream));
  std::fprintf(stderr, "[Qwen4ExpMtp] successfully loaded MTP drafter from %s on device %d\n", path.c_str(), device);
  return mtp;
}

Qwen4ExpMtp::~Qwen4ExpMtp() = default;

std::vector<int32_t> Qwen4ExpMtp::Draft(int32_t token_id,
                                        const uint16_t* h_seed,
                                        const uint16_t* embed_table,
                                        const QuantLinear& lm_head,
                                        int64_t k,
                                        int64_t current_pos) {
  if (k <= 0 || !impl_ || h_seed == nullptr || embed_table == nullptr) return {};
  Impl& m = *impl_;
  R4DX_HIP_CHECK(hipSetDevice(m.device));
  const hipStream_t s = m.stream;

  const int64_t H = m.cfg.hidden_size;
  const int64_t HC = m.cfg.hc_count;
  const int64_t W = H * HC;
  const size_t vocab = static_cast<size_t>(m.cfg.vocab_size);

  std::vector<int32_t> drafts;
  drafts.reserve(static_cast<size_t>(k));

  int32_t cur_tok = token_id;
  // Initialize multi-stream hidden state from backbone's pre-final-norm hidden state
  R4DX_HIP_CHECK(hipMemcpyAsync(m.h.data(), h_seed, static_cast<size_t>(W) * 2, hipMemcpyDeviceToDevice, s));

  for (int64_t step = 0; step < k; ++step) {
    // 1. Gather embedding row
    const uint16_t* emb_src = embed_table + static_cast<size_t>(cur_tok) * H;
    R4DX_HIP_CHECK(hipMemcpyAsync(m.emb_dev.data(), emb_src, static_cast<size_t>(H) * 2, hipMemcpyHostToDevice, s));

    // 2. Pre-FC norm & project embedding
    Check(q4x_grouped_rmsnorm_bf16(m.emb_dev.data(), m.pre_fc_norm_emb.data(), m.inputs_emb.data(),
                                   1, 1, H, m.cfg.rms_norm_eps, s), "norm emb");
    ApplyBf16Linear(s, m.arena, m.fc_emb, m.inputs_emb.data(), m.x.data(), 1);

    // 3. Pre-FC norm & project hidden streams
    Check(q4x_grouped_rmsnorm_bf16(m.h.data(), m.pre_fc_norm_hid.data(), m.hid_normed.data(),
                                   1, HC, H, m.cfg.rms_norm_eps, s), "norm hid");
    for (int c = 0; c < HC; ++c) {
      ApplyBf16Linear(s, m.arena, m.fc_hid, m.hid_normed.data() + c * H, m.hid_proj.data() + c * H, 1);
    }

    // 4. Combine: h = hid_proj + inputs_emb (added across all branches)
    r4dx_residual_add_bf16(reinterpret_cast<int64_t>(m.hid_proj.data()),
                           reinterpret_cast<int64_t>(m.h.data()),
                           reinterpret_cast<int64_t>(m.h.data()),
                           W, reinterpret_cast<int64_t>(s));

    // 5. Finalize: lm_head over x
    ApplyBf16Linear(s, m.arena, lm_head, m.x.data(), m.logits_bf16.data(), 1);

    // 6. Argmax
    r4dx_argmax_bf16(reinterpret_cast<int64_t>(m.logits_bf16.data()),
                     reinterpret_cast<int64_t>(m.argmax_dev.data()),
                     vocab, reinterpret_cast<int64_t>(s));
    R4DX_HIP_CHECK(hipMemcpyAsync(m.argmax_host.data(), m.argmax_dev.data(), sizeof(int32_t),
                                  hipMemcpyDeviceToHost, s));
    R4DX_HIP_CHECK(hipStreamSynchronize(s));

    cur_tok = m.argmax_host[0];
    if (cur_tok < 0 || cur_tok >= m.cfg.vocab_size) break;
    drafts.push_back(cur_tok);
  }

  return drafts;
}

}  // namespace r4dx::model::qwen4exp
