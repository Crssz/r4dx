// tests/model/test_model_q4exp_loader.cpp -- Qwen4ExpContainer (docs/qwen4exp-plan.md 4.11, task M1-3). Links r4dx_model.
//
//   1. CPU, always: ChooseSplit on synthetic numbers (balance, head / tail, cap, ties, refusals) and on the real
//      model's estimate (uniform 3.33 bpw experts: KB 3 gate/up, KB 4 down, 26 GiB cap -> k = 24); the
//      Qwen4ExpConfig::ToModelConfig() view of the real config.
//   2. GPU, when the tiny converter container exists (R4DX_Q4EXP_TINY_CONTAINER, default
//      E:\models\r4dx\tiny-q4exp\container-bf16.r4dx; otherwise the test ends with exit code 77 after part 1):
//      Inspect, then Load of layers [0, 2) and [2, 4) on the test device: shapes, device bytes of expert linears equal
//      the file bytes (downloaded and memcmp'd), expert views alias their class buffer, VRAM report == the bytes
//      Inspect predicted == the DeviceBuffer counter's growth, loading twice and destroying frees everything.
//   3. When R4DX_Q4EXP_TINY_TRELLIS (default E:\models\r4dx\tiny-q4exp\container-trellis.r4dx) exists: the same on a
//      trellis container -- per-layer KB, ticket slices (one buffer, disjoint, N/128 words each), widened sign
//      vectors, and a copy with one expert's KB flipped is refused for non-uniformity.
// The device is 0 when HIP_VISIBLE_DEVICES is set (ctest sets it to 1), else device 1 when there are two.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "qwen4exp_container.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace fs = std::filesystem;
using namespace r4dx::model;
using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    const bool ok = substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
    if (!ok) std::fprintf(stderr, "  (threw: %s)\n", e.what());
    return ok;
  }
  return false;
}

constexpr int64_t GiB = 1LL << 30;

// ---- part 1: ChooseSplit ----------------------------------------------------------------------------------------

void TestChooseSplitSynthetic() {
  // Uniform layers: the boundary is the middle.
  {
    const Qwen4ExpSplit s = ChooseSplit(std::vector<int64_t>(10, 100), 0, 0, 1000);
    Check(s.k == 5 && s.dev0_bytes == 500 && s.dev1_bytes == 500, "uniform 10x100 -> k 5, 500 / 500");
  }
  // The tail (lm_head) lives on device 1, so device 0 takes more layers.
  {
    const Qwen4ExpSplit s = ChooseSplit(std::vector<int64_t>(10, 100), 0, 200, 2000);
    Check(s.k == 6 && s.dev0_bytes == 600 && s.dev1_bytes == 600, "tail 200 -> k 6, 600 / 600");
  }
  // A head on device 0 moves the boundary the other way.
  {
    const Qwen4ExpSplit s = ChooseSplit(std::vector<int64_t>(10, 100), 300, 0, 2000);
    Check(s.k == 3 && s.dev0_bytes == 600 && s.dev1_bytes == 700, "head 300 -> k 3 (tie with 4 -> lower k), 600 / 700");
  }
  // A tie takes the lower boundary (an odd total: 7 layers of 100).
  {
    const Qwen4ExpSplit s = ChooseSplit(std::vector<int64_t>(7, 100), 0, 0, 1000);
    Check(s.k == 3 && s.dev0_bytes == 300 && s.dev1_bytes == 400, "7x100 -> k 3 (tie -> lower k)");
  }
  // Uneven layers: the minimum |dev0 - dev1| wins, not the middle layer index.
  {
    const Qwen4ExpSplit s = ChooseSplit({500, 50, 50, 50, 50, 50}, 0, 0, 1000);
    Check(s.k == 1 && s.dev0_bytes == 500 && s.dev1_bytes == 250, "{500,50x5} -> k 1");
  }
  // The cap rules out the balanced boundary: both cards must stay within it, and it picks the best feasible k.
  {
    // total 1000 over 10 layers, tail 0: k in [4, 6] keeps both <= 600; the balanced k = 5 is feasible.
    const Qwen4ExpSplit s = ChooseSplit(std::vector<int64_t>(10, 100), 0, 0, 600);
    Check(s.k == 5, "cap 600 keeps k 5");
    // layers {300, 100 x 7}: balance wants k = 2 (400 / 600 -> |200|), k = 3 -> (500 / 500) wait: recompute below.
    const std::vector<int64_t> v = {300, 100, 100, 100, 100, 100, 100, 100};  // total 1000
    const Qwen4ExpSplit u = ChooseSplit(v, 0, 0, 1000);
    Check(u.k == 3 && u.dev0_bytes == 500 && u.dev1_bytes == 500, "uneven -> k 3, 500 / 500");
    // A cap of 520 permits only boundaries with both sides <= 520: 480 <= dev0 <= 520 -> k 3 only.
    const Qwen4ExpSplit w = ChooseSplit(v, 0, 0, 520);
    Check(w.k == 3, "tight cap -> the one feasible k");
    // A cap below the balanced half leaves nothing feasible.
    Check(Throws([&] { ChooseSplit(v, 0, 0, 480); }, "no layer boundary keeps both cards"), "cap below half -> refused");
  }
  Check(Throws([] { ChooseSplit({100}, 0, 0, 1000); }, "at least two layers"), "one layer -> invalid_argument");
  Check(Throws([] { ChooseSplit({100, 100}, -1, 0, 1000); }, "must be >= 0"), "negative head -> invalid_argument");
  Check(Throws([] { ChooseSplit({100, -5}, 0, 0, 1000); }, "negative bytes"), "negative layer -> invalid_argument");
  Check(Throws([] { ChooseSplit({100, 100}, 0, 0, 0); }, "cap_bytes > 0"), "cap 0 -> invalid_argument");
}

// The real model, uniform 3.33 bpw experts (KB 3 gate/up, KB 4 down) and every tensor class at its on-device size
// (sign vectors widened to fp32): the estimate plan 3.1 gives (~50.8 GB) and the layer boundary it implies.
void TestChooseSplitRealEstimate() {
  const int64_t H = 2560, S = 4 * H, lr = 320, E = 512, I = 640, Si = 640, V = 248320, L = 48;
  const int64_t bits_gu = 3, bits_dn = 4;
  const auto trellis = [](int64_t N, int64_t K, int64_t bits, int64_t parts) {
    return N * K * bits / 8 + parts * K * 4 + N * 4 + N / 128 * 4;  // words + suh + svh (fp32) + tickets
  };
  const int64_t experts = E * (trellis(2 * I, H, bits_gu, 2) + trellis(H, I, bits_dn, 1));
  const int64_t hc = 2 * (S * 2 + lr * S * 2 + S * lr * 2 + 4 * S * 2);
  const int64_t dense = E * H * 2 + H * 2 + 2 * Si * H * 2 + H * Si * 2;
  const int64_t key_dim = 16 * 128, value_dim = 48 * 128, conv_dim = 2 * key_dim + value_dim;
  const int64_t gdn = trellis(conv_dim, H, 4, 1) + trellis(value_dim, H, 4, 1) + trellis(H, value_dim, 4, 1) +
                      2 * 48 * H * 2 + conv_dim * 4 * 2 + 48 * 4 * 2 + 128 * 4;
  const int64_t qdim = 24 * 256, kvdim = 2 * 256;
  const int64_t qsa = trellis(2 * qdim, H, 4, 1) + 2 * trellis(kvdim, H, 4, 1) + trellis(H, qdim, 4, 1) + 2 * 256 * 2 +
                      2 * 2 * 4 + 640 * H * 2 + 2 * 128 * 2;
  const int64_t ple = S * H * 2 + H * H * 2 + 3 * S * 2 + S * 4 * 2;
  std::vector<int64_t> per_layer;
  for (int64_t i = 0; i < L; ++i) {
    per_layer.push_back(hc + dense + experts + ((i + 1) % 4 == 0 ? qsa : gdn) + (i == 1 ? ple : 0));
  }
  const int64_t hc_final = S * 2 + lr * S * 2 + S * lr * 2;
  const int64_t lm_head = V * H / 2 + V * H / 32 * 4;  // w4a16 group 32: nibbles + one u32 (scale, zero) per 32
  int64_t total = hc_final + lm_head;
  for (int64_t b : per_layer) total += b;
  std::printf("real-model estimate: %.2f GiB of weights, %.1f MiB per layer, tail %.2f GiB\n",
              static_cast<double>(total) / GiB, static_cast<double>(per_layer[0]) / (1 << 20),
              static_cast<double>(hc_final + lm_head) / GiB);
  Check(total > 48 * GiB && total < 52 * GiB, "estimate within plan 3.1's ~50.8 GB band");
  const Qwen4ExpSplit s = ChooseSplit(per_layer, 0, hc_final + lm_head);
  std::printf("  k = %lld, device 0 %.2f GiB, device 1 %.2f GiB\n", static_cast<long long>(s.k),
              static_cast<double>(s.dev0_bytes) / GiB, static_cast<double>(s.dev1_bytes) / GiB);
  Check(s.k >= 23 && s.k <= 25, "real-model split lands near the middle (k 23..25)");
  Check(s.dev0_bytes <= kQwen4ExpDefaultWeightCapBytes && s.dev1_bytes <= kQwen4ExpDefaultWeightCapBytes,
        "both cards within the default 26 GiB cap");
  const int64_t max_layer = *std::max_element(per_layer.begin(), per_layer.end());
  const int64_t diff = s.dev0_bytes > s.dev1_bytes ? s.dev0_bytes - s.dev1_bytes : s.dev1_bytes - s.dev0_bytes;
  Check(diff <= max_layer, "imbalance at most one layer's bytes");
  Check(s.dev0_bytes + s.dev1_bytes == total, "the two totals add up to the model's bytes");
  Check(Throws([&] { ChooseSplit(per_layer, 0, hc_final + lm_head, 24 * GiB); }, "no layer boundary"),
        "a 24 GiB cap cannot hold 50.8 GB over two cards");
}

void TestToModelConfig() {
  const fs::path cfg = fs::path("E:\\models\\r4dx\\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\\config.json");
  if (!fs::exists(cfg)) {
    std::printf("SKIP: ToModelConfig on the real config (%s absent)\n", cfg.string().c_str());
    return;
  }
  std::ifstream f(cfg);
  const nlohmann::json j = nlohmann::json::parse(f);
  const Qwen4ExpConfig c = Qwen4ExpConfig::FromJson(j);
  const ModelConfig m = c.ToModelConfig();
  Check(m.arch == Arch::kQwen4Exp, "ToModelConfig: arch == kQwen4Exp");
  Check(m.hidden_size == 2560 && m.num_hidden_layers == 48 && m.vocab_size == 248320, "ToModelConfig: hidden / layers / vocab");
  Check(m.layer_types.size() == 48 && m.layer_types[0] == "linear_attention" && m.layer_types[3] == "full_attention" &&
            m.layer_types[47] == "full_attention" && m.layer_types[46] == "linear_attention",
        "ToModelConfig: layer_types spelling");
  Check(m.IsGdnLayer(0) && !m.IsGdnLayer(3), "ToModelConfig: IsGdnLayer agrees");
  Check(m.num_attention_heads == 24 && m.num_key_value_heads == 2 && m.head_dim == 256, "ToModelConfig: attention geometry");
  Check(m.linear_num_key_heads == 16 && m.linear_num_value_heads == 48 && m.linear_conv_kernel_dim == 4,
        "ToModelConfig: GDN geometry");
  Check(m.RotaryDim() == 64 && m.rope_theta == 1.0e7, "ToModelConfig: rotary dim 64, theta 1e7");
}

// ---- part 2/3: the container ------------------------------------------------------------------------------------

struct FileTensor {
  const uint8_t* data;
  size_t bytes;
};

FileTensor FileOf(const SafetensorsReader& r, const std::string& name) {
  if (!r.Has(name)) {
    std::fprintf(stderr, "FAIL: container has no tensor '%s'\n", name.c_str());
    ++g_failures;
    return {nullptr, 0};
  }
  const auto& m = r.Meta(name);
  return {r.Data(name), static_cast<size_t>(m.end - m.begin)};
}

template <class T>
std::vector<T> Download(const r4dx::core::DeviceBuffer<T>& b) {
  return b.CopyToHost();
}

// Device bytes of `b` (a bf16 / u8 / u32 buffer) equal the file tensor `name`'s bytes.
template <class T>
bool SameBytes(const SafetensorsReader& r, const std::string& name, const r4dx::core::DeviceBuffer<T>& b) {
  const FileTensor f = FileOf(r, name);
  if (f.data == nullptr || f.bytes != b.bytes()) {
    std::fprintf(stderr, "  %s: file %zu bytes, device %zu bytes\n", name.c_str(), f.bytes, b.bytes());
    return false;
  }
  const std::vector<T> host = Download(b);
  return std::memcmp(host.data(), f.data, f.bytes) == 0;
}

// fp16 file tensor == fp32 device buffer, element by element (exact: fp16 -> fp32 is lossless).
bool SameWidenedF16(const SafetensorsReader& r, const std::string& name, const r4dx::core::DeviceBuffer<float>& b) {
  const FileTensor f = FileOf(r, name);
  if (f.data == nullptr || f.bytes != b.size() * 2) return false;
  const std::vector<float> host = Download(b);
  for (size_t i = 0; i < host.size(); ++i) {
    uint16_t h;
    std::memcpy(&h, f.data + 2 * i, 2);
    if (host[i] != r4dx::core::F16ToFloat(h)) return false;
  }
  return true;
}

size_t FreeBytes() {
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return 0;
  return free_b;
}

int PickDevice() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 1) return -1;
  return std::getenv("HIP_VISIBLE_DEVICES") != nullptr ? 0 : (n > 1 ? 1 : 0);
}

int64_t SumLayers(const Qwen4ExpContainerInfo& info, int64_t a, int64_t b) {
  int64_t t = 0;
  for (int64_t i = a; i < b; ++i) t += info.layer_bytes[static_cast<size_t>(i)].Total();
  return t;
}

void CheckLinearShape(const QuantLinear& q, int64_t N, int64_t K, const std::string& what) {
  Check(q.N == N && q.K == K, what + ": N x K == " + std::to_string(N) + " x " + std::to_string(K));
}

// One expert class: views alias the class buffer back to back, are non-owning, and have the file's bytes.
void CheckExpertClass(const SafetensorsReader& r, const qwen4exp::ExpertClassStorage& st,
                      const qwen4exp::MoeWeights& moe, bool gate_up, int64_t layer, int64_t E, int64_t N, int64_t K) {
  const char* cls = gate_up ? "gate_up" : "down";
  const std::string tag = "layer " + std::to_string(layer) + " " + cls;
  Check(st.experts == E && st.N == N && st.K == K, tag + ": class storage records E, N, K");
  bool alias = true, nonowning = true, sized = true;
  const auto lin = [&](int64_t e) -> const QuantLinear& {
    return gate_up ? moe.experts[static_cast<size_t>(e)].gate_up : moe.experts[static_cast<size_t>(e)].down;
  };
  const size_t nk = static_cast<size_t>(N) * static_cast<size_t>(K);
  for (int64_t e = 0; e < E; ++e) {
    const QuantLinear& q = lin(e);
    sized = sized && q.layout == st.layout && q.N == N && q.K == K;
    switch (st.layout) {
      case Layout::kBf16:
        alias = alias && q.bf16_w.data() == st.bf16_w.data() + static_cast<size_t>(e) * nk && q.bf16_w.size() == nk;
        nonowning = nonowning && !q.bf16_w.owned();
        break;
      case Layout::kW4a16:
        alias = alias && q.wq.data() == st.wq.data() + static_cast<size_t>(e) * (nk / 2) && q.wq.size() == nk / 2 &&
                q.w4a16_wsz.size() * static_cast<size_t>(E) == st.wsz.size();
        nonowning = nonowning && !q.wq.owned() && !q.w4a16_wsz.owned();
        break;
      case Layout::kTrellis: {
        const size_t w_per = nk * static_cast<size_t>(st.bits) / 32;
        alias = alias && q.trellis_w.data() == st.trellis_w.data() + static_cast<size_t>(e) * w_per &&
                q.trellis_w.size() == w_per && q.trellis_svh.size() == static_cast<size_t>(N) &&
                q.trellis_suh.size() == static_cast<size_t>(q.trellis_parts) * static_cast<size_t>(K) &&
                q.trellis_svh.data() == st.trellis_svh.data() + static_cast<size_t>(e) * static_cast<size_t>(N);
        nonowning = nonowning && !q.trellis_w.owned() && !q.trellis_suh.owned() && !q.trellis_svh.owned();
        break;
      }
    }
  }
  Check(alias, tag + ": all " + std::to_string(E) + " views alias the class buffer back to back");
  Check(nonowning, tag + ": views are non-owning");
  Check(sized, tag + ": views carry layout / N / K");
  switch (st.layout) {
    case Layout::kBf16:
      Check(st.bf16_w.owned() && st.bf16_w.size() == static_cast<size_t>(E) * nk, tag + ": one owning bf16 buffer");
      break;
    case Layout::kW4a16:
      Check(st.wq.owned() && st.wsz.owned(), tag + ": one owning wq + one wsz buffer");
      break;
    case Layout::kTrellis:
      Check(st.trellis_w.owned() && st.trellis_suh.owned() && st.trellis_svh.owned() && st.bits >= 3,
            tag + ": one owning words / suh / svh buffer");
      break;
  }
  // Bytes: the first, a middle and the last expert, downloaded and compared with the file.
  for (int64_t e : std::set<int64_t>{0, E / 2, E - 1}) {
    const QuantLinear& q = lin(e);
    const std::string base = "text.layers." + std::to_string(layer) + ".moe.experts." + std::to_string(e) + "." + cls;
    const std::string et = tag + " expert " + std::to_string(e);
    switch (st.layout) {
      case Layout::kBf16:
        Check(SameBytes(r, base + ".bf16.w", q.bf16_w), et + ": bf16 bytes == file");
        break;
      case Layout::kW4a16:
        Check(SameBytes(r, base + ".w4a16.wq", q.wq), et + ": wq bytes == file");
        break;
      case Layout::kTrellis:
        Check(SameBytes(r, base + ".trellis.w", q.trellis_w), et + ": trellis words == file");
        Check(SameWidenedF16(r, base + ".trellis.suh", q.trellis_suh), et + ": suh == widened file fp16");
        Check(SameWidenedF16(r, base + ".trellis.svh", q.trellis_svh), et + ": svh == widened file fp16");
        break;
    }
  }
}

void CheckLayerShapes(const Qwen4ExpContainer& c, const Qwen4ExpConfig& cfg, int64_t i) {
  const qwen4exp::LayerWeights& lw = c.Layer(i);
  const int64_t H = cfg.hidden_size, S = cfg.hc_count * H, E = cfg.num_experts;
  const std::string tag = "layer " + std::to_string(i);
  Check(lw.qsa == cfg.IsQsaLayer(i) && lw.gdn.has_value() == !lw.qsa && lw.attn.has_value() == lw.qsa,
        tag + ": exactly the mixer of its kind is populated");
  Check(lw.hc_attn.norm.size() == static_cast<size_t>(S) && lw.hc_attn.down.size() == static_cast<size_t>(cfg.hc_lowrank * S) &&
            lw.hc_attn.up.size() == static_cast<size_t>(S * cfg.hc_lowrank) &&
            lw.hc_attn.inject.size() == static_cast<size_t>(cfg.hc_count * S) && lw.hc_mlp.inject.size() == lw.hc_attn.inject.size(),
        tag + ": hc shapes");
  Check(lw.moe.router.size() == static_cast<size_t>(E * H) && lw.moe.shared_gate.size() == static_cast<size_t>(H) &&
            lw.moe.experts.size() == static_cast<size_t>(E),
        tag + ": router / shared_gate / expert count");
  CheckLinearShape(lw.moe.shared.gate, cfg.shared_expert_intermediate_size, H, tag + " shared.gate");
  CheckLinearShape(lw.moe.shared.down, H, cfg.shared_expert_intermediate_size, tag + " shared.down");
  Check(lw.moe.shared.gate.layout == Layout::kBf16 && lw.moe.shared.up.layout == Layout::kBf16, tag + ": shared experts are bf16");
  if (lw.gdn) {
    const int64_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int64_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    CheckLinearShape(lw.gdn->in_proj_qkv, 2 * key_dim + value_dim, H, tag + " in_proj_qkv");
    CheckLinearShape(lw.gdn->in_proj_z, value_dim, H, tag + " in_proj_z");
    CheckLinearShape(lw.gdn->out_proj, H, value_dim, tag + " out_proj");
    Check(lw.gdn->in_proj_a.size() == static_cast<size_t>(cfg.linear_num_value_heads * H) &&
              lw.gdn->conv1d_weight.size() == static_cast<size_t>((2 * key_dim + value_dim) * cfg.linear_conv_kernel_dim) &&
              lw.gdn->A_log.size() == static_cast<size_t>(cfg.linear_num_value_heads) &&
              lw.gdn->norm_weight.size() == static_cast<size_t>(cfg.linear_value_head_dim),
          tag + ": GDN raw tensor sizes");
  } else {
    const int64_t qdim = cfg.num_attention_heads * cfg.head_dim, kvdim = cfg.num_key_value_heads * cfg.head_dim;
    CheckLinearShape(lw.attn->qg, 2 * qdim, H, tag + " attn.qg");
    CheckLinearShape(lw.attn->k, kvdim, H, tag + " attn.k");
    CheckLinearShape(lw.attn->v, kvdim, H, tag + " attn.v");
    CheckLinearShape(lw.attn->o, H, qdim, tag + " attn.o");
    Check(lw.attn->k_descale.size() == static_cast<size_t>(cfg.num_key_value_heads) &&
              lw.attn->indexer_qk.size() ==
                  static_cast<size_t>((cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim * H) &&
              lw.attn->indexer_q_norm.size() == static_cast<size_t>(cfg.indexer_head_dim),
          tag + ": QSA raw tensor sizes");
  }
  Check(lw.ple.has_value() == cfg.IsPleLayer(i), tag + ": PLE exactly on the PLE layer");
  CheckLinearShape(lw.moe.experts[0].gate_up, 2 * cfg.moe_intermediate_size, H, tag + " expert gate_up");
  CheckLinearShape(lw.moe.experts[0].down, H, cfg.moe_intermediate_size, tag + " expert down");
}

// Disjoint, in-bounds ticket slices of every trellis linear of `c`, N/128 words each, in one buffer.
void CheckTickets(const Qwen4ExpContainer& c) {
  const uint32_t* base = c.TrellisTickets().data();
  const size_t words = c.TrellisTickets().size();
  std::vector<std::pair<size_t, size_t>> spans;  // [begin, end) in words
  const auto note = [&](const QuantLinear& q) {
    if (q.layout != Layout::kTrellis) return;
    const size_t off = static_cast<size_t>(q.trellis_tickets - base);
    spans.emplace_back(off, off + static_cast<size_t>(q.N / 128));
  };
  for (int64_t i = c.FirstLayer(); i < c.EndLayer(); ++i) {
    const qwen4exp::LayerWeights& lw = c.Layer(i);
    if (lw.gdn) {
      note(lw.gdn->in_proj_qkv);
      note(lw.gdn->in_proj_z);
      note(lw.gdn->out_proj);
    }
    if (lw.attn) {
      note(lw.attn->qg);
      note(lw.attn->k);
      note(lw.attn->v);
      note(lw.attn->o);
    }
    for (const qwen4exp::ExpertLinears& e : lw.moe.experts) {
      note(e.gate_up);
      note(e.down);
    }
  }
  std::sort(spans.begin(), spans.end());
  bool disjoint = true, in_bounds = true;
  size_t covered = 0;
  for (size_t i = 0; i < spans.size(); ++i) {
    in_bounds = in_bounds && spans[i].second <= words && spans[i].first < spans[i].second;
    if (i > 0) disjoint = disjoint && spans[i - 1].second <= spans[i].first;
    covered += spans[i].second - spans[i].first;
  }
  Check(static_cast<int64_t>(spans.size()) == c.NumTrellisLinears() && !spans.empty(),
        "tickets: " + std::to_string(spans.size()) + " trellis linears each have a slice");
  Check(disjoint && in_bounds, "tickets: slices are disjoint and inside the one buffer");
  Check(covered == words, "tickets: the slices cover the buffer exactly (" + std::to_string(words) + " words)");
  std::vector<uint32_t> host = c.TrellisTickets().CopyToHost();
  Check(std::all_of(host.begin(), host.end(), [](uint32_t v) { return v == 0; }), "tickets: zero-initialised");
}

void TestContainer(const std::string& path, int device, Layout expected) {
  const bool trellis_expected = expected == Layout::kTrellis;
  // A bf16 / trellis container loads at its own layout by default; a w4a16 one is asked for explicitly.
  const std::optional<Layout> req = expected == Layout::kW4a16 ? std::optional<Layout>(expected) : std::nullopt;
  const std::string label = std::string("[") + LayoutName(expected) + "] ";
  std::printf("== %s%s\n", label.c_str(), path.c_str());
  if (hipSetDevice(device) != hipSuccess) {
    Check(false, "hipSetDevice");
    return;
  }
  const Qwen4ExpContainerInfo info = Qwen4ExpContainer::Inspect(path, req);
  const Qwen4ExpConfig& cfg = info.config;
  Check(info.has_trellis == trellis_expected && info.body_layout == expected,
        label + "Inspect: body layout");
  Check(info.model_config.arch == Arch::kQwen4Exp && info.model_config.num_hidden_layers == cfg.num_hidden_layers,
        label + "Inspect: generic model config");
  Check(static_cast<int64_t>(info.layer_bytes.size()) == cfg.num_hidden_layers && info.embed_host_bytes > 0 &&
            info.lm_head_bytes > 0 && info.hc_final_bytes > 0,
        label + "Inspect: bytes per layer + globals");
  Check(std::is_sorted(info.tensor_names.begin(), info.tensor_names.end()) &&
            std::binary_search(info.tensor_names.begin(), info.tensor_names.end(), std::string("text.embed_tokens")),
        label + "Inspect: sorted tensor names incl. text.embed_tokens");
  if (info.ngram) {
    Check(info.ngram->rows > 0 && info.ngram->sha256.size() == 64 && !info.ngram->resolved_path.empty(),
          label + "Inspect: n-gram sidecar descriptor (" + std::to_string(info.ngram->rows) + " rows)");
  } else {
    Check(false, label + "Inspect: the container carries an ngram_sidecar record");
  }
  bool kb_ok = true;
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    const int g = info.gate_up_kb[static_cast<size_t>(i)], d = info.down_kb[static_cast<size_t>(i)];
    kb_ok = kb_ok && (trellis_expected ? (g >= 3 && g <= 5 && d >= 3 && d <= 5) : (g == 0 && d == 0));
  }
  Check(kb_ok, label + "Inspect: per-layer KB of both expert classes");
  {
    const Qwen4ExpSplit s = ChooseSplit(info.PerLayerDeviceBytes(), 0, info.TailDeviceBytes(), 64LL * GiB);
    Check(s.k >= 1 && s.k < cfg.num_hidden_layers && s.dev0_bytes + s.dev1_bytes == info.TotalDeviceBytes(),
          label + "ChooseSplit on Inspect's numbers (k " + std::to_string(s.k) + ")");
  }

  const SafetensorsReader reader(Utf8ToWide(path));
  const int64_t L = cfg.num_hidden_layers, mid = L / 2;
  using r4dx::core::DeviceBufferBytes;
  const auto counter = [&] { return DeviceBufferBytes(device); };
  const int64_t counted0 = counter();
  const size_t free0 = FreeBytes();

  {
    Qwen4ExpLoadOptions o;
    o.device = device;
    o.layout = req;
    o.layer_first = 0;
    o.layer_last = mid;
    Qwen4ExpContainer c = Qwen4ExpContainer::Load(path, o);
    Check(c.FirstLayer() == 0 && c.EndLayer() == mid && c.NumLoadedLayers() == mid && !c.HasEmbed() && !c.HasHead(),
          label + "Load [0, " + std::to_string(mid) + "): range, no embed, no head");
    Check(Throws([&] { c.Layer(mid); }, "outside the loaded range"), label + "Layer(mid) is outside the range");
    Check(Throws([&] { c.EmbedTokensHost(); }, "load_embed") && Throws([&] { c.LmHead(); }, "load_head"),
          label + "embed / head accessors refuse when not loaded");
    const int64_t predicted = SumLayers(info, 0, mid);
    const int64_t counted = counter() - counted0;
    std::printf("  VRAM report %lld bytes, Inspect predicted %lld, counter growth %lld\n",
                static_cast<long long>(c.DeviceBytes()), static_cast<long long>(predicted), static_cast<long long>(counted));
    Check(c.DeviceBytes() == counted, label + "VRAM report == the DeviceBuffer counter's growth");
    Check(c.DeviceBytes() == predicted, label + "VRAM report == Inspect's per-layer bytes");
    for (int64_t i = 0; i < mid; ++i) CheckLayerShapes(c, cfg, i);
    for (int64_t i = 0; i < mid; ++i) {
      const qwen4exp::LayerWeights& lw = c.Layer(i);
      CheckExpertClass(reader, lw.moe.gate_up_store, lw.moe, true, i, cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size);
      CheckExpertClass(reader, lw.moe.down_store, lw.moe, false, i, cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size);
    }
    const qwen4exp::LayerWeights& l0 = c.Layer(0);
    const std::string b0 = "text.layers.0.";
    Check(SameBytes(reader, b0 + "hc.attn.norm", l0.hc_attn.norm) && SameBytes(reader, b0 + "hc.mlp.inject", l0.hc_mlp.inject) &&
              SameBytes(reader, b0 + "hc.attn.down", l0.hc_attn.down),
          label + "layer 0 hc tensors == file");
    Check(SameBytes(reader, b0 + "moe.router", l0.moe.router) && SameBytes(reader, b0 + "moe.shared_gate", l0.moe.shared_gate) &&
              SameBytes(reader, b0 + "moe.shared.gate.bf16.w", l0.moe.shared.gate.bf16_w) &&
              SameBytes(reader, b0 + "moe.shared.down.bf16.w", l0.moe.shared.down.bf16_w),
          label + "layer 0 router / shared expert == file");
    Check(SameBytes(reader, b0 + "gdn.conv1d_weight", l0.gdn->conv1d_weight) && SameBytes(reader, b0 + "gdn.A_log", l0.gdn->A_log) &&
              SameBytes(reader, b0 + "gdn.dt_bias", l0.gdn->dt_bias),
          label + "layer 0 GDN conv1d / A_log / dt_bias == file");
    {
      const FileTensor f = FileOf(reader, b0 + "gdn.norm_weight");
      const std::vector<float> w = Download(l0.gdn->norm_weight);
      bool ok = f.data != nullptr && w.size() * 2 == f.bytes;
      for (size_t j = 0; ok && j < w.size(); ++j) {
        uint16_t h;
        std::memcpy(&h, f.data + 2 * j, 2);
        ok = w[j] == r4dx::core::Bf16ToFloat(h);
      }
      Check(ok, label + "layer 0 gdn.norm_weight == bf16 file widened to fp32");
    }
    const qwen4exp::LayerWeights& l1 = c.Layer(1);
    Check(l1.ple.has_value() && SameBytes(reader, "text.layers.1.ple.key_proj", l1.ple->key_proj) &&
              SameBytes(reader, "text.layers.1.ple.conv1d", l1.ple->conv1d) &&
              SameBytes(reader, "text.layers.1.ple.norm_conv", l1.ple->norm_conv),
          label + "layer 1 PLE tensors == file");
    if (trellis_expected) {
      CheckTickets(c);
      Check(c.NumTrellisLinears() > 0 && c.TrellisTickets().bytes() > 0, label + "trellis tickets exist");
    } else {
      Check(c.NumTrellisLinears() == 0 && c.TrellisTickets().empty(), label + "no tickets on a non-trellis container");
    }
  }
  Check(counter() == counted0, label + "destroying the container returns the counter to its baseline");

  {
    Qwen4ExpLoadOptions o;
    o.device = device;
    o.layout = req;
    o.layer_first = mid;
    o.layer_last = -1;
    o.load_embed = true;
    o.load_head = true;
    Qwen4ExpContainer c = Qwen4ExpContainer::Load(path, o);
    Check(c.FirstLayer() == mid && c.EndLayer() == L && c.HasEmbed() && c.HasHead(),
          label + "Load [" + std::to_string(mid) + ", " + std::to_string(L) + ") + embed + head");
    const int64_t predicted = SumLayers(info, mid, L) + info.TailDeviceBytes();
    const int64_t counted = counter() - counted0;
    std::printf("  VRAM report %lld bytes, Inspect predicted %lld, counter growth %lld\n",
                static_cast<long long>(c.DeviceBytes()), static_cast<long long>(predicted), static_cast<long long>(counted));
    Check(c.DeviceBytes() == counted && c.DeviceBytes() == predicted, label + "VRAM report == counter == Inspect (layers + hc_final + lm_head)");
    for (int64_t i = mid; i < L; ++i) CheckLayerShapes(c, cfg, i);
    const int64_t qsa_layer = cfg.IsQsaLayer(L - 1) ? L - 1 : -1;
    if (qsa_layer >= 0) {
      const qwen4exp::QsaWeights& a = *c.Layer(qsa_layer).attn;
      const std::string b = "text.layers." + std::to_string(qsa_layer) + ".";
      Check(SameBytes(reader, b + "attn.k_descale", a.k_descale) && SameBytes(reader, b + "attn.v_descale", a.v_descale) &&
                SameBytes(reader, b + "attn.indexer.qk", a.indexer_qk) && SameBytes(reader, b + "attn.q_norm", a.q_norm) &&
                SameBytes(reader, b + "attn.indexer.k_norm", a.indexer_k_norm),
            label + "last QSA layer: descales / indexer / norms == file");
    }
    Check(SameBytes(reader, "text.hc_final.norm", c.HcFinal().norm) && SameBytes(reader, "text.hc_final.up", c.HcFinal().up) &&
              c.HcFinal().inject.empty(),
          label + "hc_final == file, no inject");
    CheckLinearShape(c.LmHead(), cfg.vocab_size, cfg.hidden_size, label + "lm_head");
    const FileTensor emb = FileOf(reader, "text.embed_tokens");
    Check(emb.data != nullptr && emb.bytes == static_cast<size_t>(info.embed_host_bytes) &&
              std::memcmp(c.EmbedTokensHost(), emb.data, emb.bytes) == 0,
          label + "host embedding == file");
    if (trellis_expected) CheckTickets(c);
  }
  Check(counter() == counted0, label + "counter back at its baseline after the second load");

  // Twice, concurrently alive: bytes double; destroyed: everything freed (the allocator's own free memory too).
  {
    Qwen4ExpLoadOptions o;
    o.device = device;
    o.layout = req;
    o.layer_first = 0;
    o.layer_last = mid;
    const int64_t one = SumLayers(info, 0, mid);
    {
      Qwen4ExpContainer a = Qwen4ExpContainer::Load(path, o);
      Qwen4ExpContainer b = Qwen4ExpContainer::Load(path, o);
      Check(counter() - counted0 == 2 * one, label + "two live loads hold exactly twice the bytes");
      const size_t used = free0 > FreeBytes() ? free0 - FreeBytes() : 0;
      Check(used >= static_cast<size_t>(2 * one), label + "hipMemGetInfo: free memory dropped by at least 2x the report");
      Qwen4ExpContainer moved = std::move(a);
      Check(moved.DeviceBytes() == one && counter() - counted0 == 2 * one,
            label + "a moved container keeps its buffers (views stay valid)");
      CheckExpertClass(reader, moved.Layer(0).moe.down_store, moved.Layer(0).moe, false, 0, cfg.num_experts, cfg.hidden_size,
                       cfg.moe_intermediate_size);
    }
    Check(counter() == counted0, label + "both destroyed -> counter at baseline");
    static_cast<void>(hipDeviceSynchronize());
    const size_t free1 = FreeBytes();
    const int64_t drift = static_cast<int64_t>(free0) - static_cast<int64_t>(free1);
    std::printf("  free memory before %.1f MiB, after %.1f MiB (drift %.2f MiB)\n", static_cast<double>(free0) / (1 << 20),
                static_cast<double>(free1) / (1 << 20), static_cast<double>(drift) / (1 << 20));
    Check(drift < 64LL * (1 << 20) && drift > -64LL * (1 << 20), label + "hipMemGetInfo: memory freed (drift < 64 MiB)");
  }

  // Refusals.
  {
    Qwen4ExpLoadOptions bad;
    bad.device = device;
    bad.layout = req;
    bad.layer_first = 3;
    bad.layer_last = 2;
    Check(Throws([&] { Qwen4ExpContainer::Load(path, bad); }, "outside"), label + "reversed layer range refused");
    bad.layer_first = 0;
    bad.layer_last = L + 1;
    Check(Throws([&] { Qwen4ExpContainer::Load(path, bad); }, "outside"), label + "range past the last layer refused");
    Qwen4ExpLoadOptions wrong;
    wrong.layout = expected == Layout::kTrellis ? Layout::kBf16 : Layout::kTrellis;
    Check(Throws([&] { Qwen4ExpContainer::Load(path, wrong); }), label + "a layout the container cannot serve is refused");
    Check(Throws([&] { Qwen4ExpContainer::Inspect("E:\\no\\such\\container.r4dx"); }), label + "Inspect of a missing file throws");
  }
  Check(counter() == counted0, label + "refused loads leak nothing");
}

// ---- a synthetic trellis container --------------------------------------------------------------------------------
// The converter's tiny trellis container needs the KB 3 / trellis pipeline; until it exists the loader's trellis
// paths (views, tickets, widened sign vectors, per-layer KB, uniformity) are exercised on a container derived from the
// bf16 one: every body linear's `.bf16.w` is replaced by `.trellis.{w,suh,svh}` of the right byte sizes holding
// deterministic pseudo-random words / finite fp16 sign vectors, with a `quant.trellis` block (verify "pass") naming
// them. The loader never decodes a word, so random content is a faithful test of what it moves.

uint64_t NextRandom(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string BuildSyntheticTrellis(const std::string& bf16_path) {
  const fs::path dir = "E:\\models\\r4dx\\q4exp-fixtures\\LoaderQ4";
  fs::create_directories(dir);
  const fs::path out = dir / "synthetic-trellis.r4dx";
  if (fs::exists(out) && fs::last_write_time(out) > fs::last_write_time(bf16_path)) return out.string();

  const SafetensorsReader src(Utf8ToWide(bf16_path));
  nlohmann::json hdr = nlohmann::json::parse(src.HeaderJson());
  nlohmann::json meta = hdr.at("__metadata__");
  const Qwen4ExpConfig cfg = Qwen4ExpConfig::FromJson(meta.at("model_config"));
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const int64_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const int64_t qdim = cfg.num_attention_heads * cfg.head_dim, kvdim = cfg.num_key_value_heads * cfg.head_dim;
  // The rates the kernel instantiates, rotated over (layer, class).
  std::vector<int> kbs;
  for (int b : {3, 4, 5}) {
    if (r4d_gemm_trellis_nt_m64_has_rate(b) != 0) kbs.push_back(b);
  }

  struct Out {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t bytes = 0;
    bool copy = false;  // copied from the source tensor of the same name
  };
  std::vector<Out> outs;
  nlohmann::json linears = nlohmann::json::object();
  int64_t hf_tensors = 0;
  std::vector<std::string> names = src.Names();
  std::sort(names.begin(), names.end());
  const std::string suffix = ".bf16.w";
  for (const std::string& n : names) {
    const auto& m = src.Meta(n);
    const bool bf16w = EndsWith(n, suffix);
    const std::string base = bf16w ? n.substr(0, n.size() - suffix.size()) : std::string();
    int64_t N = 0, K = 0, parts = 1;
    int bits = 4;
    bool body = false;
    if (bf16w && base.rfind("text.layers.", 0) == 0) {
      const int64_t layer = std::stoll(base.substr(12, base.find('.', 12) - 12));
      const bool expert = base.find(".moe.experts.") != std::string::npos;
      if (EndsWith(base, ".gdn.in_proj_qkv")) {
        N = 2 * key_dim + value_dim, K = H, body = true;
      } else if (EndsWith(base, ".gdn.in_proj_z")) {
        N = value_dim, K = H, body = true;
      } else if (EndsWith(base, ".gdn.out_proj")) {
        N = H, K = value_dim, body = true;
      } else if (EndsWith(base, ".attn.qg")) {
        N = 2 * qdim, K = H, body = true;
      } else if (EndsWith(base, ".attn.k") || EndsWith(base, ".attn.v")) {
        N = kvdim, K = H, body = true;
      } else if (EndsWith(base, ".attn.o")) {
        N = H, K = qdim, body = true;
      } else if (expert && EndsWith(base, ".gate_up")) {
        N = 2 * I, K = H, parts = 2, body = true;
        bits = kbs[static_cast<size_t>(layer) % kbs.size()];
      } else if (expert && EndsWith(base, ".down")) {
        N = H, K = I, body = true;
        bits = kbs[static_cast<size_t>(layer + 1) % kbs.size()];
      }
    }
    if (!body) {
      outs.push_back({n, m.dtype, m.shape, m.end - m.begin, true});
      continue;
    }
    if (static_cast<uint64_t>(N * K * 2) != m.end - m.begin) throw std::runtime_error("synthetic trellis: unexpected size of " + n);
    outs.push_back({base + ".trellis.w", "U8", {N * K * bits / 8}, static_cast<uint64_t>(N * K * bits / 8), false});
    outs.push_back({base + ".trellis.suh", "U8", {parts * K * 2}, static_cast<uint64_t>(parts * K * 2), false});
    outs.push_back({base + ".trellis.svh", "U8", {N * 2}, static_cast<uint64_t>(N * 2), false});
    nlohmann::json e = {{"bits", bits}};
    if (parts == 2) e["parts"] = {I, I};
    linears[base] = e;
    hf_tensors += parts;
  }
  nlohmann::json trellis = {
      {"format", "r4dx-trellis"}, {"version", 1}, {"codebook", "mul1"},
      {"codebook_consts", {{"mult", "0x83dcd12d"}, {"k_inv_f16", "0x1eee"}, {"k_bias_f16", "0xc931"}}},
      {"state_bits", 16}, {"tail_biting", true}, {"position_order", "exl3-tensor-core"},
      {"bitstream", "ring-u32-msb-first"}, {"tile_grid", "n32-pairs-k-major"},
      {"hadamard", {{"block", 128}, {"order", "sylvester-natural"}, {"scale", "1/sqrt(128)"},
                    {"input", "x*suh then H"}, {"output", "H then *svh"}}},
      {"prescale_log2", 0}, {"linears", linears}};
  if (!meta.contains("quant") || !meta.at("quant").is_object()) meta["quant"] = nlohmann::json::object();
  meta["quant"]["trellis"] = trellis;
  meta["r4dx_convert_run"] = {{"trellis", {{"verify", {{"result", "pass"}, {"failed", 0}, {"checked", hf_tensors}}}}}};

  nlohmann::json head = nlohmann::json::object();
  head["__metadata__"] = meta;
  uint64_t off = 0;
  for (const Out& o : outs) {
    head[o.name] = {{"dtype", o.dtype}, {"shape", o.shape}, {"data_offsets", {off, off + o.bytes}}};
    off += o.bytes;
  }
  std::string hjson = head.dump();
  while (hjson.size() % 8 != 0) hjson.push_back(' ');
  const fs::path tmp = fs::path(out).replace_extension(".tmp");
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    const uint64_t hl = hjson.size();
    f.write(reinterpret_cast<const char*>(&hl), 8);
    f.write(hjson.data(), static_cast<std::streamsize>(hjson.size()));
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    std::vector<uint8_t> buf;
    for (const Out& o : outs) {
      if (o.copy) {
        f.write(reinterpret_cast<const char*>(src.Data(o.name)), static_cast<std::streamsize>(o.bytes));
        continue;
      }
      buf.resize(static_cast<size_t>(o.bytes));
      if (EndsWith(o.name, ".trellis.suh") || EndsWith(o.name, ".trellis.svh")) {
        for (size_t i = 0; i + 1 < buf.size(); i += 2) {
          // a finite fp16 in [1, 2) with a random sign and mantissa (exponent 01111)
          const uint16_t r = static_cast<uint16_t>(NextRandom(seed));
          const uint16_t h = static_cast<uint16_t>((r & 0x83FF) | 0x3C00);
          std::memcpy(&buf[i], &h, 2);
        }
      } else {
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(NextRandom(seed) >> 24);
      }
      f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }
    if (!f) throw std::runtime_error("synthetic trellis: write failed for " + tmp.string());
  }
  fs::rename(tmp, out);
  return out.string();
}

// A copy of the trellis container with ONE expert's down KB flipped (same-width digit patch inside the JSON
// header; 4 -> 5, 3 -> 4, 5 -> 4): Inspect and Load must refuse it for non-uniformity.
void TestKbUniformity(const std::string& path, int device) {
  const SafetensorsReader reader(Utf8ToWide(path));
  const Qwen4ExpContainerInfo info = Qwen4ExpContainer::Inspect(path);
  const std::string base = "text.layers.0.moe.experts.3.down";
  const std::string meta_bytes = reader.HeaderJson();
  const std::string key = "\"" + base + "\":{";
  const size_t at = meta_bytes.find(key);
  if (at == std::string::npos) {
    Check(false, "KB uniformity: linears entry for " + base + " found in the header");
    return;
  }
  const size_t bits_at = meta_bytes.find("\"bits\":", at);
  if (bits_at == std::string::npos || bits_at > at + 200) {
    std::printf("SKIP: KB uniformity (header has no compact \"bits\" next to %s)\n", base.c_str());
    return;
  }
  const size_t digit = bits_at + 7;
  const fs::path tmp = fs::temp_directory_path() / "q4exp_loader_kb_flip.r4dx";
  {
    std::ifstream in(path, std::ios::binary);
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(1 << 22);
    uint64_t pos = 0;
    const uint64_t flip_pos = 8 + digit;  // 8 = the header-length prefix
    while (in) {
      in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
      const std::streamsize got = in.gcount();
      if (got <= 0) break;
      if (flip_pos >= pos && flip_pos < pos + static_cast<uint64_t>(got)) {
        char& ch = buf[static_cast<size_t>(flip_pos - pos)];
        ch = ch == '4' ? '5' : (ch == '5' ? '4' : '4');
      }
      out.write(buf.data(), got);
      pos += static_cast<uint64_t>(got);
    }
  }
  Check(Throws([&] { Qwen4ExpContainer::Inspect(tmp.string()); }, "uniform per layer"),
        "KB uniformity: a class whose experts disagree on KB is refused at Inspect");
  Qwen4ExpLoadOptions o;
  o.device = device;
  o.layer_first = 0;
  o.layer_last = 1;
  Check(Throws([&] { Qwen4ExpContainer::Load(tmp.string(), o); }, "uniform per layer"), "KB uniformity: ... and at Load");
  std::error_code ec;
  fs::remove(tmp, ec);
  (void)info;
}

}  // namespace

int main() {
  TestChooseSplitSynthetic();
  TestChooseSplitRealEstimate();
  TestToModelConfig();

  const char* env_bf16 = std::getenv("R4DX_Q4EXP_TINY_CONTAINER");
  const char* env_tr = std::getenv("R4DX_Q4EXP_TINY_TRELLIS");
  const std::string bf16_path = env_bf16 ? env_bf16 : "E:\\models\\r4dx\\tiny-q4exp\\container-bf16.r4dx";
  const std::string trellis_path = env_tr ? env_tr : "E:\\models\\r4dx\\tiny-q4exp\\container-trellis.r4dx";
  const std::string w4a16_path = "E:\\models\\r4dx\\tiny-q4exp\\container-w4a16.r4dx";
  const bool have_bf16 = fs::exists(bf16_path), have_trellis = fs::exists(trellis_path), have_w4a16 = fs::exists(w4a16_path);
  if (!have_bf16 && !have_trellis && !have_w4a16) {
    std::printf("SKIP: no tiny container (%s) -- ran the CPU parts only; failures so far: %d\n", bf16_path.c_str(), g_failures);
    return g_failures == 0 ? 77 : 1;
  }
  const int device = PickDevice();
  if (device < 0) {
    std::printf("SKIP: no HIP device\n");
    return g_failures == 0 ? 77 : 1;
  }
  try {
    if (have_bf16) TestContainer(bf16_path, device, Layout::kBf16);
    if (have_w4a16) TestContainer(w4a16_path, device, Layout::kW4a16);
    if (have_trellis) {
      TestContainer(trellis_path, device, Layout::kTrellis);
      TestKbUniformity(trellis_path, device);
    }
    if (have_bf16) {
      const std::string synthetic = BuildSyntheticTrellis(bf16_path);
      TestContainer(synthetic, device, Layout::kTrellis);
      TestKbUniformity(synthetic, device);
    }
    if (!have_trellis) std::printf("NOTE: the converter's trellis container %s is absent; only the synthetic one ran\n", trellis_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    ++g_failures;
  }
  std::printf("%s (%d failure(s))\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
