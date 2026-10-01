// tests/model/test_dflash_hf_weights.cpp -- CPU-only (no HIP call). docs/gemma4-plan.md D-3 / D-5: the runtime's
// DflashDraftWeights loader reads a container that `r4dx-convert --dflash-hf` (ConvertDflashHf) wrote: block 16,
// the 6 target layers of the real z-lab/gemma4-12B-it-DFlash (ids [1,10,19,27,36,45] -> inputs [2,11,20,28,37,46]),
// the optional logit_softcap / embed_scale / variant keys, the per-layer sliding pattern, and that a Qwen-style
// container (no optional keys) still parses to the defaults (softcap 0, embed_scale 1, variant ""). Also the
// loader's geometry facts the GPU side relies on (feature columns, selector codebook sizes).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "dflash_draft_weights.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/dflash2_hf.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {
int g_fail = 0;
void Check(bool c, const std::string& what) {
  if (!c) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

constexpr int64_t kH = 64, kFfn = 128, kHeads = 4, kKv = 2, kHd = 16, kLayers = 5, kVocab = 300;
const std::vector<int64_t> kIds = {1, 10, 19, 27, 36, 45};

void WriteCkpt(const fs::path& dir) {
  fs::create_directories(dir);
  json cfg = {{"architectures", {"DFlashDraftModel"}},
              {"block_size", 16},
              {"dflash_config", {{"mask_token_id", 4}, {"target_layer_ids", kIds}}},
              {"final_logit_softcapping", 30.0},
              {"head_dim", kHd},
              {"hidden_size", kH},
              {"intermediate_size", kFfn},
              {"layer_types", {"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}},
              {"max_position_embeddings", 262144},
              {"num_attention_heads", kHeads},
              {"num_hidden_layers", kLayers},
              {"num_key_value_heads", kKv},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters", {{"rope_theta", 1000000}, {"rope_type", "default"}}},
              {"sliding_window", 2048},
              {"use_sliding_window", true},
              {"vocab_size", kVocab}};
  std::ofstream(dir / "config.json") << cfg.dump();
  std::mt19937_64 rng(5);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  json header = json::object();
  std::vector<std::vector<uint16_t>> blobs;
  uint64_t off = 0;
  auto put = [&](const std::string& n, std::vector<int64_t> shape) {
    int64_t cnt = 1;
    for (auto d : shape) cnt *= d;
    std::vector<uint16_t> v(static_cast<size_t>(cnt));
    for (auto& x : v) x = r4dx::core::FloatToBf16(nd(rng));
    header[n] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {off, off + v.size() * 2}}};
    off += v.size() * 2;
    blobs.push_back(std::move(v));
  };
  put("fc.weight", {kH, static_cast<int64_t>(kIds.size()) * kH});
  put("hidden_norm.weight", {kH});
  put("norm.weight", {kH});
  for (int i = 0; i < kLayers; ++i) {
    const std::string h = "layers." + std::to_string(i) + ".";
    put(h + "input_layernorm.weight", {kH});
    put(h + "post_attention_layernorm.weight", {kH});
    put(h + "self_attn.q_proj.weight", {kHeads * kHd, kH});
    put(h + "self_attn.k_proj.weight", {kKv * kHd, kH});
    put(h + "self_attn.v_proj.weight", {kKv * kHd, kH});
    put(h + "self_attn.o_proj.weight", {kH, kHeads * kHd});
    put(h + "self_attn.q_norm.weight", {kHd});
    put(h + "self_attn.k_norm.weight", {kHd});
    put(h + "mlp.gate_proj.weight", {kFfn, kH});
    put(h + "mlp.up_proj.weight", {kFfn, kH});
    put(h + "mlp.down_proj.weight", {kH, kFfn});
  }
  const std::string hs = header.dump();
  std::ofstream f(dir / "model.safetensors", std::ios::binary);
  const uint64_t hl = hs.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const auto& b : blobs) f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size() * 2));
}
}  // namespace

int main() try {
  const fs::path root = fs::temp_directory_path() / "r4dx_test_dflash_hf_weights";
  fs::remove_all(root);
  WriteCkpt(root / "hf");
  r4dx_convert::DflashHfOptions o;
  o.input_dir = (root / "hf").string();
  o.output_path = (root / "d.r4dx").string();
  o.layout = "bf16";
  o.quiet = true;
  r4dx_convert::ConvertDflashHf(o);
  {
    auto w = std::make_unique<r4dx::model::DflashDraftWeights>(r4dx::model::DflashDraftWeights::Open(o.output_path));
    const auto& c = w->Config();
    Check(c.block_size == 16, "loader: block_size 16");
    Check(c.block_count == 5 && c.hidden_size == kH && c.feed_forward_length == kFfn, "loader: 5 layers, hidden, ffn");
    Check(c.target_layers == std::vector<int64_t>({2, 11, 20, 28, 37, 46}), "loader: target_layers [2,11,20,28,37,46] (ids + 1)");
    Check(c.mask_token_id == 4 && c.vocab_size == kVocab, "loader: mask id 4, vocab");
    Check(c.logit_softcap == 30.0 && c.embed_scale == 1.0 && c.variant == "v1_identity", "loader: softcap 30, embed_scale 1, variant");
    Check(c.rope.freq_base == 1e6 && c.rope.n_rot == kHd, "loader: rope theta 1e6, n_rot = head_dim");
    Check(c.attention.sliding_window == 2048 &&
              c.attention.sliding_window_pattern == std::vector<bool>({true, true, true, true, false}),
          "loader: window 2048, pattern 4 sliding + 1 full");
    Check(c.conv_kernel_size == 2 && c.conv_group_size == 16 && c.selector_rank == 256 && c.selector_top_k == 16,
          "loader: conv 2/16, selector 256/16");
    Check(c.layout == "bf16", "loader: layout bf16");
    Check(w->HasTensor("dflash.fc.bf16.w") && w->HasTensor("dflash.selector.predecessor") &&
              w->HasTensor("dflash.layers.4.mlp.conv.proj.bf16.w"),
          "loader: tensor directory");
    // the codebooks the host walk reads: [vocab][256] bf16 each, all zero
    const auto& m = w->TensorMeta("dflash.selector.successor");
    Check(m.end - m.begin == static_cast<uint64_t>(kVocab) * 256 * 2, "loader: successor codebook byte span");
    w.reset();  // unmap before the temp tree is removed
  }
  {
    // a Qwen-style container: write the metadata block by hand WITHOUT the optional keys -> defaults.
    r4dx_convert::DflashHfOptions o2 = o;
    o2.output_path = (root / "plain.r4dx").string();
    r4dx_convert::ConvertDflashHf(o2);
    // strip the optional keys from the header of a copy and re-open
    std::ifstream in(o2.output_path, std::ios::binary);
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    uint64_t hl = 0;
    std::memcpy(&hl, all.data(), 8);
    json header = json::parse(all.substr(8, hl));
    auto& d = header["__metadata__"]["dflash2"];
    for (const char* k : {"logit_softcap", "embed_scale", "variant", "target_layer_offset"}) d.erase(k);
    std::string nh = header.dump();
    while (nh.size() % 8 != 0) nh += ' ';
    const uint64_t nhl = nh.size();
    std::string out(reinterpret_cast<const char*>(&nhl), 8);
    out += nh;
    out += all.substr(8 + hl);
    const std::string plain = (root / "plain2.r4dx").string();
    std::ofstream(plain, std::ios::binary).write(out.data(), static_cast<std::streamsize>(out.size()));
    auto w = std::make_unique<r4dx::model::DflashDraftWeights>(r4dx::model::DflashDraftWeights::Open(plain));
    const auto& c = w->Config();
    Check(c.logit_softcap == 0.0 && c.embed_scale == 1.0 && c.variant.empty(),
          "loader: a container without the optional keys parses to softcap 0 / embed_scale 1 / variant ''");
    w.reset();
  }
  fs::remove_all(root);
  std::printf("%s\n", g_fail == 0 ? "PASS" : "FAILED");
  return g_fail == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
  return 1;
}
