// tests/model/test_inkling2_config.cpp -- CPU-only (no GPU, no container, no HIP). docs/inkling2-plan.md 5.
//
//   * Inkling2Config::FromModelConfig on the real Inkling-2-Mini config.json (tests/convert/data/inkling2_real_config.json,
//     committed; and the model directory's own copy, E:\models\OpenUncensored\Inkling-2-Mini-Uncensored\config.json, when
//     present) and on the tiny fixture's (tests/convert/data/inkling2_tiny/config.json): every field, the layer kinds,
//     the derived shapes, and every refusal;
//   * the visit table: 48 visits for the real model in HF's order, phases / passes / loop positions, the cache slots
//     (HF's layer_idx + cache_offset), each slot owned by exactly one visit; the tiny fixture's 32;
//   * Shard(2, r): the per-rank heads / kv heads / expert width / head rows, and its refusals;
//   * ResolveMaxCtx (131072 default, explicit up to 1048576) and ResolveContainerMaxCtx on a written header;
//   * ToModelConfig carries kInkling2Loop and the logits width; DetectArchFromMetadata / DetectArch for inkling2_loop
//     (model_arch, model_type, architectures), and the other arches unchanged.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "arch.h"
#include "inkling2_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"

using nlohmann::json;
using r4dx::model::Arch;
using r4dx::model::Inkling2Config;
using r4dx::model::Inkling2Visit;
namespace fs = std::filesystem;

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
    if (!ok) std::fprintf(stderr, "  (threw, but: %s)\n", e.what());
    return ok;
  }
  return false;
}
json ReadJson(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  return json::parse(f);
}

const fs::path kData = fs::path(R4DX_SOURCE_DIR) / "tests" / "convert" / "data";

void CheckReal(const json& j, const std::string& tag) {
  const Inkling2Config c = Inkling2Config::FromModelConfig(j);
  Check(c.hidden_size == 2304 && c.num_hidden_layers == 24, tag + ": hidden 2304, 24 stored layers");
  Check(c.num_attention_heads == 24 && c.num_key_value_heads == 8 && c.head_dim == 128 && c.Gqa() == 3,
        tag + ": 24 q heads, 8 kv heads, head_dim 128, GQA 3");
  Check(c.d_rel == 16 && c.rel_extent == 1024 && c.sliding_window == 1024, tag + ": d_rel 16, rel_extent 1024, window 1024");
  Check(c.vocab_size == 201024 && c.unpadded_vocab_size == 200058 && c.HeadRowsStored() == 200064,
        tag + ": vocab 201024 embedding rows, 200058 logits, 200064 stored head rows");
  Check(c.num_experts == 128 && c.num_experts_per_tok == 8 && c.moe_intermediate_size == 1152 && c.route_scale == 8.0,
        tag + ": 128 experts top-8, intermediate 1152, route_scale 8");
  Check(c.loop_start == 6 && c.loop_end == 18 && c.loop_passes == 3 && c.gate_rank == 64 && c.LoopLayers() == 12,
        tag + ": loop [6, 18) x 3, gate rank 64");
  Check(c.attn_res_block == 6 && c.final_logit_softcap == 30.0 && c.rms_norm_eps == 1e-6, tag + ": block 6, softcap 30, eps 1e-6");
  Check(std::fabs(c.qk_clip - 22.627416997969522) < 1e-12 && c.rel_clip == 8.0 && c.log_alpha == 0.1 && c.log_floor == 8192.0,
        tag + ": clips 22.63 / 8, log scaling 0.1 / 8192");
  bool kinds = true;
  for (int64_t i = 0; i < 24; ++i) kinds = kinds && c.IsFull(i) == ((i + 1) % 6 == 0);
  Check(kinds, tag + ": full attention exactly at (i + 1) % 6 == 0 (5, 11, 17, 23), from local_layer_ids");
  Check(c.IsBoundary(0) && c.IsBoundary(6) && c.IsBoundary(12) && c.IsBoundary(18) && !c.IsBoundary(5),
        tag + ": attention-residual boundaries 0, 6, 12, 18");
  Check(c.eos_token_id == 200006, tag + ": eos 200006 (top-level config)");
  Check(c.QDim() == 3072 && c.KvDim() == 1024 && c.RelExtent(5) == 1024 && c.Window(5) == 0 && c.Window(4) == 1024,
        tag + ": QDim 3072, KvDim 1024, full layers windowless");

  const std::vector<Inkling2Visit> v = c.Visits();
  Check(v.size() == 48 && c.NumCacheSlots() == 48, tag + ": 48 visits, 48 cache slots");
  bool order = v.size() == 48;
  for (int i = 0; order && i < 6; ++i) order = v[i].layer == i && v[i].phase == Inkling2Visit::Phase::kPre && v[i].cache_slot == i;
  for (int p = 0; order && p < 3; ++p)
    for (int j = 0; order && j < 12; ++j) {
      const Inkling2Visit& x = v[static_cast<size_t>(6 + 12 * p + j)];
      const int slot = p == 0 ? 6 + j : 24 + 12 * (p - 1) + j;
      order = x.layer == 6 + j && x.phase == Inkling2Visit::Phase::kLoop && x.pass == p && x.loop_j == j && x.cache_slot == slot &&
              x.HasMessages() == (p >= 1);
    }
  for (int i = 0; order && i < 6; ++i) {
    const Inkling2Visit& x = v[static_cast<size_t>(42 + i)];
    order = x.layer == 18 + i && x.phase == Inkling2Visit::Phase::kPost && x.cache_slot == 18 + i && !x.HasMessages();
  }
  Check(order, tag + ": visit order pre 0-5, three passes over 6-17 (slots 6-17, 24-35, 36-47), post 18-23");
  std::set<int> slots;
  bool idx = true;
  for (size_t i = 0; i < v.size(); ++i) {
    slots.insert(v[i].cache_slot);
    idx = idx && v[i].index == static_cast<int>(i);
  }
  Check(slots.size() == 48 && *slots.begin() == 0 && *slots.rbegin() == 47 && idx, tag + ": every slot owned by exactly one visit");
  Check(v[17].layer == 17 && v[17].pass == 0 && v[29].layer == 17 && v[29].pass == 1 && v[41].layer == 17 && v[41].pass == 2,
        tag + ": visits 17 / 29 / 41 end passes 0 / 1 / 2 (layer 17)");

  // TP=2
  const Inkling2Config r0 = Inkling2Config::Shard(c, 2, 0), r1 = Inkling2Config::Shard(c, 2, 1);
  Check(r0.num_attention_heads == 12 && r0.num_key_value_heads == 4 && r0.Gqa() == 3 && r0.moe_intermediate_size == 576,
        tag + ": TP=2 rank: 12 q heads, 4 kv heads (GQA 3), expert intermediate 576");
  Check(r0.HeadRowsLocal() == 100032 && r0.HeadRowBegin() == 0 && r1.HeadRowBegin() == 100032 && r1.tp_rank == 1,
        tag + ": TP=2 head rows 100032 per rank (rank 1 from 100032)");
  Check(r0.QDim() == 1536 && r0.QDim() % 32 == 0 && r0.KvDim() == 512 && r0.moe_intermediate_size % 32 == 0 &&
            r0.HeadRowsLocal() % 16 == 0 && (2 * r0.moe_intermediate_size) % 16 == 0,
        tag + ": TP=2 mxfp8 geometry: o / down K multiples of 32, q / kv / gate_up / head rows multiples of 16");
  Check(r0.hidden_size == 2304 && r0.num_experts == 128 && r0.unpadded_vocab_size == 200058 && r0.Visits().size() == 48,
        tag + ": TP=2: the replicated sizes stay global");
  Check(Throws([&] { Inkling2Config::Shard(r0, 2, 0); }, "already a rank shard"), tag + ": Shard of a shard refused");
  Check(Throws([&] { Inkling2Config::Shard(c, 3, 0); }, "world must be 1 or 2"), tag + ": world 3 refused");
  Check(Throws([&] { Inkling2Config::Shard(c, 2, 2); }, "rank"), tag + ": rank 2 refused");

  const r4dx::model::ModelConfig m = c.ToModelConfig();
  Check(m.arch == Arch::kInkling2Loop && m.vocab_size == 200058 && m.hidden_size == 2304 && m.num_hidden_layers == 24 &&
            m.layer_types.size() == 24 && m.layer_types[5] == "full_attention" && m.layer_types[4] == "sliding_attention" &&
            !m.IsGdnLayer(0) && m.num_experts == 128 && m.mtp_num_hidden_layers == 0,
        tag + ": ToModelConfig: kInkling2Loop, logits width 200058, layer kinds, no GDN, no MTP");
}

void CheckTiny(const json& j) {
  const Inkling2Config c = Inkling2Config::FromModelConfig(j);
  Check(c.hidden_size == 64 && c.num_hidden_layers == 16 && c.num_attention_heads == 4 && c.num_key_value_heads == 2 &&
            c.head_dim == 16 && c.d_rel == 4 && c.rel_extent == 8 && c.sliding_window == 8,
        "tiny: hidden 64, 16 layers, 4 / 2 heads of 16, d_rel 4, extent 8, window 8");
  Check(c.vocab_size == 112 && c.unpadded_vocab_size == 90 && c.HeadRowsStored() == 96, "tiny: vocab 112, 90 logits, 96 head rows");
  Check(c.num_experts == 8 && c.num_experts_per_tok == 2 && c.moe_intermediate_size == 32 && c.gate_rank == 8,
        "tiny: 8 experts top-2, intermediate 32, gate rank 8");
  bool kinds = true;
  for (int64_t i = 0; i < 16; ++i) kinds = kinds && c.IsFull(i) == ((i + 1) % 4 == 0);
  Check(kinds, "tiny: full layers at (i + 1) % 4 == 0 (layer_types and local_layer_ids agree)");
  Check(c.attn_res_block == 4 && c.loop_start == 4 && c.loop_end == 12 && c.loop_passes == 3 && c.log_floor == 16.0,
        "tiny: block 4, loop [4, 12) x 3, log floor 16");
  const std::vector<Inkling2Visit> v = c.Visits();
  Check(v.size() == 32 && c.NumCacheSlots() == 32, "tiny: 4 + 3 x 8 + 4 = 32 visits / slots");
  Check(v[4].layer == 4 && v[4].cache_slot == 4 && v[12].layer == 4 && v[12].pass == 1 && v[12].cache_slot == 16 &&
            v[20].layer == 4 && v[20].pass == 2 && v[20].cache_slot == 24 && v[27].layer == 11 && v[27].cache_slot == 31 &&
            v[28].layer == 12 && v[28].cache_slot == 12 && v[28].phase == Inkling2Visit::Phase::kPost,
        "tiny: pass 1 slots 16-23, pass 2 slots 24-31, post layers keep their own slot");
  const Inkling2Config s = Inkling2Config::Shard(c, 2, 1);
  Check(s.num_attention_heads == 2 && s.num_key_value_heads == 1 && s.moe_intermediate_size == 16 && s.HeadRowsLocal() == 48 &&
            s.HeadRowBegin() == 48,
        "tiny TP=2: 2 / 1 heads, expert intermediate 16, head rows 48 per rank");
}

void CheckRefusals(const json& real) {
  const auto with = [&](const std::function<void(json&)>& f) {
    json j = real;
    f(j.at("text_config"));
    return j;
  };
  const auto refuses = [&](const std::function<void(json&)>& f, const std::string& sub, const std::string& what) {
    const json j = with(f);
    Check(Throws([&] { (void)Inkling2Config::FromModelConfig(j); }, sub), "refused: " + what);
  };
  refuses([](json& t) { t.erase("hidden_size"); }, "hidden_size", "a missing hidden_size");
  refuses([](json& t) { t["swa_head_dim"] = 64; }, "swa_", "swa_* geometry different from the full one");
  refuses([](json& t) { t["num_key_value_heads"] = 7; t["swa_num_key_value_heads"] = 7; }, "multiple of num_key_value_heads",
          "heads not a multiple of kv heads");
  refuses([](json& t) { t["sconv_kernel_size"] = 3; }, "4 taps", "a 3-tap short conv");
  refuses([](json& t) { t["n_shared_experts"] = 1; }, "n_shared_experts", "a shared expert");
  refuses([](json& t) { t["loop_config"]["core_start"] = 5; }, "loop bounds", "loop bounds off the block grid");
  refuses([](json& t) { t["loop_config"]["core_end"] = 30; }, "loop bounds", "loop end past the layers");
  refuses([](json& t) { t["loop_config"]["is_varloop"] = true; }, "is_varloop", "a variable loop");
  refuses([](json& t) { t["mlp_up_softcap"] = 20.0; }, "softcap", "a different up softcap");
  refuses([](json& t) { t["unpadded_vocab_size"] = 300000; }, "unpadded_vocab_size", "unpadded vocab above vocab");
  refuses([](json& t) { t["n_routed_experts"] = 512; }, "router", "512 experts (the router kernel takes <= 256)");
  refuses([](json& t) { t["layer_types"] = std::vector<std::string>(24, "hybrid_sliding"); }, "disagree",
          "layer_types disagreeing with local_layer_ids");
  refuses([](json& t) { t["local_layer_ids"] = std::vector<int>{0, 99}; }, "out of range", "a local layer id out of range");
  {
    json j = real;
    j["model_type"] = "llama";
    Check(Throws([&] { (void)Inkling2Config::FromModelConfig(j); }, "inkling2_loop"), "refused: a top-level model_type other than inkling2_loop");
  }
  {
    json j = with([](json& t) { t.erase("local_layer_ids"); });
    const Inkling2Config c = Inkling2Config::FromModelConfig(j);
    Check(c.IsFull(5) && c.IsFull(23) && !c.IsFull(4), "no layer list: HF's default (i + 1) % 6 == 0");
  }
}

void CheckMaxCtxAndArch() {
  Check(Inkling2Config::ResolveMaxCtx(0) == 131072 && Inkling2Config::ResolveMaxCtx(-1) == 131072, "max_ctx default 131072");
  Check(Inkling2Config::ResolveMaxCtx(8192) == 8192 && Inkling2Config::ResolveMaxCtx(1048576) == 1048576,
        "explicit max_ctx up to 1048576 kept");
  Check(Throws([] { (void)Inkling2Config::ResolveMaxCtx(1048577); }, "1048576"), "max_ctx above 1048576 refused");

  const auto det = [](const json& m) { return r4dx::model::DetectArchFromMetadata(m); };
  Check(det(json{{"model_arch", "inkling2_loop"}}) == Arch::kInkling2Loop, "DetectArch: model_arch inkling2_loop");
  Check(det(json{{"model_config", {{"model_type", "inkling2_loop"}}}}) == Arch::kInkling2Loop, "DetectArch: model_type inkling2_loop");
  Check(det(json{{"model_config", {{"architectures", {"Inkling2LoopForConditionalGeneration"}}}}}) == Arch::kInkling2Loop,
        "DetectArch: architectures Inkling2LoopForConditionalGeneration");
  Check(std::string(r4dx::model::ArchName(Arch::kInkling2Loop)) == "inkling2_loop", "ArchName(kInkling2Loop) == inkling2_loop");
  // every arch's name round-trips, and the older detections are unchanged
  bool rt = true;
  for (Arch a : {Arch::kQwen35, Arch::kGemma4, Arch::kLfm2Moe, Arch::kQwen4Exp, Arch::kQwen35Moe, Arch::kLlama, Arch::kInkling2Loop})
    rt = rt && det(json{{"model_arch", r4dx::model::ArchName(a)}}) == a;
  Check(rt, "every ArchName round-trips through model_arch");
  Check(det(json{{"model_config", {{"model_type", "llama"}}}}) == Arch::kLlama &&
            det(json{{"model_config", {{"model_type", "lfm2_moe"}}}}) == Arch::kLfm2Moe && det(json::object()) == Arch::kQwen35,
        "llama / lfm2_moe / no-metadata detection unchanged");
  Check(Throws([&] { (void)det(json{{"model_arch", "inkling3"}}); }, "inkling2_loop"), "an unknown model_arch is refused (and lists inkling2_loop)");

  // ResolveContainerMaxCtx on a written safetensors-shell header (no tensors)
  const fs::path p = fs::temp_directory_path() / "r4dx_test_inkling2_config.r4dx";
  {
    const std::string hdr = json{{"__metadata__", {{"model_arch", "inkling2_loop"}}}}.dump();
    std::ofstream f(p, std::ios::binary);
    const uint64_t n = hdr.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(hdr.data(), static_cast<std::streamsize>(n));
  }
  Check(r4dx::model::DetectArch(p.string()) == Arch::kInkling2Loop, "DetectArch on a container header");
  Check(r4dx::model::ResolveContainerMaxCtx(p.string(), Arch::kInkling2Loop, false, 262144, false) == 131072,
        "ResolveContainerMaxCtx: no --max-ctx -> 131072");
  Check(r4dx::model::ResolveContainerMaxCtx(p.string(), Arch::kInkling2Loop, true, 524288, false) == 524288,
        "ResolveContainerMaxCtx: --max-ctx 524288 kept");
  Check(Throws([&] { (void)r4dx::model::ResolveContainerMaxCtx(p.string(), Arch::kInkling2Loop, true, 2000000, true); }),
        "ResolveContainerMaxCtx: --max-ctx 2000000 refused (even with --extended-ctx)");
  std::error_code ec;
  fs::remove(p, ec);
}

}  // namespace

int main() {
  try {
    const json real = ReadJson(kData / "inkling2_real_config.json");
    CheckReal(real, "real (committed copy)");
    const fs::path model_dir_cfg = "E:/models/OpenUncensored/Inkling-2-Mini-Uncensored/config.json";
    if (fs::exists(model_dir_cfg)) {
      CheckReal(ReadJson(model_dir_cfg), "real (model directory)");
    } else {
      std::printf("SKIP: %s not present\n", model_dir_cfg.string().c_str());
    }
    CheckTiny(ReadJson(kData / "inkling2_tiny" / "config.json"));
    CheckRefusals(real);
    CheckMaxCtxAndArch();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
