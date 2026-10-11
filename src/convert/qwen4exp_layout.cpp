#include "qwen4exp_layout.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

#include "r4dx_convert/safetensors_reader.hpp"  // Utf8ToWide

namespace r4dx_convert::qwen4exp {

namespace {

const std::string kTextPrefix = "model.language_model.";
const std::string kLayerPrefix = kTextPrefix + "layers.";
const std::string kWho = "r4dx-convert (qwen4_exp): ";

std::string ShapeStr(const std::vector<int64_t>& s) { return nlohmann::json(s).dump(); }

// The layer index of "model.language_model.layers.<i>.<rest>", or -1.
int LayerOf(const std::string& name) {
  if (name.compare(0, kLayerPrefix.size(), kLayerPrefix) != 0) return -1;
  size_t i = kLayerPrefix.size();
  int v = 0;
  bool any = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    v = v * 10 + (name[i] - '0');
    any = true;
    ++i;
  }
  return any && i < name.size() && name[i] == '.' ? v : -1;
}

bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

const nlohmann::json& Req(const nlohmann::json& j, const char* key, const char* where) {
  if (!j.is_object() || !j.contains(key))
    throw std::runtime_error(kWho + where + " has no '" + key + "'");
  return j.at(key);
}

int ReqInt(const nlohmann::json& j, const char* key, const char* where) {
  const nlohmann::json& v = Req(j, key, where);
  if (!v.is_number_integer() || v.get<int64_t>() <= 0 || v.get<int64_t>() > 0x7fffffff)
    throw std::runtime_error(kWho + where + "." + key + " is " + v.dump() + ", expected a positive integer");
  return v.get<int>();
}

}  // namespace

bool TextShape::HasPle(int i) const { return std::find(ple_layers.begin(), ple_layers.end(), i) != ple_layers.end(); }

TextShape ParseTextConfig(const nlohmann::json& config) {
  const nlohmann::json& t = Req(config, "text_config", "config.json");
  const char* where = "text_config";
  TextShape s;
  s.hidden = ReqInt(t, "hidden_size", where);
  s.layers_total = ReqInt(t, "num_hidden_layers", where);
  s.vocab = ReqInt(t, "vocab_size", where);
  s.heads = ReqInt(t, "num_attention_heads", where);
  s.kv_heads = ReqInt(t, "num_key_value_heads", where);
  s.head_dim = ReqInt(t, "head_dim", where);
  s.num_experts = ReqInt(t, "num_experts", where);
  s.moe_inter = ReqInt(t, "moe_intermediate_size", where);
  s.shared_inter = ReqInt(t, "shared_expert_intermediate_size", where);
  s.hc_count = ReqInt(t, "hc_count", where);
  s.hc_lowrank = ReqInt(t, "hc_lowrank", where);
  s.linear_key_heads = ReqInt(t, "linear_num_key_heads", where);
  s.linear_value_heads = ReqInt(t, "linear_num_value_heads", where);
  s.linear_key_dim = ReqInt(t, "linear_key_head_dim", where);
  s.linear_value_dim = ReqInt(t, "linear_value_head_dim", where);
  s.conv_kernel = ReqInt(t, "linear_conv_kernel_dim", where);
  s.indexer_heads = ReqInt(t, "indexer_n_heads", where);
  s.indexer_kv_heads = ReqInt(t, "indexer_kv_heads", where);
  s.indexer_head_dim = ReqInt(t, "indexer_head_dim", where);
  s.ple_embed_dim = ReqInt(t, "ple_embed_dim", where);
  s.ple_conv_kernel = ReqInt(t, "ple_conv_kernel_size", where);
  for (const auto& lt : Req(t, "layer_types", where)) {
    const std::string v = lt.is_string() ? lt.get<std::string>() : lt.dump();
    if (v != "linear_attention" && v != "full_attention")
      throw std::runtime_error(kWho + "layer_types has '" + v + "', expected linear_attention or full_attention");
    s.layer_types.push_back(v);
  }
  if (static_cast<int>(s.layer_types.size()) != s.layers_total)
    throw std::runtime_error(kWho + "layer_types length " + std::to_string(s.layer_types.size()) +
                             " != num_hidden_layers " + std::to_string(s.layers_total));
  for (const auto& id : Req(t, "ple_layer_ids", where)) {
    if (!id.is_number_integer() || id.get<int64_t>() < 1 || id.get<int64_t>() > s.layers_total)
      throw std::runtime_error(kWho + "ple_layer_ids has " + id.dump() + ", expected 1-based layer numbers in [1, " +
                               std::to_string(s.layers_total) + "]");
    s.ple_layers.push_back(id.get<int>() - 1);
  }
  // The layout writes an untied head from lm_head.weight; a tied checkpoint has none.
  const nlohmann::json* tie = t.contains("tie_word_embeddings") ? &t.at("tie_word_embeddings")
                              : config.contains("tie_word_embeddings") ? &config.at("tie_word_embeddings")
                                                                       : nullptr;
  if (tie && tie->is_boolean() && tie->get<bool>())
    throw std::runtime_error(kWho + "tie_word_embeddings is true: the layout converts lm_head.weight, which a "
                                    "tied checkpoint does not have");
  return s;
}

std::unordered_set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers) {
  if (layers < 0 || layers > s.layers_total)
    throw std::runtime_error(kWho + "--layers " + std::to_string(layers) + " is outside [0, " +
                             std::to_string(s.layers_total) + "]");
  std::unordered_set<std::string> consumed;
  // Existence and shape against the config before any job is registered: a mis-guessed name or a
  // checkpoint that is not the config's is a one-second error, not a mid-conversion one.
  auto expect = [&](const std::string& name, const std::vector<int64_t>& want) {
    if (!kit.has(name)) throw std::runtime_error(kWho + "checkpoint has no tensor '" + name + "'");
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want)
      throw std::runtime_error(kWho + "'" + name + "' has shape " + ShapeStr(got) + ", config says " + ShapeStr(want));
  };
  auto bf16 = [&](const std::string& hf, const std::string& name, const std::vector<int64_t>& want) {
    expect(hf, want);
    consumed.insert(hf);
    kit.add_bf16(hf, name);
  };
  auto fp32 = [&](const std::string& hf, const std::string& name, const std::vector<int64_t>& want) {
    expect(hf, want);
    consumed.insert(hf);
    kit.add_fp32_widen(hf, name);
  };
  auto linear = [&](const std::vector<std::string>& hf_names, const std::string& base, LinearRole role,
                    const std::vector<int64_t>& want_each) {
    for (const std::string& n : hf_names) {
      expect(n, want_each);
      consumed.insert(n);
    }
    kit.add_linear(hf_names, base, role);
  };

  const int64_t H = s.hidden, W = s.StreamWidth();
  const LinearRole body = LinearRole::kBody, bf16_only = LinearRole::kBf16Only;
  for (int i = 0; i < layers; ++i) {
    const std::string hf = kLayerPrefix + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    const bool full = s.IsFull(i);

    // Gated residual: one block around the mixer, one around the MoE.
    for (const char* blk : {"attn", "mlp"}) {
      const std::string h = hf + blk + "_hyper_connection.";
      const std::string b = base + "hc." + blk + ".";
      bf16(h + "hc_norm.weight", b + "norm", {W});
      bf16(h + "input_mix_weight_down.weight", b + "down", {s.hc_lowrank, W});
      bf16(h + "input_mix_weight_up.weight", b + "up", {W, s.hc_lowrank});
      bf16(h + "block_inject_weight.weight", b + "inject", {s.hc_count, W});
    }

    if (full) {
      if (kit.has(hf + "linear_attn.in_proj_qkv.weight"))
        throw std::runtime_error(kWho + "layer " + std::to_string(i) +
                                 " is full_attention in the config but the checkpoint has linear_attn tensors");
      const int64_t q_rows = static_cast<int64_t>(s.heads) * s.head_dim, kv_rows = static_cast<int64_t>(s.kv_heads) * s.head_dim;
      // q_proj is HF's fused query + output gate: view(..., heads, 2 * head_dim).chunk(2, -1), i.e. per-head
      // [q | gate] rows -- the layout attn.qg has in Qwen3.5 -- so it is converted as is.
      linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", body, {2 * q_rows, H});
      linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", body, {kv_rows, H});
      linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", body, {kv_rows, H});
      linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", body, {H, q_rows});
      bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm", {s.head_dim});
      bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm", {s.head_dim});
      kit.add_descale(base + "attn.k_descale", s.kv_heads, i, "k");
      kit.add_descale(base + "attn.v_descale", s.kv_heads, i, "v");
      bf16(hf + "self_attn.indexer.index_qk_proj.weight", base + "attn.indexer.qk",
           {static_cast<int64_t>(s.indexer_heads + s.indexer_kv_heads) * s.indexer_head_dim, H});
      bf16(hf + "self_attn.indexer.q_layernorm.weight", base + "attn.indexer.q_norm", {s.indexer_head_dim});
      bf16(hf + "self_attn.indexer.k_layernorm.weight", base + "attn.indexer.k_norm", {s.indexer_head_dim});
    } else {
      if (kit.has(hf + "self_attn.q_proj.weight"))
        throw std::runtime_error(kWho + "layer " + std::to_string(i) +
                                 " is linear_attention in the config but the checkpoint has self_attn tensors");
      const int64_t qkv = s.QkvRows(), z = static_cast<int64_t>(s.linear_value_heads) * s.linear_value_dim;
      linear({hf + "linear_attn.in_proj_qkv.weight"}, base + "gdn.in_proj_qkv", body, {qkv, H});
      linear({hf + "linear_attn.in_proj_z.weight"}, base + "gdn.in_proj_z", body, {z, H});
      bf16(hf + "linear_attn.in_proj_b.weight", base + "gdn.in_proj_b", {s.linear_value_heads, H});
      bf16(hf + "linear_attn.in_proj_a.weight", base + "gdn.in_proj_a", {s.linear_value_heads, H});
      bf16(hf + "linear_attn.conv1d.weight", base + "gdn.conv1d_weight", {qkv, 1, s.conv_kernel});
      fp32(hf + "linear_attn.A_log", base + "gdn.A_log", {s.linear_value_heads});
      fp32(hf + "linear_attn.dt_bias", base + "gdn.dt_bias", {s.linear_value_heads});
      bf16(hf + "linear_attn.norm.weight", base + "gdn.norm_weight", {s.linear_value_dim});
      linear({hf + "linear_attn.out_proj.weight"}, base + "gdn.out_proj", body, {H, z});
    }

    if (s.HasPle(i)) {
      const std::string h = hf + "ple.";
      const std::string b = base + "ple.";
      bf16(h + "key_proj.weight", b + "key_proj", {W, s.ple_embed_dim});
      bf16(h + "value_proj.weight", b + "value_proj", {H, s.ple_embed_dim});
      bf16(h + "norm_key.weight", b + "norm_key", {W});
      bf16(h + "norm_query.weight", b + "norm_query", {W});
      bf16(h + "norm_conv.weight", b + "norm_conv", {W});
      bf16(h + "conv1d.weight", b + "conv1d", {W, 1, s.ple_conv_kernel});
    }

    // MoE: router and shared expert stay bf16 (the checkpoint's own `ignore` list); the routed experts
    // are per-expert linears (never HF's fused 3-D tensors) and gate+up is ONE two-part container linear.
    bf16(hf + "mlp.gate.weight", base + "moe.router", {s.num_experts, H});
    linear({hf + "mlp.shared_expert.gate_proj.weight"}, base + "moe.shared.gate", bf16_only, {s.shared_inter, H});
    linear({hf + "mlp.shared_expert.up_proj.weight"}, base + "moe.shared.up", bf16_only, {s.shared_inter, H});
    linear({hf + "mlp.shared_expert.down_proj.weight"}, base + "moe.shared.down", bf16_only, {H, s.shared_inter});
    bf16(hf + "mlp.shared_expert_gate.weight", base + "moe.shared_gate", {1, H});
    for (int e = 0; e < s.num_experts; ++e) {
      const std::string es = std::to_string(e);
      const std::string h = hf + "mlp.experts." + es + ".";
      const std::string b = base + "moe.experts." + es + ".";
      linear({h + "gate_proj.weight", h + "up_proj.weight"}, b + "gate_up", body, {s.moe_inter, H});
      linear({h + "down_proj.weight"}, b + "down", body, {H, s.moe_inter});
    }
  }

  // Outside the layer stack.
  bf16(kTextPrefix + "embed_tokens.weight", "text.embed_tokens", {s.vocab, H});
  bf16(kTextPrefix + "hyper_connection_mixer.hc_norm.weight", "text.hc_final.norm", {W});
  bf16(kTextPrefix + "hyper_connection_mixer.input_mix_weight_down.weight", "text.hc_final.down", {s.hc_lowrank, W});
  bf16(kTextPrefix + "hyper_connection_mixer.input_mix_weight_up.weight", "text.hc_final.up", {W, s.hc_lowrank});
  const LinearRole head = LinearRole::kHead;
  linear({"lm_head.weight"}, "lm_head", head, {s.vocab, H});
  return consumed;
}

bool IsSkipped(const std::string& hf_name) {
  return StartsWith(hf_name, "model.visual.") || StartsWith(hf_name, "mtp.") ||
         hf_name.find(".ple.ple_embedding.") != std::string::npos;
}

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::unordered_set<std::string>& consumed, int layers) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0 || IsSkipped(name)) continue;
    const int layer = LayerOf(name);
    if (layer >= layers) continue;  // --layers: not converted by request
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

void CheckExpertKbUniform(const std::unordered_map<std::string, int>& kb_by_base) {
  // key "<layer> <class>" -> every (base, kb) of the group
  std::map<std::pair<int, std::string>, std::vector<std::pair<std::string, int>>> groups;
  static const std::string marker = ".moe.experts.";
  for (const auto& [base, kb] : kb_by_base) {
    if (!StartsWith(base, "text.layers.")) continue;
    const size_t m = base.find(marker);
    if (m == std::string::npos) continue;
    const size_t dot = base.rfind('.');
    const std::string cls = base.substr(dot + 1);
    if (cls != "gate_up" && cls != "down") continue;
    const int layer = std::atoi(base.c_str() + std::string("text.layers.").size());
    groups[{layer, cls}].emplace_back(base, kb);
  }
  std::string msg;
  int bad_groups = 0;
  for (auto& [key, members] : groups) {
    std::sort(members.begin(), members.end());
    int first_kb = members.front().second;
    bool uniform = true;
    for (const auto& m : members) uniform = uniform && m.second == first_kb;
    if (uniform) continue;
    ++bad_groups;
    if (bad_groups > 6) continue;
    // One representative base per KB present.
    std::map<int, std::string> rep;
    for (const auto& m : members) rep.emplace(m.second, m.first);
    msg += "\n  layer " + std::to_string(key.first) + " " + key.second + ":";
    for (const auto& [kb, base] : rep)
      msg += std::string(" ") + (kb == 0 ? "not imported" : "KB " + std::to_string(kb)) + " (" + base + ")";
  }
  if (bad_groups > 0) {
    if (bad_groups > 6) msg += "\n  ... " + std::to_string(bad_groups - 6) + " more layer/class group(s)";
    throw std::runtime_error(kWho + "the experts of " + std::to_string(bad_groups) +
                             " (layer, class) group(s) disagree on the trellis KB; all experts of one layer and "
                             "class must share one KB (plan 4.11 item 3):" + msg);
  }
}

nlohmann::json NgramSidecar::ToJson() const {
  return {{"path", path},
          {"sha256", sha256},
          {"rows", rows},
          {"row_bytes", row_bytes},
          {"version", version},
          {"heads", heads},
          {"head_dim", head_dim},
          {"ngram_size", ngram_size},
          {"heads_per_ngram", heads_per_ngram},
          {"eos_token_id", eos_token_id},
          {"seed", seed},
          {"padded_rows", padded_rows},
          {"multipliers", multipliers},
          {"head_offsets", head_offsets},
          {"head_primes", head_primes}};
}

NgramSidecar ReadNgramSidecar(const std::string& path, const TextShape& s, const nlohmann::json& config) {
  namespace fs = std::filesystem;
  constexpr size_t kHeaderBytes = 4096;
  constexpr int kNgram = 3, kHeads = 16;
  std::error_code ec;
  const fs::path abs = fs::absolute(fs::u8path(path), ec).lexically_normal();
  if (ec) throw std::runtime_error(kWho + "--ngram-sidecar " + path + ": cannot resolve the path (" + ec.message() + ")");
  const std::string who = kWho + "--ngram-sidecar " + abs.u8string() + ": ";
  const uintmax_t file_size = fs::file_size(abs, ec);
  if (ec) throw std::runtime_error(who + "cannot stat the file (" + ec.message() + ")");
  std::ifstream f(r4dx_convert::Utf8ToWide(abs.u8string()).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error(who + "cannot open the file");
  std::vector<uint8_t> h(kHeaderBytes);
  f.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(h.size()));
  if (f.gcount() != static_cast<std::streamsize>(h.size()))
    throw std::runtime_error(who + "the file is shorter than its " + std::to_string(kHeaderBytes) + "-byte header");
  if (std::memcmp(h.data(), "R4DXNGRM", 8) != 0) throw std::runtime_error(who + "bad magic (not an r4dx n-gram sidecar)");

  // Little-endian, packed (tools/reference/qwen4exp/ngram_sidecar.py): magic[8] u32 version u64 rows
  // u32 row_bytes u32 heads u32 head_dim u32 ngram_size u32 heads_per_ngram i64 eos u64 seed i64 mult[3]
  // u64 offsets[16] u64 primes[16] u64 padded_rows u8 sha256[32], zero padding.
  size_t off = 8;
  auto get = [&](auto& v) {
    std::memcpy(&v, h.data() + off, sizeof(v));
    off += sizeof(v);
  };
  NgramSidecar n;
  get(n.version);
  get(n.rows);
  get(n.row_bytes);
  get(n.heads);
  get(n.head_dim);
  get(n.ngram_size);
  get(n.heads_per_ngram);
  get(n.eos_token_id);
  get(n.seed);
  n.multipliers.resize(kNgram);
  for (auto& m : n.multipliers) get(m);
  n.head_offsets.resize(kHeads);
  for (auto& o : n.head_offsets) get(o);
  n.head_primes.resize(kHeads);
  for (auto& p : n.head_primes) get(p);
  get(n.padded_rows);
  static const char* hex = "0123456789abcdef";
  for (int i = 0; i < 32; ++i) {
    n.sha256 += hex[h[off + static_cast<size_t>(i)] >> 4];
    n.sha256 += hex[h[off + static_cast<size_t>(i)] & 15];
  }
  off += 32;
  for (size_t i = off; i < kHeaderBytes; ++i)
    if (h[i] != 0) throw std::runtime_error(who + "the header padding is not zero");
  if (n.version != 1) throw std::runtime_error(who + "version " + std::to_string(n.version) + ", this converter reads 1");
  if (n.heads != static_cast<uint32_t>(kHeads) || n.ngram_size != static_cast<uint32_t>(kNgram))
    throw std::runtime_error(who + "the header has " + std::to_string(n.heads) + " heads / ngram_size " +
                             std::to_string(n.ngram_size) + ", the layout needs " + std::to_string(kHeads) + " / " +
                             std::to_string(kNgram));
  if (static_cast<int64_t>(n.row_bytes) * n.heads != 2LL * s.ple_embed_dim)
    throw std::runtime_error(who + "row_bytes " + std::to_string(n.row_bytes) + " x " + std::to_string(n.heads) +
                             " heads != 2 x ple_embed_dim " + std::to_string(s.ple_embed_dim) + " (a bf16 row per head)");
  const uint64_t want_size = kHeaderBytes + n.rows * static_cast<uint64_t>(n.row_bytes);
  if (file_size != want_size)
    throw std::runtime_error(who + "the file is " + std::to_string(file_size) + " bytes but the header's " +
                             std::to_string(n.rows) + " rows x " + std::to_string(n.row_bytes) + " B + " +
                             std::to_string(kHeaderBytes) + " = " + std::to_string(want_size) +
                             " (truncated or still being written)");
  const nlohmann::json& eos = config.contains("text_config") && config.at("text_config").contains("eos_token_id")
                                  ? config.at("text_config").at("eos_token_id")
                                  : config.value("eos_token_id", nlohmann::json(nullptr));
  if (eos.is_number_integer() && eos.get<int64_t>() != n.eos_token_id)
    throw std::runtime_error(who + "the sidecar's eos id " + std::to_string(n.eos_token_id) +
                             " != the config's eos_token_id " + eos.dump());
  n.path = abs.u8string();
  return n;
}

}  // namespace r4dx_convert::qwen4exp
