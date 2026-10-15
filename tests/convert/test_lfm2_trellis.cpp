// tests/convert/test_lfm2_trellis.cpp -- CPU-only (ctest `convert_lfm2_trellis`). docs/lfm2-plan.md L3-2:
// `r4dx-convert --trellis-from` on an lfm2_moe checkpoint. Everything is generated here (no Python, no fixture, no
// GPU): a tiny per-expert lfm2_moe checkpoint (4 layers = conv dense, conv dense, attention MoE, conv MoE; hidden
// 256, 4 experts of I 128, top-2 -- the geometry of test_lfm2_layout.cpp) and synthetic oracle directories in the
// form tools/reference/trellis_quant.py quantize-model writes (random ring words, fp16 scales; the checkpoint's
// weights are built FROM the reconstruction, so rel_weight_err is a real, uneven ~1%).
//
// The oracle manifest keys are the checkpoint's own per-expert names (`model.layers.3.feed_forward.experts.2.w1.weight`),
// which is what the converter hands TrellisSource, so no separate "expert suffix" key form exists (plan 4 sketched
// `experts.gate_up_proj[e]` for a fused checkpoint; the real one is per-expert).
//
//   (a) the oracle covers all 40 HF body tensors (4 attention, 6 conv in/out, 6 dense, 24 expert): every
//       `.trellis.w|suh|svh` of the 30 container bases (moe.experts.{e}.gate_up = w1 rows then w3 rows, .down = w2)
//       equals the pair-grid regrid of the oracle's words / the parts' scales, no other layout of a body linear, the
//       router / expert_bias / norms / conv taps / embedding / lm_head are untouched, the verify pass 40/40.
//   (b) refusals, each before an output file exists: one expert's w3 missing from the manifest (partial gate_up),
//       a whole expert missing, a stale layer the conversion uses, and --rotate with --trellis-from.
//   (c) the reconstruction check really keys experts apart: two experts' oracle tensors swapped under their names fail.
// r4dx-convert must be built (R4DX_CONVERT_EXE); without it the test SKIPs (77).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/trellis_import.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
namespace tr = r4dx_convert::trellis;
namespace rc = r4dx_convert;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool Contains(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

// ---- the tiny lfm2_moe checkpoint (every N and K a multiple of 128) ------------------------------------------------

constexpr int64_t kHidden = 256, kHeads = 4, kKv = 2, kHd = 64, kDenseI = 256, kMoeI = 128, kExperts = 4, kVocab = 64,
                  kConvL = 3;
constexpr int kLayers = 4, kDenseLayers = 2;
constexpr int kKb = 4;  // every oracle tensor at K = 4 (32 words per tile)

bool IsAttn(int i) { return i % 4 == 2; }
bool IsDense(int i) { return i < kDenseLayers; }

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

std::string L(int i) { return "model.layers." + std::to_string(i) + "."; }

struct Lin {
  std::string hf;    // model.layers.L.<module>.weight
  int layer = 0;
  int64_t n = 0, k = 0;
  std::string base;  // container base (text.layers.L.attn.q, moe.experts.E.gate_up, ...)
  int part = 0;      // position inside the base (0 = gate / w1, 1 = up / w3)
};

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<float> v;
};

struct Ckpt {
  std::vector<Tensor> t;
  std::map<std::string, size_t> idx;
  json config;
  std::vector<Lin> lins;  // the trellis-covered HF tensors in converter order
  void Put(const std::string& name, const std::string& dtype, std::vector<int64_t> shape, std::vector<float> v) {
    idx[name] = t.size();
    t.push_back({name, dtype, std::move(shape), std::move(v)});
  }
  std::vector<float>& V(const std::string& name) { return t[idx.at(name)].v; }
  const std::vector<float>& V(const std::string& name) const { return t[idx.at(name)].v; }
};

Ckpt MakeSkeleton() {
  Ckpt c;
  std::mt19937_64 rng(0x1F42Aull);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto rnd = [&](int64_t n, double mu, double sigma) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(mu + sigma * nd(rng)));
    return v;
  };
  auto lin = [&](int i, const std::string& mod, int64_t n, int64_t k, const std::string& base, int part = 0) {
    c.lins.push_back({L(i) + mod + ".weight", i, n, k, "text.layers." + std::to_string(i) + "." + base, part});
  };
  for (int i = 0; i < kLayers; ++i) {
    c.Put(L(i) + "operator_norm.weight", "BF16", {kHidden}, rnd(kHidden, 1.0, 0.2));
    c.Put(L(i) + "ffn_norm.weight", "BF16", {kHidden}, rnd(kHidden, 1.0, 0.2));
    if (IsAttn(i)) {
      lin(i, "self_attn.q_proj", kHeads * kHd, kHidden, "attn.q");
      lin(i, "self_attn.k_proj", kKv * kHd, kHidden, "attn.k");
      lin(i, "self_attn.v_proj", kKv * kHd, kHidden, "attn.v");
      lin(i, "self_attn.out_proj", kHidden, kHeads * kHd, "attn.o");
      c.Put(L(i) + "self_attn.q_layernorm.weight", "BF16", {kHd}, rnd(kHd, 1.0, 0.2));
      c.Put(L(i) + "self_attn.k_layernorm.weight", "BF16", {kHd}, rnd(kHd, 1.0, 0.2));
    } else {
      lin(i, "conv.in_proj", 3 * kHidden, kHidden, "conv.in_proj");
      c.Put(L(i) + "conv.conv.weight", "BF16", {kHidden, 1, kConvL}, rnd(kHidden * kConvL, 0.0, 0.3));
      lin(i, "conv.out_proj", kHidden, kHidden, "conv.out_proj");
    }
    if (IsDense(i)) {
      lin(i, "feed_forward.w1", kDenseI, kHidden, "mlp.gate_up", 0);
      lin(i, "feed_forward.w3", kDenseI, kHidden, "mlp.gate_up", 1);
      lin(i, "feed_forward.w2", kHidden, kDenseI, "mlp.down");
    } else {
      c.Put(L(i) + "feed_forward.gate.weight", "BF16", {kExperts, kHidden}, rnd(kExperts * kHidden, 0.0, 0.1));
      std::vector<float> eb(static_cast<size_t>(kExperts));
      for (auto& x : eb) x = static_cast<float>(0.1 * nd(rng));
      c.Put(L(i) + "feed_forward.expert_bias", "F32", {kExperts}, eb);
      for (int e = 0; e < kExperts; ++e) {
        const std::string ex = "feed_forward.experts." + std::to_string(e) + ".";
        const std::string eb2 = "moe.experts." + std::to_string(e) + ".";
        lin(i, ex + "w1", kMoeI, kHidden, eb2 + "gate_up", 0);
        lin(i, ex + "w3", kMoeI, kHidden, eb2 + "gate_up", 1);
        lin(i, ex + "w2", kHidden, kMoeI, eb2 + "down");
      }
    }
  }
  c.Put("model.embed_tokens.weight", "BF16", {kVocab, kHidden}, rnd(kVocab * kHidden, 0.0, 0.1));
  c.Put("model.embedding_norm.weight", "BF16", {kHidden}, rnd(kHidden, 1.0, 0.2));
  json types = json::array();
  for (int i = 0; i < kLayers; ++i) types.push_back(IsAttn(i) ? "full_attention" : "conv");
  c.config = {{"architectures", {"Lfm2MoeForCausalLM"}}, {"bos_token_id", 1}, {"conv_L_cache", kConvL},
              {"conv_bias", false}, {"dtype", "bfloat16"}, {"eos_token_id", 2}, {"hidden_size", kHidden},
              {"intermediate_size", kDenseI}, {"layer_types", types}, {"max_position_embeddings", 4096},
              {"model_type", "lfm2_moe"}, {"moe_intermediate_size", kMoeI}, {"norm_eps", 1e-05},
              {"norm_topk_prob", true}, {"num_attention_heads", kHeads}, {"num_dense_layers", kDenseLayers},
              {"num_experts", kExperts}, {"num_experts_per_tok", 2}, {"num_hidden_layers", kLayers},
              {"num_key_value_heads", kKv}, {"pad_token_id", 0},
              {"rope_parameters", {{"rope_theta", 10000000.0}, {"rope_type", "default"}}},
              {"routed_scaling_factor", 1.0}, {"tie_word_embeddings", true}, {"use_cache", true},
              {"use_expert_bias", true}, {"vocab_size", kVocab}};
  return c;
}

void WriteCkpt(const Ckpt& c, const fs::path& dir) {
  fs::create_directories(dir);
  json hdr = json::object();
  std::string data;
  for (const Tensor& t : c.t) {
    const uint64_t b = data.size();
    for (float x : t.v) {
      if (t.dtype == "F32") {
        data.append(reinterpret_cast<const char*>(&x), 4);
      } else {
        const uint16_t h = r4dx::core::FloatToBf16(x);
        data.append(reinterpret_cast<const char*>(&h), 2);
      }
    }
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {b, data.size()}}};
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);
  WriteWhole(dir / "config.json", c.config.dump(2));
}

// ---- the oracle: synthetic ring words, the reconstruction, the files -------------------------------------------

struct Oracle {
  std::vector<uint32_t> words;  // [k/16][n/16][8 KB]
  std::vector<uint16_t> suh, svh;
  std::vector<float> w_hat;     // [n][k] reconstruction (what the converter's check decodes)
};

// tools/reference/trellis_quant.py's reconstruct for one tensor, as TrellisSource::Verify decodes it.
std::vector<float> Reconstruct(const std::vector<uint32_t>& words, const std::vector<uint16_t>& suh,
                               const std::vector<uint16_t>& svh, int64_t n, int64_t k) {
  const tr::RingTables& rt = tr::Ring(kKb);
  std::vector<float> out(static_cast<size_t>(n * k));
  std::vector<float> blk(static_cast<size_t>(tr::kHad * tr::kHad));
  for (int64_t row0 = 0; row0 < n; row0 += tr::kHad) {
    float sv[tr::kHad];
    for (int64_t row = 0; row < tr::kHad; ++row)
      sv[row] = r4dx::core::F16ToFloat(svh[static_cast<size_t>(row0 + row)]) * (1.0f / 128.0f);
    for (int64_t k0 = 0; k0 < k; k0 += tr::kHad) {
      float vals[tr::kTile];
      for (int64_t t8 = 0; t8 < tr::kHad / 16; ++t8)
        for (int64_t q8 = 0; q8 < tr::kHad / 16; ++q8) {
          const int64_t tn = row0 / 16 + t8, tk = k0 / 16 + q8;
          tr::DecodeTile(words.data() + (tk * (n / 16) + tn) * tr::WordsPerTile(kKb), rt, vals);
          for (int cc = 0; cc < 16; ++cc)
            for (int rr = 0; rr < 16; ++rr)
              blk[static_cast<size_t>((t8 * 16 + cc) * tr::kHad + q8 * 16 + rr)] = vals[rr * 16 + cc];
        }
      for (int64_t row = 0; row < tr::kHad; ++row) tr::Fwht128(blk.data() + row * tr::kHad);
      tr::Fwht128Rows(blk.data(), tr::kHad, tr::kHad);
      for (int64_t row = 0; row < tr::kHad; ++row)
        for (int64_t kk = 0; kk < tr::kHad; ++kk) {
          const float su = r4dx::core::F16ToFloat(suh[static_cast<size_t>(k0 + kk)]);
          out[static_cast<size_t>((row0 + row) * k + k0 + kk)] =
              blk[static_cast<size_t>(row * tr::kHad + kk)] * su * sv[row];
        }
    }
  }
  return out;
}

Oracle MakeOracle(size_t idx, int64_t n, int64_t k) {
  std::mt19937_64 rng(0xC0DEull + idx);
  Oracle o;
  o.words.resize(static_cast<size_t>(k / 16 * (n / 16) * tr::WordsPerTile(kKb)));
  for (auto& w : o.words) w = static_cast<uint32_t>(rng());
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto sgn = [&]() { return (rng() & 1) ? 1.0 : -1.0; };
  for (int64_t i = 0; i < k; ++i)
    o.suh.push_back(r4dx::core::FloatToF16(static_cast<float>(sgn() * (0.5 + u(rng)))));
  for (int64_t i = 0; i < n; ++i)
    o.svh.push_back(r4dx::core::FloatToF16(static_cast<float>(sgn() * (0.005 + 0.015 * u(rng)))));
  o.w_hat = Reconstruct(o.words, o.suh, o.svh, n, k);
  return o;
}

// W_hat * (1 + a g), a per 16 x 16 tile in [0.002, 0.02]: an uneven ~1% error, so a reconstruction check that skipped a
// block or read another tensor cannot reproduce rel_weight_err.
std::vector<float> Perturb(const std::vector<float>& w_hat, int64_t n, int64_t k, size_t idx) {
  std::mt19937_64 rng(0xFEEDull + idx);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ua(0.002, 0.02);
  std::vector<double> amp(static_cast<size_t>((n / 16) * (k / 16)));
  for (auto& a : amp) a = ua(rng);
  std::vector<float> out(w_hat.size());
  for (int64_t r = 0; r < n; ++r)
    for (int64_t c = 0; c < k; ++c)
      out[static_cast<size_t>(r * k + c)] =
          static_cast<float>(w_hat[static_cast<size_t>(r * k + c)] *
                             (1.0 + amp[static_cast<size_t>((r / 16) * (k / 16) + c / 16)] * nd(rng)));
  return out;
}

double RelErr(const std::vector<float>& w_hat, const std::vector<float>& w) {
  double e2 = 0.0, w2 = 0.0;
  for (size_t i = 0; i < w.size(); ++i) {
    const double d = static_cast<double>(w_hat[i]) - w[i];
    e2 += d * d;
    w2 += static_cast<double>(w[i]) * w[i];
  }
  return std::sqrt(e2 / w2);
}

std::string Sha(const std::string& bytes) { return rc::Sha256Hex(bytes); }

void WriteSafetensors(const fs::path& path,
                      const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::string>>& ts,
                      const json& meta) {
  json hdr = json::object();
  std::string data;
  for (const auto& [name, dtype, shape, raw] : ts) {
    hdr[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {data.size(), data.size() + raw.size()}}};
    data += raw;
  }
  hdr["__metadata__"] = meta;
  std::string h = hdr.dump();
  h.append((8 - h.size() % 8) % 8, ' ');
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(path, file);
}

struct Fixture {
  Ckpt ckpt;
  std::map<std::string, Oracle> oracle;  // hf -> oracle tensors
  std::map<std::string, double> rel;     // hf -> rel_weight_err against the bf16 checkpoint weight
  fs::path ckpt_dir, oracle_dir;
};

Fixture MakeFixture(const fs::path& root, const std::string& tag) {
  Fixture f;
  f.ckpt = MakeSkeleton();
  for (size_t i = 0; i < f.ckpt.lins.size(); ++i) {
    const Lin& l = f.ckpt.lins[i];
    Oracle o = MakeOracle(i, l.n, l.k);
    std::vector<float> w = Perturb(o.w_hat, l.n, l.k, i);
    for (auto& x : w) x = Bf16Round(x);
    f.rel[l.hf] = RelErr(o.w_hat, w);
    f.ckpt.Put(l.hf, "BF16", {l.n, l.k}, std::move(w));
    f.oracle[l.hf] = std::move(o);
  }
  f.ckpt_dir = root / ("ckpt-" + tag);
  f.oracle_dir = root / ("oracle-" + tag);
  WriteCkpt(f.ckpt, f.ckpt_dir);
  return f;
}

// Writes the oracle directory (quantize-model form): one L<ii>.safetensors per layer + weights_override.json.
// `drop`: HF names left out of the manifest and files; `swap`: two HF names whose oracle tensors are exchanged (same
// shape) while the manifest's rel_weight_err stays that of the name; `stale`: stale_layers entries.
void WriteOracle(const Fixture& f, const std::set<std::string>& drop = {},
                 const std::pair<std::string, std::string>& swap = {}, const std::vector<std::string>& stale = {}) {
  fs::create_directories(f.oracle_dir);
  const std::string config_sha = Sha(ReadWhole(f.ckpt_dir / "config.json"));
  auto src_of = [&](const std::string& hf) {
    if (!swap.first.empty() && hf == swap.first) return swap.second;
    if (!swap.first.empty() && hf == swap.second) return swap.first;
    return hf;
  };
  json tensors = json::object();
  for (int layer = 0; layer < kLayers; ++layer) {
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::string>> ts;
    std::vector<const Lin*> in_file;
    for (const Lin& l : f.ckpt.lins) {
      if (l.layer != layer || drop.count(l.hf)) continue;
      const Oracle& o = f.oracle.at(src_of(l.hf));
      ts.emplace_back(l.hf + ".trellis", "I32", std::vector<int64_t>{l.k / 16, l.n / 16, tr::WordsPerTile(kKb)},
                      std::string(reinterpret_cast<const char*>(o.words.data()), o.words.size() * 4));
      ts.emplace_back(l.hf + ".suh", "F16", std::vector<int64_t>{l.k},
                      std::string(reinterpret_cast<const char*>(o.suh.data()), o.suh.size() * 2));
      ts.emplace_back(l.hf + ".svh", "F16", std::vector<int64_t>{l.n},
                      std::string(reinterpret_cast<const char*>(o.svh.data()), o.svh.size() * 2));
      in_file.push_back(&l);
    }
    if (ts.empty()) continue;
    char name[16];
    std::snprintf(name, sizeof(name), "L%02d.safetensors", layer);
    WriteSafetensors(f.oracle_dir / name, ts, {{"format", "r4dx-weights-override"}, {"layer", std::to_string(layer)}});
    const std::string fsha = Sha(ReadWhole(f.oracle_dir / name));
    for (const Lin* l : in_file)
      tensors[l->hf] = {{"encoding", "trellis-exl3"}, {"K", static_cast<double>(kKb)}, {"codebook", "mul1"},
                        {"k", l->k}, {"n", l->n}, {"shape_hf", {l->n, l->k}},
                        {"words_shape", {l->k / 16, l->n / 16, tr::WordsPerTile(kKb)}},
                        {"rel_weight_err", f.rel.at(l->hf)}, {"hessian_basis", "matched"},
                        {"file", name}, {"file_sha256", fsha}};
  }
  json man = {{"format", "r4dx-weights-override"}, {"version", 1}, {"encoding", "trellis-exl3"},
              {"complete", true}, {"missing", json::array()}, {"missing_count", 0},
              {"bpw_target", nullptr}, {"K_uniform", static_cast<double>(kKb)},
              {"layers_done", {0, 1, 2, 3}}, {"stale_layers", stale},
              {"model_dir", f.ckpt_dir.u8string()}, {"config_sha256", config_sha},
              {"hessian_dir", "(synthetic)"}, {"hessian_manifest_sha256", std::string(64, '2')},
              {"hessian_basis", "matched"}, {"codebook", "mul1"},
              {"recipe", {{"note", "synthetic: random ring words (tests/convert/test_lfm2_trellis.cpp)"}}},
              {"code_sha256", {{"trellis_quant", std::string(64, '0')}, {"trellis_viterbi", std::string(64, '1')}}},
              {"tensors", tensors}};
  WriteWhole(f.oracle_dir / "weights_override.json", man.dump(2));
}

// ---- running the exe and reading containers --------------------------------------------------------------------

#if defined(R4DX_CONVERT_EXE)

int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

struct Container {
  bool ok = false;
  json header;
  std::map<std::string, std::string> tensors;
  json Meta() const { return header.at("__metadata__"); }
  bool Has(const std::string& n) const { return tensors.count(n) != 0; }
};

Container ReadContainer(const fs::path& path) {
  Container c;
  std::string bytes;
  try {
    bytes = ReadWhole(path);
  } catch (const std::exception&) {
    return c;
  }
  if (bytes.size() < 8) return c;
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  if (8 + hl > bytes.size()) return c;
  c.header = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  const std::string data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    c.tensors[it.key()] = data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

std::string Q(const fs::path& p) { return "\"" + p.u8string() + "\""; }

void CheckTrellisContainer(const Container& c, const Fixture& f) {
  std::map<std::string, std::vector<const Lin*>> by_base;
  for (const Lin& l : f.ckpt.lins) by_base[l.base].push_back(&l);
  Check(f.ckpt.lins.size() == 40, "fixture: 40 HF body tensors");
  Check(by_base.size() == 30, "fixture: 30 body bases (4 attention + 6 conv + 4 dense + 16 expert)");
  bool bytes_ok = true, no_other_layout = true;
  for (const auto& [base, parts] : by_base) {
    std::vector<tr::OracleWords> ow;
    std::string suh, svh;
    int64_t N = 0;
    for (const Lin* l : parts) {  // push order = w1 then w3 (gate before up)
      const Oracle& o = f.oracle.at(l->hf);
      ow.push_back({reinterpret_cast<const uint8_t*>(o.words.data()), l->n});
      suh.append(reinterpret_cast<const char*>(o.suh.data()), o.suh.size() * 2);
      svh.append(reinterpret_cast<const char*>(o.svh.data()), o.svh.size() * 2);
      N += l->n;
    }
    const int64_t K = parts[0]->k;
    std::vector<uint32_t> grid(static_cast<size_t>(N * K * kKb / 32));
    tr::RegridToPairGrid(ow, K, kKb, grid.data(), 2);
    const std::string want_w(reinterpret_cast<const char*>(grid.data()), grid.size() * 4);
    bytes_ok = bytes_ok && c.Has(base + ".trellis.w") && c.tensors.at(base + ".trellis.w") == want_w &&
               c.Has(base + ".trellis.suh") && c.Has(base + ".trellis.svh") &&
               c.tensors.at(base + ".trellis.svh") == svh && c.tensors.at(base + ".trellis.suh") == suh;
    for (const char* other : {".bf16.w", ".w4a16.wq", ".w4a16.wsz"})
      no_other_layout = no_other_layout && !c.Has(base + other);
  }
  Check(bytes_ok, "every .trellis.w / .svh is the oracle's regrid / the parts' scales (w1 then w3)");
  Check(no_other_layout, "no .bf16.w / .w4a16.* next to a trellis linear");
  Check(c.Has("text.layers.3.moe.experts.2.gate_up.trellis.w") && c.Has("text.layers.3.moe.experts.2.down.trellis.w") &&
            c.Has("text.layers.2.moe.experts.0.gate_up.trellis.w"),
        "expert tensors are keyed moe.experts.{e}.gate_up / .down");
  Check(c.Has("text.layers.2.moe.router"),
        "the router stays a plain bf16 tensor");
  Check(c.Has("text.layers.2.moe.expert_bias") && !c.Has("text.layers.2.moe.router.trellis.w"),
        "expert_bias present, the router is not trellis");
  Check(c.Has("text.layers.0.conv.weight") && c.Has("text.embed_tokens") && c.Has("lm_head.bf16.w") &&
            !c.Has("lm_head.trellis.w"),
        "conv taps / embedding stay bf16; lm_head is not a body linear (not trellis)");
  const json q = c.Meta().at("quant").at("trellis");
  std::set<std::string> in_meta, want_meta;
  for (auto it = q.at("linears").begin(); it != q.at("linears").end(); ++it) in_meta.insert(it.key());
  for (const auto& kv : by_base) want_meta.insert(kv.first);
  Check(q.value("format", "") == "r4dx-trellis" && in_meta == want_meta &&
            q.at("linears").at("text.layers.3.moe.experts.1.gate_up").at("parts") == json::array({kMoeI, kMoeI}),
        "__metadata__.quant.trellis lists the 30 bases (an expert gate_up with parts [128, 128])");
  const json run = c.Meta().at("r4dx_convert_run").at("trellis");
  Check(run.value("hf_tensors", 0) == 40 && run.at("verify").value("result", "").rfind("pass 40/40", 0) == 0 &&
            run.at("verify").value("failed", -1) == 0 && run.at("verify").value("worst", 1.0) <= 1e-4,
        "the reconstruction check passed 40/40 (worst |rel - rec| / rec <= 1e-4): " +
            run.at("verify").value("result", ""));
}

void TestAll() {
  const fs::path root = fs::temp_directory_path() / "r4dx_test_lfm2_trellis";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  const fs::path log = root / "convert.log";
  auto log_text = [&]() { return ReadWhole(log); };
  const std::string common = " --threads 2 --layouts bf16 --lm-head bf16 ";
  auto convert = [&](const Fixture& f, const fs::path& out, const std::string& extra) {
    std::error_code e2;
    fs::remove(out, e2);
    return Run("--input " + Q(f.ckpt_dir) + " --output " + Q(out) + common + extra, log);
  };
  auto no_output = [&](const fs::path& out) {
    return !fs::exists(out) && !fs::exists(out.u8string() + ".partial") && !fs::exists(out.u8string() + ".verify-failed");
  };

  std::printf("---- (a) --trellis-from on the tiny lfm2_moe checkpoint ----\n");
  Fixture fa = MakeFixture(root, "plain");
  WriteOracle(fa);
  {
    const fs::path out = root / "plain.r4dx";
    const int rcode = convert(fa, out, "--trellis-from " + Q(fa.oracle_dir));
    Check(rcode == 0, "r4dx-convert --trellis-from converts lfm2_moe (rc=" + std::to_string(rcode) + ")");
    if (rcode != 0) std::fprintf(stderr, "%s\n", log_text().c_str());
    const Container c = ReadContainer(out);
    Check(c.ok, "the container is readable");
    if (c.ok) {
      CheckTrellisContainer(c, fa);
      Check(c.Meta().value("model_arch", "") == "lfm2_moe" && c.Meta().value("norm_kind", "") == "plain",
            "model_arch lfm2_moe, norm_kind plain");
    }
  }

  std::printf("---- (b) refusals before an output exists ----\n");
  auto refuses = [&](const Fixture& fx, const std::string& extra, const std::string& needle, const std::string& what) {
    const fs::path o = root / "refused.r4dx";
    const int rc3 = convert(fx, o, extra);
    const bool ok = rc3 != 0 && Contains(log_text(), needle) && no_output(o);
    Check(ok, what + " (looking for '" + needle + "')");
    if (!ok) std::fprintf(stderr, "%s\n", log_text().c_str());
  };
  {
    const std::string w3 = L(3) + "feed_forward.experts.2.w3.weight";
    Fixture fb = MakeFixture(root, "missing-w3");
    WriteOracle(fb, {w3});
    refuses(fb, "--trellis-from " + Q(fb.oracle_dir), "experts.2.gate_up",
            "refused: an expert's w3 missing from the manifest (partial gate_up)");
    Fixture fc = MakeFixture(root, "missing-expert");
    std::set<std::string> d;
    for (const char* w : {"w1", "w2", "w3"}) d.insert(L(2) + "feed_forward.experts.1." + w + ".weight");
    WriteOracle(fc, d);
    refuses(fc, "--trellis-from " + Q(fc.oracle_dir), "body linear(s) not (fully) covered",
            "refused: a whole expert missing from the manifest");
    Fixture fd = MakeFixture(root, "stale");
    WriteOracle(fd, {}, {}, {"L03.json"});
    refuses(fd, "--trellis-from " + Q(fd.oracle_dir), "stale_layers",
            "refused: a stale layer this conversion uses");
    refuses(fa, "--rotate q2a --trellis-from " + Q(fa.oracle_dir), "--rotate",
            "refused: --rotate with --trellis-from (the LFM2 trellis is unrotated)");
  }

  std::printf("---- (c) experts are keyed apart ----\n");
  {
    Fixture fe = MakeFixture(root, "swap");
    WriteOracle(fe, {}, {L(2) + "feed_forward.experts.0.w2.weight", L(2) + "feed_forward.experts.3.w2.weight"});
    const fs::path o = root / "swap.r4dx";
    const int rc4 = convert(fe, o, "--trellis-from " + Q(fe.oracle_dir));
    Check(rc4 != 0 && Contains(log_text(), "do not reconstruct") && !fs::exists(o),
          "two experts' oracle tensors swapped under their names fail the reconstruction check");
    if (!(rc4 != 0 && Contains(log_text(), "do not reconstruct"))) std::fprintf(stderr, "%s\n", log_text().c_str());
  }
  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(R4DX_CONVERT_EXE)
  try {
    TestAll();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
    return 1;
  }
  if (g_failures > 0) {
    std::fprintf(stderr, "%d of %d check(s) FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("ALL PASS (%d checks)\n", g_checks);
  return 0;
#else
  std::printf("SKIP: r4dx-convert is not built in this configuration\n");
  return 77;
#endif
}
