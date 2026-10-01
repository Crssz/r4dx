// tests/convert/test_gemma_trellis.cpp -- CPU-only (ctest `convert_gemma_trellis`). docs/gemma4-plan.md
// M1-28 prerequisites: `r4dx-convert --trellis-from` (and --kv-calib) on a gemma4_unified checkpoint, plain
// and ROTATED. Everything is generated here (no Python, no fixture, no GPU): the tiny single-file Gemma
// checkpoint of test_gemma_layout.cpp's shape (hidden 768 = 3 x 256, one sliding + one full layer, no v_proj
// on the full layer), and synthetic oracle directories in the form tools/reference/trellis_quant.py
// quantize-model writes (random ring words, fp16 scales; the checkpoint's weights are built FROM the
// reconstruction, like tools/reference/trellis_import_golden.py, so rel_weight_err is a real, uneven ~1%).
//
//   (a) unrotated: the oracle covers exactly the HF tensors Gemma has (7 on the sliding layer, 6 on the full
//       layer -- no v_proj, k_eq_v), every `.trellis.w|suh|svh` equals the pair-grid regrid of the oracle's
//       words / the parts' scales (gate before up for mlp.gate_up), no other layout for a body linear,
//       __metadata__.quant.trellis, the verify pass 13/13, lm_head written w4a16 (group 32 by rule, with the
//       bf16 companion) -- also LDLQ'd against a Hessian for the tied head -- and --kv-calib descales per
//       layer (2 sliding heads, 1 full).
//   (b) rotated (q2ab, and q2a): `r4dx-convert --rotation-out` writes the exact Q / Hb the container gets
//       (the test regenerates them with rotation.hpp and compares every value, and the fingerprint the oracle
//       must carry); an oracle that quantized fold(W) (the checkpoint is built so that fold(W) is near the
//       reconstruction) imports, and the reconstruction check compares against fold(W): the same bits, with
//       a manifest that quantized the UNFOLDED weights but claims the rotation, fail it. The container carries
//       the rotation tensors / metadata, the norms folded to ones, and r4dx_convert_run.trellis.rotation.
//   (c) every refusal, each before an output file exists: a rotated manifest without --rotate, an unrotated
//       manifest with --rotate, another seed, another tensors_sha256, a manifest that lists a v_proj of the
//       full layer, a manifest missing a Gemma body linear, --trellis-from with a Qwen-shaped --rotate.
// The Qwen byte-identity guards stay convert_trellis_import and the other convert_* tests; this file never
// runs a Qwen conversion. r4dx-convert must be built (R4DX_CONVERT_EXE); without it the test SKIPs (77).
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/rotation.hpp"
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

// ---- the tiny Gemma checkpoint (test_gemma_layout.cpp's geometry; every N and K a multiple of 128) -----------

constexpr int64_t kHidden = 768;  // rotation block 256 x 3
constexpr int64_t kHeads = 4, kKvS = 2, kKvF = 1;
constexpr int64_t kHdS = 64, kHdF = 128;
constexpr int64_t kInter = 512;
constexpr int64_t kVocab = 64;
constexpr int kKb = 4;  // every oracle tensor at K = 4 (32 words per tile)

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct Lin {
  std::string hf;           // model.language_model.layers.L.<module>.weight
  int layer = 0;
  std::string module;       // self_attn.q_proj ...
  int64_t n = 0, k = 0;
  std::string base;         // container base (text.layers.L.attn.q, mlp.gate_up, ...)
};

struct Ckpt {
  std::map<std::string, std::pair<std::vector<int64_t>, std::vector<float>>> t;  // bf16-exact values
  std::vector<std::string> order;
  json config;
  std::vector<Lin> lins;  // the trellis-covered HF tensors in converter order
};

std::string L(int i) { return "model.language_model.layers." + std::to_string(i) + "."; }

// Builds the checkpoint's norms / embeddings now; the 13 body weights are filled later (they depend on
// the oracle's reconstruction), so `lins` records their names and shapes.
Ckpt MakeSkeleton() {
  Ckpt c;
  std::mt19937_64 rng(0x6E3A5ull);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto put = [&](const std::string& name, std::vector<int64_t> shape, std::vector<float> v) {
    c.order.push_back(name);
    c.t[name] = {std::move(shape), std::move(v)};
  };
  auto rnd = [&](int64_t n, double mu, double sigma) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(mu + sigma * nd(rng)));
    return v;
  };
  for (int i = 0; i < 2; ++i) {
    const bool full = i == 1;
    const int64_t hd = full ? kHdF : kHdS, kv = full ? kKvF : kKvS;
    for (const char* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm"})
      put(L(i) + n + ".weight", {kHidden}, rnd(kHidden, 1.0, 0.2));  // plain weights around 1
    put(L(i) + "self_attn.q_norm.weight", {hd}, rnd(hd, 1.0, 0.2));
    put(L(i) + "self_attn.k_norm.weight", {hd}, rnd(hd, 1.0, 0.2));
    put(L(i) + "layer_scalar", {1}, {Bf16Round(i == 0 ? 0.375f : 0.8125f)});
    auto lin = [&](const char* module, int64_t n, int64_t k, const char* base) {
      c.lins.push_back({L(i) + module + ".weight", i, module, n, k,
                        "text.layers." + std::to_string(i) + "." + base});
    };
    lin("self_attn.q_proj", kHeads * hd, kHidden, "attn.q");
    lin("self_attn.k_proj", kv * hd, kHidden, "attn.k");
    if (!full) lin("self_attn.v_proj", kv * hd, kHidden, "attn.v");
    lin("self_attn.o_proj", kHidden, kHeads * hd, "attn.o");
    lin("mlp.gate_proj", kInter, kHidden, "mlp.gate_up");
    lin("mlp.up_proj", kInter, kHidden, "mlp.gate_up");
    lin("mlp.down_proj", kHidden, kInter, "mlp.down");
  }
  put("model.language_model.embed_tokens.weight", {kVocab, kHidden}, rnd(kVocab * kHidden, 0.0, 0.1));
  put("model.language_model.norm.weight", {kHidden}, rnd(kHidden, 1.0, 0.2));
  c.config = {{"architectures", {"Gemma4UnifiedForConditionalGeneration"}},
              {"model_type", "gemma4_unified"},
              {"eos_token_id", {1, 106}},
              {"text_config",
               {{"model_type", "gemma4_unified_text"},
                {"hidden_size", kHidden},
                {"num_hidden_layers", 2},
                {"layer_types", {"sliding_attention", "full_attention"}},
                {"num_attention_heads", kHeads},
                {"num_key_value_heads", kKvS},
                {"num_global_key_value_heads", kKvF},
                {"head_dim", kHdS},
                {"global_head_dim", kHdF},
                {"intermediate_size", kInter},
                {"vocab_size", kVocab},
                {"attention_k_eq_v", true},
                {"tie_word_embeddings", true}}}};
  return c;
}

void WriteCkpt(const Ckpt& c, const fs::path& dir) {
  fs::create_directories(dir);
  json hdr = json::object();
  std::string data;
  for (const auto& name : c.order) {
    const auto& [shape, v] = c.t.at(name);
    const uint64_t b = data.size();
    for (float x : v) {
      const uint16_t h = r4dx::core::FloatToBf16(x);
      data.append(reinterpret_cast<const char*>(&h), 2);
    }
    hdr[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {b, data.size()}}};
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model.safetensors", file);  // one file, no index (like the real Huihui checkpoint)
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
          out[static_cast<size_t>((row0 + row) * k + k0 + kk)] = blk[static_cast<size_t>(row * tr::kHad + kk)] * su * sv[row];
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

// W_hat * (1 + a g), a per 16 x 16 tile in [0.002, 0.02]: an uneven, ~1% error, so a reconstruction check
// that skipped a block or compared against the wrong basis cannot reproduce rel_weight_err.
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

std::string Sha(const std::string& bytes) { return rc::Sha256Hex(bytes); }

void WriteSafetensors(const fs::path& path, const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::string>>& ts,
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

// ---- the rotation, as the converter generates it (kind, default seed) ------------------------------------------

struct Rot {
  rc::RotationKind kind = rc::RotationKind::kNone;
  uint64_t seed = rc::kDefaultRotationSeed;
  rc::RotationSet set;
  std::vector<std::vector<float>> norms;  // not used; norms come from the checkpoint
  std::string Name() const { return rc::RotationKindName(kind); }
};

Rot MakeRot(rc::RotationKind kind, uint64_t seed = rc::kDefaultRotationSeed) {
  Rot r;
  r.kind = kind;
  r.seed = seed;
  rc::RotationShape shape;
  shape.hidden = kHidden;
  shape.block = rc::ChooseRotationBlock(kHidden);
  shape.k_down = kInter;
  shape.k_o = kHeads * kHdS;
  shape.k_o_full = kHeads * kHdF;
  shape.b_o = shape.b_o_full = rc::kHadBlockO;
  r.set = rc::GenerateRotationSet(kind, seed, shape);
  return r;
}

// What the converter does to the K side of one HF tensor under `r`: in-projections W diag(w) Q, q2ab's
// Hadamard on o / down. `which`: 0 none, 1 in-projection (with `norm`), 2 Hadamard (with `hb`).
struct FoldSpec {
  int which = 0;
  const std::vector<float>* norm = nullptr;
  const rc::BlockHadamard* hb = nullptr;
};

FoldSpec SpecFor(const Rot& r, const Ckpt& c, const Lin& l) {
  FoldSpec s;
  if (r.kind == rc::RotationKind::kNone) return s;
  const bool mlp_in = l.module == "mlp.gate_proj" || l.module == "mlp.up_proj";
  const bool attn_in = l.module == "self_attn.q_proj" || l.module == "self_attn.k_proj" || l.module == "self_attn.v_proj";
  if (attn_in || mlp_in) {
    s.which = 1;
    s.norm = &c.t.at(L(l.layer) + (mlp_in ? "pre_feedforward_layernorm.weight" : "input_layernorm.weight")).second;
  } else if (r.kind == rc::RotationKind::kQ2ab) {
    s.which = 2;
    s.hb = l.module == "mlp.down_proj" ? &r.set.had_down : (l.layer == 1 ? &r.set.had_o_full : &r.set.had_o);
  }
  return s;
}

std::vector<float> Fold(const Rot& r, const FoldSpec& s, std::vector<float> w, int64_t n, int64_t k) {
  if (s.which == 1) rc::FoldRowsQ(w, n, k, s.norm->data(), r.set.q, 1, /*norm_offset=*/0.0);
  if (s.which == 2) rc::FoldRowsHadamard(w, n, k, *s.hb, 1);
  return w;
}

// The inverse of Fold, in double: the checkpoint weight whose fold is `target` (before bf16 rounding).
std::vector<float> Unfold(const Rot& r, const FoldSpec& s, const std::vector<float>& target, int64_t n, int64_t k) {
  std::vector<float> out(target.size());
  std::vector<double> v(static_cast<size_t>(k)), tmp(static_cast<size_t>(k));
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t j = 0; j < k; ++j) v[static_cast<size_t>(j)] = target[static_cast<size_t>(row * k + j)];
    if (s.which == 1) {
      r.set.q.ApplyT(v.data(), tmp.data());
      for (int64_t j = 0; j < k; ++j) v[static_cast<size_t>(j)] /= static_cast<double>((*s.norm)[static_cast<size_t>(j)]);
    } else if (s.which == 2) {
      s.hb->ApplyT(v.data());
    }
    for (int64_t j = 0; j < k; ++j) out[static_cast<size_t>(row * k + j)] = static_cast<float>(v[static_cast<size_t>(j)]);
  }
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

// ---- one fixture: a checkpoint + the oracle that matches it ---------------------------------------------------

struct Fixture {
  Ckpt ckpt;
  std::map<std::string, Oracle> oracle;  // hf -> oracle tensors
  std::map<std::string, double> rel;     // hf -> rel_weight_err against what the converter will compare with
  fs::path ckpt_dir, oracle_dir;
};

// `r`: the rotation the oracle "quantized in" (kNone = the plain basis). The checkpoint's weight is the
// unfold of the perturbed reconstruction, rounded to bf16; rel is then measured against fold(bf16 weight) --
// exactly the weight the converter's check forms. `claim` = fold the manifest's rel against instead (a
// manifest that did not fold: rel against the raw weight).
Fixture MakeFixture(const fs::path& root, const std::string& tag, const Rot& r, bool oracle_folded = true) {
  Fixture f;
  f.ckpt = MakeSkeleton();
  for (size_t i = 0; i < f.ckpt.lins.size(); ++i) {
    const Lin& l = f.ckpt.lins[i];
    Oracle o = MakeOracle(i, l.n, l.k);
    const std::vector<float> pert = Perturb(o.w_hat, l.n, l.k, i);
    const FoldSpec spec = SpecFor(r, f.ckpt, l);
    std::vector<float> w = Unfold(r, spec, pert, l.n, l.k);
    for (auto& x : w) x = Bf16Round(x);
    f.ckpt.order.push_back(l.hf);
    f.ckpt.t[l.hf] = {{l.n, l.k}, w};
    f.oracle[l.hf] = std::move(o);
  }
  f.ckpt_dir = root / ("ckpt-" + tag);
  f.oracle_dir = root / ("oracle-" + tag);
  WriteCkpt(f.ckpt, f.ckpt_dir);
  for (const Lin& l : f.ckpt.lins) {
    const FoldSpec spec = SpecFor(r, f.ckpt, l);
    const std::vector<float>& w = f.ckpt.t.at(l.hf).second;
    f.rel[l.hf] = RelErr(f.oracle.at(l.hf).w_hat, oracle_folded ? Fold(r, spec, w, l.n, l.k) : w);
  }
  return f;
}

json Fingerprint(const Rot& r) {
  std::string blob = "r4dx-rotation-v1\n";
  auto add = [&](const std::string& name, const std::vector<float>& v) {
    blob += name + "\n";
    const uint64_t n = v.size();
    blob.append(reinterpret_cast<const char*>(&n), sizeof(n));
    blob.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
  };
  add("rotation.signs", r.set.q.signs);
  add(rc::RotationMixName(r.set.q.nblk), r.set.q.mix);
  if (r.kind == rc::RotationKind::kQ2ab) {
    add("rotation.had_down_signs", r.set.had_down.signs);
    add("rotation.had_o_signs", r.set.had_o.signs);
    add("rotation.had_o_full_signs", r.set.had_o_full.signs);
  }
  return {{"kind", r.Name()}, {"seed", r.seed}, {"tensors_sha256", Sha(blob)}, {"hidden", kHidden},
          {"block", r.set.q.block}};
}

// Writes the oracle directory (quantize-model form): one L<ii>.safetensors per layer + weights_override.json.
// `drop`: HF names left out of the manifest and files; `extra`: a fabricated record to add (a v_proj of the
// full layer); `rotation`: the manifest's "rotation" block (null = none).
void WriteOracle(const Fixture& f, const json& rotation, const std::set<std::string>& drop = {},
                 const std::vector<std::string>& extra = {}) {
  fs::create_directories(f.oracle_dir);
  const std::string config_sha = Sha(ReadWhole(f.ckpt_dir / "config.json"));
  json tensors = json::object();
  std::vector<std::string> hf_all;
  for (const Lin& l : f.ckpt.lins) hf_all.push_back(l.hf);
  for (const auto& e : extra) hf_all.push_back(e);
  for (int layer = 0; layer < 2; ++layer) {
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::string>> ts;
    std::vector<std::string> in_file;
    for (const std::string& hf : hf_all) {
      if (drop.count(hf)) continue;
      if (hf.rfind(L(layer), 0) != 0) continue;
      const bool is_extra = f.oracle.count(hf) == 0;
      const Lin* l = nullptr;
      for (const Lin& x : f.ckpt.lins)
        if (x.hf == hf) l = &x;
      // An "extra" record borrows q_proj's oracle tensors (its shape is that of the full layer's k_proj).
      const std::string src = is_extra ? L(layer) + "self_attn.k_proj.weight" : hf;
      const Oracle& o = f.oracle.at(src);
      int64_t n = 0, k = 0;
      for (const Lin& x : f.ckpt.lins)
        if (x.hf == src) n = x.n, k = x.k;
      (void)l;
      ts.emplace_back(hf + ".trellis", "I32", std::vector<int64_t>{k / 16, n / 16, tr::WordsPerTile(kKb)},
                      std::string(reinterpret_cast<const char*>(o.words.data()), o.words.size() * 4));
      ts.emplace_back(hf + ".suh", "F16", std::vector<int64_t>{k},
                      std::string(reinterpret_cast<const char*>(o.suh.data()), o.suh.size() * 2));
      ts.emplace_back(hf + ".svh", "F16", std::vector<int64_t>{n},
                      std::string(reinterpret_cast<const char*>(o.svh.data()), o.svh.size() * 2));
      in_file.push_back(hf);
    }
    if (ts.empty()) continue;
    char name[16];
    std::snprintf(name, sizeof(name), "L%02d.safetensors", layer);
    WriteSafetensors(f.oracle_dir / name, ts, {{"format", "r4dx-weights-override"}, {"layer", std::to_string(layer)}});
    const std::string fsha = Sha(ReadWhole(f.oracle_dir / name));
    for (const std::string& hf : in_file) {
      const bool is_extra = f.oracle.count(hf) == 0;
      const std::string src = is_extra ? L(layer) + "self_attn.k_proj.weight" : hf;
      int64_t n = 0, k = 0;
      for (const Lin& x : f.ckpt.lins)
        if (x.hf == src) n = x.n, k = x.k;
      tensors[hf] = {{"encoding", "trellis-exl3"}, {"K", static_cast<double>(kKb)}, {"codebook", "mul1"},
                     {"k", k}, {"n", n}, {"shape_hf", {n, k}},
                     {"words_shape", {k / 16, n / 16, tr::WordsPerTile(kKb)}},
                     {"rel_weight_err", f.rel.at(src)}, {"hessian_basis", "matched"},
                     {"file", name}, {"file_sha256", fsha}};
    }
  }
  json man = {{"format", "r4dx-weights-override"}, {"version", 1}, {"encoding", "trellis-exl3"},
              {"complete", true}, {"missing", json::array()}, {"missing_count", 0},
              {"bpw_target", nullptr}, {"K_uniform", static_cast<double>(kKb)},
              {"layers_done", {0, 1}}, {"stale_layers", json::array()},
              {"model_dir", f.ckpt_dir.u8string()}, {"config_sha256", config_sha},
              {"hessian_dir", "(synthetic)"}, {"hessian_manifest_sha256", std::string(64, '2')},
              {"hessian_basis", "matched"}, {"codebook", "mul1"},
              {"recipe", {{"note", "synthetic: random ring words (tests/convert/test_gemma_trellis.cpp)"}}},
              {"code_sha256", {{"trellis_quant", std::string(64, '0')}, {"trellis_viterbi", std::string(64, '1')}}},
              {"tensors", tensors}};
  if (!rotation.is_null()) man["rotation"] = rotation;
  WriteWhole(f.oracle_dir / "weights_override.json", man.dump(2));
}

void RewriteManifest(const fs::path& oracle_dir, const std::function<void(json&)>& edit) {
  json man = json::parse(ReadWhole(oracle_dir / "weights_override.json"));
  edit(man);
  WriteWhole(oracle_dir / "weights_override.json", man.dump(2));
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
  std::vector<float> Floats(const std::string& n) const {
    const std::string& b = tensors.at(n);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
  }
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

// What a trellis container must hold for the Gemma fixture, and that nothing else of a body linear exists.
void CheckTrellisContainer(const Container& c, const Fixture& f, const std::string& what, bool lm_bf16 = true) {
  std::map<std::string, std::vector<const Lin*>> by_base;
  for (const Lin& l : f.ckpt.lins) by_base[l.base].push_back(&l);
  Check(by_base.size() == 11, what + ": 11 body bases (6 sliding incl. v, 5 full without v: k_eq_v)");
  bool bytes_ok = true, no_other_layout = true;
  const tr::RingTables& unused = tr::Ring(kKb);
  (void)unused;
  for (const auto& [base, parts] : by_base) {
    std::vector<tr::OracleWords> ow;
    std::string suh, svh;
    int64_t N = 0;
    for (const Lin* l : parts) {
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
               c.Has(base + ".trellis.suh") && c.tensors.at(base + ".trellis.suh") == suh &&
               c.Has(base + ".trellis.svh") && c.tensors.at(base + ".trellis.svh") == svh;
    for (const char* other : {".bf16.w", ".w4a16.wq", ".w4a16.wsz"}) no_other_layout = no_other_layout && !c.Has(base + other);
  }
  Check(bytes_ok, what + ": every .trellis.w / .suh / .svh is the oracle's regrid / the parts' scales (gate before up)");
  Check(no_other_layout, what + ": no .bf16.w / .w4a16.* next to a trellis linear");
  Check(!c.Has("text.layers.1.attn.v.trellis.w") && !c.Has("text.layers.1.attn.v.bf16.w"),
        what + ": no attn.v on the full layer");
  const json q = c.Meta().at("quant").at("trellis");
  std::set<std::string> in_meta;
  for (auto it = q.at("linears").begin(); it != q.at("linears").end(); ++it) in_meta.insert(it.key());
  std::set<std::string> want_meta;
  for (const auto& kv : by_base) want_meta.insert(kv.first);
  Check(q.value("format", "") == "r4dx-trellis" && in_meta == want_meta &&
            q.at("linears").at("text.layers.0.mlp.gate_up").at("parts") == json::array({kInter, kInter}),
        what + ": __metadata__.quant.trellis lists the 11 bases (gate_up with parts [512, 512])");
  const json run = c.Meta().at("r4dx_convert_run").at("trellis");
  Check(run.value("hf_tensors", 0) == 13 && run.at("verify").value("result", "").rfind("pass 13/13", 0) == 0 &&
            run.at("verify").value("failed", -1) == 0 && run.at("verify").value("worst", 1.0) <= 1e-4,
        what + ": the reconstruction check passed 13/13 (worst |rel - rec| / rec <= 1e-4): " +
            run.at("verify").value("result", ""));
  // lm_head: the tied head is not a body linear; it stays w4a16 (+ the default bf16 companion).
  Check(c.Has("lm_head.bf16.w") == lm_bf16 && !c.Has("lm_head.trellis.w"),
        what + ": lm_head is not trellis (" + (lm_bf16 ? "bf16 companion present" : "no bf16 companion") + ")");
}

void TestAll() {
  const fs::path root = fs::temp_directory_path() / "r4dx_test_gemma_trellis";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root);
  const fs::path log = root / "convert.log";
  auto log_text = [&]() { return ReadWhole(log); };
  auto convert = [&](const Fixture& f, const fs::path& out, const std::string& extra) {
    std::error_code e2;
    fs::remove(out, e2);
    return Run("--input " + Q(f.ckpt_dir) + " --output " + Q(out) + " --threads 2 " + extra, log);
  };
  auto no_output = [&](const fs::path& out) {
    return !fs::exists(out) && !fs::exists(out.u8string() + ".partial") && !fs::exists(out.u8string() + ".verify-failed");
  };

  // ======================= (a) unrotated ===================================================================
  std::printf("---- (a) --trellis-from on the tiny gemma4_unified checkpoint (unrotated) ----\n");
  const Rot none;
  Fixture fa = MakeFixture(root, "plain", none);
  WriteOracle(fa, nullptr);
  {
    // --kv-calib: per-layer head counts (2 sliding, 1 full). lm_head: LDLQ at group 32 against a Hessian.
    json calib = {{"0", {{"k_amax", {4.48, 8.96}}, {"v_amax", {44.8, 89.6}}}},
                  {"1", {{"k_amax", {13.44}}, {"v_amax", {22.4}}}}};
    WriteWhole(root / "kvcalib.json", calib.dump());
    fs::create_directories(root / "hess");
    {
      std::string h(64, '\0');
      std::memcpy(h.data(), "R4DXHES1", 8);
      const uint32_t K = kHidden, flags = 1;
      const uint64_t rows = 8 * kHidden;
      std::memcpy(h.data() + 8, &K, 4);
      std::memcpy(h.data() + 12, &flags, 4);
      std::memcpy(h.data() + 16, &rows, 8);
      std::string body;
      double trace = 0.0;
      for (int64_t i = 0; i < kHidden; ++i)
        for (int64_t j = i; j < kHidden; ++j) {
          const float v = static_cast<float>(std::pow(0.9, static_cast<double>(j - i)) * (1.0 + 0.5 * ((i % 7) == 0)));
          if (i == j) trace += v;
          body.append(reinterpret_cast<const char*>(&v), 4);
        }
      std::memcpy(h.data() + 24, &trace, 8);
      WriteWhole(root / "hess" / "lm_head.hess", h + body);
      json hm = {{"format", "r4dx-hessian"}, {"version", 1},
                 {"files", {{"lm_head.hess", {{"K", kHidden}, {"rows", rows}, {"trace", trace}}}}},
                 {"keys", {{"lm_head", "lm_head.hess"}}}};
      WriteWhole(root / "hess" / "hessian.json", hm.dump(2));
    }
    const fs::path out = root / "plain.r4dx";
    const int rcode = convert(fa, out, "--trellis-from " + Q(fa.oracle_dir) + " --kv-calib " + Q(root / "kvcalib.json") +
                                           " --no-bf16 --lm-head w4a16 --hessian-dir " + Q(root / "hess") + " --ldlq \"^lm_head$\"" +
                                           " --w4a16-group-rule \"^lm_head$=32\"");
    Check(rcode == 0, "plain: r4dx-convert --trellis-from on gemma4_unified converts (rc=" + std::to_string(rcode) + ")");
    if (rcode != 0) std::fprintf(stderr, "%s\n", log_text().c_str());
    const Container c = ReadContainer(out);
    Check(c.ok, "plain: the container is readable");
    if (c.ok) {
      CheckTrellisContainer(c, fa, "plain", /*lm_bf16=*/false);
      Check(!c.Meta().contains("rotation") && !c.Meta().at("r4dx_convert_run").at("trellis").contains("rotation"),
            "plain: no rotation metadata");
      Check(c.Meta().value("model_arch", "") == "gemma4_unified" && c.Meta().value("norm_kind", "") == "plain",
            "plain: model_arch gemma4_unified, norm_kind plain");
      // lm_head w4a16 at group 32 (LDLQ'd), tied embedding bytes in the bf16 companion.
      bool wsz32 = false;
      for (const auto& kv : c.tensors)
        if (kv.first.rfind("lm_head.w4a16.wsz", 0) == 0 && Contains(kv.first, "g32")) wsz32 = true;
      Check(wsz32 && c.Has("lm_head.w4a16.wq"), "plain: lm_head written w4a16 at group 32 (lm_head.w4a16.wsz.g32)");
      const json run = c.Meta().at("r4dx_convert_run");
      Check(run.value("ldlq", "") == "^lm_head$", "plain: the tied lm_head was rounded by LDLQ (r4dx_convert_run.ldlq)");
      // --kv-calib: descale = amax / 448 per kv head of THAT layer.
      auto approx = [](const std::vector<float>& a, std::vector<float> b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
          if (std::fabs(a[i] - b[i]) > 1e-6f) return false;
        return true;
      };
      Check(approx(c.Floats("text.layers.0.attn.k_descale"), {4.48f / 448.0f, 8.96f / 448.0f}) &&
                approx(c.Floats("text.layers.0.attn.v_descale"), {44.8f / 448.0f, 89.6f / 448.0f}) &&
                approx(c.Floats("text.layers.1.attn.k_descale"), {13.44f / 448.0f}) &&
                approx(c.Floats("text.layers.1.attn.v_descale"), {22.4f / 448.0f}),
            "plain: --kv-calib k/v descale per layer (2 sliding heads, 1 full)");
      Check(c.Floats("text.layers.0.layer_scalar") == std::vector<float>({0.375f}), "plain: layer_scalar widened");
      Check(c.Has("text.layers.1.input_layernorm") && !c.Has("text.layers.1.input_layernorm.rotated"),
            "plain: norms stored raw, not folded");
    }
  }

  // ======================= (b) rotated =====================================================================
  for (const rc::RotationKind kind : {rc::RotationKind::kQ2ab, rc::RotationKind::kQ2a}) {
    const Rot r = MakeRot(kind);
    const std::string tag = r.Name();
    std::printf("---- (b) rotated trellis, --rotate %s ----\n", tag.c_str());
    // --rotation-out: the exact tensors and fingerprint.
    const fs::path rot_file = root / ("rot-" + tag + ".safetensors");
    Fixture fb = MakeFixture(root, tag, r);
    {
      const int rcode = Run("--input " + Q(fb.ckpt_dir) + " --rotate " + tag + " --rotation-out " + Q(rot_file), log);
      Check(rcode == 0 && fs::exists(rot_file), "--rotation-out " + tag + ": writes the rotation file (rc=" + std::to_string(rcode) + ")");
      if (rcode != 0) std::fprintf(stderr, "%s\n", log_text().c_str());
      const Container rf = ReadContainer(rot_file);
      Check(rf.ok && rf.header.contains("__metadata__"), "--rotation-out " + tag + ": a safetensors file with metadata");
      if (rf.ok) {
        const json md = rf.Meta();
        bool same = rf.Floats("rotation.signs") == r.set.q.signs && rf.Floats("rotation.mix") == r.set.q.mix;
        if (kind == rc::RotationKind::kQ2ab)
          same = same && rf.Floats("rotation.had_down_signs") == r.set.had_down.signs &&
                 rf.Floats("rotation.had_o_signs") == r.set.had_o.signs &&
                 rf.Floats("rotation.had_o_full_signs") == r.set.had_o_full.signs;
        Check(same, "--rotation-out " + tag + ": every tensor equals rotation.hpp's generation (signs, mix 3x3, had_*)");
        Check(rf.header.at("rotation.mix").at("shape") == json::array({3, 3}) &&
                  rf.header.at("rotation.signs").at("shape") == json::array({kHidden}) &&
                  rf.header.at("rotation.signs").at("dtype") == "F32",
              "--rotation-out " + tag + ": F32 with natural shapes (mix [3, 3])");
        const json fp = json::parse(md.at("fingerprint").get<std::string>());
        Check(fp == Fingerprint(r), "--rotation-out " + tag + ": the fingerprint is kind + seed + sha256 of the tensors: " + fp.dump());
        const json rotmd = json::parse(md.at("rotation").get<std::string>());
        Check(rotmd.value("kind", "") == tag && rotmd.value("out_fold", "") == "had_only",
              "--rotation-out " + tag + ": the container's __metadata__.rotation block (Gemma option A)");
      }
    }
    WriteOracle(fb, Fingerprint(r));
    const fs::path out = root / ("rot-" + tag + ".r4dx");
    const int rcode = convert(fb, out, "--rotate " + tag + " --trellis-from " + Q(fb.oracle_dir));
    Check(rcode == 0, "rotated " + tag + ": r4dx-convert --rotate " + tag + " --trellis-from converts (rc=" + std::to_string(rcode) + ")");
    if (rcode != 0) std::fprintf(stderr, "%s\n", log_text().c_str());
    const Container c = ReadContainer(out);
    Check(c.ok, "rotated " + tag + ": the container is readable");
    if (c.ok) {
      CheckTrellisContainer(c, fb, "rotated " + tag);
      const json md = c.Meta();
      Check(md.contains("rotation") && md.at("rotation").value("kind", "") == tag && md.at("rotation").value("out_fold", "") == "had_only",
            "rotated " + tag + ": __metadata__.rotation (kind, option A)");
      Check(md.at("r4dx_convert_run").at("trellis").at("rotation") == Fingerprint(r),
            "rotated " + tag + ": r4dx_convert_run.trellis.rotation is the shared fingerprint");
      bool tensors_same = c.Floats("rotation.signs") == r.set.q.signs && c.Floats("rotation.mix") == r.set.q.mix;
      if (kind == rc::RotationKind::kQ2ab)
        tensors_same = tensors_same && c.Floats("rotation.had_o_full_signs") == r.set.had_o_full.signs;
      Check(tensors_same, "rotated " + tag + ": the container's rotation tensors are the --rotation-out ones");
      Check(c.Has("text.layers.0.input_layernorm.rotated") && !c.Has("text.layers.0.input_layernorm") &&
                c.Has("text.layers.1.pre_feedforward_layernorm.rotated") && c.Has("text.layers.0.post_attention_layernorm"),
            "rotated " + tag + ": folded norms stored as ones under .rotated; the post-norms (not folded) stay");
    }

    // The reconstruction check forms fold(W): an oracle that did NOT fold (its rel is measured against the raw
    // weight) but claims the rotation cannot pass it -- the check would otherwise be vacuous for a rotated run.
    {
      const Fixture fu = MakeFixture(root, tag + "-unfolded", r, /*oracle_folded=*/false);
      WriteOracle(fu, Fingerprint(r));
      const fs::path out2 = root / ("rot-" + tag + "-unfolded.r4dx");
      const int rc2 = convert(fu, out2, "--rotate " + tag + " --trellis-from " + Q(fu.oracle_dir));
      Check(rc2 != 0 && Contains(log_text(), "do not reconstruct") && fs::exists(out2.u8string() + ".verify-failed") &&
                !fs::exists(out2),
            "rotated " + tag + ": a manifest whose rel was measured against the UNFOLDED weight fails the reconstruction check");
    }

    // (c) refusals, each before an output exists.
    auto refuses = [&](const Fixture& fx, const std::string& extra, const std::string& needle, const std::string& what) {
      const fs::path o = root / "refused.r4dx";
      const int rc3 = convert(fx, o, extra);
      Check(rc3 != 0 && Contains(log_text(), needle) && no_output(o), what + " (looking for '" + needle + "')");
      if (!(rc3 != 0 && Contains(log_text(), needle))) std::fprintf(stderr, "%s\n", log_text().c_str());
    };
    refuses(fb, "--trellis-from " + Q(fb.oracle_dir), "quantized ROTATED weights",
            "refused (" + tag + "): a rotated manifest without --rotate");
    refuses(fa, "--rotate " + tag + " --trellis-from " + Q(fa.oracle_dir), "has none",
            "refused (" + tag + "): an unrotated manifest with --rotate");
    refuses(fb, "--rotate " + tag + " --rotation-seed 7 --trellis-from " + Q(fb.oracle_dir), "rotation.",
            "refused (" + tag + "): another --rotation-seed than the oracle folded");
    {
      const Fixture fw = MakeFixture(root, tag + "-wrongsha", r);
      WriteOracle(fw, Fingerprint(r));
      RewriteManifest(fw.oracle_dir, [](json& m) { m["rotation"]["tensors_sha256"] = std::string(64, 'a'); });
      refuses(fw, "--rotate " + tag + " --trellis-from " + Q(fw.oracle_dir), "tensors_sha256",
              "refused (" + tag + "): another tensors_sha256 (other sign tensors)");
    }
  }

  // ======================= (c) layout refusals (unrotated) ================================================
  std::printf("---- (c) Gemma layout refusals of --trellis-from ----\n");
  {
    const Fixture fx = MakeFixture(root, "vproj", none);
    // A manifest that lists a v_proj of the FULL layer: the checkpoint has none, so the entry is unused (a
    // warning), never imported as a body linear.
    WriteOracle(fx, nullptr, {}, {L(1) + "self_attn.v_proj.weight"});
    const fs::path out = root / "vproj.r4dx";
    const int rcode = convert(fx, out, "--trellis-from " + Q(fx.oracle_dir));
    Check(rcode == 0 && Contains(log_text(), "not used by this conversion") && Contains(log_text(), "self_attn.v_proj"),
          "a manifest entry for the full layer's v_proj (k_eq_v) is reported unused, not imported");
    const Container c = ReadContainer(out);
    Check(c.ok && !c.Has("text.layers.1.attn.v.trellis.w"), "no attn.v is written for the full layer");
  }
  {
    const Fixture fx = MakeFixture(root, "missing", none);
    WriteOracle(fx, nullptr, {L(1) + "self_attn.k_proj.weight"});
    const fs::path out = root / "missing.r4dx";
    const int rcode = convert(fx, out, "--trellis-from " + Q(fx.oracle_dir));
    Check(rcode != 0 && Contains(log_text(), "not (fully) covered") && Contains(log_text(), "text.layers.1.attn.k") && no_output(out),
          "a missing Gemma body linear (full layer's k_proj) is refused before an output exists");
  }
  {
    // --rotate with a Qwen-shaped config keeps the Qwen refusal (the unrotated-only rule); here only the
    // message of the exe's argument stage is checked on a Gemma config that is NOT rotated: --rotation-out
    // needs --rotate.
    const int rcode = Run("--input " + Q(fa.ckpt_dir) + " --rotation-out " + Q(root / "x.safetensors"), log);
    Check(rcode != 0 && Contains(log_text(), "--rotation-out needs --rotate"), "--rotation-out without --rotate is refused");
    const int rc2 = Run("--input " + Q(fa.ckpt_dir) + " --output " + Q(root / "y.r4dx") + " --rotate q2ab --rotation-out " + Q(root / "x.safetensors"), log);
    Check(rc2 != 0 && Contains(log_text(), "--rotation-out takes --input and --rotate only"), "--rotation-out with --output is refused");
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
