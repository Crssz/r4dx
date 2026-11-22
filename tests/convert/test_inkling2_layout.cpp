// r4dx-convert's inkling2_loop branch (src/convert/inkling2_layout.{hpp,cpp}, main.cpp; docs/inkling2-plan.md 3.2, 4).
//
//   (a) the layout over the REAL checkpoint's geometry -- the 1244-tensor safetensors header (tests/convert/data/
//       inkling2_real_header.tsv: name, dtype, shape; no tensor bytes) and the real config.json -- through a recording
//       kit: every tensor consumed or allow-listed, container names / shapes, the attn_queries index shift, [3, H] queries
//       on full layers, MXFP8 raw sources exactly in layers 20-23, the padded head, every refusal. No weights, no exe.
//   (b) when r4dx-convert is built: the exe on the tiny synthetic checkpoint (tests/convert/data/inkling2_tiny, made by
//       tools/convert_ref/make_tiny_inkling2.py, BOTH expert encodings): `--layouts mxfp8 --no-bf16` and `--layouts bf16`
//       containers byte-checked against the source tensors; the checkpoint's MXFP8 experts copied BIT-EXACT (and shown to
//       differ from a re-encode on the edge groups, so the raw copy is load-bearing); NVFP4 experts = decode + de-interleave
//       + EncodeGroup; --keep-bf16; thread-count independence; --plan-only; the coverage audit failing BEFORE the output
//       file exists; every refusal. CPU-only; writes a few MB under %TEMP% and removes them.
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
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "inkling2_layout.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_mxfp8.hpp"

namespace fs = std::filesystem;
namespace ik = r4dx_convert::inkling2;
namespace mx = r4dx_convert::mxfp8;
using nlohmann::json;

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
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  return false;
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

const fs::path kData = fs::path(R4DX_CONVERT_DATA_DIR);

std::vector<int64_t> With(std::vector<int64_t> s, int64_t w) {
  s.push_back(w);
  return s;
}

// ---- (a) the recording kit over a name / dtype / shape table ---------------------------------------------------------

struct Meta {
  std::string dtype;
  std::vector<int64_t> shape;
};
using Table = std::map<std::string, Meta>;

Table LoadTsv(const fs::path& p) {
  Table t;
  std::istringstream in(ReadWhole(p));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const size_t a = line.find('\t'), b = line.find('\t', a + 1);
    Meta m;
    m.dtype = line.substr(a + 1, b - a - 1);
    std::istringstream sh(line.substr(b + 1));
    std::string d;
    while (std::getline(sh, d, ',')) m.shape.push_back(std::stoll(d));
    t[line.substr(0, a)] = m;
  }
  return t;
}

struct Linear {
  int64_t N = 0, K = 0;
  bool head = false, source = false, raw_mxfp8 = false;
};

struct Recorder {
  Table ckpt;
  std::map<std::string, std::vector<int64_t>> shape;  // container name -> container shape of non-linears
  std::map<std::string, Linear> linears;
  std::map<std::string, std::string> origin;  // container name -> hf name (non-linears)
  ik::Kit kit;

  explicit Recorder(Table t) : ckpt(std::move(t)) {
    kit.all_names = [this]() {
      std::vector<std::string> n;
      for (const auto& kv : ckpt) n.push_back(kv.first);
      return n;
    };
    kit.has = [this](const std::string& n) { return ckpt.count(n) != 0; };
    kit.shape = [this](const std::string& n) { return ckpt.at(n).shape; };
    kit.dtype = [this](const std::string& n) { return ckpt.at(n).dtype; };
    kit.data = [](const std::string&) -> const uint8_t* { return nullptr; };
    kit.add_bf16 = [this](const std::string& hf, const std::string& name) {
      shape[name] = With(ckpt.at(hf).shape, 2);
      origin[name] = hf;
    };
    kit.add_bf16_shaped = [this](const std::string& hf, const std::string& name, const std::vector<int64_t>& s) {
      shape[name] = With(s, 2);
      origin[name] = hf;
    };
    kit.add_fp32_widen = [this](const std::string& hf, const std::string& name) {
      shape[name] = With(ckpt.at(hf).shape, 4);
      origin[name] = hf;
    };
    kit.add_msg_queries = [this](const std::string& q, const std::string& kv, const std::string& name, int64_t L, int64_t H) {
      shape[name] = {L, 4, H, 4};
      origin[name] = q + "+" + kv;
    };
    kit.add_linear = [this](const std::vector<std::string>& hf, const std::string& base, bool head) {
      linears[base] = {ckpt.at(hf.at(0)).shape.at(0), ckpt.at(hf.at(0)).shape.at(1), head, false, false};
    };
    kit.add_source_linear = [this](const std::string& base, bool head, ik::LinearSource src) {
      linears[base] = {src.N, src.K, head, true, static_cast<bool>(src.raw_mxfp8)};
    };
  }
};

json RealConfig() { return json::parse(ReadWhole(kData / "inkling2_real_config.json")); }

void TestConfig() {
  std::printf("---- (a) ParseConfig ----\n");
  const ik::TextShape s = ik::ParseConfig(RealConfig());
  Check(s.hidden == 2304 && s.layers == 24 && s.heads == 24 && s.kv_heads == 8 && s.head_dim == 128 && s.d_rel == 16 &&
            s.rel_extent == 1024 && s.conv_kernel == 4 && s.vocab == 201024 && s.unpadded_vocab == 200058 && s.experts == 128 &&
            s.moe_intermediate == 1152 && s.loop_start == 6 && s.loop_end == 18 && s.loop_passes == 3 && s.gate_rank == 64 &&
            s.attn_res_block == 6,
        "the real config's geometry (loop 6..18 is [start, end): 12 loop layers)");
  std::vector<int64_t> full;
  for (int64_t i = 0; i < s.layers; ++i)
    if (s.IsFull(i)) full.push_back(i);
  Check(full == std::vector<int64_t>({5, 11, 17, 23}) && s.LoopLayers() == 12 && s.HeadRowsStored() == 200064,
        "full layers 5, 11, 17, 23 (from local_layer_ids: the real config has no layer_types); head stored rows 200064");

  auto with = [&](const char* key, json v) {
    json c = RealConfig();
    c["text_config"][key] = v;
    return c;
  };
  Check(Throws([&] { ik::ParseConfig(with("swa_head_dim", 64)); }, "swa_"), "swa_* geometry different from the full one is refused");
  Check(Throws([&] { ik::ParseConfig(with("n_shared_experts", 1)); }, "shared"), "shared experts are refused");
  Check(Throws([&] { ik::ParseConfig(with("loop_config", {{"core_start", 6}, {"core_end", 17}, {"num_passes", 3}, {"gate_rank", 64}})); }, "multiples"),
        "loop bounds that are not multiples of n_layers_per_attn_res_block are refused (HF validate_architecture)");
  Check(Throws([&] { ik::ParseConfig(with("loop_config", {{"core_start", 0}, {"core_end", 18}})); }, "multiples"), "loop start 0 is refused");
  json lt = json::array();
  for (int i = 0; i < 24; ++i) lt.push_back(i % 5 == 4 ? "hybrid" : "hybrid_sliding");
  Check(Throws([&] { ik::ParseConfig(with("layer_types", lt)); }, "disagree"), "layer_types and local_layer_ids that disagree are refused");
  json lt2 = json::array();
  for (int i = 0; i < 24; ++i) lt2.push_back((i + 1) % 6 == 0 ? "hybrid" : "hybrid_sliding");
  Check(!Throws([&] { ik::ParseConfig(with("layer_types", lt2)); }), "layer_types consistent with local_layer_ids is accepted");
  json noids = RealConfig();
  noids["text_config"].erase("local_layer_ids");
  const ik::TextShape d = ik::ParseConfig(noids);
  Check(d.IsFull(5) && d.IsFull(23) && !d.IsFull(4) && !d.IsFull(0), "no layer_types / local_layer_ids: HF's default ((i + 1) % 6 == 0 is full)");
  json bad = RealConfig();
  bad["model_type"] = "qwen3_5_moe";
  Check(Throws([&] { ik::ParseConfig(bad); }, "model_type"), "another model_type is refused");
  json nokey = RealConfig();
  nokey["text_config"].erase("hidden_size");
  Check(Throws([&] { ik::ParseConfig(nokey); }, "hidden_size"), "a missing key is named");
}

void TestMerge() {
  std::printf("---- (a) MergeMessageQueries ----\n");
  const int64_t L = 3, H = 4;
  std::vector<float> q(2 * L * H), kv(2 * L * H), out(static_cast<size_t>(L * 4 * H), -1.0f);
  for (int64_t r = 0; r < 2 * L; ++r)
    for (int64_t c = 0; c < H; ++c) {
      q[static_cast<size_t>(r * H + c)] = 100.0f + 10.0f * r + c;   // rows: q0, mlp0, q1, mlp1, ...
      kv[static_cast<size_t>(r * H + c)] = 200.0f + 10.0f * r + c;  // rows: k0, v0, k1, v1, ...
    }
  ik::MergeMessageQueries(q.data(), kv.data(), L, H, out.data());
  bool ok = true;
  for (int64_t j = 0; j < L; ++j)
    for (int64_t c = 0; c < H; ++c) {
      const auto at = [&](int64_t kind) { return out[static_cast<size_t>((j * 4 + kind) * H + c)]; };
      ok = ok && at(0) == q[static_cast<size_t>((2 * j) * H + c)] && at(1) == kv[static_cast<size_t>((2 * j) * H + c)] &&
           at(2) == kv[static_cast<size_t>((2 * j + 1) * H + c)] && at(3) == q[static_cast<size_t>((2 * j + 1) * H + c)];
    }
  Check(ok, "per loop layer j: [q[2j], kv[2j], kv[2j+1], q[2j+1]] = [query, key, value, mlp] (Inkling2LoopMergeMessageQueries)");
}

void TestRealGeometry() {
  std::printf("---- (a) the layout over the real 1244-tensor header ----\n");
  Table t = LoadTsv(kData / "inkling2_real_header.tsv");
  Check(t.size() == 1244, "header table has 1244 tensors (822 text + 422 model.visual.*)");
  const ik::TextShape s = ik::ParseConfig(RealConfig());
  Recorder rec(t);
  std::set<std::string> consumed;
  try {
    consumed = ik::AddInkling2TextStack(rec.kit, s);
  } catch (const std::exception& e) {
    Check(false, std::string("AddInkling2TextStack on the real header threw: ") + e.what());
    return;
  }
  const std::vector<std::string> unconsumed = ik::UnconsumedTensors(rec.kit.all_names(), consumed);
  Check(unconsumed.empty(), "coverage audit: every real tensor is consumed or allow-listed (model.visual.*)");
  size_t visual = 0;
  for (const auto& kv : t) visual += ik::IsAllowListed(kv.first) ? 1 : 0;
  Check(visual == 422 && consumed.size() == 822 && consumed.size() + visual == t.size(),
        "822 text tensors consumed + 422 allow-listed model.visual.* = 1244");
  bool vis_disjoint = true;
  for (const auto& n : consumed) vis_disjoint = vis_disjoint && !ik::IsAllowListed(n);
  Check(vis_disjoint, "no allow-listed tensor was consumed");

  // Linears: 24 x 6 attention + 24 x 128 x 2 experts + the head.
  size_t body = 0, src = 0, heads = 0;
  for (const auto& kv : rec.linears) {
    body += kv.second.source ? 0 : 1;
    src += kv.second.source ? 1 : 0;
    heads += kv.second.head ? 1 : 0;
  }
  Check(body == 144 && src == 24 * 128 * 2 + 1 && heads == 1 && rec.linears.size() == 144 + 6144 + 1,
        "144 attention linears + 6144 expert linears + 1 head (6289 linears)");
  const auto L = [&](const std::string& n) -> const Linear& { return rec.linears.at(n); };
  Check(L("text.layers.0.attn.q").N == 3072 && L("text.layers.0.attn.q").K == 2304 && L("text.layers.0.attn.k").N == 1024 &&
            L("text.layers.7.attn.v").N == 1024 && L("text.layers.7.attn.gate").N == 3072 && L("text.layers.7.attn.o").N == 2304 &&
            L("text.layers.7.attn.o").K == 3072 && L("text.layers.23.attn.rel").N == 384 && L("text.layers.23.attn.rel").K == 2304,
        "attention linear shapes q/gate [3072,2304], k/v [1024,2304], o [2304,3072], rel [384,2304]");
  Check(L("text.layers.3.moe.experts.127.gate_up").N == 2304 && L("text.layers.3.moe.experts.127.gate_up").K == 2304 &&
            L("text.layers.3.moe.experts.0.down").N == 2304 && L("text.layers.3.moe.experts.0.down").K == 1152,
        "expert linears gate_up [2304, 2304] (de-interleaved w13) and down [2304, 1152]");
  bool raw_ok = true;
  for (int64_t i = 0; i < 24; ++i)
    for (int64_t e : {0, 77, 127})
      for (const char* w : {"gate_up", "down"})
        raw_ok = raw_ok && rec.linears.at("text.layers." + std::to_string(i) + ".moe.experts." + std::to_string(e) + "." + w).raw_mxfp8 == (i >= 20);
  Check(raw_ok, "a bit-exact raw MXFP8 source exists exactly for layers 20-23 (NVFP4 layers 0-19 are re-encoded)");
  const Linear& head = L("lm_head");
  Check(head.head && head.N == 200064 && head.K == 2304 && !head.raw_mxfp8, "lm_head: unembed rows padded 200058 -> 200064 (N % 16 for the fragment layout), K 2304");

  const auto& sh = rec.shape;
  Check(sh.at("text.embed_tokens") == std::vector<int64_t>({201024, 2304, 2}), "embed_tokens keeps all 201024 rows (bf16)");
  Check(sh.at("text.embed_norm") == std::vector<int64_t>({2304, 2}) && sh.at("text.final_norm") == std::vector<int64_t>({2304, 2}) &&
            sh.at("text.final_query") == std::vector<int64_t>({2304, 4}),
        "embed_norm / final_norm bf16, final_query f32");
  Check(sh.count("text.layers.0.attn_query") == 0 && sh.at("text.layers.1.attn_query") == std::vector<int64_t>({2304, 4}) &&
            sh.at("text.layers.5.attn_query") == std::vector<int64_t>({3, 2304, 4}) && sh.at("text.layers.23.attn_query") == std::vector<int64_t>({3, 2304, 4}) &&
            sh.at("text.layers.6.attn_query") == std::vector<int64_t>({2304, 4}),
        "attn_query: none for layer 0, [H] on sliding layers, [3, H] on full layers 5 / 23");
  Check(rec.origin.at("text.layers.5.attn_query") == "model.llm.attn_res.attn_queries.4" && rec.origin.at("text.layers.1.attn_query") == "model.llm.attn_res.attn_queries.0" &&
            rec.origin.at("text.layers.23.attn_query") == "model.llm.attn_res.attn_queries.22" && rec.origin.at("text.layers.0.mlp_query") == "model.llm.attn_res.mlp_queries.0" &&
            rec.origin.at("text.layers.23.mlp_query") == "model.llm.attn_res.mlp_queries.23",
        "attn_queries.{i} -> layer i + 1 (attn_queries.4 is layer 5's [3, H]); mlp_queries.{i} -> layer i");
  Check(sh.at("text.msg_queries") == std::vector<int64_t>({12, 4, 2304, 4}), "msg_queries merged to [12, 4, 2304] f32");
  Check(sh.at("text.layers.0.attn.k_conv") == std::vector<int64_t>({1024, 4, 2}) && sh.at("text.layers.0.attn_conv") == std::vector<int64_t>({2304, 4, 2}) &&
            sh.at("text.layers.0.mlp_conv") == std::vector<int64_t>({2304, 4, 2}) && sh.count("text.layers.0.attn.k_conv_static") == 0,
        "sliding layer: conv taps [C, 1, 4] squeezed to [C, 4]");
  Check(sh.at("text.layers.5.attn.k_conv_static") == std::vector<int64_t>({4, 1024, 2}) && sh.at("text.layers.5.attn.v_conv_proj") == std::vector<int64_t>({32, 2304, 2}) &&
            sh.count("text.layers.5.attn.k_conv") == 0 && sh.count("text.layers.5.attn_norm") == 0 && sh.at("text.layers.5.q_norm") == std::vector<int64_t>({2304, 2}) &&
            sh.at("text.layers.5.v_norm") == std::vector<int64_t>({2304, 2}) && sh.at("text.layers.0.attn_norm") == std::vector<int64_t>({2304, 2}) &&
            sh.count("text.layers.0.q_norm") == 0,
        "full layer: static conv [4, 1024] + proj [32, 2304], q/k/v norms instead of attn_norm");
  Check(sh.at("text.layers.2.attn.sink") == std::vector<int64_t>({24, 2304, 2}) && sh.at("text.layers.2.attn.q_scale") == std::vector<int64_t>({24, 128, 2}) &&
            sh.at("text.layers.2.attn.rel_scale") == std::vector<int64_t>({24, 16, 2}) && sh.at("text.layers.2.attn.rel_proj") == std::vector<int64_t>({16, 1024, 2}),
        "sink [24, 2304], q_scale [24, 128], rel_scale [24, 16], rel_proj [16, 1024] bf16 passthrough");
  bool msg_ok = true;
  for (int64_t i = 0; i < 24; ++i) {
    const bool has = sh.count("text.layers." + std::to_string(i) + ".msg.q.down") != 0;
    msg_ok = msg_ok && has == (i >= 6 && i < 18);
  }
  Check(msg_ok && sh.at("text.layers.6.msg.mlp.gain") == std::vector<int64_t>({2304, 64, 2}) && sh.at("text.layers.17.msg.k.down") == std::vector<int64_t>({64, 4608, 2}) &&
            rec.origin.at("text.layers.9.msg.q.down") == "model.llm.layers.9.controllers.attn.down_weight" &&
            rec.origin.at("text.layers.9.msg.k.forget") == "model.llm.layers.9.controllers.k.forget_weight",
        "msg.{q,k,v,mlp}.{down,forget,gain} exactly on loop layers 6..17 (controllers.attn = q)");
  Check(sh.at("text.layers.9.moe.router") == std::vector<int64_t>({128, 2304, 2}) && sh.at("text.layers.9.moe.bias") == std::vector<int64_t>({128, 4}),
        "router bf16 [128, 2304], selection bias f32 [128]");

  // Refusals: each starts from a fresh recorder over a mutated table.
  auto run = [&](const std::function<void(Table&)>& mutate, const std::string& substr) {
    Table m = t;
    mutate(m);
    Recorder r(m);
    return Throws([&] { ik::AddInkling2TextStack(r.kit, s); }, substr);
  };
  Check(run([](Table& m) { m.erase("model.llm.layers.9.mlp.gate.bias"); }, "mlp.gate.bias"), "a missing tensor is named");
  Check(run([](Table& m) { m["model.llm.layers.9.attn.wq_du.weight"].shape = {3072, 2303}; }, "shape"), "a wrong shape is refused");
  Check(run([](Table& m) { m["model.llm.layers.9.attn.wq_du.weight"].dtype = "F32"; }, "dtype"), "a wrong dtype is refused");
  Check(run([](Table& m) { m["model.llm.attn_res.attn_queries.4"].shape = {2304}; }, "attn_queries.4"), "a full layer's [H] query (want [3, H]) is refused");
  Check(run([](Table& m) { m["model.llm.attn_res.attn_queries.3"].shape = {3, 2304}; }, "attn_queries.3"), "a sliding layer's [3, H] query (want [H]) is refused");
  Check(run([](Table& m) { m.erase("model.llm.layers.3.mlp.experts.w13_weight_amax"); }, "without"), "a denominator without its amax is refused");
  Check(run([](Table& m) { m.erase("model.llm.layers.3.mlp.experts.w13_weight_amax"); m.erase("model.llm.layers.3.mlp.experts.w13_weight_outer_scale_denominator"); }, "dtype"),
        "an NVFP4 weight that lost its amax and denominator reads as MXFP8 and fails on dtype");
  Check(run([](Table& m) { m.erase("model.llm.layers.3.mlp.experts.w13_weight_outer_scale_denominator"); }, "outer_scale_denominator"), "amax without its denominator is refused");
  Check(run([](Table& m) { m["model.llm.layers.21.mlp.experts.w2_weight_scale"].shape = {128, 2304, 35}; }, "w2_weight_scale"), "an MXFP8 scale of the wrong width is refused");
  Check(run([](Table& m) { m["model.llm.layers.22.mlp.experts.w13_weight_outer_scale_denominator"] = {"F32", {1}}; }, "without"), "a denominator on an MXFP8 group is refused");
  Check(run([](Table& m) { m.erase("model.llm.middle_cycle_cross_pass_kv_queries"); }, "middle_cycle_cross_pass_kv_queries"), "a missing cross-pass tensor is named");
  Check(run([](Table& m) { m["model.llm.unembed.weight"].shape = {200058, 2304}; }, "unembed"), "an already unpadded unembed (not [vocab, H]) is refused");
  // The audit.
  {
    Table m = t;
    m["model.llm.layers.0.mystery.weight"] = {"BF16", {4}};
    m["model.visual.extra.weight"] = {"BF16", {4}};
    Recorder r(m);
    const auto c = ik::AddInkling2TextStack(r.kit, s);
    const auto u = ik::UnconsumedTensors(r.kit.all_names(), c);
    Check(u == std::vector<std::string>({"model.llm.layers.0.mystery.weight"}), "audit: an extra model.llm tensor is reported, an extra model.visual.* is allow-listed");
    Table lm = t;
    lm["lm_head.weight"] = {"BF16", {200058, 2304}};
    Recorder r2(lm);
    const auto c2 = ik::AddInkling2TextStack(r2.kit, s);
    Check(ik::UnconsumedTensors(r2.kit.all_names(), c2) == std::vector<std::string>({"lm_head.weight"}), "audit: a tensor the checkpoint grew (lm_head.weight) is reported");
  }
}

// ---- (b) the exe on the tiny checkpoint --------------------------------------------------------------------------------

#if defined(R4DX_CONVERT_EXE)

struct Src {
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};
using Ckpt = std::map<std::string, Src>;

Ckpt ReadSafetensors(const fs::path& p) {
  const std::string f = ReadWhole(p);
  uint64_t hl = 0;
  std::memcpy(&hl, f.data(), 8);
  const json h = json::parse(f.substr(8, static_cast<size_t>(hl)));
  Ckpt c;
  for (auto it = h.begin(); it != h.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0], e = it.value()["data_offsets"][1];
    c[it.key()] = {it.value()["dtype"].get<std::string>(), it.value()["shape"].get<std::vector<int64_t>>(),
                   f.substr(static_cast<size_t>(8 + hl + b), static_cast<size_t>(e - b))};
  }
  return c;
}

void WriteSafetensors(const fs::path& p, const Ckpt& c) {
  json hdr = json::object();
  std::string data;
  for (const auto& kv : c) {
    const uint64_t b = data.size();
    data += kv.second.bytes;
    hdr[kv.first] = {{"dtype", kv.second.dtype}, {"shape", kv.second.shape}, {"data_offsets", {b, data.size()}}};
  }
  std::string h = hdr.dump();
  h.append((8 - h.size() % 8) % 8, ' ');
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  WriteWhole(p, file + h + data);
}

struct Container {
  bool ok = false;
  json header;
  std::map<std::string, std::string> tensors;
  json Meta() const { return header.at("__metadata__"); }
  std::vector<int64_t> Shape(const std::string& n) const { return header.at(n).at("shape").get<std::vector<int64_t>>(); }
  const std::string& Bytes(const std::string& n) const { return tensors.at(n); }
  bool Has(const std::string& n) const { return tensors.count(n) != 0; }
  std::set<std::string> Names() const {
    std::set<std::string> n;
    for (const auto& kv : tensors) n.insert(kv.first);
    return n;
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
  c.header = json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  const std::string data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0], e = it.value()["data_offsets"][1];
    c.tensors[it.key()] = data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" + log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

fs::path TempDir(const std::string& tag) {
  const fs::path d = fs::temp_directory_path() / ("r4dx_test_inkling2_layout_" + tag);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

std::vector<float> Bf16Floats(const std::string& b) {
  std::vector<float> v(b.size() / 2);
  for (size_t i = 0; i < v.size(); ++i) {
    uint16_t h;
    std::memcpy(&h, b.data() + 2 * i, 2);
    v[i] = r4dx::core::Bf16ToFloat(h);
  }
  return v;
}

std::vector<float> F32Floats(const std::string& b) {
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

std::vector<uint8_t> U8(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }
std::string Str(const std::vector<uint8_t>& v) { return std::string(v.begin(), v.end()); }

// A linear's expected mxfp8 tensors from f32 weights (the converter's path: EncodeGroup per (row, 32-K group), packed).
void ExpectMxfp8(const std::vector<float>& w, int N, int K, std::string* wq, std::string* ws) {
  std::vector<uint8_t> codes, scales;
  r4dx_convert::QuantizeMxfp8(w.data(), N, K, 1, codes, scales);
  *wq = Str(r4dx_convert::PackMxfp8Codes(codes, N, K, 1));
  *ws = Str(r4dx_convert::PackMxfp8Scales(scales, N, K));
}

// The expert group of layer `layer` (w13 | w2) as the codec's view over the source bytes.
struct ExpertSrc {
  ik::ExpertTensors t;
  bool nvfp4 = false;
  std::string w, s, amax_bytes;
};

std::shared_ptr<ExpertSrc> OpenExpert(const Ckpt& c, int layer, const std::string& which, int64_t E, int64_t rows, int64_t cols) {
  const std::string p = "model.llm.layers." + std::to_string(layer) + ".mlp.experts." + which + "_weight";
  auto x = std::make_shared<ExpertSrc>();
  x->nvfp4 = c.count(p + "_amax") != 0;
  x->w = c.at(p).bytes;
  x->s = c.at(p + "_scale").bytes;
  x->t.format = x->nvfp4 ? ik::ExpertFormat::kNvfp4 : ik::ExpertFormat::kMxfp8;
  x->t.experts = E;
  x->t.rows = rows;
  x->t.cols = cols;
  x->t.weight = reinterpret_cast<const uint8_t*>(x->w.data());
  x->t.scale = reinterpret_cast<const uint8_t*>(x->s.data());
  if (x->nvfp4) {
    x->amax_bytes = c.at(p + "_amax").bytes;
    x->t.amax = reinterpret_cast<const uint8_t*>(x->amax_bytes.data());
    x->t.denominator = F32Floats(c.at(p + "_outer_scale_denominator").bytes).at(0);
  }
  return x;
}

struct Tiny {
  fs::path dir;
  json config;
  ik::TextShape s;
  Ckpt src;
};

Tiny LoadTiny() {
  Tiny t;
  t.dir = kData / "inkling2_tiny";
  t.config = json::parse(ReadWhole(t.dir / "config.json"));
  t.s = ik::ParseConfig(t.config);
  t.src = ReadSafetensors(t.dir / "model.safetensors");
  return t;
}

// What a container of the tiny checkpoint must hold: non-linear names plus the linear bases (suffixes added per layout).
struct Expect {
  std::set<std::string> plain;
  std::vector<std::string> linears;
};

Expect ExpectedTiny(const ik::TextShape& s) {
  Expect e;
  e.plain = {"text.embed_tokens", "text.embed_norm", "text.final_norm", "text.final_query", "text.msg_queries"};
  e.linears.push_back("lm_head");
  for (int64_t i = 0; i < s.layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    if (i >= 1) e.plain.insert(b + "attn_query");
    for (const char* n : {"mlp_query", "attn.sink", "attn.q_scale", "attn.rel_scale", "attn.rel_proj", "mlp_norm", "attn_conv", "mlp_conv", "moe.router", "moe.bias"})
      e.plain.insert(b + n);
    for (const char* n : {"attn.q", "attn.k", "attn.v", "attn.gate", "attn.o", "attn.rel"}) e.linears.push_back(b + n);
    if (s.IsFull(i)) {
      for (const char* n : {"attn.k_conv_static", "attn.v_conv_static", "attn.k_conv_proj", "attn.v_conv_proj", "q_norm", "k_norm", "v_norm"}) e.plain.insert(b + n);
    } else {
      for (const char* n : {"attn.k_conv", "attn.v_conv", "attn_norm"}) e.plain.insert(b + n);
    }
    if (s.IsLoop(i))
      for (const char* m : {"q", "k", "v", "mlp"})
        for (const char* p : {"down", "forget", "gain"}) e.plain.insert(b + "msg." + m + "." + p);
    for (int64_t x = 0; x < s.experts; ++x) {
      e.linears.push_back(b + "moe.experts." + std::to_string(x) + ".gate_up");
      e.linears.push_back(b + "moe.experts." + std::to_string(x) + ".down");
    }
  }
  return e;
}

std::set<std::string> NamesFor(const Expect& e, const std::string& wq, const std::string& ws,
                               const std::function<bool(const std::string&)>& kept = nullptr) {
  std::set<std::string> n = e.plain;
  for (const std::string& l : e.linears) {
    if ((kept && kept(l)) || wq.empty()) {
      n.insert(l + ".bf16.w");
    } else {
      n.insert(l + wq);
      n.insert(l + ws);
    }
  }
  return n;
}

void CheckNames(const Container& c, const std::set<std::string>& want, const std::string& what) {
  const std::set<std::string> got = c.Names();
  Check(got == want, what);
  if (got != want) {
    std::set<std::string> extra, missing;
    std::set_difference(got.begin(), got.end(), want.begin(), want.end(), std::inserter(extra, extra.end()));
    std::set_difference(want.begin(), want.end(), got.begin(), got.end(), std::inserter(missing, missing.end()));
    for (const auto& x : extra) std::fprintf(stderr, "  extra: %s\n", x.c_str());
    for (const auto& x : missing) std::fprintf(stderr, "  missing: %s\n", x.c_str());
  }
}

// The non-linear tensors: container name -> hf name.
std::map<std::string, std::string> Passthroughs(const ik::TextShape& s) {
  std::map<std::string, std::string> m = {{"text.embed_tokens", "model.llm.embed.weight"},
                                          {"text.embed_norm", "model.llm.embed_norm.weight"},
                                          {"text.final_norm", "model.llm.norm.weight"},
                                          {"text.final_query", "model.llm.attn_res.output_query"}};
  for (int64_t i = 0; i < s.layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".", h = "model.llm.layers." + std::to_string(i) + ".";
    if (i >= 1) m[b + "attn_query"] = "model.llm.attn_res.attn_queries." + std::to_string(i - 1);
    m[b + "mlp_query"] = "model.llm.attn_res.mlp_queries." + std::to_string(i);
    m[b + "attn.sink"] = h + "attn.attn_sink_proj.weight";
    m[b + "attn.q_scale"] = h + "attn.q_l2_clip.scale";
    m[b + "attn.rel_scale"] = h + "attn.rel_l2_clip.scale";
    m[b + "attn.rel_proj"] = h + "attn.rel_logits_proj.proj";
    m[b + "mlp_norm"] = h + "mlp_norm.weight";
    m[b + "attn_conv"] = h + "attn_sconv.weight";
    m[b + "mlp_conv"] = h + "mlp_sconv.weight";
    m[b + "moe.router"] = h + "mlp.gate.weight";
    m[b + "moe.bias"] = h + "mlp.gate.bias";
    if (s.IsFull(i)) {
      for (const char* kv : {"k", "v"}) {
        m[b + "attn." + kv + "_conv_static"] = h + "attn." + kv + "_sconv.static_w";
        m[b + "attn." + kv + "_conv_proj"] = h + "attn." + kv + "_sconv.weight_proj";
      }
      for (const char* n : {"q", "k", "v"}) m[b + n + "_norm"] = h + n + "_input_norm.weight";
    } else {
      for (const char* kv : {"k", "v"}) m[b + "attn." + kv + "_conv"] = h + "attn." + kv + "_sconv.weight";
      m[b + "attn_norm"] = h + "attn_norm.weight";
    }
    if (s.IsLoop(i)) {
      const char* hf[4] = {"attn", "k", "v", "mlp"};
      const char* ct[4] = {"q", "k", "v", "mlp"};
      for (int c = 0; c < 4; ++c)
        for (const char* p : {"down", "forget", "gain"}) m[b + "msg." + ct[c] + "." + p] = h + "controllers." + hf[c] + "." + p + "_weight";
    }
  }
  return m;
}

// The bf16 body linears: container base -> hf name.
std::map<std::string, std::string> BodyLinears(const ik::TextShape& s) {
  std::map<std::string, std::string> m;
  for (int64_t i = 0; i < s.layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".attn.", h = "model.llm.layers." + std::to_string(i) + ".attn.";
    m[b + "q"] = h + "wq_du.weight";
    m[b + "k"] = h + "wk_dv.weight";
    m[b + "v"] = h + "wv_dv.weight";
    m[b + "gate"] = h + "gate_attn.weight";
    m[b + "o"] = h + "wo_ud.weight";
    m[b + "rel"] = h + "wr_du.weight";
  }
  return m;
}

// The f32 values an expert linear must carry: HF-semantics decode of expert `e`, w13 de-interleaved to [gate; up].
std::vector<float> ExpertF32(const ExpertSrc& x, int64_t e, bool w13) {
  std::vector<float> raw(static_cast<size_t>(x.t.rows * x.t.cols)), out(raw.size());
  ik::DecodeExpert(x.t, e, raw.data());
  if (w13) {
    ik::DeinterleaveGateUp(raw.data(), out.data(), x.t.rows, x.t.cols);
    return out;
  }
  return raw;
}

std::string Bf16Bytes(const std::vector<float>& v) {
  const std::vector<uint8_t> b = r4dx_convert::EncodeBf16(v);
  return Str(b);
}

void TestExe() {
  std::printf("---- (b) r4dx-convert on the tiny inkling2_loop checkpoint ----\n");
  const Tiny tiny = LoadTiny();
  const ik::TextShape& s = tiny.s;
  const int64_t H = s.hidden, I = s.moe_intermediate, E = s.experts;
  const fs::path root = TempDir("exe");
  const fs::path out = root / "out.r4dx", log = root / "convert.log";
  const std::string in = " --input \"" + tiny.dir.u8string() + "\" ";
  auto convert = [&](const std::string& extra, const fs::path& output = fs::path()) {
    const fs::path o = output.empty() ? out : output;
    std::error_code ec;
    fs::remove(o, ec);
    return Run(in + "--output \"" + o.u8string() + "\" " + extra, log);
  };
  auto log_has = [&](const std::string& x) { return ReadWhole(log).find(x) != std::string::npos; };
  const Expect expect = ExpectedTiny(s);
  const auto pass = Passthroughs(s);
  const auto body = BodyLinears(s);

  // ---- mxfp8 ----
  int rc = convert("--layouts mxfp8 --no-bf16 --threads 2");
  Check(rc == 0, "mxfp8 container: converts the tiny checkpoint (rc=" + std::to_string(rc) + ")");
  const Container c = ReadContainer(out);
  Check(c.ok, "mxfp8 container: readable");
  if (!c.ok) return;
  CheckNames(c, NamesFor(expect, mx::kWqSuffix, mx::kWsSuffix), "mxfp8 container: exactly the plan's tensor set (4.x names; no vision, no mtp, no .bf16.w)");
  const json md = c.Meta();
  Check(md.value("model_arch", "") == "inkling2_loop" && md.value("norm_kind", "") == "plain" && md.at("model_config") == tiny.config &&
            md.value("model_id", "") == "inkling2_tiny" && md.at("r4dx_convert_run").value("arch", "") == "inkling2_loop",
        "metadata: model_arch inkling2_loop, norm_kind plain, model_config verbatim, model_id = directory name");
  const json& q = md.at("quant").at("mxfp8");
  Check(q.value("version", 0) == 1 && q.value("group", 0) == 32 && q.value("elem", "") == "e4m3fn" && q.value("scale", "") == "e8m0" &&
            q.value("fragment_permutation", "") == mx::kMetaFragmentPermutation && !md.at("quant").contains("w8a16"),
        "metadata: quant.mxfp8 {version 1, group 32, e4m3fn, e8m0, fragment permutation}, no w8a16 block");
  const json& ikm = md.at("inkling2");
  Check(ikm.value("lm_head_rows", 0) == 90 && ikm.value("lm_head_rows_stored", 0) == 96 && ikm.at("expert_source_nvfp4_layers").size() == 12 &&
            ikm.at("expert_source_mxfp8_layers").size() == 4 && ikm.at("expert_source_mxfp8_layers")[0] == "12" &&
            ikm.at("expert_source_nvfp4_layers")[11] == "11",
        "metadata: inkling2 {lm_head_rows 90 / stored 96, NVFP4 layers 0-11, MXFP8 layers 12-15}");

  // passthroughs: raw bytes (bf16 / F32) unchanged, shapes carry the trailing element width
  bool pass_ok = true, shape_ok = true;
  for (const auto& kv : pass) {
    const Src& sr = tiny.src.at(kv.second);
    pass_ok = pass_ok && c.Bytes(kv.first) == sr.bytes;
    std::vector<int64_t> want = sr.shape;
    if (sr.dtype == "BF16" && want.size() == 3 && want[1] == 1) want = {want[0], want[2]};  // conv [C,1,K] -> [C,K]
    shape_ok = shape_ok && c.Shape(kv.first) == With(want, sr.dtype == "F32" ? 4 : 2);
  }
  Check(pass_ok, "every non-linear tensor is the source's bytes unchanged (bf16 and F32; conv taps squeezed, same bytes)");
  Check(shape_ok, "non-linear shapes: source shape + element width (conv [C,1,4] -> [C,4,2]; F32 queries [3,H,4])");
  Check(c.Shape("text.layers.3.attn_query") == std::vector<int64_t>({3, H, 4}) && c.Shape("text.layers.2.attn_query") == std::vector<int64_t>({H, 4}) &&
            !c.Has("text.layers.0.attn_query"),
        "attn_query: layer 3 (full) [3, H], layer 2 [H], none for layer 0");

  // msg_queries
  {
    const std::vector<float> qq = F32Floats(tiny.src.at("model.llm.middle_cycle_cross_pass_queries").bytes);
    const std::vector<float> kv = F32Floats(tiny.src.at("model.llm.middle_cycle_cross_pass_kv_queries").bytes);
    const std::vector<float> got = F32Floats(c.Bytes("text.msg_queries"));
    bool manual = got.size() == static_cast<size_t>(s.LoopLayers() * 4 * H);
    for (int64_t j = 0; j < s.LoopLayers() && manual; ++j)
      for (int64_t k = 0; k < H; ++k) {
        const auto at = [&](int64_t kind) { return got[static_cast<size_t>((j * 4 + kind) * H + k)]; };
        manual = manual && at(0) == qq[static_cast<size_t>((2 * j) * H + k)] && at(1) == kv[static_cast<size_t>((2 * j) * H + k)] &&
                 at(2) == kv[static_cast<size_t>((2 * j + 1) * H + k)] && at(3) == qq[static_cast<size_t>((2 * j + 1) * H + k)];
      }
    Check(c.Shape("text.msg_queries") == std::vector<int64_t>({s.LoopLayers(), 4, H, 4}) && manual,
          "msg_queries [loop_layers, 4, H] f32 in order q, k, v, mlp");
  }

  // body linears: EncodeGroup + packers of the bf16 weights
  bool lin_ok = true;
  for (const auto& kv : body) {
    const Src& sr = tiny.src.at(kv.second);
    std::string wq, ws;
    ExpectMxfp8(Bf16Floats(sr.bytes), static_cast<int>(sr.shape[0]), static_cast<int>(sr.shape[1]), &wq, &ws);
    lin_ok = lin_ok && c.Bytes(kv.first + ".mxfp8.wq") == wq && c.Bytes(kv.first + ".mxfp8.ws") == ws &&
             c.Shape(kv.first + ".mxfp8.wq") == std::vector<int64_t>({sr.shape[0] * sr.shape[1]}) &&
             c.Shape(kv.first + ".mxfp8.ws") == std::vector<int64_t>({sr.shape[0] * sr.shape[1] / 32});
  }
  Check(lin_ok, "attention linears (q, k, v, gate, o, rel x 16 layers): wq / ws = EncodeGroup + w8a16 fragment packing of the bf16 weight");

  // head: rows [0, 90) of unembed, rows 90..95 zero
  std::vector<float> head_w = Bf16Floats(tiny.src.at("model.llm.unembed.weight").bytes);
  head_w.resize(static_cast<size_t>(s.HeadRowsStored() * H), 0.0f);
  for (size_t i = static_cast<size_t>(s.unpadded_vocab * H); i < head_w.size(); ++i) head_w[i] = 0.0f;
  {
    std::string wq, ws;
    ExpectMxfp8(head_w, static_cast<int>(s.HeadRowsStored()), static_cast<int>(H), &wq, &ws);
    Check(c.Bytes("lm_head.mxfp8.wq") == wq && c.Bytes("lm_head.mxfp8.ws") == ws && c.Shape("lm_head.mxfp8.wq") == std::vector<int64_t>({96 * H}),
          "lm_head: unembed rows [0, 90) + 6 zero rows = 96 rows (N % 16), mxfp8");
    const std::vector<float> back = r4dx_convert::DequantMxfp8(U8(c.Bytes("lm_head.mxfp8.wq")), U8(c.Bytes("lm_head.mxfp8.ws")), 96, static_cast<int>(H));
    bool zero_rows = true;
    for (size_t i = static_cast<size_t>(90 * H); i < back.size(); ++i) zero_rows = zero_rows && back[i] == 0.0f;
    Check(zero_rows, "lm_head: the 6 padding rows dequantize to exactly zero");
  }

  // experts
  bool raw_ok = true, nv_ok = true, nv_err_ok = true;
  size_t raw_n = 0, nv_n = 0;
  for (int64_t layer = 0; layer < s.layers; ++layer)
    for (const char* which : {"w13", "w2"}) {
      const bool w13 = std::string(which) == "w13";
      const int64_t rows = w13 ? 2 * I : H, cols = w13 ? H : I;
      const auto xp = OpenExpert(tiny.src, static_cast<int>(layer), which, E, rows, cols);
      const ExpertSrc& x = *xp;
      for (int64_t e = 0; e < E; ++e) {
        const std::string base = "text.layers." + std::to_string(layer) + ".moe.experts." + std::to_string(e) + (w13 ? ".gate_up" : ".down");
        const std::string wq = c.Bytes(base + ".mxfp8.wq"), ws = c.Bytes(base + ".mxfp8.ws");
        const std::vector<float> f = ExpertF32(x, e, w13);
        if (!x.nvfp4) {
          // bit-exact: the checkpoint's codes and scale bytes, de-interleaved (w13) and fragment-packed, nothing re-encoded
          std::vector<uint8_t> codes(static_cast<size_t>(rows * cols)), scales(static_cast<size_t>(rows * cols / 32));
          std::vector<uint8_t> rc(codes.size()), rs(scales.size());
          ik::RawMxfp8Expert(x.t, e, rc.data(), rs.data());
          if (w13) {
            ik::DeinterleaveGateUp(rc.data(), codes.data(), rows, cols);
            ik::DeinterleaveGateUp(rs.data(), scales.data(), rows, cols / 32);
          } else {
            codes = rc;
            scales = rs;
          }
          raw_ok = raw_ok && wq == Str(r4dx_convert::PackMxfp8Codes(codes, static_cast<int>(rows), static_cast<int>(cols), 1)) &&
                   ws == Str(r4dx_convert::PackMxfp8Scales(scales, static_cast<int>(rows), static_cast<int>(cols)));
          // and the container decodes to the checkpoint's own values
          raw_ok = raw_ok && r4dx_convert::DequantMxfp8(U8(wq), U8(ws), static_cast<int>(rows), static_cast<int>(cols)) == f;
          ++raw_n;
        } else {
          std::string ewq, ews;
          ExpectMxfp8(f, static_cast<int>(rows), static_cast<int>(cols), &ewq, &ews);
          nv_ok = nv_ok && wq == ewq && ws == ews;
          const std::vector<float> back = r4dx_convert::DequantMxfp8(U8(wq), U8(ws), static_cast<int>(rows), static_cast<int>(cols));
          const int gpr = static_cast<int>(cols / 32);
          const std::vector<uint8_t> wsb = U8(ws);
          for (int64_t r = 0; r < rows && nv_err_ok; ++r)
            for (int64_t k = 0; k < cols; ++k) {
              // error bound of the ceil rule: 2^-4 relative on normals plus half a subnormal step of the group scale
              const uint8_t sb = wsb[static_cast<size_t>((r / 16) * gpr * 16 + (k / 32) * 16 + r % 16)];
              const double bound = std::fabs(f[static_cast<size_t>(r * cols + k)]) * 0.0625 + static_cast<double>(mx::ScaleValue(sb)) * std::ldexp(1.0, -10) * 1.0000001;
              nv_err_ok = nv_err_ok && std::fabs(back[static_cast<size_t>(r * cols + k)] - f[static_cast<size_t>(r * cols + k)]) <= bound;
            }
          ++nv_n;
        }
      }
    }
  Check(raw_n == 4 * 2 * static_cast<size_t>(E) && raw_ok,
        "MXFP8 experts (layers 12-15, w13 + w2, all " + std::to_string(raw_n) + " slices): codes and E8M0 bytes are the checkpoint's, de-interleaved + packed, bit-exact");
  Check(nv_n == 12 * 2 * static_cast<size_t>(E) && nv_ok,
        "NVFP4 experts (layers 0-11, all " + std::to_string(nv_n) + " slices): HF-semantics f32 decode, w13 de-interleaved, EncodeGroup + packing");
  Check(nv_err_ok, "NVFP4 -> mxfp8 error: |container - f32| <= 2^-4 |v| + 2^-10 * group scale on every element");
  {
    // The edge groups of layer 12 expert 0 (a zero group with a nonzero scale, scale byte 0 with subnormal codes, a group whose max code is far
    // below 224): a re-encode of the decoded values is NOT the checkpoint's bytes there; the converter's output is.
    const auto xp = OpenExpert(tiny.src, static_cast<int>(s.layers - 4), "w13", E, 2 * I, H);
    std::vector<float> f(static_cast<size_t>(2 * I * H));
    ik::DecodeExpert(xp->t, 0, f.data());
    std::vector<uint8_t> codes, scales;
    r4dx_convert::QuantizeMxfp8(f.data(), static_cast<int>(2 * I), static_cast<int>(H), 1, codes, scales);
    std::vector<uint8_t> rc(codes.size()), rs(scales.size());
    ik::RawMxfp8Expert(xp->t, 0, rc.data(), rs.data());
    Check(codes != rc && scales != rs, "layer 12 expert 0 (edge groups): re-encoding the decoded values would NOT reproduce the checkpoint's bytes -- the raw copy is load-bearing");
    Check(rs[0] == 130 && scales[0] == 0, "  e.g. the zero group with scale byte 130 would re-encode to scale byte 0 (row 0, group 0 of the interleaved source)");
  }

  // ---- determinism ----
  {
    const fs::path out2 = root / "out_t1.r4dx", out3 = root / "out_t3.r4dx";
    const int r1 = convert("--layouts mxfp8 --no-bf16 --threads 1", out2);
    const int r3 = convert("--layouts mxfp8 --no-bf16 --threads 3", out3);
    const Container c1 = ReadContainer(out2), c3 = ReadContainer(out3);
    Check(r1 == 0 && r3 == 0 && c1.ok && c3.ok && c1.tensors == c.tensors && c3.tensors == c.tensors, "--threads 1 / 2 / 3: identical tensor bytes");
  }

  // ---- bf16 ----
  rc = convert("--layouts bf16 --threads 2");
  Check(rc == 0, "bf16 container: converts (rc=" + std::to_string(rc) + ")");
  const Container b = ReadContainer(out);
  Check(b.ok, "bf16 container: readable");
  if (b.ok) {
    CheckNames(b, NamesFor(expect, "", ""), "bf16 container: the same tensors, every linear as <base>.bf16.w");
    Check(!b.Meta().at("quant").contains("mxfp8") && b.Meta().value("model_arch", "") == "inkling2_loop", "bf16 container: no quant.mxfp8 block");
    bool bf_pass = true, bf_lin = true, bf_exp = true;
    for (const auto& kv : pass) bf_pass = bf_pass && b.Bytes(kv.first) == c.Bytes(kv.first);
    for (const auto& kv : body) bf_lin = bf_lin && b.Bytes(kv.first + ".bf16.w") == tiny.src.at(kv.second).bytes;
    for (int64_t layer : {0, 5, 12, 15})
      for (const char* which : {"w13", "w2"}) {
        const bool w13 = std::string(which) == "w13";
        const auto xp = OpenExpert(tiny.src, static_cast<int>(layer), which, E, w13 ? 2 * I : H, w13 ? H : I);
        for (int64_t e : {0, 3, 7})
          bf_exp = bf_exp && b.Bytes("text.layers." + std::to_string(layer) + ".moe.experts." + std::to_string(e) + (w13 ? ".gate_up" : ".down") + ".bf16.w") ==
                                 Bf16Bytes(ExpertF32(*xp, e, w13));
      }
    Check(bf_pass && bf_lin, "bf16 container: non-linears and attention linears are the source's bytes");
    Check(bf_exp, "bf16 container: experts (NVFP4 and MXFP8 layers) are bf16 of the f32 decode, w13 de-interleaved");
    Check(b.Bytes("lm_head.bf16.w") == Bf16Bytes(head_w) && b.Shape("lm_head.bf16.w") == std::vector<int64_t>({96, H, 2}),
          "bf16 container: lm_head [96, H] with the 6 zero padding rows");
  }

  // ---- --keep-bf16 with mxfp8 ----
  rc = convert("--layouts mxfp8 --no-bf16 --threads 2 --keep-bf16 \"attn\\.o$|moe\\.experts\\.3\\.|^lm_head$\"");
  Check(rc == 0, "--keep-bf16 with mxfp8: converts (rc=" + std::to_string(rc) + ")");
  const Container k = ReadContainer(out);
  if (k.ok) {
    auto kept = [](const std::string& n) {
      return (n.size() >= 6 && n.compare(n.size() - 6, 6, "attn.o") == 0) || n.find("moe.experts.3.") != std::string::npos || n == "lm_head";
    };
    CheckNames(k, NamesFor(expect, mx::kWqSuffix, mx::kWsSuffix, kept), "--keep-bf16: matched linears are <base>.bf16.w ONLY, the rest mxfp8");
    const auto xp = OpenExpert(tiny.src, 12, "w13", E, 2 * I, H);
    Check(k.Bytes("text.layers.12.moe.experts.3.gate_up.bf16.w") == Bf16Bytes(ExpertF32(*xp, 3, true)) &&
              k.Bytes("text.layers.12.moe.experts.4.gate_up.mxfp8.wq") == c.Bytes("text.layers.12.moe.experts.4.gate_up.mxfp8.wq") &&
              k.Bytes("text.layers.4.attn.q.mxfp8.wq") == c.Bytes("text.layers.4.attn.q.mxfp8.wq"),
          "--keep-bf16: a kept MXFP8-source expert is bf16 of its decode; unmatched linears are unchanged");
    Check(k.Meta().at("r4dx_convert_run").at("keep_bf16_linears").size() == 16 + 2 * 16 + 1 && k.Meta().at("quant").contains("mxfp8"),
          "--keep-bf16: run record lists 49 kept linears (16 attn.o + 32 experts + lm_head); quant.mxfp8 still present");
    Check(k.Shape("lm_head.bf16.w") == std::vector<int64_t>({96, H, 2}), "--keep-bf16 lm_head: [96, H] bf16");
  } else {
    Check(false, "--keep-bf16 container unreadable");
  }

  // ---- --plan-only ----
  {
    std::error_code ec;
    fs::remove(out, ec);
    const int r = Run(in + "--output \"" + out.u8string() + "\" --layouts mxfp8 --no-bf16 --plan-only", log);
    Check(r == 0 && !fs::exists(out) && log_has("plan-only"), "--plan-only: plans every tensor, writes no output (the header-only audit path)");
  }

  // ---- refusals ----
  {
    int r = convert("");
    Check(r != 0 && log_has("mxfp8 or bf16") && !fs::exists(out), "default --layouts (w4a16) is refused with a pointer to mxfp8 / bf16");
    r = convert("--layouts mxfp8");
    Check(r != 0 && log_has("exactly one body layout") && !fs::exists(out), "--layouts mxfp8 without --no-bf16 (a bf16 companion) is refused");
    r = convert("--layouts w8a16 --no-bf16");
    Check(r != 0 && log_has("mxfp8 or bf16"), "--layouts w8a16 is refused");
    r = convert("--layouts mxfp8 --no-bf16 --layers 4");
    Check(r != 0 && log_has("--layers"), "--layers N < layer count is refused");
    r = convert("--layouts mxfp8 --no-bf16 --vision on");
    Check(r != 0 && log_has("--vision"), "--vision on is refused");
    r = convert("--layouts mxfp8 --no-bf16 --mtp on");
    Check(r != 0 && log_has("--mtp"), "--mtp on is refused");
    r = convert("--layouts mxfp8 --no-bf16 --rotate q2a");
    Check(r != 0 && log_has("--rotate"), "--rotate is refused");
    r = convert("--layouts mxfp8 --no-bf16 --quant search");
    Check(r != 0 && log_has("--quant"), "--quant search is refused");
    r = convert("--layouts mxfp8 --no-bf16 --keep-bf16 \"(\"");
    Check(r != 0 && log_has("invalid") && !fs::exists(out), "an invalid --keep-bf16 regex fails before any output");
  }

  // ---- the coverage audit, on mutated copies of the checkpoint ----
  {
    auto broken = [&](const std::string& tag, const std::function<void(Ckpt&)>& mutate) {
      const fs::path d = root / ("ckpt_" + tag);
      fs::create_directories(d);
      Ckpt m = tiny.src;
      mutate(m);
      WriteSafetensors(d / "model.safetensors", m);
      WriteWhole(d / "config.json", tiny.config.dump(2));
      std::error_code ec;
      fs::remove(out, ec);
      return Run("--input \"" + d.u8string() + "\" --output \"" + out.u8string() + "\" --layouts mxfp8 --no-bf16 --threads 2", log);
    };
    int r = broken("extra", [](Ckpt& m) { m["model.llm.layers.0.mystery.weight"] = {"BF16", {4}, std::string(8, '\0')}; });
    Check(r != 0 && log_has("mystery") && log_has("not allow-listed") && !fs::exists(out), "audit: an unconsumed tensor fails the run and NO output file is created");
    r = broken("visual", [](Ckpt& m) { m["model.visual.extra.weight"] = {"BF16", {4}, std::string(8, '\0')}; });
    Check(r == 0 && fs::exists(out) && log_has("allow-listed and skipped"), "audit: model.visual.* is allow-listed and logged");
    r = broken("missing", [](Ckpt& m) { m.erase("model.llm.layers.3.mlp.gate.bias"); });
    Check(r != 0 && log_has("mlp.gate.bias") && !fs::exists(out), "a missing tensor fails before the output exists");
    r = broken("shape", [](Ckpt& m) {
      m["model.llm.layers.5.attn.wq_du.weight"].shape = {64, 63};
      m["model.llm.layers.5.attn.wq_du.weight"].bytes.resize(64 * 63 * 2);
    });
    Check(r != 0 && log_has("shape") && !fs::exists(out), "a wrong shape fails before the output exists");
    r = broken("nan", [](Ckpt& m) { m["model.llm.layers.13.mlp.experts.w2_weight"].bytes[10] = static_cast<char>(0x7F); });
    Check(r != 0 && log_has("NaN"), "a NaN e4m3 code in a checkpoint expert fails the conversion");
    r = broken("scale255", [](Ckpt& m) { m["model.llm.layers.14.mlp.experts.w13_weight_scale"].bytes[5] = static_cast<char>(0xFF); });
    Check(r != 0 && log_has("255"), "E8M0 scale byte 255 in a checkpoint expert fails the conversion");
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main() {
  // An exception escaping a stage would otherwise end in std::terminate (0xc0000409 on Windows) with the buffered
  // stdout lost, which tells nothing about where it failed.
  const char* stage = "start";
  try {
    stage = "TestConfig";
    TestConfig();
    stage = "TestMerge";
    TestMerge();
    stage = "TestRealGeometry";
    TestRealGeometry();
#if defined(R4DX_CONVERT_EXE)
    stage = "TestExe";
    TestExe();
#else
    std::printf("SKIP: (b) needs the r4dx-convert target\n");
#endif
  } catch (const std::exception& e) {
    std::fflush(stdout);
    std::fprintf(stderr, "uncaught exception in %s: %s\n", stage, e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all inkling2 layout checks passed\n");
  return 0;
}
