// tests/convert/test_convert_qwen4exp_awq.cpp -- CPU-only (ctest `convert_qwen4exp_awq`), plan M1-2.
//
// The virtual AWQ source (awq_source.{hpp,cpp}): a compressed-tensors pack-quantized quad
// `<b>.weight_packed/_scale/_zero_point/_shape` as one bf16 `<b>.weight` through ShardedModel's overlay.
//   (a) DequantRows against an independent model of the format (a packer written here, bf16 rounding by
//       neighbour comparison, not by the bit trick): whole tensors, row ranges, a row count that is not a
//       multiple of 8 (padded zero-point nibbles), group 128, negative / zero scales;
//   (b) the overlay inside a ShardedModel (two shards + index, plain tensors next to the quads): Has / Meta /
//       Data / DataRows / AllNames (component names hidden, virtual names added), ReadTensorAsFloat and
//       CopyRawBf16 on a virtual tensor, and 8 threads reading concurrently (per-thread buffers);
//   (c) every malformed quad and every unsupported quantization_config is refused with a precise message;
//   (d) against the Python reference: tools/reference/qwen4exp/convert_fixtures.py awq-mini writes quads with
//       compressed-tensors' own packer and the expected bf16 from awq_dequant.py -- every one must be equal bit
//       for bit -- and, when present, `real-spot` does the same for tensors of the real checkpoint.
//       (d) prints SKIP when the fixtures are not generated.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "awq_source.hpp"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace fs = std::filesystem;
namespace awq = r4dx_convert::awq;
using nlohmann::json;
using r4dx_convert::ShardedModel;

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
    if (!ok) std::fprintf(stderr, "  (threw, but not with '%s': %s)\n", substr.c_str(), e.what());
    return ok;
  }
  return false;
}

struct TempDir {
  fs::path p;
  TempDir() {
    static std::atomic<int> n{0};
    p = fs::temp_directory_path() / ("r4dx_q4exp_awq_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                                      std::to_string(n++));
    fs::create_directories(p);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(p, ec);
  }
};

// ---- an independent model of the format ----------------------------------------------------------------------

// bf16 round-to-nearest-even of a finite float, by comparing the two neighbouring bf16 values.
uint16_t Bf16Nearest(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lo_bits = u & 0xFFFF0000u;           // truncation toward zero
  const uint32_t hi_bits = lo_bits + 0x10000u;        // the next magnitude up
  float lo, hi;
  std::memcpy(&lo, &lo_bits, 4);
  std::memcpy(&hi, &hi_bits, 4);
  const double dlo = std::fabs(static_cast<double>(f) - lo), dhi = std::fabs(static_cast<double>(hi) - f);
  if (dlo < dhi) return static_cast<uint16_t>(lo_bits >> 16);
  if (dhi < dlo) return static_cast<uint16_t>(hi_bits >> 16);
  return static_cast<uint16_t>(((lo_bits >> 16) & 1) == 0 ? lo_bits >> 16 : hi_bits >> 16);  // tie: even mantissa
}

float Bf16Value(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Quad {
  int64_t N = 0, K = 0;
  std::vector<uint8_t> packed, scale, zp, shape;  // raw little-endian bytes of the four tensors
  std::vector<uint16_t> expected;                  // bf16 bits [N, K]
};

Quad MakeQuad(int64_t N, int64_t K, uint32_t seed) {
  const int G = 128;
  std::mt19937 rng(seed);
  Quad q;
  q.N = N;
  q.K = K;
  const int64_t groups = K / G;
  std::vector<uint8_t> nib(static_cast<size_t>(N * K)), z(static_cast<size_t>(N * groups));
  std::vector<uint16_t> sc(static_cast<size_t>(N * groups));
  for (auto& v : nib) v = static_cast<uint8_t>(rng() & 15);
  for (auto& v : z) v = static_cast<uint8_t>(rng() & 15);
  for (size_t i = 0; i < sc.size(); ++i) {
    float s = (static_cast<float>(rng() % 2000) + 1.0f) * 1e-5f * ((rng() % 7 == 0) ? -1.0f : 1.0f);
    if (rng() % 11 == 0) s = 0.0f;
    sc[i] = Bf16Nearest(s);
  }
  // packed [N, K/8]: nibble j of word c is column 8c + j
  q.packed.resize(static_cast<size_t>(N * (K / 8) * 4));
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < K / 8; ++c) {
      uint32_t w = 0;
      for (int j = 0; j < 8; ++j) w |= static_cast<uint32_t>(nib[static_cast<size_t>(n * K + c * 8 + j)]) << (4 * j);
      std::memcpy(q.packed.data() + static_cast<size_t>((n * (K / 8) + c) * 4), &w, 4);
    }
  // zero points [ceil(N/8), G]: nibble n % 8 of row n / 8
  const int64_t zrows = (N + 7) / 8;
  q.zp.assign(static_cast<size_t>(zrows * groups * 4), 0);
  for (int64_t r = 0; r < zrows; ++r)
    for (int64_t g = 0; g < groups; ++g) {
      uint32_t w = 0;
      for (int j = 0; j < 8; ++j) {
        const int64_t n = r * 8 + j;
        if (n < N) w |= static_cast<uint32_t>(z[static_cast<size_t>(n * groups + g)]) << (4 * j);
        else w |= static_cast<uint32_t>(rng() & 15) << (4 * j);  // padding nibbles are dropped
      }
      std::memcpy(q.zp.data() + static_cast<size_t>((r * groups + g) * 4), &w, 4);
    }
  q.scale.resize(sc.size() * 2);
  std::memcpy(q.scale.data(), sc.data(), q.scale.size());
  q.shape.resize(16);
  std::memcpy(q.shape.data(), &N, 8);
  std::memcpy(q.shape.data() + 8, &K, 8);
  q.expected.resize(static_cast<size_t>(N * K));
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k) {
      const int64_t g = k / G;
      const int d = static_cast<int>(nib[static_cast<size_t>(n * K + k)]) - static_cast<int>(z[static_cast<size_t>(n * groups + g)]);
      q.expected[static_cast<size_t>(n * K + k)] = Bf16Nearest(static_cast<float>(d) * Bf16Value(sc[static_cast<size_t>(n * groups + g)]));
    }
  return q;
}

// ---- safetensors writing ---------------------------------------------------------------------------------------

struct Entry {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

void WriteSafetensors(const fs::path& path, const std::vector<Entry>& entries) {
  json header = json::object();
  uint64_t off = 0;
  for (const Entry& e : entries) {
    header[e.name] = {{"dtype", e.dtype}, {"shape", e.shape}, {"data_offsets", {off, off + e.bytes.size()}}};
    off += e.bytes.size();
  }
  std::string h = header.dump();
  h.append((8 - h.size() % 8) % 8, ' ');
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  const uint64_t hl = h.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const Entry& e : entries) f.write(reinterpret_cast<const char*>(e.bytes.data()), static_cast<std::streamsize>(e.bytes.size()));
}

void AddQuad(std::vector<Entry>& out, const std::string& base, const Quad& q, bool with_scale = true) {
  const int64_t groups = q.K / 128;
  out.push_back({base + ".weight_packed", "I32", {q.N, q.K / 8}, q.packed});
  if (with_scale) out.push_back({base + ".weight_scale", "BF16", {q.N, groups}, q.scale});
  out.push_back({base + ".weight_zero_point", "I32", {(q.N + 7) / 8, groups}, q.zp});
  out.push_back({base + ".weight_shape", "I64", {2}, q.shape});
}

Entry PlainBf16(const std::string& name, std::vector<int64_t> shape, uint32_t seed) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  std::mt19937 rng(seed);
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (auto& x : v) x = Bf16Nearest(static_cast<float>(static_cast<int>(rng() % 2001) - 1000) * 1e-3f);
  Entry e{name, "BF16", std::move(shape), {}};
  e.bytes.resize(v.size() * 2);
  std::memcpy(e.bytes.data(), v.data(), e.bytes.size());
  return e;
}

void WriteIndex(const fs::path& dir, const std::map<std::string, std::string>& weight_map) {
  json wm = json::object();
  for (const auto& kv : weight_map) wm[kv.first] = kv.second;
  std::ofstream f(dir / "model.safetensors.index.json");
  f << json{{"metadata", json::object()}, {"weight_map", wm}}.dump();
}

json CheckpointQuantConfig() {
  return {{"quantization_config",
           {{"format", "pack-quantized"},
            {"quant_method", "compressed-tensors"},
            {"config_groups",
             {{"group_0",
               {{"format", "pack-quantized"},
                {"input_activations", nullptr},
                {"output_activations", nullptr},
                {"targets", {"Linear"}},
                {"weights",
                 {{"actorder", nullptr},
                  {"block_structure", nullptr},
                  {"dynamic", false},
                  {"group_size", 128},
                  {"num_bits", 4},
                  {"observer", "memoryless_minmax"},
                  {"strategy", "group"},
                  {"symmetric", false},
                  {"type", "int"},
                  {"zp_dtype", "torch.int8"}}}}}}}}}};
}

std::vector<uint8_t> AsBytes(const std::vector<uint16_t>& v) {
  std::vector<uint8_t> b(v.size() * 2);
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// ---- (a) DequantRows ------------------------------------------------------------------------------------------

void TestDequantRows() {
  std::printf("---- (a) DequantRows vs an independent model of the format ----\n");
  for (const auto& [N, K] : std::vector<std::pair<int64_t, int64_t>>{{8, 128}, {11, 256}, {24, 384}, {640, 128}, {5, 640}}) {
    const Quad q = MakeQuad(N, K, static_cast<uint32_t>(N * 31 + K));
    std::vector<uint16_t> out(static_cast<size_t>(N * K), 0xDEAD);
    awq::DequantRows(q.packed.data(), q.scale.data(), q.zp.data(), N, K, 128, 0, N, out.data());
    Check(out == q.expected, "whole tensor [" + std::to_string(N) + ", " + std::to_string(K) + "] is bit-equal");
    if (N >= 11) {
      const int64_t r0 = 3, nr = N - 5;
      std::vector<uint16_t> part(static_cast<size_t>(nr * K), 0xDEAD);
      awq::DequantRows(q.packed.data(), q.scale.data(), q.zp.data(), N, K, 128, r0, nr, part.data());
      Check(std::equal(part.begin(), part.end(), q.expected.begin() + static_cast<std::ptrdiff_t>(r0 * K)),
            "rows [3, " + std::to_string(N - 2) + ") alone are bit-equal (the zero point of row n is nibble n % 8)");
    }
  }
  Check(Throws([] { std::vector<uint16_t> o(1); awq::DequantRows(nullptr, nullptr, nullptr, 8, 192, 128, 0, 1, o.data()); },
               "K=192 is not a multiple of group 128"),
        "K not a multiple of the group is refused");
  Check(Throws([] { std::vector<uint16_t> o(1); awq::DequantRows(nullptr, nullptr, nullptr, 8, 128, 128, 7, 2, o.data()); },
               "outside N=8"),
        "a row range outside N is refused");
}

// ---- (b) the overlay in a ShardedModel -----------------------------------------------------------------------

void TestOverlay() {
  std::printf("---- (b) the overlay in a ShardedModel ----\n");
  TempDir t;
  const Quad a = MakeQuad(24, 256, 1), b = MakeQuad(16, 128, 2), c = MakeQuad(11, 384, 3);
  std::vector<Entry> s1, s2;
  AddQuad(s1, "m.a", a);
  s1.push_back(PlainBf16("m.plain.weight", {5, 7}, 9));
  AddQuad(s2, "m.b", b);
  AddQuad(s2, "m.c", c);
  s2.push_back(PlainBf16("m.norm.weight", {64}, 10));
  WriteSafetensors(t.p / "s1.safetensors", s1);
  WriteSafetensors(t.p / "s2.safetensors", s2);
  std::map<std::string, std::string> wm;
  for (const auto& e : s1) wm[e.name] = "s1.safetensors";
  for (const auto& e : s2) wm[e.name] = "s2.safetensors";
  WriteIndex(t.p, wm);

  ShardedModel model(t.p.string());
  const size_t before = model.AllNames().size();
  Check(before == wm.size(), "no overlay: AllNames is the index (" + std::to_string(before) + " names)");
  Check(awq::InstallAwqSource(model, json::object()) == 0 && model.AllNames().size() == before, "a config without quantization_config installs nothing");
  Check(awq::InstallAwqSource(model, json{{"quantization_config", {{"format", "other"}}}}) == 0, "another quantization format installs nothing");
  json cfg = CheckpointQuantConfig();
  Check(awq::InstallAwqSource(model, cfg) == 3, "three pack-quantized linears found");

  std::vector<std::string> names = model.AllNames();
  std::sort(names.begin(), names.end());
  const std::vector<std::string> want = {"m.a.weight", "m.b.weight", "m.c.weight", "m.norm.weight", "m.plain.weight"};
  Check(names == want, "AllNames: the four component names of every quad are gone, `<b>.weight` appears (and plain tensors stay)");
  Check(model.Has("m.a.weight") && model.Has("m.a.weight_packed") && !model.Has("m.zzz.weight"), "Has: virtual and raw names, not an unknown one");

  const Quad* quads[3] = {&a, &b, &c};
  const char* qn[3] = {"m.a.weight", "m.b.weight", "m.c.weight"};
  bool all_ok = true;
  for (int i = 0; i < 3; ++i) {
    const r4dx_convert::TensorMeta& m = model.Meta(qn[i]);
    all_ok = all_ok && m.dtype == "BF16" && m.shape == std::vector<int64_t>{quads[i]->N, quads[i]->K} &&
             m.ElemCount() * 2 == static_cast<int64_t>(quads[i]->expected.size() * 2);
    const uint8_t* d = model.Data(qn[i]);
    all_ok = all_ok && std::memcmp(d, quads[i]->expected.data(), quads[i]->expected.size() * 2) == 0;
  }
  Check(all_ok, "Meta (BF16 [N, K]) and Data (bit-equal to the model) of every virtual tensor");

  std::vector<uint8_t> scratch;
  const uint8_t* rows = model.DataRows("m.c.weight", 4, 6, scratch);
  Check(std::memcmp(rows, reinterpret_cast<const uint8_t*>(c.expected.data()) + 4 * 384 * 2, 6 * 384 * 2) == 0 && scratch.size() == 6 * 384 * 2,
        "DataRows of a virtual tensor: the rows' bytes, built in the caller's scratch");
  std::vector<uint8_t> scratch2;
  const uint8_t* plain_rows = model.DataRows("m.plain.weight", 2, 2, scratch2);
  Check(plain_rows == model.Data("m.plain.weight") + 2 * 7 * 2 && scratch2.empty(),
        "DataRows of a checkpoint tensor: straight out of the mapping, scratch untouched");
  Check(Throws([&] { model.DataRows("m.plain.weight", 4, 2, scratch2); }, "not inside the 2-D tensor"), "DataRows outside the tensor is refused");

  // the codec readers see the virtual tensor as bf16
  const std::vector<float> f = r4dx_convert::ReadTensorAsFloat(model, "m.b.weight");
  bool f_ok = f.size() == b.expected.size();
  for (size_t i = 0; f_ok && i < f.size(); ++i) f_ok = f[i] == Bf16Value(b.expected[i]);
  Check(f_ok, "ReadTensorAsFloat on a virtual tensor");
  Check(r4dx_convert::CopyRawBf16(model, "m.a.weight") == AsBytes(a.expected), "CopyRawBf16 on a virtual tensor");

  // 8 threads, 300 reads each, random tensor and range: per-thread buffers never mix
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int th = 0; th < 8; ++th)
    threads.emplace_back([&, th] {
      std::mt19937 rng(100 + th);
      std::vector<uint8_t> sc;
      for (int it = 0; it < 300; ++it) {
        const int i = static_cast<int>(rng() % 3);
        const Quad& q = *quads[i];
        if (rng() % 2) {
          if (std::memcmp(model.Data(qn[i]), q.expected.data(), q.expected.size() * 2) != 0) ++bad;
        } else {
          const int64_t r0 = static_cast<int64_t>(rng() % static_cast<uint32_t>(q.N)), nr = 1 + static_cast<int64_t>(rng() % static_cast<uint32_t>(q.N - r0));
          const uint8_t* d = model.DataRows(qn[i], r0, nr, sc);
          if (std::memcmp(d, reinterpret_cast<const uint8_t*>(q.expected.data()) + r0 * q.K * 2, static_cast<size_t>(nr * q.K * 2)) != 0) ++bad;
        }
      }
    });
  for (auto& th : threads) th.join();
  Check(bad == 0, "8 threads x 300 concurrent Data / DataRows reads, every one bit-equal");
}

// ---- (c) refusals ---------------------------------------------------------------------------------------------

void TestRefusals() {
  std::printf("---- (c) malformed quads and unsupported schemes ----\n");
  const json cfg = CheckpointQuantConfig();
  auto install = [&](const std::vector<Entry>& entries, const json& config) {
    TempDir t;
    WriteSafetensors(t.p / "s.safetensors", entries);
    std::map<std::string, std::string> wm;
    for (const auto& e : entries) wm[e.name] = "s.safetensors";
    WriteIndex(t.p, wm);
    ShardedModel model(t.p.string());
    return awq::InstallAwqSource(model, config);
  };
  const Quad q = MakeQuad(16, 256, 5);
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    Check(install(e, cfg) == 1, "the well-formed quad installs (control)");
  }
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q, /*with_scale=*/false);
    Check(Throws([&] { install(e, cfg); }, "has no companion 'x.lin.weight_scale'"), "a missing weight_scale is named");
  }
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    e.push_back(PlainBf16("x.lin.weight", {2, 2}, 1));
    Check(Throws([&] { install(e, cfg); }, "'x.lin.weight' exists next to"), "a quad next to a plain weight of the same name is refused");
  }
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    for (auto& en : e)
      if (en.name == "x.lin.weight_zero_point") en.shape = {1, 2}, en.bytes.resize(8);
    Check(Throws([&] { install(e, cfg); }, "'x.lin.weight_zero_point' is I32 [1,2], expected I32 [2,2]"), "a wrong zero-point shape is refused");
  }
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    for (auto& en : e)
      if (en.name == "x.lin.weight_scale") en.dtype = "F32", en.bytes.resize(en.bytes.size() * 2);
    Check(Throws([&] { install(e, cfg); }, "'x.lin.weight_scale' is F32"), "a non-bf16 scale is refused");
  }
  {
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    const int64_t wrong[2] = {16, 128};
    for (auto& en : e)
      if (en.name == "x.lin.weight_shape") std::memcpy(en.bytes.data(), wrong, 16);
    Check(Throws([&] { install(e, cfg); }, "says [16, 128] but weight_packed implies [16, 256]"), "a weight_shape that disagrees with weight_packed is refused");
  }
  {
    // K = 192: packed [N, 24], scale [N, 1.5] cannot exist -- a 128-group scale for 192 columns is refused by K % group
    std::vector<Entry> e;
    AddQuad(e, "x.lin", q);
    for (auto& en : e)
      if (en.name == "x.lin.weight_packed") en.shape = {16, 24}, en.bytes.resize(16 * 24 * 4);
    Check(Throws([&] { install(e, cfg); }, "K = 192 is not a multiple of the group size 128"), "K not a multiple of 128 is refused");
  }
  // quantization_config variants
  auto cfg_with = [&](const std::function<void(json&)>& mut) {
    json c = cfg;
    mut(c);
    return c;
  };
  auto w = [](json& c) -> json& { return c["quantization_config"]["config_groups"]["group_0"]["weights"]; };
  struct Case {
    const char* what;
    std::function<void(json&)> mut;
    const char* msg;
  };
  const std::vector<Case> cases = {
      {"num_bits 8", [&](json& c) { w(c)["num_bits"] = 8; }, "weights.num_bits is 8"},
      {"group_size 64", [&](json& c) { w(c)["group_size"] = 64; }, "weights.group_size is 64"},
      {"symmetric", [&](json& c) { w(c)["symmetric"] = true; }, "weights.symmetric is true"},
      {"strategy channel", [&](json& c) { w(c)["strategy"] = "channel"; }, "weights.strategy is \"channel\""},
      {"type float", [&](json& c) { w(c)["type"] = "float"; }, "weights.type is \"float\""},
      {"actorder", [&](json& c) { w(c)["actorder"] = "group"; }, "weights.actorder is \"group\""},
      {"dynamic", [&](json& c) { w(c)["dynamic"] = true; }, "weights.dynamic is true"},
      {"activations", [&](json& c) { c["quantization_config"]["config_groups"]["group_0"]["input_activations"] = {{"num_bits", 8}}; },
       "input_activations"},
      {"two groups", [&](json& c) { c["quantization_config"]["config_groups"]["group_1"] = c["quantization_config"]["config_groups"]["group_0"]; },
       "exactly one group"},
      {"quant_method", [&](json& c) { c["quantization_config"]["quant_method"] = "awq"; }, "quant_method is \"awq\""},
  };
  for (const Case& cs : cases) {
    const json c = cfg_with(cs.mut);
    Check(Throws([&] { awq::PackQuantizedScheme(c, nullptr); }, cs.msg), std::string("quantization_config refused: ") + cs.what);
  }
  awq::PackScheme sch;
  Check(awq::PackQuantizedScheme(cfg, &sch) && sch.num_bits == 4 && sch.group_size == 128, "the checkpoint's scheme (int4, group 128, asymmetric) parses");
  Check(!awq::PackQuantizedScheme(json{{"quantization_config", nullptr}}, nullptr) && !awq::PackQuantizedScheme(json::object(), nullptr), "null / absent quantization_config: inactive");
}

// ---- (d) the Python reference ---------------------------------------------------------------------------------

fs::path FixturesDir() {
  if (const char* e = std::getenv("R4DX_Q4EXP_FIXTURES")) return e;
  return "E:/models/r4dx/q4exp-fixtures/ConvertQ4";
}

void CompareToExpected(ShardedModel& model, r4dx_convert::SafetensorsReader& expected, const std::string& label) {
  size_t n = 0, bad = 0;
  uint64_t elems = 0;
  for (const std::string& name : expected.Names()) {
    ++n;
    const r4dx_convert::TensorMeta& em = expected.Meta(name);
    if (!model.Has(name)) {
      std::fprintf(stderr, "  missing virtual tensor %s\n", name.c_str());
      ++bad;
      continue;
    }
    const r4dx_convert::TensorMeta& m = model.Meta(name);
    const uint64_t bytes = static_cast<uint64_t>(em.ElemCount()) * 2;
    elems += static_cast<uint64_t>(em.ElemCount());
    if (m.dtype != "BF16" || m.shape != em.shape || std::memcmp(model.Data(name), expected.Data(name), bytes) != 0) {
      std::fprintf(stderr, "  differs: %s\n", name.c_str());
      ++bad;
    }
  }
  Check(n > 0 && bad == 0, label + ": " + std::to_string(n) + " dequantized linears (" + std::to_string(elems) + " weights) bit-equal to awq_dequant.py, " +
                               std::to_string(bad) + " different");
}

void TestPythonReference() {
  std::printf("---- (d) against tools/reference/qwen4exp/awq_dequant.py ----\n");
  const fs::path mini = FixturesDir() / "awq-mini";
  if (!fs::exists(mini / "expected_bf16.safetensors")) {
    std::printf("SKIP: %s not generated (python tools/reference/qwen4exp/convert_fixtures.py awq-mini)\n", mini.string().c_str());
  } else {
    ShardedModel model((mini / "hf").string());
    const json cfg = json::parse(std::ifstream(mini / "hf" / "config.json"));
    const size_t count = awq::InstallAwqSource(model, cfg);
    r4dx_convert::SafetensorsReader expected(r4dx_convert::Utf8ToWide((mini / "expected_bf16.safetensors").string()));
    Check(count == expected.Names().size(), "awq-mini: the overlay found " + std::to_string(count) + " linears");
    CompareToExpected(model, expected, "awq-mini (compressed-tensors' own packer)");
  }
  const fs::path spot = FixturesDir() / "real-spot";
  const fs::path real = "E:/models/r4dx/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ";
  if (!fs::exists(spot / "expected_bf16.safetensors") || !fs::exists(real / "config.json")) {
    std::printf("SKIP: real-spot not generated or the real checkpoint is absent (convert_fixtures.py real-spot)\n");
  } else {
    ShardedModel model(real.string());
    const json cfg = json::parse(std::ifstream(real / "config.json"));
    const size_t count = awq::InstallAwqSource(model, cfg);
    Check(count == 73884, "real checkpoint: 73,884 pack-quantized linears (QUANTIZATION_MANIFEST.json packed_linears)");
    r4dx_convert::SafetensorsReader expected(r4dx_convert::Utf8ToWide((spot / "expected_bf16.safetensors").string()));
    CompareToExpected(model, expected, "real checkpoint spot tensors");
  }
}

}  // namespace

int main() {
  TestDequantRows();
  TestOverlay();
  TestRefusals();
  TestPythonReference();
  std::printf("%s (%d failure(s))\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}
