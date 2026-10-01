// r4dx_convert::ShardedModel single-file fallback (docs/gemma4-plan.md M1-3): a directory with no
// model.safetensors.index.json but one model.safetensors (Huihui Gemma 4 12B) opens that file. The
// test writes tiny safetensors files into a fresh temp directory, so it needs no fixture and no
// weights. Also pins the unchanged paths: an index still wins and routes across shards, and a
// directory with neither throws. CPU-only, no HIP.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/safetensors_reader.hpp"

namespace fs = std::filesystem;

namespace {

bool g_ok = true;

bool Check(const char* label, bool cond) {
  std::printf("%-56s %s\n", label, cond ? "OK" : "FAIL");
  if (!cond) g_ok = false;
  return cond;
}

struct T {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// Writes a valid .safetensors file: u64 header length, JSON header, then the tensors back to back.
void WriteSafetensors(const fs::path& path, const std::vector<T>& tensors) {
  nlohmann::json h = nlohmann::json::object();
  h["__metadata__"] = {{"format", "pt"}};
  uint64_t off = 0;
  for (const T& t : tensors) {
    h[t.name] = {{"dtype", t.dtype},
                 {"shape", t.shape},
                 {"data_offsets", {off, off + t.bytes.size()}}};
    off += t.bytes.size();
  }
  const std::string js = h.dump();
  const uint64_t len = js.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&len), 8);
  f.write(js.data(), static_cast<std::streamsize>(js.size()));
  for (const T& t : tensors) f.write(reinterpret_cast<const char*>(t.bytes.data()), t.bytes.size());
}

T Make(const std::string& name, std::vector<int64_t> shape, uint8_t seed) {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  T t{name, "BF16", std::move(shape), {}};
  for (int64_t i = 0; i < n * 2; ++i) t.bytes.push_back(static_cast<uint8_t>(seed + i));
  return t;
}

bool BytesEqual(r4dx_convert::ShardedModel& m, const T& t) {
  const auto& meta = m.Meta(t.name);
  return meta.dtype == t.dtype && meta.shape == t.shape &&
         std::memcmp(m.Data(t.name), t.bytes.data(), t.bytes.size()) == 0;
}

template <class Fn>
bool Throws(Fn fn) {
  try {
    fn();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  using r4dx_convert::ShardedModel;
  const fs::path root = fs::temp_directory_path() / ("r4dx_sharded_single_" + std::to_string(::GetCurrentProcessId()));
  { std::error_code ec; fs::remove_all(root, ec); }

  // ---- single file, no index ----
  const T a = Make("model.language_model.embed_tokens.weight", {4, 8}, 1);
  const T b = Make("model.language_model.layers.0.self_attn.q_proj.weight", {2, 16}, 77);
  const T c = Make("model.vision_tower.norm.weight", {3}, 200);
  {
    const fs::path d = root / "single";
    fs::create_directories(d);
    WriteSafetensors(d / "model.safetensors", {a, b, c});
    ShardedModel m(d.string());
    Check("single: Has every tensor", m.Has(a.name) && m.Has(b.name) && m.Has(c.name));
    Check("single: Has(missing) false", !m.Has("nope"));
    Check("single: AllNames has all 3", m.AllNames().size() == 3);
    Check("single: tensor a dtype/shape/bytes", BytesEqual(m, a));
    Check("single: tensor b dtype/shape/bytes", BytesEqual(m, b));
    Check("single: tensor c dtype/shape/bytes", BytesEqual(m, c));
    Check("single: missing tensor throws", Throws([&] { m.Meta("nope"); }));
  }

  // ---- sharded with an index: the index still wins and routes per shard ----
  {
    const fs::path d = root / "sharded";
    fs::create_directories(d);
    WriteSafetensors(d / "model-00001-of-00002.safetensors", {a});
    WriteSafetensors(d / "model-00002-of-00002.safetensors", {b});
    // A stray model.safetensors that must be ignored when an index exists.
    WriteSafetensors(d / "model.safetensors", {c});
    nlohmann::json idx = {{"metadata", {{"total_size", 0}}},
                          {"weight_map",
                           {{a.name, "model-00001-of-00002.safetensors"},
                            {b.name, "model-00002-of-00002.safetensors"}}}};
    std::ofstream(d / "model.safetensors.index.json") << idx.dump();
    ShardedModel m(d.string());
    Check("sharded: index tensors present", m.Has(a.name) && m.Has(b.name));
    Check("sharded: stray model.safetensors ignored", !m.Has(c.name) && m.AllNames().size() == 2);
    Check("sharded: tensor a bytes", BytesEqual(m, a));
    Check("sharded: tensor b bytes", BytesEqual(m, b));
  }

  // ---- neither ----
  {
    const fs::path d = root / "empty";
    fs::create_directories(d);
    Check("neither: throws", Throws([&] { ShardedModel m(d.string()); }));
  }

  // ---- single file that is truncated reports the file's own error, not "not found" ----
  {
    const fs::path d = root / "trunc";
    fs::create_directories(d);
    WriteSafetensors(d / "model.safetensors", {a, b});
    fs::resize_file(d / "model.safetensors", fs::file_size(d / "model.safetensors") - 4);
    bool got = false;
    try {
      ShardedModel m(d.string());
    } catch (const std::runtime_error& e) {
      got = std::string(e.what()).find("truncated") != std::string::npos;
    }
    Check("single: truncated file surfaces the reader's error", got);
  }

  { std::error_code ec; fs::remove_all(root, ec); }
  std::printf(g_ok ? "test_sharded_single_file: PASS\n" : "test_sharded_single_file: FAIL\n");
  return g_ok ? 0 : 1;
}
