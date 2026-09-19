// r4dx_convert::SafetensorsReader -- memory-mapped reader for one .safetensors shard, plus
// ShardedModel which follows a model.safetensors.index.json across every shard and opens each
// shard's mapping lazily (so a --layers N run that only touches a handful of shards never maps
// the rest -- streaming per-tensor reads over a partially-downloaded checkpoint, per the task
// brief). Windows-only (CreateFileMappingW/MapViewOfFile); this repo only ever builds on Windows.
#pragma once

#ifndef NOMINMAX
#define NOMINMAX  // windows.h's min/max macros shadow std::min/std::max otherwise
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert {

// Byte width of one element for the dtypes this checkpoint format can carry. Returns 0 for an
// unrecognized dtype, which callers treat as "skip the element-count cross-check" rather than a
// hard error (an unknown-but-in-bounds tensor is not this reader's problem to reject).
inline uint64_t SafetensorsDtypeSize(const std::string& dtype) {
  if (dtype == "BF16" || dtype == "F16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  return 0;
}

struct TensorMeta {
  std::string dtype;            // "BF16", "F32", ...
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // byte offsets into the shard's data region

  int64_t ElemCount() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

inline std::string WideToUtf8(const std::wstring& w) {
  if (w.empty()) return std::string();
  int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                 nullptr, nullptr);
  std::string s(len, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), len, nullptr,
                       nullptr);
  return s;
}

// One memory-mapped .safetensors file.
class SafetensorsReader {
 public:
  explicit SafetensorsReader(const std::wstring& path) : path_display_(WideToUtf8(path)) {
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("safetensors: cannot open shard (CreateFileW failed)");
    }
    LARGE_INTEGER sz{};
    GetFileSizeEx(file_, &sz);
    file_size_ = static_cast<uint64_t>(sz.QuadPart);

    mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) throw std::runtime_error("safetensors: CreateFileMappingW failed");
    view_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!view_) throw std::runtime_error("safetensors: MapViewOfFile failed");

    if (file_size_ < 8) throw std::runtime_error("safetensors: file too small for header");
    uint64_t header_len = 0;
    std::memcpy(&header_len, view_, 8);
    if (8 + header_len > file_size_) throw std::runtime_error("safetensors: header length overruns file");

    std::string header_json(reinterpret_cast<const char*>(view_ + 8), header_len);
    data_start_ = 8 + header_len;
    nlohmann::json j = nlohmann::json::parse(header_json);
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      TensorMeta m;
      m.dtype = it.value().at("dtype").get<std::string>();
      for (auto& d : it.value().at("shape")) m.shape.push_back(d.get<int64_t>());
      auto off = it.value().at("data_offsets");
      m.begin = off.at(0).get<uint64_t>();
      m.end = off.at(1).get<uint64_t>();

      // Review finding (major): this shard may be a partially-downloaded checkpoint (the task's
      // own working situation), and the header comment above advertises "streaming per-tensor
      // reads over a partially-downloaded checkpoint" as a design goal. Without this check, a
      // truncated shard reads garbage/zeros (bytes in the final mapped page past EOF) or crashes
      // with an unhelpful access violation, and either way the resulting container looks valid
      // (correct header, correct config_sha256) with silently wrong weight data baked in.
      if (m.end < m.begin) {
        throw std::runtime_error("safetensors: tensor '" + it.key() + "' in " + path_display_ +
                                  " has data_offsets end < begin (" + std::to_string(m.end) + " < " +
                                  std::to_string(m.begin) + ")");
      }
      if (data_start_ + m.end > file_size_) {
        throw std::runtime_error(
            "safetensors: tensor '" + it.key() + "' in " + path_display_ + " needs bytes [" +
            std::to_string(data_start_ + m.begin) + "," + std::to_string(data_start_ + m.end) +
            ") but the file is only " + std::to_string(file_size_) +
            " bytes -- shard is truncated or still downloading");
      }
      const uint64_t dsize = SafetensorsDtypeSize(m.dtype);
      if (dsize != 0) {
        const uint64_t expected = static_cast<uint64_t>(m.ElemCount()) * dsize;
        if (m.end - m.begin != expected) {
          throw std::runtime_error(
              "safetensors: tensor '" + it.key() + "' in " + path_display_ + " has byte span " +
              std::to_string(m.end - m.begin) + " but shape*dtype implies " +
              std::to_string(expected) + " bytes (dtype " + m.dtype + ")");
        }
      }
      tensors_.emplace(it.key(), m);
    }
  }

  ~SafetensorsReader() {
    if (view_) UnmapViewOfFile(view_);
    if (mapping_) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
  }

  SafetensorsReader(const SafetensorsReader&) = delete;
  SafetensorsReader& operator=(const SafetensorsReader&) = delete;

  bool Has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorMeta& Meta(const std::string& name) const { return tensors_.at(name); }
  const uint8_t* Data(const std::string& name) const {
    const auto& m = tensors_.at(name);
    return view_ + data_start_ + m.begin;
  }

 private:
  std::string path_display_;
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
  const uint8_t* view_ = nullptr;
  uint64_t file_size_ = 0;
  uint64_t data_start_ = 0;
  std::unordered_map<std::string, TensorMeta> tensors_;
};

inline std::wstring Utf8ToWide(const std::string& s) {
  if (s.empty()) return L"";
  int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(len, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), len);
  return w;
}

// Follows model.safetensors.index.json across every shard it names, opening each shard's mapping
// the first time one of its tensors is requested.
class ShardedModel {
 public:
  explicit ShardedModel(const std::string& model_dir) : model_dir_(model_dir) {
    // Wide open, matching SafetensorsReader's CreateFileW (review finding, minor: narrow
    // ifstream/fopen and CreateFileW can resolve a non-ASCII path to different files under a
    // non-UTF-8 ANSI codepage).
    const std::string index_path = model_dir + "\\model.safetensors.index.json";
    std::ifstream f(Utf8ToWide(index_path).c_str());
    if (!f) throw std::runtime_error("ShardedModel: model.safetensors.index.json not found in " + model_dir);
    nlohmann::json j;
    f >> j;
    const auto& weight_map = j.at("weight_map");
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      name_to_shard_[it.key()] = it.value().get<std::string>();
    }
  }

  bool Has(const std::string& name) const { return name_to_shard_.count(name) != 0; }

  const TensorMeta& Meta(const std::string& name) { return Shard(name).Meta(name); }
  const uint8_t* Data(const std::string& name) { return Shard(name).Data(name); }

  // Every top-level tensor name in the checkpoint (all shards), for enumeration/mapping checks.
  std::vector<std::string> AllNames() const {
    std::vector<std::string> names;
    names.reserve(name_to_shard_.size());
    for (auto& kv : name_to_shard_) names.push_back(kv.first);
    return names;
  }

 private:
  SafetensorsReader& Shard(const std::string& tensor_name) {
    auto it = name_to_shard_.find(tensor_name);
    if (it == name_to_shard_.end())
      throw std::runtime_error("ShardedModel: tensor not in index: " + tensor_name);
    const std::string& shard_file = it->second;
    auto cached = open_shards_.find(shard_file);
    if (cached != open_shards_.end()) return *cached->second;
    auto reader = std::make_unique<SafetensorsReader>(Utf8ToWide(model_dir_ + "\\" + shard_file));
    auto& ref = *reader;
    open_shards_.emplace(shard_file, std::move(reader));
    return ref;
  }

  std::string model_dir_;
  std::unordered_map<std::string, std::string> name_to_shard_;
  std::unordered_map<std::string, std::unique_ptr<SafetensorsReader>> open_shards_;
};

}  // namespace r4dx_convert
