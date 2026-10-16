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
    try {
      Open(path);
    } catch (...) {
      Release();  // a throwing constructor never runs the destructor
      throw;
    }
  }

  ~SafetensorsReader() { Release(); }

  SafetensorsReader(const SafetensorsReader&) = delete;
  SafetensorsReader& operator=(const SafetensorsReader&) = delete;

 private:
  void Release() {
    if (view_) UnmapViewOfFile(view_);
    if (mapping_) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    view_ = nullptr;
    mapping_ = nullptr;
    file_ = INVALID_HANDLE_VALUE;
  }

  void Open(const std::wstring& path) {
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
    if (header_len > file_size_ - 8) throw std::runtime_error("safetensors: header length overruns file");

    std::string header_json(reinterpret_cast<const char*>(view_ + 8), static_cast<size_t>(header_len));
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
      if (m.end > file_size_ - data_start_) {
        throw std::runtime_error(
            "safetensors: tensor '" + it.key() + "' in " + path_display_ + " needs bytes [" +
            std::to_string(data_start_ + m.begin) + "," + std::to_string(data_start_ + m.end) +
            ") but the file is only " + std::to_string(file_size_) +
            " bytes -- shard is truncated or still downloading");
      }
      for (int64_t d : m.shape) {
        if (d < 0) {
          throw std::runtime_error("safetensors: tensor '" + it.key() + "' in " + path_display_ +
                                    " has a negative shape dimension");
        }
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

 public:
  bool Has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorMeta& Meta(const std::string& name) const { return tensors_.at(name); }
  const uint8_t* Data(const std::string& name) const {
    const auto& m = tensors_.at(name);
    return view_ + data_start_ + m.begin;
  }
  // The JSON header's bytes exactly as in the file (from the mapping, so the same bytes the tensor
  // directory was parsed from).
  std::string HeaderJson() const {
    return std::string(reinterpret_cast<const char*>(view_ + 8), static_cast<size_t>(data_start_ - 8));
  }
  // Every tensor name in the file (no particular order; __metadata__ excluded).
  std::vector<std::string> Names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    return out;
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

// Tensors layered over a checkpoint's own (awq_source.hpp's virtual dequantised linears). With none
// installed ShardedModel is exactly what it was. An overlay tensor is an ordinary read-only 2-D (or
// N-D) tensor to every user of ShardedModel; its bytes are produced on demand.
class TensorOverlay {
 public:
  virtual ~TensorOverlay() = default;
  virtual bool Provides(const std::string& name) const = 0;
  // Provides(name) only. Shape/dtype of the tensor as the rest of the converter sees it.
  virtual const TensorMeta& Meta(const std::string& name) const = 0;
  // Provides(name) only. The whole tensor's bytes in a buffer owned by the overlay: valid only until
  // the next Data()/DataRows() of an overlay tensor on the SAME thread, and private to the calling
  // thread (any number of threads may call concurrently). Copy it before asking for another.
  virtual const uint8_t* Data(const std::string& name) const = 0;
  // Provides(name) only, 2-D: the row-major bytes of rows [row0, row0 + nrows), same lifetime rule
  // as Data(); `scratch` is the caller's buffer to build them in.
  virtual const uint8_t* DataRows(const std::string& name, int64_t row0, int64_t nrows,
                                  std::vector<uint8_t>& scratch) const = 0;
  // AllNames(): drop the checkpoint names the overlay replaces, add the ones it provides.
  virtual void RewriteNames(std::vector<std::string>& names) const = 0;
};

// Follows model.safetensors.index.json across every shard it names, opening each shard's mapping
// the first time one of its tensors is requested. A directory with no index but a single
// model.safetensors opens that file instead (all tensors, mapped at construction).
class ShardedModel {
 public:
  explicit ShardedModel(const std::string& model_dir) : model_dir_(model_dir) {
    // Wide open, matching SafetensorsReader's CreateFileW (review finding, minor: narrow
    // ifstream/fopen and CreateFileW can resolve a non-ASCII path to different files under a
    // non-UTF-8 ANSI codepage).
    const std::string index_path = model_dir + "\\model.safetensors.index.json";
    std::ifstream f(Utf8ToWide(index_path).c_str());
    if (!f) {
      // No index: a checkpoint small enough to be one file (Huihui Gemma 4 12B is a single 23.9 GB
      // model.safetensors) has no model.safetensors.index.json. Map every tensor of that file to it;
      // the reader stays open as the one shard.
      const std::string single = "model.safetensors";
      const std::wstring single_path = Utf8ToWide(model_dir + "\\" + single);
      if (GetFileAttributesW(single_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
        throw std::runtime_error("ShardedModel: neither model.safetensors.index.json nor model.safetensors "
                                 "found in " + model_dir);
      }
      auto reader = std::make_unique<SafetensorsReader>(single_path);  // a corrupt file throws its own error
      for (const std::string& n : reader->Names()) name_to_shard_[n] = single;
      open_shards_.emplace(single, std::move(reader));
      return;
    }
    nlohmann::json j;
    f >> j;
    const auto& weight_map = j.at("weight_map");
    for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
      name_to_shard_[it.key()] = it.value().get<std::string>();
    }
  }

  // Installed once, before any worker thread reads (the overlay is read-only afterwards).
  void SetOverlay(std::shared_ptr<const TensorOverlay> overlay) { overlay_ = std::move(overlay); }

  bool Has(const std::string& name) const {
    return name_to_shard_.count(name) != 0 || (overlay_ && overlay_->Provides(name));
  }

  const TensorMeta& Meta(const std::string& name) {
    if (overlay_ && overlay_->Provides(name)) return overlay_->Meta(name);
    return Shard(name).Meta(name);
  }
  // Whole-tensor bytes; an overlay tensor's live only until this thread's next overlay read
  // (TensorOverlay::Data).
  const uint8_t* Data(const std::string& name) {
    if (overlay_ && overlay_->Provides(name)) return overlay_->Data(name);
    return Shard(name).Data(name);
  }
  // Rows [row0, row0 + nrows) of a 2-D tensor, row-major: straight out of the mapping for a
  // checkpoint tensor (scratch untouched), built into `scratch` for an overlay tensor. Safe from
  // several threads once the tensor's shard has been opened (a prior Meta() does that).
  const uint8_t* DataRows(const std::string& name, int64_t row0, int64_t nrows,
                          std::vector<uint8_t>& scratch) {
    if (overlay_ && overlay_->Provides(name)) return overlay_->DataRows(name, row0, nrows, scratch);
    const TensorMeta& m = Shard(name).Meta(name);
    if (m.shape.size() != 2 || row0 < 0 || nrows < 0 || row0 + nrows > m.shape[0])
      throw std::runtime_error("ShardedModel::DataRows: rows [" + std::to_string(row0) + ", " +
                               std::to_string(row0 + nrows) + ") are not inside the 2-D tensor " + name);
    const uint64_t row_bytes = static_cast<uint64_t>(m.shape[1]) * SafetensorsDtypeSize(m.dtype);
    return Shard(name).Data(name) + static_cast<uint64_t>(row0) * row_bytes;
  }
  // The checkpoint's own tensor, the overlay bypassed (what an overlay builds its tensors from).
  const TensorMeta& RawMeta(const std::string& name) { return Shard(name).Meta(name); }
  const uint8_t* RawData(const std::string& name) { return Shard(name).Data(name); }

  // Every top-level tensor name in the checkpoint (all shards), for enumeration/mapping checks; with
  // an overlay, the names the overlay replaces are replaced by the ones it provides.
  std::vector<std::string> AllNames() const {
    std::vector<std::string> names;
    names.reserve(name_to_shard_.size());
    for (auto& kv : name_to_shard_) names.push_back(kv.first);
    if (overlay_) overlay_->RewriteNames(names);
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
  std::shared_ptr<const TensorOverlay> overlay_;
};

}  // namespace r4dx_convert
