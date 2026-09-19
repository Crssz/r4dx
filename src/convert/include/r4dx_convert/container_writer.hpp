// r4dx_convert::ContainerWriter -- writer for the container docs/container-format.md defines:
// safetensors-shaped shell (8-byte JSON-header length, JSON header, raw tensor bytes), every
// tensor dtype "U8", `__metadata__` carrying r4dx_format_version/model_id/config_sha256/
// produced_by/quant_summary/model_config.
//
// Two-phase so the whole model never has to sit in RAM at once: `Plan()` every tensor by name/
// shape/byte-size first (byte sizes are a pure function of shape+layout, no data needed), which
// fixes every tensor's offset; `FinalizeHeader()` writes the 8-byte length + JSON header and
// pre-sizes the file; then `WriteTensor()` seeks to that tensor's fixed offset and writes its
// bytes, called once per tensor in any order (streaming per-tensor from the source shards, one
// tensor's worth of packed bytes resident at a time -- never the whole model).
#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"  // Utf8ToWide -- see FinalizeHeader's _wfopen note

namespace r4dx_convert {

class ContainerWriter {
 public:
  // Registers a tensor of `nbytes` raw bytes and `shape` (as it appears in the JSON header --
  // r4dx always stores dtype "U8", so `shape` already carries any trailing byte-width axis the
  // caller wants, e.g. [rows,cols,2] for a bf16 tensor per docs/container-format.md). Must be
  // called for every tensor before FinalizeHeader(); order doesn't matter.
  void Plan(const std::string& name, std::vector<int64_t> shape, uint64_t nbytes) {
    if (finalized_) throw std::runtime_error("ContainerWriter: Plan() after FinalizeHeader()");
    if (index_.count(name)) throw std::runtime_error("ContainerWriter: duplicate tensor name " + name);
    Entry e;
    e.shape = std::move(shape);
    e.offset = cursor_;
    e.nbytes = nbytes;
    cursor_ += nbytes;
    index_[name] = static_cast<int64_t>(plan_.size());
    plan_.push_back(std::move(e));
    plan_names_.push_back(name);
  }

  // Writes the 8-byte header length + JSON header to `path` and pre-sizes the file to header +
  // all planned tensor bytes. `metadata` is the full `__metadata__` object (already assembled by
  // the caller: format version, model id, config hash, quant summary, model_config).
  void FinalizeHeader(const std::string& path, const nlohmann::json& metadata) {
    if (finalized_) throw std::runtime_error("ContainerWriter: FinalizeHeader() called twice");
    nlohmann::json header = nlohmann::json::object();
    header["__metadata__"] = metadata;
    for (size_t i = 0; i < plan_.size(); ++i) {
      const Entry& e = plan_[i];
      header[plan_names_[i]] = {
          {"dtype", "U8"},
          {"shape", e.shape},
          {"data_offsets", {e.offset, e.offset + e.nbytes}},
      };
    }
    const std::string header_str = header.dump();
    const uint64_t header_len = header_str.size();
    data_start_ = 8 + header_len;

    // _wfopen, not fopen: safetensors_reader.hpp deliberately opens shards via CreateFileW/
    // Utf8ToWide because ANSI-codepage fopen() can silently resolve a non-ASCII path differently
    // (or fail); the writer previously used narrow fopen() here, an inconsistency flagged by
    // review (minor). --output is typically ASCII in practice, but this keeps every path-open in
    // the component on the same UTF-8-safe code path.
    if (_wfopen_s(&file_, Utf8ToWide(path).c_str(), L"wb") != 0 || !file_)
      throw std::runtime_error("ContainerWriter: cannot create output file " + path);
    std::fwrite(&header_len, 1, 8, file_);
    std::fwrite(header_str.data(), 1, header_str.size(), file_);
    // Pre-size the file so every WriteTensor() below is a pure seek+write, no growth races.
    const uint64_t total = data_start_ + cursor_;
    if (total > 8 + header_len) {
      if (_fseeki64(file_, static_cast<int64_t>(total - 1), SEEK_SET) != 0)
        throw std::runtime_error("ContainerWriter: pre-size seek failed");
      const char zero = 0;
      std::fwrite(&zero, 1, 1, file_);
    }
    std::fflush(file_);
    finalized_ = true;
  }

  // Writes `nbytes` raw bytes for a previously-planned tensor. Thread-safe (guarded by a mutex
  // around the seek+write pair) so tensor-level work can be parallelized later without touching
  // this class.
  void WriteTensor(const std::string& name, const void* data, uint64_t nbytes) {
    if (!finalized_) throw std::runtime_error("ContainerWriter: WriteTensor() before FinalizeHeader()");
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("ContainerWriter: unplanned tensor " + name);
    Entry& e = plan_[static_cast<size_t>(it->second)];
    if (nbytes != e.nbytes) {
      throw std::runtime_error("ContainerWriter: size mismatch for " + name + " (planned " +
                                std::to_string(e.nbytes) + ", got " + std::to_string(nbytes) + ")");
    }
    {
      std::lock_guard<std::mutex> lock(io_mutex_);
      if (_fseeki64(file_, static_cast<int64_t>(data_start_ + e.offset), SEEK_SET) != 0)
        throw std::runtime_error("ContainerWriter: seek failed for " + name);
      if (nbytes > 0 && std::fwrite(data, 1, nbytes, file_) != nbytes)
        throw std::runtime_error("ContainerWriter: write failed for " + name);
    }
    e.written = true;
  }

  // Review finding (minor): FinalizeHeader pre-sizes the whole file with zeros, so a planned
  // tensor whose emit_job is missing (e.g. a plan_jobs/emit_jobs lambda-list typo in main.cpp)
  // silently ships as a valid-looking all-zero tensor with a valid header and a valid
  // config_sha256 -- nothing anywhere would say so. Call this once after every WriteTensor() call
  // is expected to have happened (RunConvert/RunSelftest, right before printing success) to turn
  // that into a loud, named error instead.
  void Finish() const {
    std::string missing;
    for (size_t i = 0; i < plan_.size(); ++i) {
      if (!plan_[i].written) {
        if (!missing.empty()) missing += ", ";
        missing += plan_names_[i];
      }
    }
    if (!missing.empty())
      throw std::runtime_error("ContainerWriter: planned tensor(s) never written: " + missing);
  }

  uint64_t PlannedDataBytes() const { return cursor_; }
  size_t PlannedTensorCount() const { return plan_.size(); }

  ~ContainerWriter() {
    if (file_) std::fclose(file_);
  }

 private:
  struct Entry {
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
    bool written = false;
  };
  std::vector<Entry> plan_;
  std::vector<std::string> plan_names_;
  std::unordered_map<std::string, int64_t> index_;
  uint64_t cursor_ = 0;
  uint64_t data_start_ = 0;
  bool finalized_ = false;
  std::FILE* file_ = nullptr;
  std::mutex io_mutex_;
};

}  // namespace r4dx_convert
