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

#include <io.h>     // _commit, _fileno (Sync)
#include <share.h>  // _SH_DENYWR (FinalizeHeader)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"  // Utf8ToWide -- see FinalizeHeader's _wfopen note
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/threadpool.hpp"          // ParallelEach (ReadBackDigests)

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
    header_str_ = header.dump();
    const std::string& header_str = header_str_;
    const uint64_t header_len = header_str.size();
    data_start_ = 8 + header_len;

    // _wfsopen, not fopen: safetensors_reader.hpp deliberately opens shards via CreateFileW/
    // Utf8ToWide because ANSI-codepage fopen() can silently resolve a non-ASCII path differently
    // (or fail); the writer previously used narrow fopen() here, an inconsistency flagged by
    // review (minor). --output is typically ASCII in practice, but this keeps every path-open in
    // the component on the same UTF-8-safe code path. _SH_DENYWR (not _wfopen_s, whose files are
    // not sharable at all): no other writer while this one holds the file, but ReadBackDigests'
    // own read handles may open it.
    file_ = _wfsopen(Utf8ToWide(path).c_str(), L"wb", _SH_DENYWR);
    if (!file_) throw std::runtime_error("ContainerWriter: cannot create output file " + path);
    path_ = path;
    if (std::fwrite(&header_len, 1, 8, file_) != 8 ||
        std::fwrite(header_str.data(), 1, header_str.size(), file_) != header_str.size())
      throw std::runtime_error("ContainerWriter: header write failed for " + path);
    // Pre-size the file so every WriteTensor() below is a pure seek+write, no growth races.
    const uint64_t total = data_start_ + cursor_;
    if (total > 8 + header_len) {
      if (_fseeki64(file_, static_cast<int64_t>(total - 1), SEEK_SET) != 0)
        throw std::runtime_error("ContainerWriter: pre-size seek failed");
      const char zero = 0;
      if (std::fwrite(&zero, 1, 1, file_) != 1)
        throw std::runtime_error("ContainerWriter: pre-size write failed for " + path);
    }
    std::fflush(file_);
    finalized_ = true;
  }

  // Writes `nbytes` raw bytes for a previously-planned tensor. Thread-safe (guarded by a mutex
  // around the seek+write pair) so tensor-level work can be parallelized later without touching
  // this class.
  void WriteTensor(const std::string& name, const void* data, uint64_t nbytes) {
    if (!finalized_) throw std::runtime_error("ContainerWriter: WriteTensor() before FinalizeHeader()");
    if (!file_) throw std::runtime_error("ContainerWriter: WriteTensor() after Close()");
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

  // WriteTensor for a source too large to want in one call -- r4dx-convert --reuse-tensors-from
  // copies tensors of up to a few GB straight out of the baseline's mapping: same checks, one seek,
  // then `chunk`-byte writes, so the pages of `data` are touched front to back in bounded steps.
  void WriteTensorChunked(const std::string& name, const uint8_t* data, uint64_t nbytes,
                          uint64_t chunk = uint64_t{64} << 20) {
    if (!finalized_) throw std::runtime_error("ContainerWriter: WriteTensorChunked() before FinalizeHeader()");
    if (!file_) throw std::runtime_error("ContainerWriter: WriteTensorChunked() after Close()");
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
      for (uint64_t done = 0; done < nbytes;) {
        const uint64_t n = std::min<uint64_t>(chunk, nbytes - done);
        if (std::fwrite(data + done, 1, static_cast<size_t>(n), file_) != n)
          throw std::runtime_error("ContainerWriter: write failed for " + name);
        done += n;
      }
    }
    e.written = true;
  }

  // Planned tensor `i` (plan order = data order): name, shape (as in the header), byte size.
  const std::string& PlannedName(size_t i) const { return plan_names_.at(i); }
  const std::vector<int64_t>& PlannedShape(size_t i) const { return plan_.at(i).shape; }
  uint64_t PlannedBytes(size_t i) const { return plan_.at(i).nbytes; }

  // How often `needle` occurs in the header FinalizeHeader wrote.
  size_t HeaderOccurrences(const std::string& needle) const {
    size_t n = 0;
    for (size_t p = header_str_.find(needle); p != std::string::npos;
         p = header_str_.find(needle, p + 1))
      ++n;
    return n;
  }

  // Everything written so far is on disk: the CRT buffer is flushed to the OS, and the OS cache to
  // the device (_commit = FlushFileBuffers). Without the second half the OS may write pages back
  // in any order, so after a crash or power loss a header patched "complete" could sit over tensor
  // pages that never reached the disk (the pre-sized file's zeros).
  void Sync() {
    if (!file_) throw std::runtime_error("ContainerWriter: Sync() without an open file");
    std::lock_guard<std::mutex> lock(io_mutex_);
    SyncLocked();
  }

  // Rewrites the one occurrence of `from` in the on-disk header as `to` (same length, so no offset
  // moves). Durable on both sides: every byte written before the call is on disk before the patch
  // is written (so a completion marker can never reach the disk ahead of the data it vouches for),
  // and the patch itself is on disk when the call returns. r4dx-convert's reuse guard uses it to
  // record its data digest and flip its completion marker after Finish(); check
  // HeaderOccurrences(from) == 1 right after FinalizeHeader so a header that cannot be patched fails
  // before the emit pass, not after it.
  void PatchHeader(const std::string& from, const std::string& to) {
    if (!finalized_) throw std::runtime_error("ContainerWriter: PatchHeader() before FinalizeHeader()");
    if (!file_) throw std::runtime_error("ContainerWriter: PatchHeader() after Close()");
    if (from.size() != to.size())
      throw std::logic_error("ContainerWriter::PatchHeader: '" + from + "' and '" + to +
                             "' differ in length");
    if (HeaderOccurrences(from) != 1)
      throw std::logic_error("ContainerWriter::PatchHeader: '" + from +
                             "' does not occur exactly once in the header");
    const size_t pos = header_str_.find(from);
    std::lock_guard<std::mutex> lock(io_mutex_);
    SyncLocked();
    if (_fseeki64(file_, static_cast<int64_t>(8 + pos), SEEK_SET) != 0 ||
        std::fwrite(to.data(), 1, to.size(), file_) != to.size())
      throw std::runtime_error("ContainerWriter: header patch failed");
    SyncLocked();
    header_str_.replace(pos, to.size(), to);
  }

  // sha256 (hex) of every planned tensor's bytes AS THEY ARE IN THE FILE, in plan order: Sync()s,
  // then reads each tensor back through its own read handle, `threads` tensors at a time, the
  // largest first. What r4dx-convert's reuse guard digests -- the file, not the buffers handed to
  // WriteTensor. Call after Finish().
  std::vector<std::string> ReadBackDigests(int threads) {
    if (!finalized_) throw std::runtime_error("ContainerWriter: ReadBackDigests() before FinalizeHeader()");
    Sync();
    std::vector<size_t> order(plan_.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return plan_[a].nbytes > plan_[b].nbytes; });
    std::vector<std::string> out(plan_.size());
    const std::wstring wpath = Utf8ToWide(path_);
    ParallelEach(order.size(), threads, [&](size_t k) {
      const size_t i = order[k];
      const Entry& e = plan_[i];
      std::ifstream f(wpath.c_str(), std::ios::binary);
      if (!f) throw std::runtime_error("ContainerWriter: cannot reopen " + path_ + " to read it back");
      f.seekg(static_cast<std::streamoff>(data_start_ + e.offset));
      std::vector<char> buf(static_cast<size_t>(std::min<uint64_t>(e.nbytes, uint64_t{8} << 20)));
      Sha256 h;
      for (uint64_t done = 0; done < e.nbytes;) {
        const uint64_t n = std::min<uint64_t>(buf.size(), e.nbytes - done);
        f.read(buf.data(), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("ContainerWriter: short read-back of " + plan_names_[i]);
        h.Update(reinterpret_cast<const uint8_t*>(buf.data()), static_cast<size_t>(n));
        done += n;
      }
      out[i] = h.HexDigest();
    });
    return out;
  }

  // Closes the file, reporting a failed close (the destructor cannot). Idempotent.
  void Close() {
    if (!file_) return;
    std::FILE* f = file_;
    file_ = nullptr;
    if (std::fclose(f) != 0) throw std::runtime_error("ContainerWriter: closing " + path_ + " failed");
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

  void SyncLocked() {
    if (std::fflush(file_) != 0 || _commit(_fileno(file_)) != 0)
      throw std::runtime_error("ContainerWriter: flushing " + path_ + " to disk failed");
  }

  std::vector<Entry> plan_;
  std::vector<std::string> plan_names_;
  std::string path_;
  std::string header_str_;  // the JSON header as written (PatchHeader keeps it in sync)
  std::unordered_map<std::string, int64_t> index_;
  uint64_t cursor_ = 0;
  uint64_t data_start_ = 0;
  bool finalized_ = false;
  std::FILE* file_ = nullptr;
  std::mutex io_mutex_;
};

}  // namespace r4dx_convert
