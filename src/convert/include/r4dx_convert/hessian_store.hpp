// r4dx_convert::hessian_store -- reads the per-tap Hessians tools/reference/hessian_capture.py
// writes (docs/quant2.md section 2.1) and hands out LDLQ factors (quant_ldlq.hpp) for them.
//
// ---- On disk ---------------------------------------------------------------------------------
// <dir>/<file>.hess, little-endian (x86 is the only host this converter builds for):
//   [8]  magic "R4DXHES1"
//   [4]  u32 K
//   [4]  u32 flags      bit 0 = packed upper triangle. v1 always sets it and nothing else; any
//                       other value is rejected rather than guessed at.
//   [8]  u64 rows       tokens accumulated
//   [8]  f64 trace      sum of the stored fp32 diagonal, for a cheap integrity check
//   [32] reserved       zeros (checked)
//   then K*(K+1)/2 float32: H[i][j] for i = 0..K-1, j = i..K-1, H = X^T X / rows.
// <dir>/hessian.json:
//   { "format": "r4dx-hessian", "version": 1,
//     "files": { "<file>": { "K": int, "rows": int, "trace": float } },
//     "keys":  { "<container base name, e.g. text.layers.3.mlp.down>": "<file>" }, ...provenance }
// Several container bases share one file when they share an input (gdn.in_proj_qkv / in_proj_z;
// attn.qg / k / v): the Hessian is a property of the linear's INPUT, not of its weight.
//
// ---- Validation ------------------------------------------------------------------------------
// ReadHessFile checks magic, flags, reserved bytes, the exact file size for K, that every stored
// value is finite, that the diagonal is non-negative (X^T X cannot have a negative diagonal; one
// means a corrupt file, not an unusual activation), and that the recomputed trace matches the header
// to a relative 1e-3 -- a truncated-then-zero-padded or half-rewritten file fails one of these
// before any factorization time is spent on it. HessianStore additionally checks every key's file
// against the manifest's K (and rows, when present). CheckFile runs the header + size half of this
// (no payload read) at planning time, so a missing or truncated file fails before the emit pass.
//
// ---- Memory ---------------------------------------------------------------------------------
// The largest tap (mlp.down, K = 17408) is 606 MB packed and 1.21 GB expanded. ReadHessFile reads
// the packed triangle straight into the TAIL of the full K*K buffer and expands it in place, row by
// row from the top (row i's full span [i*K, (i+1)*K) never reaches the packed data of rows > i:
// that starts at K*K - K(K+1)/2 + (i+1)K - i(i+1)/2 >= (i+1)K), so the peak is one K*K buffer, not
// packed + full. FactorHessian then holds H and one working copy (it retries from the original H),
// and the store drops its previous factor BEFORE building the next, so at most ~3 K*K buffers are
// live at once.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>  // sha256.hpp uses std::snprintf without including it
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/safetensors_reader.hpp"  // Utf8ToWide (wide open, as every other reader here)
#include "r4dx_convert/sha256.hpp"

namespace r4dx_convert {

namespace hessian_detail {

inline constexpr char kMagic[8] = {'R', '4', 'D', 'X', 'H', 'E', 'S', '1'};
inline constexpr size_t kHeaderBytes = 64;
inline constexpr uint32_t kFlagPackedUpper = 1u;

inline std::string JoinPath(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  const char last = dir.back();
  if (last == '/' || last == '\\') return dir + name;
  return dir + "\\" + name;
}

template <typename T>
T ReadLe(const unsigned char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));  // little-endian host
  return v;
}

inline std::string ReadWholeFile(const std::string& path, const char* who) {
  std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error(std::string(who) + ": cannot open " + path);
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (f.bad()) throw std::runtime_error(std::string(who) + ": read failed for " + path);
  return bytes;
}

struct HessHeader {
  int64_t K = 0;
  uint64_t rows = 0;
  double trace = 0.0;
};

// Opens `path` and validates everything checkable without reading the payload: magic, flags,
// reserved bytes, K > 0, rows > 0, and the exact file size for K. Leaves `f` positioned at the
// packed triangle. Shared by ReadHessFile and HessianStore::CheckFile (the planning-time check).
inline HessHeader OpenHessFile(const std::string& path, const std::string& who, std::ifstream& f) {
  f.open(Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error(who + ": cannot open");

  f.seekg(0, std::ios::end);
  const std::streamoff file_size = f.tellg();
  f.seekg(0, std::ios::beg);
  if (file_size < static_cast<std::streamoff>(kHeaderBytes))
    throw std::runtime_error(who + ": file is shorter than the 64-byte header");

  unsigned char hdr[kHeaderBytes];
  f.read(reinterpret_cast<char*>(hdr), kHeaderBytes);
  if (!f) throw std::runtime_error(who + ": header read failed");
  if (std::memcmp(hdr, kMagic, 8) != 0) throw std::runtime_error(who + ": bad magic (not R4DXHES1)");
  const uint32_t K32 = ReadLe<uint32_t>(hdr + 8);
  const uint32_t flags = ReadLe<uint32_t>(hdr + 12);
  HessHeader h;
  h.rows = ReadLe<uint64_t>(hdr + 16);
  h.trace = ReadLe<double>(hdr + 24);
  if (flags != kFlagPackedUpper)
    throw std::runtime_error(who + ": unsupported flags " + std::to_string(flags) +
                             " (v1 requires exactly 1 = packed upper triangle)");
  for (size_t i = 32; i < kHeaderBytes; ++i)
    if (hdr[i] != 0) throw std::runtime_error(who + ": reserved header bytes are not zero");
  if (K32 == 0) throw std::runtime_error(who + ": K = 0");
  if (h.rows == 0) throw std::runtime_error(who + ": rows = 0 (no tokens accumulated)");

  h.K = K32;
  const uint64_t packed_n = static_cast<uint64_t>(h.K) * static_cast<uint64_t>(h.K + 1) / 2;
  const uint64_t expect = kHeaderBytes + packed_n * sizeof(float);
  if (static_cast<uint64_t>(file_size) != expect)
    throw std::runtime_error(who + ": size " + std::to_string(file_size) + " bytes, expected " +
                             std::to_string(expect) + " for K=" + std::to_string(h.K));
  return h;
}

}  // namespace hessian_detail

// Full symmetric K x K (row-major) from one .hess file; *K and *rows receive the header values.
inline std::vector<float> ReadHessFile(const std::string& path, int64_t* K_out, uint64_t* rows_out) {
  using namespace hessian_detail;
  const std::string who = "ReadHessFile(" + path + ")";
  std::ifstream f;
  const HessHeader h = OpenHessFile(path, who, f);
  const int64_t K = h.K;
  const uint64_t rows = h.rows;
  const double trace_hdr = h.trace;
  const uint64_t packed_n = static_cast<uint64_t>(K) * static_cast<uint64_t>(K + 1) / 2;

  const uint64_t full_n = static_cast<uint64_t>(K) * static_cast<uint64_t>(K);
  std::vector<float> H(static_cast<size_t>(full_n));
  const uint64_t base = full_n - packed_n;  // packed triangle lives in the tail, see header comment
  {
    char* dst = reinterpret_cast<char*>(H.data() + base);
    uint64_t left = packed_n * sizeof(float);
    constexpr uint64_t kChunk = uint64_t{1} << 28;
    while (left > 0) {
      const uint64_t take = left < kChunk ? left : kChunk;
      f.read(dst, static_cast<std::streamsize>(take));
      if (!f) throw std::runtime_error(who + ": short read of the packed triangle");
      dst += take;
      left -= take;
    }
  }

  // Validate on the packed data (each value once), then expand in place.
  double trace = 0.0;
  {
    const float* p = H.data() + base;
    for (int64_t i = 0; i < K; ++i) {
      const float d = p[0];
      if (!(d >= 0.0f) || !std::isfinite(d))
        throw std::runtime_error(who + ": diagonal H[" + std::to_string(i) + "][" +
                                 std::to_string(i) + "] = " + std::to_string(d) +
                                 " (negative or non-finite)");
      trace += static_cast<double>(d);
      for (int64_t j = 1; j < K - i; ++j)
        if (!std::isfinite(p[j]))
          throw std::runtime_error(who + ": non-finite H[" + std::to_string(i) + "][" +
                                   std::to_string(i + j) + "]");
      p += K - i;
    }
  }
  if (!(std::fabs(trace - trace_hdr) <= 1e-3 * std::fabs(trace_hdr)) &&
      !(trace == 0.0 && trace_hdr == 0.0)) {
    throw std::runtime_error(who + ": trace mismatch (header " + std::to_string(trace_hdr) +
                             ", data " + std::to_string(trace) + ")");
  }

  uint64_t off = base;  // start of packed row i
  for (int64_t i = 0; i < K; ++i) {
    float* row = H.data() + static_cast<uint64_t>(i) * K;
    const uint64_t len = static_cast<uint64_t>(K - i);
    // dest [i*K + i, (i+1)*K) starts at or before the source; memmove is overlap-safe.
    std::memmove(row + i, H.data() + off, len * sizeof(float));
    for (int64_t j = 0; j < i; ++j) row[j] = H[static_cast<uint64_t>(j) * K + i];
    off += len;
  }

  if (K_out) *K_out = K;
  if (rows_out) *rows_out = rows;
  return H;
}

// An optional change of basis applied to H after ReadHessFile has validated it and before it is
// factored (docs/quant2.md sections 3-4): a linear folded by r4dx-convert --rotate sees its input as
// x M, so LDLQ must round against H' = M^T H M (rotation.hpp's TransformHessianQ /
// TransformHessianHadamard), not the captured H. `id` names the transform completely -- kind plus a
// hash of whatever it depends on (the folded norm weights) -- because it is part of the factor
// cache key: two linears sharing a tap file but not a transform must never share a factor, and two
// sharing both (attn.qg / k / v under one input_layernorm) still should.
struct HessianTransform {
  std::string id;
  std::function<void(std::vector<float>& H, int64_t K, int nthreads)> apply;
};

class HessianStore {
 public:
  explicit HessianStore(const std::string& dir) : dir_(dir) {
    const std::string manifest_path = hessian_detail::JoinPath(dir_, "hessian.json");
    const std::string bytes = hessian_detail::ReadWholeFile(manifest_path, "HessianStore");
    sha256_ = Sha256Hex(bytes);

    nlohmann::json j;
    try {
      j = nlohmann::json::parse(bytes);
    } catch (const std::exception& e) {
      throw std::runtime_error("HessianStore: " + manifest_path + " is not valid JSON: " + e.what());
    }
    const std::string who = "HessianStore(" + manifest_path + ")";
    if (!j.is_object() || !j.contains("format") || !j["format"].is_string() ||
        j["format"].get<std::string>() != "r4dx-hessian")
      throw std::runtime_error(who + ": \"format\" is not \"r4dx-hessian\"");
    if (!j.contains("version") || !j["version"].is_number_integer() ||
        j["version"].get<int64_t>() != 1)
      throw std::runtime_error(who + ": unsupported \"version\" (this converter reads version 1)");
    if (!j.contains("files") || !j["files"].is_object())
      throw std::runtime_error(who + ": missing \"files\" object");
    if (!j.contains("keys") || !j["keys"].is_object())
      throw std::runtime_error(who + ": missing \"keys\" object");

    for (auto it = j["files"].begin(); it != j["files"].end(); ++it) {
      const nlohmann::json& e = it.value();
      if (!e.is_object() || !e.contains("K") || !e["K"].is_number_integer() ||
          e["K"].get<int64_t>() <= 0)
        throw std::runtime_error(who + ": files[\"" + it.key() + "\"] has no positive integer K");
      FileInfo fi;
      fi.K = e["K"].get<int64_t>();
      if (e.contains("rows")) {
        if (!e["rows"].is_number_unsigned() && !e["rows"].is_number_integer())
          throw std::runtime_error(who + ": files[\"" + it.key() + "\"].rows is not an integer");
        fi.rows = e["rows"].get<uint64_t>();
        fi.has_rows = true;
      }
      files_[it.key()] = fi;
    }
    for (auto it = j["keys"].begin(); it != j["keys"].end(); ++it) {
      if (!it.value().is_string())
        throw std::runtime_error(who + ": keys[\"" + it.key() + "\"] is not a file name");
      const std::string file = it.value().get<std::string>();
      if (!files_.count(file))
        throw std::runtime_error(who + ": keys[\"" + it.key() + "\"] names \"" + file +
                                 "\", which is not listed under \"files\"");
      keys_[it.key()] = file;
    }
  }

  bool Has(const std::string& key) const { return keys_.count(key) != 0; }

  int64_t KOf(const std::string& key) const { return files_.at(File(key)).K; }

  std::string File(const std::string& key) const {
    auto it = keys_.find(key);
    if (it == keys_.end())
      throw std::runtime_error("HessianStore: no Hessian for '" + key + "' in " + dir_);
    return it->second;
  }

  const std::string& ManifestSha256() const { return sha256_; }
  const std::string& Dir() const { return dir_; }

  // Planning-time check of `key`'s file on disk: header valid, exact size for its K, K and rows
  // equal to the manifest's. Reads 64 bytes, never the payload, and each distinct file once -- so a
  // missing or truncated file (an interrupted capture, a partial copy) fails before the converter
  // writes a header, not hours into the emit pass. The payload checks (finiteness, trace) still run
  // in ReadHessFile at Factor() time.
  void CheckFile(const std::string& key) {
    const std::string file = File(key);
    if (checked_files_.count(file)) return;
    const FileInfo& fi = files_.at(file);
    const std::string path = hessian_detail::JoinPath(dir_, file);
    std::ifstream f;
    const hessian_detail::HessHeader h =
        hessian_detail::OpenHessFile(path, "HessianStore(" + path + ")", f);
    if (h.K != fi.K)
      throw std::runtime_error("HessianStore: " + path + " header says K=" + std::to_string(h.K) +
                               " but hessian.json says K=" + std::to_string(fi.K));
    if (fi.has_rows && h.rows != fi.rows)
      throw std::runtime_error("HessianStore: " + path + " header says rows=" +
                               std::to_string(h.rows) + " but hessian.json says rows=" +
                               std::to_string(fi.rows));
    checked_files_.insert(file);
  }

  // The factor for `key`'s tap. The returned reference stays valid until the next Factor() call
  // that needs a DIFFERENT (file, damp, transform id): only the last factor is cached, which is all
  // the emit loop needs -- the bases sharing a tap (in_proj_qkv / in_proj_z, qg / k / v) are
  // emitted back to back. `xf` (nullable) is applied to H before factoring; see HessianTransform.
  const LdlqFactor& Factor(const std::string& key, int64_t K, float damp, int nthreads,
                           const HessianTransform* xf = nullptr) {
    const std::string file = File(key);
    const FileInfo& fi = files_.at(file);
    if (fi.K != K)
      throw std::runtime_error("HessianStore: '" + key + "' has K=" + std::to_string(K) +
                               " but its Hessian " + file + " is K=" + std::to_string(fi.K));
    const std::string xf_id = xf ? xf->id : std::string();
    if (xf && xf_id.empty())
      throw std::runtime_error("HessianStore: a HessianTransform needs a non-empty id (cache key)");
    if (cached_valid_ && cached_file_ == file && cached_xf_id_ == xf_id &&
        std::memcmp(&cached_damp_, &damp, sizeof(float)) == 0)
      return cached_;

    // Free the previous factor first: at K = 17408 it is 1.2 GB.
    cached_valid_ = false;
    cached_ = LdlqFactor{};

    const std::string path = hessian_detail::JoinPath(dir_, file);
    int64_t fk = 0;
    uint64_t frows = 0;
    std::vector<float> H = ReadHessFile(path, &fk, &frows);
    if (fk != fi.K)
      throw std::runtime_error("HessianStore: " + path + " header says K=" + std::to_string(fk) +
                               " but hessian.json says K=" + std::to_string(fi.K));
    if (fi.has_rows && frows != fi.rows)
      throw std::runtime_error("HessianStore: " + path + " header says rows=" +
                               std::to_string(frows) + " but hessian.json says rows=" +
                               std::to_string(fi.rows));
    if (xf) xf->apply(H, K, nthreads);
    cached_ = FactorHessian(std::move(H), K, damp, nthreads);
    cached_file_ = file;
    cached_xf_id_ = xf_id;
    cached_damp_ = damp;
    cached_valid_ = true;
    return cached_;
  }

 private:
  struct FileInfo {
    int64_t K = 0;
    uint64_t rows = 0;
    bool has_rows = false;
  };

  std::string dir_;
  std::string sha256_;
  std::map<std::string, FileInfo> files_;
  std::map<std::string, std::string> keys_;
  std::set<std::string> checked_files_;  // CheckFile's memo

  LdlqFactor cached_;
  std::string cached_file_;
  std::string cached_xf_id_;  // "" = untransformed
  float cached_damp_ = 0.0f;
  bool cached_valid_ = false;
};

}  // namespace r4dx_convert
