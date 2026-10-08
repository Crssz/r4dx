// HIP-free host side of the hybrid mode's one-device gates (docs/pp-tp2-hybrid.md 4, 9 "P2"): a Model's LIVE state as host bytes
// (hybrid::LiveImage, reshard_plan.h) + its scalars (StageSyncState), the canonical form of that image, and a checked blob file so
// a TP=1 reference run in one process can dump its state and the emulated hybrid in a LATER process can load it (Gate B: the two
// would not fit in 32 GiB together). Model::DebugExportFullState / DebugImportFullState are the device ends of this.
//
// Canonical form: a KV image holds whole blocks, ceil(live_rows / block) of them, and the rows of the last block past live_rows are
// ZERO (a device cache keeps stale bytes there; they are not state and must not make two images of the same state differ).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "live_digest.h"
#include "reshard_plan.h"
#include "stage_sync.h"

namespace r4dx::model::hybrid {

// Zeroes the rows >= live_rows of every block of a KV image ([block][head][token][row_bytes]) and drops blocks past the last live one.
inline void CanonicalizeKv(std::vector<uint8_t>* image, int64_t kv_heads, int64_t block_tokens, int64_t row_bytes, int64_t live_rows) {
  const size_t head_block = static_cast<size_t>(block_tokens * row_bytes), block = head_block * static_cast<size_t>(kv_heads);
  const int64_t need = live_rows > 0 ? (live_rows + block_tokens - 1) / block_tokens : 0;
  if (image->size() % block != 0 || static_cast<int64_t>(image->size() / block) < need) {
    throw std::invalid_argument("CanonicalizeKv: the image holds fewer whole blocks than " + std::to_string(live_rows) + " rows need");
  }
  image->resize(static_cast<size_t>(need) * block);
  const int64_t rem = live_rows % block_tokens;
  if (need == 0 || rem == 0) return;
  uint8_t* last = image->data() + static_cast<size_t>(need - 1) * block;
  for (int64_t h = 0; h < kv_heads; ++h) {
    std::memset(last + static_cast<size_t>(h) * head_block + static_cast<size_t>(rem * row_bytes), 0, static_cast<size_t>((block_tokens - rem) * row_bytes));
  }
}

struct LiveState {
  LiveImage image;
  StageSyncState scalars;
  std::map<std::string, std::vector<uint8_t>> extra;  // test payloads that travel with the state (DFlash features, rope rows, ...)
};

// ---- the blob container -----------------------------------------------------------------------------------------------------
// "R4DXLST1" | u32 count | count x (u32 name_len, name, u64 size, bytes) | u64 FNV-1a of everything before it. Little endian (the host).
struct Blob {
  std::string name;
  std::vector<uint8_t> bytes;
};

namespace detail {
inline void Put(std::vector<uint8_t>* out, const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  out->insert(out->end(), b, b + n);
}
inline void PutU32(std::vector<uint8_t>* out, uint32_t v) { Put(out, &v, 4); }
inline void PutU64(std::vector<uint8_t>* out, uint64_t v) { Put(out, &v, 8); }
struct Reader {
  const uint8_t* p;
  size_t n, at = 0;
  void Need(size_t k, const char* what) const {
    if (k > n - at) throw std::runtime_error(std::string("state blob: truncated at ") + what);
  }
  uint32_t U32(const char* what) {
    Need(4, what);
    uint32_t v;
    std::memcpy(&v, p + at, 4);
    at += 4;
    return v;
  }
  uint64_t U64(const char* what) {
    Need(8, what);
    uint64_t v;
    std::memcpy(&v, p + at, 8);
    at += 8;
    return v;
  }
};
constexpr char kBlobMagic[8] = {'R', '4', 'D', 'X', 'L', 'S', 'T', '1'};
}  // namespace detail

inline std::vector<uint8_t> EncodeBlobs(const std::vector<Blob>& blobs) {
  std::vector<uint8_t> out;
  detail::Put(&out, detail::kBlobMagic, sizeof(detail::kBlobMagic));
  detail::PutU32(&out, static_cast<uint32_t>(blobs.size()));
  for (const Blob& b : blobs) {
    detail::PutU32(&out, static_cast<uint32_t>(b.name.size()));
    detail::Put(&out, b.name.data(), b.name.size());
    detail::PutU64(&out, static_cast<uint64_t>(b.bytes.size()));
    detail::Put(&out, b.bytes.data(), b.bytes.size());
  }
  detail::PutU64(&out, FnvUpdate(kFnvInit, out.data(), out.size()));
  return out;
}

inline std::vector<Blob> DecodeBlobs(const uint8_t* data, size_t n) {
  if (n < sizeof(detail::kBlobMagic) + 4 + 8 || std::memcmp(data, detail::kBlobMagic, sizeof(detail::kBlobMagic)) != 0) {
    throw std::runtime_error("state blob: not a live-state file (bad magic)");
  }
  uint64_t want;
  std::memcpy(&want, data + n - 8, 8);
  if (FnvUpdate(kFnvInit, data, n - 8) != want) throw std::runtime_error("state blob: checksum mismatch (the file is damaged or truncated)");
  detail::Reader r{data, n - 8};
  r.at = sizeof(detail::kBlobMagic);
  const uint32_t count = r.U32("count");
  std::vector<Blob> out;
  for (uint32_t i = 0; i < count; ++i) {
    Blob b;
    const uint32_t name_len = r.U32("name length");
    r.Need(name_len, "name");
    b.name.assign(reinterpret_cast<const char*>(data) + r.at, name_len);
    r.at += name_len;
    const uint64_t size = r.U64("payload size");
    r.Need(static_cast<size_t>(size), "payload");
    b.bytes.assign(data + r.at, data + r.at + static_cast<size_t>(size));
    r.at += static_cast<size_t>(size);
    for (const Blob& o : out) {
      if (o.name == b.name) throw std::runtime_error("state blob: duplicate record '" + b.name + "'");
    }
    out.push_back(std::move(b));
  }
  if (r.at != n - 8) throw std::runtime_error("state blob: trailing bytes after the last record");
  return out;
}

inline void WriteBlobFile(const std::string& path, const std::vector<Blob>& blobs) {
  const std::vector<uint8_t> bytes = EncodeBlobs(blobs);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("state blob: cannot open '" + path + "' for writing");
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  f.flush();
  if (!f) throw std::runtime_error("state blob: writing '" + path + "' failed");
}

inline std::vector<Blob> ReadBlobFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("state blob: cannot open '" + path + "'");
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(bytes.data()), n);
  if (!f) throw std::runtime_error("state blob: reading '" + path + "' failed");
  return DecodeBlobs(bytes.data(), bytes.size());
}

// ---- LiveState <-> blobs ----------------------------------------------------------------------------------------------------
inline std::vector<uint8_t> EncodeScalars(const StageSyncState& s) {
  std::vector<uint8_t> out;
  detail::PutU64(&out, static_cast<uint64_t>(s.pos));
  const uint8_t flags = static_cast<uint8_t>((s.started ? 1 : 0) | (s.mrope_active ? 2 : 0) | (s.mtp_seed_valid ? 4 : 0));
  detail::Put(&out, &flags, 1);
  detail::PutU64(&out, static_cast<uint64_t>(s.mrope_delta));
  detail::PutU32(&out, static_cast<uint32_t>(s.mtp_seed.size()));
  detail::Put(&out, s.mtp_seed.data(), s.mtp_seed.size() * sizeof(uint16_t));
  return out;
}
inline StageSyncState DecodeScalars(const std::vector<uint8_t>& b) {
  detail::Reader r{b.data(), b.size()};
  StageSyncState s;
  s.pos = static_cast<int64_t>(r.U64("pos"));
  r.Need(1, "flags");
  const uint8_t flags = b[r.at++];
  s.started = (flags & 1) != 0;
  s.mrope_active = (flags & 2) != 0;
  s.mtp_seed_valid = (flags & 4) != 0;
  s.mrope_delta = static_cast<int64_t>(r.U64("mrope delta"));
  const uint32_t n = r.U32("seed length");
  r.Need(static_cast<size_t>(n) * 2, "seed");
  s.mtp_seed.resize(n);
  std::memcpy(s.mtp_seed.data(), b.data() + r.at, static_cast<size_t>(n) * 2);
  r.at += static_cast<size_t>(n) * 2;
  if (r.at != b.size()) throw std::runtime_error("state blob: trailing bytes in the scalars record");
  return s;
}

inline std::vector<Blob> LiveStateToBlobs(const LiveState& st) {
  std::vector<Blob> out;
  out.push_back({"scalars", EncodeScalars(st.scalars)});
  for (const auto& [l, b] : st.image.kv) out.push_back({DigestName("kv", l), b});
  if (!st.image.mtp_kv.empty()) out.push_back({"mtp.kv", st.image.mtp_kv});
  for (const auto& [l, b] : st.image.gdn_rec) out.push_back({DigestName("gdn.rec", l), b});
  for (const auto& [l, c] : st.image.gdn_conv) {
    Blob b{DigestName("gdn.conv", l), {}};
    b.bytes.resize(c.size() * sizeof(uint16_t));
    std::memcpy(b.bytes.data(), c.data(), b.bytes.size());
    out.push_back(std::move(b));
  }
  for (const auto& [name, bytes] : st.extra) out.push_back({"x." + name, bytes});
  return out;
}

inline LiveState LiveStateFromBlobs(const std::vector<Blob>& blobs) {
  LiveState st;
  bool have_scalars = false;
  const auto layer_of = [](const std::string& name, const std::string& prefix) { return static_cast<int64_t>(std::stoll(name.substr(prefix.size()))); };
  for (const Blob& b : blobs) {
    const auto starts = [&](const char* p) { return b.name.rfind(p, 0) == 0; };
    if (b.name == "scalars") {
      st.scalars = DecodeScalars(b.bytes);
      have_scalars = true;
    } else if (b.name == "mtp.kv") {
      st.image.mtp_kv = b.bytes;
    } else if (starts("kv.")) {
      st.image.kv[layer_of(b.name, "kv.")] = b.bytes;
    } else if (starts("gdn.rec.")) {
      st.image.gdn_rec[layer_of(b.name, "gdn.rec.")] = b.bytes;
    } else if (starts("gdn.conv.")) {
      if (b.bytes.size() % sizeof(uint16_t) != 0) throw std::runtime_error("state blob: a conv record is not whole bf16 values");
      std::vector<uint16_t>& c = st.image.gdn_conv[layer_of(b.name, "gdn.conv.")];
      c.resize(b.bytes.size() / sizeof(uint16_t));
      std::memcpy(c.data(), b.bytes.data(), b.bytes.size());
    } else if (starts("x.")) {
      st.extra[b.name.substr(2)] = b.bytes;
    } else {
      throw std::runtime_error("state blob: unknown record '" + b.name + "'");
    }
  }
  if (!have_scalars) throw std::runtime_error("state blob: no scalars record");
  return st;
}

inline void WriteLiveState(const std::string& path, const LiveState& st) { WriteBlobFile(path, LiveStateToBlobs(st)); }
inline LiveState ReadLiveState(const std::string& path) { return LiveStateFromBlobs(ReadBlobFile(path)); }

}  // namespace r4dx::model::hybrid
