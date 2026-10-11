// NgramSidecar: header parse / open / body hash. See ngram_sidecar.h.
#include "r4dx/model/qwen4exp/ngram_sidecar.h"

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace r4dx::model::qwen4exp {

namespace {

constexpr char kMagic[8] = {'R', '4', 'D', 'X', 'N', 'G', 'R', 'M'};
// struct.Struct("<8sIQIIIIIqQ3q16Q16QQ32s").size, then zero padding to 4096.
constexpr size_t kPackedBytes = 376;

[[noreturn]] void Bad(const std::string& what) { throw std::runtime_error("ngram sidecar header: " + what); }

class Cursor {
 public:
  explicit Cursor(const uint8_t* p) : p_(p) {}
  template <class T>
  T Get() {
    T v;
    std::memcpy(&v, p_ + off_, sizeof(T));
    off_ += sizeof(T);
    return v;
  }
  size_t offset() const { return off_; }

 private:
  const uint8_t* p_;
  size_t off_ = 0;
};

void CheckNt(NTSTATUS st, const char* what) {
  if (st < 0) throw std::runtime_error(std::string("ngram sidecar: ") + what + " failed, NTSTATUS " +
                                       std::to_string(static_cast<long>(st)));
}

}  // namespace

NgramSidecarHeader NgramSidecarHeader::Parse(const uint8_t* buf, size_t len) {
  static_assert(sizeof(uint8_t) == 1 && sizeof(int64_t) == 8, "layout assumptions");
  if (len < kHeaderBytes) Bad("only " + std::to_string(len) + " bytes, expected " + std::to_string(kHeaderBytes));
  if (std::memcmp(buf, kMagic, sizeof(kMagic)) != 0) Bad("bad magic (not an R4DXNGRM file)");
  Cursor c(buf + sizeof(kMagic));
  NgramSidecarHeader h;
  h.version = c.Get<uint32_t>();
  h.rows = c.Get<uint64_t>();
  h.row_bytes = c.Get<uint32_t>();
  h.heads = c.Get<uint32_t>();
  h.head_dim = c.Get<uint32_t>();
  h.ngram_size = c.Get<uint32_t>();
  h.heads_per_ngram = c.Get<uint32_t>();
  h.eos_token_id = c.Get<int64_t>();
  h.seed = c.Get<uint64_t>();
  for (auto& m : h.multipliers) m = c.Get<int64_t>();
  for (auto& o : h.head_offsets) o = c.Get<uint64_t>();
  for (auto& p : h.head_primes) p = c.Get<uint64_t>();
  h.padded_rows = c.Get<uint64_t>();
  for (auto& b : h.body_sha256) b = c.Get<uint8_t>();
  if (sizeof(kMagic) + c.offset() != kPackedBytes) Bad("internal layout error (packed size)");

  if (h.version != kVersion) Bad("unsupported version " + std::to_string(h.version) + " (this build reads " + std::to_string(kVersion) + ")");
  if (h.heads != kHeads) Bad("heads " + std::to_string(h.heads) + " != " + std::to_string(kHeads));
  if (h.head_dim != kHeadDim) Bad("head_dim " + std::to_string(h.head_dim) + " != " + std::to_string(kHeadDim));
  if (h.row_bytes != kRowBytes) Bad("row_bytes " + std::to_string(h.row_bytes) + " != " + std::to_string(kRowBytes));
  if (h.ngram_size != kNgramSize) Bad("ngram_size " + std::to_string(h.ngram_size) + " != " + std::to_string(kNgramSize));
  if (h.heads_per_ngram != kHeadsPerNgram)
    Bad("heads_per_ngram " + std::to_string(h.heads_per_ngram) + " != " + std::to_string(kHeadsPerNgram));
  if (h.heads != (h.ngram_size - 1) * h.heads_per_ngram) Bad("heads != (ngram_size - 1) * heads_per_ngram");
  if (h.rows == 0) Bad("rows == 0");
  if (h.padded_rows < h.rows) Bad("padded_rows " + std::to_string(h.padded_rows) + " < rows " + std::to_string(h.rows));
  if (h.rows > (UINT64_MAX - kHeaderBytes) / kRowBytes) Bad("rows " + std::to_string(h.rows) + " overflows the file size");
  for (size_t i = 0; i < h.multipliers.size(); ++i)
    if (h.multipliers[i] <= 0) Bad("multiplier " + std::to_string(i) + " is not positive");
  for (size_t i = 0; i < h.head_primes.size(); ++i)
    if (h.head_primes[i] == 0) Bad("head prime " + std::to_string(i) + " is 0");
  for (size_t i = kPackedBytes; i < kHeaderBytes; ++i)
    if (buf[i] != 0) Bad("padding is not zero at byte " + std::to_string(i));
  return h;
}

NgramSidecar NgramSidecar::Open(const std::filesystem::path& path, bool allow_truncated) {
  const std::string shown = path.string();
  std::error_code ec;
  const uint64_t size = std::filesystem::file_size(path, ec);
  if (ec) throw std::runtime_error("ngram sidecar: cannot stat " + shown + ": " + ec.message());
  if (size < NgramSidecarHeader::kHeaderBytes)
    throw std::runtime_error("ngram sidecar: " + shown + " is " + std::to_string(size) + " bytes, shorter than the " +
                             std::to_string(NgramSidecarHeader::kHeaderBytes) + "-byte header");
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("ngram sidecar: cannot open " + shown);
  std::vector<uint8_t> buf(NgramSidecarHeader::kHeaderBytes);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  if (static_cast<size_t>(f.gcount()) != buf.size())
    throw std::runtime_error("ngram sidecar: short read of the header of " + shown);

  NgramSidecar s;
  try {
    s.header_ = NgramSidecarHeader::Parse(buf.data(), buf.size());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(e.what()) + " [" + shown + "]");
  }
  s.path_ = path;
  s.file_bytes_ = size;
  const uint64_t want = s.header_.FileBytes();
  if (size != want && !(allow_truncated && size < want))
    throw std::runtime_error("ngram sidecar: " + shown + " is " + std::to_string(size) + " bytes, header says " +
                             std::to_string(s.header_.rows) + " rows = " + std::to_string(want) + " bytes");
  return s;
}

std::array<uint8_t, 32> NgramSidecar::ComputeBodySha256(const std::function<void(uint64_t, uint64_t)>& progress) const {
  const uint64_t total = header_.BodyBytes();
  if (file_bytes_ < header_.FileBytes())
    throw std::runtime_error("ngram sidecar: " + path_.string() + " is truncated (" + std::to_string(file_bytes_) +
                             " bytes, header says " + std::to_string(header_.FileBytes()) + ")");
  std::ifstream f(path_, std::ios::binary);
  if (!f) throw std::runtime_error("ngram sidecar: cannot open " + path_.string());
  f.seekg(static_cast<std::streamoff>(NgramSidecarHeader::kHeaderBytes));

  BCRYPT_ALG_HANDLE alg = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  CheckNt(BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0), "BCryptOpenAlgorithmProvider");
  struct Closer {
    BCRYPT_ALG_HANDLE& a;
    BCRYPT_HASH_HANDLE& h;
    ~Closer() {
      if (h) BCryptDestroyHash(h);
      if (a) BCryptCloseAlgorithmProvider(a, 0);
    }
  } closer{alg, hash};
  CheckNt(BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0), "BCryptCreateHash");

  constexpr size_t kChunk = size_t{16} << 20;
  std::vector<uint8_t> chunk(kChunk);
  uint64_t done = 0;
  while (done < total) {
    const size_t want = static_cast<size_t>(std::min<uint64_t>(kChunk, total - done));
    f.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(want));
    if (static_cast<size_t>(f.gcount()) != want)
      throw std::runtime_error("ngram sidecar: short read hashing " + path_.string() + " at byte " +
                               std::to_string(NgramSidecarHeader::kHeaderBytes + done));
    CheckNt(BCryptHashData(hash, chunk.data(), static_cast<ULONG>(want), 0), "BCryptHashData");
    done += want;
    if (progress) progress(done, total);
  }
  std::array<uint8_t, 32> out{};
  CheckNt(BCryptFinishHash(hash, out.data(), static_cast<ULONG>(out.size()), 0), "BCryptFinishHash");
  return out;
}

bool NgramSidecar::VerifyBodySha256(const std::function<void(uint64_t, uint64_t)>& progress) const {
  return ComputeBodySha256(progress) == header_.body_sha256;
}

}  // namespace r4dx::model::qwen4exp
