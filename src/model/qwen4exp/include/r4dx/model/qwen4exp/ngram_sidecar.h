// r4dx::model::qwen4exp::NgramSidecar -- metadata and open of the qwen4_exp n-gram table sidecar
// (docs/qwen4exp-plan.md 2.3, 4.2, 4.8; writer: tools/reference/qwen4exp/ngram_sidecar.py).
//
// The sidecar is one file: a fixed 4096-byte header followed by `rows` rows of `row_bytes` (320 = 160
// bf16) bytes in global row-id order, so row r lives at byte 4096 + r * 320. The header (little-endian,
// packed, zero padded to 4096 bytes):
//
//     off  type        field
//       0  char[8]     magic "R4DXNGRM"
//       8  u32         version (1)
//      12  u64         rows (rows in the body)
//      20  u32         row_bytes (head_dim * 2)
//      24  u32         heads (16)
//      28  u32         head_dim (160)
//      32  u32         ngram_size (3)
//      36  u32         heads_per_ngram (8)
//      40  i64         eos token id
//      48  u64         seed
//      56  i64[3]      layer multipliers (splitmix64 of the seed; they come from here, never recomputed)
//      80  u64[16]     head offsets
//     208  u64[16]     head primes (per-head vocabulary sizes)
//     336  u64         padded_rows
//     344  u8[32]      sha256 of the body
//     376              zero padding up to 4096
//
// Open parses and validates the header and checks the file size; it never reads the body (95 GiB for the
// real table). VerifyBodySha256 hashes the whole body and is meant for tools, not for model load.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>

namespace r4dx::model::qwen4exp {

struct NgramSidecarHeader {
  static constexpr size_t kHeaderBytes = 4096;
  static constexpr uint32_t kVersion = 1;
  static constexpr uint32_t kHeads = 16;
  static constexpr uint32_t kHeadDim = 160;
  static constexpr uint32_t kRowBytes = kHeadDim * 2;  // bf16
  static constexpr uint32_t kNgramSize = 3;
  static constexpr uint32_t kHeadsPerNgram = 8;

  uint32_t version = 0;
  uint64_t rows = 0;
  uint32_t row_bytes = 0;
  uint32_t heads = 0;
  uint32_t head_dim = 0;
  uint32_t ngram_size = 0;
  uint32_t heads_per_ngram = 0;
  int64_t eos_token_id = 0;
  uint64_t seed = 0;
  std::array<int64_t, kNgramSize> multipliers{};
  std::array<uint64_t, kHeads> head_offsets{};
  std::array<uint64_t, kHeads> head_primes{};
  uint64_t padded_rows = 0;
  std::array<uint8_t, 32> body_sha256{};

  // Strict parse of the first 4096 bytes (`len` must be >= kHeaderBytes): magic, version, the geometry the
  // runtime is built for (heads 16, head_dim 160, row_bytes 320, 3-grams, 8 heads per order), zero padding,
  // positive multipliers and primes. Throws std::runtime_error naming the offending field.
  static NgramSidecarHeader Parse(const uint8_t* buf, size_t len);

  // Byte offset of row `row` in the file.
  static constexpr uint64_t RowOffset(uint64_t row) { return kHeaderBytes + row * kRowBytes; }
  uint64_t BodyBytes() const { return rows * row_bytes; }
  uint64_t FileBytes() const { return kHeaderBytes + BodyBytes(); }
};

class NgramSidecar {
 public:
  // Reads and validates the header; throws std::runtime_error when the file is missing, the header is bad
  // or (unless allow_truncated) the file is not exactly header + rows * row_bytes long. allow_truncated
  // exists for tests that exercise the reader's short-read path; a runtime load never sets it.
  static NgramSidecar Open(const std::filesystem::path& path, bool allow_truncated = false);

  const std::filesystem::path& path() const { return path_; }
  const NgramSidecarHeader& header() const { return header_; }
  uint64_t file_bytes() const { return file_bytes_; }  // actual size on disk

  // SHA-256 of the body bytes [4096, 4096 + rows * 320), streamed in 16 MiB reads. `progress` (optional)
  // gets (bytes hashed, body bytes) after each read. Throws std::runtime_error on an I/O failure or when
  // the file is shorter than the header claims.
  std::array<uint8_t, 32> ComputeBodySha256(const std::function<void(uint64_t, uint64_t)>& progress = {}) const;
  // ComputeBodySha256() == header().body_sha256.
  bool VerifyBodySha256(const std::function<void(uint64_t, uint64_t)>& progress = {}) const;

 private:
  std::filesystem::path path_;
  NgramSidecarHeader header_;
  uint64_t file_bytes_ = 0;
};

}  // namespace r4dx::model::qwen4exp
