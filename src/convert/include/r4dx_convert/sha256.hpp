// r4dx_convert::Sha256Hex -- a small self-contained SHA-256 (public-domain algorithm, FIPS 180-4)
// so `__metadata__.config_sha256` (docs/container-format.md) doesn't need an extra dependency for
// one hash of one JSON file.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace r4dx_convert {

class Sha256 {
 public:
  Sha256() { Reset(); }

  void Update(const uint8_t* data, size_t len) {
    total_len_ += len;
    while (len > 0) {
      size_t take = 64 - buf_len_;
      if (take > len) take = len;
      std::memcpy(buf_ + buf_len_, data, take);
      buf_len_ += take;
      data += take;
      len -= take;
      if (buf_len_ == 64) {
        Transform(buf_);
        buf_len_ = 0;
      }
    }
  }

  std::string HexDigest() {
    uint64_t bitlen = total_len_ * 8;
    uint8_t pad = 0x80;
    Update(&pad, 1);
    uint8_t zero = 0;
    while (buf_len_ != 56) Update(&zero, 1);
    uint8_t lenbytes[8];
    for (int i = 0; i < 8; ++i) lenbytes[i] = static_cast<uint8_t>(bitlen >> (56 - 8 * i));
    Update(lenbytes, 8);

    char hex[65];
    for (int i = 0; i < 8; ++i) {
      std::snprintf(hex + i * 8, 9, "%08x", h_[i]);
    }
    return std::string(hex, 64);
  }

 private:
  void Reset() {
    h_[0] = 0x6a09e667u; h_[1] = 0xbb67ae85u; h_[2] = 0x3c6ef372u; h_[3] = 0xa54ff53au;
    h_[4] = 0x510e527fu; h_[5] = 0x9b05688cu; h_[6] = 0x1f83d9abu; h_[7] = 0x5be0cd19u;
    buf_len_ = 0;
    total_len_ = 0;
  }

  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void Transform(const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) | (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  uint32_t h_[8];
  uint8_t buf_[64];
  size_t buf_len_ = 0;
  uint64_t total_len_ = 0;
};

inline std::string Sha256Hex(const std::string& data) {
  Sha256 s;
  s.Update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return s.HexDigest();
}

}  // namespace r4dx_convert
