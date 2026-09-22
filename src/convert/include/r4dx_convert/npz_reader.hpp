// r4dx_convert::NpzReader -- a minimal reader for the importance-matrix file
// tools/reference/imatrix_capture.py writes with `numpy.savez`.
//
// Scope, deliberately narrow: `np.savez` produces a ZIP archive whose members are STORED
// (compression method 0, no deflate) `.npy` v1/v2 files, and every member of an imatrix is a 1-D
// little-endian float32 vector whose name is the converter's own container base (e.g.
// "text.layers.3.attn.qg"), so this reader supports exactly that and refuses everything else with
// a message that says what to do instead. No zlib, no third-party zip library, no mmap.
//
// Zip64 matters here: numpy writes `allowZip64=True`, so even a 10 MiB archive carries 0xFFFFFFFF
// size placeholders in the local file header with the real 64-bit sizes in the 0x0001 extra field
// (verified against D:\models\r4dx\qwen38-27b.imatrix.npz, whose very first member is written that
// way). Reading only the 32-bit fields would produce a 4 GiB "size" and a nonsense walk.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace r4dx_convert {

class NpzReader {
 public:
  // `path` is a narrow (UTF-8 / ACP) path. The whole archive is read into memory once -- an imatrix
  // for this checkpoint is ~10 MiB, so streaming would be complexity for nothing.
  explicit NpzReader(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("r4dx_convert: cannot open imatrix npz " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff len = f.tellg();
    if (len <= 0) throw std::runtime_error("r4dx_convert: empty imatrix npz " + path);
    f.seekg(0, std::ios::beg);
    buf_.resize(static_cast<size_t>(len));
    f.read(reinterpret_cast<char*>(buf_.data()), len);
    if (!f) throw std::runtime_error("r4dx_convert: short read on imatrix npz " + path);
    Parse(path);
    buf_.clear();
    buf_.shrink_to_fit();
  }

  bool Has(const std::string& name) const { return entries_.count(name) != 0; }
  size_t Count() const { return entries_.size(); }

  // Returns a pointer to this reader's copy of the vector (valid for the reader's lifetime) plus
  // the element count, or {nullptr, 0} if `name` is absent. The payload is copied out during Parse
  // rather than aliased in place: numpy aligns the .npy DATA to 64 bytes inside each member, but a
  // member's own start inside the zip is only 30 + filename + extra-field bytes in, so the payload
  // is not reliably 4-byte aligned in the archive and `reinterpret_cast<const float*>` on it would
  // be UB.
  const float* Vector(const std::string& name, int64_t* out_len) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
      *out_len = 0;
      return nullptr;
    }
    *out_len = static_cast<int64_t>(it->second.size());
    return it->second.data();
  }

  std::vector<std::string> Names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& kv : entries_) out.push_back(kv.first);
    return out;
  }

 private:
  uint16_t U16(size_t p) const {
    Need(p, 2);
    uint16_t v;
    std::memcpy(&v, buf_.data() + p, 2);
    return v;
  }
  uint32_t U32(size_t p) const {
    Need(p, 4);
    uint32_t v;
    std::memcpy(&v, buf_.data() + p, 4);
    return v;
  }
  uint64_t U64(size_t p) const {
    Need(p, 8);
    uint64_t v;
    std::memcpy(&v, buf_.data() + p, 8);
    return v;
  }
  void Need(size_t p, size_t n) const {
    if (p + n > buf_.size()) throw std::runtime_error("r4dx_convert: truncated imatrix npz");
  }

  void Parse(const std::string& path) {
    size_t p = 0;
    while (p + 4 <= buf_.size()) {
      const uint32_t sig = U32(p);
      if (sig == 0x02014b50u || sig == 0x06054b50u || sig == 0x06064b50u) break;  // central dir / EOCD
      if (sig != 0x04034b50u)
        throw std::runtime_error("r4dx_convert: not a zip/npz archive (bad local header) " + path);

      const uint16_t flags = U16(p + 6);
      const uint16_t method = U16(p + 8);
      uint64_t csize = U32(p + 18);
      uint64_t usize = U32(p + 22);
      const uint16_t fnlen = U16(p + 26);
      const uint16_t extralen = U16(p + 28);
      const size_t name_at = p + 30;
      Need(name_at, fnlen);
      std::string name(reinterpret_cast<const char*>(buf_.data() + name_at), fnlen);
      const size_t extra_at = name_at + fnlen;
      Need(extra_at, extralen);

      if (flags & 0x8u)
        throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                                 "' uses a streamed data descriptor (unsupported)");
      if (method != 0)
        throw std::runtime_error(
            "r4dx_convert: imatrix npz member '" + name +
            "' is compressed (zip method " + std::to_string(method) +
            "); re-save with numpy.savez, not numpy.savez_compressed");

      // Zip64 extended information extra field (id 0x0001): 8-byte uncompressed size then 8-byte
      // compressed size, each present only if the corresponding 32-bit field was 0xFFFFFFFF.
      if (csize == 0xFFFFFFFFull || usize == 0xFFFFFFFFull) {
        size_t e = extra_at;
        const size_t extra_end = extra_at + extralen;
        bool found = false;
        while (e + 4 <= extra_end) {
          const uint16_t id = U16(e), sz = U16(e + 2);
          if (id == 0x0001u) {
            size_t q = e + 4;
            if (usize == 0xFFFFFFFFull) {
              usize = U64(q);
              q += 8;
            }
            if (csize == 0xFFFFFFFFull) csize = U64(q);
            found = true;
            break;
          }
          e += 4u + sz;
        }
        if (!found)
          throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                                   "' declares zip64 sizes but has no 0x0001 extra field");
      }
      if (csize != usize)
        throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                                 "' is not stored verbatim");

      const size_t data_at = extra_at + extralen;
      Need(data_at, static_cast<size_t>(csize));
      if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0)
        entries_.emplace(name.substr(0, name.size() - 4),
                         ParseNpy(name, data_at, static_cast<size_t>(usize)));
      p = data_at + static_cast<size_t>(csize);
    }
    if (entries_.empty())
      throw std::runtime_error("r4dx_convert: imatrix npz " + path + " contains no .npy members");
  }

  // .npy v1/v2: magic "\x93NUMPY", major, minor, then a 2-byte (v1) or 4-byte (v2) little-endian
  // header length followed by that many bytes of ASCII dict literal. Only '<f4', C order and a 1-D
  // shape are accepted -- a float64 or 2-D imatrix would be a Stage-1 bug, not something to coerce.
  std::vector<float> ParseNpy(const std::string& name, size_t at, size_t total) const {
    static const unsigned char kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    Need(at, 10);
    if (std::memcmp(buf_.data() + at, kMagic, 6) != 0)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name + "' is not a .npy file");
    const uint8_t major = buf_[at + 6];
    size_t hdr_len, hdr_at;
    if (major == 1) {
      hdr_len = U16(at + 8);
      hdr_at = at + 10;
    } else if (major == 2 || major == 3) {
      hdr_len = U32(at + 8);
      hdr_at = at + 12;
    } else {
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name + "' has .npy version " +
                               std::to_string(major));
    }
    Need(hdr_at, hdr_len);
    const std::string hdr(reinterpret_cast<const char*>(buf_.data() + hdr_at), hdr_len);

    if (hdr.find("'<f4'") == std::string::npos && hdr.find("\"<f4\"") == std::string::npos)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                               "' is not little-endian float32 (header: " + hdr + ")");
    if (hdr.find("'fortran_order': False") == std::string::npos)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                               "' is not C-ordered (header: " + hdr + ")");

    // shape: exactly one dimension, "(<n>,)".
    const size_t sp = hdr.find("'shape'");
    if (sp == std::string::npos)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name + "' has no shape");
    const size_t open = hdr.find('(', sp);
    const size_t close = hdr.find(')', open);
    if (open == std::string::npos || close == std::string::npos)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name + "' has a malformed shape");
    const std::string shape = hdr.substr(open + 1, close - open - 1);
    if (shape.find(',') != shape.rfind(','))
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name +
                               "' is not 1-D (shape (" + shape + "))");
    const int64_t count = std::stoll(shape);

    const size_t payload_at = hdr_at + hdr_len;
    const size_t payload_bytes = total - (payload_at - at);
    if (count < 0 || payload_bytes != static_cast<size_t>(count) * 4)
      throw std::runtime_error("r4dx_convert: imatrix npz member '" + name + "' payload is " +
                               std::to_string(payload_bytes) + " bytes, expected " +
                               std::to_string(count * 4));
    std::vector<float> out(static_cast<size_t>(count));
    if (count > 0) std::memcpy(out.data(), buf_.data() + payload_at, payload_bytes);
    return out;
  }

  std::vector<uint8_t> buf_;
  std::map<std::string, std::vector<float>> entries_;
};

}  // namespace r4dx_convert
