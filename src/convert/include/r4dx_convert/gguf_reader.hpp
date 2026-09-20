// r4dx_convert::GgufReader -- minimal, memory-mapped GGUF v3 reader (little-endian; GGUF defines
// no other byte order). Independent, from-scratch implementation (task A1 rule: "Do NOT import
// gguf-py from the ROCmFPX checkout in committed code -- write your own minimal GGUF/Q8_0
// parsing") -- cross-checked, not copied, against the read-only reference at
// C:\Users\user\dev\ROCmFPX\gguf-py\gguf\gguf_reader.py (binary layout) and quants.py (Q8_0
// block dequant: value = f16(d) * int8(qs), block = 32 elements = 2+32 bytes) via
// $env:R4DX_REFERENCE_VENV-independent plain byte inspection of the real file, not by reading
// gguf-py's own code into this file.
//
// GGUF v3 layout (all little-endian):
//   [4]  magic "GGUF"
//   [4]  uint32 version (this reader requires 3; a different version could reshuffle metadata
//        value-type widths, so failing loudly beats guessing)
//   [8]  int64  tensor_count
//   [8]  int64  metadata_kv_count
//   metadata_kv_count x { string key; uint32 value_type; <value> }
//   tensor_count x { string name; uint32 n_dims; uint64 ne[n_dims]; uint32 ggml_type;
//                     uint64 offset (relative to the aligned data section start) }
//   padding to general.alignment (default 32, an optional uint32 metadata key)
//   tensor data, one blob per tensor at data_start + offset
//
// A GGUF `string` is uint64 length + that many UTF-8 bytes (no NUL terminator, no trailing pad).
// A GGUF `array` is uint32 element_type + uint64 count + that many elements (recursively, though
// this reader only encounters flat arrays of scalars/strings in practice).
//
// `ne` (tensor shape) is stored innermost-dimension-first: ne[0] is the fastest-varying axis, i.e.
// for a 2D linear weight ne[0] = in_features (K), ne[1] = out_features (N) -- row-major storage
// with row = one output feature, K contiguous (docs/container-format.md's own convention, and
// confirmed against the real Qwen3.8-27B-DFlash2-Q8_0.gguf: attn_conv_proj.weight has ne=
// [5120,1280], i.e. K=5120 in, N=1280 out, matching "attn_conv_proj [5120 -> 1280]").
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "r4dx/core/dtype.hpp"  // Bf16ToFloat/FloatToBf16/F16ToFloat/FloatToF16
#include "r4dx_convert/safetensors_reader.hpp"  // Utf8ToWide, WideToUtf8 -- shared UTF-8/Win32 path helpers

namespace r4dx_convert {

// ---- metadata value types ---------------------------------------------------------------------

enum class GgufValueType : uint32_t {
  kUInt8 = 0,
  kInt8 = 1,
  kUInt16 = 2,
  kInt16 = 3,
  kUInt32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUInt64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

// Byte width of a scalar GGUF value type; throws for kString/kArray (variable-width, handled
// separately) or an unrecognized numeric value (a GGUF version/type this reader doesn't know).
inline uint64_t GgufScalarTypeSize(GgufValueType t) {
  switch (t) {
    case GgufValueType::kUInt8:
    case GgufValueType::kInt8:
    case GgufValueType::kBool:
      return 1;
    case GgufValueType::kUInt16:
    case GgufValueType::kInt16:
      return 2;
    case GgufValueType::kUInt32:
    case GgufValueType::kInt32:
    case GgufValueType::kFloat32:
      return 4;
    case GgufValueType::kUInt64:
    case GgufValueType::kInt64:
    case GgufValueType::kFloat64:
      return 8;
    default:
      throw std::runtime_error("GgufScalarTypeSize: not a fixed-width scalar type");
  }
}

// One metadata value. Only one of the payload members is meaningful, selected by `type`:
// integers/bool widen into `u`/`i` (sign-extended for signed types), floats into `f`, strings into
// `s`, arrays into `arr` (each element a full GgufValue, `arr_elem_type` giving the common
// declared element type).
struct GgufValue {
  GgufValueType type = GgufValueType::kUInt8;
  uint64_t u = 0;
  int64_t i = 0;
  double f = 0.0;
  bool b = false;
  std::string s;
  GgufValueType arr_elem_type = GgufValueType::kUInt8;
  std::vector<GgufValue> arr;

  int64_t AsInt() const {
    switch (type) {
      case GgufValueType::kUInt8:
      case GgufValueType::kUInt16:
      case GgufValueType::kUInt32:
      case GgufValueType::kUInt64:
        return static_cast<int64_t>(u);
      case GgufValueType::kInt8:
      case GgufValueType::kInt16:
      case GgufValueType::kInt32:
      case GgufValueType::kInt64:
        return i;
      case GgufValueType::kBool:
        return b ? 1 : 0;
      default:
        throw std::runtime_error("GgufValue::AsInt: not an integer/bool type");
    }
  }
  double AsFloat() const {
    if (type == GgufValueType::kFloat32 || type == GgufValueType::kFloat64) return f;
    // A caller asking a plain integer KV for AsFloat() (e.g. a metadata schema that could be
    // either) is a reasonable, non-lossy widen -- allow it.
    return static_cast<double>(AsInt());
  }
  bool AsBool() const {
    if (type == GgufValueType::kBool) return b;
    return AsInt() != 0;
  }
  const std::string& AsString() const {
    if (type != GgufValueType::kString) throw std::runtime_error("GgufValue::AsString: not a string");
    return s;
  }
  std::vector<int64_t> AsIntArray() const {
    if (type != GgufValueType::kArray) throw std::runtime_error("GgufValue::AsIntArray: not an array");
    std::vector<int64_t> out;
    out.reserve(arr.size());
    for (auto& e : arr) out.push_back(e.AsInt());
    return out;
  }
  std::vector<bool> AsBoolArray() const {
    if (type != GgufValueType::kArray) throw std::runtime_error("GgufValue::AsBoolArray: not an array");
    std::vector<bool> out;
    out.reserve(arr.size());
    for (auto& e : arr) out.push_back(e.AsBool());
    return out;
  }
  std::vector<std::string> AsStringArray() const {
    if (type != GgufValueType::kArray) throw std::runtime_error("GgufValue::AsStringArray: not an array");
    std::vector<std::string> out;
    out.reserve(arr.size());
    for (auto& e : arr) out.push_back(e.AsString());
    return out;
  }
};

// ---- tensor dtypes (ggml_type) ------------------------------------------------------------------

// Only the subset this converter actually needs to read; any other ggml_type is a hard error
// rather than a silent misinterpretation. Numeric values match ggml.h's `enum ggml_type`
// (cross-checked against the read-only ROCmFPX gguf-py's GGMLQuantizationType, not imported).
enum class GgmlType : uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ8_0 = 8,
  kI8 = 24,
  kI16 = 25,
  kI32 = 26,
  kI64 = 27,
  kF64 = 28,
  kBF16 = 30,
};

// Elements per quantization block (1 for every unquantized type).
inline int64_t GgmlBlockSize(GgmlType t) {
  switch (t) {
    case GgmlType::kQ8_0:
      return 32;
    default:
      return 1;
  }
}

// On-disk bytes per block (== per element for unquantized types).
inline int64_t GgmlTypeSize(GgmlType t) {
  switch (t) {
    case GgmlType::kF32:
    case GgmlType::kI32:
      return 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
    case GgmlType::kI16:
      return 2;
    case GgmlType::kI8:
      return 1;
    case GgmlType::kI64:
    case GgmlType::kF64:
      return 8;
    case GgmlType::kQ8_0:
      return 34;  // 2 (f16 scale) + 32 (int8) per block of 32 elements
    default:
      throw std::runtime_error("GgmlTypeSize: unsupported/unknown ggml_type " +
                                std::to_string(static_cast<uint32_t>(t)));
  }
}

inline const char* GgmlTypeName(GgmlType t) {
  switch (t) {
    case GgmlType::kF32: return "F32";
    case GgmlType::kF16: return "F16";
    case GgmlType::kBF16: return "BF16";
    case GgmlType::kQ8_0: return "Q8_0";
    case GgmlType::kI8: return "I8";
    case GgmlType::kI16: return "I16";
    case GgmlType::kI32: return "I32";
    case GgmlType::kI64: return "I64";
    case GgmlType::kF64: return "F64";
    default: return "UNKNOWN";
  }
}

struct GgufTensorInfo {
  std::string name;
  std::vector<int64_t> ne;  // GGUF order: ne[0] fastest/innermost
  GgmlType type = GgmlType::kF32;
  uint64_t offset = 0;  // relative to the (aligned) data section start
  uint64_t nbytes = 0;  // computed from ne + type block/type size

  int64_t ElemCount() const {
    int64_t n = 1;
    for (auto d : ne) n *= d;
    return n;
  }
};

// ---- reader --------------------------------------------------------------------------------

class GgufReader {
 public:
  explicit GgufReader(const std::wstring& path) : path_display_(WideToUtf8(path)) {
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE)
      throw std::runtime_error("gguf: cannot open " + path_display_ + " (CreateFileW failed)");
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file_, &sz))
      throw std::runtime_error("gguf: GetFileSizeEx failed for " + path_display_);
    file_size_ = static_cast<uint64_t>(sz.QuadPart);
    mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) throw std::runtime_error("gguf: CreateFileMappingW failed for " + path_display_);
    view_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!view_) throw std::runtime_error("gguf: MapViewOfFile failed for " + path_display_);

    Cursor c{view_, file_size_, 0, path_display_};
    char magic[4];
    c.Read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0)
      throw std::runtime_error("gguf: bad magic in " + path_display_);
    version_ = c.ReadU32();
    if (version_ != 3)
      throw std::runtime_error("gguf: unsupported version " + std::to_string(version_) +
                                " in " + path_display_ + " (this reader only handles v3)");
    const int64_t n_tensors = static_cast<int64_t>(c.ReadU64());
    const int64_t n_kv = static_cast<int64_t>(c.ReadU64());
    if (n_tensors < 0 || n_kv < 0)
      throw std::runtime_error("gguf: negative tensor/kv count in " + path_display_);

    for (int64_t i = 0; i < n_kv; ++i) {
      std::string key = c.ReadString();
      GgufValue val = ReadValue(c);
      kv_.emplace(std::move(key), std::move(val));
    }

    if (HasKey("general.alignment")) {
      alignment_ = static_cast<uint64_t>(Kv("general.alignment").AsInt());
      if (alignment_ == 0) throw std::runtime_error("gguf: general.alignment is 0 in " + path_display_);
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
      GgufTensorInfo info;
      info.name = c.ReadString();
      const uint32_t n_dims = c.ReadU32();
      info.ne.resize(n_dims);
      for (uint32_t d = 0; d < n_dims; ++d) info.ne[d] = static_cast<int64_t>(c.ReadU64());
      info.type = static_cast<GgmlType>(c.ReadU32());
      info.offset = c.ReadU64();
      const int64_t elems = info.ElemCount();
      const int64_t block = GgmlBlockSize(info.type);
      if (block <= 0 || elems % block != 0) {
        throw std::runtime_error("gguf: tensor '" + info.name + "' element count " +
                                  std::to_string(elems) + " not divisible by its type's block size " +
                                  std::to_string(block));
      }
      info.nbytes = static_cast<uint64_t>((elems / block) * GgmlTypeSize(info.type));
      tensor_order_.push_back(info.name);
      tensors_.emplace(info.name, std::move(info));
    }

    // Data section starts at the next `alignment_`-aligned offset after the tensor-info table.
    const uint64_t after_infos = c.pos;
    data_start_ = (after_infos % alignment_ == 0) ? after_infos
                                                   : after_infos + (alignment_ - after_infos % alignment_);

    for (auto& kv : tensors_) {
      GgufTensorInfo& info = kv.second;
      if (data_start_ + info.offset + info.nbytes > file_size_) {
        throw std::runtime_error("gguf: tensor '" + info.name + "' in " + path_display_ +
                                  " needs bytes [" + std::to_string(data_start_ + info.offset) + "," +
                                  std::to_string(data_start_ + info.offset + info.nbytes) +
                                  ") but the file is only " + std::to_string(file_size_) +
                                  " bytes -- file truncated or still downloading");
      }
    }
  }

  ~GgufReader() {
    if (view_) UnmapViewOfFile(view_);
    if (mapping_) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
  }
  GgufReader(const GgufReader&) = delete;
  GgufReader& operator=(const GgufReader&) = delete;

  uint32_t Version() const { return version_; }
  uint64_t Alignment() const { return alignment_; }

  bool HasKey(const std::string& key) const { return kv_.count(key) != 0; }
  const GgufValue& Kv(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing metadata key '" + key + "' in " + path_display_);
    return it->second;
  }
  const std::unordered_map<std::string, GgufValue>& AllKv() const { return kv_; }

  int64_t GetInt(const std::string& key) const { return Kv(key).AsInt(); }
  double GetFloat(const std::string& key) const { return Kv(key).AsFloat(); }
  bool GetBool(const std::string& key) const { return Kv(key).AsBool(); }
  const std::string& GetString(const std::string& key) const { return Kv(key).AsString(); }
  std::vector<int64_t> GetIntArray(const std::string& key) const { return Kv(key).AsIntArray(); }
  std::vector<bool> GetBoolArray(const std::string& key) const { return Kv(key).AsBoolArray(); }
  std::vector<std::string> GetStringArray(const std::string& key) const { return Kv(key).AsStringArray(); }

  bool HasTensor(const std::string& name) const { return tensors_.count(name) != 0; }
  const GgufTensorInfo& TensorInfo(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("gguf: missing tensor '" + name + "' in " + path_display_);
    return it->second;
  }
  const std::vector<std::string>& TensorNames() const { return tensor_order_; }

  const uint8_t* TensorData(const std::string& name) const {
    const GgufTensorInfo& info = TensorInfo(name);
    return view_ + data_start_ + info.offset;
  }

  // ---- dequant helpers, one tensor at a time (the mmap pages the source bytes in on demand; the
  // whole file is never explicitly read into a heap buffer) --------------------------------------

  // F32/F16/BF16/Q8_0 -> row-major float32.
  std::vector<float> DequantToF32(const std::string& name) const {
    const GgufTensorInfo& info = TensorInfo(name);
    const int64_t n = info.ElemCount();
    std::vector<float> out(static_cast<size_t>(n));
    const uint8_t* data = TensorData(name);
    switch (info.type) {
      case GgmlType::kF32: {
        std::memcpy(out.data(), data, static_cast<size_t>(n) * 4);
        break;
      }
      case GgmlType::kF16: {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = r4dx::core::F16ToFloat(src[i]);
        break;
      }
      case GgmlType::kBF16: {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(data);
        for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = r4dx::core::Bf16ToFloat(src[i]);
        break;
      }
      case GgmlType::kQ8_0: {
        DequantQ8_0ToF32(data, n, out.data());
        break;
      }
      default:
        throw std::runtime_error(std::string("gguf: DequantToF32 unsupported ggml_type ") +
                                  GgmlTypeName(info.type) + " for '" + name + "'");
    }
    return out;
  }

  // Same, but rounds to bf16 (round-to-nearest-even via r4dx::core::FloatToBf16) instead of
  // returning float32 -- for tensors the container stores as bf16.
  std::vector<uint16_t> DequantToBf16(const std::string& name) const {
    std::vector<float> f = DequantToF32(name);
    std::vector<uint16_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) out[i] = r4dx::core::FloatToBf16(f[i]);
    return out;
  }

  // Raw bf16-typed tensor's bytes, unchanged (no float round-trip) -- for tensors already stored
  // BF16 in the GGUF that the container also stores as bf16 (avoids a needless requantize).
  std::vector<uint8_t> CopyRawBf16(const std::string& name) const {
    const GgufTensorInfo& info = TensorInfo(name);
    if (info.type != GgmlType::kBF16)
      throw std::runtime_error("gguf: CopyRawBf16: '" + name + "' is not BF16 (" +
                                GgmlTypeName(info.type) + ")");
    const uint8_t* data = TensorData(name);
    return std::vector<uint8_t>(data, data + info.nbytes);
  }

  // Standalone Q8_0 block dequant (also exposed free-standing below for fixture-free unit tests).
  static void DequantQ8_0ToF32(const uint8_t* data, int64_t n_elems, float* out) {
    if (n_elems % 32 != 0)
      throw std::runtime_error("gguf: Q8_0 element count not a multiple of 32");
    const int64_t n_blocks = n_elems / 32;
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t* block = data + static_cast<size_t>(b) * 34;
      uint16_t d16;
      std::memcpy(&d16, block, 2);
      const float d = r4dx::core::F16ToFloat(d16);
      const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
      float* row_out = out + b * 32;
      for (int k = 0; k < 32; ++k) row_out[k] = d * static_cast<float>(qs[k]);
    }
  }

 private:
  // Bounds-checked little-endian cursor over the mmap'd view (shared by the header/KV/tensor-info
  // parse, which all happen before any tensor DATA read).
  struct Cursor {
    const uint8_t* base;
    uint64_t size;
    uint64_t pos;
    const std::string& path_display;

    void Require(uint64_t n) const {
      // Written as `n > size - pos` (not `pos + n > size`) so a corrupt/huge length field can't
      // wrap the 64-bit addition and slip past the check (pos <= size is an invariant maintained
      // by every mutator below, so `size - pos` never underflows).
      if (n > size - pos)
        throw std::runtime_error("gguf: " + path_display + " truncated (need " + std::to_string(n) +
                                  " bytes at offset " + std::to_string(pos) + ", file is " +
                                  std::to_string(size) + " bytes)");
    }
    void Read(void* dst, uint64_t n) {
      Require(n);
      std::memcpy(dst, base + pos, n);
      pos += n;
    }
    uint32_t ReadU32() { uint32_t v; Read(&v, 4); return v; }
    uint64_t ReadU64() { uint64_t v; Read(&v, 8); return v; }
    std::string ReadString() {
      const uint64_t len = ReadU64();
      Require(len);
      std::string s(reinterpret_cast<const char*>(base + pos), static_cast<size_t>(len));
      pos += len;
      return s;
    }
  };

  GgufValue ReadValue(Cursor& c) {
    GgufValue v;
    v.type = static_cast<GgufValueType>(c.ReadU32());
    ReadValuePayload(c, v.type, v);
    return v;
  }

  void ReadValuePayload(Cursor& c, GgufValueType type, GgufValue& out) {
    switch (type) {
      case GgufValueType::kString:
        out.s = c.ReadString();
        return;
      case GgufValueType::kArray: {
        out.arr_elem_type = static_cast<GgufValueType>(c.ReadU32());
        const uint64_t count = c.ReadU64();
        out.arr.resize(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
          GgufValue elem;
          elem.type = out.arr_elem_type;
          ReadValuePayload(c, out.arr_elem_type, elem);
          out.arr[static_cast<size_t>(i)] = std::move(elem);
        }
        return;
      }
      case GgufValueType::kBool: {
        uint8_t b;
        c.Read(&b, 1);
        out.b = b != 0;
        return;
      }
      case GgufValueType::kFloat32: {
        float f;
        c.Read(&f, 4);
        out.f = f;
        return;
      }
      case GgufValueType::kFloat64: {
        double f;
        c.Read(&f, 8);
        out.f = f;
        return;
      }
      case GgufValueType::kUInt8: { uint8_t x; c.Read(&x, 1); out.u = x; return; }
      case GgufValueType::kUInt16: { uint16_t x; c.Read(&x, 2); out.u = x; return; }
      case GgufValueType::kUInt32: { uint32_t x; c.Read(&x, 4); out.u = x; return; }
      case GgufValueType::kUInt64: { uint64_t x; c.Read(&x, 8); out.u = x; return; }
      case GgufValueType::kInt8: { int8_t x; c.Read(&x, 1); out.i = x; return; }
      case GgufValueType::kInt16: { int16_t x; c.Read(&x, 2); out.i = x; return; }
      case GgufValueType::kInt32: { int32_t x; c.Read(&x, 4); out.i = x; return; }
      case GgufValueType::kInt64: { int64_t x; c.Read(&x, 8); out.i = x; return; }
      default:
        throw std::runtime_error("gguf: unknown metadata value type " +
                                  std::to_string(static_cast<uint32_t>(type)));
    }
  }

  std::string path_display_;
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
  const uint8_t* view_ = nullptr;
  uint64_t file_size_ = 0;
  uint32_t version_ = 0;
  uint64_t alignment_ = 32;
  uint64_t data_start_ = 0;
  std::unordered_map<std::string, GgufValue> kv_;
  std::vector<std::string> tensor_order_;
  std::unordered_map<std::string, GgufTensorInfo> tensors_;
};

}  // namespace r4dx_convert
