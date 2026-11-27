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
  kQ5_K = 13,
  kIQ3_S = 21,
  kIQ4_XS = 23,
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
    case GgmlType::kQ5_K:
    case GgmlType::kIQ3_S:
    case GgmlType::kIQ4_XS:
      return 256;
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
      return 34;   // 2 (f16 scale) + 32 (int8) per block of 32 elements
    case GgmlType::kQ5_K:
      return 176;  // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) per 256 elements
    case GgmlType::kIQ3_S:
      return 110;  // 2 (d) + 64 (qs) + 8 (qh) + 32 (signs) + 4 (scales) per 256 elements
    case GgmlType::kIQ4_XS:
      return 136;  // 2 (d) + 2 (scales_h) + 4 (scales_l) + 128 (qs) per 256 elements
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
    case GgmlType::kQ5_K: return "Q5_K";
    case GgmlType::kIQ3_S: return "IQ3_S";
    case GgmlType::kIQ4_XS: return "IQ4_XS";
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
      case GgmlType::kQ5_K: {
        DequantQ5_KToF32(data, n, out.data());
        break;
      }
      case GgmlType::kIQ3_S: {
        DequantIQ3_SToF32(data, n, out.data());
        break;
      }
      case GgmlType::kIQ4_XS: {
        DequantIQ4_XSToF32(data, n, out.data());
        break;
      }
      default:
        throw std::runtime_error(std::string("gguf: DequantToF32 unsupported ggml_type ") +
                                  GgmlTypeName(info.type) + " for '" + name + "'");
    }
    return out;
  }

  // Dequantize one contiguous slice (e.g. expert `slice_idx` of a 3D `[K, N, E]` tensor) to float32.
  std::vector<float> DequantSliceToF32(const std::string& name, int64_t slice_idx, int64_t slice_elems) const {
    const GgufTensorInfo& info = TensorInfo(name);
    const int64_t total = info.ElemCount();
    if (slice_idx < 0 || slice_elems <= 0 || (slice_idx + 1) * slice_elems > total) {
      throw std::runtime_error("gguf: DequantSliceToF32 out of range for '" + name + "'");
    }
    const int64_t block = GgmlBlockSize(info.type);
    if (slice_elems % block != 0) {
      throw std::runtime_error("gguf: DequantSliceToF32 slice_elems not divisible by block size for '" + name + "'");
    }
    const uint64_t slice_bytes = static_cast<uint64_t>((slice_elems / block) * GgmlTypeSize(info.type));
    const uint8_t* data = TensorData(name) + static_cast<size_t>(slice_idx) * slice_bytes;
    std::vector<float> out(static_cast<size_t>(slice_elems));
    DequantRawToF32(info.type, data, slice_elems, out.data(), name);
    return out;
  }

  static void DequantRawToF32(GgmlType type, const uint8_t* data, int64_t n, float* out,
                              const std::string& name_for_err = "") {
    switch (type) {
      case GgmlType::kF32:
        std::memcpy(out, data, static_cast<size_t>(n) * 4);
        break;
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
      case GgmlType::kQ8_0:
        DequantQ8_0ToF32(data, n, out);
        break;
      case GgmlType::kQ5_K:
        DequantQ5_KToF32(data, n, out);
        break;
      case GgmlType::kIQ3_S:
        DequantIQ3_SToF32(data, n, out);
        break;
      case GgmlType::kIQ4_XS:
        DequantIQ4_XSToF32(data, n, out);
        break;
      default:
        throw std::runtime_error(std::string("gguf: DequantRawToF32 unsupported ggml_type ") +
                                 GgmlTypeName(type) + " for '" + name_for_err + "'");
    }
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

  static void DequantQ5_KToF32(const uint8_t* data, int64_t n_elems, float* out) {
    if (n_elems % 256 != 0)
      throw std::runtime_error("gguf: Q5_K element count not a multiple of 256");
    const int64_t n_blocks = n_elems / 256;
    auto get_scale_min_k4 = [](int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
      if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
      } else {
        d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
      }
    };
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t* blk = data + static_cast<size_t>(b) * 176;
      uint16_t d16, dmin16;
      std::memcpy(&d16, blk, 2);
      std::memcpy(&dmin16, blk + 2, 2);
      const float d = r4dx::core::F16ToFloat(d16);
      const float dmin = r4dx::core::F16ToFloat(dmin16);
      const uint8_t* scales = blk + 4;
      const uint8_t* qh = blk + 16;
      const uint8_t* ql = blk + 48;
      float* y = out + b * 256;
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
      }
    }
  }

  static void DequantIQ4_XSToF32(const uint8_t* data, int64_t n_elems, float* out) {
    if (n_elems % 256 != 0)
      throw std::runtime_error("gguf: IQ4_XS element count not a multiple of 256");
    static constexpr int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const int64_t n_blocks = n_elems / 256;
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t* blk = data + static_cast<size_t>(b) * 136;
      uint16_t d16, scales_h;
      std::memcpy(&d16, blk, 2);
      std::memcpy(&scales_h, blk + 2, 2);
      const float d = r4dx::core::F16ToFloat(d16);
      const uint8_t* scales_l = blk + 4;
      const uint8_t* qs = blk + 8;
      float* y = out + b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
          y[j + 0] = dl * static_cast<float>(kIq4nl[qs[j] & 0xf]);
          y[j + 16] = dl * static_cast<float>(kIq4nl[qs[j] >> 4]);
        }
        y += 32;
        qs += 16;
      }
    }
  }

  static void DequantIQ3_SToF32(const uint8_t* data, int64_t n_elems, float* out) {
    if (n_elems % 256 != 0)
      throw std::runtime_error("gguf: IQ3_S element count not a multiple of 256");
    static const uint32_t* grid = []() -> const uint32_t* {
      static uint32_t tbl[512];
      static constexpr uint8_t kMap[8] = {0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f};
      static const char kHex[] =
          "0000010002000500070010001100120014001600200021002500330040004200"
          "4500470051005300600062007100740077000001010102010401100111011501"
          "2001230127013101350144016101650172010002010205020702100213021602"
          "2102250230023402420245024702510253027002730203031103150320032203"
          "3103330336034403500352036703710375030004130417042104240432044004"
          "4304510470040205040520052205260533054105450547056605730506061106"
          "1306310652067106000702070407200722072607330750075407001001100210"
          "0410101011101310151017102010221031103410361054105610611072100011"
          "0111031106111011141121113011331141115011521170117611001212121512"
          "1712201224123212401243125512601272120113041307131013131321132713"
          "3013341341136213701303140514121414143114331442144614501454140115"
          "1015131521153015321551152016241627164416461601170317101712172117"
          "3517411762177017002001200320052007201020122014201620212023202720"
          "3020322041204320452050205220672070207320752000210221102113211721"
          "2221252131213421422151210122042207222122232230223722412253225722"
          "7122742200230223052311232223242331233323422350236623012407242024"
          "2324322435244124722475240425112522253725402553257025002602260726"
          "2126552661260527112726273027432750270230113013301530173022303130"
          "3330353042304430473051306330713001310331053114312131233140316031"
          "7231763100321232203232323432503201331033143321332333273330334133"
          "4333473355337333033411341634223431345234603464340135103512352535"
          "3235443556357335163641360137033720372237353700400440124020402440"
          "2740324041405040704002410741114113412241304135414341514155410142"
          "0342104215422142334240425742624270420443114313432043224331433543"
          "0044024424443744404471440545074521456245134634466046104715473047"
          "4347514702501050145022504050445047505250665074500151035105511251"
          "2151325172510052115223523052365253520253075310532753445351536553"
          "7353015404542054325446541255265551555355425602570457225711601360"
          "1560316033606060006120612761646112623462426255626262706200631463"
          "2163406325644364626400650365346560650566406611671367007004700770"
          "2070227036704070547062700271117124714371457101720472107216722172"
          "3072517202733273357353730174057413742074507422754275027631760077";
      auto hex = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        return static_cast<uint8_t>(c - 'a' + 10);
      };
      for (int i = 0; i < 512; ++i) {
        const uint8_t b0 = static_cast<uint8_t>((hex(kHex[4 * i + 0]) << 4) | hex(kHex[4 * i + 1]));
        const uint8_t b1 = static_cast<uint8_t>((hex(kHex[4 * i + 2]) << 4) | hex(kHex[4 * i + 3]));
        const uint8_t g0 = kMap[b0 & 7];
        const uint8_t g1 = kMap[(b0 >> 4) & 7];
        const uint8_t g2 = kMap[b1 & 7];
        const uint8_t g3 = kMap[(b1 >> 4) & 7];
        tbl[i] = static_cast<uint32_t>(g0) | (static_cast<uint32_t>(g1) << 8) |
                 (static_cast<uint32_t>(g2) << 16) | (static_cast<uint32_t>(g3) << 24);
      }
      return tbl;
    }();
    static constexpr uint8_t kmask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const int64_t n_blocks = n_elems / 256;
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t* blk = data + static_cast<size_t>(b) * 110;
      uint16_t d16;
      std::memcpy(&d16, blk, 2);
      const float d = r4dx::core::F16ToFloat(d16);
      const uint8_t* qs = blk + 2;
      const uint8_t* qh = blk + 66;
      const uint8_t* signs = blk + 74;
      const uint8_t* scales = blk + 106;
      float* y = out + b * 256;
      for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * static_cast<float>(1 + 2 * (scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
          const uint32_t g1 = grid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)];
          const uint32_t g2 = grid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)];
          for (int j = 0; j < 4; ++j) {
            const float v1 = static_cast<float>((g1 >> (8 * j)) & 0xff);
            const float v2 = static_cast<float>((g2 >> (8 * j)) & 0xff);
            y[j + 0] = db1 * v1 * ((signs[l] & kmask[j + 0]) ? -1.0f : 1.0f);
            y[j + 4] = db1 * v2 * ((signs[l] & kmask[j + 4]) ? -1.0f : 1.0f);
          }
          y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; ++l) {
          const uint32_t g1 = grid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)];
          const uint32_t g2 = grid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)];
          for (int j = 0; j < 4; ++j) {
            const float v1 = static_cast<float>((g1 >> (8 * j)) & 0xff);
            const float v2 = static_cast<float>((g2 >> (8 * j)) & 0xff);
            y[j + 0] = db2 * v1 * ((signs[l] & kmask[j + 0]) ? -1.0f : 1.0f);
            y[j + 4] = db2 * v2 * ((signs[l] & kmask[j + 4]) ? -1.0f : 1.0f);
          }
          y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
      }
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
