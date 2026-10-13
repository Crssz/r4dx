// r4dx-convert's virtual AWQ source (docs/qwen4exp-plan.md 2.5, 4.2, 4.11 item 3; task M1-2).
//
// A compressed-tensors `pack-quantized` checkpoint stores every quantized linear `<b>` as four tensors
//
//   <b>.weight_packed       int32 [N, K/8]        8 unsigned nibbles per int32 along K (nibble j of word c = column 8c + j)
//   <b>.weight_scale        bf16  [N, K/G]
//   <b>.weight_zero_point   int32 [ceil(N/8), K/G] 8 unsigned nibbles per int32 along N (nibble n % 8 of row n / 8)
//   <b>.weight_shape        int64 [2] = (N, K)
//
// and the rest of r4dx-convert reads only BF16/F32 tensors named `<b>.weight`. This overlay (installed on
// the ShardedModel with SetOverlay) makes each quad appear as ONE BF16 tensor `<b>.weight` of shape
// [N, K], dequantized on demand and never materialised as a whole checkpoint:
//
//   W[n, k] = bf16_rne( float(u_q[n, k] - u_zp[n, g]) * float(scale[n, g]) ),   g = k / G
//
// -- exactly tools/reference/qwen4exp/awq_dequant.py (the product of a 4-bit integer and a bf16 scale is
// exact in fp32, so this is the one rounding compressed-tensors' bf16 arithmetic performs). The four
// component names disappear from ShardedModel::AllNames() (so a coverage audit never sees them as
// unconsumed); they stay readable through RawMeta/RawData.
//
// Memory and threads: Data() dequantizes the whole tensor into a buffer private to the CALLING THREAD
// and reused by that thread's next Data()/DataRows() of an overlay tensor, so the footprint is one
// tensor per worker thread (52 MB for the largest, in_proj_qkv) whatever the checkpoint's size, and
// concurrent readers never share a buffer. Nothing else is mutable after Create(). Create() opens every
// shard the quads live in (ShardedModel opens shards lazily and is not thread-safe for that), reads
// every weight_shape and cross-checks all four tensors of every linear, so a malformed checkpoint
// fails before anything is planned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx_convert::awq {

struct PackScheme {
  int num_bits = 4;
  int group_size = 128;
};

// config.json's quantization_config: false when it is absent, null, or its `format` is not
// "pack-quantized" (another format is another converter's business; the qwen4_exp branch refuses it
// itself). When the format IS pack-quantized, anything but int4, group 128, strategy group, asymmetric,
// no activation quantization, no actorder, static weights throws std::runtime_error naming the field.
bool PackQuantizedScheme(const nlohmann::json& config, PackScheme* out);

// Rows [row0, row0 + nrows) of the dequantized bf16 [N, K] weight into out[nrows * K]. `packed` is the
// int32 [N, K/8] tensor's bytes, `scale` the bf16 [N, K/group] one's, `zp` the int32 [ceil(N/8),
// K/group] one's (little-endian, any alignment: they are mapped straight out of the shards);
// K % group == 0 and group % 8 == 0. Pure function (no allocation), the unit the bit-equality test pins.
void DequantRows(const uint8_t* packed, const uint8_t* scale, const uint8_t* zp, int64_t N, int64_t K,
                 int group, int64_t row0, int64_t nrows, uint16_t* out);

class AwqOverlay final : public TensorOverlay {
 public:
  // Scans model.AllNames() for `<b>.weight_packed`; every such `<b>` must have its three companions
  // with the dtypes and shapes above, a weight_shape equal to (N, K), N % 8-padded zero points, and no
  // `<b>.weight` of its own. Throws std::runtime_error naming the tensor otherwise. Installs nothing.
  static std::shared_ptr<AwqOverlay> Create(ShardedModel& model, const PackScheme& scheme);

  size_t LinearCount() const { return linears_.size(); }

  bool Provides(const std::string& name) const override;
  const TensorMeta& Meta(const std::string& name) const override;
  const uint8_t* Data(const std::string& name) const override;
  const uint8_t* DataRows(const std::string& name, int64_t row0, int64_t nrows,
                          std::vector<uint8_t>& scratch) const override;
  void RewriteNames(std::vector<std::string>& names) const override;

 private:
  struct Linear {
    TensorMeta meta;  // BF16 [N, K], begin 0, end N * K * 2
    int64_t N = 0, K = 0;
  };
  AwqOverlay(ShardedModel& model, const PackScheme& scheme) : model_(model), scheme_(scheme) {}

  const Linear& Find(const std::string& name, std::string* base) const;
  void Dequant(const std::string& base, const Linear& l, int64_t row0, int64_t nrows, uint8_t* out) const;

  ShardedModel& model_;
  PackScheme scheme_;
  std::unordered_map<std::string, Linear> linears_;  // key: `<b>.weight`
};

// config -> overlay: parses the scheme, and when the checkpoint is pack-quantized creates the overlay
// and installs it on `model`; returns the number of virtual linears (0 = not pack-quantized, nothing
// installed).
size_t InstallAwqSource(ShardedModel& model, const nlohmann::json& config);

}  // namespace r4dx_convert::awq
