#include "vision_weights.h"

#include <cmath>
#include <stdexcept>

namespace r4dx::vision {

namespace {

using r4dx_convert::SafetensorsReader;

// The converter writes `vision.*` as raw bytes with a trailing 2-wide element axis (dtype U8,
// shape [..., 2]), the same passthrough form every bare bf16 tensor in a container has -- so the
// element count comes from the byte span, not from the declared shape. Mirrors
// src/model/container.cpp's ElemCountBySize, which does this for the text side's bare tensors.
int64_t ElemCountBySize(const SafetensorsReader& r, const std::string& name) {
  if (!r.Has(name)) {
    throw std::runtime_error("r4dx::vision: container has no tensor '" + name + "'");
  }
  const auto& m = r.Meta(name);
  const uint64_t span = m.end - m.begin;
  if (span % 2 != 0) {
    throw std::runtime_error("r4dx::vision: tensor '" + name + "' byte span is odd (not bf16)");
  }
  return static_cast<int64_t>(span / 2);
}

core::DeviceBuffer<uint16_t> Upload(const SafetensorsReader& r, const std::string& name,
                                     int64_t expect_elems) {
  const int64_t n = ElemCountBySize(r, name);
  if (expect_elems > 0 && n != expect_elems) {
    throw std::runtime_error("r4dx::vision: tensor '" + name + "' has " + std::to_string(n) +
                              " bf16 elements, the vision_config implies " +
                              std::to_string(expect_elems));
  }
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint16_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

struct LoadCounter {
  int64_t tensors = 0;
  int64_t bytes = 0;
  void Add(const core::DeviceBuffer<uint16_t>& b) {
    ++tensors;
    bytes += static_cast<int64_t>(b.bytes());
  }
};

VisionLinear LoadLinear(const SafetensorsReader& r, const std::string& base, int64_t N, int64_t K,
                        LoadCounter* c) {
  VisionLinear l;
  l.N = N;
  l.K = K;
  l.weight = Upload(r, base + ".weight", N * K);
  l.bias = Upload(r, base + ".bias", N);
  c->Add(l.weight);
  c->Add(l.bias);
  return l;
}

VisionLayerNorm LoadLayerNorm(const SafetensorsReader& r, const std::string& base, int64_t hidden,
                              LoadCounter* c) {
  VisionLayerNorm n;
  n.weight = Upload(r, base + ".weight", hidden);
  n.bias = Upload(r, base + ".bias", hidden);
  c->Add(n.weight);
  c->Add(n.bias);
  return n;
}

}  // namespace

int64_t VisionConfig::NumGridPerSide() const {
  const int64_t side = static_cast<int64_t>(std::llround(std::sqrt(
      static_cast<double>(num_position_embeddings))));
  if (side * side != num_position_embeddings) {
    throw std::runtime_error(
        "r4dx::vision: num_position_embeddings (" + std::to_string(num_position_embeddings) +
        ") is not a perfect square -- the learned position grid is square by construction "
        "(Qwen3_5VisionModel.num_grid_per_side = int(num_position_embeddings ** 0.5))");
  }
  return side;
}

VisionConfig VisionConfig::FromJson(const nlohmann::json& vision_cfg) {
  VisionConfig c;
  auto geti = [&](const char* key, int64_t* dst) {
    if (vision_cfg.contains(key)) *dst = vision_cfg.at(key).get<int64_t>();
  };
  geti("depth", &c.depth);
  geti("hidden_size", &c.hidden_size);
  geti("num_heads", &c.num_heads);
  geti("intermediate_size", &c.intermediate_size);
  geti("in_channels", &c.in_channels);
  geti("patch_size", &c.patch_size);
  geti("temporal_patch_size", &c.temporal_patch_size);
  geti("spatial_merge_size", &c.spatial_merge_size);
  geti("out_hidden_size", &c.out_hidden_size);
  geti("num_position_embeddings", &c.num_position_embeddings);
  if (vision_cfg.contains("hidden_act")) c.hidden_act = vision_cfg.at("hidden_act").get<std::string>();
  // rope_theta lives under "rope_parameters" in a full HF vision_config; this checkpoint's
  // container metadata carries neither, so the Qwen3_5VisionConfig default (1e4) stands.
  if (vision_cfg.contains("rope_parameters")) {
    const auto& rp = vision_cfg.at("rope_parameters");
    if (rp.contains("rope_theta")) c.rope_theta = rp.at("rope_theta").get<double>();
  } else if (vision_cfg.contains("rope_theta")) {
    c.rope_theta = vision_cfg.at("rope_theta").get<double>();
  }

  if (c.num_heads <= 0 || c.hidden_size % c.num_heads != 0) {
    throw std::runtime_error("r4dx::vision: hidden_size is not divisible by num_heads");
  }
  // r4d_attn_vit_h72_bf16 is compiled for head_dim 72 only (r4d_attn_vit_dims); a container whose
  // vision tower has a different head size would return -1 at the first launch, which is a much
  // worse place to discover it than here.
  if (c.HeadDim() != 72) {
    throw std::runtime_error(
        "r4dx::vision: head_dim is " + std::to_string(c.HeadDim()) +
        ", but r4d_attn_vit_h72_bf16 is compiled for 72 only (third_party/libr4d, see "
        "r4d_attn_vit_dims)");
  }
  if (c.hidden_act != "gelu_pytorch_tanh") {
    throw std::runtime_error(
        "r4dx::vision: vision_config.hidden_act is '" + c.hidden_act +
        "', but the encoder MLP path implements gelu_pytorch_tanh (docs/vision.md)");
  }
  return c;
}

bool HasVisionTensors(const SafetensorsReader& reader) {
  return reader.Has("vision.patch_embed.proj.weight");
}

VisionWeights LoadVisionWeights(const SafetensorsReader& reader, const nlohmann::json& vision_cfg) {
  VisionWeights w;
  w.config = VisionConfig::FromJson(vision_cfg);
  const VisionConfig& c = w.config;
  LoadCounter counter;

  // patch_embed.proj: the checkpoint's Conv3d weight [hidden, in_ch, t, p, p] read as the
  // [hidden, 1536] matrix those same bytes already are (docs/vision.md "Patch embedding").
  w.patch_embed.N = c.hidden_size;
  w.patch_embed.K = c.PatchDim();
  w.patch_embed.weight =
      Upload(reader, "vision.patch_embed.proj.weight", c.hidden_size * c.PatchDim());
  w.patch_embed.bias = Upload(reader, "vision.patch_embed.proj.bias", c.hidden_size);
  counter.Add(w.patch_embed.weight);
  counter.Add(w.patch_embed.bias);

  w.pos_embed_table =
      Upload(reader, "vision.pos_embed.weight", c.num_position_embeddings * c.hidden_size);
  counter.Add(w.pos_embed_table);

  w.blocks.resize(static_cast<size_t>(c.depth));
  for (int64_t i = 0; i < c.depth; ++i) {
    const std::string base = "vision.blocks." + std::to_string(i) + ".";
    VisionBlockWeights& b = w.blocks[static_cast<size_t>(i)];
    b.norm1 = LoadLayerNorm(reader, base + "norm1", c.hidden_size, &counter);
    b.norm2 = LoadLayerNorm(reader, base + "norm2", c.hidden_size, &counter);
    b.qkv = LoadLinear(reader, base + "attn.qkv", 3 * c.hidden_size, c.hidden_size, &counter);
    b.proj = LoadLinear(reader, base + "attn.proj", c.hidden_size, c.hidden_size, &counter);
    b.fc1 = LoadLinear(reader, base + "mlp.linear_fc1", c.intermediate_size, c.hidden_size,
                       &counter);
    b.fc2 = LoadLinear(reader, base + "mlp.linear_fc2", c.hidden_size, c.intermediate_size,
                       &counter);
  }

  // The merger's norm is over hidden_size, NOT over the post-view merger width: the reference
  // constructs it as `nn.LayerNorm(config.hidden_size)` because `use_postshuffle_norm` is False
  // (Qwen3_5VisionModel.__init__), and only then reshapes to [-1, 4608].
  w.merger_norm = LoadLayerNorm(reader, "vision.merger.norm", c.hidden_size, &counter);
  w.merger_fc1 = LoadLinear(reader, "vision.merger.linear_fc1", c.MergerHidden(), c.MergerHidden(),
                            &counter);
  w.merger_fc2 = LoadLinear(reader, "vision.merger.linear_fc2", c.out_hidden_size,
                            c.MergerHidden(), &counter);

  w.tensor_count = counter.tensors;
  w.bytes = counter.bytes;
  return w;
}

}  // namespace r4dx::vision
