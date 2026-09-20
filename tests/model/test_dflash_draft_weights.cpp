// r4dx::model::DflashDraftWeights CPU-only test (task A1 item 3): builds a tiny dflash2_draft
// container via the SAME r4dx_convert writer helpers src/convert/main.cpp's --dflash-gguf mode
// uses, from tests/convert/fixtures/dflash_mini.gguf, then opens it with DflashDraftWeights and
// checks Config()/HasTensor()/TensorMeta() -- no HIP anywhere in this binary (this target links
// only r4dx_convert + r4dx_core, see tests/model/CMakeLists.txt).
#include <cstdio>
#include <cstdlib>
#include <string>

#include "dflash_draft_weights.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/dflash2_container.hpp"
#include "r4dx_convert/gguf_reader.hpp"

namespace {
bool g_ok = true;
bool Check(const char* label, bool cond) {
  std::printf("%-56s %s\n", label, cond ? "OK" : "FAIL");
  if (!cond) g_ok = false;
  return cond;
}
}  // namespace

int main() {
  using namespace r4dx_convert;

#ifndef R4DX_CONVERT_FIXTURES_DIR
#error "R4DX_CONVERT_FIXTURES_DIR must be defined by CMake"
#endif
  const std::string gguf_path = std::string(R4DX_CONVERT_FIXTURES_DIR) + "\\dflash_mini.gguf";
  GgufReader gguf(Utf8ToWide(gguf_path));
  Dflash2Metadata meta = ReadDflash2Metadata(gguf);

  LayoutSet layouts;
  layouts.bf16 = true;

  const std::string out_path = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") +
                                "\\r4dx_test_dflash_draft_weights.r4dx";
  {
    ContainerWriter writer;
    PlanDflash2Linear(writer, gguf, {"blk.0.attn_q.weight", "dflash.layers.0.self_attn.q_proj"}, layouts);
    PlanDflash2F32(writer, gguf, "blk.0.attn_norm.weight", "dflash.layers.0.input_layernorm");

    nlohmann::json metadata;
    metadata["r4dx_format_version"] = "1";
    metadata["container_kind"] = "dflash2_draft";
    metadata["dflash2"] = BuildDflash2MetadataJson(meta);
    metadata["dflash2"]["layout"] = "bf16";
    writer.FinalizeHeader(out_path, metadata);

    EmitDflash2Linear(writer, gguf, {"blk.0.attn_q.weight", "dflash.layers.0.self_attn.q_proj"}, layouts, 1);
    EmitDflash2F32(writer, gguf, "blk.0.attn_norm.weight", "dflash.layers.0.input_layernorm");
    writer.Finish();
  }

  auto w = r4dx::model::DflashDraftWeights::Open(out_path);
  const auto& cfg = w.Config();
  Check("hidden_size", cfg.hidden_size == 64);
  Check("block_count", cfg.block_count == 2);
  Check("feed_forward_length", cfg.feed_forward_length == 128);
  Check("attention.head_count", cfg.attention.head_count == 4);
  Check("attention.head_count_kv", cfg.attention.head_count_kv == 2);
  Check("attention.key_length", cfg.attention.key_length == 16);
  Check("attention.causal false", cfg.attention.causal == false);
  Check("attention.sliding_window", cfg.attention.sliding_window == 64);
  Check("attention.sliding_window_pattern size", cfg.attention.sliding_window_pattern.size() == 2);
  Check("rope.n_rot", cfg.rope.n_rot == 16);
  Check("rope.pairing", cfg.rope.pairing == "neox_split_half");
  Check("selector_rank", cfg.selector_rank == 32);
  Check("selector_top_k", cfg.selector_top_k == 4);
  Check("target_layers", cfg.target_layers == std::vector<int64_t>({1, 2}));
  Check("mask_token_id", cfg.mask_token_id == 500);
  Check("vocab_size", cfg.vocab_size == 512);
  Check("layout", cfg.layout == "bf16");

  Check("HasTensor q_proj.bf16.w", w.HasTensor("dflash.layers.0.self_attn.q_proj.bf16.w"));
  Check("HasTensor input_layernorm", w.HasTensor("dflash.layers.0.input_layernorm"));
  Check("!HasTensor bogus", !w.HasTensor("dflash.layers.0.does_not_exist"));
  Check("LayerTensor helper", r4dx::model::DflashDraftWeights::LayerTensor(3, "mlp.down_proj") ==
                                    "dflash.layers.3.mlp.down_proj");

  const auto& m = w.TensorMeta("dflash.layers.0.self_attn.q_proj.bf16.w");
  Check("q_proj.bf16.w shape", m.shape == std::vector<int64_t>({64, 64, 2}));

  Check("Open() rejects a non-dflash2_draft container", [&] {
    // Build a plain container with no container_kind (or a mismatched one) and confirm Open()
    // throws rather than silently misreading it as a draft container.
    const std::string bad_path = out_path + ".bad";
    {
      ContainerWriter writer2;
      writer2.Plan("x", {1}, 1);
      nlohmann::json md;
      md["r4dx_format_version"] = "1";
      writer2.FinalizeHeader(bad_path, md);
      uint8_t z = 0;
      writer2.WriteTensor("x", &z, 1);
      writer2.Finish();
    }
    bool threw = false;
    try {
      (void)r4dx::model::DflashDraftWeights::Open(bad_path);
    } catch (const std::exception&) {
      threw = true;
    }
    std::remove(bad_path.c_str());
    return threw;
  }());

  std::remove(out_path.c_str());
  if (!g_ok) return 1;
  std::printf("PASS\n");
  return 0;
}
