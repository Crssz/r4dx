// r4dx::model::DflashDraftWeights CPU-only test (task A1 item 3): builds a tiny dflash2_draft
// container via the SAME r4dx_convert writer helpers src/convert/main.cpp's --dflash-gguf mode
// uses, from tests/convert/fixtures/dflash_mini.gguf, then opens it with DflashDraftWeights and
// checks Config()/HasTensor()/TensorMeta() -- no HIP CALL anywhere in this binary, so it needs no
// device (this target links only r4dx_convert + r4dx_core, see tests/model/CMakeLists.txt; since
// Milestone 11 the header does reach r4d_core for r4d_gemm_w4a16_nt_m64_group(), a host function).
//
// It also gates the scope of that group guard: a dflash2 container records a w4a16 group whether
// or not it holds a single w4a16 tensor, so Open() must refuse a mismatch only for the ones that
// do -- see the last block of main().
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

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

  // ---- the w4a16 group guard, and its scope (Milestone 11 + adversarial-review fix) ------------
  // r4dx-convert writes __metadata__.quant unconditionally, so EVERY dflash2 container records a
  // w4a16 group -- including one whose linears are bf16 or mxfp4 and which therefore contains no
  // .w4a16.* tensor at all. DflashDraftWeights::Open must refuse a group mismatch only when the
  // container actually carries w4a16 bytes; refusing the others rejects containers that this build
  // can read byte-for-byte correctly (qwen38-27b-dflash2-bf16.r4dx was one).
  {
    const int kernel_group = r4d_gemm_w4a16_nt_m64_group();
    const int wrong_group = (kernel_group == 128) ? 64 : 128;

    auto write_draft_container = [&](const std::string& path, int recorded_group,
                                      bool with_w4a16_tensor) {
      ContainerWriter writer2;
      writer2.Plan("dflash.layers.0.input_layernorm", {64, 4}, 64 * 4);
      if (with_w4a16_tensor) writer2.Plan("dflash.fc.w4a16.wq", {32}, 32);
      nlohmann::json md;
      md["r4dx_format_version"] = "1";
      md["container_kind"] = "dflash2_draft";
      md["dflash2"] = BuildDflash2MetadataJson(meta);
      md["dflash2"]["layout"] = with_w4a16_tensor ? "w4a16" : "bf16";
      md["quant"] = {{"w4a16", {{"group", recorded_group}}}};
      writer2.FinalizeHeader(path, md);
      std::vector<uint8_t> zeros(64 * 4, 0);
      writer2.WriteTensor("dflash.layers.0.input_layernorm", zeros.data(), 64 * 4);
      if (with_w4a16_tensor) writer2.WriteTensor("dflash.fc.w4a16.wq", zeros.data(), 32);
      writer2.Finish();
    };
    auto opens = [](const std::string& path) {
      try {
        (void)r4dx::model::DflashDraftWeights::Open(path);
        return std::string();
      } catch (const std::exception& e) {
        return std::string(e.what());
      }
    };

    const std::string p_bf16 = out_path + ".g_bf16";
    const std::string p_w4a16_bad = out_path + ".g_w4a16_bad";
    const std::string p_w4a16_ok = out_path + ".g_w4a16_ok";
    write_draft_container(p_bf16, wrong_group, /*with_w4a16_tensor=*/false);
    write_draft_container(p_w4a16_bad, wrong_group, /*with_w4a16_tensor=*/true);
    write_draft_container(p_w4a16_ok, kernel_group, /*with_w4a16_tensor=*/true);

    const std::string e_bf16 = opens(p_bf16);
    const std::string e_bad = opens(p_w4a16_bad);
    const std::string e_ok = opens(p_w4a16_ok);
    Check("no-w4a16-tensor container opens despite a mismatched recorded group", e_bf16.empty());
    if (!e_bf16.empty()) std::printf("  unexpected: %s\n", e_bf16.c_str());
    Check("w4a16 container with a mismatched group is refused",
          e_bad.find("was packed with w4a16 group=") != std::string::npos &&
              e_bad.find(std::to_string(wrong_group)) != std::string::npos &&
              e_bad.find(std::to_string(kernel_group)) != std::string::npos);
    Check("w4a16 container at this build's own group opens", e_ok.empty());
    if (!e_ok.empty()) std::printf("  unexpected: %s\n", e_ok.c_str());
    std::printf("  kernel w4a16 group=%d, mismatched group used in the test=%d\n", kernel_group,
                wrong_group);
    std::remove(p_bf16.c_str());
    std::remove(p_w4a16_bad.c_str());
    std::remove(p_w4a16_ok.c_str());
  }

  std::remove(out_path.c_str());
  if (!g_ok) return 1;
  std::printf("PASS\n");
  return 0;
}
