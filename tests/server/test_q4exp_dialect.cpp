// tests/server/test_q4exp_dialect.cpp -- the container-header half of the qwen4_exp dialect wiring
// (docs/qwen4exp.md): a synthetic safetensors shell whose `__metadata__.model_arch` is "qwen4_exp" must
// resolve to the qwen4_exp dialect with no flag, a Gemma request against it must be refused, and an explicit
// request must still win. CPU only (r4dx_model_arch + r4dx_server_core); the pure-dialect checks are in
// test_dialect.cpp.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "arch.h"
#include "dialect.h"
#include "nlohmann/json.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

using namespace r4dx::server;
namespace fs = std::filesystem;
using r4dx::model::Arch;

// A safetensors shell: u64 LE header length, then the JSON header holding only `__metadata__` (no tensors).
std::string WriteShell(const fs::path& path, const nlohmann::json& metadata) {
  const std::string header = nlohmann::json{{"__metadata__", metadata}}.dump();
  const uint64_t len = header.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&len), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  return path.string();
}

void WriteFile(const fs::path& path, const std::string& body) { std::ofstream(path, std::ios::binary) << body; }

DialectKind Resolve(const std::string& container, std::optional<DialectKind> requested, const std::string& tok_dir) {
  return ResolveDialectKindForArch(requested, tok_dir, r4dx::model::ArchName(r4dx::model::DetectArch(container)));
}

}  // namespace

int main() {
  // arch name <-> dialect name round trip
  CHECK(std::string(r4dx::model::ArchName(Arch::kQwen4Exp)) == "qwen4_exp");
  CHECK(r4dx::model::DetectArchFromMetadata({{"model_arch", "qwen4_exp"}}) == Arch::kQwen4Exp);
  DialectKind parsed;
  CHECK(ParseDialectName(r4dx::model::ArchName(Arch::kQwen4Exp), &parsed) && parsed == DialectKind::kQwen4Exp);
  CHECK(std::string(Qwen4ExpDialect().name) == r4dx::model::ArchName(Arch::kQwen4Exp));

  const fs::path root = fs::temp_directory_path() / "r4dx_test_q4exp_dialect";
  fs::remove_all(root);
  fs::create_directories(root / "qwen_tok");
  fs::create_directories(root / "gemma_tok");
  WriteFile(root / "qwen_tok" / "tokenizer_config.json", R"({"tokenizer_class":"Qwen2Tokenizer"})");
  WriteFile(root / "qwen_tok" / "tokenizer.json", "{}");
  WriteFile(root / "gemma_tok" / "tokenizer_config.json", R"({"tokenizer_class":"GemmaTokenizer"})");
  WriteFile(root / "gemma_tok" / "tokenizer.json", "{}");
  const std::string qwen_tok = (root / "qwen_tok").string();
  const std::string gemma_tok = (root / "gemma_tok").string();

  const std::string q4 = WriteShell(root / "q4exp.r4dx", {{"model_arch", "qwen4_exp"}, {"model_id", "swift/q4exp-test"}});
  const std::string q35 = WriteShell(root / "q35.r4dx", {{"model_arch", "qwen3_5"}});
  const std::string gem = WriteShell(root / "gemma.r4dx", {{"model_arch", "gemma4_unified"}});
  // model_config alone (no model_arch) also names qwen4_exp
  const std::string q4_cfg_only =
      WriteShell(root / "q4cfg.r4dx", {{"model_config", {{"model_type", "qwen4_exp"}}}});

  CHECK(r4dx::model::DetectArch(q4) == Arch::kQwen4Exp);
  CHECK(r4dx::model::DetectArch(q4_cfg_only) == Arch::kQwen4Exp);

  // no flag, no tokenizer dir: the container header alone picks the dialect
  CHECK(Resolve(q4, std::nullopt, "") == DialectKind::kQwen4Exp);
  CHECK(Resolve(q4_cfg_only, std::nullopt, "") == DialectKind::kQwen4Exp);
  CHECK(Resolve(q35, std::nullopt, "") == DialectKind::kQwen35);
  CHECK(Resolve(gem, std::nullopt, "") == DialectKind::kGemma4);
  // a Qwen tokenizer dir does not change a qwen4_exp container's dialect
  CHECK(Resolve(q4, std::nullopt, qwen_tok) == DialectKind::kQwen4Exp);
  CHECK(Resolve(q35, std::nullopt, qwen_tok) == DialectKind::kQwen35);
  // a Gemma tokenizer on a qwen4_exp container resolves to gemma4 ... and the arch check refuses that
  CHECK(Resolve(q4, std::nullopt, gemma_tok) == DialectKind::kGemma4);
  const std::string arch = r4dx::model::ArchName(r4dx::model::DetectArchFromMetadata(r4dx::model::ReadContainerMetadata(q4)));
  CHECK(arch == "qwen4_exp");
  CHECK(!CheckDialectAgainstArch(DialectFor(Resolve(q4, std::nullopt, gemma_tok)), arch).empty());
  CHECK(!CheckDialectAgainstArch(Gemma4Dialect(), arch).empty());
  // the auto-resolved dialect passes its own container's check
  CHECK(CheckDialectAgainstArch(DialectFor(Resolve(q4, std::nullopt, "")), arch).empty());
  // an explicit --dialect wins over the container and the tokenizer dir
  CHECK(Resolve(q4, DialectKind::kQwen35, "") == DialectKind::kQwen35);
  CHECK(Resolve(q35, DialectKind::kQwen4Exp, qwen_tok) == DialectKind::kQwen4Exp);
  CHECK(Resolve(q4, DialectKind::kGemma4, "") == DialectKind::kGemma4);

  // --max-ctx: qwen4_exp keeps the caller's value (the file is not read)
  CHECK(r4dx::model::ResolveContainerMaxCtx(q4, Arch::kQwen4Exp, /*requested_given=*/false, 262144,
                                            /*extended_ctx=*/false) == 262144);
  CHECK(r4dx::model::ResolveContainerMaxCtx(q4, Arch::kQwen4Exp, /*requested_given=*/true, 32768,
                                            /*extended_ctx=*/false) == 32768);

  fs::remove_all(root);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("test_q4exp_dialect: OK");
  return 0;
}
