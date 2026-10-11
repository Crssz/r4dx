// r4dx_convert::SafetensorsReader hardening against malformed shards (src/convert/include/r4dx_convert/
// safetensors_reader.hpp): a header length or a tensor data_offsets end near 2^64 must be refused with a
// std::runtime_error (the old `8 + header_len > file_size` / `data_start + end > file_size` checks wrapped
// around and let the read run off the mapping), a negative shape dimension must be refused (two negative
// dimensions have a positive element count that would otherwise match the byte span), and a constructor that
// throws must release its file / mapping handles (the file is deleted right afterwards: Windows refuses to
// delete a file that is still open). The files are written by hand (u64 LE header length + JSON header +
// data) into a fresh temp directory, so the test needs no fixture. CPU-only, no HIP.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/safetensors_reader.hpp"

namespace fs = std::filesystem;

namespace {

bool g_ok = true;

bool Check(const std::string& label, bool cond) {
  std::printf("%-72s %s\n", label.c_str(), cond ? "OK" : "FAIL");
  if (!cond) g_ok = false;
  return cond;
}

// Writes `header_len_field` as the 8-byte little-endian prefix, then `header` verbatim, then `data`.
void WriteRaw(const fs::path& path, uint64_t header_len_field, const std::string& header, const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&header_len_field), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

// A well-formed prefix: the header length field is the header's real size.
void WriteFile(const fs::path& path, const std::string& header, const std::vector<uint8_t>& data) {
  WriteRaw(path, header.size(), header, data);
}

// Opens `path` through the reader. Returns true when it threw std::runtime_error whose message contains `substr`
// (any message when empty). Any other exception type escapes (and fails the test run).
bool OpenThrows(const fs::path& path, const std::string& substr, std::string* what = nullptr) {
  try {
    r4dx_convert::SafetensorsReader r(path.wstring());
  } catch (const std::runtime_error& e) {
    if (what) *what = e.what();
    return substr.empty() || std::string(e.what()).find(substr) != std::string::npos;
  }
  if (what) *what = "(no exception)";
  return false;
}

// The file can be deleted right after a throwing open only if the reader released every handle.
bool RemovableNow(const fs::path& path) {
  std::error_code ec;
  const bool removed = fs::remove(path, ec);
  return removed && !ec;
}

void CaseThrows(const fs::path& path, const std::string& label, const std::string& substr) {
  std::string what;
  const bool ok = OpenThrows(path, substr, &what);
  Check(label + " -> runtime_error" + (substr.empty() ? "" : " mentioning \"" + substr + "\""), ok);
  if (!ok) std::printf("    got: %s\n", what.c_str());
  Check(label + ": the failed open leaked no handle (file deletable)", RemovableNow(path));
}

}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("r4dx_safetensors_reader_" + std::to_string(::GetCurrentProcessId()));
  { std::error_code ec; fs::remove_all(root, ec); }
  fs::create_directories(root);
  const fs::path p = root / "t.safetensors";
  const std::string kOne = R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}})";
  const std::vector<uint8_t> k12(12, 0x5A);

  // ---- control: a well-formed file opens and reads back ----
  {
    WriteFile(p, kOne, k12);
    bool ok = false;
    try {
      r4dx_convert::SafetensorsReader r(p.wstring());
      ok = r.Has("w") && r.Meta("w").shape == std::vector<int64_t>{2, 3} && r.Meta("w").end == 12 && r.Data("w")[0] == 0x5A;
    } catch (const std::exception& e) {
      std::printf("    unexpected: %s\n", e.what());
    }
    Check("control: a well-formed shard opens and reads", ok);
    Check("control: ... and is deletable once closed", RemovableNow(p));
  }

  // ---- header length near 2^64 (8 + header_len wrapped to a small number) ----
  WriteRaw(p, UINT64_MAX, "{}", {});
  CaseThrows(p, "header_len = 2^64 - 1", "header length overruns file");
  WriteRaw(p, UINT64_MAX - 7, "{}", {});  // 8 + header_len == 1 (mod 2^64)
  CaseThrows(p, "header_len = 2^64 - 7 (8 + len wraps to 1)", "header length overruns file");
  WriteRaw(p, 0x8000000000000000ull, kOne, k12);
  CaseThrows(p, "header_len = 2^63", "header length overruns file");
  WriteRaw(p, kOne.size() + 1, kOne, {});  // one byte longer than the file has: still refused (not near 2^64)
  CaseThrows(p, "header_len = real size + 1 (no data after it)", "header length overruns file");

  // ---- tensor data_offsets end near 2^64 (data_start + end wrapped back inside the file) ----
  {
    const std::string h = R"({"w":{"dtype":"U8","shape":[16],"data_offsets":[0,18446744073709551615]}})";
    WriteFile(p, h, std::vector<uint8_t>(16, 1));
    CaseThrows(p, "data_offsets end = 2^64 - 1", "truncated");
  }
  {
    // data_start + end == data_start - 8 (mod 2^64): the wrapped sum is below the file size, the real one is not.
    const uint64_t end = UINT64_MAX - 7;  // 2^64 - 8
    const std::string h = std::string(R"({"w":{"dtype":"U8","shape":[16],"data_offsets":[0,)") + std::to_string(end) + "]}}";
    WriteFile(p, h, std::vector<uint8_t>(16, 1));
    CaseThrows(p, "data_offsets end = 2^64 - 8 (data_start + end wraps)", "truncated");
  }
  {
    const std::string h = R"({"w":{"dtype":"U8","shape":[8],"data_offsets":[18446744073709551608,18446744073709551615]}})";
    WriteFile(p, h, std::vector<uint8_t>(16, 1));
    CaseThrows(p, "data_offsets begin/end both near 2^64", "truncated");
  }
  {
    const std::string h = R"({"w":{"dtype":"U8","shape":[4],"data_offsets":[8,4]}})";
    WriteFile(p, h, std::vector<uint8_t>(16, 1));
    CaseThrows(p, "data_offsets end < begin", "end < begin");
  }

  // ---- negative shape dimension ----
  {
    // [-2, -3] has element count 6 -> 12 bytes of BF16, which equals the span: only the explicit check refuses it.
    const std::string h = R"({"w":{"dtype":"BF16","shape":[-2,-3],"data_offsets":[0,12]}})";
    WriteFile(p, h, k12);
    CaseThrows(p, "shape [-2, -3] (positive product, matching span)", "negative shape dimension");
  }
  {
    const std::string h = R"({"w":{"dtype":"BF16","shape":[3,-1],"data_offsets":[0,0]}})";
    WriteFile(p, h, {});
    CaseThrows(p, "shape [3, -1]", "negative shape dimension");
  }

  // ---- header_len 0 / file shorter than the prefix ----
  {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    const char few[4] = {1, 2, 3, 4};
    f.write(few, 4);
  }
  CaseThrows(p, "file shorter than the 8-byte prefix", "too small");

  { std::error_code ec; fs::remove_all(root, ec); }
  std::printf(g_ok ? "test_safetensors_reader: PASS\n" : "test_safetensors_reader: FAIL\n");
  return g_ok ? 0 : 1;
}
