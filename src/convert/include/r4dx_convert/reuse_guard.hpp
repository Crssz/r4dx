// r4dx_convert reuse guard -- the library half of r4dx-convert's --record-reuse-guard and
// --reuse-tensors-from <baseline.r4dx> (docs/quant2.md 5.2; src/convert/main.cpp has the per-tensor
// decision and the header comment).
//
// A Q3 sweep converts ~40 containers that differ from one baseline only in ONE candidate: a
// --w4a16-group-rule, or a --keep-bf16 that keeps more (or fewer) linears in bf16. Every linear is
// quantized independently (LDLQ: its own Hessian, no sequential dependency), so such a candidate
// differs from the baseline only in the linears whose RESOLVED layout set (LayoutSetId: the w4a16
// group, or bf16-only) differs. --reuse-tensors-from copies every other tensor verbatim from the
// baseline instead of recomputing it. That is sound only if the baseline was produced by THE SAME
// converter binary on the same CPU from THE SAME inputs with THE SAME flags except the per-linear
// layout choices, and only if its bytes are still the ones that conversion wrote. So a run that
// records a guard writes __metadata__.r4dx_convert_run.reuse_guard -- the identity of everything the
// bytes can depend on, the resolved layout set of every linear, and a digest of the bytes themselves
// -- and a reuse run refuses a baseline whose guard differs from its own in any identity field, or
// whose data no longer matches its digest, and recomputes exactly the linears whose recorded layout
// set differs from its own.
//
// This header holds the pieces that do not need main.cpp's AppArgs: file / checkpoint hashing, the
// running binary's and CPU's identity, the data digest, the guard comparison and the baseline's
// reader.
#pragma once

#if defined(_MSC_VER)
#include <intrin.h>  // __cpuid, __cpuidex, _xgetbv
#endif
#include <immintrin.h>

#include <math.h>  // _get_FMA3_enable (x64 UCRT)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/linear_layouts.hpp"      // LayoutSetId (reuse_guard.linears)
#include "r4dx_convert/safetensors_reader.hpp"  // SafetensorsReader, Utf8ToWide, WideToUtf8
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/threadpool.hpp"          // ParallelEach

namespace r4dx_convert {

// Bumped whenever the guard's fields or their meaning change: a baseline with another version is
// refused like any other guard mismatch. 2: converter.cpu, inputs.hessian_files, data_sha256.
// 3: args.keep_bf16 (the regex text) is gone; `linears` records every linear's resolved layout set
// and a reuse recomputes the linears whose set differs, so --keep-bf16 may differ like the group rules.
constexpr int kReuseGuardVersion = 3;

// reuse_guard.linears: {"<container base>": LayoutSetId(its resolved LayoutSet)} for every linear the
// run writes through add_linear (and the MTP draft head) -- after --w4a16-group-rule and --keep-bf16,
// i.e. what the linear IS in this container, not the flags that made it so. NOT part of the identity
// ReuseGuardMismatches compares (it is the one thing a reuse may change): src/convert/main.cpp's
// PlanReuse compares it linear by linear, recomputes every linear whose set differs, and checks the
// baseline's tensors against the baseline's own record.
constexpr const char* kReuseLinearsKey = "linears";

// The two completion fields inside reuse_guard. FinalizeHeader writes them as placeholders (the
// header goes to disk before the emit pass); once writer.Finish() has confirmed that every planned
// tensor was written, the converter reads every tensor back from the file, records the digest
// (ContainerDataSha256) over data_sha256's placeholder and flips emit_complete from 0 to 1 -- both
// patched in place, each durably (ContainerWriter::PatchHeader syncs the data before and the patch
// after). ContainerWriter pre-sizes the file with zeros, so an interrupted or failed conversion
// leaves a valid-looking header over zero-filled tensors: a baseline whose marker is still 0 is
// refused, and so is one whose bytes do not hash to its data_sha256 (a partial copy, a later
// overwrite). Neither field is part of the identity ReuseGuardMismatches compares.
constexpr const char* kReuseCompleteKey = "emit_complete";
constexpr const char* kReuseDataKey = "data_sha256";
inline std::string ReuseCompleteNeedle(int v) {
  return std::string("\"") + kReuseCompleteKey + "\":" + std::to_string(v);
}
inline std::string ReuseDataPlaceholder() { return std::string(64, '0'); }
inline std::string ReuseDataNeedle(const std::string& hex) {
  return std::string("\"") + kReuseDataKey + "\":\"" + hex + "\"";
}
inline bool IsSha256Hex(const std::string& s) {
  if (s.size() != 64) return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

// The data digest of a container: sha256 over "<name>\n<sha256 of the tensor's bytes>\n" for every
// tensor, names in byte order. Per tensor, so the tensors hash in parallel and a reuse run can check
// each copied tensor on its own; by name, so it binds the tensor directory as well as the bytes (a
// header whose offsets were swapped reads other bytes under a name).
inline std::string ContainerDataSha256(std::vector<std::pair<std::string, std::string>> digests) {
  std::sort(digests.begin(), digests.end());
  Sha256 h;
  for (const auto& kv : digests) {
    const std::string line = kv.first + "\n" + kv.second + "\n";
    h.Update(reinterpret_cast<const uint8_t*>(line.data()), line.size());
  }
  return h.HexDigest();
}

// sha256 of a whole file, streamed (8 MiB at a time). `bytes_out`: the file's length.
inline std::string Sha256File(const std::string& path, uint64_t* bytes_out = nullptr) {
  // Wide open, like every other path open in this component (non-ASCII paths on an ANSI codepage).
  std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path + " to hash it");
  Sha256 h;
  std::vector<char> buf(static_cast<size_t>(8) << 20);
  uint64_t total = 0;
  for (;;) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = f.gcount();
    if (got > 0) {
      h.Update(reinterpret_cast<const uint8_t*>(buf.data()), static_cast<size_t>(got));
      total += static_cast<uint64_t>(got);
    }
    if (!f) break;
  }
  if (f.bad()) throw std::runtime_error("read error while hashing " + path);
  if (bytes_out) *bytes_out = total;
  return h.HexDigest();
}

struct FileDigest {
  uint64_t bytes = 0;
  std::string sha256;
};

// Sha256File of every path, `threads` files at a time, the largest first (the files are few and
// uneven: checkpoint shards of a few GB, Hessians from 50 MB to 0.6 GB). Result in `paths` order.
inline std::vector<FileDigest> Sha256Files(const std::vector<std::string>& paths, int threads) {
  std::vector<uint64_t> size(paths.size(), 0);
  for (size_t i = 0; i < paths.size(); ++i) {
    std::error_code ec;
    const auto s = std::filesystem::file_size(std::filesystem::u8path(paths[i]), ec);
    size[i] = ec ? 0 : static_cast<uint64_t>(s);  // a missing file fails in Sha256File, named
  }
  std::vector<size_t> order(paths.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return size[a] > size[b]; });
  std::vector<FileDigest> out(paths.size());
  ParallelEach(order.size(), threads, [&](size_t k) {
    const size_t i = order[k];
    out[i].sha256 = Sha256File(paths[i], &out[i].bytes);
  });
  return out;
}

// A loaded module's file path (GetModuleFileNameW), UTF-8; nullptr = the running executable.
inline std::string ModulePath(HMODULE module = nullptr) {
  std::wstring buf(1024, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0) throw std::runtime_error("GetModuleFileNameW failed");
    if (n < buf.size()) {
      buf.resize(n);
      return WideToUtf8(buf);
    }
    buf.resize(buf.size() * 2);
  }
}

inline std::string ExecutablePath() { return ModulePath(nullptr); }

// sha256 of the C/C++ runtime DLLs this process has loaded (the executable's hash cannot cover them):
// the rotation draws rotation.mix5 through ucrt's log/cos/sin, so a runtime update between the
// baseline and a reuse run could move its last bits -- and a copied rotation.mix5 next to linears
// folded with a recomputed one would be a container no full run writes. "not loaded" when absent.
inline nlohmann::json RuntimeModulesIdentity() {
  nlohmann::json j = nlohmann::json::object();
  for (const wchar_t* name : {L"ucrtbase.dll", L"msvcp140.dll", L"vcruntime140.dll"}) {
    const HMODULE h = GetModuleHandleW(name);
    j[WideToUtf8(name)] = h ? Sha256File(ModulePath(h)) : std::string("not loaded");
  }
  return j;
}

// The CPU, as far as it can choose a code path. The same ucrtbase.dll picks its transcendental
// implementation at run time -- the FMA3 one when the CPU (and the OS's saved AVX state) has it and
// _get_FMA3_enable() allows it -- and the two can differ in the last bit; dense_linalg.hpp picks
// AVX-512 or scalar the same way. So a baseline converted on one machine and a reuse run on another
// could fold rotation.mix5 (always copied) and the recomputed linears with different Q. Recorded
// whole (vendor, brand, signature, the path-selecting feature bits, XCR0): conservative, since a
// sweep runs on one machine anyway.
inline nlohmann::json CpuIdentity() {
  nlohmann::json j = nlohmann::json::object();
#if (defined(_M_X64) || defined(__x86_64__)) && defined(_MSC_VER)
  int r[4];
  __cpuid(r, 0);
  const int max_leaf = r[0];
  char vendor[13] = {};
  std::memcpy(vendor + 0, &r[1], 4);
  std::memcpy(vendor + 4, &r[3], 4);
  std::memcpy(vendor + 8, &r[2], 4);
  __cpuid(r, 1);
  const uint32_t sig = static_cast<uint32_t>(r[0]), ecx1 = static_cast<uint32_t>(r[2]);
  uint32_t family = (sig >> 8) & 0xF, model = (sig >> 4) & 0xF;
  const uint32_t stepping = sig & 0xF;
  if (family == 0xF) family += (sig >> 20) & 0xFF;
  if (family == 0x6 || family >= 0xF) model += ((sig >> 16) & 0xF) << 4;
  const bool osxsave = (ecx1 >> 27) & 1;
  uint32_t ebx7 = 0;
  if (max_leaf >= 7) {
    __cpuidex(r, 7, 0);
    ebx7 = static_cast<uint32_t>(r[1]);
  }
  char brand[49] = {};
  __cpuid(r, static_cast<int>(0x80000000u));
  if (static_cast<uint32_t>(r[0]) >= 0x80000004u) {
    for (int i = 0; i < 3; ++i) {
      __cpuid(r, static_cast<int>(0x80000002u + i));
      std::memcpy(brand + 16 * i, r, 16);
    }
  }
  std::string b(brand);
  while (!b.empty() && b.back() == ' ') b.pop_back();
  j["vendor"] = std::string(vendor);
  j["brand"] = b;
  j["family"] = family;
  j["model"] = model;
  j["stepping"] = stepping;
  j["fma3"] = ((ecx1 >> 12) & 1) != 0;
  j["avx"] = ((ecx1 >> 28) & 1) != 0;
  j["avx2"] = ((ebx7 >> 5) & 1) != 0;
  j["avx512f"] = ((ebx7 >> 16) & 1) != 0;
  j["xcr0"] = osxsave ? static_cast<uint64_t>(_xgetbv(0)) : uint64_t{0};
  j["crt_fma3_enable"] = _get_FMA3_enable();
#else
  j["unsupported_host"] = true;  // this converter only builds for x64 Windows
#endif
  return j;
}

// The checkpoint's files RunConvert reads from --input: model.safetensors.index.json (its text)
// and every shard it names, by file name.
struct CheckpointFiles {
  std::string index_text;
  std::vector<std::string> shards;
};

inline CheckpointFiles ReadCheckpointIndex(const std::string& dir) {
  const std::string index_path = dir + "\\model.safetensors.index.json";
  CheckpointFiles cf;
  {
    std::ifstream f(Utf8ToWide(index_path).c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("reuse guard: cannot open " + index_path);
    std::ostringstream ss;
    ss << f.rdbuf();
    cf.index_text = ss.str();
  }
  const nlohmann::json index = nlohmann::json::parse(cf.index_text);
  std::set<std::string> names;
  for (auto it = index.at("weight_map").begin(); it != index.at("weight_map").end(); ++it)
    names.insert(it.value().get<std::string>());
  cf.shards.assign(names.begin(), names.end());
  return cf;
}

// The checkpoint's identity: config.json and model.safetensors.index.json by content, and every
// shard the index names by (length, sha256) -- the WHOLE file, including shards a --layers N run
// never opens (conservative: one identity per checkpoint, not per run). `shard_digests` is
// Sha256Files over cf.shards (main.cpp hashes them in one pass with the Hessians). The directory
// path is deliberately NOT part of it: the content hashes are strictly stronger, and a moved
// checkpoint is the same checkpoint.
inline nlohmann::json CheckpointIdentity(const std::string& config_text, const CheckpointFiles& cf,
                                         const std::vector<FileDigest>& shard_digests) {
  nlohmann::json shards = nlohmann::json::object();
  for (size_t i = 0; i < cf.shards.size(); ++i)
    shards[cf.shards[i]] = {{"bytes", shard_digests.at(i).bytes},
                            {"sha256", shard_digests.at(i).sha256}};
  return {{"config_sha256", Sha256Hex(config_text)},
          {"index_sha256", Sha256Hex(cf.index_text)},
          {"shards", shards}};
}

// Every leaf of `j` as (path, value): objects are descended ("a.b.c"), everything else -- arrays and
// empty objects included -- is one leaf. Paths are for messages; the caller also compares the whole
// objects, so a key containing '.' cannot make two different guards compare equal.
inline void FlattenJsonLeaves(const nlohmann::json& j, const std::string& prefix,
                              std::map<std::string, nlohmann::json>& out) {
  if (j.is_object() && !j.empty()) {
    for (auto it = j.begin(); it != j.end(); ++it)
      FlattenJsonLeaves(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
    return;
  }
  out[prefix] = j;
}

// The guard fields that differ between the baseline's guard and this run's, one line each
// ("<path>: baseline <v>, this run <v>"), ignoring the completion fields kReuseCompleteKey and
// kReuseDataKey (checked on their own) and the per-linear record kReuseLinearsKey (PlanReuse's).
// Empty means the guards are identical.
inline std::vector<std::string> ReuseGuardMismatches(const nlohmann::json& baseline,
                                                     const nlohmann::json& current) {
  nlohmann::json a = baseline, b = current;
  for (nlohmann::json* g : {&a, &b}) {
    if (!g->is_object()) continue;
    g->erase(kReuseCompleteKey);
    g->erase(kReuseDataKey);
    g->erase(kReuseLinearsKey);
  }
  std::vector<std::string> out;
  if (a == b) return out;
  std::map<std::string, nlohmann::json> fa, fb;
  FlattenJsonLeaves(a, "reuse_guard", fa);
  FlattenJsonLeaves(b, "reuse_guard", fb);
  std::set<std::string> keys;
  for (const auto& kv : fa) keys.insert(kv.first);
  for (const auto& kv : fb) keys.insert(kv.first);
  auto show = [](const std::map<std::string, nlohmann::json>& m, const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? std::string("(absent)") : it->second.dump();
  };
  for (const auto& k : keys) {
    const std::string va = show(fa, k), vb = show(fb, k);
    if (va != vb) out.push_back(k + ": baseline " + va + ", this run " + vb);
  }
  if (out.empty()) out.push_back("reuse_guard: the objects differ in structure (baseline " + a.dump() +
                                 ", this run " + b.dump() + ")");
  return out;
}

// The baseline container: its header (raw bytes, for the identity recorded in reused_from, and
// parsed), its __metadata__, and SafetensorsReader -- the reader the runtime loads containers with
// (src/model/container.cpp) -- for the tensor directory, the bounds checks and the mapped data. The
// reader opens the file share-read only, so no process can write it while this object lives: the
// digests VerifyData computes stay true for the bytes the copy pass later reads.
struct BaselineContainer {
  std::string path;
  std::string header_bytes;
  nlohmann::json metadata;
  std::unique_ptr<SafetensorsReader> reader;
  std::map<std::string, std::string> tensor_sha256;  // VerifyData: every tensor's digest
  std::string data_sha256;                           // VerifyData: the recorded (= computed) root

  static std::unique_ptr<BaselineContainer> Open(const std::string& path) {
    const std::string who = "--reuse-tensors-from: " + path;
    auto b = std::make_unique<BaselineContainer>();
    b->path = path;
    {
      std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
      if (!f) throw std::runtime_error("--reuse-tensors-from: cannot open baseline " + path);
      uint64_t hl = 0;
      f.read(reinterpret_cast<char*>(&hl), 8);
      if (!f || hl == 0 || hl > (uint64_t{1} << 30))
        throw std::runtime_error(who + " is not an r4dx container (bad header length)");
    }
    // Validates every tensor's byte range against the file length (a truncated copy is refused
    // here) and locks the file against writers (see the struct comment).
    try {
      b->reader = std::make_unique<SafetensorsReader>(Utf8ToWide(path));
    } catch (const std::exception& e) {
      throw std::runtime_error(who + " is not a readable r4dx container: " + e.what());
    }
    b->header_bytes = b->reader->HeaderJson();
    nlohmann::json header;
    try {
      header = nlohmann::json::parse(b->header_bytes);
    } catch (const std::exception& e) {
      throw std::runtime_error(who + " is not an r4dx container (header is not JSON: " + e.what() + ")");
    }
    if (!header.is_object() || !header.contains("__metadata__") || !header["__metadata__"].is_object())
      throw std::runtime_error(who + " has no __metadata__ object");
    b->metadata = header.at("__metadata__");
    return b;
  }

  // Hashes every tensor straight from the mapping, `threads` tensors at a time (the largest first),
  // and refuses the baseline unless ContainerDataSha256 of them equals `recorded` (its
  // reuse_guard.data_sha256): a completed baseline that was later partly overwritten, or copied
  // by a tool that died half way (a full-length file, its header intact, its tail zeros), is never
  // copied from. Keeps the per-tensor digests: the copy pass checks each copied tensor against them.
  void VerifyData(const std::string& recorded, int threads) {
    const std::string who = "--reuse-tensors-from " + path;
    if (!IsSha256Hex(recorded) || recorded == ReuseDataPlaceholder())
      throw std::runtime_error(who + ": the baseline's reuse_guard." + kReuseDataKey + " is not a "
                               "recorded digest -- an interrupted or failed conversion; convert it "
                               "again");
    std::vector<std::string> names = reader->Names();
    std::vector<uint64_t> size(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      const TensorMeta& m = reader->Meta(names[i]);
      size[i] = m.end - m.begin;
    }
    std::vector<size_t> order(names.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return size[a] > size[c]; });
    std::vector<std::string> digest(names.size());
    ParallelEach(order.size(), threads, [&](size_t k) {
      const size_t i = order[k];
      Sha256 h;
      h.Update(reader->Data(names[i]), static_cast<size_t>(size[i]));
      digest[i] = h.HexDigest();
    });
    std::vector<std::pair<std::string, std::string>> named(names.size());
    for (size_t i = 0; i < names.size(); ++i) named[i] = {names[i], digest[i]};
    const std::string got = ContainerDataSha256(named);
    if (got != recorded)
      throw std::runtime_error(who + ": the baseline's tensor data does not match its reuse_guard." +
                               kReuseDataKey + " (recorded " + recorded + ", the file hashes to " +
                               got + ") -- it was modified or only partly copied after its "
                               "conversion finished; refusing to copy from it");
    for (auto& kv : named) tensor_sha256.emplace(std::move(kv.first), std::move(kv.second));
    data_sha256 = recorded;
  }
};

}  // namespace r4dx_convert
