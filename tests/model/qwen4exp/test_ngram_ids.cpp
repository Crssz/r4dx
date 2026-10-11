// tests/model/qwen4exp/test_ngram_ids.cpp -- CPU only, no sidecar body needed (docs/qwen4exp-plan.md 4.8).
//
// NgramIds (C++) against the ids tools/reference/qwen4exp/ngram_ids.py computes, bit for bit:
//   * every case of ids.bin generated in the Python chunking (including 0- and 1-token calls) and in one call;
//   * the 2-token history carried across calls (final state == the Python final context);
//   * snapshot / restore (prefix reuse) and Reset;
//   * tokens that would overflow int64 are refused and leave the state untouched;
//   * the header parser refuses a bad magic, version, geometry, padding and a truncated header.
// Fixtures: <R4DX_Q4EXP_FIXTURES or <models root>\r4dx\q4exp-fixtures>\NgramReader\{header.bin,ids.bin}, made by
//   D:\venvs\r4dx-qwen4-ref\Scripts\python.exe tools\reference\qwen4exp\ngram_fixtures.py
// Exit code 77 = skipped (fixture missing).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/model/qwen4exp/ngram_ids.h"
#include "r4dx/model/qwen4exp/ngram_sidecar.h"
#include "r4dx/models_root.h"

using namespace r4dx::model::qwen4exp;

namespace {

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

template <class F>
bool Throws(F&& f, const std::string& substr = "") {
  try {
    f();
  } catch (const std::exception& e) {
    if (substr.empty() || std::string(e.what()).find(substr) != std::string::npos) return true;
    std::fprintf(stderr, "  threw, but without \"%s\": %s\n", substr.c_str(), e.what());
    return false;
  }
  return false;
}

std::filesystem::path FixtureDir() {
  std::string base;
  char* env = nullptr;
  size_t len = 0;
  if (_dupenv_s(&env, &len, "R4DX_Q4EXP_FIXTURES") == 0 && env != nullptr) base = env;
  std::free(env);
  if (base.empty()) base = r4dx::ModelsPath("r4dx/q4exp-fixtures");
  return std::filesystem::path(base) / "NgramReader";
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Cursor {
  const std::vector<uint8_t>& buf;
  size_t off = 0;
  template <class T>
  T Get() {
    if (off + sizeof(T) > buf.size()) throw std::runtime_error("fixture truncated");
    T v;
    std::memcpy(&v, buf.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
  }
  template <class T>
  std::vector<T> GetVec(size_t n) {
    if (off + n * sizeof(T) > buf.size()) throw std::runtime_error("fixture truncated");
    std::vector<T> v(n);
    if (n) std::memcpy(v.data(), buf.data() + off, n * sizeof(T));
    off += n * sizeof(T);
    return v;
  }
};

struct Case {
  std::vector<uint32_t> chunks;
  bool has_ctx = false;
  NgramIds::State ctx;
  std::vector<int32_t> tokens;
  std::vector<int64_t> ids;  // [T, 16]
  NgramIds::State final_ctx;
};

std::vector<Case> LoadCases(const std::vector<uint8_t>& buf) {
  Cursor c{buf};
  if (std::memcmp(c.GetVec<char>(8).data(), "NGID0001", 8) != 0) throw std::runtime_error("ids.bin: bad magic");
  const uint32_t n = c.Get<uint32_t>();
  std::vector<Case> cases(n);
  for (Case& k : cases) {
    const uint32_t T = c.Get<uint32_t>();
    const uint32_t nc = c.Get<uint32_t>();
    k.chunks = c.GetVec<uint32_t>(nc);
    k.has_ctx = c.Get<uint32_t>() != 0;
    auto ctx = c.GetVec<int64_t>(2);
    k.ctx.tokens = {ctx[0], ctx[1]};
    k.tokens = c.GetVec<int32_t>(T);
    k.ids = c.GetVec<int64_t>(size_t{T} * NgramIds::kHeads);
    auto fin = c.GetVec<int64_t>(2);
    k.final_ctx.tokens = {fin[0], fin[1]};
  }
  if (c.off != buf.size()) throw std::runtime_error("ids.bin: trailing bytes");
  return cases;
}

size_t Mismatches(const std::vector<int64_t>& got, const std::vector<int64_t>& want) {
  if (got.size() != want.size()) return got.size() + want.size() + 1;
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
  return bad;
}

}  // namespace

int main() {
  const std::filesystem::path dir = FixtureDir();
  if (!std::filesystem::exists(dir / "header.bin") || !std::filesystem::exists(dir / "ids.bin")) {
    std::printf("SKIP: fixtures missing in %s (run tools/reference/qwen4exp/ngram_fixtures.py)\n", dir.string().c_str());
    return 77;
  }
  const std::vector<uint8_t> raw_header = ReadFile(dir / "header.bin");
  const NgramSidecarHeader header = NgramSidecarHeader::Parse(raw_header.data(), raw_header.size());
  std::printf("header: rows %llu, eos %lld, multipliers %lld %lld %lld\n", static_cast<unsigned long long>(header.rows),
              static_cast<long long>(header.eos_token_id), static_cast<long long>(header.multipliers[0]),
              static_cast<long long>(header.multipliers[1]), static_cast<long long>(header.multipliers[2]));
  const std::vector<Case> cases = LoadCases(ReadFile(dir / "ids.bin"));
  Check(!cases.empty(), "ids.bin has cases (" + std::to_string(cases.size()) + ")");

  NgramIds gen(header);
  std::printf("max accepted token id %lld\n", static_cast<long long>(gen.max_token_id()));
  Check(gen.max_token_id() >= 248319, "every id of the 248320-token vocabulary is accepted");

  size_t total_ids = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const Case& k = cases[ci];
    const std::string tag = "case " + std::to_string(ci) + " (" + std::to_string(k.tokens.size()) + " tokens, " +
                            std::to_string(k.chunks.size()) + " calls)";
    total_ids += k.ids.size();
    const auto start = [&] {
      if (k.has_ctx)
        gen.Restore(k.ctx);
      else
        gen.Reset();
    };

    // Python's chunking
    start();
    std::vector<int64_t> got(k.ids.size());
    size_t pos = 0;
    for (uint32_t c : k.chunks) {
      gen.Generate(k.tokens.data() + pos, c, got.data() + pos * NgramIds::kHeads);
      pos += c;
    }
    Check(Mismatches(got, k.ids) == 0, tag + ": chunked ids equal the reference");
    Check(gen.Snapshot() == k.final_ctx, tag + ": carried 2-token history equals the reference final context");

    // one call
    start();
    std::vector<int64_t> one(k.ids.size());
    gen.Generate(k.tokens.data(), k.tokens.size(), one.data());
    Check(Mismatches(one, k.ids) == 0, tag + ": single-call ids equal the reference");

    // token-by-token (the decode path)
    start();
    std::vector<int64_t> single(k.ids.size());
    for (size_t i = 0; i < k.tokens.size(); ++i) gen.Generate(k.tokens.data() + i, 1, single.data() + i * NgramIds::kHeads);
    Check(Mismatches(single, k.ids) == 0 && gen.Snapshot() == k.final_ctx, tag + ": one-token calls equal the reference");

    // snapshot after the first half, finish, restore, finish again; and Reset reproduces a fresh start
    if (k.tokens.size() >= 4) {
      start();
      const size_t half = k.tokens.size() / 2;
      std::vector<int64_t> a(k.ids.size()), b(k.ids.size());
      gen.Generate(k.tokens.data(), half, a.data());
      const NgramIds::State snap = gen.Snapshot();
      gen.Generate(k.tokens.data() + half, k.tokens.size() - half, a.data() + half * NgramIds::kHeads);
      gen.Restore(snap);
      gen.Generate(k.tokens.data() + half, k.tokens.size() - half, b.data() + half * NgramIds::kHeads);
      Check(Mismatches(std::vector<int64_t>(a.begin() + half * NgramIds::kHeads, a.end()),
                       std::vector<int64_t>(b.begin() + half * NgramIds::kHeads, b.end())) == 0 &&
                Mismatches(a, k.ids) == 0 && gen.Snapshot() == k.final_ctx,
            tag + ": snapshot/restore replays the second half identically");
    }
  }
  std::printf("compared %zu ids against ngram_ids.py\n", total_ids);

  {  // Reset == a fresh generator; a snapshot taken at the start is {EOS, EOS}
    NgramIds fresh(header);
    Check(fresh.Snapshot().tokens[0] == header.eos_token_id && fresh.Snapshot().tokens[1] == header.eos_token_id,
          "initial state is {EOS, EOS}");
    const int32_t t[3] = {7, 8, 9};
    int64_t x[3 * 16], y[3 * 16];
    fresh.Generate(t, 3, x);
    fresh.Reset();
    fresh.Generate(t, 3, y);
    Check(std::memcmp(x, y, sizeof(x)) == 0, "Reset restarts the sequence");
  }

  {  // overflow guard: refuse, state untouched
    NgramIds g(header);
    const int32_t good[2] = {11, 22};
    int64_t out[2 * 16];
    g.Generate(good, 2, out);
    const NgramIds::State before = g.Snapshot();
    const int32_t neg[2] = {5, -1};
    Check(Throws([&] { g.Generate(neg, 2, out); }, "outside"), "negative token refused");
    Check(g.Snapshot() == before, "state untouched after a refused call");
    if (g.max_token_id() < INT32_MAX) {
      const int32_t big[2] = {5, static_cast<int32_t>(g.max_token_id() + 1)};
      Check(Throws([&] { g.Generate(big, 2, out); }, "outside"), "token above the overflow-free bound refused");
      const int32_t edge[1] = {static_cast<int32_t>(g.max_token_id())};
      g.Generate(edge, 1, out);
      Check(out[0] >= 0, "the largest accepted token yields non-negative ids");
    }
    NgramIds::State bad;
    bad.tokens = {-3, 4};
    Check(Throws([&] { g.Restore(bad); }, "snapshot"), "Restore refuses an out-of-range snapshot");
  }

  {  // header parser refusals
    auto mutated = [&](size_t off, uint8_t v) {
      std::vector<uint8_t> h = raw_header;
      h[off] = v;
      return h;
    };
    auto parse = [](const std::vector<uint8_t>& h) { NgramSidecarHeader::Parse(h.data(), h.size()); };
    Check(Throws([&] { parse(mutated(0, 'X')); }, "bad magic"), "bad magic refused");
    Check(Throws([&] { parse(mutated(8, 2)); }, "unsupported version"), "version 2 refused");
    Check(Throws([&] { parse(mutated(20, 0xC0)); }, "row_bytes"), "row_bytes != 320 refused");
    Check(Throws([&] { parse(mutated(24, 8)); }, "heads"), "heads != 16 refused");
    Check(Throws([&] { parse(mutated(28, 128)); }, "head_dim"), "head_dim != 160 refused");
    Check(Throws([&] { parse(mutated(32, 4)); }, "ngram_size"), "ngram_size != 3 refused");
    Check(Throws([&] { parse(mutated(4000, 1)); }, "padding"), "non-zero padding refused");
    std::vector<uint8_t> shortbuf(raw_header.begin(), raw_header.begin() + 4095);
    Check(Throws([&] { parse(shortbuf); }, "4095 bytes"), "4095-byte header refused");
    std::vector<uint8_t> zero_mult = raw_header;
    std::memset(zero_mult.data() + 56, 0, 8);
    Check(Throws([&] { parse(zero_mult); }, "multiplier 0"), "zero multiplier refused");
    // a header whose heads index past padded_rows cannot build the id generator
    std::vector<uint8_t> small_rows = raw_header;
    const uint64_t one = 1;
    std::memcpy(small_rows.data() + 336, &one, 8);  // padded_rows 1 < rows
    Check(Throws([&] { parse(small_rows); }, "padded_rows"), "padded_rows < rows refused");
    NgramSidecarHeader tight = header;
    tight.padded_rows = header.head_offsets[15];  // last head no longer fits
    Check(Throws([&] { NgramIds g(tight); }, "beyond padded_rows"), "NgramIds refuses a header whose heads exceed padded_rows");
  }

  if (g_failures) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
