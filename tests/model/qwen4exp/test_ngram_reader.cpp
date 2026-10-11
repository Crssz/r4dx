// tests/model/qwen4exp/test_ngram_reader.cpp -- CPU only (no GPU), reads the REAL 95 GiB sidecar (docs/qwen4exp-plan.md 4.8).
//
//   1. correctness: NgramReader (unbuffered, overlapped, IOCP) rows == the bytes ngram_fixtures.py read from the
//      real sidecar (ids.bin case 0's first 24 tokens, 200 random rows, edge rows, sector-straddling rows, a
//      batch with duplicate ids, several batches outstanding at once, an empty batch); a random 4096-row batch
//      is also checked against an independent buffered std::ifstream read of every row;
//   2. error paths: truncated copies of the sidecar (a short read and a read past EOF must throw with the file
//      offset), out-of-range ids, a consumed handle, strict Open of a truncated file, destruction with a batch
//      unfinished; the reader stays usable after a failed batch;
//   3. NgramSidecar::VerifyBodySha256 on small.r4dx-ngram (hashlib sha256 in its header) and on a corrupted copy;
//   4. measurements (informational, no pass/fail threshold): 4096-row batches (one 256-token prefill chunk) x 20
//      with fresh random tokens, then the same batch replayed, then a buffered reader (OS file cache) fresh and
//      replayed to show what the cache does, a queue-depth sweep, and the single-token latency (16 rows) over
//      200 trials.
// Environment: R4DX_NGRAM_SIDECAR (default <models root>\r4dx\swift15-flash-next.r4dx-ngram), R4DX_Q4EXP_FIXTURES
// (default <models root>\r4dx\q4exp-fixtures; this test reads <that>\NgramReader\). Exit code 77 = skipped.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "r4dx/model/qwen4exp/ngram_ids.h"
#include "r4dx/model/qwen4exp/ngram_reader.h"
#include "r4dx/model/qwen4exp/ngram_sidecar.h"
#include "r4dx/models_root.h"

using namespace r4dx::model::qwen4exp;
namespace fs = std::filesystem;

namespace {

constexpr size_t kRow = NgramSidecarHeader::kRowBytes;
constexpr size_t kHeads = NgramIds::kHeads;

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// True when f throws a message containing every substring.
template <class F>
bool Throws(F&& f, std::initializer_list<const char*> substrs) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string what = e.what();
    for (const char* s : substrs)
      if (what.find(s) == std::string::npos) {
        std::fprintf(stderr, "  threw, but without \"%s\": %s\n", s, what.c_str());
        return false;
      }
    std::printf("  (threw: %s)\n", what.c_str());
    return true;
  }
  std::fprintf(stderr, "  did not throw\n");
  return false;
}

std::string Env(const char* name) {
  std::string v;
  char* env = nullptr;
  size_t len = 0;
  if (_dupenv_s(&env, &len, name) == 0 && env != nullptr) v = env;
  std::free(env);
  return v;
}

std::vector<uint8_t> ReadAll(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void WriteAll(const fs::path& p, const uint8_t* data, size_t n) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
  if (!f) throw std::runtime_error("cannot write " + p.string());
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
  const uint8_t* Take(size_t n) {
    if (off + n > buf.size()) throw std::runtime_error("fixture truncated");
    const uint8_t* p = buf.data() + off;
    off += n;
    return p;
  }
};

using RowBytes = std::array<uint8_t, kRow>;

// ids.bin case 0: only the first 24 tokens' ids are needed here.
std::vector<int64_t> LoadFirstCaseIds(const std::vector<uint8_t>& buf, size_t tokens_wanted) {
  Cursor c{buf};
  if (std::memcmp(c.Take(8), "NGID0001", 8) != 0) throw std::runtime_error("ids.bin: bad magic");
  c.Get<uint32_t>();  // n cases
  const uint32_t T = c.Get<uint32_t>();
  const uint32_t nc = c.Get<uint32_t>();
  c.Take(size_t{nc} * 4 + 4 + 16);  // chunk lens, has_ctx, ctx
  c.Take(size_t{T} * 4);            // tokens
  std::vector<int64_t> ids(tokens_wanted * kHeads);
  std::memcpy(ids.data(), c.Take(ids.size() * 8), ids.size() * 8);
  return ids;
}

std::map<int64_t, RowBytes> LoadRows(const std::vector<uint8_t>& buf) {
  Cursor c{buf};
  if (std::memcmp(c.Take(8), "NGRW0001", 8) != 0) throw std::runtime_error("rows.bin: bad magic");
  const uint32_t n = c.Get<uint32_t>();
  std::map<int64_t, RowBytes> rows;
  for (uint32_t i = 0; i < n; ++i) {
    const int64_t id = c.Get<int64_t>();
    RowBytes r;
    std::memcpy(r.data(), c.Take(kRow), kRow);
    auto [it, fresh] = rows.emplace(id, r);
    if (!fresh && it->second != r) throw std::runtime_error("rows.bin: conflicting copies of row " + std::to_string(id));
  }
  return rows;
}

size_t CountBad(const std::vector<int64_t>& ids, const std::vector<uint8_t>& dst, const std::map<int64_t, RowBytes>& want) {
  size_t bad = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    auto it = want.find(ids[i]);
    if (it == want.end() || std::memcmp(dst.data() + i * kRow, it->second.data(), kRow) != 0) ++bad;
  }
  return bad;
}

// Independent reference: buffered std::ifstream, one seek + read per row.
size_t CountBadAgainstFile(const fs::path& path, const std::vector<int64_t>& ids, const std::vector<uint8_t>& dst) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  size_t bad = 0;
  RowBytes r;
  for (size_t i = 0; i < ids.size(); ++i) {
    f.clear();
    f.seekg(static_cast<std::streamoff>(NgramSidecarHeader::RowOffset(static_cast<uint64_t>(ids[i]))));
    f.read(reinterpret_cast<char*>(r.data()), kRow);
    if (static_cast<size_t>(f.gcount()) != kRow || std::memcmp(dst.data() + i * kRow, r.data(), kRow) != 0) ++bad;
  }
  return bad;
}

NgramReader::Result ReadBatch(NgramReader& rd, const std::vector<int64_t>& ids, std::vector<uint8_t>& dst) {
  dst.assign(ids.size() * kRow, 0xCD);  // sentinel: an undelivered row cannot pass for data
  return rd.Wait(rd.Submit(ids.data(), ids.size(), dst.data()));
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}
double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

constexpr int64_t kVocab = 248320;
using Clock = std::chrono::steady_clock;

struct BatchStats {
  std::vector<double> ms, copy_ms;
  size_t sectors = 0;
  size_t rows = 0;
};

std::vector<int64_t> RandomIds(NgramIds& gen, std::mt19937_64& rng, size_t tokens) {
  std::vector<int32_t> t(tokens);
  for (auto& x : t) x = static_cast<int32_t>(rng() % kVocab);
  std::vector<int64_t> ids(tokens * kHeads);
  gen.Generate(t.data(), t.size(), ids.data());
  return ids;
}

void Record(BatchStats& s, NgramReader& rd, const std::vector<int64_t>& ids, std::vector<uint8_t>& dst) {
  const auto t0 = Clock::now();
  const NgramReader::Result r = rd.Wait(rd.Submit(ids.data(), ids.size(), dst.data()));
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  s.ms.push_back(ms);
  s.copy_ms.push_back(r.copy_seconds * 1e3);
  s.sectors = r.sectors_read;
  s.rows = r.rows;
}

// n batches of `tokens` fresh random tokens each.
BatchStats RunFresh(NgramReader& rd, NgramIds& gen, std::mt19937_64& rng, int n, size_t tokens) {
  BatchStats s;
  std::vector<uint8_t> dst(tokens * kHeads * kRow);
  for (int i = 0; i < n; ++i) Record(s, rd, RandomIds(gen, rng, tokens), dst);
  return s;
}

// the same batch n times
BatchStats RunSame(NgramReader& rd, const std::vector<int64_t>& ids, int n) {
  BatchStats s;
  std::vector<uint8_t> dst(ids.size() * kRow);
  for (int i = 0; i < n; ++i) Record(s, rd, ids, dst);
  return s;
}

void Report(const std::string& label, const BatchStats& s) {
  const double med = Median(s.ms);
  const double mn = *std::min_element(s.ms.begin(), s.ms.end());
  const double mx = *std::max_element(s.ms.begin(), s.ms.end());
  const double delivered_mb = static_cast<double>(s.rows * kRow) / 1e6;
  const double disk_mb = static_cast<double>(s.sectors * 4096) / 1e6;
  std::printf("  %-44s n=%-3zu median %8.3f ms (min %.3f, max %.3f)  %7.1f MB/s delivered, %7.1f MB/s sector traffic, %8.0f sector-reads/s, "
              "%zu rows -> %zu sectors, copy %.3f ms (%.1f%%)\n",
              label.c_str(), s.ms.size(), med, mn, mx, delivered_mb / (med * 1e-3), disk_mb / (med * 1e-3),
              static_cast<double>(s.sectors) / (med * 1e-3), s.rows, s.sectors, Median(s.copy_ms), 100.0 * Median(s.copy_ms) / med);
}

}  // namespace

int main() {
  std::string sidecar_path = Env("R4DX_NGRAM_SIDECAR");
  if (sidecar_path.empty()) sidecar_path = r4dx::ModelsPath("r4dx/swift15-flash-next.r4dx-ngram");
  std::string fixture_base = Env("R4DX_Q4EXP_FIXTURES");
  if (fixture_base.empty()) fixture_base = r4dx::ModelsPath("r4dx/q4exp-fixtures");
  const fs::path fixtures = fs::path(fixture_base) / "NgramReader";
  for (const char* f : {"header.bin", "ids.bin", "rows.bin", "small.r4dx-ngram"})
    if (!fs::exists(fixtures / f)) {
      std::printf("SKIP: fixture %s missing (run tools/reference/qwen4exp/ngram_fixtures.py)\n", (fixtures / f).string().c_str());
      return 77;
    }
  if (!fs::exists(sidecar_path)) {
    std::printf("SKIP: sidecar %s missing\n", sidecar_path.c_str());
    return 77;
  }
  const fs::path tmp = fs::temp_directory_path() / "r4dx_ngram_reader_test";
  fs::remove_all(tmp);
  fs::create_directories(tmp);

  // ---- open ---------------------------------------------------------------------------------------------
  const NgramSidecar sidecar = NgramSidecar::Open(sidecar_path);
  const NgramSidecarHeader& hdr = sidecar.header();
  {
    std::ifstream f(sidecar_path, std::ios::binary);
    std::vector<uint8_t> first(NgramSidecarHeader::kHeaderBytes);
    f.read(reinterpret_cast<char*>(first.data()), static_cast<std::streamsize>(first.size()));
    Check(first == ReadAll(fixtures / "header.bin"), "the sidecar's first 4096 bytes equal header.bin");
  }
  std::printf("sidecar %s: %llu rows, %.2f GiB, OS file size %llu\n", sidecar_path.c_str(), static_cast<unsigned long long>(hdr.rows),
              static_cast<double>(hdr.FileBytes()) / (1ull << 30), static_cast<unsigned long long>(sidecar.file_bytes()));
  Check(sidecar.file_bytes() == hdr.FileBytes(), "file size == header + rows * 320");

  const auto rows_fixture = LoadRows(ReadAll(fixtures / "rows.bin"));
  std::printf("rows.bin: %zu distinct rows\n", rows_fixture.size());

  // ---- 1. correctness -----------------------------------------------------------------------------------
  {
    NgramReader rd(sidecar);
    std::printf("default reader: queue_depth %zu, completion threads %zu, unbuffered %s\n", rd.options().queue_depth,
                rd.options().completion_threads, rd.unbuffered() ? "yes (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)" : "NO (fell back to buffered)");

    std::vector<int64_t> ids;
    for (const auto& kv : rows_fixture) ids.push_back(kv.first);
    std::mt19937_64 rng(7);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (size_t i = 0; i < 100; ++i) ids.push_back(ids[i]);  // duplicates in one batch
    size_t straddle = 0;
    for (int64_t id : ids) straddle += (NgramSidecarHeader::RowOffset(static_cast<uint64_t>(id)) % 4096) + kRow > 4096;
    std::vector<uint8_t> dst;
    const NgramReader::Result r = ReadBatch(rd, ids, dst);
    std::printf("fixture batch: %zu rows (%zu straddle two sectors) -> %zu sector reads\n", r.rows, straddle, r.sectors_read);
    Check(straddle >= 5, "the fixture batch contains sector-straddling rows");
    Check(r.sectors_read < ids.size() + straddle, "identical sectors of a batch are read once");
    Check(CountBad(ids, dst, rows_fixture) == 0, "all fixture rows (random, edges, straddling, duplicates) are byte-exact");

    const std::vector<int64_t> tok_ids = LoadFirstCaseIds(ReadAll(fixtures / "ids.bin"), 24);
    ReadBatch(rd, tok_ids, dst);
    Check(CountBad(tok_ids, dst, rows_fixture) == 0, "rows of the fixture ids of 24 tokens ([24,16,160] bf16) are byte-exact");

    // several batches outstanding at once, waited in another order
    {
      std::vector<std::vector<int64_t>> parts(3);
      std::vector<std::vector<uint8_t>> dsts(3);
      for (size_t i = 0; i < ids.size(); ++i) parts[i % 3].push_back(ids[i]);
      std::vector<NgramReader::Handle> h(3);
      for (int i = 0; i < 3; ++i) {
        dsts[i].assign(parts[i].size() * kRow, 0xCD);
        h[i] = rd.Submit(parts[i].data(), parts[i].size(), dsts[i].data());
      }
      for (int i = 2; i >= 0; --i) rd.Wait(h[i]);
      size_t bad = 0;
      for (int i = 0; i < 3; ++i) bad += CountBad(parts[i], dsts[i], rows_fixture);
      Check(bad == 0, "three batches in flight at once all deliver byte-exact rows");
    }
    {
      const NgramReader::Result e = rd.Wait(rd.Submit(nullptr, 0, nullptr));
      Check(e.rows == 0 && e.sectors_read == 0, "an empty batch completes at once");
    }
    {  // 4096 random rows (one 256-token chunk) against an independent buffered read of every row
      NgramIds gen(hdr);
      std::mt19937_64 r2(99);
      const std::vector<int64_t> big = RandomIds(gen, r2, 256);
      ReadBatch(rd, big, dst);
      Check(CountBadAgainstFile(sidecar_path, big, dst) == 0, "4096 random rows equal an independent buffered ifstream read");
    }
  }
  for (auto [qd, th] : {std::pair<size_t, size_t>{1, 1}, {3, 4}, {256, 1}}) {
    NgramReader::Options o;
    o.queue_depth = qd;
    o.completion_threads = th;
    NgramReader rd(sidecar, o);
    std::vector<int64_t> ids;
    for (const auto& kv : rows_fixture) ids.push_back(kv.first);
    std::vector<uint8_t> dst;
    ReadBatch(rd, ids, dst);
    Check(CountBad(ids, dst, rows_fixture) == 0, "queue depth " + std::to_string(qd) + ", " + std::to_string(th) + " completion thread(s): byte-exact");
  }
  {
    NgramReader::Options o;
    o.unbuffered = false;
    NgramReader rd(sidecar, o);
    std::vector<int64_t> ids;
    for (const auto& kv : rows_fixture) ids.push_back(kv.first);
    std::vector<uint8_t> dst;
    ReadBatch(rd, ids, dst);
    Check(!rd.unbuffered() && CountBad(ids, dst, rows_fixture) == 0, "buffered mode: byte-exact");
  }

  // ---- 3. sha256 ----------------------------------------------------------------------------------------
  const std::vector<uint8_t> small_bytes = ReadAll(fixtures / "small.r4dx-ngram");
  {
    const NgramSidecar small = NgramSidecar::Open(fixtures / "small.r4dx-ngram");
    Check(small.header().rows == 8 && small.file_bytes() == 4096 + 8 * kRow, "small sidecar: 8 rows, 6656 bytes");
    Check(small.VerifyBodySha256(), "VerifyBodySha256 accepts the body whose hashlib digest is in the header");
    std::vector<uint8_t> bad = small_bytes;
    bad[4096 + 100] ^= 1;
    WriteAll(tmp / "corrupt.r4dx-ngram", bad.data(), bad.size());
    Check(!NgramSidecar::Open(tmp / "corrupt.r4dx-ngram").VerifyBodySha256(), "VerifyBodySha256 rejects a body with one flipped bit");

    // the last, partially present sector (file ends 2560 bytes into sector 1) must read fine
    NgramReader rd(small);
    std::vector<int64_t> ids = {7, 0, 3, 7, 6, 1, 2, 4, 5};
    std::vector<uint8_t> dst;
    ReadBatch(rd, ids, dst);
    bool ok = true;
    for (size_t i = 0; i < ids.size(); ++i) ok &= std::memcmp(dst.data() + i * kRow, small_bytes.data() + 4096 + static_cast<size_t>(ids[i]) * kRow, kRow) == 0;
    Check(ok, "small sidecar: rows 0..7 read byte-exact (last sector is only partly present)");
    const int64_t past[1] = {8};
    uint8_t one[kRow];
    Check(Throws([&] { rd.Submit(past, 1, one); }, {"row id 8", "outside"}), "row id == rows refused at Submit");
  }

  // ---- 2. error paths -----------------------------------------------------------------------------------
  {
    NgramReader rd(sidecar);
    uint8_t one[kRow];
    const int64_t neg[1] = {-1};
    const int64_t over[1] = {static_cast<int64_t>(hdr.rows)};
    Check(Throws([&] { rd.Submit(neg, 1, one); }, {"row id -1", "outside"}), "negative row id refused");
    Check(Throws([&] { rd.Submit(over, 1, one); }, {"outside"}), "row id == rows refused");
    const int64_t ok_id[1] = {5};
    const NgramReader::Handle h = rd.Submit(ok_id, 1, one);
    rd.Wait(h);
    Check(Throws([&] { rd.Wait(h); }, {"unknown or already consumed"}), "a consumed handle cannot be waited twice");
    Check(Throws([&] { rd.Wait(12345); }, {"unknown"}), "an unknown handle is refused");
  }
  {
    // real header (320 M rows) + only 8 rows on disk: a truncated copy
    const std::vector<uint8_t> real_header = ReadAll(fixtures / "header.bin");
    std::vector<uint8_t> trunc(real_header);
    trunc.insert(trunc.end(), small_bytes.begin() + 4096, small_bytes.end());  // 4096 + 2560
    WriteAll(tmp / "trunc8.r4dx-ngram", trunc.data(), trunc.size());
    Check(Throws([&] { NgramSidecar::Open(tmp / "trunc8.r4dx-ngram"); }, {"bytes, header says"}), "strict Open refuses a truncated sidecar");
    const NgramSidecar ts = NgramSidecar::Open(tmp / "trunc8.r4dx-ngram", /*allow_truncated=*/true);
    NgramReader rd(ts);
    std::vector<uint8_t> dst(4 * kRow);

    const int64_t good[2] = {3, 7};
    rd.Wait(rd.Submit(good, 2, dst.data()));
    Check(std::memcmp(dst.data(), small_bytes.data() + 4096 + 3 * kRow, kRow) == 0 &&
              std::memcmp(dst.data() + kRow, small_bytes.data() + 4096 + 7 * kRow, kRow) == 0,
          "truncated copy: rows that are on disk read fine (row 7 ends exactly at EOF)");
    const int64_t short_row[1] = {8};  // sector 1 starts at 4096, file ends at 6656: sector read returns 2560 < 2880 bytes
    Check(Throws([&] { rd.Wait(rd.Submit(short_row, 1, dst.data())); }, {"short read", "file offset 4096"}),
          "row beyond EOF inside the last sector: short read throws with the offset");
    const int64_t past_eof[1] = {20};  // byte 10496: sector 2 (offset 8192) is entirely past EOF
    Check(Throws([&] { rd.Wait(rd.Submit(past_eof, 1, dst.data())); }, {"short read", "file offset 8192"}),
          "row in a sector past EOF: throws with the offset");
    const int64_t mixed[3] = {0, 3, 20};
    Check(Throws([&] { rd.Wait(rd.Submit(mixed, 3, dst.data())); }, {"short read"}), "a batch with one bad row throws");
    rd.Wait(rd.Submit(good, 2, dst.data()));
    Check(std::memcmp(dst.data() + kRow, small_bytes.data() + 4096 + 7 * kRow, kRow) == 0, "the reader is usable after a failed batch");

    // file cut in the middle of row 7 (100 bytes short)
    WriteAll(tmp / "trunc_mid.r4dx-ngram", trunc.data(), trunc.size() - 100);
    const NgramSidecar tm = NgramSidecar::Open(tmp / "trunc_mid.r4dx-ngram", true);
    NgramReader rd2(tm);
    const int64_t r6[1] = {6}, r7[1] = {7};
    rd2.Wait(rd2.Submit(r6, 1, dst.data()));
    Check(std::memcmp(dst.data(), small_bytes.data() + 4096 + 6 * kRow, kRow) == 0, "file cut mid row 7: row 6 still reads");
    Check(Throws([&] { rd2.Wait(rd2.Submit(r7, 1, dst.data())); }, {"short read", "got 2460", "need 2560"}), "file cut mid row 7: row 7 throws a short read");

    // destruction with a batch unfinished must not hang or crash
    {
      std::vector<int64_t> many(4096);
      std::mt19937_64 rng(5);
      for (auto& id : many) id = static_cast<int64_t>(rng() % hdr.rows);
      std::vector<uint8_t> big(many.size() * kRow);
      {
        NgramReader rd3(sidecar);
        rd3.Submit(many.data(), many.size(), big.data());
      }
      Check(true, "reader destroyed with a 4096-row batch unfinished (no hang, no crash)");
    }
  }

  // ---- 4. measurements ----------------------------------------------------------------------------------
  std::printf("\nMEASUREMENTS (informational). Test host: %u hardware threads. 'delivered' = rows x 320 B copied to the destination, "
              "'sector traffic' = 4096 B per sector read.\n",
              std::thread::hardware_concurrency());
  {
    NgramIds gen(hdr);
    std::mt19937_64 rng(2026);
    // NOTE on ordering: all unbuffered measurements come first and no buffered handle of the sidecar is open
    // while they run. A cached handle open on the same file at the same time makes Windows serialize the
    // non-cached reads (measured at the end of this test), which is why the runtime must never open the
    // sidecar through the file cache (ifstream, mmap) while the reader is alive.
    NgramReader rd(sidecar);
    std::printf("unbuffered, queue depth 64, 2 completion threads, 256-token chunk = 4096 rows:\n");
    RunFresh(rd, gen, rng, 3, 256);  // thread / slot warm-up, not reported
    Report("fresh random batches (never read before)", RunFresh(rd, gen, rng, 20, 256));
    const std::vector<int64_t> one_batch = RandomIds(gen, rng, 256);
    Report("one more fresh batch (first read)", RunSame(rd, one_batch, 1));
    Report("same batch replayed x10", RunSame(rd, one_batch, 10));

    std::printf("single token (16 rows), unbuffered, queue depth 64, 200 fresh trials (Submit + Wait wall time):\n");
    {
      std::vector<double> lat;
      std::vector<uint8_t> dst(kHeads * kRow);
      for (int i = 0; i < 10; ++i) {
        BatchStats w;
        Record(w, rd, RandomIds(gen, rng, 1), dst);
      }
      for (int i = 0; i < 200; ++i) {
        BatchStats w;
        Record(w, rd, RandomIds(gen, rng, 1), dst);
        lat.push_back(w.ms[0]);
      }
      std::printf("  p50 %.3f ms, p90 %.3f ms, p99 %.3f ms, min %.3f ms, max %.3f ms\n", Median(lat), Percentile(lat, 0.90),
                  Percentile(lat, 0.99), *std::min_element(lat.begin(), lat.end()), *std::max_element(lat.begin(), lat.end()));
    }

    std::printf("unbuffered queue-depth / completion-thread sweep (10 fresh 4096-row batches each):\n");
    for (auto [qd, th] : {std::pair<size_t, size_t>{1, 1}, {4, 1}, {8, 1}, {16, 1}, {32, 1}, {64, 1}, {64, 2}, {64, 4}, {128, 2}, {256, 2}}) {
      NgramReader::Options o;
      o.queue_depth = qd;
      o.completion_threads = th;
      NgramReader srd(sidecar, o);
      RunFresh(srd, gen, rng, 1, 256);
      Report("queue depth " + std::to_string(qd) + ", " + std::to_string(th) + " thread(s)", RunFresh(srd, gen, rng, 10, 256));
    }
    Report("default reader again, fresh batches", RunFresh(rd, gen, rng, 10, 256));

    // the buffered reader last; while its handle is open the unbuffered reader is measured once more
    NgramReader::Options bo;
    bo.unbuffered = false;
    NgramReader brd(sidecar, bo);
    std::printf("buffered (OS file cache), same settings:\n");
    Report("fresh random batches (cold OS cache)", RunFresh(brd, gen, rng, 20, 256));
    const std::vector<int64_t> bbatch = RandomIds(gen, rng, 256);
    Report("one more fresh batch (first read)", RunSame(brd, bbatch, 1));
    Report("same batch replayed x10 (warm OS cache)", RunSame(brd, bbatch, 10));
    std::printf("unbuffered reader (opened before the buffered one) while the buffered handle is still open:\n");
    Report("same batch, unbuffered x10", RunSame(rd, bbatch, 10));
    Report("fresh random batches, unbuffered", RunFresh(rd, gen, rng, 10, 256));
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
