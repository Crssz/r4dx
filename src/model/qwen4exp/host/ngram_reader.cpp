// NgramReader: overlapped unbuffered sector reads of the n-gram sidecar. See ngram_reader.h.
#include "r4dx/model/qwen4exp/ngram_reader.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace r4dx::model::qwen4exp {

namespace {

constexpr uint64_t kSector = 4096;
constexpr ULONG_PTR kKeyPump = 1;
constexpr ULONG_PTR kKeyShutdown = 2;
constexpr size_t kMaxQueueDepth = 4096;

using Clock = std::chrono::steady_clock;

std::string WinError(DWORD err) {
  return "Windows error " + std::to_string(static_cast<unsigned long>(err)) + " (" +
         std::system_category().message(static_cast<int>(err)) + ")";
}

}  // namespace

struct NgramReader::Impl {
  // One copy of `len` bytes: sector bytes [src_off, src_off + len) -> row `row` bytes [dst_off, dst_off + len).
  struct Piece {
    uint32_t row;
    uint16_t src_off;
    uint16_t len;
    uint16_t dst_off;
  };
  // One 4 KiB sector read and the pieces it serves (indices into Batch::pieces).
  struct Unit {
    uint64_t sector;
    uint32_t piece_begin;
    uint32_t piece_end;
    uint32_t need;  // bytes of the sector the pieces use (the last sector of a file is only partly present)
  };
  struct Batch {
    Handle id = 0;
    uint8_t* dst = nullptr;
    size_t rows = 0;
    std::vector<Piece> pieces;
    std::vector<Unit> units;
    // guarded by Impl::mu
    size_t next = 0;  // next unit to issue
    size_t outstanding = 0;
    bool failed = false;
    bool done = false;
    std::string error;
    Clock::time_point start, end;
    std::atomic<int64_t> copy_ns{0};
  };
  struct Slot {
    OVERLAPPED ov;  // first member: the completion port hands back &ov
    uint8_t* buf = nullptr;
    std::shared_ptr<Batch> batch;
    size_t unit = 0;
  };

  Impl(const NgramSidecar& sc, const Options& opt) : path(sc.path()), header(sc.header()), file_bytes(sc.file_bytes()), options(opt) {
    if (options.queue_depth == 0 || options.queue_depth > kMaxQueueDepth)
      throw std::runtime_error("ngram reader: queue_depth " + std::to_string(options.queue_depth) + " is outside [1, " +
                               std::to_string(kMaxQueueDepth) + "]");
    if (options.completion_threads == 0 || options.completion_threads > 64)
      throw std::runtime_error("ngram reader: completion_threads " + std::to_string(options.completion_threads) + " is outside [1, 64]");
    OpenFile();
    try {
      port = CreateIoCompletionPort(file, nullptr, 0, 0);
      if (!port) throw std::runtime_error("ngram reader: CreateIoCompletionPort failed for " + path.string() + ": " + WinError(GetLastError()));
      slot_mem = static_cast<uint8_t*>(VirtualAlloc(nullptr, options.queue_depth * kSector, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
      if (!slot_mem) throw std::runtime_error("ngram reader: VirtualAlloc of the bounce buffers failed: " + WinError(GetLastError()));
      slots.resize(options.queue_depth);
      for (size_t i = 0; i < slots.size(); ++i) {
        std::memset(&slots[i].ov, 0, sizeof(OVERLAPPED));
        slots[i].buf = slot_mem + i * kSector;
        free_slots.push_back(&slots[i]);
      }
      for (size_t i = 0; i < options.completion_threads; ++i) workers.emplace_back([this] { WorkerLoop(); });
    } catch (...) {
      Shutdown();
      throw;
    }
  }

  ~Impl() { Shutdown(); }

  void OpenFile() {
    const DWORD base = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
    if (options.unbuffered) {
      file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, base | FILE_FLAG_NO_BUFFERING, nullptr);
      if (file != INVALID_HANDLE_VALUE) {
        is_unbuffered = true;
        return;
      }
      const DWORD err = GetLastError();
      std::fprintf(stderr,
                   "[r4dx::model::qwen4exp::NgramReader] WARNING: unbuffered open of %s failed (%s); falling back to buffered "
                   "reads through the OS file cache\n",
                   path.string().c_str(), WinError(err).c_str());
    }
    file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, base, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error("ngram reader: cannot open " + path.string() + ": " + WinError(GetLastError()));
    is_unbuffered = false;
  }

  // Aborts the unfinished batches, waits for the reads in flight, stops the threads, releases everything.
  void Shutdown() {
    {
      std::unique_lock<std::mutex> lk(mu);
      shutting_down = true;
      for (auto& b : active) {
        b->next = b->units.size();
        if (!b->failed) {
          b->failed = true;
          b->error = "ngram reader destroyed with the batch unfinished";
        }
      }
      active.clear();
      cv.wait(lk, [&] { return in_flight == 0; });
    }
    for (size_t i = 0; i < workers.size(); ++i) PostQueuedCompletionStatus(port, 0, kKeyShutdown, nullptr);
    for (auto& t : workers)
      if (t.joinable()) t.join();
    workers.clear();
    if (slot_mem) VirtualFree(slot_mem, 0, MEM_RELEASE);
    slot_mem = nullptr;
    if (port) CloseHandle(port);
    port = nullptr;
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    file = INVALID_HANDLE_VALUE;
  }

  // ---- slot accounting (mu held) ------------------------------------------------------------------------
  void FinishUnitLocked(Slot* s, const std::string* error) {
    Batch& b = *s->batch;
    if (error) {
      if (!b.failed) {
        b.failed = true;
        b.error = *error;
      }
      b.next = b.units.size();  // nothing further is issued for a failed batch
    }
    --b.outstanding;
    --in_flight;
    bool finished_now = false;
    if (b.outstanding == 0 && b.next >= b.units.size() && !b.done) {
      b.done = true;
      b.end = Clock::now();
      finished_now = true;
    }
    s->batch.reset();
    free_slots.push_back(s);
    // Waiters care about a finished batch, Shutdown about the last read in flight.
    if (finished_now || (shutting_down && in_flight == 0)) cv.notify_all();
  }

  // Issues reads while there is a free slot and an unissued sector (mu held).
  void PumpLocked() {
    while (!free_slots.empty() && !active.empty()) {
      std::shared_ptr<Batch> b = active.front();
      if (b->next >= b->units.size()) {
        active.pop_front();
        continue;
      }
      Slot* s = free_slots.back();
      free_slots.pop_back();
      const size_t unit = b->next++;
      s->batch = b;
      s->unit = unit;
      ++b->outstanding;
      ++in_flight;
      const uint64_t offset = b->units[unit].sector * kSector;
      std::memset(&s->ov, 0, sizeof(OVERLAPPED));
      s->ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
      s->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
      if (!ReadFile(file, s->buf, static_cast<DWORD>(kSector), nullptr, &s->ov)) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
          // Refused at submission: no completion packet will follow.
          const std::string msg = ReadError(offset, err, 0, 0);
          FinishUnitLocked(s, &msg);
        }
      }
    }
  }

  std::string ReadError(uint64_t offset, DWORD err, DWORD got, uint32_t need) const {
    const std::string where = "ngram sidecar " + path.string() + ": read at file offset " + std::to_string(offset);
    if (err == ERROR_HANDLE_EOF)
      return where + " is at/after the end of the file (" + std::to_string(file_bytes) + " bytes): short read";
    if (err != 0) return where + " failed: " + WinError(err);
    return where + ": short read, got " + std::to_string(got) + " bytes, need " + std::to_string(need) + " (file is " +
           std::to_string(file_bytes) + " bytes)";
  }

  void WorkerLoop() {
    for (;;) {
      DWORD bytes = 0;
      ULONG_PTR key = 0;
      OVERLAPPED* pov = nullptr;
      const BOOL ok = GetQueuedCompletionStatus(port, &bytes, &key, &pov, INFINITE);
      if (!pov) {
        if (!ok || key == kKeyShutdown) return;
        if (key == kKeyPump) {
          std::lock_guard<std::mutex> lk(mu);
          PumpLocked();
        }
        continue;
      }
      const DWORD err = ok ? 0 : GetLastError();
      Slot* s = reinterpret_cast<Slot*>(pov);
      Batch& b = *s->batch;
      const Unit& u = b.units[s->unit];
      std::string error;
      if (err != 0 || bytes < u.need) error = ReadError(u.sector * kSector, err, bytes, u.need);
      if (error.empty()) {
        const auto t0 = Clock::now();
        for (uint32_t i = u.piece_begin; i < u.piece_end; ++i) {
          const Piece& p = b.pieces[i];
          std::memcpy(b.dst + static_cast<size_t>(p.row) * NgramSidecarHeader::kRowBytes + p.dst_off, s->buf + p.src_off, p.len);
        }
        b.copy_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count(), std::memory_order_relaxed);
      }
      std::lock_guard<std::mutex> lk(mu);
      FinishUnitLocked(s, error.empty() ? nullptr : &error);
      PumpLocked();
    }
  }

  Handle Submit(const int64_t* row_ids, size_t n_rows, void* dst) {
    if (n_rows > UINT32_MAX) throw std::runtime_error("ngram reader: batch of " + std::to_string(n_rows) + " rows is too large");
    if (n_rows != 0 && (!row_ids || !dst)) throw std::runtime_error("ngram reader: null row id or destination pointer");
    constexpr uint64_t kRow = NgramSidecarHeader::kRowBytes;

    struct Tagged {
      uint64_t sector;
      Piece piece;
    };
    std::vector<Tagged> tagged;
    tagged.reserve(n_rows * 2);
    for (size_t i = 0; i < n_rows; ++i) {
      const int64_t id = row_ids[i];
      if (id < 0 || static_cast<uint64_t>(id) >= header.rows)
        throw std::runtime_error("ngram reader: row id " + std::to_string(id) + " (batch row " + std::to_string(i) + ") is outside [0, " +
                                 std::to_string(header.rows) + ")");
      const uint64_t off = NgramSidecarHeader::RowOffset(static_cast<uint64_t>(id));
      const uint64_t sector = off / kSector;
      const uint64_t so = off % kSector;
      const uint32_t row = static_cast<uint32_t>(i);
      if (so + kRow <= kSector) {
        tagged.push_back({sector, {row, static_cast<uint16_t>(so), static_cast<uint16_t>(kRow), 0}});
      } else {
        const uint64_t first = kSector - so;
        tagged.push_back({sector, {row, static_cast<uint16_t>(so), static_cast<uint16_t>(first), 0}});
        tagged.push_back({sector + 1, {row, 0, static_cast<uint16_t>(kRow - first), static_cast<uint16_t>(first)}});
      }
    }
    std::sort(tagged.begin(), tagged.end(), [](const Tagged& a, const Tagged& b) { return a.sector < b.sector; });

    auto batch = std::make_shared<Batch>();
    batch->dst = static_cast<uint8_t*>(dst);
    batch->rows = n_rows;
    batch->pieces.reserve(tagged.size());
    for (size_t i = 0; i < tagged.size(); ++i) {
      if (batch->units.empty() || batch->units.back().sector != tagged[i].sector)
        batch->units.push_back({tagged[i].sector, static_cast<uint32_t>(i), static_cast<uint32_t>(i), 0});
      Unit& u = batch->units.back();
      u.piece_end = static_cast<uint32_t>(i + 1);
      u.need = std::max<uint32_t>(u.need, tagged[i].piece.src_off + tagged[i].piece.len);
      batch->pieces.push_back(tagged[i].piece);
    }
    batch->start = Clock::now();

    {
      std::lock_guard<std::mutex> lk(mu);
      if (shutting_down) throw std::runtime_error("ngram reader: Submit during shutdown");
      batch->id = ++next_handle;
      if (batch->units.empty()) {
        batch->done = true;
        batch->end = batch->start;
      } else {
        active.push_back(batch);
      }
      batches.emplace(batch->id, batch);
    }
    if (!batch->units.empty() && !PostQueuedCompletionStatus(port, 0, kKeyPump, nullptr))
      throw std::runtime_error("ngram reader: PostQueuedCompletionStatus failed: " + WinError(GetLastError()));
    return batch->id;
  }

  Result Wait(Handle handle) {
    std::shared_ptr<Batch> b;
    {
      std::unique_lock<std::mutex> lk(mu);
      auto it = batches.find(handle);
      if (it == batches.end()) throw std::runtime_error("ngram reader: Wait on an unknown or already consumed handle " + std::to_string(handle));
      b = it->second;
      cv.wait(lk, [&] { return b->done; });
      batches.erase(it);
    }
    if (b->failed) throw std::runtime_error(b->error);
    Result r;
    r.rows = b->rows;
    r.sectors_read = b->units.size();
    r.elapsed_seconds = std::chrono::duration<double>(b->end - b->start).count();
    r.copy_seconds = static_cast<double>(b->copy_ns.load(std::memory_order_relaxed)) * 1e-9;
    return r;
  }

  std::filesystem::path path;
  NgramSidecarHeader header;
  uint64_t file_bytes;
  Options options;
  bool is_unbuffered = false;

  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE port = nullptr;
  uint8_t* slot_mem = nullptr;
  std::vector<Slot> slots;
  std::vector<std::thread> workers;

  std::mutex mu;
  std::condition_variable cv;
  std::vector<Slot*> free_slots;
  std::deque<std::shared_ptr<Batch>> active;  // batches with unissued sectors
  std::unordered_map<Handle, std::shared_ptr<Batch>> batches;
  size_t in_flight = 0;
  bool shutting_down = false;
  Handle next_handle = 0;
};

NgramReader::NgramReader(const NgramSidecar& sidecar, const Options& options) : impl_(std::make_unique<Impl>(sidecar, options)) {}
NgramReader::~NgramReader() = default;

NgramReader::Handle NgramReader::Submit(const int64_t* row_ids, size_t n_rows, void* dst) { return impl_->Submit(row_ids, n_rows, dst); }
NgramReader::Result NgramReader::Wait(Handle handle) { return impl_->Wait(handle); }
bool NgramReader::unbuffered() const { return impl_->is_unbuffered; }
const NgramReader::Options& NgramReader::options() const { return impl_->options; }

}  // namespace r4dx::model::qwen4exp
