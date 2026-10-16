// r4dx::model::qwen4exp::NgramReader -- asynchronous NVMe row reader of the n-gram sidecar
// (docs/qwen4exp-plan.md 4.8). Windows only.
//
// A token needs 16 rows of 320 bytes at byte 4096 + id * 320 of a 95 GiB file: random small reads, far
// beyond what fits in RAM. The reader opens the file with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (the
// OS file cache is bypassed; if the volume refuses an unbuffered open it falls back to a buffered overlapped
// open and prints a warning to stderr), associates it with an I/O completion port and keeps up to
// `queue_depth` 4096-byte sector reads in flight:
//
//   * every row is rounded to the 4 KiB sectors it spans (a 320-byte row straddles two sectors when its
//     offset % 4096 > 3776), identical sectors of one batch are read once;
//   * each in-flight read owns one 4096-byte page-aligned bounce buffer (slot); the completion thread
//     scatters the wanted bytes of that sector into the destination rows, then re-issues the slot;
//   * Submit returns immediately with a handle; Wait(handle) blocks until the whole batch is delivered.
//
// Destination layout: row i of the batch lands at dst + i * 320, so a batch of n tokens x 16 heads is
// [n, 16, 160] bf16 contiguous (= [n, 2560]); `dst` is typically pinned host memory. It must stay valid
// and untouched until Wait(handle) returns. A short read, an I/O error or a row id outside the table makes
// Wait throw std::runtime_error naming the file offset; a row is never filled with zeros instead.
//
// Thread-safety: Submit / Wait may be called from any one thread at a time per reader (several batches may
// be outstanding, sharing the slot pool); completions run on the reader's internal threads.
//
// Do not open the sidecar through the OS file cache (ifstream, mmap, NgramSidecar::ComputeBodySha256,
// Options::unbuffered = false) while an unbuffered reader of the same file is in use: measured on the dev
// machine, one live cached handle made the non-cached reads of the other handle ~20x slower (4096 rows:
// 23 ms -> 540 ms; the queue depth no longer mattered), recovering once the cached handle was closed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "r4dx/model/qwen4exp/ngram_sidecar.h"

namespace r4dx::model::qwen4exp {

class NgramReader {
 public:
  struct Options {
    size_t queue_depth = 64;       // sector reads in flight (= bounce slots)
    size_t completion_threads = 2;  // threads draining the completion port
    bool unbuffered = true;        // false: plain buffered overlapped reads (the OS cache is used)
  };

  struct Result {
    size_t rows = 0;               // rows delivered
    size_t sectors_read = 0;       // 4 KiB sector reads issued (after the per-batch dedupe)
    double elapsed_seconds = 0;    // Submit -> last row delivered
    double copy_seconds = 0;       // bounce buffer -> destination copying, summed over the completion threads
  };

  using Handle = uint64_t;

  NgramReader(const NgramSidecar& sidecar, const Options& options);
  explicit NgramReader(const NgramSidecar& sidecar) : NgramReader(sidecar, Options{}) {}
  ~NgramReader();  // aborts unfinished batches, waits for the reads in flight, closes the file
  NgramReader(const NgramReader&) = delete;
  NgramReader& operator=(const NgramReader&) = delete;

  // Starts reading `n_rows` rows (global row ids, e.g. the [n_tokens, 16] output of NgramIds::Generate with
  // n_rows = n_tokens * 16) into dst[n_rows * 320]. Throws at once when an id is outside [0, rows).
  Handle Submit(const int64_t* row_ids, size_t n_rows, void* dst);

  // Blocks until the batch is fully delivered; throws the batch's first I/O error. The handle is consumed.
  Result Wait(Handle handle);

  // False when the unbuffered open was refused and the reader fell back to buffered reads.
  bool unbuffered() const;
  const Options& options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace r4dx::model::qwen4exp
