// DirectStorage file -> VRAM loading (docs/direct-storage.md).
//
// Weights normally go SSD -> page cache (the safetensors mapping) -> pageable hipMemcpy -> VRAM.
// DirectStorageLoader instead has DirectStorage read the file straight into a small ring of D3D12
// buffers that are imported into HIP (hipImportExternalMemory), and copies each chunk device to
// device into the caller's own hipMalloc'd destination. No file byte enters the page cache or a
// process-owned host buffer.
//
// No D3D12 / DirectStorage / HIP header leaks through this one: the implementation is
// src/core/direct_storage.cpp (target r4dx_direct_storage).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace r4dx::core {

class DirectStorageLoader {
 public:
  // One chunk of the staging ring; also DirectStorage's largest single request.
  static constexpr uint64_t kChunkBytes = 32ull << 20;
  static constexpr int kRingSlots = 4;

  // A loader for the HIP device current on the calling thread: the D3D12 device is created on the
  // adapter with that device's LUID. nullptr, with the reason in *why_not (when given), if any of
  // dstorage.dll, D3D12, the adapter match or the HIP import is unavailable.
  static std::unique_ptr<DirectStorageLoader> Create(std::string* why_not);
  ~DirectStorageLoader();
  DirectStorageLoader(const DirectStorageLoader&) = delete;
  DirectStorageLoader& operator=(const DirectStorageLoader&) = delete;

  // Reads file[offset, offset + bytes) into the device memory at `dst` (on this loader's device).
  // Blocking: the bytes are in `dst` when it returns true. false, with nothing written, when
  // DirectStorage cannot open `path` at all (an unsupported volume or file system); a read that
  // fails after the open throws std::runtime_error, and the loader is unusable afterwards.
  bool Read(const std::wstring& path, uint64_t offset, uint64_t bytes, void* dst);

  uint64_t bytes_read() const;
  double read_seconds() const;

 private:
  DirectStorageLoader();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The loader the current thread's container load reads through. Constructed at the top of a
// container load (Container::Load and friends); nested scopes on a thread are no-ops. Disabled
// by R4DX_DIRECT_STORAGE=0. If DirectStorage cannot start, one line on stderr says why and every
// TryRead() returns false, so the caller's own (memory-mapped) path runs.
class DirectLoadScope {
 public:
  // Tensors below this are cheaper through the mapping than through a DirectStorage round trip.
  static constexpr uint64_t kMinBytes = 1ull << 20;

  DirectLoadScope();
  ~DirectLoadScope();
  DirectLoadScope(const DirectLoadScope&) = delete;
  DirectLoadScope& operator=(const DirectLoadScope&) = delete;

  // true: file[offset, offset + bytes) is in the device memory at `dst`. false: nothing written,
  // the caller copies from its own mapping (no scope, disabled, too small, unsupported volume).
  static bool TryRead(const std::wstring& path, uint64_t offset, uint64_t bytes, void* dst);

 private:
  bool owner_ = false;
};

}  // namespace r4dx::core
