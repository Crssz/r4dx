#include "r4dx/core/direct_storage.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <dstorage.h>

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <mutex>
#include <unordered_set>

namespace r4dx::core {

using Microsoft::WRL::ComPtr;

namespace {

std::string Hex(HRESULT hr) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%08lX", static_cast<unsigned long>(hr));
  return b;
}

// dstorage.dll is loaded by name from the process directory first, then from the copy vendored in
// the source tree (third_party/directstorage/bin/x64, baked in by CMake as R4DX_DSTORAGE_DIR).
// dstorage.dll finds dstoragecore.dll through the process's search path, NOT next to itself, so the
// vendored core is loaded first by full path: the later by-name request then binds to that module.
// (The core module is never unloaded; the loader's own refcount on dstorage.dll is.)
HMODULE LoadDStorage() {
  if (HMODULE m = LoadLibraryW(L"dstorage.dll")) return m;
#ifdef R4DX_DSTORAGE_DIR
  const std::wstring dir = L"" R4DX_DSTORAGE_DIR;
  if (!LoadLibraryW((dir + L"\\dstoragecore.dll").c_str())) return nullptr;
  return LoadLibraryW((dir + L"\\dstorage.dll").c_str());
#else
  return nullptr;
#endif
}

using GetFactoryFn = HRESULT(WINAPI*)(REFIID, void**);

}  // namespace

struct DirectStorageLoader::Impl {
  struct Slot {
    ComPtr<ID3D12Resource> res;
    HANDLE shared = nullptr;
    hipExternalMemory_t ext = nullptr;
    void* ptr = nullptr;
    hipEvent_t copied = nullptr;  // the D2D copy out of this slot has finished
    uint64_t fence_value = 0;     // signalled when DirectStorage has filled the slot
  };

  HMODULE dstorage = nullptr;
  ComPtr<ID3D12Device> d3d;
  ComPtr<IDStorageFactory> factory;
  ComPtr<IDStorageQueue> queue;
  ComPtr<ID3D12Fence> fence;
  HANDLE fence_event = nullptr;
  uint64_t next_fence = 0;
  Slot slots[kRingSlots];
  hipStream_t stream = nullptr;
  std::unordered_map<std::wstring, ComPtr<IDStorageFile>> files;
  std::unordered_set<std::wstring> unsupported;
  bool broken = false;
  uint64_t total_bytes = 0;
  double total_seconds = 0;

  ~Impl() {
    if (stream) static_cast<void>(hipStreamSynchronize(stream));
    queue.Reset();
    for (Slot& s : slots) {
      if (s.copied) static_cast<void>(hipEventDestroy(s.copied));
      if (s.ptr) static_cast<void>(hipFree(s.ptr));  // the mapped buffer is released before its external memory
      if (s.ext) static_cast<void>(hipDestroyExternalMemory(s.ext));
      if (s.shared) CloseHandle(s.shared);
      s.res.Reset();
    }
    if (stream) static_cast<void>(hipStreamDestroy(stream));
    if (fence_event) CloseHandle(fence_event);
    files.clear();
    fence.Reset();
    factory.Reset();
    d3d.Reset();
    if (dstorage) FreeLibrary(dstorage);
  }

  // Returns the reason on failure, empty on success.
  std::string Init() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return "no current HIP device";
    char luid[8];
    unsigned mask = 0;
    if (hipDeviceGetLuid(luid, &mask, dev) != hipSuccess) return "hipDeviceGetLuid failed";

    dstorage = LoadDStorage();
    if (!dstorage) return "dstorage.dll not found";
    auto get_factory = reinterpret_cast<GetFactoryFn>(GetProcAddress(dstorage, "DStorageGetFactory"));
    if (!get_factory) return "dstorage.dll has no DStorageGetFactory";

    ComPtr<IDXGIFactory4> dxgi;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi));
    if (FAILED(hr)) return "CreateDXGIFactory1 " + Hex(hr);
    LUID l;
    std::memcpy(&l, luid, sizeof l);
    ComPtr<IDXGIAdapter1> adapter;
    hr = dxgi->EnumAdapterByLuid(l, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) return "no DXGI adapter with the HIP device's LUID " + Hex(hr);
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d));
    if (FAILED(hr)) return "D3D12CreateDevice " + Hex(hr);

    hr = get_factory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return "DStorageGetFactory " + Hex(hr);
    // The factory is process-wide (one per process, shared by every rank thread's loader): its staging
    // size is set once, before the first queue exists -- a later call is refused while a queue is live.
    static std::once_flag staging_once;
    static HRESULT staging_hr = S_OK;
    std::call_once(staging_once, [&] { staging_hr = factory->SetStagingBufferSize(static_cast<UINT32>(2 * kChunkBytes)); });
    if (FAILED(staging_hr)) return "SetStagingBufferSize " + Hex(staging_hr);

    DSTORAGE_QUEUE_DESC qd{};
    qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    qd.Priority = DSTORAGE_PRIORITY_NORMAL;
    qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    qd.Device = d3d.Get();
    hr = factory->CreateQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) return "IDStorageFactory::CreateQueue " + Hex(hr);
    hr = d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) return "CreateFence " + Hex(hr);
    fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event) return "CreateEventW failed";
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) return "hipStreamCreate failed";

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = kChunkBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (Slot& s : slots) {
      hr = d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(&s.res));
      if (FAILED(hr)) return "CreateCommittedResource " + Hex(hr);
      hr = d3d->CreateSharedHandle(s.res.Get(), nullptr, GENERIC_ALL, nullptr, &s.shared);
      if (FAILED(hr)) return "CreateSharedHandle " + Hex(hr);
      hipExternalMemoryHandleDesc d{};
      d.type = hipExternalMemoryHandleTypeD3D12Resource;
      d.handle.win32.handle = s.shared;
      d.size = kChunkBytes;
      d.flags = hipExternalMemoryDedicated;
      hipError_t e = hipImportExternalMemory(&s.ext, &d);
      if (e != hipSuccess) return std::string("hipImportExternalMemory: ") + hipGetErrorString(e);
      hipExternalMemoryBufferDesc b{};
      b.size = kChunkBytes;
      e = hipExternalMemoryGetMappedBuffer(&s.ptr, s.ext, &b);
      if (e != hipSuccess) return std::string("hipExternalMemoryGetMappedBuffer: ") + hipGetErrorString(e);
      if (hipEventCreateWithFlags(&s.copied, hipEventDisableTiming) != hipSuccess) return "hipEventCreate failed";
    }
    return {};
  }

  IDStorageFile* File(const std::wstring& path) {
    auto it = files.find(path);
    if (it != files.end()) return it->second.Get();
    if (unsupported.count(path)) return nullptr;
    ComPtr<IDStorageFile> f;
    const HRESULT hr = factory->OpenFile(path.c_str(), IID_PPV_ARGS(&f));
    if (FAILED(hr)) {
      std::fprintf(stderr, "r4dx: DirectStorage cannot open the file (%s) -- that file loads through the mapping\n",
                   Hex(hr).c_str());
      unsupported.insert(path);
      return nullptr;
    }
    return files.emplace(path, std::move(f)).first->second.Get();
  }

  void Submit(IDStorageFile* file, uint64_t off, uint32_t n, Slot& s) {
    DSTORAGE_REQUEST r{};
    r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    r.Source.File.Source = file;
    r.Source.File.Offset = off;
    r.Source.File.Size = n;
    r.Destination.Buffer.Resource = s.res.Get();
    r.Destination.Buffer.Offset = 0;
    r.Destination.Buffer.Size = n;
    r.UncompressedSize = n;
    queue->EnqueueRequest(&r);
    s.fence_value = ++next_fence;
    queue->EnqueueSignal(fence.Get(), s.fence_value);
    queue->Submit();
  }

  void Fail(const std::string& what) {
    broken = true;
    throw std::runtime_error("r4dx::core::DirectStorageLoader: " + what);
  }

  void Check(hipError_t e, const char* what) {
    if (e != hipSuccess) Fail(std::string(what) + ": " + hipGetErrorString(e));
  }

  // Blocks until DirectStorage has filled `s` (or the queue reports an error).
  void WaitFilled(Slot& s) {
    if (fence->GetCompletedValue() < s.fence_value) {
      fence->SetEventOnCompletion(s.fence_value, fence_event);
      const HANDLE waits[2] = {fence_event, queue->GetErrorEvent()};
      WaitForMultipleObjects(2, waits, FALSE, INFINITE);
    }
    DSTORAGE_ERROR_RECORD rec{};
    queue->RetrieveErrorRecord(&rec);
    if (FAILED(rec.FirstFailure.HResult)) {
      Fail("a DirectStorage request failed (" + Hex(rec.FirstFailure.HResult) + ")");
    }
    if (fence->GetCompletedValue() < s.fence_value) Fail("the DirectStorage queue stopped before the request completed");
  }

  bool Read(const std::wstring& path, uint64_t offset, uint64_t bytes, void* dst) {
    if (broken) throw std::runtime_error("r4dx::core::DirectStorageLoader: an earlier read failed");
    if (bytes == 0) return true;
    IDStorageFile* file = File(path);
    if (!file) return false;

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t n_chunks = (bytes + kChunkBytes - 1) / kChunkBytes;
    const auto chunk_off = [&](uint64_t i) { return i * kChunkBytes; };
    const auto chunk_len = [&](uint64_t i) {
      return static_cast<uint32_t>(std::min<uint64_t>(kChunkBytes, bytes - chunk_off(i)));
    };
    uint64_t submitted = 0, consumed = 0;
    while (consumed < n_chunks) {
      // Keep the ring full. A slot is reused only after the copy out of its previous chunk finished.
      while (submitted < n_chunks && submitted - consumed < static_cast<uint64_t>(kRingSlots)) {
        Slot& s = slots[submitted % kRingSlots];
        if (submitted >= static_cast<uint64_t>(kRingSlots)) Check(hipEventSynchronize(s.copied), "hipEventSynchronize");
        Submit(file, offset + chunk_off(submitted), chunk_len(submitted), s);
        ++submitted;
      }
      Slot& s = slots[consumed % kRingSlots];
      WaitFilled(s);
      const hipError_t e = hipMemcpyAsync(static_cast<char*>(dst) + chunk_off(consumed), s.ptr, chunk_len(consumed),
                                          hipMemcpyDeviceToDevice, stream);
      if (e != hipSuccess) Fail(std::string("hipMemcpyAsync: ") + hipGetErrorString(e));
      Check(hipEventRecord(s.copied, stream), "hipEventRecord");
      ++consumed;
    }
    const hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess) Fail(std::string("hipStreamSynchronize: ") + hipGetErrorString(e));
    total_bytes += bytes;
    total_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
  }
};

DirectStorageLoader::DirectStorageLoader() : impl_(std::make_unique<Impl>()) {}
DirectStorageLoader::~DirectStorageLoader() = default;

std::unique_ptr<DirectStorageLoader> DirectStorageLoader::Create(std::string* why_not) {
  std::unique_ptr<DirectStorageLoader> l(new DirectStorageLoader());
  const std::string why = l->impl_->Init();
  if (!why.empty()) {
    if (why_not) *why_not = why;
    return nullptr;
  }
  return l;
}

bool DirectStorageLoader::Read(const std::wstring& path, uint64_t offset, uint64_t bytes, void* dst) {
  return impl_->Read(path, offset, bytes, dst);
}
uint64_t DirectStorageLoader::bytes_read() const { return impl_->total_bytes; }
double DirectStorageLoader::read_seconds() const { return impl_->total_seconds; }

namespace {
thread_local std::unique_ptr<DirectStorageLoader> t_loader;
// Creating and destroying a loader (a D3D12 device, a DirectStorage queue, imported buffers) from two rank threads at once crashed inside
// dstoragecore.dll (a short --tp 2 load of a tiny container: one rank's scope ended while the other was still starting). The loaders' READS stay
// concurrent; only their creation and destruction take turns.
std::mutex g_lifecycle_mu;

bool Disabled() {
  char v[16] = {};
  if (GetEnvironmentVariableA("R4DX_DIRECT_STORAGE", v, sizeof v) == 0) return false;
  return !std::strcmp(v, "0") || !std::strcmp(v, "off") || !std::strcmp(v, "false");
}
}  // namespace

DirectLoadScope::DirectLoadScope() {
  if (t_loader || Disabled()) return;
  std::string why;
  std::lock_guard<std::mutex> lifecycle(g_lifecycle_mu);
  t_loader = DirectStorageLoader::Create(&why);
  if (!t_loader) {
    std::fprintf(stderr, "r4dx: DirectStorage unavailable (%s) -- loading through the memory mapping\n", why.c_str());
    return;
  }
  owner_ = true;
}

DirectLoadScope::~DirectLoadScope() {
  if (!owner_) return;
  if (t_loader->bytes_read() > 0) {
    const double gib = static_cast<double>(t_loader->bytes_read()) / (1ull << 30);
    const double s = t_loader->read_seconds();
    std::fprintf(stderr, "r4dx: DirectStorage read %.2f GiB straight to VRAM in %.2f s (%.2f GB/s)\n", gib, s,
                 s > 0 ? gib * 1.073741824 / s : 0.0);
  }
  std::lock_guard<std::mutex> lifecycle(g_lifecycle_mu);
  t_loader.reset();
}

bool DirectLoadScope::TryRead(const std::wstring& path, uint64_t offset, uint64_t bytes, void* dst) {
  if (!t_loader || bytes < kMinBytes) return false;
  return t_loader->Read(path, offset, bytes, dst);
}

}  // namespace r4dx::core
