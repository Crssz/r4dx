// tests/core/test_direct_storage.cpp -- DirectStorageLoader / DirectLoadScope (src/core/direct_storage.cpp,
// docs/direct-storage.md): file bytes land in VRAM exactly, including unaligned offsets, lengths that
// are not a multiple of the chunk, and ranges long enough to reuse every staging-ring slot; the scope's
// refusal rules; a read past the end of the file throws and poisons the loader. Exit 77 (skip) when
// there is no HIP device or DirectStorage cannot start here.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/direct_storage.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;

void Check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what);
  }
}

std::vector<uint8_t> WriteRandomFile(const std::wstring& path, size_t bytes) {
  std::vector<uint8_t> data(bytes);
  std::mt19937_64 rng(0xD57u);
  for (size_t i = 0; i + 8 <= bytes; i += 8) {
    const uint64_t v = rng();
    std::memcpy(data.data() + i, &v, 8);
  }
  FILE* f = _wfopen(path.c_str(), L"wb");
  if (!f) throw std::runtime_error("cannot create the temp file");
  std::fwrite(data.data(), 1, bytes, f);
  std::fclose(f);
  return data;
}

bool DeviceMatches(const DeviceBuffer<uint8_t>& dev, const uint8_t* expect, size_t n) {
  std::vector<uint8_t> back(n);
  dev.CopyToHost(back.data(), n);
  return std::memcmp(back.data(), expect, n) == 0;
}

}  // namespace

int Run() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  std::string why;
  std::unique_ptr<DirectStorageLoader> loader = DirectStorageLoader::Create(&why);
  if (!loader) {
    std::printf("SKIP: DirectStorage cannot start here: %s\n", why.c_str());
    return 77;
  }

  // 168 MiB: more than kRingSlots chunks (every slot is refilled after its copy-out) and room for the
  // five-chunk range at an odd offset below.
  const size_t kFile = 168ull << 20;
  wchar_t tmp[MAX_PATH];
  GetTempPathW(MAX_PATH, tmp);
  const std::wstring path = std::wstring(tmp) + L"r4dx_test_direct_storage.bin";
  const std::vector<uint8_t> data = WriteRandomFile(path, kFile);

  {
    DeviceBuffer<uint8_t> dev(kFile);
    Check(loader->Read(path, 0, kFile, dev.data()), "whole file read is accepted");
    Check(DeviceMatches(dev, data.data(), kFile), "whole file bytes match");
  }
  {
    // Odd offset, length crossing five chunks and ending mid-chunk.
    const uint64_t off = 4097, len = 5 * DirectStorageLoader::kChunkBytes + 13;
    DeviceBuffer<uint8_t> dev(len);
    Check(loader->Read(path, off, len, dev.data()), "unaligned range is accepted");
    Check(DeviceMatches(dev, data.data() + off, len), "unaligned range bytes match (ring slots reused)");
  }
  {
    const uint64_t off = kFile - 3000, len = 3000;
    DeviceBuffer<uint8_t> dev(len);
    Check(loader->Read(path, off, len, dev.data()), "short tail read is accepted");
    Check(DeviceMatches(dev, data.data() + off, len), "short tail bytes match");
  }
  Check(loader->bytes_read() == kFile + 5 * DirectStorageLoader::kChunkBytes + 13 + 3000, "bytes_read counts every read");
  Check(!loader->Read(L"Z:\\no\\such\\dir\\x.bin", 0, 4096, nullptr), "a file DirectStorage cannot open is refused, not thrown");

  loader.reset();

  {
    DeviceBuffer<uint8_t> dev(DirectLoadScope::kMinBytes);
    Check(!DirectLoadScope::TryRead(path, 0, DirectLoadScope::kMinBytes, dev.data()), "no scope: TryRead refuses");
    DirectLoadScope scope;
    Check(DirectLoadScope::TryRead(path, 123, DirectLoadScope::kMinBytes, dev.data()), "scope: a tensor at the threshold is read");
    Check(DeviceMatches(dev, data.data() + 123, DirectLoadScope::kMinBytes), "scope: bytes match");
    {
      DirectLoadScope nested;
      Check(DirectLoadScope::TryRead(path, 0, DirectLoadScope::kMinBytes, dev.data()), "nested scope reuses the outer loader");
    }
    Check(DirectLoadScope::TryRead(path, 0, DirectLoadScope::kMinBytes, dev.data()), "outer scope survives the nested one");
    Check(!DirectLoadScope::TryRead(path, 0, DirectLoadScope::kMinBytes - 1, dev.data()), "below the threshold: TryRead refuses");
  }
  {
    DeviceBuffer<uint8_t> dev(DirectLoadScope::kMinBytes);
    Check(!DirectLoadScope::TryRead(path, 0, DirectLoadScope::kMinBytes, dev.data()), "scope destroyed: TryRead refuses again");
  }

  {
    auto l = DirectStorageLoader::Create(&why);
    DeviceBuffer<uint8_t> dev(8u << 20);
    bool threw = false;
    try {
      l->Read(path, kFile - 1000, 8u << 20, dev.data());  // runs past the end of the file
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "a read past the end of the file throws");
    threw = false;
    try {
      l->Read(path, 0, 4096, dev.data());
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "the loader is unusable after a failed read");
  }

  DeleteFileW(path.c_str());
  std::printf(g_failures == 0 ? "ALL PASS\n" : "%d FAILED\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}

int main() {
  try {
    return Run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
    return 1;
  }
}
