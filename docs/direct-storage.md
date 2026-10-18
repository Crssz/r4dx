# DirectStorage: weights from the SSD straight to VRAM

Before: SSD -> page cache (the safetensors `MapViewOfFile`) -> pageable `hipMemcpy` -> VRAM.
Now, for every tensor of at least 1 MiB: SSD -> DirectStorage -> a D3D12 buffer in VRAM -> device-to-device copy
into the tensor's own `hipMalloc` buffer. No file byte enters the page cache or a process-owned host buffer.

## What it is, and is not

* `r4dx::core::DirectStorageLoader` (`src/core/direct_storage.cpp`, target `r4dx_direct_storage`) owns, per HIP device:
  a D3D12 device on the adapter whose LUID equals `hipDeviceGetLuid`, a DirectStorage queue, and a ring of
  `kRingSlots` = 4 shared D3D12 buffers of `kChunkBytes` = 32 MiB, each imported into HIP with
  `hipImportExternalMemory(hipExternalMemoryHandleTypeD3D12Resource)`.
* `Read(path, offset, bytes, dst)` pipelines the file in 32 MiB requests: DirectStorage fills slot *i + 1..3* while the GPU
  copies slot *i* into `dst`. The caller's destination stays an ordinary `DeviceBuffer`; nothing downstream knows.
* It is **not** peer-to-peer NVMe -> GPU DMA. On consumer GPUs DirectStorage stages through its own 64 MiB system
  buffer (`SetStagingBufferSize(2 * kChunkBytes)`); what is removed is the page cache and the mapping.
  GPUDirect Storage (cuFile) is NVIDIA-only; there is no ROCm equivalent on Windows.
* Cost while loading: 4 x 32 MiB = 128 MiB of extra VRAM per device, freed when the load returns.

Feasibility was checked, not assumed: the AMD Windows HIP runtime implements D3D12-resource import on both R9700s
(a 1 GiB DirectStorage read read back through HIP matched the file byte for byte).

## Where it hooks in

`core::DirectLoadScope` (thread-local, RAII) is the first statement of `Container::Load`, `Container::LoadShard`,
`GemmaContainer::Load`, `Lfm2Container::Load` and `Qwen4ExpContainer::Load`. Nested scopes on a thread are no-ops, so a
TP rank thread has one loader per rank, each on that rank's own device. Two seams ask it:

| seam | covers |
|---|---|
| `UploadRaw{U8,U16,F32,U32,I32}` (`container_load_util.cpp`, one template `UploadRawT`) | every TP=1 tensor: `wq`, `trellis.w`, `bf16.w`, norms, descales, embeddings that are mirrored to the device |
| `ShardLoader::Upload` (`shard_loader.h`, `ReadRunsDirect`) | every TP=2 slice: each byte run is one request, written back to back into the rank's buffer; if any run is under 1 MiB the whole tensor takes the mapping path |

`DirectLoadScope::TryRead` returns `false` (and the caller's old mapped path runs) when there is no scope, the tensor is
under `kMinBytes` = 1 MiB, or DirectStorage cannot open that file (unsupported volume). A failed read after a successful
open throws, and that loader refuses further reads.

Not covered, by design: tensors that are transformed on the host before upload (`UploadWidenedF32/F16`, rotation tables),
and the embedding table's pinned **host** copy (`Container::embed_tokens_`), which exists to be gathered on the CPU.

## Running it

* On by default. `R4DX_DIRECT_STORAGE=0` (also `off`, `false`) restores the memory-mapped path.
* If DirectStorage cannot start (no `dstorage.dll`, no D3D12 adapter for the HIP device, no import support), one stderr line
  says why and the load proceeds through the mapping: `r4dx: DirectStorage unavailable (...) -- loading through the memory mapping`.
* `dstorage.dll` is loaded at run time (`LoadLibrary`), by name first, then from the vendored copy in
  `third_party/directstorage/bin/x64`. The core DLL is preloaded from there by full path because `dstorage.dll` resolves
  `dstoragecore.dll` through the process search path, not its own directory (without that, `DStorageGetFactory` returns
  `E_NOTIMPL`). A shipped exe can instead carry both DLLs beside it.
* The staging-buffer size is a process-wide factory setting, applied once (`std::call_once`) before the first queue.
* A completed load prints `r4dx: DirectStorage read X GiB straight to VRAM in S s (G GB/s)`.

## Measured (2026-10-18, R9700 x2, E: NVMe, `huihui-qwen38-27b-abl-trellis-mix45m.r4dx`, 18.3 GB)

| run | container load | note |
|---|---|---|
| TP=1, DirectStorage | 7.67 s / 7.32 s | 13.79 GiB through DirectStorage at 3.5-3.6 GB/s; the other ~3.3 s is host work (embedding copy, widened tables, VRAM probes) |
| TP=1, `R4DX_DIRECT_STORAGE=0` | 15.92 s / 17.19 s | |
| TP=2 (real), DirectStorage | 6.64 s | 4.89 GiB per rank at 2.1 GB/s each (both ranks share one SSD) |
| TP=2 (real), `R4DX_DIRECT_STORAGE=0` | 17.45 s | |

Greedy output (`--temperature 0`, 24 tokens) is byte-identical with the switch on and off, TP=1 and TP=2; VRAM use is identical.
The mapped-path runs did not get faster on their second run (15.92 s, then 17.19 s), so the page cache was not absorbing this 18.3 GB
file here. A load whose file is fully hot in the cache (`docs/perf.md` records 5.4-5.6 s for such a case) is where DirectStorage
helps least; that case was not measured.

## Test

`tests/core/test_direct_storage.cpp` (ctest `test_direct_storage`, device 1, skip 77 without a device or DirectStorage): whole
file, an unaligned offset with a length ending mid-chunk across more chunks than ring slots (every slot refilled), a short tail,
the scope's refusal rules (no scope, under the threshold, after the scope ends, nested scope), an unopenable file refused rather
than thrown, and a read past the end of the file throwing and poisoning the loader.
