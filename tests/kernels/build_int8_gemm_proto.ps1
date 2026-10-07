# tests/kernels/build_int8_gemm_proto.ps1 -- builds tool_int8_gemm_proto.exe with hipcc, outside CMake (the same way
# build_m256_bench.ps1 builds the M = 256 bench): the shipped M = 256 trellis unit (the f16 baseline of the
# comparison) is compiled from libr4d's source with third_party/CMakeLists.txt's R4D_BASE_FLAGS, the bench with the
# same flags, and the two are linked.
#   powershell -File tests\kernels\build_int8_gemm_proto.ps1 [-Out <R4DX_MODELS_ROOT>\r4dx\int8gemm\obj] [-Lite] [-Isa]
#     -Lite  instantiates only skw 4 (a fast compile for a smoke test; the bench then sweeps only that)
#     -Isa   also writes the bench's device listing (hipcc -S) next to the exe, prints each kernel's VGPRs / scratch
#            and counts of v_wmma_i32_16x16x16_iu8, v_wmma_f32_16x16x16_f16 and the packed-f16 quantizer ops
param([string]$Out = (Join-Path $(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { "E:\models" }) "r4dx\int8gemm\obj"),
      [string]$Rocm = "C:/opt/rocm", [switch]$Lite, [switch]$Isa, [string[]]$Define = @())
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force $Out | Out-Null
$env:PATH = "$Rocm/bin;$Rocm/lib/llvm/bin;$env:PATH"
$inc = Join-Path $root "third_party/libr4d"
$flags = $Define + @("-O3", "-std=c++17", "--offload-arch=gfx1201", "-Wno-unused-result", "-ffp-contract=off",
           "--rocm-path=$Rocm", "--rocm-device-lib-path=$Rocm/lib/llvm/amdgcn/bitcode", "-DNDEBUG",
           "-D_DLL", "-D_MT", "-D_CRT_SECURE_NO_WARNINGS", "-Xclang", "--dependent-lib=msvcrt",
           "-I", $inc, "-I", $PSScriptRoot, "-I", (Join-Path $root "src/core/include"))
if ($Lite) { $flags += "-DI8G_LITE" }
function Compile($src, $obj, $extra = @()) {
  & "$Rocm/bin/hipcc.exe" @flags @extra -c $src -o $obj
  if ($LASTEXITCODE -ne 0) { throw "hipcc failed for $src" }
}
$m256 = Join-Path $Out "m256.obj"
$m256src = Join-Path $inc "r4d_gemm_trellis_nt_m256.hip"
if (-not (Test-Path $m256) -or (Get-Item $m256).LastWriteTime -lt (Get-Item $m256src).LastWriteTime) { Compile $m256src $m256 }
$exe = "tool_int8_gemm_proto.exe"
$bench = Join-Path $PSScriptRoot "tool_int8_gemm_proto.hip"
$benchobj = Join-Path $Out ("tool_int8_gemm_proto" + $(if ($Lite) { "_lite" } else { "" }) + ".obj")
Compile $bench $benchobj
& "$Rocm/bin/hipcc.exe" $benchobj $m256 "--rocm-path=$Rocm" "-o" (Join-Path $Out $exe)
if ($LASTEXITCODE -ne 0) { throw "link failed" }
Write-Host "built $(Join-Path $Out $exe)"
if ($Isa) {
  $isaPath = Join-Path $Out "tool_int8_gemm_proto-gfx1201.s"
  & "$Rocm/bin/hipcc.exe" @flags --cuda-device-only -S $bench -o $isaPath
  if ($LASTEXITCODE -ne 0) { throw "hipcc -S failed" }
  $lines = Get-Content $isaPath
  $cur = $null; $counts = @{}; $meta = @{}; $priv = 0; $lastsym = $null
  foreach ($l in $lines) {
    if ($l -match '^(_Z[A-Za-z0-9_$.]+):\s*(;.*)?$') { $cur = $Matches[1]; $counts[$cur] = @{iu8 = 0; f16 = 0; pkfma = 0; valu = 0; ds = 0; mem = 0; sbar = 0}; continue }
    if ($cur) {
      if ($l -match '^\s+v_wmma_i32_16x16x16_iu8') { $counts[$cur].iu8++ }
      elseif ($l -match '^\s+v_wmma_f32_16x16x16_f16') { $counts[$cur].f16++ }
      elseif ($l -match '^\s+v_pk_fma_f16') { $counts[$cur].pkfma++; $counts[$cur].valu++ }
      elseif ($l -match '^\s+v_') { $counts[$cur].valu++ }
      elseif ($l -match '^\s+s_barrier') { $counts[$cur].sbar++ }
      elseif ($l -match '^\s+ds_') { $counts[$cur].ds++ }
      elseif ($l -match '^\s+(global|flat|buffer)_') { $counts[$cur].mem++ }
    }
    if ($l -match '^\s+\.private_segment_fixed_size:\s+(\d+)') { $priv = [int]$Matches[1] }
    if ($l -match '^\s+\.symbol:\s+(\S+)\.kd') { $lastsym = $Matches[1]; $meta[$lastsym] = @{scratch = $priv; vgpr = -1} }
    if ($lastsym -and $l -match '^\s+\.vgpr_count:\s+(\d+)') { $meta[$lastsym].vgpr = [int]$Matches[1]; $lastsym = $null }
  }
  Write-Host "kernel (mangled; i8g_kernel<TRELLIS, KB, FWHT, RESC, SKW>)        VGPRs scratch  iu8-wmma f16-wmma pk_fma_f16 VALU ds mem s_barrier"
  foreach ($k in $meta.Keys | Sort-Object) {
    $c = $counts[$k]
    Write-Host ("{0,-60} {1,4} {2,5}  {3,5} {4,5} {5,5} {6,6} {7,3} {8,4} {9,3}" -f $k.Substring(0, [Math]::Min(60, $k.Length)), $meta[$k].vgpr, $meta[$k].scratch, $c.iu8, $c.f16, $c.pkfma, $c.valu, $c.ds, $c.mem, $c.sbar)
  }
  Write-Host "listing: $isaPath"
}
