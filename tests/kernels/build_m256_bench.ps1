# tests/kernels/build_m256_bench.ps1 -- builds tool_trellis_m256_bench.exe with hipcc, outside CMake:
# the shipped M = 64 trellis kernel unit and the prototype M = 256 unit are compiled from libr4d's
# sources with third_party/CMakeLists.txt's R4D_BASE_FLAGS (so the M = 64 side is the object r4d_core
# holds, flag for flag), the bench with the same flags, and the three linked.
#   powershell -File tests\kernels\build_m256_bench.ps1 [-Out D:\models\r4dx\linear\A2\obj]
param([string]$Out = "D:\models\r4dx\linear\A2\obj", [string]$Rocm = "C:/opt/rocm")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force $Out | Out-Null
$env:PATH = "$Rocm/bin;$Rocm/lib/llvm/bin;$env:PATH"
$inc = Join-Path $root "third_party/libr4d"
$flags = @("-O3", "-std=c++17", "--offload-arch=gfx1201", "-Wno-unused-result", "-ffp-contract=off",
           "--rocm-path=$Rocm", "--rocm-device-lib-path=$Rocm/lib/llvm/amdgcn/bitcode", "-DNDEBUG",
           "-D_DLL", "-D_MT", "-Xclang", "--dependent-lib=msvcrt", "-I", $inc)
function Compile($src, $obj) {
  & "$Rocm/bin/hipcc.exe" @flags -c $src -o $obj
  if ($LASTEXITCODE -ne 0) { throw "hipcc failed for $src" }
}
Compile (Join-Path $inc "r4d_gemm_trellis_nt_m64.hip") (Join-Path $Out "m64.obj")
Compile (Join-Path $inc "r4d_gemm_trellis_nt_m256.hip") (Join-Path $Out "m256.obj")
Compile (Join-Path $PSScriptRoot "tool_trellis_m256_bench.hip") (Join-Path $Out "bench.obj")
& "$Rocm/bin/hipcc.exe" (Join-Path $Out "bench.obj") (Join-Path $Out "m64.obj") (Join-Path $Out "m256.obj") `
  "--rocm-path=$Rocm" "-o" (Join-Path $Out "tool_trellis_m256_bench.exe")
if ($LASTEXITCODE -ne 0) { throw "link failed" }
Write-Host "built $(Join-Path $Out 'tool_trellis_m256_bench.exe')"
