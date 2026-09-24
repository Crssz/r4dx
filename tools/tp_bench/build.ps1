# tools/tp_bench/build.ps1 -- builds the TP=2 all-reduce benchmark (ar_bench.hip) directly via hipcc.exe.
#
# Same approach and flags as tools/bench/build.ps1: the root CMakeLists.txt does not
# add_subdirectory(tools), so this drives hipcc.exe directly. Output goes to tools/tp_bench/build/
# (in-tree like tools/bench/build/; the repo-wide `build/` rule in .gitignore already ignores it).
# Compiling is CPU-only; this script never runs the benchmark.
#
# Usage: powershell -File tools\tp_bench\build.ps1 [-Asm]
#   -Asm   also emit the gfx1201 device assembly (build\ar_bench-gfx1201.s) to inspect the protocol
#          kernels' memory instructions (scope:SCOPE_SYS stores/loads, global_wb/global_inv, s_sleep).

param([switch]$Asm)

$ErrorActionPreference = 'Stop'

$RocmBin = 'C:\opt\rocm\bin'
$RocmLlvmBin = 'C:\opt\rocm\lib\llvm\bin'
$Hipcc = Join-Path $RocmBin 'hipcc.exe'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Here 'build'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:PATH = "$RocmBin;$RocmLlvmBin;$env:PATH"

$CommonFlags = @(
    '-O3', '-std=c++20', '--offload-arch=gfx1201',
    '-Wno-unused-result',
    '--rocm-path=C:/opt/rocm', '--rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode',
    '-DNDEBUG', '-D_DLL', '-D_MT', '-Xclang', '--dependent-lib=msvcrt'
)

function Invoke-Hipcc {
    param([string[]]$HipArgs)
    # Run via cmd /c so stdout+stderr merge inside cmd's own redirection (avoids PowerShell 5.1
    # wrapping a native exe's stderr into NativeCommandError / $?=false even on success).
    $quotedArgs = ($HipArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $cmdLine = '"' + $Hipcc + '" ' + $quotedArgs + ' 2>&1'
    $out = & cmd /c $cmdLine
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host "    $_" }
    return $code
}

$src = Join-Path $Here 'ar_bench.hip'
$exe = Join-Path $OutDir 'ar_bench.exe'
Write-Host "==> building ar_bench.hip -> build\ar_bench.exe"
$code = Invoke-Hipcc ($CommonFlags + @($src, '-o', $exe))
if ($code -ne 0) {
    Write-Host "    FAILED (exit $code)" -ForegroundColor Red
    exit 1
}
Write-Host "    OK"

if ($Asm) {
    $asmOut = Join-Path $OutDir 'ar_bench-gfx1201.s'
    Write-Host "==> device assembly -> build\ar_bench-gfx1201.s"
    $code = Invoke-Hipcc ($CommonFlags + @('--cuda-device-only', '-S', $src, '-o', $asmOut))
    if ($code -ne 0) {
        Write-Host "    FAILED (exit $code)" -ForegroundColor Red
        exit 1
    }
    Write-Host "    OK"
}
exit 0
