# tools/bench/build.ps1 -- builds the r9700-microbench HIP programs directly via hipcc.exe.
#
# The repo root CMakeLists.txt (C:\Users\user\dev\r4dx\CMakeLists.txt) does NOT
# add_subdirectory(tools) today, and this benchmark suite's own instructions say not to touch the
# root CMakeLists.txt -- so this script drives hipcc.exe directly, the same way
# third_party/CMakeLists.txt does for r4d_core (see docs/build-windows.md), instead of wiring a
# tools/CMakeLists.txt that would need that root add_subdirectory to ever be built.
#
# Usage: powershell -File tools\bench\build.ps1

$ErrorActionPreference = 'Stop'

$RocmBin = 'C:\opt\rocm\bin'
$RocmLlvmBin = 'C:\opt\rocm\lib\llvm\bin'
$Hipcc = Join-Path $RocmBin 'hipcc.exe'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Here 'build'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:PATH = "$RocmBin;$RocmLlvmBin;$env:PATH"

$CommonFlags = @(
    '-O3', '-std=c++17', '--offload-arch=gfx1201',
    '-Wno-unused-result',
    '--rocm-path=C:/opt/rocm', '--rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode',
    '-DNDEBUG', '-D_DLL', '-D_MT', '-Xclang', '--dependent-lib=msvcrt'
)

function Build-One {
    param([string]$Src, [string]$OutExe, [string[]]$ExtraFlags = @())
    $srcPath = Join-Path $Here $Src
    $outPath = Join-Path $OutDir $OutExe
    Write-Host "==> building $Src -> build\$OutExe"
    $allArgs = $CommonFlags + $ExtraFlags + @($srcPath, '-o', $outPath)
    # Run via cmd /c so stdout+stderr merge inside cmd's own redirection (avoids PowerShell 5.1
    # wrapping a native exe's stderr into NativeCommandError / $?=false even on success -- see
    # this tool's own PowerShell caveats).
    $quotedArgs = ($allArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $cmdLine = '"' + $Hipcc + '" ' + $quotedArgs + ' 2>&1'
    $buildOutput = & cmd /c $cmdLine
    $code = $LASTEXITCODE
    $buildOutput | ForEach-Object { Write-Host "    $_" }
    if ($code -ne 0) {
        Write-Host "    FAILED (exit $code)" -ForegroundColor Red
    } else {
        Write-Host "    OK"
    }
    return @{ Success = ($code -eq 0); Output = ($buildOutput -join "`n") }
}

$results = @{}

$results['bench_bandwidth'] = Build-One -Src 'bench_bandwidth.hip' -OutExe 'bench_bandwidth.exe'
$results['launch_overhead'] = Build-One -Src 'launch_overhead.hip' -OutExe 'launch_overhead.exe'
$results['lds_bandwidth'] = Build-One -Src 'lds_bandwidth.hip' -OutExe 'lds_bandwidth.exe'

# WMMA probes: one compile per WMMA_KIND, each isolated so one failure doesn't block the rest.
$wmmaKinds = @(
    @{ Id = 0; Name = 'f32_16x16x16_f16' },
    @{ Id = 1; Name = 'f32_16x16x16_bf16' },
    @{ Id = 2; Name = 'f32_16x16x16_fp8_fp8' },
    @{ Id = 3; Name = 'f32_16x16x16_bf8_bf8' },
    @{ Id = 4; Name = 'i32_16x16x16_iu8' },
    @{ Id = 5; Name = 'i32_16x16x16_iu4' },
    @{ Id = 6; Name = 'i32_16x16x32_iu4' }
)

$wmmaResults = @{}
foreach ($k in $wmmaKinds) {
    $exe = "wmma_probe_$($k.Name).exe"
    $r = Build-One -Src 'wmma_probe.hip' -OutExe $exe -ExtraFlags @("-DWMMA_KIND=$($k.Id)")
    $wmmaResults[$k.Name] = $r
}
$results['wmma_probes'] = $wmmaResults

$results | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $OutDir 'build_report.json') -Encoding utf8
Write-Host "`nBuild report written to build\build_report.json"
