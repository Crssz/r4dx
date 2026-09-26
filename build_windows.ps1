#requires -Version 5.1
<#
.SYNOPSIS
  Build r4d.pyd (the R4D kernel library's pybind11 extension) on Windows, single GPU, gfx1201.

.DESCRIPTION
  Mirrors build.sh: compiles every translation unit in UNITS to an object file with hipcc, then
  links them all into one pybind11 extension module. Unlike build.sh this targets the MSVC ABI
  (clang-cl/lld-link under hipcc) instead of a Linux .so, and skips -fPIC (PE has no PIC concept).

.PARAMETER RocmRoot
  Root of the ROCm SDK devel tree (hipcc.exe under bin\, clang under lib\llvm\bin\, headers under
  include\hip). Defaults to the ROCm SDK vendored inside vLLM_for_AMD's .venv-rocm10, falling back
  to C:\opt\rocm if hipcc.exe is not found there.

.PARAMETER PythonRoot
  Root of the base CPython 3.12 interpreter used to build against (Python.h, libs\python312.lib).

.PARAMETER Pybind11Include
  Directory containing pybind11\pybind11.h. Defaults to the torch-vendored copy under
  .venv-rocm10 if present; otherwise a throwaway uv venv is created at
  $env:USERPROFILE\dev\libr4d-buildenv and pybind11 is installed into it with uv.

.PARAMETER OutDir
  Directory objects and r4d.pyd are written to. Default build-win.

.PARAMETER Jobs
  Max concurrent hipcc compiles. Default 6 (the vLLM_for_AMD build may also be using the CPU).

.EXAMPLE
  .\build_windows.ps1
  .\build_windows.ps1 -Jobs 4 -OutDir build-win2
#>
[CmdletBinding()]
param(
    [string]$RocmRoot = "",
    [string]$PythonRoot = "",
    [string]$Pybind11Include = "",
    [string]$OutDir = "build-win",
    [int]$Jobs = 6
)

$ErrorActionPreference = "Stop"

# Machine-specific locations are derived from the environment rather than hardcoded, so that no
# local account name is committed to the repository. Override $env:LIBR4D_REFERENCE_VENV, or pass
# -RocmRoot / -PythonRoot explicitly, if these live elsewhere.
$VenvRoot = if ($env:LIBR4D_REFERENCE_VENV) { $env:LIBR4D_REFERENCE_VENV }
            else { Join-Path $env:USERPROFILE "dev\vLLM_for_AMD\.venv-rocm10" }
if (-not $RocmRoot) { $RocmRoot = Join-Path $VenvRoot "Lib\site-packages\_rocm_sdk_devel" }
if (-not $PythonRoot) {
    $PythonRoot = Join-Path $env:APPDATA "uv\python\cpython-3.12.11-windows-x86_64-none"
}
Set-Location $PSScriptRoot

# ---------------------------------------------------------------------------------------------
# Resolve RocmRoot: fall back to C:\opt\rocm if hipcc.exe is not where we expect it.
# ---------------------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $RocmRoot "bin\hipcc.exe"))) {
    $fallback = "C:\opt\rocm"
    if (Test-Path (Join-Path $fallback "bin\hipcc.exe")) {
        Write-Warning "hipcc.exe not found under -RocmRoot '$RocmRoot'; falling back to '$fallback'."
        $RocmRoot = $fallback
    } else {
        throw "hipcc.exe not found under '$RocmRoot' or the fallback '$fallback'."
    }
}
$HIPCC = Join-Path $RocmRoot "bin\hipcc.exe"
Write-Output "[build_windows] RocmRoot   = $RocmRoot"
Write-Output "[build_windows] HIPCC      = $HIPCC"

# ---------------------------------------------------------------------------------------------
# Resolve PythonRoot.
# ---------------------------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $PythonRoot "include\Python.h"))) {
    throw "Python.h not found under -PythonRoot '$PythonRoot'."
}
if (-not (Test-Path (Join-Path $PythonRoot "libs\python312.lib"))) {
    throw "libs\python312.lib not found under -PythonRoot '$PythonRoot'."
}
Write-Output "[build_windows] PythonRoot = $PythonRoot"

# ---------------------------------------------------------------------------------------------
# Resolve Pybind11Include: prefer the torch-vendored copy inside .venv-rocm10; otherwise stand up
# a throwaway uv venv and install pybind11 into it.
# ---------------------------------------------------------------------------------------------
if (-not $Pybind11Include) {
    $torchInc = Join-Path $VenvRoot "Lib\site-packages\torch\include"
    if (Test-Path (Join-Path $torchInc "pybind11\pybind11.h")) {
        $Pybind11Include = $torchInc
        Write-Output "[build_windows] pybind11 include (torch-vendored): $Pybind11Include"
    } else {
        $buildEnvDir = Join-Path $env:USERPROFILE "dev\libr4d-buildenv"
        $uv = Join-Path $env:USERPROFILE ".hermes\bin\uv.exe"
        if (-not (Test-Path (Join-Path $buildEnvDir "Scripts\python.exe"))) {
            Write-Output "[build_windows] no pybind11 headers found; creating throwaway venv at $buildEnvDir"
            & $uv venv --python 3.12 $buildEnvDir
            if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
        }
        $buildEnvPy = Join-Path $buildEnvDir "Scripts\python.exe"
        & $uv pip install --python $buildEnvPy pybind11
        if ($LASTEXITCODE -ne 0) { throw "uv pip install pybind11 failed with exit code $LASTEXITCODE" }
        $site = & $buildEnvPy -c "import pybind11; print(pybind11.get_include())"
        if ($LASTEXITCODE -ne 0 -or -not $site) { throw "failed to resolve pybind11 include dir from throwaway venv" }
        $Pybind11Include = $site.Trim()
        Write-Output "[build_windows] pybind11 include (throwaway venv): $Pybind11Include"
    }
} else {
    if (-not (Test-Path (Join-Path $Pybind11Include "pybind11\pybind11.h"))) {
        throw "pybind11\pybind11.h not found under -Pybind11Include '$Pybind11Include'."
    }
}

# ---------------------------------------------------------------------------------------------
# Output directory + log.
# ---------------------------------------------------------------------------------------------
$OutDir = Join-Path $PSScriptRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogPath = Join-Path $OutDir "build.log"
if (Test-Path $LogPath) { Remove-Item $LogPath -Force }

function Log($msg) {
    Write-Output $msg
    Add-Content -Path $LogPath -Value $msg -Encoding utf8
}

# Windows PowerShell 5.1 runs on .NET Framework, where ProcessStartInfo.ArgumentList does not
# exist -- only the single Arguments string. Quote each token that needs it.
function Format-Args([string[]]$argv) {
    $parts = foreach ($a in $argv) {
        if ($a -match '[\s"]') {
            '"' + ($a -replace '"', '\"') + '"'
        } else {
            $a
        }
    }
    return ($parts -join ' ')
}

# Make sure hipcc can find its own clang / lld-link / device libs.
$env:PATH = "$RocmRoot\bin;$RocmRoot\lib\llvm\bin;$env:PATH"

$GFX_ARCH = "gfx1201"
$DeviceLibPath = Join-Path $RocmRoot "lib\llvm\amdgcn\bitcode"

$BASE = @(
    "-O3", "-std=c++17", "--offload-arch=$GFX_ARCH",
    "-Wno-unused-result", "-ffp-contract=off",
    "--rocm-path=$RocmRoot",
    "-DNDEBUG",
    "-I$PythonRoot\include",
    "-isystem", "$Pybind11Include"
)
if (Test-Path $DeviceLibPath) {
    $BASE += "--rocm-device-lib-path=$DeviceLibPath"
}
# python312.dll uses the dynamic CRT (msvcrt.dll family); make sure the object files agree so the
# linker does not pick the static CRT by default and collide (LNK4098) or fail to resolve symbols.
$BASE += @("-D_DLL", "-D_MT", "-Xclang", "--dependent-lib=msvcrt")

Log "[build_windows] BASE flags: $($BASE -join ' ')"

# translation unit : extra flags (mirrors build.sh's UNITS array)
$UNITS = [ordered]@{
    "r4d_attn_paged_h256_gqa6"                              = @()
    "r4d_attn_vit_h72_bf16"                                 = @()
    "r4d_gdn_chunk_scan_k128_v128_c64_bf16"                 = @("-mcumode")
    "r4d_gdn_conv_w4_h128_bf16"                              = @()
    "r4d_gdn_kkt_solve_k128_c64_bf16"                        = @()
    "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state"      = @()
    "r4d_gdn_gated_rmsnorm_h128_bf16"                        = @()
    "r4d_ar_oneshot_2rank_exact"                             = @()
    "r4d_ar_oneshot_2rank_wht6"                              = @()
    "r4d_ar_oneshot_Nrank_exact"                             = @()
    "r4d_ar_twoshot_Nrank_exact"                             = @()
    "r4d_ar_twoshot_Nrank_ti8"                                = @()
    "r4d_gemm_bf16_nt_m16"                                   = @()
    "r4d_gemm_bf16_nt_m64"                                   = @()
    "r4d_gemm_w4a16_nt_m64"                                  = @()
    "r4d_gemm_trellis_nt_m64"                                = @()
    "r4d_gemm_w4a8_nt_m64"                                   = @("-DR4D_GEMM_W4A8_GROUP=128")
    "r4d_gemm_mxfp4a8_nt_m64"                                = @()
    "r4d_quant_act_i8"                                       = @()
    "r4d_dflash_conv_t2_g16_bf16"                            = @()
    "r4d_registry"                                           = @()
    "r4d_module"                                             = @()
}

# ---------------------------------------------------------------------------------------------
# Compile, up to $Jobs concurrent hipcc processes.
# ---------------------------------------------------------------------------------------------
$procs = @{}   # Process -> unit name, for logging when it finishes
$failed = $false

function Wait-OneSlot {
    while ($procs.Count -ge $Jobs) {
        Start-Sleep -Milliseconds 200
        foreach ($p in @($procs.Keys)) {
            if ($p.HasExited) {
                Reap-Proc $p
            }
        }
    }
}

function Reap-Proc($p) {
    $unit = $procs[$p]
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    Log "----- $unit (exit $($p.ExitCode)) -----"
    if ($stdout) { Log $stdout }
    if ($stderr) { Log $stderr }
    if ($p.ExitCode -ne 0) { $script:failed = $true }
    $procs.Remove($p) | Out-Null
}

foreach ($name in $UNITS.Keys) {
    Wait-OneSlot
    $extra = $UNITS[$name]
    $src = Join-Path $PSScriptRoot "$name.hip"
    $obj = Join-Path $OutDir "$name.obj"
    $argList = @($BASE + $extra + @("-c", $src, "-o", $obj))
    Log "[hipcc] $name.obj ($GFX_ARCH)$(if ($extra) { ' ' + ($extra -join ' ') })"
    Log "  $HIPCC $($argList -join ' ')"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $HIPCC
    $psi.Arguments = Format-Args $argList
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $procs[$p] = $name
}
while ($procs.Count -gt 0) {
    Start-Sleep -Milliseconds 200
    foreach ($p in @($procs.Keys)) {
        if ($p.HasExited) { Reap-Proc $p }
    }
}

if ($failed) {
    throw "one or more hipcc compiles failed; see $LogPath"
}

# ---------------------------------------------------------------------------------------------
# Link.
# ---------------------------------------------------------------------------------------------
$objs = $UNITS.Keys | ForEach-Object { Join-Path $OutDir "$_.obj" }
$pyd = Join-Path $OutDir "r4d.pyd"
$linkArgs = @(
    "-shared", "-fuse-ld=lld", "--offload-arch=$GFX_ARCH",
    "--rocm-path=$RocmRoot",
    "-D_DLL", "-D_MT", "-Xclang", "--dependent-lib=msvcrt"
) + $objs + @(
    "$PythonRoot\libs\python312.lib",
    "-o", $pyd
)
Log "[hipcc] $pyd (link)"
Log "  $HIPCC $($linkArgs -join ' ')"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $HIPCC
$psi.Arguments = Format-Args $linkArgs
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$p = [System.Diagnostics.Process]::Start($psi)
$p.WaitForExit()
$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()
if ($stdout) { Log $stdout }
if ($stderr) { Log $stderr }
if ($p.ExitCode -ne 0) {
    throw "link failed with exit code $($p.ExitCode); see $LogPath"
}

Log "[build_windows] built $pyd"

# ---------------------------------------------------------------------------------------------
# Verify PyInit_r4d is exported. llvm-nm reads the COFF symbol table, which an optimized/stripped
# DLL does not have (it reports "no symbols" even though the PE export directory is populated);
# llvm-objdump -p reads the actual export directory, so use that instead. Non-fatal either way --
# a missing export fails the import test right after this, which is the authoritative check.
# ---------------------------------------------------------------------------------------------
$llvmObjdump = Join-Path $RocmRoot "lib\llvm\bin\llvm-objdump.exe"
if (Test-Path $llvmObjdump) {
    try {
        $dump = & $llvmObjdump -p $pyd 2>$null
    } catch {
        $dump = $null
    }
    $exports = $dump | Select-String "PyInit_r4d"
    if ($exports) {
        Log "[build_windows] PyInit_r4d exported: OK ($exports)"
    } else {
        Log "[build_windows] WARNING: PyInit_r4d not found in llvm-objdump -p output"
    }
} else {
    Log "[build_windows] llvm-objdump.exe not found at $llvmObjdump, skipping export check"
}

Write-Output "[build_windows] done. Log: $LogPath"
