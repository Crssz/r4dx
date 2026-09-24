#requires -Version 5.1
<#
.SYNOPSIS
  Run r4dx's ctest suite on HIP device 1 (default), or the opt-in two-GPU tests (-TwoGpu).

.DESCRIPTION
  Only HIP device 1 (of the two R9700s in this machine) may be used by the default suite -- see the
  GPU rule in README.md / docs/build-windows.md. Sets HIP_VISIBLE_DEVICES=1 (so device index 0
  inside the test process is physical device 1) and runs ctest against the 'win-hip' preset's build
  directory with the venv's CMake (or the one on PATH when the venv is absent). No other
  environment variable is needed: the tests pick the containers matching the build's w4a16 group
  themselves (tests/model/test_container_path.h).

  The default run excludes the tensor-parallel two-GPU tests (ctest -LE tp2gpu) and makes sure
  R4DX_TP2GPU is not set, so they SKIP even if selected some other way (docs/tp.md 10.1).

  -TwoGpu runs ONLY the tp2gpu-labelled tests, on both GPUs (the user authorized TP work on both
  cards): it refuses to start while r4dx-server is running, removes HIP_VISIBLE_DEVICES from the
  process environment, sets R4DX_TP2GPU=1 and runs `ctest --test-dir build\<preset> -L tp2gpu`
  -- deliberately not `--preset`, whose test environment would inject HIP_VISIBLE_DEVICES=1 into
  every test. Both variables are restored afterwards.

.PARAMETER Preset
  CMake preset name. Default 'win-hip'.

.PARAMETER TwoGpu
  Run only the two-GPU (tp2gpu) tests, on both GPUs.

.EXAMPLE
  .\tests\run_tests.ps1
  .\tests\run_tests.ps1 -TwoGpu
#>
[CmdletBinding()]
param(
    [string]$Preset = "win-hip",
    [switch]$TwoGpu
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$VenvRoot = if ($env:R4DX_REFERENCE_VENV) { $env:R4DX_REFERENCE_VENV }
            else { Join-Path $env:USERPROFILE "dev\vLLM_for_AMD\.venv-rocm10" }
$Cmake = Join-Path $VenvRoot "Scripts\cmake.exe"
# Same fallback as build.ps1: the reference venv is a machine-local environment that can be absent;
# the cmake/ctest on PATH run this preset just as well.
if (-not (Test-Path $Cmake)) {
    $PathCmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if (-not $PathCmake) { throw "cmake.exe not found in $VenvRoot\Scripts nor on PATH" }
    $Cmake = $PathCmake.Source
}
$Ctest = Join-Path (Split-Path $Cmake) "ctest.exe"

if ($TwoGpu) {
    if (Get-Process r4dx-server -ErrorAction SilentlyContinue) {
        throw "stop the production server first (r4dx-server is running; the two-GPU tests use both cards)"
    }
    $SavedHip = $env:HIP_VISIBLE_DEVICES
    $SavedOptIn = $env:R4DX_TP2GPU
    try {
        Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $env:R4DX_TP2GPU = "1"
        $TestDir = Join-Path "build" $Preset
        Write-Output "[run_tests] two-GPU tests: ctest --test-dir $TestDir -L tp2gpu (HIP_VISIBLE_DEVICES unset, R4DX_TP2GPU=1)"
        & $Ctest --test-dir $TestDir -L tp2gpu --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw "ctest (tp2gpu) failed with exit code $LASTEXITCODE" }
    } finally {
        if ($null -ne $SavedHip) { $env:HIP_VISIBLE_DEVICES = $SavedHip }
        else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
        if ($null -ne $SavedOptIn) { $env:R4DX_TP2GPU = $SavedOptIn }
        else { Remove-Item env:R4DX_TP2GPU -ErrorAction SilentlyContinue }
    }
    Write-Output "[run_tests] all two-GPU tests passed"
    return
}

$env:HIP_VISIBLE_DEVICES = "1"
# The two-GPU tests are opt-in only (docs/tp.md 10.1): never let a stray R4DX_TP2GPU reach them.
Remove-Item env:R4DX_TP2GPU -ErrorAction SilentlyContinue

Write-Output "[run_tests] ctest (preset $Preset, -LE tp2gpu), HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES)"
& $Cmake --build --preset $Preset --target smoke_r4d
if ($LASTEXITCODE -ne 0) { throw "build of smoke_r4d failed with exit code $LASTEXITCODE" }

& $Ctest --preset $Preset --output-on-failure -LE tp2gpu
if ($LASTEXITCODE -ne 0) { throw "ctest failed with exit code $LASTEXITCODE" }

Write-Output "[run_tests] all tests passed"
