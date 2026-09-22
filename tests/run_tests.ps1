#requires -Version 5.1
<#
.SYNOPSIS
  Run r4dx's ctest suite on HIP device 1.

.DESCRIPTION
  Only HIP device 1 (of the two R9700s in this machine) may be used -- see the GPU rule in
  README.md / docs/build-windows.md. Sets HIP_VISIBLE_DEVICES=1 (so device index 0 inside the
  test process is physical device 1) and runs ctest against the 'win-hip' preset's build
  directory with the venv's CMake (or the one on PATH when the venv is absent). No other
  environment variable is needed: the tests pick the containers matching the build's w4a16 group
  themselves (tests/model/test_container_path.h).

.PARAMETER Preset
  CMake preset name. Default 'win-hip'.

.EXAMPLE
  .\tests\run_tests.ps1
#>
[CmdletBinding()]
param(
    [string]$Preset = "win-hip"
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

$env:HIP_VISIBLE_DEVICES = "1"

Write-Output "[run_tests] ctest (preset $Preset), HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES)"
& $Cmake --build --preset $Preset --target smoke_r4d
if ($LASTEXITCODE -ne 0) { throw "build of smoke_r4d failed with exit code $LASTEXITCODE" }

$Ctest = Join-Path (Split-Path $Cmake) "ctest.exe"
& $Ctest --preset $Preset --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "ctest failed with exit code $LASTEXITCODE" }

Write-Output "[run_tests] all tests passed"
