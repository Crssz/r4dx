#requires -Version 5.1
<#
.SYNOPSIS
  Configure + build r4dx with the 'win-hip' CMake preset.

.DESCRIPTION
  Uses the vLLM_for_AMD venv's CMake 4.4.2 / Ninja when that venv exists, otherwise the cmake/ninja
  on PATH (CMake 3.31 + Ninja from PATH configure and build the 'win-hip' preset too -- verified
  2026-09-22 once the venv had been deleted from this machine), against the ROCm SDK at C:\opt\rocm.
  r4d_core's 15 libr4d translation units are compiled by hipcc.exe directly (see
  third_party/CMakeLists.txt); everything else is plain clang-cl C++.

.PARAMETER Preset
  CMake preset name. Default 'win-hip'.

.PARAMETER Clean
  Delete the preset's build directory first.

.EXAMPLE
  .\build.ps1
  .\build.ps1 -Clean
#>
[CmdletBinding()]
param(
    [string]$Preset = "win-hip",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Derived from the environment rather than hardcoded, so no local account name is baked into the
# repo. Override with $env:R4DX_REFERENCE_VENV if the reference venv lives elsewhere.
$VenvRoot = if ($env:R4DX_REFERENCE_VENV) { $env:R4DX_REFERENCE_VENV }
            else { Join-Path $env:USERPROFILE "dev\vLLM_for_AMD\.venv-rocm10" }
$VenvScripts = Join-Path $VenvRoot "Scripts"
$Cmake = Join-Path $VenvScripts "cmake.exe"
$Ninja = Join-Path $VenvScripts "ninja.exe"

# The reference venv is not a hard requirement -- fall back to whatever cmake/ninja are on PATH when
# it is absent (it is a machine-local Python environment that can be deleted or recreated
# independently of this repo). CMake 3.31 + Ninja 1.x from PATH configure and build this preset
# correctly; the venv copies are simply the newer ones the project was originally developed against.
if (-not (Test-Path $Cmake) -or -not (Test-Path $Ninja)) {
    $PathCmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
    $PathNinja = (Get-Command ninja.exe -ErrorAction SilentlyContinue)
    if (-not $PathCmake) { throw "cmake.exe not found in $VenvScripts nor on PATH" }
    if (-not $PathNinja) { throw "ninja.exe not found in $VenvScripts nor on PATH" }
    Write-Output "[build] reference venv not found at $VenvRoot -- using PATH cmake/ninja"
    $Cmake = $PathCmake.Source
    $Ninja = $PathNinja.Source
} else {
    # So CMake's Ninja generator resolves `ninja` to the venv's 1.13 build rather than any older copy
    # earlier on PATH.
    $env:PATH = "$VenvScripts;$env:PATH"
}

$BuildDir = Join-Path $PSScriptRoot "build\$Preset"
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Output "[build] removing $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

Write-Output "[build] configure (preset $Preset)"
& $Cmake --preset $Preset -DCMAKE_MAKE_PROGRAM="$Ninja"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

Write-Output "[build] build (preset $Preset)"
& $Cmake --build --preset $Preset
if ($LASTEXITCODE -ne 0) { throw "cmake build failed with exit code $LASTEXITCODE" }

Write-Output "[build] done: $BuildDir"
