#requires -Version 5.1
<#
.SYNOPSIS
  Configure + build r4dx with the 'win-hip' CMake preset.

.DESCRIPTION
  Uses the vLLM_for_AMD venv's CMake 4.4.2 / Ninja (the PATH cmake is 3.31, which cannot
  configure clang-cl builds the way this project needs) against the ROCm SDK at C:\opt\rocm.
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

$VenvScripts = "C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts"
$Cmake = Join-Path $VenvScripts "cmake.exe"
$Ninja = Join-Path $VenvScripts "ninja.exe"

foreach ($tool in @($Cmake, $Ninja)) {
    if (-not (Test-Path $tool)) { throw "required tool not found: $tool" }
}

# So CMake's Ninja generator resolves `ninja` to the venv's 1.13 build rather than any older copy
# earlier on PATH.
$env:PATH = "$VenvScripts;$env:PATH"

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
