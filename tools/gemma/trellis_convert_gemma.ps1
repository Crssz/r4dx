# Gemma 4 rotated trellis container (docs/gemma4-plan.md M1-28, 9.12, 9.13, 10): r4dx-convert --trellis-from a
# mix4.5m oracle directory made by tools\gemma\trellis_oracle_gemma.ps1. The Gemma twin of
# tools\quant2\trellis_convert.ps1 (Qwen-only, unchanged). CPU only (the converter's reconstruction check is
# --trellis-verify full); it never launches a GPU kernel.
#
#   .\tools\gemma\trellis_convert_gemma.ps1 -Oracle D:\models\r4dx\huihui-gemma\trellis-q\mix4.5m `
#       -Output D:\models\r4dx\huihui-gemma\trellis-mix45m.r4dx
#
# -Rotate / -RotationSeed / -RotationFile MUST be the ones the oracle was quantized with: the converter
# refuses a manifest whose rotation fingerprint (kind, seed, tensors_sha256) differs, and a rotated manifest
# without --rotate. The manifest pin (--trellis-manifest-sha256) is the sha256 of the oracle's
# weights_override.json as it is NOW (Gemma has no weights-override KL run to take it from; the quantized
# container's KL is measured on the container itself), and is printed and logged.
#
# lm_head: w4a16 g32 LDLQ'd against hessian-v1's lm_head.hess (the tied head is the embedding table; plan 10.1).
param(
  [Parameter(Mandatory = $true)][string]$Oracle,
  [Parameter(Mandatory = $true)][string]$Output,
  [string]$Checkpoint = 'D:\models\Huihui-gemma-4-12B-it-abliterated',
  [string]$HessianDir = 'D:\models\r4dx\huihui-gemma\hessian-v1',
  [string]$KvCalib = 'D:\models\r4dx\huihui-gemma\kvcalib.json',
  [string]$RotationFile = 'D:\models\r4dx\huihui-gemma\rotation-q2ab.safetensors',
  [string]$Rotate = 'q2ab',
  [int]$RotationSeed = 1,
  [string]$LogDir = 'D:\models\r4dx\huihui-gemma\logs',
  [string]$Exe = '',
  [int]$Threads = 32,
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' })
)
$ErrorActionPreference = 'Stop'
$PSDefaultParameterValues = @{ 'Out-File:Encoding' = 'utf8' }
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
  foreach ($c in 'build\win-hip-m3\src\convert\r4dx-convert.exe', 'build\win-hip\src\convert\r4dx-convert.exe') {
    if (Test-Path (Join-Path $repo $c)) { $Exe = Join-Path $repo $c; break }
  }
}
if (-not $Exe -or -not (Test-Path $Exe)) { throw "[gemma-trellis] no converter; pass -Exe" }
$manifest = Join-Path $Oracle 'weights_override.json'
if (-not (Test-Path $manifest)) { throw "[gemma-trellis] no weights_override.json in $Oracle" }
foreach ($f in $KvCalib, $RotationFile, (Join-Path $HessianDir 'hessian.json')) {
  if (-not (Test-Path $f)) { throw "[gemma-trellis] missing $f" }
}
$sha = (Get-FileHash $manifest -Algorithm SHA256).Hash.ToLower()
New-Item -ItemType Directory -Force $LogDir | Out-Null
$log = Join-Path $LogDir ([IO.Path]::GetFileNameWithoutExtension($Output) + '.log')
Write-Host "[gemma-trellis] manifest pin $sha ($manifest)"

$cargs = @('--input', $Checkpoint, '--output', $Output,
           '--rotate', $Rotate, '--rotation-seed', "$RotationSeed",
           '--trellis-from', $Oracle, '--trellis-manifest-sha256', $sha, '--trellis-verify', 'full',
           '--kv-calib', $KvCalib, '--no-bf16', '--lm-head', 'w4a16', '--w4a16-group-rule', '^lm_head$=32',
           '--hessian-dir', $HessianDir, '--ldlq', '^lm_head$', '--threads', "$Threads")
Write-Host "[gemma-trellis] $Exe $($cargs -join ' ')"
$t0 = Get-Date
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
try { & $Exe @cargs *> $log } finally { $ErrorActionPreference = $prev }
$rc = $LASTEXITCODE
Add-Content -Path $log -Encoding utf8 -Value "[trellis_convert_gemma.ps1] exit $rc, manifest pin $sha"
if ($rc -ne 0) { throw "[gemma-trellis] r4dx-convert failed (exit $rc), see $log" }
Write-Host ("[gemma-trellis] converted in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
Select-String -Path $log -Pattern 'trellis verify: \d+/\d+|wrote ' | ForEach-Object { Write-Host "  $($_.Line)" }
