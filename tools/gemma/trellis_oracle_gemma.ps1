# Gemma 4 (gemma4_unified) mix4.5m oracle (docs/gemma4-plan.md 9.12, 9.13, 11): the ROTATED K4 and K5
# trellis oracles, then the per-tensor 4/5 mix chosen by tools/reference/gemma_mix.py from the two runs'
# Hessian proxies. The Gemma twin of tools\quant2\trellis_oracle.ps1 (which stays Qwen-only and is unchanged).
#
#   step 0  r4dx-convert --rotate q2ab --rotation-seed S --rotation-out rot.safetensors   (CPU; the fingerprint)
#   step 1  trellis_quant.py quantize-model --arch gemma4_unified --rotation rot ... --K 4   [GPU encoder]
#   step 2  the same with --K 5                                                             [GPU encoder]
#   step 3  trellis_quant.py mix --bpw 4.5 --src K4m --src K5m --out-dir mix4.5m            [CPU, seconds]
#
# Steps 1-2 run the encoder on a GPU: this script REFUSES to start them without -Gpu (device id), so a
# plain run does steps 0 and 3 only when the K4/K5 directories already exist. -DryRun runs every
# quantize-model with --dry-run (CPU: tap groups, Hessian headers, rotation folds) and skips the mix.
# Each step resumes (quantize-model skips layers it already wrote with the same code).
#
#   .\tools\gemma\trellis_oracle_gemma.ps1 -DryRun
#   .\tools\gemma\trellis_oracle_gemma.ps1 -Gpu 0           # when the user has cleared a device
#   .\tools\gemma\trellis_oracle_gemma.ps1 -MixOnly         # re-mix (e.g. -Bpw 4.4 or -Ranking prior), CPU
#
# The mix directory also gets mix_ranking.json (every group, score, final K): the audit trail of the ranking.
# The KL of the assembled container is measured with the r4dx runtime on the container (the oracle manifest
# itself has no Gemma weights-override KL driver): kl_report.py --gate gemma-fp32 --base-dir <r4dx bf16 dump>.
param(
  [string]$Checkpoint = 'D:\models\Huihui-gemma-4-12B-it-abliterated',
  [string]$HessianDir = 'D:\models\r4dx\huihui-gemma\hessian-v1',
  [string]$QDir = 'D:\models\r4dx\huihui-gemma\trellis-q',
  [string]$RotationFile = 'D:\models\r4dx\huihui-gemma\rotation-q2ab.safetensors',
  [string]$Rotate = 'q2ab',
  [int]$RotationSeed = 1,
  [double]$Bpw = 4.5,
  [ValidateSet('auto', 'exl3', 'prior', 'gemma')][string]$Ranking = 'auto',
  [string]$Out = '',                                  # default <QDir>\mix<bpw>m
  [string]$Gpu = '',                                  # HIP device id for the encoder (required for steps 1-2)
  [switch]$DryRun,
  [switch]$MixOnly,
  [string]$Exe = '',
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' })
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$env:PYTHONIOENCODING = 'utf-8'
if (-not $Out) { $Out = Join-Path $QDir ('mix{0}m' -f $Bpw.ToString([Globalization.CultureInfo]::InvariantCulture)) }
$tq = Join-Path $repo 'tools\reference\trellis_quant.py'
New-Item -ItemType Directory -Force $QDir | Out-Null

function Invoke-Native([string]$what, [scriptblock]$cmd) {
  Write-Host ("[gemma-oracle] {0:HH:mm} {1}" -f (Get-Date), $what)
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[gemma-oracle] $what failed (exit $LASTEXITCODE)" }
}

function Test-Complete([string]$dir) {
  $m = Join-Path $dir 'weights_override.json'
  if (-not (Test-Path $m)) { return $false }
  $c = & $Python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['complete'])" $m
  return ($LASTEXITCODE -eq 0 -and "$c".Trim() -eq 'True')
}

if (-not $MixOnly) {
  # step 0: the rotation file (CPU). The oracle folds with THIS file; the converter later checks its fingerprint.
  if (-not (Test-Path $RotationFile)) {
    if (-not $Exe) {
      foreach ($c in 'build\win-hip-m3\src\convert\r4dx-convert.exe', 'build\win-hip\src\convert\r4dx-convert.exe', 'build\win-hip\r4dx-convert.exe') {
        if (Test-Path (Join-Path $repo $c)) { $Exe = Join-Path $repo $c; break }
      }
    }
    if (-not $Exe -or -not (Test-Path $Exe)) { throw "[gemma-oracle] no r4dx-convert to write $RotationFile; pass -Exe or -RotationFile" }
    Invoke-Native "rotation-out $RotationFile" { & $Exe --input $Checkpoint --rotate $Rotate --rotation-seed $RotationSeed --rotation-out $RotationFile }
  }
  foreach ($k in 4, 5) {
    $dir = Join-Path $QDir "K${k}m"
    if (-not $DryRun -and (Test-Complete $dir)) { Write-Host "[gemma-oracle] K${k}m complete"; continue }
    $qargs = @('quantize-model', '--arch', 'gemma4_unified', '--model-dir', $Checkpoint, '--rotation', $RotationFile,
               '--K', "$k", '--hessian-basis', 'matched', '--hessian-dir', $HessianDir, '--out-dir', $dir)
    if ($DryRun) {
      $env:HIP_VISIBLE_DEVICES = '-1'; $env:CUDA_VISIBLE_DEVICES = '-1'
      Invoke-Native "K$k dry-run" { & $Python $tq @qargs --dry-run --device cpu }
      continue
    }
    if ($Gpu -eq '') { throw "[gemma-oracle] K$k needs the GPU encoder: pass -Gpu <device id> once a device is free (or -DryRun / -MixOnly)" }
    $env:HIP_VISIBLE_DEVICES = $Gpu
    Invoke-Native "quantize K$k on device $Gpu" { & $Python $tq @qargs --device cuda }
    if (-not (Test-Complete $dir)) { throw "[gemma-oracle] $dir incomplete after quantize-model" }
  }
}
if ($DryRun) { Write-Host '[gemma-oracle] dry-run done'; return }

# step 3: the mix (CPU only: a manifest). Never touches a GPU.
$env:HIP_VISIBLE_DEVICES = '-1'; $env:CUDA_VISIBLE_DEVICES = '-1'
Invoke-Native "mix $Bpw ($Ranking) -> $Out" { & $Python $tq mix --bpw $Bpw --ranking $Ranking --src (Join-Path $QDir 'K4m') --src (Join-Path $QDir 'K5m') --out-dir $Out }
Write-Host "[gemma-oracle] $Out\weights_override.json (+ mix_ranking.json); next: tools\gemma\trellis_convert_gemma.ps1 -Oracle $Out"
