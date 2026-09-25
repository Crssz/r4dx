# quant2 experiment driver: for each variant, convert (CPU) then rung-4 KL (HIP device 1) against a
# fixed bf16 reference, appending one row per variant to results.json. Sequential and resumable:
# a variant whose kl_<name>.json exists is skipped; a container is reused only if its
# <name>.done marker exists (written after a successful convert), otherwise it is rebuilt.
#
# Every variant starts from the v6 recipe (w4a16 only, --no-bf16, MTP + vision, KV calib,
# search+imatrix, attn.k/v kept bf16) and appends its own converter flags:
#
#   .\tools\quant2\run_variants.ps1 -Variant 'q2ab_ldlq=--rotate q2ab --hessian-dir D:\models\r4dx\hessian-v1 --ldlq .'
param(
  [Parameter(Mandatory = $true)][string[]]$Variant,
  [string]$OutDir = 'D:\models\r4dx\kl-q2',
  [string]$RefDir = 'D:\models\r4dx\kl-q1\ref',
  [string]$ContainerDir = 'D:\models\r4dx',
  [string]$Checkpoint = 'C:\AI\models\Qwen3.8-27B',
  [string]$Python = 'C:\Users\pay20\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe',
  [switch]$DeleteContainers
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
New-Item -ItemType Directory -Force $OutDir | Out-Null
$conv = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe'
$tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe'
$tokens = 'tools\reference\kl_corpus\tokens.json'
if (-not (Test-Path (Join-Path $RefDir 'reference_run.json'))) { throw "[var] no reference in $RefDir" }

$base = @('--input', $Checkpoint, '--layouts', 'w4a16', '--lm-head', '4bit', '--no-bf16', '--mtp', 'on',
          '--vision', 'on', '--kv-calib', 'D:\models\r4dx\qwen38-27b.kvcalib-full.json',
          '--quant', 'search', '--imatrix', 'D:\models\r4dx\qwen38-27b.imatrix.npz',
          '--keep-bf16', '^text\.layers\.[0-9]+\.attn\.[kv]$')

function Run([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[var] $name -> $log"
  # PS 5.1 makes native stderr lines error records; judge by exit code only.
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd *> $log } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[var] $name failed (exit $LASTEXITCODE), see $log" }
}

$resultsPath = Join-Path $OutDir 'results.json'
$results = @(if (Test-Path $resultsPath) { Get-Content $resultsPath -Raw | ConvertFrom-Json })

foreach ($v in $Variant) {
  $name, $extra = $v -split '=', 2
  $klPath = Join-Path $OutDir "kl_$name.json"
  if (Test-Path $klPath) { Write-Host "[var] $name already measured"; continue }
  $container = Join-Path $ContainerDir "qwen38-27b-$name.r4dx"
  $done = Join-Path $OutDir "$name.done"
  if (-not ((Test-Path $container) -and (Test-Path $done))) {
    Remove-Item -Force $container, $done -ErrorAction SilentlyContinue
    $xargs = @(if ($extra) { $extra -split ' +' | Where-Object { $_ } })
    $t0 = Get-Date
    Run "convert_$name" { & $conv @base --output $container @xargs }
    $convMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Set-Content -Encoding utf8 $done $convMin
  }
  $env:HIP_VISIBLE_DEVICES = '1'
  if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[var] stop r4dx-server first' }
  $dir = Join-Path $OutDir "tf_$name"
  New-Item -ItemType Directory -Force $dir | Out-Null
  Run "tf_$name" { & $tool --model $container --layout w4a16 --tokens $tokens --out-dir $dir --max-ctx 4096 --vision off }
  Run "kl_$name" { & $Python tools\reference\kl_report.py --ref-dir $RefDir --test-dir $dir --tokens $tokens --out $klPath }
  Remove-Item -Recurse -Force $dir
  $j = Get-Content $klPath -Raw | ConvertFrom-Json
  $segs = [ordered]@{}; foreach ($s in $j.segments) { $segs[$s.name] = [math]::Round($s.mean_kl, 5) }
  $results += [pscustomobject]@{ name = $name; flags = $extra; mean_kl = $j.overall.mean_kl;
    median_kl = $j.overall.median_kl; top1_pct = $j.overall.top1_agreement_pct; p99_kl = $j.overall.p99_kl;
    segments = $segs; bytes = (Get-Item $container).Length; convert_min = [string](Get-Content $done -Raw).Trim() }
  ConvertTo-Json -InputObject @($results) -Depth 4 | Set-Content -Encoding utf8 $resultsPath
  Write-Host ("[var] {0,-16} mean KL {1:N5}  top-1 {2:N2}%" -f $name, $j.overall.mean_kl, $j.overall.top1_agreement_pct)
  if ($DeleteContainers) { Remove-Item -Force $container }
}
