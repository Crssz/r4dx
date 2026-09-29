# HISTORICAL (Q1/Q2 experiment driver): its defaults name the base checkpoint and hessian-v1, both
# removed (2026-09-29, docs/huihui.md "Retired files"), and the kl-q1 reference, whose directory still
# exists but holds only the *.meta.json and reference_run.json files (the *.logprobs.f16 dumps are gone,
# so KL scoring against it cannot run); pass every path explicitly to re-run it on another model.
# quant2 experiment driver: for each variant, convert (CPU) then rung-4 KL (HIP device 1) against a
# fixed bf16 reference, appending one row per variant to results.json. Sequential and resumable:
# a variant whose kl_<name>.json exists is skipped; a container is reused only if its
# <name>.done marker exists (written after a successful convert), otherwise it is rebuilt.
#
# Every variant starts from the v6 recipe (w4a16 only, --no-bf16, MTP + vision, KV calib,
# search+imatrix, attn.k/v kept bf16) and appends its own converter flags:
#
#   .\tools\quant2\run_variants.ps1 -Variant 'q2ab_ldlq=--rotate q2ab --hessian-dir D:\models\r4dx\hessian-v1 --ldlq .'
#
# -ExtraKl name=tokens.json=refdir (repeatable) also scores each container on another tokens file
# against its own bf16 reference, stored as extra.<name> on the row (kl_<variant>.<name>.json). The
# default adds the canonically tokenized Thai segment (docs/quant2.md 3.4); tokens.json's Thai
# segment is the AutoTokenizer (split-mark) form, kept for comparability with earlier rows.
param(
  [Parameter(Mandatory = $true)][string[]]$Variant,
  [string]$OutDir = 'D:\models\r4dx\kl-q2',
  [string]$RefDir = 'D:\models\r4dx\kl-q1\ref',
  [string]$ContainerDir = 'D:\models\r4dx',
  [string]$Checkpoint = 'C:\AI\models\Qwen3.8-27B',
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' }),
  [string[]]$ExtraKl = @('thai_canon=tools\reference\kl_corpus\tokens_thai_canon.json=D:\models\r4dx\kl-thai-canon\ref'),
  # Where the per-variant log-prob dumps go while kl_report.py reads them (~2 GB for 4 segments,
  # deleted right after): a drive with room, when -OutDir's is full of containers.
  [string]$ScratchDir = '',
  [switch]$DeleteContainers
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
New-Item -ItemType Directory -Force $OutDir | Out-Null
if (-not $ScratchDir) { $ScratchDir = $OutDir }
New-Item -ItemType Directory -Force $ScratchDir | Out-Null
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
# PS 5.1's ConvertFrom-Json emits a JSON array as ONE pipeline object; wrapping that in @() nests it
# (and the next save writes {"value": [...], "Count": n}). foreach enumerates the rows themselves.
$results = @()
if (Test-Path $resultsPath) { foreach ($row in (Get-Content $resultsPath -Raw | ConvertFrom-Json)) { $results += $row } }

foreach ($v in $Variant) {
  $name, $extra = $v -split '=', 2
  $klPath = Join-Path $OutDir "kl_$name.json"
  if (Test-Path $klPath) { Write-Host "[var] $name already measured"; continue }
  $container = Join-Path $ContainerDir "qwen38-27b-$name.r4dx"
  $done = Join-Path $OutDir "$name.done"
  if (-not ((Test-Path $container) -and (Test-Path $done))) {
    Remove-Item -Force $container, $done -ErrorAction SilentlyContinue
    $xargs = @(if ($extra) { $extra -split ' +' | Where-Object { $_ } })
    # '--no-keep-kv' (a driver token, not a converter flag) drops the v6 recipe's --keep-bf16 on
    # attn.k/attn.v, so those linears are quantized like every other one.
    $vbase = $base
    if ($xargs -contains '--no-keep-kv') {
      $xargs = @($xargs | Where-Object { $_ -ne '--no-keep-kv' })
      $i = [array]::IndexOf($base, '--keep-bf16')
      # index filter, not a range: PS ranges count DOWN when the pair is last (6..5).
      $vbase = @(for ($k = 0; $k -lt $base.Count; $k++) { if ($k -ne $i -and $k -ne $i + 1) { $base[$k] } })
    }
    $t0 = Get-Date
    Run "convert_$name" { & $conv @vbase --output $container @xargs }
    $convMin = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Set-Content -Encoding utf8 $done $convMin
  }
  $env:HIP_VISIBLE_DEVICES = '1'
  if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[var] stop r4dx-server first' }
  # Extra tokens files first: kl_<name>.json (written last) is what marks the variant measured.
  $extraRows = [ordered]@{}
  foreach ($x in $ExtraKl) {
    $xn, $xtok, $xref = $x -split '=', 3
    if (-not (Test-Path (Join-Path $xref 'reference_run.json'))) { throw "[var] -ExtraKl ${xn}: no reference in $xref" }
    $xkl = Join-Path $OutDir "kl_$name.$xn.json"
    if (-not (Test-Path $xkl)) {
      $xdir = Join-Path $ScratchDir "tf_$name.$xn"
      New-Item -ItemType Directory -Force $xdir | Out-Null
      Run "tf_$name.$xn" { & $tool --model $container --layout w4a16 --tokens $xtok --out-dir $xdir --max-ctx 4096 --vision off }
      Run "kl_$name.$xn" { & $Python tools\reference\kl_report.py --ref-dir $xref --test-dir $xdir --tokens $xtok --out $xkl }
      Remove-Item -Recurse -Force $xdir
    }
    $xj = Get-Content $xkl -Raw | ConvertFrom-Json
    $extraRows[$xn] = [ordered]@{ mean_kl = $xj.overall.mean_kl; top1_pct = $xj.overall.top1_agreement_pct }
  }
  $dir = Join-Path $ScratchDir "tf_$name"
  New-Item -ItemType Directory -Force $dir | Out-Null
  Run "tf_$name" { & $tool --model $container --layout w4a16 --tokens $tokens --out-dir $dir --max-ctx 4096 --vision off }
  Run "kl_$name" { & $Python tools\reference\kl_report.py --ref-dir $RefDir --test-dir $dir --tokens $tokens --out $klPath }
  Remove-Item -Recurse -Force $dir
  $j = Get-Content $klPath -Raw | ConvertFrom-Json
  $segs = [ordered]@{}; foreach ($s in $j.segments) { $segs[$s.name] = [math]::Round($s.mean_kl, 5) }
  $results += [pscustomobject]@{ name = $name; flags = $extra; mean_kl = $j.overall.mean_kl;
    median_kl = $j.overall.median_kl; top1_pct = $j.overall.top1_agreement_pct; p99_kl = $j.overall.p99_kl;
    segments = $segs; extra = $extraRows; bytes = (Get-Item $container).Length
    convert_min = [string](Get-Content $done -Raw).Trim() }
  ConvertTo-Json -InputObject @($results) -Depth 5 | Set-Content -Encoding utf8 $resultsPath
  Write-Host ("[var] {0,-16} mean KL {1:N5}  top-1 {2:N2}%  {3}" -f $name, $j.overall.mean_kl, $j.overall.top1_agreement_pct,
    (($extraRows.Keys | ForEach-Object { "{0} {1:N5}" -f $_, $extraRows[$_].mean_kl }) -join '  '))
  if ($DeleteContainers) { Remove-Item -Force $container }
}
