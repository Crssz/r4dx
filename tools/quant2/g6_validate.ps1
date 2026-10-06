# quant2 gate G6 on one container (HIP device 1 only): the functional checks a production candidate
# must pass, each script's full output in -OutDir\<step>.log and a one-line verdict per step in
# summary.json. Every step runs even when an earlier one fails; exit 1 if any failed.
#
#   .\tools\quant2\g6_validate.ps1 -Model <models-root>\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx
#   .\tools\quant2\g6_validate.ps1 -Model <a w4a16 container> -Layout w4a16
#
# -Layout (default trellis, the production container's layout) is the container's body layout:
# passed as -Layouts to both validators and as -Layout to the three smoke steps
# (docs/trellis-kernel.md gate A4; a trellis container refuses any other). The DFlash drafter keeps
# its own layout. -Model has no default (Mandatory); the production container is the Huihui abliterated
# trellis mix4.5m above. G6 stores no expected values: its verdict is the exit codes and the
# [PASS]/[FAIL] counts in summary.json (recorded for the production container in docs/huihui.md).
#
# TP=2 runs in --tp-mode emulate (both ranks on device 1): the byte-exact reference of the sharded
# math, which is what a rotated container changes; real mode only adds the device-0 transport.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$Dflash = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx",
  [string]$Layout = 'trellis',
  [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not (Test-Path $Model)) { throw "[g6] $Model not found" }
if (-not $OutDir) { $OutDir = Join-Path (Split-Path $Model) ('g6-' + [IO.Path]::GetFileNameWithoutExtension($Model)) }
New-Item -ItemType Directory -Force $OutDir | Out-Null
foreach ($p in 'r4dx-server', 'r4dx-cli') {
  if (Get-Process $p -ErrorAction SilentlyContinue) { throw "[g6] $p is running; one GPU process at a time" }
}
$env:HIP_VISIBLE_DEVICES = '1'

$steps = @(
  @{ name = 'validate_dflash'; file = 'tools\validate_dflash.ps1'
     args = @('-Model', $Model, '-Dflash', $Dflash, '-Layouts', $Layout) },
  @{ name = 'validate_spec_sampling'; file = 'tools\validate_spec_sampling.ps1'
     args = @('-Model', $Model, '-Dflash', $Dflash, '-Layouts', $Layout, '-AllowBatchedVerifyDivergence') },
  @{ name = 'smoke_dflash_tools_vision'; file = 'tools\server\smoke.ps1'
     args = @('-Model', $Model, '-Layout', $Layout, '-Layers', '-1', '-Dflash', $Dflash, '-ToolRoundTrip', '-Vision') },
  @{ name = 'smoke_mtp3'; file = 'tools\server\smoke.ps1'
     args = @('-Model', $Model, '-Layout', $Layout, '-Layers', '-1', '-Mtp', '3') },
  @{ name = 'smoke_tp2_emulate_dflash'; file = 'tools\server\smoke.ps1'
     args = @('-Model', $Model, '-Layout', $Layout, '-Layers', '-1', '-Dflash', $Dflash, '-Tp', '2', '-TpMode', 'emulate') }
)

$summary = @()
foreach ($s in $steps) {
  $log = Join-Path $OutDir "$($s.name).log"
  Write-Host "[g6] $($s.name) -> $log"
  $t0 = Get-Date
  # A child powershell per step: each script sets its own preferences/env and may exit; its output
  # is stringified and judged by the exit code (PS 5.1 turns native stderr into error records).
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $s.file @($s.args) *>&1 |
      ForEach-Object { "$_" } | Set-Content -Encoding utf8 $log
    $code = $LASTEXITCODE
  } finally { $ErrorActionPreference = $prev }
  $env:HIP_VISIBLE_DEVICES = '1'
  $text = Get-Content $log -Raw
  $pass = @([regex]::Matches($text, '\[PASS\]')).Count
  $fail = @([regex]::Matches($text, '\[FAIL\]')).Count
  $warn = @([regex]::Matches($text, '\bWARN\b')).Count
  $row = [ordered]@{ step = $s.name; exit = $code; pass = $pass; fail = $fail; warn = $warn
                     minutes = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
                     last = [string](@(Get-Content $log | Where-Object { $_.Trim() }) | Select-Object -Last 1) }
  $summary += [pscustomobject]$row
  Write-Host ("[g6] {0,-26} exit {1}  PASS {2}  FAIL {3}  WARN {4}  ({5} min)" -f $s.name, $code, $pass, $fail, $warn, $row.minutes)
  ConvertTo-Json -InputObject @($summary) -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'summary.json')
}
$bad = @($summary | Where-Object { $_.exit -ne 0 })
Write-Host ("[g6] {0}: {1} / {2} steps exit 0" -f $(if ($bad.Count) { 'FAILED' } else { 'PASSED' }), ($summary.Count - $bad.Count), $summary.Count)
exit $(if ($bad.Count) { 1 } else { 0 })
