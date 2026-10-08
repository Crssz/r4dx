#requires -Version 5.1
<#
.SYNOPSIS
  The pipeline-parallel-prefill soak (docs/pp-prefill.md Phase 2, gate G2c): tool_pp_soak on both GPUs under a TDR watch.

.DESCRIPTION
  Runs build\<preset>\tests\model\tool_pp_soak.exe with EVERY argument given to this script passed through unchanged (see
  that tool's header for its options; --json is required) through tools\tp\tdr_watch.psm1's Invoke-TdrWatched with
  HIP_VISIBLE_DEVICES=1,0 (ordinal 0 = the headless decode card = stage B, ordinal 1 = the desktop card = stage A; restored
  afterwards): the r4dx-server pre-flight, tdr_check.ps1 -Since <start> every 20 s WHILE the soak runs -- the first TDR
  stops the soak at once, it is never retried (docs/tp.md Appendix B N44, N64) -- then a 30 s wait and a final TDR check,
  and a scan of the log for HIP error 719 (a suspected TDR).

  G2c verdict, exit code 0 only when ALL hold; otherwise 1, with every reason printed:
    - no TDR (WER LiveKernelEvent 141 / System 4101) and no suspected TDR;
    - the soak exited 0 (every monolithic/pipelined comparison equal, buffer drift <= 64 MiB per device);
    - this run's JSON log has a "summary" line with exit_code 0 AND a "teardown" line with exit_code 0 (the PpModel was
      destroyed cleanly).
  Stage it: 5 minutes first, then 60, each with its own --json file. Stage A runs the desktop card at ~100% duty for the
  whole prefill (2-19 s at 8k-64k), so run the first 64k+ prefills with the user present (docs/pp-prefill.md 4: power and
  the TDR hazard of the display card).

  No param() block on purpose: every argument, `--minutes 5` included, belongs to tool_pp_soak. Set $env:R4DX_SOAK_PRESET to
  use a build preset other than win-hip. A relative --json path is relative to the repository root, where the soak runs.

.EXAMPLE
  .\tools\pp\soak.ps1 --layout trellis --minutes 5 --max-ctx 36864 --json build\logs\pp_soak_5min.jsonl
  .\tools\pp\soak.ps1 --layout trellis --minutes 60 --max-ctx 36864 --verify --json build\logs\pp_soak_60min.jsonl
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot)
Import-Module (Join-Path $Root "tools\tp\tdr_watch.psm1") -Force
$Preset = if ($env:R4DX_SOAK_PRESET) { $env:R4DX_SOAK_PRESET } else { "win-hip" }
$Exe = Join-Path $Root "build\$Preset\tests\model\tool_pp_soak.exe"

$ToolArgs = @($args | ForEach-Object { [string]$_ })
$json = $null
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--json') { $json = $ToolArgs[$i + 1] }
}
if (-not $json) { throw "soak.ps1: --json <log.jsonl> is required (the G2c verdict reads it)" }
$JsonAbs = if ([System.IO.Path]::IsPathRooted($json)) { $json } else { Join-Path $Root $json }

$r = Invoke-TdrWatched -Exe $Exe -ToolArgs $ToolArgs -WorkingDirectory $Root -JsonLog $JsonAbs -JsonAppends `
                       -Label 'pp/soak.ps1' -HipVisibleDevices '1,0'

$fail = New-Object System.Collections.Generic.List[string]
if ($r.Tdr) { $fail.Add("a TDR since the start (the soak was stopped: $($r.Stopped))") }
if ($r.SuspectedTdr) { $fail.Add("suspected TDR (HIP error 719 in the log)") }
if ($r.ExitCode -ne 0) { $fail.Add("tool_pp_soak exit code $($r.ExitCode)") }
$summary = @($r.NewLines | Where-Object { $_ -match '"type":"summary"' }) | Select-Object -Last 1
$teardown = @($r.NewLines | Where-Object { $_ -match '"type":"teardown"' }) | Select-Object -Last 1
if (-not $summary) { $fail.Add("no summary line in this run's log") }
elseif (($summary | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the summary line has exit_code $(($summary | ConvertFrom-Json).exit_code)") }
if (-not $teardown) { $fail.Add("no teardown line in this run's log (the process died before or inside ~PpModel)") }
elseif (($teardown | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the teardown line has exit_code $(($teardown | ConvertFrom-Json).exit_code)") }
if ($summary) {
    $s = $summary | ConvertFrom-Json
    Write-Output ("[pp/soak.ps1] {0} iterations ({1} compared), prefill speedup mean x{2:N2} best x{3:N2}" -f $s.iterations, $s.compared, $s.mean_prefill_speedup, $s.best_prefill_speedup)
}

if ($fail.Count -gt 0) {
    Write-Output "[pp/soak.ps1] FAIL (G2c):"
    foreach ($f in $fail) { Write-Output "    $f" }
    exit 1
}
Write-Output "[pp/soak.ps1] PASS (G2c): soak exit 0, summary and teardown exit_code 0, no TDR since the start"
exit 0
