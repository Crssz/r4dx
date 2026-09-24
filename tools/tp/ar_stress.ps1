#requires -Version 5.1
<#
.SYNOPSIS
  A device-0 endurance run of tool_tp_ar_stress (the G5 10M decode-pattern retry, docs/tp.md
  Appendix B N62, N64) under a TDR watch.

.DESCRIPTION
  Runs build\<preset>\tests\kernels\tool_tp_ar_stress.exe with every argument given to this script
  passed through unchanged, through tools\tp\tdr_watch.psm1's Invoke-TdrWatched: the r4dx-server
  pre-flight, HIP_VISIBLE_DEVICES unset (restored afterwards), tdr_check.ps1 -Since <start> every
  20 s WHILE the stress runs -- the first TDR stops it at once, never retried -- a final TDR check
  after a 30 s wait, and a scan of the --json report for HIP error 719 (a suspected TDR).

  Refuses to start unless the arguments bound the submission with `--flush-every N --max-inflight 1`
  (N >= 1): P3's unbounded decode stress TDR'd twice in ~200 s with the GPU kept 100% busy (N44),
  and --max-inflight 0 or >= 2 keeps it just as busy (K >= 2 always has a unit queued). Only K = 1,
  a synchronize per unit, leaves the GPU idle at each unit; add `--idle-us U` for a longer gap.
  --json is required (the report is scanned). Run it in pieces (e.g. --count 1000000, a few minutes
  each) with pauses in between rather than as one 23-minute run.

  Exit code 0 only when the stress exited 0 and no (suspected) TDR happened; otherwise 1. Set
  $env:R4DX_SOAK_PRESET to use a build preset other than win-hip. A relative --json path is relative
  to the repository root, where the tool runs.

.EXAMPLE
  .\tools\tp\ar_stress.ps1 --count 1000000 --pattern decode --flush-every 16 --max-inflight 1 --idle-us 500 --json build\logs\ar_stress_1.json
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot)
Import-Module (Join-Path $PSScriptRoot "tdr_watch.psm1") -Force
$Preset = if ($env:R4DX_SOAK_PRESET) { $env:R4DX_SOAK_PRESET } else { "win-hip" }
$Exe = Join-Path $Root "build\$Preset\tests\kernels\tool_tp_ar_stress.exe"

$ToolArgs = @($args | ForEach-Object { [string]$_ })
$opt = @{}
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -in @('--flush-every', '--max-inflight', '--json')) { $opt[$ToolArgs[$i]] = $ToolArgs[$i + 1] }
}
$flush = 0
if (-not $opt.ContainsKey('--flush-every') -or -not [int]::TryParse($opt['--flush-every'], [ref]$flush) -or $flush -lt 1 -or
    $opt['--max-inflight'] -ne '1') {
    throw ("ar_stress.ps1: a device-0 endurance run needs --flush-every N (N >= 1) --max-inflight 1: any other " +
           "bounding keeps the GPU busy without a gap, as P3's TDR'd stress did (docs/tp.md Appendix B N44, N64)")
}
if (-not $opt.ContainsKey('--json')) { throw "ar_stress.ps1: --json <report.json> is required (it is scanned for HIP error 719)" }
$json = $opt['--json']
$JsonAbs = if ([System.IO.Path]::IsPathRooted($json)) { $json } else { Join-Path $Root $json }

$r = Invoke-TdrWatched -Exe $Exe -ToolArgs $ToolArgs -WorkingDirectory $Root -JsonLog $JsonAbs -Label 'ar_stress.ps1'
if ($r.Tdr -or $r.SuspectedTdr -or $r.ExitCode -ne 0) {
    Write-Output ("[ar_stress.ps1] FAIL: stress exit code {0}, TDR {1}, suspected TDR {2}, stopped {3}" -f `
                  $r.ExitCode, $r.Tdr, $r.SuspectedTdr, $r.Stopped)
    exit 1
}
Write-Output "[ar_stress.ps1] PASS: stress exit 0, no TDR since the start"
exit 0
