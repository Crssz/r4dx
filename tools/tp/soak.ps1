#requires -Version 5.1
<#
.SYNOPSIS
  The tensor-parallel soak (docs/tp.md 10.5, gate G8): tool_tp_soak on both GPUs under a TDR watch.

.DESCRIPTION
  Runs build\<preset>\tests\model\tool_tp_soak.exe with EVERY argument given to this script passed
  through unchanged (see that tool's header for its options; --json is required) through
  tools\tp\tdr_watch.psm1's Invoke-TdrWatched: the r4dx-server pre-flight (9.2), HIP_VISIBLE_DEVICES
  unset (restored afterwards), tdr_check.ps1 -Since <start> every 20 s WHILE the soak runs -- the
  first TDR stops the soak at once, it is never retried (Appendix B N44, N64) -- then a 30 s wait and
  a final TDR check, and a scan of the log for HIP error 719 (a suspected TDR).

  G8 verdict, exit code 0 only when ALL hold; otherwise 1, with every reason printed:
    - no TDR (WER LiveKernelEvent 141 / System 4101, Appendix B N55) and no suspected TDR;
    - the soak exited 0;
    - this run's JSON log has a "summary" line with exit_code 0 AND a "teardown" line with exit_code
      0 (the TpModel was destroyed cleanly; an exit crash after the summary, N63, fails G8).
  Stage it: 5 minutes first, then 60 (Appendix C q2), each with its own --json file. Refuses
  --tp-max-inflight other than 1 (the default): only K = 1 leaves the GPU an idle gap at each unit
  (Appendix B N64); short A/B measurements with other settings call the tool directly.

  HYBRID SOAK (docs/pp-tp2-hybrid.md 9 P4, gate G-H4): with `--pp 2` tool_tp_soak runs the hybrid serving mode
  (--tp 2 --pp 2) with long prompts, warm "turns" conversations and hybrid canaries (see that tool's header); the
  verdict is the same plus: the summary's "hybrid" object must show the hybrid engaged and >= 1 pipelined call.
  tools\hybrid\soak.ps1 is the convenience wrapper that supplies --pp 2 and the hybrid's soak defaults.

  No param() block on purpose: every argument, `--minutes 5` included, belongs to tool_tp_soak. Set
  $env:R4DX_SOAK_PRESET to use a build preset other than win-hip. A relative --json path is relative
  to the repository root, where the soak runs.

.EXAMPLE
  .\tools\tp\soak.ps1 --layout w4a16 --minutes 5 --max-ctx 8192 --json build\logs\tp_soak_5min.jsonl
  .\tools\tp\soak.ps1 --layout w4a16 --minutes 60 --max-ctx 8192 --json build\logs\tp_soak_60min.jsonl
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot)
Import-Module (Join-Path $PSScriptRoot "tdr_watch.psm1") -Force
$Preset = if ($env:R4DX_SOAK_PRESET) { $env:R4DX_SOAK_PRESET } else { "win-hip" }
$Exe = Join-Path $Root "build\$Preset\tests\model\tool_tp_soak.exe"

$ToolArgs = @($args | ForEach-Object { [string]$_ })
$json = $null
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--json') { $json = $ToolArgs[$i + 1] }
}
if (-not $json) { throw "soak.ps1: --json <log.jsonl> is required (the G8 verdict reads it)" }
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--tp-max-inflight' -and $ToolArgs[$i + 1] -ne '1') {
        throw ("soak.ps1: an endurance run keeps --tp-max-inflight 1 (the default): any other K records events and " +
               "leaves the GPU no idle gap inside a prefill chunk (docs/tp.md Appendix B N64)")
    }
}
$IsHybrid = $false
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--pp' -and $ToolArgs[$i + 1] -eq '2') { $IsHybrid = $true }
}
$Gate = if ($IsHybrid) { "G-H4" } else { "G8" }
$JsonAbs = if ([System.IO.Path]::IsPathRooted($json)) { $json } else { Join-Path $Root $json }

$r = Invoke-TdrWatched -Exe $Exe -ToolArgs $ToolArgs -WorkingDirectory $Root -JsonLog $JsonAbs -JsonAppends `
                       -Label 'soak.ps1'

$fail = New-Object System.Collections.Generic.List[string]
if ($r.Tdr) { $fail.Add("a TDR since the start (the soak was stopped: $($r.Stopped))") }
if ($r.SuspectedTdr) { $fail.Add("suspected TDR (HIP error 719 in the log)") }
if ($r.ExitCode -ne 0) { $fail.Add("tool_tp_soak exit code $($r.ExitCode)") }
$summary = @($r.NewLines | Where-Object { $_ -match '"type":"summary"' }) | Select-Object -Last 1
$teardown = @($r.NewLines | Where-Object { $_ -match '"type":"teardown"' }) | Select-Object -Last 1
if (-not $summary) { $fail.Add("no summary line in this run's log") }
elseif (($summary | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the summary line has exit_code $(($summary | ConvertFrom-Json).exit_code)") }
if (-not $teardown) { $fail.Add("no teardown line in this run's log (the process died before or inside ~TpModel)") }
elseif (($teardown | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the teardown line has exit_code $(($teardown | ConvertFrom-Json).exit_code)") }
if ($IsHybrid -and $summary) {
    $hy = ($summary | ConvertFrom-Json).hybrid
    if ($null -eq $hy) {
        $fail.Add("--pp 2 but the summary line has no 'hybrid' object")
    } else {
        $refText = if ([int]$hy.tp2_ref.compared -gt 0) {
            ("{0} iterations compared, {1} differ at the first token, mean agreement {2:P1}") -f $hy.tp2_ref.compared, $hy.tp2_ref.first_token_diff, $hy.tp2_ref.mean_agree
        } else {
            "no iteration compared (--tp2-ref-every not given or no greedy iteration reached it)"
        }
        $hyLine = ("[soak.ps1] hybrid: split k={0}, S={1} tokens, min rows {2}; {3} pipelined / {4} TP-prefill calls, DFlash tail rows {5}; " +
                   "TP=2 reference: {6}") -f `
            $hy.split, $hy.stage_ctx, $hy.min_rows, $hy.pipelined, $hy.tp_prefill, $hy.tail_rows, $refText
        Write-Output $hyLine
        if (-not $hy.engaged) { $fail.Add("the hybrid mode was not engaged at the end of the soak") }
        if ($hy.pipelined -lt 1) { $fail.Add("the hybrid soak made no pipelined call") }
    }
}

if ($fail.Count -gt 0) {
    Write-Output "[soak.ps1] FAIL ($Gate):"
    foreach ($f in $fail) { Write-Output "    $f" }
    exit 1
}
Write-Output "[soak.ps1] PASS ($Gate): soak exit 0, summary and teardown exit_code 0, no TDR since the start"
exit 0
