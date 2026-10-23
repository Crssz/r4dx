#requires -Version 5.1
<#
.SYNOPSIS
  The --tp 2 soak of Ornith-1.5-122B (docs/qwen35moe-tp-design.md P5, gate G8'): tool_q35moe_tp_soak on both GPUs under a TDR watch.

.DESCRIPTION
  tools\tp\soak.ps1's contract for the qwen3_5_moe tool: takes the machine-wide mutexes `gpu` then `gpu-dev0` (tools\with_lock.ps1 names), runs
  build\<preset>\tests\model\qwen35moe\tool_q35moe_tp_soak.exe with EVERY argument passed through (--json is required) through tdr_watch.psm1's
  Invoke-TdrWatched: the r4dx-server pre-flight, HIP_VISIBLE_DEVICES unset, tdr_check.ps1 every 20 s WHILE the soak runs -- the first TDR stops
  the soak at once and is never retried -- then a 30 s wait and a final check, and a scan of the log for HIP error 719.

  G8' verdict, exit code 0 only when ALL hold; otherwise 1 with every reason printed:
    - no TDR and no suspected TDR;
    - the soak exited 0;
    - this run's JSON log has a "summary" line with exit_code 0 AND a "teardown" line with exit_code 0.
  Stage it: 5 minutes first, then 60 (the 60-minute run needs the owner's go-ahead: dual-GPU sustained load has reset the display on this machine).

.EXAMPLE
  .\tools\tp\q35moe_soak.ps1 --minutes 5 --max-ctx 8192 --json build\logs\q35moe_soak_5min.jsonl
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot)
Import-Module (Join-Path $PSScriptRoot "tdr_watch.psm1") -Force
$Preset = if ($env:R4DX_SOAK_PRESET) { $env:R4DX_SOAK_PRESET } else { "win-hip" }
$Exe = Join-Path $Root "build\$Preset\tests\model\qwen35moe\tool_q35moe_tp_soak.exe"

$ToolArgs = @($args | ForEach-Object { [string]$_ })
$json = $null
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--json') { $json = $ToolArgs[$i + 1] }
}
if (-not $json) { throw "q35moe_soak.ps1: --json <log.jsonl> is required (the G8' verdict reads it)" }
for ($i = 0; $i + 1 -lt $ToolArgs.Count; $i++) {
    if ($ToolArgs[$i] -eq '--tp-max-inflight' -and $ToolArgs[$i + 1] -ne '1') {
        throw "q35moe_soak.ps1: an endurance run keeps --tp-max-inflight 1 (the default): any other K records events and leaves the GPU no idle gap (docs/tp.md Appendix B N64)"
    }
}
$JsonAbs = if ([System.IO.Path]::IsPathRooted($json)) { $json } else { Join-Path $Root $json }
New-Item -ItemType Directory -Force (Split-Path $JsonAbs) | Out-Null

$held = @()
try {
    foreach ($n in 'gpu', 'gpu-dev0') {
        $m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$n")
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes(240)) }
        catch [System.Threading.AbandonedMutexException] { $got = $true }
        if (-not $got) { Write-Error "q35moe_soak.ps1: timed out waiting for lock '$n'"; exit 75 }
        $held += $m
        Write-Host ("[q35moe_soak] lock {0} taken {1:u}" -f $n, (Get-Date))
    }
    $env:R4DX_ALLOW_GPU0 = '1'
    $r = Invoke-TdrWatched -Exe $Exe -ToolArgs $ToolArgs -WorkingDirectory $Root -JsonLog $JsonAbs -JsonAppends -Label 'q35moe_soak'

    $fail = New-Object System.Collections.Generic.List[string]
    if ($r.Tdr) { $fail.Add("a TDR since the start (the soak was stopped: $($r.Stopped))") }
    if ($r.SuspectedTdr) { $fail.Add("suspected TDR (HIP error 719 in the log)") }
    if ($r.ExitCode -ne 0) { $fail.Add("tool_q35moe_tp_soak exit code $($r.ExitCode)") }
    $summary = @($r.NewLines | Where-Object { $_ -match '"type":"summary"' }) | Select-Object -Last 1
    $teardown = @($r.NewLines | Where-Object { $_ -match '"type":"teardown"' }) | Select-Object -Last 1
    if (-not $summary) { $fail.Add("no summary line in this run's log") }
    elseif (($summary | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the summary line has exit_code $(($summary | ConvertFrom-Json).exit_code)") }
    if (-not $teardown) { $fail.Add("no teardown line in this run's log (the process died before or inside the model's destructor)") }
    elseif (($teardown | ConvertFrom-Json).exit_code -ne 0) { $fail.Add("the teardown line has exit_code $(($teardown | ConvertFrom-Json).exit_code)") }
    if ($fail.Count -gt 0) {
        Write-Output "[q35moe_soak] FAIL (G8'):"
        foreach ($f in $fail) { Write-Output "    $f" }
        exit 1
    }
    Write-Output "[q35moe_soak] PASS (G8'): soak exit 0, summary and teardown exit_code 0, no TDR since the start"
    exit 0
}
finally {
    for ($i = $held.Count - 1; $i -ge 0; $i--) { $held[$i].ReleaseMutex() }
}
