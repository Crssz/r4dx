#requires -Version 5.1
<#
.SYNOPSIS
  The --tp 2 half of the Ornith Rung 4 KL gate (docs/qwen35moe-tp-design.md gate G4'): tools\quant2\qwen35moe\r4dx_dump.ps1 with `--tp 2`, on both GPUs
  under a TDR watch.

.DESCRIPTION
  Runs build\win-hip\tests\model\tool_teacher_forced_logprobs.exe --tp 2 (tensor-parallel Qwen35MoeTpModel, real mode: rank 0 = HIP device 1, rank 1 = the
  desktop card) over every segment of -Tokens (Prefill of 1 token, then DecodeStep per token: the serving path, fp8 KV cache and all) and writes
  <name>.logprobs.f16 + <name>.meta.json into -OutDir, exactly the files r4dx_dump.ps1 writes for the layer split. Compare them with the HF dumps on the CPU:

    python tools\reference\qwen35moe\gate_q35.py eval --tokens E:\models\r4dx\q35moe\kl\tokens_gate.json --stock <hf_bf16> --truth <hf_fp32> `
        --test layer-split=E:\models\r4dx\q35moe\kl\r4dx_kv --test tp2=<OutDir> --out <report>

  Takes the machine-wide mutexes `gpu` then `gpu-dev0` (tools\with_lock.ps1 names) for the whole run, refuses to start while an r4dx-server runs, and goes
  through tdr_watch.psm1's Invoke-TdrWatched: tdr_check every 20 s, the FIRST TDR stops the run (no retry). Exit 0 only when the tool exited 0 and no TDR /
  suspected TDR was seen.

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\tp\q35moe_tp_dump.ps1 `
      -Container E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx -Tokens E:\models\r4dx\q35moe\kl\tokens_gate.json -OutDir E:\models\r4dx\q35moe\kl\r4dx_tp2
#>
param(
    [Parameter(Mandatory = $true)][string]$Container,
    [Parameter(Mandatory = $true)][string]$Tokens,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$MaxCtx = 12288,
    [string]$TpMode = 'real',
    [int]$TimeoutMinutes = 240
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $PSScriptRoot 'tdr_watch.psm1') -Force
$exe = Join-Path $Root 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe'
if (-not (Test-Path $exe)) { throw "tool_teacher_forced_logprobs.exe not found (build it through tools\with_lock.ps1 -Name build)" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$held = @()
try {
    foreach ($n in 'gpu', 'gpu-dev0') {
        $m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$n")
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes($TimeoutMinutes)) }
        catch [System.Threading.AbandonedMutexException] { $got = $true }
        if (-not $got) { Write-Error "q35moe_tp_dump.ps1: timed out waiting for lock '$n'"; exit 75 }
        $held += $m
        Write-Host ("[q35moe_tp_dump] lock {0} taken {1:u}" -f $n, (Get-Date))
    }
    $env:R4DX_ALLOW_GPU0 = '1'
    $toolArgs = @('--model', $Container, '--layout', 'trellis', '--tokens', $Tokens, '--out-dir', $OutDir, '--max-ctx', "$MaxCtx", '--vision', 'off',
                  '--quiet', '--tp', '2', '--tp-mode', $TpMode)
    $r = Invoke-TdrWatched -Exe $exe -ToolArgs $toolArgs -WorkingDirectory $Root -Label 'q35moe_tp_dump'
    $fail = @()
    if ($r.Tdr) { $fail += "a TDR since the start (the run was stopped: $($r.Stopped))" }
    if ($r.SuspectedTdr) { $fail += 'suspected TDR (HIP error 719)' }
    if ($r.ExitCode -ne 0) { $fail += "tool exit code $($r.ExitCode)" }
    if ($fail.Count -gt 0) {
        Write-Output '[q35moe_tp_dump] FAIL:'
        $fail | ForEach-Object { Write-Output "    $_" }
        exit 1
    }
    Write-Output '[q35moe_tp_dump] PASS: tool exit 0, no TDR since the start'
    exit 0
}
finally {
    for ($i = $held.Count - 1; $i -ge 0; $i--) { $held[$i].ReleaseMutex() }
}
