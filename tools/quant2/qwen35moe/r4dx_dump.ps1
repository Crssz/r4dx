<#
.SYNOPSIS
  Teacher-forced log-prob dump of an Ornith-1.5-122B (qwen3_5_moe) r4dx container: the r4dx half of the Rung 4 KL gate.

.DESCRIPTION
  Runs build\win-hip\tests\model\tool_teacher_forced_logprobs.exe (Prefill of 1 token, then DecodeStep per token: the serving path,
  fp8 KV cache and all) over every segment of -Tokens and writes <name>.logprobs.f16 + <name>.meta.json into -OutDir; tools\reference\qwen35moe\gate_q35.py
  compares those with the HF dumps on the CPU.

  The model needs BOTH cards (26 GiB each), so this takes the machine-wide mutexes `gpu` and `gpu-dev0` (tools\with_lock.ps1 names;
  always in that order) for the whole run, and R4DX_ALLOW_GPU0=1. Never run it while the HF reference (ref_run.py) is on a card.
  Waits up to -TimeoutMinutes for the locks (exit 75 on timeout).

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\quant2\qwen35moe\r4dx_dump.ps1 `
      -Container E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx -Tokens E:\models\r4dx\q35moe\kl\tokens_gate.json -OutDir E:\models\r4dx\q35moe\kl\r4dx_kv
#>
param(
    [Parameter(Mandatory = $true)][string]$Container,
    [Parameter(Mandatory = $true)][string]$Tokens,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$MaxCtx = 12288,
    [int]$TimeoutMinutes = 240
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = 'C:\Users\pay20\dev\r4dx\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
if (-not (Test-Path $exe)) { throw "tool_teacher_forced_logprobs.exe not found (build it through tools\with_lock.ps1 -Name build)" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$held = @()
try {
    foreach ($n in 'gpu', 'gpu-dev0') {
        $m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$n")
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes($TimeoutMinutes)) }
        catch [System.Threading.AbandonedMutexException] { $got = $true }
        if (-not $got) { Write-Error "r4dx_dump.ps1: timed out waiting for lock '$n'"; exit 75 }
        $held += $m
        Write-Host ("[r4dx_dump] lock {0} taken {1:u}" -f $n, (Get-Date))
    }
    Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $env:R4DX_ALLOW_GPU0 = '1'
    & $exe --model $Container --layout trellis --tokens $Tokens --out-dir $OutDir --max-ctx $MaxCtx --vision off --quiet
    $rc = $LASTEXITCODE
    Write-Host "[r4dx_dump] tool exit code $rc"
    exit $rc
}
finally {
    for ($i = $held.Count - 1; $i -ge 0; $i--) { $held[$i].ReleaseMutex() }
}
