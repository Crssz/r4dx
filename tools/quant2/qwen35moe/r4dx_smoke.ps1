<#
.SYNOPSIS
  Smoke + speed test of Ornith-1.5-122B (qwen3_5_moe) r4dx containers with r4dx-cli: the France prompt, then a long prompt
  (-LongPrompt, default the ~7k-token plan doc E:\models\r4dx\q35moe\long_prompt.txt) for prefill / decode speed.

.DESCRIPTION
  Needs BOTH cards, so it takes the machine-wide mutexes `gpu` then `gpu-dev0` (the tools\with_lock.ps1 names) for the whole run and sets
  R4DX_ALLOW_GPU0=1. Several containers are tested back to back inside one locked window so the speed numbers are comparable.
  Output (the `--stats` lines included) goes to -Log.

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\quant2\qwen35moe\r4dx_smoke.ps1 `
      -Containers E:\models\r4dx\ornith-1.5-122b-trellis-mix.r4dx,E:\models\r4dx\ornith-1.5-122b-trellis-mix-p2.r4dx -Log E:\models\r4dx\q35moe\logs\smoke-p2.log
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Containers,
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$LongPrompt = 'E:\models\r4dx\q35moe\long_prompt.txt',
    [string]$TokenizerDir = 'E:\models\r4dx\Ornith-1.5-122B',
    [int]$TimeoutMinutes = 240
)
$ErrorActionPreference = 'Stop'
$Containers = @($Containers | ForEach-Object { $_ -split ',' })   # powershell -File passes a comma list as one string
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = 'C:\Users\pay20\dev\r4dx\build\win-hip\src\cli\r4dx-cli.exe'
if (-not (Test-Path $exe)) { $exe = Join-Path $repo 'build\win-hip\src\cli\r4dx-cli.exe' }
if (-not (Test-Path $exe)) { throw 'r4dx-cli.exe not found' }
'' | Set-Content $Log

$held = @()
try {
    foreach ($n in 'gpu', 'gpu-dev0') {
        $m = New-Object System.Threading.Mutex($false, "Global\r4dx-lock-$n")
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes($TimeoutMinutes)) }
        catch [System.Threading.AbandonedMutexException] { $got = $true }
        if (-not $got) { Write-Error "r4dx_smoke.ps1: timed out waiting for lock '$n'"; exit 75 }
        $held += $m
    }
    Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $env:R4DX_ALLOW_GPU0 = '1'
    $ErrorActionPreference = 'Continue'   # r4dx-cli logs to stderr; with Stop a log line would end the script
    $rc = 0
    foreach ($c in $Containers) {
        foreach ($t in @(
            @('France', @('--prompt', 'What is the capital of France? Answer in one sentence.', '--think', 'off', '--max-tokens', '40', '--max-ctx', '4096')),
            @('long prompt', @('--prompt-file', $LongPrompt, '--think', 'off', '--max-tokens', '128', '--max-ctx', '16384')))) {
            "===== $c : $($t[0])" | Add-Content $Log
            & $exe --model $c --layout trellis --tokenizer-dir $TokenizerDir --temperature 0 --stats @($t[1]) *>> $Log
            "exit $LASTEXITCODE" | Add-Content $Log
            if ($LASTEXITCODE -ne 0) { $rc = 1 }
        }
    }
    exit $rc
}
finally {
    for ($i = $held.Count - 1; $i -ge 0; $i--) { $held[$i].ReleaseMutex() }
}
