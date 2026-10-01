#requires -Version 5.1
<#
.SYNOPSIS
  Gemma 4 tensor-parallel gate (docs/gemma4-plan.md M1b-2, section 6.3 "TP identity"): TP=1 byte-identical to a frozen
  pre-TP baseline, and TP=2 vs TP=1 compared by KL (<= 3x the noise floor) and top-1 agreement (>= 99.5%).

.DESCRIPTION
  RUNS THE GPU. Ask before launching it (both GPUs under -TpMode real; one under emulate). Nothing here is run by the
  agent that wrote it. -SkipDump re-scores existing dumps (CPU only).

  Steps (all dumps are tool_teacher_forced_logprobs on the chat corpus, the same one the M1 gate uses):
    1 tp1       candidate build, --tp 1, HIP device 1  -> <OutDir>\tp1
    2 identity  -Baseline (a flat dir or build tree of the FROZEN pre-TP tool, r4dx-baselines\gemma-tp1-<commit>): the same
                run -> <OutDir>\base; every *.logprobs.f16 must be byte-identical to step 1 (SHA-256). Skipped without
                -Baseline (then only step 3 is meaningful and the TP=1 guard is NOT established).
    3 tp2       candidate build, --tp 2 --tp-mode <TpMode> -> <OutDir>\tp2
    4 score     kl_report.py --ref-dir tp1 --test-dir tp2 --tokens <Chat> -> <OutDir>\tp2_vs_tp1.json, and the noise floor
                kl_report.py --ref-dir <TruthDir> --test-dir <NoiseDir> (HF bf16 sdpa vs the fp32 truth, the term the M1 gate
                is relative to) -> <OutDir>\noise.json. PASS iff overall mean KL <= 3 x noise mean KL and overall top-1
                agreement >= 99.5 %. (Proposed thresholds, plan 6.3.)

  TP=2 reduces the row-parallel outputs (o_proj, down_proj) in a different order than TP=1's single GEMM K loop, so TP=2 is
  NOT bit-identical to TP=1 by design; the gate is distributional, exactly like the trellis one. Do not expect EQUAL there.

  Environment: R4DX_GEMMA_KV / R4DX_GEMMA_RESID / R4DX_GEMMA_ATTN are honoured (set by the caller; run both tp1 and tp2 with
  the same ones). -Layout trellis needs a rotated trellis container (Gemma option A); the bf16 container is the reference
  (TP=2 holds ~11 GB of weights per rank; a bf16 TP=1 run needs the whole ~29 GB card).

.PARAMETER Model
  The Gemma .r4dx container (required unless -SkipDump).

.PARAMETER Baseline
  Directory of the frozen pre-TP tool (flat, or a build tree). Optional; enables step 2.

.EXAMPLE
  # both GPUs (HIP_VISIBLE_DEVICES unset for the tp2 step; the script manages it):
  powershell -File tools\tp\gemma_tp_identity.ps1 -Model D:\models\r4dx\huihui-gemma\bf16.r4dx -Layout bf16 `
      -OutDir D:\models\r4dx\huihui-gemma\kl\tp -Baseline C:\Users\pay20\dev\r4dx-baselines\gemma-tp1-<commit>
  # one GPU, both ranks on it (needs a body that fits twice: a trellis container):
  powershell -File tools\tp\gemma_tp_identity.ps1 -Model <trellis>.r4dx -Layout trellis -TpMode emulate -OutDir <dir>
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Layout = "bf16",
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Baseline = "",
    [string]$Candidate = "build\win-hip",
    [ValidateSet("real", "emulate")][string]$TpMode = "real",
    [int]$Device = 1,
    [int]$MaxCtx = 4096,
    [string]$Chat = "tools\reference\kl_corpus\chat_gemma.json",
    [string]$TruthDir = "D:\models\r4dx\huihui-gemma\kl\fp32\truth",
    [string]$NoiseDir = "D:\models\r4dx\huihui-gemma\kl\fp32\bf16sdpa",
    [double]$KlFactor = 3.0,
    [double]$Top1MinPct = 99.5,
    [string]$Python = "",
    [switch]$AllowMismatch,
    [switch]$SkipDump
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
function Resolve-Full([string]$p) { if ([System.IO.Path]::IsPathRooted($p)) { return $p } else { return (Join-Path $RepoRoot $p) } }
function Find-Tool([string]$Dir) {
    foreach ($c in @((Join-Path $Dir "tool_teacher_forced_logprobs.exe"), (Join-Path $Dir "tests\model\tool_teacher_forced_logprobs.exe"))) {
        if (Test-Path $c) { return $c }
    }
    throw "gemma_tp_identity: tool_teacher_forced_logprobs.exe not found under $Dir"
}
function Get-Sha([string]$Path) { return (Get-FileHash -Algorithm SHA256 $Path).Hash }
if (-not $Python) {
    $Python = "D:\venvs\r4dx-gemma-ref\Scripts\python.exe"
    if (-not (Test-Path $Python)) { $Python = "C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe" }
}
$Out = Resolve-Full $OutDir
$ChatPath = Resolve-Full $Chat
foreach ($p in $ChatPath, $TruthDir, $NoiseDir, $Python) { if (-not (Test-Path $p)) { throw "gemma_tp_identity: $p not found" } }
New-Item -ItemType Directory -Force $Out | Out-Null

function Invoke-Dump([string]$Tool, [string]$Dir, [string[]]$Extra, [string]$HipDevices) {
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $saved = $env:HIP_VISIBLE_DEVICES
    if ($HipDevices -ne $null -and $HipDevices -ne "") { $env:HIP_VISIBLE_DEVICES = $HipDevices } else { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    try {
        $argv = @("--model", $Model, "--layout", $Layout, "--tokens", $ChatPath, "--out-dir", $Dir, "--max-ctx", $MaxCtx,
                  "--vision", "off", "--quiet") + $Extra
        Write-Host "[gemma_tp_identity] $Tool $($argv -join ' ')  (HIP_VISIBLE_DEVICES='$env:HIP_VISIBLE_DEVICES')"
        & $Tool @argv *> (Join-Path $Dir "stdout_stderr.txt")
        if ($LASTEXITCODE -ne 0) { throw "gemma_tp_identity: the tool failed (exit $LASTEXITCODE), see $Dir\stdout_stderr.txt" }
    } finally {
        if ($null -eq $saved) { Remove-Item env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:HIP_VISIBLE_DEVICES = $saved }
    }
}

$cand = Find-Tool (Resolve-Full $Candidate)
if (-not $SkipDump) {
    if (-not $Model) { throw "gemma_tp_identity: -Model is required (unless -SkipDump)" }
    if (Get-Process r4dx-server -ErrorAction SilentlyContinue) {
        throw "gemma_tp_identity: an r4dx-server process is running -- stop it first (it holds the GPU memory)"
    }
    # 1. TP=1 on the candidate, one GPU.
    Invoke-Dump $cand (Join-Path $Out "tp1") @() "$Device"
    # 2. TP=1 byte identity against the frozen baseline.
    if ($Baseline) {
        $base = Find-Tool (Resolve-Full $Baseline)
        Invoke-Dump $base (Join-Path $Out "base") @() "$Device"
        $files = @(Get-ChildItem (Join-Path $Out "base\*.logprobs.f16") | Sort-Object Name)
        if ($files.Count -eq 0) { throw "gemma_tp_identity: the baseline wrote no *.logprobs.f16" }
        $diff = @()
        foreach ($f in $files) {
            $c = Join-Path $Out ("tp1\" + $f.Name)
            if (-not (Test-Path $c) -or (Get-Sha $f.FullName) -ne (Get-Sha $c)) { $diff += $f.Name }
        }
        if ($diff.Count -gt 0) { throw "gemma_tp_identity: TP=1 DIFFERS from the frozen baseline on: $($diff -join ', ')" }
        Write-Host "[gemma_tp_identity] TP=1 PASS: $($files.Count) logprobs.f16 file(s) byte-identical to the frozen baseline"
    } else {
        Write-Host "[gemma_tp_identity] no -Baseline: the TP=1 byte-identity guard is NOT checked"
    }
    # 3. TP=2.
    $tp2 = @("--tp", "2", "--tp-mode", $TpMode)
    if ($TpMode -eq "emulate") {
        Invoke-Dump $cand (Join-Path $Out "tp2") $tp2 "$Device"
    } else {
        Invoke-Dump $cand (Join-Path $Out "tp2") $tp2 ""   # real: both GPUs visible, rank 0 = device 1, rank 1 = device 0
    }
}

# 4. Score (CPU).
$allow = @(); if ($AllowMismatch) { $allow = @("--allow-mismatch") }
& $Python (Resolve-Full "tools\reference\kl_report.py") --ref-dir (Join-Path $Out "tp1") --test-dir (Join-Path $Out "tp2") `
    --tokens $ChatPath --out (Join-Path $Out "tp2_vs_tp1.json") @allow
if ($LASTEXITCODE -ne 0) { throw "gemma_tp_identity: kl_report.py (tp2 vs tp1) failed (exit $LASTEXITCODE); a dump-identity refusal can be overridden with -AllowMismatch (read it first)" }
& $Python (Resolve-Full "tools\reference\kl_report.py") --ref-dir $TruthDir --test-dir $NoiseDir --tokens $ChatPath `
    --out (Join-Path $Out "noise.json") --allow-mismatch | Out-Null
if ($LASTEXITCODE -ne 0) { throw "gemma_tp_identity: kl_report.py (noise floor) failed (exit $LASTEXITCODE)" }
$r = Get-Content (Join-Path $Out "tp2_vs_tp1.json") -Raw | ConvertFrom-Json
$n = Get-Content (Join-Path $Out "noise.json") -Raw | ConvertFrom-Json
$kl = [double]$r.overall.mean_kl
$noise = [double]$n.overall.mean_kl
$top1 = [double]$r.overall.top1_agreement_pct
$okKl = $kl -le ($KlFactor * $noise)
$okTop1 = $top1 -ge $Top1MinPct
Write-Host ("[gemma_tp_identity] TP2 vs TP1: mean KL {0:E3} (limit {1} x noise {2:E3} = {3:E3}) -> {4}; top-1 {5:F2} % (min {6}) -> {7}" -f `
    $kl, $KlFactor, $noise, ($KlFactor * $noise), $(if ($okKl) { "ok" } else { "FAIL" }), $top1, $Top1MinPct, $(if ($okTop1) { "ok" } else { "FAIL" }))
if (-not ($okKl -and $okTop1)) { throw "gemma_tp_identity: TP=2 gate FAIL" }
Write-Host "[gemma_tp_identity] TP=2 gate PASS"
