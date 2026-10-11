#requires -Version 5.1
<#
.SYNOPSIS
  qwen4_exp quality gates (plan 4.10 / Q10) of one r4dx container: teacher-forced log-probs per segment on both cards,
  then the ORIGINAL gate (vs the bf16 reference) and the noise-relative gate (vs truth-fp32) through
  tools\reference\qwen4exp\gate_eval.py.

.DESCRIPTION
  1. For every requested scope and segment runs build\win-hip\tests\model\tool_teacher_forced_logprobs.exe (Prefill(1)/DecodeStep
     per token) on the segment's tokens file, each segment under its OWN `with_lock.ps1 -Name gpu` acquisition, with both cards
     visible (HIP_VISIBLE_DEVICES removed) and R4DX_Q4_DEVICES / R4DX_Q4_SPLIT set from -Devices / -Split. Writes
     <Out>\<segment>.logprobs.f16 + .meta.json + <segment>.tool.log + <segment>.gate_done.json (the resume marker).
  2. Evaluates (CPU, no lock) and writes <Out>\gate_report.json / gate_report.md:
       A  original gate: KL(bf16 reference E:\models\r4dx\q4exp\ref\canon || r4dx), pooled mean KL <= 0.02 and top-1 >= 93 %
       B  noise-relative gate: KL(truth-fp32 || r4dx) per group english/thai/code/ALL, <= 1.5 x N + 0.005 and top-1 >= N - 1 point
     probe8k / probe32k are reported only. Exit code 0 = every counted gate PASSed, 1 = a gate FAILed, 2 = tool / dump error.
  Resumable: a segment whose marker matches this container (path, size, mtime), layout, devices, split and whose files exist is skipped.
  -SelfTest: no GPU; the reference dumps against themselves (REAL_REF.md section 1 (a): canon KL 0.01386 / 95.19 %).

.PARAMETER Container   the .r4dx container (required unless -SelfTest)
.PARAMETER Layout      body layout, default trellis
.PARAMETER Devices     R4DX_Q4_DEVICES "a,b" (HIP ordinals of stage 0 / stage 1; physical 1 = headless, 0 = desktop). Default 0,1
.PARAMETER Split       R4DX_Q4_SPLIT layer boundary (0 = the model's own choice)
.PARAMETER Segments    canon (4 x 1023 rows) | probe8k | probe32k, comma list. Default canon
.PARAMETER Out         dump + report directory. Default E:\models\r4dx\q4exp\gate\<container file name without extension>
.PARAMETER Tokens      plumbing mode (tiny container): a tokens.json instead of -Segments (with -RefDir; gate A against that dump only)
.PARAMETER RefDir      plumbing mode: the reference dump directory of -Tokens
.PARAMETER MaxCtx      tool --max-ctx; default per scope (canon 2048, probe8k 8192, probe32k 32768, -Tokens: 2048)
.PARAMETER Layers      tool --layers (-1 = all; a 4-layer test container needs none because the tiny container declares 4)
.PARAMETER DryRun      print every command, run nothing
.EXAMPLE
  .\tools\qwen4exp\gate.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx
.EXAMPLE
  .\tools\qwen4exp\gate.ps1 -Container E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx -Segments canon,probe8k,probe32k
.EXAMPLE
  .\tools\qwen4exp\gate.ps1 -SelfTest
.EXAMPLE
  # plumbing proof on the tiny container (vocab 4096, no real reference)
  .\tools\qwen4exp\gate.ps1 -Container E:\models\r4dx\tiny-q4exp\container-bf16.r4dx -Layout bf16 -Tokens E:\models\r4dx\tiny-q4exp\kl\recent\tokens_used.json -RefDir E:\models\r4dx\tiny-q4exp\kl\recent -Out E:\models\r4dx\tiny-q4exp\gate_tmp
#>
[CmdletBinding()]
param(
    [string]$Container = "",
    [string]$Layout = "trellis",
    [string]$Devices = "0,1",
    [int]$Split = 0,
    [string[]]$Segments = @("canon"),
    [string]$Out = "",
    [string]$Tokens = "",
    [string]$RefDir = "",
    [int]$MaxCtx = 0,
    [int]$Layers = -1,
    [switch]$SelfTest,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = if (Test-Path "D:\venvs\r4dx-qwen4-ref\Scripts\python.exe") { "D:\venvs\r4dx-qwen4-ref\Scripts\python.exe" } else { "python" }
$GateEval = "tools\reference\qwen4exp\gate_eval.py"
$Tool = "build\win-hip\tests\model\tool_teacher_forced_logprobs.exe"
$RefRoot = "E:\models\r4dx\q4exp\ref"
$WithLock = "tools\with_lock.ps1"

# A path with whitespace survives with_lock.ps1's -Run (a command line cmd.exe re-parses) only with a literal quote.
function Q { param([string]$S) if ($S -match '\s') { '\"' + $S + '\"' } else { $S } }

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block, [string]$Shown)
    Write-Host "[gate] $Label"
    Write-Host "[gate]   > $Shown"
    if ($DryRun) { return 0 }
    & $Block | Out-Host
    return $LASTEXITCODE
}

if ($SelfTest) {
    $cmd = "$Py $GateEval --self-test"
    $env:CUDA_VISIBLE_DEVICES = ""; $env:HIP_VISIBLE_DEVICES = ""
    $rc = Invoke-Step "self-test (CPU)" { & $Py $GateEval --self-test } $cmd
    exit $rc
}

if (-not $Container) { throw "-Container is required (or -SelfTest)" }
if (-not $DryRun -and -not (Test-Path -LiteralPath $Container)) { throw "container not found: $Container" }
if ($Devices -notmatch '^\d+,\d+$') { throw "-Devices must be 'a,b' (HIP ordinals), got '$Devices'" }
if (-not $Out) { $Out = Join-Path "E:\models\r4dx\q4exp\gate" ([IO.Path]::GetFileNameWithoutExtension($Container)) }

# scope -> tokens file, the segment names to run, tool --max-ctx
$Segments = @($Segments | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$plan = @()   # {Scope, TokensFile, Segment, MaxCtx}
function Get-SegmentNames { param([string]$File)
    if (-not (Test-Path -LiteralPath $File)) { if ($DryRun) { return @("<segments of $File>") } else { throw "tokens file not found: $File" } }
    @((Get-Content -Raw -LiteralPath $File | ConvertFrom-Json).segments | ForEach-Object { $_.name })
}
if ($Tokens) {
    if (-not $RefDir) { throw "-Tokens needs -RefDir (the reference dump of that tokens file)" }
    foreach ($s in (Get-SegmentNames $Tokens)) { $plan += [pscustomobject]@{ Scope = "custom"; Tokens = $Tokens; Segment = $s; MaxCtx = $(if ($MaxCtx) { $MaxCtx } else { 2048 }) } }
    $ScopeArg = $null
} else {
    $canonTokens = Join-Path $Repo "tools\reference\kl_corpus\tokens_canon.json"
    $probeTokens = Join-Path $RefRoot "probes\tokens_probes.json"
    foreach ($sc in $Segments) {
        switch ($sc) {
            "canon"    { foreach ($s in (Get-SegmentNames $canonTokens)) { $plan += [pscustomobject]@{ Scope = $sc; Tokens = $canonTokens; Segment = $s; MaxCtx = $(if ($MaxCtx) { $MaxCtx } else { 2048 }) } } }
            "probe8k"  { $plan += [pscustomobject]@{ Scope = $sc; Tokens = $probeTokens; Segment = "probe_8k"; MaxCtx = $(if ($MaxCtx) { $MaxCtx } else { 8192 }) } }
            "probe32k" { $plan += [pscustomobject]@{ Scope = $sc; Tokens = $probeTokens; Segment = "probe_32k"; MaxCtx = $(if ($MaxCtx) { $MaxCtx } else { 32768 }) } }
            default    { throw "-Segments: unknown scope '$sc' (canon | probe8k | probe32k)" }
        }
    }
    $ScopeArg = ($Segments -join ",")
}

$saved = @{ HIP = $env:HIP_VISIBLE_DEVICES; DEV = $env:R4DX_Q4_DEVICES; SPL = $env:R4DX_Q4_SPLIT }
$ci = if (Test-Path -LiteralPath $Container) { Get-Item -LiteralPath $Container } else { $null }
$overall = 0
try {
    Write-Output "[gate] container=$Container layout=$Layout devices=$Devices split=$Split out=$Out"
    Write-Output "[gate] env for every tool run: HIP_VISIBLE_DEVICES removed (both cards), R4DX_Q4_DEVICES=$Devices$(if ($Split) { ", R4DX_Q4_SPLIT=$Split" })"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $env:R4DX_Q4_DEVICES = $Devices
        if ($Split) { $env:R4DX_Q4_SPLIT = "$Split" } else { Remove-Item Env:R4DX_Q4_SPLIT -ErrorAction SilentlyContinue }
    }
    foreach ($p in $plan) {
        $marker = Join-Path $Out "$($p.Segment).gate_done.json"
        $want = if ($ci) { "$($ci.FullName)|$($ci.Length)|$($ci.LastWriteTimeUtc.Ticks)|$Layout|$Devices|$Split" } else { "" }
        $files = (Test-Path (Join-Path $Out "$($p.Segment).logprobs.f16")) -and (Test-Path (Join-Path $Out "$($p.Segment).meta.json"))
        if (-not $DryRun -and $files -and (Test-Path $marker) -and ((Get-Content -Raw $marker | ConvertFrom-Json).identity -eq $want)) {
            Write-Output "[gate] $($p.Segment): resumed (marker matches this container; skipping the GPU run)"
            continue
        }
        $toolArgs = "--model $(Q $Container) --layout $Layout --tokens $(Q $p.Tokens) --out-dir $(Q $Out) --segment $($p.Segment) --max-ctx $($p.MaxCtx) --vision off --quiet"
        if ($Layers -ge 0) { $toolArgs += " --layers $Layers" }
        $log = Join-Path $Out "$($p.Segment).tool.log"
        $shown = "powershell -NoProfile -ExecutionPolicy Bypass -File $WithLock -Name gpu -Run `"$Tool $toolArgs`"   (lock held for this segment only; log $log)"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $rc = Invoke-Step "$($p.Scope)/$($p.Segment): teacher-forced log-probs" {
            $ErrorActionPreference = "Continue"
            & powershell -NoProfile -ExecutionPolicy Bypass -File $WithLock -Name gpu -Run "$Tool $toolArgs" 2>&1 | Tee-Object -FilePath $log | ForEach-Object { "$_" } | Write-Output
            $ErrorActionPreference = "Stop"
        } $shown
        if ($DryRun) { continue }
        if ($rc -ne 0) { Write-Output "[gate] $($p.Segment): tool exited $rc (see $log)"; $overall = 2; break }
        @{ identity = $want; segment = $p.Segment; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); finished = (Get-Date).ToString("o") } |
            ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding ASCII
        Write-Output ("[gate] {0}: done in {1:N0} s" -f $p.Segment, $sw.Elapsed.TotalSeconds)
    }
} finally {
    $env:HIP_VISIBLE_DEVICES = $saved.HIP; $env:R4DX_Q4_DEVICES = $saved.DEV; $env:R4DX_Q4_SPLIT = $saved.SPL   # $null removes the variable
}
if ($overall -ne 0) { exit $overall }

$evalArgs = @($GateEval, "--test-dir", $Out, "--out-json", (Join-Path $Out "gate_report.json"), "--out-md", (Join-Path $Out "gate_report.md"))
if ($Tokens) { $evalArgs += @("--custom-tokens", $Tokens, "--custom-ref", $RefDir) } else { $evalArgs += @("--scopes", $ScopeArg) }
$rc = Invoke-Step "evaluate gate A + gate B (CPU, no GPU lock)" {
    $env:CUDA_VISIBLE_DEVICES = ""; $env:HIP_VISIBLE_DEVICES = ""
    & $Py @evalArgs
    Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
} "$Py $($evalArgs -join ' ')"
if ($DryRun) { Write-Output "[gate] dry run: nothing was run"; exit 0 }
Write-Output "[gate] reports: $(Join-Path $Out 'gate_report.json'), $(Join-Path $Out 'gate_report.md')"
exit $rc
