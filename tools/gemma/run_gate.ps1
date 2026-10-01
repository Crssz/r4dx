# One-line Gemma M1 gate run (docs/gemma4-plan.md section 9, items 9-13): dumps the engine's teacher-forced
# log-probs on the chat corpus (full) and the raw corpus (first 512 tokens per segment, matching the
# reference dumps' --max-tokens 512) into ONE out dir, then runs kl_report.py --gate gemma-fp32.
#
#   # bf16 container, noise-relative rule:
#   $env:R4DX_GEMMA_KV='bf16'; .\tools\gemma\run_gate.ps1 -Model D:\models\r4dx\huihui-gemma\<bf16>.r4dx -Layout bf16 -OutDir D:\models\r4dx\huihui-gemma\kl\r4dx-x
#   # quantized container, KL increment over the r4dx bf16 dump (<= 0.01):
#   .\tools\gemma\run_gate.ps1 -Model <q>.r4dx -Layout trellis -OutDir <dir> -BaseDir D:\models\r4dx\huihui-gemma\kl\r4dx-bf16kv-f32res
#
# KV / residual / attention settings are the engine's env vars, set by the caller (or via -Kv / -Resid /
# -Attn, which just set R4DX_GEMMA_KV / R4DX_GEMMA_RESID / R4DX_GEMMA_ATTN for this run). The tool prints
# the effective [gemma] kv/resid line in its header. This runs the GPU tool: ask before launching it.
# -SkipDump re-scores an existing OutDir (kl_report only, no GPU). Exit status is the gate's.
param(
  [string]$Model,
  [string]$Layout = 'bf16',
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$BaseDir = '',
  [string]$Kv = '',
  [string]$Resid = '',
  [string]$Attn = '',
  [int]$Device = -1,
  [int]$MaxCtx = 4096,
  [int]$RawMaxTokens = 512,
  [string]$TruthDir = 'D:\models\r4dx\huihui-gemma\kl\fp32\truth',
  [string]$NoiseDir = 'D:\models\r4dx\huihui-gemma\kl\fp32\bf16sdpa',
  [string]$Chat = 'tools\reference\kl_corpus\chat_gemma.json',
  [string]$Raw = 'tools\reference\kl_corpus\tokens_gemma.json',
  [string]$Tool = '',
  [string]$Python = '',
  # Extra arguments for tool_teacher_forced_logprobs, e.g. -ToolArgs '--tp','2','--tp-mode','real' for the TP=2 run of the gate
  # (M1b-2; with --tp 2 real leave -Device unset so both GPUs are visible).
  [string[]]$ToolArgs = @(),
  [switch]$SkipDump
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not $Tool) { $Tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
if (-not $Python) {
  $Python = 'D:\venvs\r4dx-gemma-ref\Scripts\python.exe'
  if (-not (Test-Path $Python)) { $Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe' }
}
foreach ($p in $Chat, $Raw, $TruthDir, $NoiseDir, $Python) { if (-not (Test-Path $p)) { throw "[run_gate] $p not found" } }
if ($BaseDir -and -not (Test-Path $BaseDir)) { throw "[run_gate] -BaseDir $BaseDir not found" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

if ($Kv) { $env:R4DX_GEMMA_KV = $Kv }
if ($Resid) { $env:R4DX_GEMMA_RESID = $Resid }
if ($Attn) { $env:R4DX_GEMMA_ATTN = $Attn }
if ($Device -ge 0) { $env:HIP_VISIBLE_DEVICES = "$Device" }
Write-Host "[run_gate] KV=$env:R4DX_GEMMA_KV RESID=$env:R4DX_GEMMA_RESID ATTN=$env:R4DX_GEMMA_ATTN HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES (unset = engine defaults)"

if (-not $SkipDump) {
  if (-not $Model) { throw '[run_gate] -Model is required (unless -SkipDump)' }
  foreach ($p in $Tool, $Model) { if (-not (Test-Path $p)) { throw "[run_gate] $p not found" } }
  $common = @('--model', $Model, '--layout', $Layout, '--out-dir', $OutDir, '--max-ctx', $MaxCtx, '--vision', 'off', '--quiet') + $ToolArgs
  & $Tool @common --tokens $Chat
  if ($LASTEXITCODE -ne 0) { throw "[run_gate] tool failed on the chat corpus (exit $LASTEXITCODE)" }
  & $Tool @common --tokens $Raw --max-tokens $RawMaxTokens
  if ($LASTEXITCODE -ne 0) { throw "[run_gate] tool failed on the raw corpus (exit $LASTEXITCODE)" }
}

$gate = @('tools\reference\kl_report.py', '--gate', 'gemma-fp32', '--truth-dir', $TruthDir, '--noise-dir', $NoiseDir,
          '--test-dir', $OutDir, '--tokens', $Chat, '--raw-tokens', $Raw, '--raw-max-tokens', $RawMaxTokens,
          '--out', (Join-Path $OutDir 'gate.json'))
if ($BaseDir) { $gate += @('--base-dir', $BaseDir) }
& $Python @gate
exit $LASTEXITCODE
