# docs/quant2.md gate G3: the Q1 pilot. v6's recipe with LDLQ on the MLP linears only, against v6
# itself, both measured by rung 4 (tool_teacher_forced_logprobs + kl_report.py) against a freshly
# regenerated bf16 reference.
#
#   -Convert   CPU only: build the pilot container from the Hessians in -HessianDir.
#   -Kl        GPU (HIP device 1): bf16 reference dump, both teacher-forced runs, both reports.
#
# Everything is logged under -OutDir so a user-run can be read back. Stops at the first failure.
param(
  [switch]$Convert,
  [switch]$Kl,
  [string]$Checkpoint = 'C:\AI\models\Qwen3.8-27B',
  [string]$HessianDir = 'D:\models\r4dx\hessian-v1',
  [string]$Pilot = 'D:\models\r4dx\qwen38-27b-q1pilot.r4dx',
  [string]$V6 = 'D:\models\r4dx\qwen38-27b-v6.r4dx',
  [string]$OutDir = 'D:\models\r4dx\kl-q1',
  [string]$Python = 'C:\Users\pay20\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe',
  [string]$Tool = 'C:\Users\pay20\dev\r4dx\build\win-hip\tests\model\tool_teacher_forced_logprobs.exe'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
New-Item -ItemType Directory -Force $OutDir | Out-Null
$tokens = 'tools\reference\kl_corpus\tokens.json'

function Run([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[q1] $name -> $log"
  # Windows PowerShell 5.1 turns every stderr line of a native program into an error record, and
  # under 'Stop' the first one (a transformers fallback warning, the engine's VRAM line) would abort
  # the script. Success is judged by the exit code alone.
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd *> $log } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[q1] $name failed (exit $LASTEXITCODE), see $log" }
}

if (-not ($Convert -or $Kl)) { throw 'pass -Convert and/or -Kl' }

if ($Convert) {
  if (-not (Test-Path (Join-Path $HessianDir 'hessian.json'))) {
    throw "[q1] $HessianDir has no hessian.json (capture missing or its gates failed)"
  }
  $conv = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe'
  Run 'convert_pilot' {
    & $conv --input $Checkpoint --output $Pilot `
      --layouts w4a16 --lm-head 4bit --no-bf16 --mtp on --vision on `
      --kv-calib D:\models\r4dx\qwen38-27b.kvcalib-full.json `
      --quant search --imatrix D:\models\r4dx\qwen38-27b.imatrix.npz `
      --keep-bf16 '^text\.layers\.[0-9]+\.attn\.[kv]$' `
      --hessian-dir $HessianDir --ldlq 'mlp\.'
  }
}

if ($Kl) {
  if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[q1] stop r4dx-server first' }
  $env:HIP_VISIBLE_DEVICES = '1'
  $ref = Join-Path $OutDir 'ref'
  if (-not (Test-Path (Join-Path $ref 'reference_run.json'))) {
    Run 'ref_dump' { & $Python tools\reference\full_logits_golden.py --device cuda --tokens $tokens --out-dir $ref }
  }
  foreach ($c in @(@{n = 'v6'; m = $V6 }, @{n = 'q1pilot'; m = $Pilot })) {
    $dir = Join-Path $OutDir $c.n
    New-Item -ItemType Directory -Force $dir | Out-Null  # the tool does not create --out-dir
    Run "tf_$($c.n)" { & $Tool --model $c.m --layout w4a16 --tokens $tokens --out-dir $dir --max-ctx 4096 --vision off }
    Run "kl_$($c.n)" { & $Python tools\reference\kl_report.py --ref-dir $ref --test-dir $dir --tokens $tokens --out (Join-Path $OutDir "kl_$($c.n).json") }
  }
  foreach ($n in 'v6', 'q1pilot') {
    $j = Get-Content (Join-Path $OutDir "kl_$n.json") -Raw | ConvertFrom-Json
    Write-Host ("[q1] {0,-8} mean KL {1:N5}  top-1 {2:N2}%" -f $n, $j.overall.mean_kl, $j.overall.top1_agreement_pct)
  }
}
