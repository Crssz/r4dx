# quant2 GPU stage 2 (HIP device 1, run by the user). Logs + summary.json under -OutDir.
#
#   1. the three GPU tests stage 1 skipped for missing golden fixtures (now copied in)
#   2. gate G5, rotation correctness: 4-layer bf16 containers converted unrotated / q2a / q2ab from
#      the same checkpoint, teacher-forced on the KL corpus, rotated vs unrotated KL. Rotation is
#      exact in math, so only bf16 rounding separates them: the gate is mean KL <= 2 x the bf16
#      reference's own self-noise (3.9e-4 nats, docs/validation.md) = 7.8e-4.
#   3. gate G3, the Q1 pilot: tools/quant2/q1_pilot.ps1 -Kl (bf16 reference dump, v6, pilot)
param(
  [string]$OutDir = 'D:\models\r4dx\quant2-gpu2',
  [string]$L4Dir = 'D:\models\r4dx\quant2-l4'
)
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[stage2] stop r4dx-server first' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:HIP_VISIBLE_DEVICES = '1'
$tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe'
$py = 'C:\Users\pay20\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe'
$tokens = Join-Path $repo 'tools\reference\kl_corpus\tokens.json'
$summary = [ordered]@{}

function Step([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[stage2] $name -> $log"
  $t0 = Get-Date
  & $cmd *> $log
  $summary[$name] = [ordered]@{ exit = $LASTEXITCODE; minutes = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1) }
  Write-Host ("[stage2] {0}: exit {1}" -f $name, $LASTEXITCODE)
  $summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $OutDir 'summary.json')
}

Step 'ctest_skipped' {
  ctest --test-dir (Join-Path $repo 'build\win-hip') -R '^(test_gdn_layer|test_final_lm_head|test_attn_layer)$' --output-on-failure
}

foreach ($r in 'none', 'q2a', 'q2ab') {
  Step "g5_tf_$r" {
    & $tool --model (Join-Path $L4Dir "l4-$r.r4dx") --layout bf16 --layers 4 --tokens $tokens `
      --out-dir (Join-Path $OutDir "g5_$r") --max-ctx 4096 --vision off
  }
}
foreach ($r in 'q2a', 'q2ab') {
  Step "g5_kl_$r" {
    & $py (Join-Path $repo 'tools\reference\kl_report.py') --ref-dir (Join-Path $OutDir 'g5_none') `
      --test-dir (Join-Path $OutDir "g5_$r") --tokens $tokens --out (Join-Path $OutDir "g5_kl_$r.json")
  }
  $f = Join-Path $OutDir "g5_kl_$r.json"
  if (Test-Path $f) {
    $o = (Get-Content $f -Raw | ConvertFrom-Json).overall
    $summary["g5_$r"] = [ordered]@{ mean_kl = $o.mean_kl; top1_pct = $o.top1_agreement_pct; pass = ($o.mean_kl -le 7.8e-4) }
  }
}

Step 'q1_pilot_kl' { & (Join-Path $PSScriptRoot 'q1_pilot.ps1') -Kl -Tool $tool }
foreach ($n in 'v6', 'q1pilot') {
  $f = "D:\models\r4dx\kl-q1\kl_$n.json"
  if (Test-Path $f) {
    $o = (Get-Content $f -Raw | ConvertFrom-Json).overall
    $summary["q1_$n"] = [ordered]@{ mean_kl = $o.mean_kl; top1_pct = $o.top1_agreement_pct }
  }
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $OutDir 'summary.json')
$summary | ConvertTo-Json -Depth 4
