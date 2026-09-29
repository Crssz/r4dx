# HISTORICAL (quant2 stage-1 driver): its defaults name files that were removed before and on 2026-09-29
# (docs/huihui.md "Retired files"); pass every path explicitly to re-run it.
# quant2 GPU stage 1 (HIP device 1, run by the user): everything on the GPU that does not wait for
# a CPU-built container. Each step logs to -OutDir and the exit codes go to summary.json, so the
# results can be read back without the console. A failing test does NOT stop the Hessian capture
# (it does not depend on the new kernels); the capture itself is last because it is the long one.
#
#   1. GPU ctests for the new Q2 kernels, the Q3 grouped dispatch, and the default-path regressions
#   2. libr4d's grouped w4a16 GEMM correctness test (all three groups vs a CPU dequant reference)
#   3. the Q1 Hessian capture (~30-60 min, ~48 GiB)
#
# -Tokenizer (docs/quant2.md 3.4): hessian-v1, the default -HessianDir, is an hf-auto (AutoTokenizer)
# set, so this script captures hf-auto by default. A canonical capture needs -Tokenizer canonical AND
# another -HessianDir; a canonical set under the hessian-v1 name is refused.
param(
  [string]$OutDir = 'D:\models\r4dx\quant2-gpu',
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' }),
  [string]$HessianDir = 'D:\models\r4dx\hessian-v1',
  [ValidateSet('hf-auto', 'canonical')][string]$Tokenizer = 'hf-auto',
  [switch]$SkipCapture
)
$ErrorActionPreference = 'Continue'
if (-not $SkipCapture -and $Tokenizer -ne 'hf-auto' -and (Split-Path -Leaf $HessianDir) -eq 'hessian-v1') {
  throw "[stage1] hessian-v1 is the hf-auto set (docs/quant2.md 3.4): pass another -HessianDir for a $Tokenizer capture"
}
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[stage1] stop r4dx-server first' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:HIP_VISIBLE_DEVICES = '1'
$summary = [ordered]@{}

function Step([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[stage1] $name -> $log"
  $t0 = Get-Date
  & $cmd *> $log
  $summary[$name] = [ordered]@{ exit = $LASTEXITCODE; minutes = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1) }
  Write-Host ("[stage1] {0}: exit {1}" -f $name, $LASTEXITCODE)
  $summary | ConvertTo-Json -Depth 3 | Set-Content -Encoding utf8 (Join-Path $OutDir 'summary.json')
}

$tests = 'test_rotate_residual|test_attn_gate_mul_hadamard|test_pick_tuning|test_tp_loader|' +
         'test_dflash_draft_weights|test_core|test_silu_mul|test_forward_smoke|test_final_lm_head|' +
         'test_mtp|test_keep_bf16|test_tp_emulation|test_gdn_layer|test_attn_layer'
Step 'ctest_gpu' { ctest --test-dir (Join-Path $repo 'build\win-hip') -R "^($tests)$" --output-on-failure }

Step 'libr4d_w4a16_groups' {
  Push-Location (Join-Path $repo 'third_party\libr4d')
  try { $env:PYTHONPATH = 'build-win'; & $Python test_w4a16_gemm_groups.py } finally { Pop-Location }
}

if (-not $SkipCapture) {
  Step 'hessian_capture' {
    Push-Location $repo
    try { & $Python -u tools\reference\hessian_capture.py --out-dir $HessianDir --tokenizer $Tokenizer } finally { Pop-Location }
  }
}
Write-Host "[stage1] done: $(Join-Path $OutDir 'summary.json')"
