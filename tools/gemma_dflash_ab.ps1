#requires -Version 5.1
<#
.SYNOPSIS
  Gemma 4 DFlash v1 drafter: convert (CPU), then the greedy-identity + acceptance A/B against plain decode (GPU).
  docs/gemma4-plan.md D-7, docs/dflash2.md section 13.6.

.DESCRIPTION
  Step 1 (CPU, always): r4dx-convert --dflash-hf D:\models\z-lab-gemma4-12B-it-DFlash -> the dflash2 container
  (identity conv, zero selector). Skipped when the output exists.
  Step 2 (GPU, ONLY with -RunGpu; never run it while another job owns the device): for every prompt, r4dx-cli
  --temperature 0 once WITHOUT --dflash (ground truth) and once WITH it, SHA-256 of the raw stdout compared, and the
  `[stats] dflash:` line (tokens per round, accept rate) printed. Without -RunGpu the exact commands are printed.
  A byte mismatch is not automatically a bug: verify rows use a different GEMM shape than single-row decode (the
  docs/mtp.md / tools/validate_dflash.ps1 precedent). Read the diff position: one token flipping to a coherent
  continuation is the near-tie signature; dropped / duplicated / garbage tokens are a bookkeeping bug.
  A/B knobs: -LayerOffset 0 reconverts with --dflash-target-layer-offset 0; -EmbedScale takes a list (default 1,62: both arms; set via R4DX_DFLASH_EMBED_SCALE).
#>
[CmdletBinding()]
param(
  [string]$Convert = "",  # r4dx-convert.exe; default: build\win-hip\src\convert\r4dx-convert.exe
  [string]$Cli = "",      # r4dx-cli.exe;     default: build\win-hip\src\cli\r4dx-cli.exe
  [string]$HfDir = "D:\models\z-lab-gemma4-12B-it-DFlash",
  [string]$Model = "D:\models\r4dx\huihui-gemma\bf16.r4dx",
  [string]$Out = "D:\models\r4dx\gemma4-12b-dflash-bf16.r4dx",
  [int]$K = 7,
  [int]$MaxTokens = 256,
  [ValidateSet(0, 1)][int]$LayerOffset = 1,
  [double[]]$EmbedScale = @(1.0, 62.0),  # z-lab's inference path is 1.0 (docs/dflash2.md 13.1); 62 = Gemma-scaled rows: both arms run by default
  [string[]]$Prompts = @(
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a C++ function that reverses a singly linked list and explain its complexity.",
    "Summarise the causes of the French Revolution in five bullet points."),
  [ValidateSet(0, 1)][int]$Device = 1,  # HIP device for the GPU step
  [switch]$RunGpu
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $Convert) { $Convert = "build\win-hip\src\convert\r4dx-convert.exe" }
if (-not $Cli) { $Cli = "build\win-hip\src\cli\r4dx-cli.exe" }

if ($LayerOffset -eq 0) { $Out = $Out -replace "\.r4dx$", "-off0.r4dx" }
if (-not (Test-Path $Out)) {
  & $Convert --dflash-hf $HfDir --out $Out --layout bf16 --dflash-target-layer-offset $LayerOffset --threads 4
  if ($LASTEXITCODE -ne 0) { throw "r4dx-convert --dflash-hf failed" }
}
Write-Output "[ab] drafter container: $Out (layer offset $LayerOffset, embed scales $($EmbedScale -join ','), k $K)"

# PowerShell 5.1 turns a native program's stderr lines (r4dx-cli's load log) into terminating errors under "Stop".
$ErrorActionPreference = "Continue"
$common = @("--model", $Model, "--temperature", "0", "--max-tokens", "$MaxTokens", "--stats")  # --prompt is a one-shot chat turn
$i = 0
foreach ($p in $Prompts) {
  $i++
  $plain = @($common + @("--prompt", $p))
  $spec = @($common + @("--prompt", $p, "--dflash", $Out, "--dflash-k", "$K"))
  if (-not $RunGpu) {
    Write-Output "[ab] prompt $i (GPU, not run; pass -RunGpu):"
    Write-Output ("  `$env:HIP_VISIBLE_DEVICES='$Device'; $Cli " + ($plain -join " "))
    foreach ($sc in $EmbedScale) {
      Write-Output ("  `$env:HIP_VISIBLE_DEVICES='$Device'; `$env:R4DX_DFLASH_EMBED_SCALE='$sc'; $Cli " + ($spec -join " "))
    }
    continue
  }
  $env:HIP_VISIBLE_DEVICES = "$Device"
  $a = & $Cli @plain 2>$null
  $ha = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($a -join "`n")))).Replace("-", "")
  foreach ($sc in $EmbedScale) {
    $env:R4DX_DFLASH_EMBED_SCALE = "$sc"
    $errFile = [System.IO.Path]::GetTempFileName()
    $b = & $Cli @spec 2>$errFile
    $stats = (Get-Content $errFile | Select-String "\[stats\] dflash:" | Select-Object -Last 1)
    Remove-Item $errFile -ErrorAction SilentlyContinue
    Remove-Item Env:\R4DX_DFLASH_EMBED_SCALE -ErrorAction SilentlyContinue
    $hb = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes(($b -join "`n")))).Replace("-", "")
    $same = $ha -eq $hb
    Write-Output ("[ab] prompt {0} embed_scale {1}: identical={2}  {3}" -f $i, $sc, $same, $stats)
    if (-not $same) {
      $ta = $a -join "`n"; $tb = $b -join "`n"
      $n = [Math]::Min($ta.Length, $tb.Length); $d = 0
      while ($d -lt $n -and $ta[$d] -eq $tb[$d]) { $d++ }
      Write-Output ("      first difference at char {0} of {1}/{2}" -f $d, $ta.Length, $tb.Length)
    }
  }
}
