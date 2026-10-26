#requires -Version 5.1
<#
.SYNOPSIS
  Nanbeige4.3 DFlash2: plain greedy vs DFlash rounds on fixed prompts (both GPUs, --tp 2). docs/nanbeige-dflash.md.

.DESCRIPTION
  For every prompt file in -PromptDir: one plain r4dx-cli run (the ground truth), then one run per -Variant with --dflash.
  Prints, per run, decode tok/s, the `[stats] dflash:` line and whether stdout is byte-identical to the plain run (with
  the first differing character when not). A variant is "name|drafter.r4dx|ENV=V;ENV=V|cli args" (env and args optional;
  the args default to "--dflash-k <K>"), e.g. "pmin5|E:\...\nanbeige43-dflash2-w4a16.r4dx||--dflash-k 7 --dflash-p-min 0.5".
  Never run it while another job owns a GPU.
#>
[CmdletBinding()]
param(
  [string]$Cli = "build\win-hip\src\cli\r4dx-cli.exe",
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx",
  [string]$PromptDir = "tools\nanbeige\dflash_prompts",
  [string[]]$Variant = @("w4a16|$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\nanbeige\nanbeige43-dflash2-w4a16.r4dx|"),
  [int]$K = 7,
  [int]$MaxTokens = 512,
  [ValidateSet("on", "off")][string]$Think = "on",
  [double]$Temperature = 0,
  [string]$OutDir = "$env:TEMP\r4dx-nanbeige-dflash-ab"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
New-Item -ItemType Directory -Force $OutDir | Out-Null
Remove-Item Env:\HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue  # --tp 2 needs both devices
$ErrorActionPreference = "Continue"  # r4dx-cli's load log goes to stderr
$common = @("--model", $Model, "--layout", "w8a16", "--tp", "2", "--max-ctx", "8192", "--temperature", "$Temperature",
            "--max-tokens", "$MaxTokens", "--think", $Think, "--stats")

function Stat([string]$errFile, [string]$pattern) {
  $l = Get-Content $errFile | Select-String $pattern | Select-Object -Last 1
  if ($l) { $l.Line } else { "" }
}
function Rate([string]$errFile) {
  $l = Stat $errFile "\[stats\] prefill:"
  if ($l -match "decode: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\)") { return [double]$Matches[3] }
  return 0.0
}

foreach ($pf in Get-ChildItem $PromptDir -Filter *.txt | Sort-Object Name) {
  $name = $pf.BaseName
  $plainOut = "$OutDir\$name.plain.out"; $plainErr = "$OutDir\$name.plain.err"
  & $Cli @common --prompt-file $pf.FullName > $plainOut 2> $plainErr
  $plainRate = Rate $plainErr
  $ta = [IO.File]::ReadAllText($plainOut)
  Write-Output ("[ab] {0,-8} plain   {1,6:F2} tok/s" -f $name, $plainRate)
  foreach ($v in $Variant) {
    $parts = $v.Split("|")
    $vn = $parts[0]; $drafter = $parts[1]; $envs = if ($parts.Count -gt 2) { $parts[2] } else { "" }
    $extra = if ($parts.Count -gt 3 -and $parts[3]) { $parts[3].Split(" ") } else { @("--dflash-k", "$K") }
    $set = @()
    foreach ($kv in ($envs.Split(";") | Where-Object { $_ })) { $e = $kv.Split("=", 2); Set-Item "Env:\$($e[0])" $e[1]; $set += $e[0] }
    $o = "$OutDir\$name.$vn.out"; $er = "$OutDir\$name.$vn.err"
    & $Cli @common --prompt-file $pf.FullName --dflash $drafter @extra > $o 2> $er
    foreach ($s in $set) { Remove-Item "Env:\$s" -ErrorAction SilentlyContinue }
    $tb = [IO.File]::ReadAllText($o)
    $same = $ta -eq $tb
    $diff = ""
    if (-not $same) {
      $n = [Math]::Min($ta.Length, $tb.Length); $d = 0
      while ($d -lt $n -and $ta[$d] -eq $tb[$d]) { $d++ }
      $diff = " (first diff at char $d of $($ta.Length))"
    }
    $r = Rate $er
    $dl = (Stat $er "\[stats\] dflash:") -replace "^\[stats\] dflash: ", ""
    $err = Stat $er "error"
    Write-Output ("[ab] {0,-8} {1,-7} {2,6:F2} tok/s  x{3:F2}  identical={4}{5}  {6} {7}" -f $name, $vn, $r,
                  $(if ($plainRate -gt 0) { $r / $plainRate } else { 0 }), $same, $diff, $dl, $err)
  }
}
