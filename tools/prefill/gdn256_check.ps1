# tools/prefill/gdn256_check.ps1 -- GPU validation of the GDN 256-row super-chunk change (docs/trellis-m256.md
# "GDN sequence ops"): the GDN sequence ops run once per 256-row super-chunk (default) instead of four
# 64-row sub-slices (R4DX_GDN_SLICE=64), and the opt-in r4d_gdn_conv_prep2 (R4DX_GDN_CONV=2) instead of
# the original r4d_gdn_conv_prep (default). A 64-row Model (R4DX_PREFILL_CHUNK=0) ignores both knobs and
# runs the pre-change kernels. Every step must show the same bytes every way; the profile and the [stats]
# lines are the perf A/B.
#
#   .\tools\prefill\gdn256_check.ps1                      # device 1, everything
#   .\tools\prefill\gdn256_check.ps1 -Lengths 8k -SkipIdentity
#
# Steps (each on HIP device -Device only; all output under build\logs\gdn256\<timestamp>\, the summary in
# gdn256_check.log there):
#   1. tests\model\test_gdn_seq256_identity  (one call vs 4 x 64, conv v1 vs v2, byte for byte)
#   2. tests\model\test_gdn_layer            (layer 0 against the golden, as before)
#   3. tests\kernels\test_gdn_chunk_scan     (the scan against the fp64 recurrence, as before)
#   4. tests\model\test_prefill_chunk_identity, three times: defaults, R4DX_GDN_CONV=2 and
#      R4DX_GDN_SLICE=64 (logits + KV / GDN state digest of the 256-row Model against the 64-row one, which
#      is the true pre-change path every time; -SkipIdentity skips it)
#   5. r4dx-cli greedy (temperature 0, -MaxTokens tokens) at each length, four configurations:
#        new   = defaults (one call per super-chunk, r4d_gdn_conv_prep)
#        conv2 = R4DX_GDN_CONV=2 (one call per super-chunk, r4d_gdn_conv_prep2)
#        old   = R4DX_GDN_SLICE=64 (the pre-change 256-row path)
#        c64   = R4DX_PREFILL_CHUNK=0 (the 64-row chunk path, the bit-identity reference)
#      the generated text's SHA-256 must match across all four; the [stats] prefill line of each is the
#      unprofiled TTFT A/B (gate the change on these, not on the profile shares).
#   6. r4dx-cli --profile-prefill at 8k, new, conv2 and old (the gdn.conv_prep / kkt_solve / chunk_scan /
#      gated_rmsnorm rows of the three tables).
# Exit 0 only when every test passed (or skipped for missing data) and every hash matched.
param(
  [int]$Device = 1,
  [string[]]$Lengths = @('8k', '32k'),
  [int]$MaxTokens = 64,
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx",
  [string]$Layout = 'trellis',
  [string]$TasksDir = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\prefill-m0\tasks",
  [string]$ProfileLength = '8k',
  [switch]$SkipIdentity,
  [switch]$AllowOthers
)
$ErrorActionPreference = 'Stop'
[string]$LengthList = ($Lengths -join ',')  # a,b arrives as an array
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build = Join-Path $repo 'build\win-hip'
$cli = Join-Path $build 'src\cli\r4dx-cli.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = Join-Path $repo "build\logs\gdn256\$stamp"
New-Item -ItemType Directory -Force $outDir | Out-Null
$summary = Join-Path $outDir 'gdn256_check.log'
function Say([string]$s) { Write-Output $s; $s | Out-File -Append -Encoding utf8 $summary }

$others = @(Get-Process r4dx-server, r4dx-cli, tool_teacher_forced_logprobs -ErrorAction SilentlyContinue)
if ($others.Count -gt 0 -and -not $AllowOthers) {
  throw "[gdn256] other GPU jobs are running ($(($others | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ')); pass -AllowOthers if they use another device"
}
$env:HIP_VISIBLE_DEVICES = "$Device"
$gitHead = (git -C $repo rev-parse --short HEAD)
Say "[gdn256] $stamp git $gitHead HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) -> $outDir"

$knobs = @('R4DX_GDN_SLICE', 'R4DX_GDN_CONV', 'R4DX_PREFILL_CHUNK')
# Runs $exe with exactly the env overrides in $envSet (every other knob unset), stdout/stderr to files.
function Invoke-WithEnv([string]$exe, [string[]]$argList, [hashtable]$envSet, [string]$logBase) {
  $saved = @{}
  foreach ($k in $knobs) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process'); [Environment]::SetEnvironmentVariable($k, $null, 'Process') }
  foreach ($k in $envSet.Keys) { [Environment]::SetEnvironmentVariable($k, $envSet[$k], 'Process') }
  try {
    $t0 = Get-Date
    # Start-Process (not &): PowerShell 5.1 would wrap every stderr line in an ErrorRecord.
    $sp = @{ FilePath = $exe; NoNewWindow = $true; Wait = $true; PassThru = $true
             RedirectStandardOutput = "$logBase.out.txt"; RedirectStandardError = "$logBase.err.txt" }
    if ($argList.Count -gt 0) { $sp.ArgumentList = $argList }
    $p = Start-Process @sp
    return [pscustomobject]@{ Exit = $p.ExitCode; Wall = ((Get-Date) - $t0).TotalSeconds
                              Out = "$logBase.out.txt"; Err = "$logBase.err.txt" }
  } finally {
    foreach ($k in $knobs) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
  }
}
function EnvText([hashtable]$e) { if ($e.Count -eq 0) { 'defaults' } else { ($e.Keys | Sort-Object | ForEach-Object { "$_=$($e[$_])" }) -join ' ' } }

$fail = 0

# ---- 1-4: tests -------------------------------------------------------------------------------------
$tests = @(
  @{ Name = 'test_gdn_seq256_identity'; Exe = 'tests\model\test_gdn_seq256_identity.exe'; Env = @{} },
  @{ Name = 'test_gdn_layer'; Exe = 'tests\model\test_gdn_layer.exe'; Env = @{} },
  @{ Name = 'test_gdn_chunk_scan'; Exe = 'tests\kernels\test_gdn_chunk_scan.exe'; Env = @{} }
)
if (-not $SkipIdentity) {
  $tests += @{ Name = 'test_prefill_chunk_identity'; Exe = 'tests\model\test_prefill_chunk_identity.exe'; Env = @{} }
  $tests += @{ Name = 'test_prefill_chunk_identity_conv2'; Exe = 'tests\model\test_prefill_chunk_identity.exe'
               Env = @{ R4DX_GDN_CONV = '2' } }
  $tests += @{ Name = 'test_prefill_chunk_identity_gdnslice64'; Exe = 'tests\model\test_prefill_chunk_identity.exe'
               Env = @{ R4DX_GDN_SLICE = '64' } }
}
foreach ($t in $tests) {
  $exe = Join-Path $build $t.Exe
  if (-not (Test-Path $exe)) { Say "[FAIL] $($t.Name): $exe not built"; $fail++; continue }
  $r = Invoke-WithEnv $exe @() $t.Env (Join-Path $outDir $t.Name)
  $verdict = if ($r.Exit -eq 0) { 'PASS' } elseif ($r.Exit -eq 77) { 'SKIP' } else { 'FAIL' }
  if ($verdict -eq 'FAIL') { $fail++ }
  Say ("[{0}] {1} ({2}) exit {3}, {4:N1}s; {5}" -f $verdict, $t.Name, (EnvText $t.Env), $r.Exit, $r.Wall, $r.Out)
  Get-Content $r.Out | Select-String -Pattern 'PASS|FAIL|SKIP|differ' | Select-Object -Last 40 | ForEach-Object { Say "    $($_.Line)" }
}

# ---- 5: greedy r4dx-cli, text hash across the three configurations ------------------------------------
$lenTokens = @{ '4k' = 4096; '8k' = 8192; '16k' = 16384; '32k' = 32768; '64k' = 65536; '128k' = 131072 }
$configs = [ordered]@{
  new = @{}
  conv2 = @{ R4DX_GDN_CONV = '2' }
  old = @{ R4DX_GDN_SLICE = '64' }
  c64 = @{ R4DX_PREFILL_CHUNK = '0' }
}
function CliArgs([string]$len, [switch]$Prof) {
  $prompt = Join-Path $TasksDir "prompts\ttft_$len.txt"
  if (-not (Test-Path $prompt)) { throw "[gdn256] missing $prompt (run tools\prefill\build_tasks.py)" }
  $maxCtx = [int]([math]::Ceiling(($lenTokens[$len] + 256 + $MaxTokens) / 1024.0) * 1024)
  $a = @('--model', $Model, '--layout', $Layout, '--prompt-file', $prompt, '--max-ctx', "$maxCtx",
         '--vision', 'off', '--temperature', '0', '--max-tokens', "$MaxTokens", '--stats')
  if ($Prof) { $a += '--profile-prefill' }
  return , $a
}
if (-not (Test-Path $cli)) { Say "[FAIL] $cli not built"; $fail++ }
else {
  foreach ($len in $LengthList.Split(',')) {
    $len = $len.Trim()
    $hashes = [ordered]@{}
    foreach ($name in $configs.Keys) {
      $r = Invoke-WithEnv $cli (CliArgs $len) $configs[$name] (Join-Path $outDir "cli_${len}_$name")
      $text = Get-Content $r.Err -Raw
      $m = [regex]::Match($text, '\[stats\] prefill: (\d+) tok in ([\d.]+)s \(([\d.]+) tok/s\)')
      $hash = if (Test-Path $r.Out) { (Get-FileHash -Algorithm SHA256 $r.Out).Hash.Substring(0, 16) } else { 'none' }
      $hashes[$name] = $hash
      $stats = if ($m.Success) { "{0} tok in {1}s ({2} tok/s)" -f $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value } else { 'no [stats] prefill line' }
      if ($r.Exit -ne 0) { $fail++ }
      Say ("[cli] {0} {1,-5} ({2}): exit {3}, text sha256 {4}, prefill {5}" -f $len, $name, (EnvText $configs[$name]), $r.Exit, $hash, $stats)
    }
    $distinct = @($hashes.Values | Select-Object -Unique)
    if ($distinct.Count -eq 1 -and $distinct[0] -ne 'none') { Say "[PASS] $len greedy text identical across new / conv2 / old / c64" }
    else { Say "[FAIL] $len greedy text differs: $(($hashes.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')"; $fail++ }
  }

  # ---- 6: --profile-prefill, new vs conv2 vs old ---------------------------------------------------
  foreach ($name in @('new', 'conv2', 'old')) {
    $r = Invoke-WithEnv $cli (CliArgs $ProfileLength -Prof) $configs[$name] (Join-Path $outDir "profile_${ProfileLength}_$name")
    if ($r.Exit -ne 0) { $fail++ }
    Say ("[profile] {0} {1} ({2}): exit {3}; table in {4}" -f $ProfileLength, $name, (EnvText $configs[$name]), $r.Exit, $r.Err)
    Get-Content $r.Err, $r.Out -ErrorAction SilentlyContinue |
      Select-String -Pattern 'gdn\.(conv_prep|kkt_solve|chunk_scan|gated_rmsnorm)|gpu_sum|wall' |
      ForEach-Object { Say "    $($_.Line)" }
  }
}

Say $(if ($fail -eq 0) { "[gdn256] ALL PASS ($outDir)" } else { "[gdn256] $fail FAILED ($outDir)" })
exit $(if ($fail -eq 0) { 0 } else { 1 })

