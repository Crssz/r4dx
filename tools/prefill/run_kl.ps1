# tools/prefill/run_kl.ps1 -- long-prefix teacher-forced log-prob dump for the prefill KL harness.
#
# Runs tests/model/tool_teacher_forced_logprobs --tail-rows <tail> over the segments of a
# make_kl_tokens.py file: each segment's (T - tail)-token prefix goes through ONE chunked Prefill call
# (the path under test), then the tail is fed token by token (DecodeStep, or one-token Prefill calls
# with -TailPath prefill) and only the tail's rows are written ([tail, V] fp16 per segment, ~121 MiB
# at tail 256) plus a sidecar with prefill_seconds. Compare two such directories with kl_compare.py.
#
#   .\tools\prefill\run_kl.ps1 -Device 1 -OutDir <models-root>\r4dx\prefill-m0\kl\dense
#   .\tools\prefill\run_kl.ps1 -Device 1 -Segment prose_8k,recall_8k -OutDir ...\kl\smoke
#   .\tools\prefill\run_kl.ps1 -Tp 2 -OutDir <models-root>\r4dx\prefill-m0\kl\dense-tp2
#   .\tools\prefill\run_kl.ps1 -Device 0 -Tool <other build>\tool_teacher_forced_logprobs.exe -OutDir ...\kl\variant
#
# HIP_VISIBLE_DEVICES is set explicitly: -Device N at TP=1, '0,1' with -Tp 2. Without -Segment one
# tool process loads the model once and runs every segment; with -Segment a,b one process per
# segment. --max-ctx = longest selected segment + 64, rounded up to 1024. Refuses when another r4dx
# GPU job runs unless -AllowOthers.
param(
  [int]$Device = 1,
  [ValidateSet(1, 2)][int]$Tp = 1,
  [string]$Tokens = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\prefill-m0\kl\tokens_long.json",
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string[]]$Segment = @(),
  [ValidateSet('decode', 'prefill')][string]$TailPath = 'decode',
  [string]$Model = "$(if ($env:R4DX_MODELS_ROOT) { $env:R4DX_MODELS_ROOT } else { 'E:\models' })\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx",
  [string]$Layout = 'trellis',
  [string]$Tool = '',
  [string[]]$ExtraArgs = @(),
  [string]$Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe',
  [switch]$AllowOthers
)
$ErrorActionPreference = 'Stop'
[string]$SegmentList = ($Segment -join ',')  # -Segment a,b arrives as an array
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Tool) { $Tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
foreach ($p in $Tool, $Model, $Tokens) { if (-not (Test-Path $p)) { throw "[run_kl] missing $p" } }
$others = @(Get-Process r4dx-server, r4dx-cli, tool_teacher_forced_logprobs -ErrorAction SilentlyContinue)
if ($others.Count -gt 0) {
  $desc = ($others | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', '
  if (-not $AllowOthers) { throw "[run_kl] other GPU jobs are running ($desc); stop them or pass -AllowOthers" }
  Write-Output "[run_kl] WARNING: other GPU jobs running ($desc) -- make sure they use another device"
}

# Segment lengths and tail from the tokens file (python: the file is tens of MB of JSON).
$info = & $Python -c @"
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
want = set(s for s in sys.argv[2][4:].split(',') if s)
segs = [s for s in d['segments'] if not want or s['name'] in want]
if not segs: sys.exit('no matching segment')
print(d.get('tail_rows', 256), max(len(s['token_ids']) for s in segs), ','.join(s['name'] for s in segs))
"@ $Tokens "sel=$SegmentList"
if ($LASTEXITCODE -ne 0) { throw "[run_kl] cannot read $Tokens" }
$tail, $maxT, $names = $info.Trim().Split(' ')
$maxCtx = [int]([math]::Ceiling(([int]$maxT + 64) / 1024.0) * 1024)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$argList = @('--model', $Model, '--layout', $Layout, '--tokens', $Tokens, '--out-dir', $OutDir,
             '--max-ctx', "$maxCtx", '--tail-rows', "$tail", '--tail-path', $TailPath, '--quiet')
if ($Tp -eq 2) { $env:HIP_VISIBLE_DEVICES = '0,1'; $argList += @('--tp', '2') } else { $env:HIP_VISIBLE_DEVICES = "$Device" }
$argList += $ExtraArgs
$segList = $names.Split(',')
[ordered]@{ started = (Get-Date).ToString('s'); tool = $Tool; args = ($argList -join ' '); segments = $segList;
  hip_visible_devices = $env:HIP_VISIBLE_DEVICES; git_head = (git -C $repo rev-parse HEAD);
  git_dirty = [bool](git -C $repo status --porcelain) } | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $OutDir 'run_info.json')

# Start-Process (not &): PowerShell 5.1 turns native stderr lines into terminating errors under
# ErrorActionPreference Stop. stdout -> <name>.log, stderr -> <name>.err.log.
function Invoke-Tool([string[]]$a, [string]$name) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[run_kl] HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) $Tool $($a -join ' ') -> $log"
  $p = Start-Process -FilePath $Tool -ArgumentList $a -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError (Join-Path $OutDir "$name.err.log")
  Get-Content $log | Where-Object { $_ -match '^\[(prefill|segment|total|PASS|model|vram)' } | ForEach-Object { Write-Host $_ }
  Get-Content (Join-Path $OutDir "$name.err.log") | Where-Object { $_ -match 'FAIL|rror' } | ForEach-Object { Write-Host $_ }
  return $p.ExitCode
}

$rc = 0
if (-not $SegmentList) {
  # Every segment in one process (one model load).
  $code = Invoke-Tool $argList 'tool'
  if ($code -ne 0) { Write-Output "[run_kl] FAILED (exit $code)"; $rc = 1 }
} else {
  # The tool takes one --segment: one process per selected segment.
  foreach ($s in $segList) {
    $code = Invoke-Tool ($argList + @('--segment', $s)) $s
    if ($code -ne 0) { Write-Output "[run_kl] segment $s FAILED (exit $code)"; $rc = 1 }
  }
}
exit $rc
