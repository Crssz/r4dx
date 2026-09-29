# Rung-4 KL gate for one container (docs/huihui.md "Frozen values"): teacher-forced log-probs of the
# runtime (tool_teacher_forced_logprobs, canonical tokens, --max-ctx 4096, --vision off) against the
# model's OWN bf16 reference dump, judged against the FROZEN numbers of the production container --
# the Huihui abliterated trellis mix4.5m: mean KL 0.00788 nats, top-1 agreement 95.70%.
#
#   .\tools\quant2\kl_rung4.ps1 -OutDir D:\models\r4dx\huihui\kl\rt-check
#   .\tools\quant2\kl_rung4.ps1 -Device 0 -OutDir <dir>       # HIP device 0 (the desktop card; ~3 min)
#   .\tools\quant2\kl_rung4.ps1 -Model <other.r4dx> -Layout w4a16 -RefDir <its bf16 ref> -ExpectKl 0 -NoGate ...
#
# A KL number is only meaningful against the bf16 reference of the SAME model: the default -RefDir is
# huihui\kl-ref (tools/reference/full_logits_golden.py on the Huihui checkpoint, tokens_canon.json).
# The base Qwen3.8-27B reference (kl-canon\ref) belongs to the retired base containers.
#
# The gate (exit 1 when it fails):
#   1. mean KL rounds to -ExpectKl at 5 decimals and top-1 to -ExpectTop1 at 2 (kl_report.py's overall);
#   2. unless -CompareDir '' the *.logprobs.f16 of every segment are byte-identical (SHA-256) to that
#      directory's -- the default is the recorded run of 2026-09-28 (huihui\kl\rt-mix45m), so a kernel or
#      loader change that moves a single log-prob bit is caught even when the mean would not show it.
# -NoGate only prints the numbers. One GPU process at a time: refuses while another r4dx job runs.
param(
  [string]$Model = 'D:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx',
  [string]$Layout = 'trellis',
  [string]$RefDir = 'D:\models\r4dx\huihui\kl-ref',
  [string]$Tokens = 'tools\reference\kl_corpus\tokens_canon.json',
  [Parameter(Mandatory = $true)][string]$OutDir,
  [int]$Device = 1,
  [double]$ExpectKl = 0.00788,
  [double]$ExpectTop1 = 95.70,
  [string]$CompareDir = 'D:\models\r4dx\huihui\kl\rt-mix45m',
  [string]$Tool = '',
  [string]$Python = 'C:\Users\pay20\AppData\Local\Programs\Python\Python312\python.exe',
  [switch]$NoGate
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not $Tool) { $Tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
foreach ($p in $Tool, $Model, $Tokens, (Join-Path $RefDir 'reference_run.json')) {
  if (-not (Test-Path $p)) { throw "[kl_rung4] $p not found" }
}
# One job per device: jobs on device 0 and device 1 may run side by side (the caller staggers their
# starts by >= 30 s); a production server holds ~28 GiB of device 1, so it must not be running.
if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[kl_rung4] r4dx-server is running; stop it first' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:HIP_VISIBLE_DEVICES = "$Device"
$env:PYTHONIOENCODING = 'utf-8'

Write-Host ("[kl_rung4] {0:HH:mm:ss} tool_teacher_forced_logprobs on device {1}: {2} ({3})" -f (Get-Date), $Device, $Model, $Layout)
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
try {
  & $Tool --model $Model --layout $Layout --tokens $Tokens --out-dir $OutDir --max-ctx 4096 --vision off *> (Join-Path $OutDir 'tool.log')
  $rc = $LASTEXITCODE
} finally { $ErrorActionPreference = $prev }
if ($rc -ne 0) { throw "[kl_rung4] tool_teacher_forced_logprobs exited $rc (see $OutDir\tool.log)" }

$klJson = Join-Path $OutDir 'kl_canon.json'
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
try {
  & $Python tools\reference\kl_report.py --ref-dir $RefDir --test-dir $OutDir --tokens $Tokens --out $klJson *> (Join-Path $OutDir 'kl_report.log')
  $rc = $LASTEXITCODE
} finally { $ErrorActionPreference = $prev }
if ($rc -ne 0) { throw "[kl_rung4] kl_report.py exited $rc (see $OutDir\kl_report.log)" }

$sumPy = Join-Path $OutDir 'read_kl.py'
[IO.File]::WriteAllText($sumPy, @'
import json, sys
j = json.load(open(sys.argv[1], encoding="utf-8"))
o = j["overall"]
seg = " ".join(f"{s['name']}={s['mean_kl']:.5f}" for s in j["segments"])
print(f"{o['mean_kl']:.9f} {o['top1_agreement_pct']:.6f} {seg}")
'@, (New-Object Text.UTF8Encoding($false)))
$line = (& $Python $sumPy $klJson) -join ' '
$parts = $line -split ' '
$kl = [double]::Parse($parts[0], [Globalization.CultureInfo]::InvariantCulture)
$top1 = [double]::Parse($parts[1], [Globalization.CultureInfo]::InvariantCulture)
Write-Host ("[kl_rung4] mean KL {0:F5}  top-1 {1:F2}%   segments: {2}" -f $kl, $top1, ($parts[2..($parts.Count - 1)] -join ' '))

$fail = @()
if (-not $NoGate) {
  if ([math]::Round($kl, 5) -ne [math]::Round($ExpectKl, 5)) { $fail += ("mean KL {0:F5} != frozen {1:F5}" -f $kl, $ExpectKl) }
  if ([math]::Round($top1, 2) -ne [math]::Round($ExpectTop1, 2)) { $fail += ("top-1 {0:F2}% != frozen {1:F2}%" -f $top1, $ExpectTop1) }
  if ($CompareDir) {
    $ref = @(Get-ChildItem (Join-Path $CompareDir '*.logprobs.f16') | Sort-Object Name)
    if ($ref.Count -eq 0) { $fail += "no *.logprobs.f16 in $CompareDir" }
    foreach ($f in $ref) {
      $c = Join-Path $OutDir $f.Name
      if (-not (Test-Path $c)) { $fail += "$($f.Name) missing in $OutDir"; continue }
      $same = ((Get-FileHash -Algorithm SHA256 $f.FullName).Hash -eq (Get-FileHash -Algorithm SHA256 $c).Hash)
      Write-Host ("[kl_rung4]   {0,-28} {1} vs {2}" -f $f.Name, $(if ($same) { 'byte-identical' } else { 'DIFFERENT' }), $CompareDir)
      if (-not $same) { $fail += "$($f.Name) differs from $CompareDir" }
    }
  }
}
if ($fail.Count -gt 0) { Write-Host ("[kl_rung4] FAILED: " + ($fail -join '; ')); exit 1 }
Write-Host $(if ($NoGate) { '[kl_rung4] done (no gate)' } else { '[kl_rung4] PASSED: KL and top-1 equal the frozen values' + $(if ($CompareDir) { ', log-probs byte-identical' } else { '' }) })
exit 0
