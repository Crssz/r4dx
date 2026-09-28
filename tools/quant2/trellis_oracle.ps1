# Trellis accuracy oracle (docs/trellis.md 14): quantize the 400 decoder linears with
# trellis_quant.py, then run rung-4 KL (weights only) against the bf16 reference on HIP device 1.
# Sequential and resumable: quantize-model skips layers it already wrote with the same code, and a
# point whose kl_canon.json exists is not re-measured. Each step logs to <KlDir>\<step>.log, and
# <KlDir>\summary.json is rewritten after every KL so partial results can be read while it runs.
#
#   .\tools\quant2\trellis_oracle.ps1                       # the default plan below
#   .\tools\quant2\trellis_oracle.ps1 -Points K4m,mix4.5m   # a subset
#
# A point is K<rate>[m] (a uniform rate; the m suffix = --hessian-basis matched) or mix<bpw>[m]
# (EXL3's 4/5 allocation at that bpw over K4[m] + K5[m], e.g. mix4.5m or mix4.25m; a manifest only).
# A mix quantizes its sources first.
#
# -Python defaults to $env:R4DX_REFERENCE_VENV\Scripts\python.exe when that is set, else python on
# PATH. -HessianDir is hessian-v2 (docs/quant2.md 3.3), moved from C:\AI\r4dx-hessian on 2026-09-28.
param(
  [string[]]$Points = @('K4m', 'mix4.5m', 'K3.5m', 'K4'),
  [string]$QDir = 'D:\models\r4dx\trellis-q',
  [string]$HessianDir = 'D:\models\r4dx\hessian\hessian-v2',
  [string]$KlDir = 'D:\models\r4dx\kl-trellis',
  [string]$RefDir = 'D:\models\r4dx\kl-canon\ref',
  [string]$Tokens = 'tools\reference\kl_corpus\tokens_canon.json',
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' })
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$env:HIP_VISIBLE_DEVICES = '1'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force $QDir, $KlDir | Out-Null
if (-not (Test-Path (Join-Path $RefDir 'reference_run.json'))) { throw "[trellis] no reference in $RefDir" }

function Run([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $KlDir "$name.log"
  Write-Host ("[trellis] {0:HH:mm} {1} -> {2}" -f (Get-Date), $name, $log)
  # PS 5.1 makes native stderr lines error records; judge by exit code only.
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd *> $log } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[trellis] $name failed (exit $LASTEXITCODE), see $log" }
}

function Assert-GpuFree {
  if (Get-Process r4dx-server, r4dx-cli, tool_teacher_forced_logprobs, r4dx-convert -ErrorAction SilentlyContinue) {
    throw '[trellis] an r4dx process is running; stop it first'
  }
  $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'full_logits_golden|hessian_capture|trellis_quant' }
  if ($busy) { throw "[trellis] another reference python is running (pid $($busy.ProcessId -join ','))" }
}

function Complete([string]$dir) {
  $m = Join-Path $dir 'weights_override.json'
  if (-not (Test-Path $m)) { return $false }
  # Not ConvertFrom-Json: the manifest's tensor records carry both 'K' (rate) and 'k' (rows), and
  # PS 5.1 refuses keys that differ only in case.
  $c = & $Python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['complete'])" $m
  return ($LASTEXITCODE -eq 0 -and "$c".Trim() -eq 'True')
}

function Quantize([string]$point) {
  if ($point -notmatch '^K([0-9.]+)(m?)$') { throw "[trellis] bad point $point" }
  $dir = Join-Path $QDir $point
  if (Complete $dir) { return }
  $basis = if ($Matches[2]) { 'matched' } else { 'exl3' }
  $k = $Matches[1]
  Assert-GpuFree
  Run "quantize_$point" { & $Python tools\reference\trellis_quant.py quantize-model --device cuda --K $k --hessian-basis $basis --hessian-dir $HessianDir --out-dir $dir }
  if (-not (Complete $dir)) { throw "[trellis] $dir incomplete after quantize-model" }
}

$summaryPath = Join-Path $KlDir 'summary.json'
foreach ($p in $Points) {
  $out = Join-Path $KlDir $p
  $klPath = Join-Path $out 'kl_canon.json'
  if (Test-Path $klPath) { Write-Host "[trellis] $p already measured"; continue }
  $src = Join-Path $QDir $p
  if ($p -match '^mix([0-9.]+)(m?)$') {
    $bpw = $Matches[1]; $sfx = $Matches[2]
    Quantize "K4$sfx"; Quantize "K5$sfx"
    if (-not (Complete $src)) {
      Run "mix_$p" { & $Python tools\reference\trellis_quant.py mix --bpw $bpw --src (Join-Path $QDir "K4$sfx") --src (Join-Path $QDir "K5$sfx") --out-dir $src }
    }
  } else {
    Quantize $p
  }
  Assert-GpuFree
  # A partial dump from an interrupted run would be refused (or worse, mixed); start it clean.
  if (Test-Path $out) { Remove-Item -Recurse -Force $out }
  Run "golden_$p" { & $Python tools\reference\full_logits_golden.py --device cuda --tokens $Tokens --weights-override $src --out-dir $out }
  $klTmp = Join-Path $KlDir "kl_$p.tmp.json"
  Run "kl_$p" { & $Python tools\reference\kl_report.py --ref-dir $RefDir --test-dir $out --tokens $Tokens --out $klTmp }
  Move-Item -Force $klTmp $klPath

  # reference_run.json has the same K/k records, so the summary is built in Python too.
  $summarize = @'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1]); rows = []
for d in sorted(p for p in root.iterdir() if (p / "kl_canon.json").exists()):
    j = json.loads((d / "kl_canon.json").read_text(encoding="utf-8"))
    w = json.loads((d / "reference_run.json").read_text(encoding="utf-8")).get("weights_override", {})
    o = j["overall"]
    rows.append({"point": d.name, "mean_kl": o["mean_kl"], "top1_pct": o["top1_agreement_pct"], "p99_kl": o["p99_kl"],
                 "segments": {s["name"]: round(s["mean_kl"], 5) for s in j["segments"]},
                 "bpw": w.get("summary_used", {}).get("bpw"), "checked": w.get("checked", {}).get("count")})
pathlib.Path(sys.argv[2]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
r = next(x for x in rows if x["point"] == sys.argv[3])
print(f"mean KL {r['mean_kl']:.5f}  top-1 {r['top1_pct']:.2f}%  bpw {r['bpw']:.4f}  checked {r['checked']}  {r['segments']}")
'@
  # A file, not -c: PS 5.1 strips the embedded double quotes from native arguments.
  $sumPy = Join-Path $KlDir 'summarize.py'
  [IO.File]::WriteAllText($sumPy, $summarize, (New-Object Text.UTF8Encoding($false)))
  $line = & $Python $sumPy $KlDir $summaryPath $p
  if ($LASTEXITCODE -ne 0) { throw "[trellis] summary failed for $p" }
  Write-Host ("[trellis] {0:HH:mm} {1,-9} {2}" -f (Get-Date), $p, $line)
}
Write-Host '[trellis] done'
