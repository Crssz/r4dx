# Trellis containers (docs/trellis-kernel.md M3): r4dx-convert --trellis-from, CPU only.
#
# Imports an oracle directory's trellis bits (tools/reference/trellis_quant.py quantize-model or mix,
# pinned by the manifest's sha256) for all 400 decoder body linears, and writes everything else
# exactly as q2ab_hv2_q3 (docs/quant2.md 7) does, minus the q2ab rotation that trellis excludes:
#
#   lm_head          w4a16 g32, LDLQ against hessian-v2's "lm_head" key (the q3 rule "^lm_head$=32")
#   mtp.attn.qg|o, mtp.mlp.gate_up|down
#                    w4a16 at the default group g64, LDLQ against hessian-v2's mtp.* keys
#   mtp.attn.k|v, mtp.fc, mtp norms, gdn.in_proj_a|b, conv1d, A_log, dt_bias, every norm, embeddings,
#   vision           bf16 / fp32 as in every container (unrotated: the norms keep their weights)
#   attn.k|v_descale the full KV calibration
#
# None of those tensors is rotated in q2ab_hv2_q3 either (they sit outside the rotated stack), and
# their Hessians are the unrotated captures, so with these flags their bytes equal q2ab_hv2_q3's
# (checked by tools/quant2/compare_containers.py --tensors, see -Compare). Deliberate differences
# from the q3 recipe: no --rotate; no --keep-bf16 of attn.k|v (the oracle quantized all 400 linears);
# the q3 group rules on body linears are dropped (a trellis linear has no w4a16 layout), only
# "^lm_head$=32" stays; --lm-head w4a16 instead of 4bit (--layout trellis loads the w4a16 head, so the
# other-layout copies of the head would be dead weight). -LmHead bf16 writes the A2 twin (same body bytes).
#
#   .\tools\quant2\trellis_convert.ps1 -Oracle mix4.5m -Output D:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx
#   .\tools\quant2\trellis_convert.ps1 -Oracle K4m -Output D:\models\r4dx\huihui-qwen38-27b-abl-trellis-k4m.r4dx
#   .\tools\quant2\trellis_convert.ps1 -Oracle K4m -LmHead bf16 -Output <...>-k4m-lmbf16.r4dx
#
# Every default below is the HUIHUI abliterated model's (docs/huihui.md, D:\models\r4dx\huihui\RECIPE.md):
# its checkpoint, hessian-v2, trellis-q oracle dir, KL runs and imatrix / kvcalib. The base Qwen3.8-27B
# files were retired on 2026-09-29 (docs/huihui.md); another model needs every path passed explicitly.
#
# The log goes to -LogDir\<output name>.log; the run fails unless the converter exits 0 (its
# --trellis-verify full reconstruction check included) and prints decode bytes at the end.
#
# The manifest pin (--trellis-manifest-sha256) is the manifest whose KL was MEASURED: by default
# weights_override.manifest_sha256 of the A0 KL run's reference_run.json (-KlRoot\<oracle>\, written by
# full_logits_golden.py --weights-override), so a manifest edited or regenerated since that run is
# refused instead of silently re-pinned. -ManifestSha256 overrides it (e.g. an oracle with no KL run).
#
# -Exe defaults to build\win-hip-m3's converter when that build exists (the M3 build dir while M2
# and M3 share this worktree), else build\win-hip's; the path, time and sha256 of the one used are
# printed and logged.
param(
  [Parameter(Mandatory = $true)][string]$Oracle,          # a directory under -OracleRoot, or a path
  [Parameter(Mandatory = $true)][string]$Output,
  [ValidateSet('w4a16', 'bf16')][string]$LmHead = 'w4a16',
  [string]$OracleRoot = 'D:\models\r4dx\huihui\trellis-q',
  [string]$KlRoot = 'D:\models\r4dx\huihui\kl',
  [string]$ManifestSha256 = '',
  [string]$Checkpoint = 'D:\models\Huihui-Qwen3.8-27B-abliterated',
  [string]$HessianDir = 'D:\models\r4dx\huihui\hessian-v2',
  [string]$KvCalib = 'D:\models\r4dx\huihui-qwen38-27b-abl.kvcalib-full.json',
  [string]$Imatrix = 'D:\models\r4dx\huihui-qwen38-27b-abl.imatrix.npz',
  [string]$LogDir = 'D:\models\r4dx\huihui\logs',
  [string]$Exe = '',
  [int]$Threads = 32,
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' }),
  [string]$Compare = ''                                   # a container to byte-compare the non-trellis tensors against (same model + calibration only)
)
$ErrorActionPreference = 'Stop'
# Script scope: the log's *> redirection writes UTF-8 (PS 5.1's own default is UTF-16), so the
# Add-Content lines below and every reader of the log see one encoding.
$PSDefaultParameterValues = @{ 'Out-File:Encoding' = 'utf8' }
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# Everything this script runs in Python (the pin, decode_bytes.py, compare_containers.py) is stdlib
# only, so any Python 3 does: -Python defaults to $env:R4DX_REFERENCE_VENV's interpreter when that is
# set, else python on PATH, and falls back to python on PATH when the one given is absent.
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if (-not $cmd) { throw "[trellis] no Python at $Python nor on PATH" }
  Write-Host "[trellis] $Python is absent: using $($cmd.Source)"
  $Python = $cmd.Source
}
if (-not $Exe) {
  $Exe = Join-Path $repo 'build\win-hip-m3\src\convert\r4dx-convert.exe'
  if (-not (Test-Path $Exe)) { $Exe = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe' }
}
if (-not (Test-Path $Exe)) { throw "[trellis] no converter at $Exe" }
$odir = if (Test-Path $Oracle) { $Oracle } else { Join-Path $OracleRoot $Oracle }
$manifest = Join-Path $odir 'weights_override.json'
if (-not (Test-Path $manifest)) { throw "[trellis] no weights_override.json in $odir" }
if ($ManifestSha256) {
  $sha = $ManifestSha256.ToLower()
  $pinFrom = '-ManifestSha256'
} else {
  # Python, not ConvertFrom-Json: PS 5.1 refuses JSON keys that differ only in case ('K' / 'k').
  $ref = Join-Path (Join-Path $KlRoot (Split-Path -Leaf $odir)) 'reference_run.json'
  if (-not (Test-Path $ref)) {
    throw "[trellis] no KL run at $ref to take the manifest pin from; pass -ManifestSha256 <sha of the manifest whose KL you measured>"
  }
  $code = @'
import json, os, sys
wo = json.load(open(sys.argv[1], encoding="utf-8")).get("weights_override") or {}
m, s = wo.get("manifest"), wo.get("manifest_sha256")
if not s or not m:
    sys.exit("reference_run.json has no weights_override.manifest / manifest_sha256")
if os.path.normcase(os.path.abspath(m)) != os.path.normcase(os.path.abspath(sys.argv[2])):
    sys.exit("its KL run measured " + m + ", not " + sys.argv[2])
print(s)
'@
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { $sha = ($code | & $Python - $ref $manifest 2>&1 | ForEach-Object { "$_" }) -join "`n" } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0 -or $sha -notmatch '^[0-9a-f]{64}$') { throw "[trellis] manifest pin from ${ref}: $sha" }
  $pinFrom = $ref
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
$log = Join-Path $LogDir ([IO.Path]::GetFileNameWithoutExtension($Output) + '.log')
$exeInfo = Get-Item $Exe
$exeSha = (Get-FileHash $Exe -Algorithm SHA256).Hash.ToLower()
Write-Host "[trellis] manifest pin $sha (from $pinFrom)"
Write-Host ("[trellis] converter {0} ({1:yyyy-MM-dd HH:mm}, sha256 {2})" -f $Exe, $exeInfo.LastWriteTime, $exeSha)

$cargs = @('--input', $Checkpoint, '--output', $Output,
           '--trellis-from', $odir, '--trellis-manifest-sha256', $sha, '--trellis-verify', 'full',
           '--layouts', 'w4a16', '--lm-head', $LmHead, '--no-bf16', '--mtp', 'on', '--vision', 'on',
           '--kv-calib', $KvCalib, '--quant', 'search', '--imatrix', $Imatrix,
           '--hessian-dir', $HessianDir, '--ldlq', '.', '--threads', "$Threads")
if ($LmHead -eq 'w4a16') { $cargs += @('--w4a16-group-rule', '^lm_head$=32') }

$free = (Get-PSDrive ([IO.Path]::GetPathRoot((Resolve-Path (Split-Path -Parent $Output)).Path).Substring(0, 1))).Free
Write-Host ("[trellis] {0} -> {1} ({2:N0} GB free), log {3}" -f $odir, $Output, ($free / 1e9), $log)
Write-Host "[trellis] $Exe $($cargs -join ' ')"
$t0 = Get-Date
# PS 5.1 makes native stderr lines error records; judge by exit code only.
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
try { & $Exe @cargs *> $log } finally { $ErrorActionPreference = $prev }
$rc = $LASTEXITCODE
Add-Content -Path $log -Encoding utf8 -Value @(
  ("[trellis_convert.ps1] converter {0} ({1:yyyy-MM-dd HH:mm}, sha256 {2}), exit {3}" -f $Exe, $exeInfo.LastWriteTime, $exeSha, $rc),
  "[trellis_convert.ps1] manifest pin $sha (from $pinFrom)")
$min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
if ($rc -ne 0) { throw "[trellis] r4dx-convert failed (exit $rc) after $min min, see $log" }
Write-Host "[trellis] converted in $min min"
Select-String -Path $log -Pattern 'trellis verify: \d+/\d+|hashed in|wrote ' | ForEach-Object { Write-Host "  $($_.Line)" }
& $Python (Join-Path $repo 'tools\quant2\decode_bytes.py') $Output --by-class
if ($Compare) {
  # The tensors outside the trellis body must be byte-identical to the w4a16 reference container's.
  & $Python (Join-Path $repo 'tools\quant2\compare_containers.py') $Output $Compare --no-metadata `
      --tensors '^(lm_head\.w4a16\.|mtp\.|vision\.|text\.embed_tokens$)'
}
