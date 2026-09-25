# docs/quant2.md section 5 (Q3): the per-tensor w4a16 group sweep. Each candidate moves ONE set of
# linears -- a tensor class in one depth half, or lm_head -- from the build's default w4a16 group to
# another (default: 32 and 128) on top of the v6-style recipe, and is measured on its own by rung 4
# (tool_teacher_forced_logprobs + kl_report.py) against a bf16 reference. The result is the
# candidates.json tools/quant2/alloc_groups.py ranks by nats of KL per GiB and fills a byte budget
# from, printing the --w4a16-group-rule flags of the chosen set.
#
#   -Convert   CPU only: r4dx-convert each candidate container (w4a16 only) plus the no-rule
#              baseline, with --w4a16-group-rule "<bases_regex>=<group>". Writes <name>.meta.json
#              next to the logs: the rule and __metadata__.r4dx_convert_run.w4a16_group_extra_bytes,
#              the candidate's exact byte delta against the baseline.
#   -Kl        GPU (HIP device 1): bf16 reference dump if -RefDir has none, then per candidate the
#              teacher-forced run and kl_report.py. Then candidates.json and alloc_groups.py.
#
# A container is ~20 GB and there are 31 of them, so -Convert -Kl together runs candidate by
# candidate (convert, measure, delete the container and its log-prob dump) and -Convert alone stops
# when the disk gets below -MinFreeGB. Everything is logged under -OutDir; a rerun skips every
# candidate whose kl_<name>.json and <name>.meta.json already exist, so an interrupted sweep
# resumes. The converter writes <name>.r4dx.partial, renamed to <name>.r4dx only once it exited 0
# and <name>.meta.json is written: a container or a KL without its meta.json is a failed conversion
# and is redone, never measured or collected. Stops at the first failure.
#
#   .\tools\quant2\group_sweep.ps1 -Convert -Kl
#   .\tools\quant2\group_sweep.ps1 -Convert -Kl -Only '^mlp\.down' -RefDir D:\models\r4dx\kl-q1\ref
#   .\tools\quant2\group_sweep.ps1 -Kl      # re-collect candidates.json + rerun the allocation
#                                           # (measures any converted, unmeasured one; skips the rest)
#   ... -ExtraArgs '--hessian-dir','D:\models\r4dx\hessian-v1','--ldlq','mlp\.'   # on top of Q1
#
# -CandidateFile takes [{"name", "bases_regex", "group"}, ...] instead of the default list below.
param(
  [switch]$Convert,
  [switch]$Kl,
  [string]$Checkpoint = 'C:\AI\models\Qwen3.8-27B',
  [string]$OutDir = 'D:\models\r4dx\q3-sweep',
  [string]$ContainerDir = '',   # default: -OutDir
  [string]$RefDir = '',         # default: <OutDir>\ref; point it at an existing rung-4 ref to reuse it
  [string[]]$Recipe = @(
    '--layouts', 'w4a16', '--lm-head', 'w4a16', '--no-bf16', '--mtp', 'on', '--vision', 'on',
    '--kv-calib', 'D:\models\r4dx\qwen38-27b.kvcalib-full.json',
    '--quant', 'search', '--imatrix', 'D:\models\r4dx\qwen38-27b.imatrix.npz',
    '--keep-bf16', '^text\.layers\.[0-9]+\.attn\.[kv]$'),
  [string[]]$ExtraArgs = @(),   # appended to -Recipe for every container, baseline included
  [int[]]$Groups = @(32, 128),
  [string]$CandidateFile = '',
  [string]$Only = '',           # regex over candidate names; the baseline is always included
  [double]$BudgetGib = 0,       # alloc_groups.py --budget-gib; 0 = the baseline's own weights
  [double]$BaselineWeightsGib = 16.4065,  # used only if the baseline's tf log has no weights= line
  [double]$MinFreeGB = 40,
  [switch]$KeepContainers,
  [switch]$KeepLogprobs,
  [string]$Python = 'C:\Users\pay20\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe',
  [string]$Tool = ''            # default: this worktree's tool_teacher_forced_logprobs.exe
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
New-Item -ItemType Directory -Force $OutDir | Out-Null
if (-not $ContainerDir) { $ContainerDir = $OutDir }
New-Item -ItemType Directory -Force $ContainerDir | Out-Null
if (-not $RefDir) { $RefDir = Join-Path $OutDir 'ref' }
# Per-tensor groups need a runtime that reads __metadata__.quant.w4a16.groups: this worktree's build,
# not the main checkout's (which q1_pilot.ps1 can use, its containers being single-group).
if (-not $Tool) { $Tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
$conv = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe'
$tokens = 'tools\reference\kl_corpus\tokens.json'

function Run([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[q3] $name -> $log"
  # PS 5.1 makes native stderr lines error records; under 'Stop' the first one would abort the
  # sweep. Success is judged by the exit code alone.
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd *> $log } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[q3] $name failed (exit $LASTEXITCODE), see $log" }
}

if (-not ($Convert -or $Kl)) { throw 'pass -Convert and/or -Kl' }

# The build default group (the one a candidate at that group would not change): read from the same
# CMakeCache.txt the converter and the tool were built from, like the other scripts' -Model lookup.
$cache = Join-Path $repo 'build\win-hip\CMakeCache.txt'
$defaultGroup = 64
if (Test-Path $cache) {
  $m = Select-String -Path $cache -Pattern '^R4DX_W4A16_GROUP:STRING=(\d+)' | Select-Object -First 1
  if ($m) { $defaultGroup = [int]$m.Matches[0].Groups[1].Value }
}

# ---- candidates --------------------------------------------------------------------------------
# Default: 7 body classes x 2 depth halves x -Groups, plus lm_head (one tensor, no halves) x -Groups.
# attn.qg / attn.o exist on the full-attention layers only (every 4th); attn.k / attn.v are bf16 in
# the recipe (--keep-bf16), so they are not candidates; mtp.* keeps the default group.
$candidates = @()
if ($CandidateFile) {
  # The parentheses matter: Windows PowerShell 5.1's ConvertFrom-Json emits a JSON array as ONE
  # pipeline object, so without them ForEach-Object would see the whole array as $_.
  $candidates = @((Get-Content $CandidateFile -Raw | ConvertFrom-Json) | ForEach-Object {
      [pscustomobject]@{ name = $_.name; bases_regex = $_.bases_regex; group = [int]$_.group } })
} else {
  $classes = [ordered]@{
    'gdn.in_proj_qkv' = 'gdn\.in_proj_qkv'; 'gdn.in_proj_z' = 'gdn\.in_proj_z'
    'gdn.out_proj' = 'gdn\.out_proj'; 'attn.qg' = 'attn\.qg'; 'attn.o' = 'attn\.o'
    'mlp.gate_up' = 'mlp\.gate_up'; 'mlp.down' = 'mlp\.down'
  }
  $halves = [ordered]@{ 'L0-31' = '([0-9]|[12][0-9]|3[01])'; 'L32-63' = '(3[2-9]|[45][0-9]|6[0-3])' }
  foreach ($g in $Groups) {
    foreach ($c in $classes.Keys) {
      foreach ($h in $halves.Keys) {
        $candidates += [pscustomobject]@{
          name = "$c.$h.g$g"; bases_regex = "^text\.layers\.$($halves[$h])\.$($classes[$c])$"; group = $g }
      }
    }
    $candidates += [pscustomobject]@{ name = "lm_head.g$g"; bases_regex = '^lm_head$'; group = $g }
  }
}
foreach ($c in $candidates) {
  if (@(32, 64, 128) -notcontains $c.group) { throw "[q3] candidate $($c.name): group $($c.group) is not 32/64/128" }
}
$skipped = @($candidates | Where-Object { $_.group -eq $defaultGroup })
if ($skipped) { Write-Host "[q3] skipping $($skipped.Count) candidate(s) at the build default group $defaultGroup" }
$candidates = @($candidates | Where-Object { $_.group -ne $defaultGroup })
if ($Only) { $candidates = @($candidates | Where-Object { $_.name -match $Only }) }
$all = @([pscustomobject]@{ name = 'baseline'; bases_regex = ''; group = $defaultGroup }) + $candidates
Write-Host "[q3] default group $defaultGroup; $($candidates.Count) candidate(s) + baseline"

# __metadata__ of a container: 8-byte little-endian header length, then the JSON header.
function Read-Metadata([string]$path) {
  $fs = [IO.File]::OpenRead($path)
  try {
    $b = New-Object byte[] 8
    if ($fs.Read($b, 0, 8) -ne 8) { throw "[q3] $path is too short" }
    $n = [int][BitConverter]::ToUInt64($b, 0)
    $h = New-Object byte[] $n
    $got = 0
    while ($got -lt $n) { $got += $fs.Read($h, $got, $n - $got) }
    return ([Text.Encoding]::UTF8.GetString($h) | ConvertFrom-Json).__metadata__
  } finally { $fs.Close() }
}

function Free-GB([string]$dir) {
  $root = [IO.Path]::GetPathRoot((Resolve-Path $dir).Path)
  return (New-Object IO.DriveInfo $root).AvailableFreeSpace / 1GB
}

if ($Kl) {
  if (Get-Process r4dx-server -ErrorAction SilentlyContinue) { throw '[q3] stop r4dx-server first' }
  $env:HIP_VISIBLE_DEVICES = '1'
  if (-not (Test-Path (Join-Path $RefDir 'reference_run.json'))) {
    Run 'ref_dump' { & $Python tools\reference\full_logits_golden.py --device cuda --tokens $tokens --out-dir $RefDir }
  }
}

foreach ($c in $all) {
  $n = $c.name
  $container = Join-Path $ContainerDir "$n.r4dx"
  $partial = "$container.partial"
  $metaPath = Join-Path $OutDir "$n.meta.json"
  $klPath = Join-Path $OutDir "kl_$n.json"
  # <name>.meta.json is written only after the converter exited 0 and its header was read back, and
  # the container only gets its final name after that. ContainerWriter sizes the whole file before
  # the emit pass, so a failed or interrupted conversion leaves a full-size, valid-header,
  # zero-weight file: without meta.json neither a container nor a KL measured on one counts.
  $converted = (Test-Path $container) -and (Test-Path $metaPath)
  $measured = (Test-Path $klPath) -and (Test-Path $metaPath)
  if ((Test-Path $klPath) -and -not $measured) {
    Write-Host "[q3] $n has kl_$n.json but no $n.meta.json (a failed conversion was measured) -- discarding it"
    Remove-Item -Force $klPath
  }

  if ($Convert -and -not $measured -and -not $converted) {
    foreach ($stale in @($container, $partial)) {
      if (Test-Path $stale) { Write-Host "[q3] removing unfinished $stale"; Remove-Item -Force $stale }
    }
    $free = Free-GB $ContainerDir
    if ($free -lt $MinFreeGB) {
      throw "[q3] $([math]::Round($free, 1)) GB free on $ContainerDir (< -MinFreeGB $MinFreeGB) before $n -- run -Convert -Kl together (it deletes each container once measured) or free space"
    }
    $rule = @()
    if ($c.bases_regex) { $rule = @('--w4a16-group-rule', "$($c.bases_regex)=$($c.group)") }
    Run "convert_$n" { & $conv --input $Checkpoint --output $partial @Recipe @ExtraArgs @rule }
    $md = Read-Metadata $partial
    $extra = 0
    if ($c.bases_regex) {
      $run = $md.r4dx_convert_run
      if (-not $run -or $null -eq $run.w4a16_group_extra_bytes) { throw "[q3] $partial has no w4a16_group_extra_bytes" }
      if (-not $run.w4a16_groups -or @($run.w4a16_groups.PSObject.Properties).Count -eq 0) {
        throw "[q3] $n selected no linear (rule '$($c.bases_regex)=$($c.group)'), see convert_$n.log"
      }
      $extra = [int64]$run.w4a16_group_extra_bytes
    }
    [ordered]@{ name = $n; bases_regex = $c.bases_regex; group = $c.group; extra_bytes = $extra
      linears = if ($c.bases_regex) { @($md.r4dx_convert_run.w4a16_groups.PSObject.Properties).Count } else { 0 }
      convert_args = @($Recipe + $ExtraArgs + $rule) } |
      ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $metaPath
    Move-Item -Force $partial $container
    $converted = $true
  }

  if ($Kl -and -not $measured) {
    if (-not $converted) {
      # Only the baseline is required (it prices every candidate). A candidate with nothing to
      # measure -- not converted yet, e.g. outside an earlier -Only -- is left out of this
      # collection, so a plain -Kl re-collects whatever has been measured.
      if ($n -eq 'baseline' -or $Convert) {
        throw "[q3] $container (with $n.meta.json) missing -- run with -Convert"
      }
      Write-Host "[q3] $n not converted (no $n.r4dx with $n.meta.json) -- skipped; -Convert -Kl measures it"
      continue
    }
    $dir = Join-Path $OutDir "tf_$n"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Run "tf_$n" { & $Tool --model $container --layout w4a16 --tokens $tokens --out-dir $dir --max-ctx 4096 --vision off }
    Run "kl_$n" { & $Python tools\reference\kl_report.py --ref-dir $RefDir --test-dir $dir --tokens $tokens --out $klPath }
    if (-not $KeepLogprobs) { Remove-Item -Recurse -Force $dir }
    if ($Convert -and -not $KeepContainers) { Remove-Item -Force $container }
    $j = Get-Content $klPath -Raw | ConvertFrom-Json
    Write-Host ("[q3] {0,-28} mean KL {1:N5}  top-1 {2:N2}%" -f $n, $j.overall.mean_kl, $j.overall.top1_agreement_pct)
  }
}

if ($Kl) {
  # candidates.json for alloc_groups.py: exact byte deltas from the converter, KL from kl_report.py,
  # the baseline's weights from its own load line (Model's "VRAM breakdown ... weights=").
  $weights = $BaselineWeightsGib
  $tfLog = Join-Path $OutDir 'tf_baseline.log'
  if (Test-Path $tfLog) {
    $m = Select-String -Path $tfLog -Pattern 'weights=([0-9.]+) GiB' | Select-Object -First 1
    if ($m) { $weights = [double]$m.Matches[0].Groups[1].Value }
  }
  foreach ($f in @('kl_baseline.json', 'baseline.meta.json')) {
    if (-not (Test-Path (Join-Path $OutDir $f))) { throw "[q3] $OutDir\$f missing -- the baseline is not measured" }
  }
  $bj = Get-Content (Join-Path $OutDir 'kl_baseline.json') -Raw | ConvertFrom-Json
  $rows = @()
  $unmeasured = @()
  foreach ($c in $candidates) {
    $klPath = Join-Path $OutDir "kl_$($c.name).json"
    $metaPath = Join-Path $OutDir "$($c.name).meta.json"
    if (-not (Test-Path $klPath) -or -not (Test-Path $metaPath)) { $unmeasured += $c.name; continue }
    $k = Get-Content $klPath -Raw | ConvertFrom-Json
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    $rows += [ordered]@{ name = $c.name; bases_regex = $c.bases_regex; group = $c.group
      delta_gib = [double]$meta.extra_bytes / 1GB; kl = $k.overall.mean_kl
      top1 = $k.overall.top1_agreement_pct; linears = $meta.linears }
  }
  $out = [ordered]@{
    baseline = [ordered]@{ kl = $bj.overall.mean_kl; top1 = $bj.overall.top1_agreement_pct; weights_gib = $weights }
    default_group = $defaultGroup
    recipe = @($Recipe + $ExtraArgs)
    candidates = $rows
  }
  if ($BudgetGib -gt 0) { $out.budget_gib = $BudgetGib }
  $cj = Join-Path $OutDir 'candidates.json'
  $out | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $cj
  Write-Host "[q3] $($rows.Count) measured candidate(s) -> $cj"
  if ($unmeasured) { Write-Host "[q3] $($unmeasured.Count) not measured, left out: $($unmeasured -join ', ')" }
  Run 'alloc' { & $Python tools\quant2\alloc_groups.py $cj --json-out (Join-Path $OutDir 'picks.json') }
  Get-Content (Join-Path $OutDir 'alloc.log')
}
