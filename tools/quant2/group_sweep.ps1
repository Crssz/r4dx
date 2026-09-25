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
# Reuse (docs/quant2.md 5.2), the default: the baseline is converted with --record-reuse-guard and is
# every candidate's --reuse-tensors-from source, so a candidate recomputes only the linears its rule
# moves and copies every other tensor from the baseline -- the container is byte-identical to a full
# conversion (header aside: reused_from), and r4dx-convert refuses the baseline if anything but the
# rule differs (binary, CPU, checkpoint, recipe, input files) or its data no longer matches its
# digest. The baseline container is therefore kept after it is measured. One left by an older sweep
# without a completed guard is converted again while any candidate still needs converting. A MEASURED
# baseline whose container is gone is converted again only to be checked: the new container must
# have the data_sha256 and guard identity its meta.json recorded when it was measured, or the sweep
# stops (the binary, recipe or inputs changed: kl_baseline.json and every measured candidate are
# stale -- start a new -OutDir). Its meta.json is never rewritten.
#   -ReuseFrom <container>  another source (it must carry a completed guard); the sweep's own
#                           baseline then reuses from it too, and is deleted once measured as before
#   -NoReuse                the old path: every container a full conversion, no guard
# <name>.meta.json records reused_from, the recomputed linears, the container's data_sha256 and its
# guard identity (guard_json); -Kl leaves out a candidate whose guard identity differs from the
# baseline's.
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
  [string]$ReuseFrom = '',      # default: the sweep's own baseline container (see the header)
  [switch]$NoReuse,             # every candidate a full conversion (the pre-reuse path)
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

# A container r4dx-convert accepts as --reuse-tensors-from: its guard is there and its emit pass
# finished (r4dx_convert_run.reuse_guard.emit_complete, patched to 1 only after the last tensor, and
# data_sha256, the digest of its tensors patched in with it). The converter itself then checks the
# data against that digest.
function Test-GuardComplete($g) {
  return ($null -ne $g) -and ($g.emit_complete -eq 1) -and ("$($g.data_sha256)" -match '^[0-9a-f]{64}$') -and
    ("$($g.data_sha256)" -ne ('0' * 64))
}
function Test-ReuseGuard([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  return Test-GuardComplete (Read-Metadata $path).r4dx_convert_run.reuse_guard
}

# What a guard identifies -- binary, CPU, checkpoint, input files, flags -- as one comparable string:
# the guard without its two completion fields, compact JSON. Every container of one sweep must share
# it (only the group rules may differ, and they are not part of it); $null without a guard.
function Get-GuardIdentity($g) {
  if ($null -eq $g) { return $null }
  return ($g | Select-Object -Property * -ExcludeProperty emit_complete, data_sha256 |
    ConvertTo-Json -Depth 30 -Compress)
}

# ---- reuse source (docs/quant2.md 5.2; header comment) ------------------------------------------
if ($NoReuse -and $ReuseFrom) { throw '[q3] -ReuseFrom and -NoReuse exclude each other' }
$autoReuse = (-not $NoReuse) -and (-not $ReuseFrom)
$reuseSrc = ''
if ($ReuseFrom) {
  if ($Convert -and -not (Test-ReuseGuard $ReuseFrom)) {
    throw "[q3] -ReuseFrom $ReuseFrom is missing or has no completed reuse guard (convert it with --record-reuse-guard)"
  }
  $reuseSrc = (Resolve-Path $ReuseFrom).Path
  # Naming the sweep's own baseline is the default mode (it must not be treated as a stale file).
  if ($reuseSrc -ieq [IO.Path]::GetFullPath((Join-Path $ContainerDir 'baseline.r4dx'))) { $autoReuse = $true }
} elseif ($autoReuse) {
  $reuseSrc = Join-Path $ContainerDir 'baseline.r4dx'
}
# Converted (container + meta.json) or measured (kl + meta.json): nothing left to convert for it.
function Test-Done([string]$name) {
  if (-not (Test-Path (Join-Path $OutDir "$name.meta.json"))) { return $false }
  return (Test-Path (Join-Path $OutDir "kl_$name.json")) -or (Test-Path (Join-Path $ContainerDir "$name.r4dx"))
}
$pending = @($candidates | Where-Object { -not (Test-Done $_.name) })
if ($Convert) {
  $mode = if ($NoReuse) { 'off (-NoReuse): full conversions' } elseif ($autoReuse) { "the sweep's baseline $reuseSrc" } else { $reuseSrc }
  Write-Host "[q3] reuse source: $mode; $($pending.Count) candidate(s) still to convert"
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

  # The sweep's own baseline as the reuse source: it must exist with a completed guard while any
  # candidate still needs converting, even if it was measured already. A MEASURED baseline is never
  # replaced silently: kl_baseline.json (and every candidate's delta against it) holds only for the
  # container that was measured, whose data_sha256 and guard identity its meta.json recorded. So a
  # measured baseline is converted again only when that meta.json can prove the new container is the
  # same one (same data_sha256, same guard) -- checked after the conversion, before anything is
  # replaced; the meta.json is kept as written when it was measured. An existing container that no
  # longer matches its meta.json is refused the same way.
  $isSource = $autoReuse -and ($n -eq 'baseline')
  $mustConvert = $Convert -and -not $measured -and -not $converted
  $prevMeta = $null
  if ($isSource -and $Convert -and $pending.Count -gt 0) {
    $sourceOk = $converted -and (Test-ReuseGuard $container)
    if ($measured) {
      $prevMeta = Get-Content $metaPath -Raw | ConvertFrom-Json
      if (-not $prevMeta.data_sha256 -or -not $prevMeta.guard_json) {
        if (-not $sourceOk) {
          throw ("[q3] the baseline was measured (kl_baseline.json) on a container whose data digest and guard $n.meta.json does not record " +
                 "(an older sweep's), and that container is gone or has no completed reuse guard: a new conversion cannot be shown to be " +
                 "the container that was measured. Finish this sweep with -NoReuse, or start a new -OutDir.")
        }
        throw ("[q3] $n.meta.json (measured) does not record the data_sha256 / guard of $container, so it cannot be checked " +
               "against the measured container. Finish this sweep with -NoReuse, or start a new -OutDir.")
      }
      if ($sourceOk) {
        $g = (Read-Metadata $container).r4dx_convert_run.reuse_guard
        if ($g.data_sha256 -ne $prevMeta.data_sha256 -or (Get-GuardIdentity $g) -ne $prevMeta.guard_json) {
          throw ("[q3] $container is not the container kl_baseline.json was measured on (data_sha256 $($g.data_sha256), " +
                 "measured $($prevMeta.data_sha256); guard identical: $((Get-GuardIdentity $g) -eq $prevMeta.guard_json)). " +
                 "Restore the measured container, or start a new -OutDir.")
        }
      }
    }
    if (-not $sourceOk) {
      if ($converted) { Write-Host "[q3] $container has no completed reuse guard (an older sweep's) -- converting the baseline again as the reuse source" }
      elseif ($measured) { Write-Host "[q3] the baseline was measured but its container is gone -- converting it again as the reuse source; it must come out identical to the measured one (data_sha256 $($prevMeta.data_sha256))" }
      $mustConvert = $true
    }
  }

  if ($mustConvert) {
    foreach ($stale in @($container, $partial)) {
      if (Test-Path $stale) { Write-Host "[q3] removing $stale (unfinished, or replaced as the reuse source)"; Remove-Item -Force $stale }
    }
    $free = Free-GB $ContainerDir
    if ($free -lt $MinFreeGB) {
      throw "[q3] $([math]::Round($free, 1)) GB free on $ContainerDir (< -MinFreeGB $MinFreeGB) before $n -- run -Convert -Kl together (it deletes each container once measured) or free space"
    }
    $rule = @()
    if ($c.bases_regex) { $rule = @('--w4a16-group-rule', "$($c.bases_regex)=$($c.group)") }
    # The source records its guard; everything else copies from the source when there is one.
    $reuse = @()
    if ($isSource) { $reuse = @('--record-reuse-guard') }
    elseif ($reuseSrc) {
      if (Test-ReuseGuard $reuseSrc) { $reuse = @('--reuse-tensors-from', $reuseSrc) }
      else { Write-Host "[q3] reuse source $reuseSrc missing or without a completed guard -- $n converts in full" }
    }
    Run "convert_$n" { & $conv --input $Checkpoint --output $partial @Recipe @ExtraArgs @rule @reuse }
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
    # reused_from: the source and what the converter recomputed (null for a full conversion).
    $rf = $md.r4dx_convert_run.reused_from
    if (($reuse -contains '--reuse-tensors-from') -and -not $rf) { throw "[q3] $partial has no r4dx_convert_run.reused_from" }
    $reusedFrom = $null
    if ($rf) {
      $reusedFrom = [ordered]@{ path = $rf.path; header_sha256 = $rf.header_sha256; data_sha256 = $rf.data_sha256
        tensors_copied = $rf.tensors_copied; tensors_recomputed = $rf.tensors_recomputed
        linears_recomputed = @($rf.linears_recomputed) }
    }
    # The container's identity for later runs: its data digest and its guard identity (null without
    # a guard, i.e. -NoReuse or a -ReuseFrom-less full conversion).
    $guard = $md.r4dx_convert_run.reuse_guard
    if ($reuse.Count -gt 0 -and -not (Test-GuardComplete $guard)) { throw "[q3] $partial has no completed reuse guard" }
    $dataSha = if ($guard) { "$($guard.data_sha256)" } else { $null }
    $guardId = Get-GuardIdentity $guard
    if ($prevMeta) {
      # A measured baseline converted again: it must be the container that was measured.
      if ($dataSha -ne $prevMeta.data_sha256 -or $guardId -ne $prevMeta.guard_json) {
        Remove-Item -Force $partial
        throw ("[q3] the re-converted baseline is not the container kl_baseline.json was measured on: data_sha256 $dataSha, " +
               "measured $($prevMeta.data_sha256); guard identical: $($guardId -eq $prevMeta.guard_json). The converter binary, CPU, " +
               "recipe (-Recipe/-ExtraArgs) or an input file changed since it was measured, so that KL and every candidate measured " +
               "against it are stale. Start a new -OutDir (or restore the measured baseline.r4dx).")
      }
      Write-Host "[q3] the re-converted baseline is identical to the measured one (data_sha256 $dataSha, same guard) -- keeping $metaPath as measured"
    } else {
      [ordered]@{ name = $n; bases_regex = $c.bases_regex; group = $c.group; extra_bytes = $extra
        linears = if ($c.bases_regex) { @($md.r4dx_convert_run.w4a16_groups.PSObject.Properties).Count } else { 0 }
        convert_args = @($Recipe + $ExtraArgs + $rule); reused_from = $reusedFrom
        data_sha256 = $dataSha; guard_json = $guardId } |
        ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $metaPath
    }
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
    # The reuse source stays: every later candidate copies from it.
    if ($isSource) { Write-Host "[q3] keeping $container (the candidates' --reuse-tensors-from source)" }
    elseif ($Convert -and -not $KeepContainers) { Remove-Item -Force $container }
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
  $bm = Get-Content (Join-Path $OutDir 'baseline.meta.json') -Raw | ConvertFrom-Json
  $rows = @()
  $unmeasured = @()
  $stale = @()
  foreach ($c in $candidates) {
    $klPath = Join-Path $OutDir "kl_$($c.name).json"
    $metaPath = Join-Path $OutDir "$($c.name).meta.json"
    if (-not (Test-Path $klPath) -or -not (Test-Path $metaPath)) { $unmeasured += $c.name; continue }
    $k = Get-Content $klPath -Raw | ConvertFrom-Json
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    # A candidate converted by another binary / CPU / recipe / input set than the measured baseline
    # (both guarded, identities differ) is not a group-rule delta against it: left out, named.
    if ($bm.guard_json -and $meta.guard_json -and $meta.guard_json -ne $bm.guard_json) { $stale += $c.name; continue }
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
  if ($stale) { Write-Host "[q3] $($stale.Count) converted with another guard identity than the measured baseline (binary, CPU, recipe or inputs changed), left out: $($stale -join ', ')" }
  Run 'alloc' { & $Python tools\quant2\alloc_groups.py $cj --json-out (Join-Path $OutDir 'picks.json') }
  Get-Content (Join-Path $OutDir 'alloc.log')
}
