# HISTORICAL (Q3 experiment driver): its defaults name the base checkpoint, base imatrix/kvcalib, hessian-v1
# and the q3-sweep reference (2026-09-29: the base checkpoint was removed, docs/huihui.md "Retired files");
# pass every path explicitly to re-run it on another model.
# docs/quant2.md sections 5 and 5.3 (Q3): the per-tensor precision sweep. Each candidate makes ONE
# change to one set of linears on top of the v6-style recipe, and is measured on its own by rung 4
# (tool_teacher_forced_logprobs + kl_report.py) against a bf16 reference:
#   kind 'group'   the set moves from the build's default w4a16 group to another (--w4a16-group-rule
#                  "<bases_regex>=<group>"; default: 32) -- a tensor class in one depth half,
#                  or lm_head;
#   kind 'keep'    the set is written in bf16 (added to the recipe's --keep-bf16): a byte SPENDER;
#   kind 'unkeep'  the set, which the recipe keeps in bf16, is quantized like every other linear
#                  (taken out of the recipe's --keep-bf16; LDLQ applies if the recipe has it): a SAVER.
# The result is the candidates.json tools/quant2/alloc_groups.py ranks by nats of KL per GiB and fills
# a byte budget from, printing the flags of the chosen set.
#
#   -Convert   CPU only: r4dx-convert each candidate container (w4a16 only) plus the no-change
#              baseline. Writes <name>.meta.json next to the logs (below).
#   -Kl        GPU (HIP device 1): bf16 reference dump if -RefDir has none, then per candidate the
#              teacher-forced run and kl_report.py. Then candidates.json and alloc_groups.py.
#   -ListCandidates   prints the candidate list (and, once the baseline is converted, each one's
#              resolved linear count and byte delta) and exits.
#
# --keep-bf16 is ONE regex (r4dx-convert keeps the last one given), so a keep / unkeep candidate never
# appends a second one: the recipe's --keep-bf16 (the last in -Recipe + -ExtraArgs) is replaced by ONE
# merged ECMAScript regex -- keep: "(?:<recipe>)|(?:<bases_regex>)"; unkeep:
# "^(?![\s\S]*?(?:<bases_regex>))[\s\S]*?(?:<recipe>)" (matches iff the recipe's regex matches
# somewhere and the candidate's nowhere, anchored or not). Every candidate is then VERIFIED against the
# converter's own output, not the regex text: before converting, its intended set is the baseline's
# linears that bases_regex matches (each must be in the state the kind needs: w4a16 for group / keep,
# bf16 for unkeep); after converting, the set of linears whose layout differs from the baseline's (read
# from the two tensor directories) must be exactly that set, each at the candidate's precision, the
# converter's keep_bf16_linears must be the baseline's plus / minus it, and a reuse run's
# reused_from.linears_recomputed must be that set too. Anything else fails the candidate.
#
# The byte delta (meta.json extra_bytes) is what a `--layout w4a16` load reads, from the two tensor
# directories: per changed linear, its w4a16 tensors (wq + wsz at its group) or its bf16.w, candidate
# minus baseline -- exact for every kind and every recipe. It is cross-checked against the converter's
# own accounting (w4a16_group_extra_bytes for a group candidate; keep_bf16_extra_bytes, whose baseline
# already carries the recipe's k/v, for keep / unkeep when every linear has exactly one layout).
#
# A container is ~20 GB and there are ~40 of them, so -Convert -Kl together runs candidate by
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
#   .\tools\quant2\group_sweep.ps1 -ListCandidates
#
# The default -Recipe alone is the UNROTATED v6 recipe. The Q3 round (docs/quant2.md 5.3) measures on
# the current one, q2ab rotation + LDLQ (section 7's q2ab_ldlq), whose sensitivities differ -- so it runs
#   .\tools\quant2\group_sweep.ps1 -Convert -Kl -ExtraArgs '--rotate','q2ab','--hessian-dir','D:\models\r4dx\hessian-v1','--ldlq','.'
# and every later run on that -OutDir passes the same -Recipe / -ExtraArgs (-Kl refuses one whose
# --keep-bf16 is not the measured baseline's; a -Convert would fail its candidates' verification).
#
# Reuse (docs/quant2.md 5.2), the default: the baseline is converted with --record-reuse-guard and is
# every candidate's --reuse-tensors-from source, so a candidate recomputes only the linears whose
# layout set it changes (a group rule or a keep) and copies every other tensor from the baseline -- the
# container is byte-identical to a full conversion (header aside: reused_from), and r4dx-convert
# refuses the baseline if anything but the per-linear layout choices differs (binary, CPU,
# checkpoint, recipe, input files) or its data no longer matches its digest. The baseline container
# is therefore kept after it is measured. One left by an older sweep
# without a completed guard is converted again while any candidate still needs converting. A MEASURED
# baseline whose container is gone is converted again only to be checked: the new container must
# have the data_sha256 and guard identity its meta.json recorded when it was measured, or the sweep
# stops (the binary, recipe or inputs changed: kl_baseline.json and every measured candidate are
# stale -- start a new -OutDir). Its meta.json is never rewritten.
#   -ReuseFrom <container>  another source (it must carry a completed guard); the sweep's own
#                           baseline then reuses from it too, and is deleted once measured as before
#   -NoReuse                the old path: every container a full conversion, no guard
# <name>.meta.json records kind, precision, the exact linear set (linears), extra_bytes, the
# converter's own byte accounting, the merged --keep-bf16, reused_from, the container's data_sha256
# and its guard identity (guard_json: the guard without its completion fields and its per-linear
# record), and the data_sha256 of the baseline it was verified against (baseline_data_sha256);
# baseline.meta.json also records every linear's layout and N*K (linear_layouts) and the keep list,
# which every candidate is verified against. The guard identity no longer carries the recipe's
# --keep-bf16 (keep candidates change it), so -Kl leaves out a candidate whose guard identity
# differs from the baseline's, whose keep_bf16_linears is not the baseline's plus (keep) / minus
# (unkeep) / unchanged by (group) its set, or that was verified against another baseline container;
# and candidates.json carries the baseline's own convert_args as the recipe.
#
# -CandidateFile takes [{"name", "kind", "bases_regex", "group"}, ...] instead of the default list
# below: kind "group" (the default when absent, i.e. every older file) needs group 32 or 64; "keep"
# takes no group; "unkeep" may give the group the un-kept linears must land at (default: the build
# default).
param(
  [switch]$Convert,
  [switch]$Kl,
  [switch]$ListCandidates,
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
  [int[]]$Groups = @(32),
  [string]$CandidateFile = '',
  [string]$Only = '',           # regex over candidate names; the baseline is always included
  [double]$BudgetGib = 0,       # alloc_groups.py --budget-gib; 0 = the baseline's own weights
  [double]$BaselineWeightsGib = 16.4065,  # used only if the baseline's tf log has no weights= line
  [double]$MinFreeGB = 40,
  [string]$ReuseFrom = '',      # default: the sweep's own baseline container (see the header)
  [switch]$NoReuse,             # every candidate a full conversion (the pre-reuse path)
  [switch]$KeepContainers,
  [switch]$KeepLogprobs,
  [string]$Python = $(if ($env:R4DX_REFERENCE_VENV) { Join-Path $env:R4DX_REFERENCE_VENV 'Scripts\python.exe' } else { 'python' }),
  [string]$Tool = '',           # default: this worktree's tool_teacher_forced_logprobs.exe
  # The rung-4 tokens file every candidate is scored on (and -RefDir's reference was dumped from).
  # tokens.json's thai_prose is the AutoTokenizer split-mark form; tokens_canon.json is the same four
  # texts tokenized as r4dx serves them (docs/quant2.md 3.4) -- the one to allocate against.
  [string]$Tokens = 'tools\reference\kl_corpus\tokens.json'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not ($Convert -or $Kl -or $ListCandidates)) { throw 'pass -Convert and/or -Kl (or -ListCandidates)' }
if (-not $ContainerDir) { $ContainerDir = $OutDir }
if ($Convert -or $Kl) {
  New-Item -ItemType Directory -Force $OutDir | Out-Null
  New-Item -ItemType Directory -Force $ContainerDir | Out-Null
}
if (-not $RefDir) { $RefDir = Join-Path $OutDir 'ref' }
# Per-tensor groups need a runtime that reads __metadata__.quant.w4a16.groups: this worktree's build,
# not the main checkout's (which q1_pilot.ps1 can use, its containers being single-group).
if (-not $Tool) { $Tool = Join-Path $repo 'build\win-hip\tests\model\tool_teacher_forced_logprobs.exe' }
$conv = Join-Path $repo 'build\win-hip\src\convert\r4dx-convert.exe'
$tokens = $Tokens
if (-not (Test-Path $tokens)) { throw "[q3] -Tokens $tokens not found" }

function Run([string]$name, [scriptblock]$cmd) {
  $log = Join-Path $OutDir "$name.log"
  Write-Host "[q3] $name -> $log"
  # PS 5.1 makes native stderr lines error records; under 'Stop' the first one would abort the
  # sweep. Success is judged by the exit code alone.
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $cmd *> $log } finally { $ErrorActionPreference = $prev }
  if ($LASTEXITCODE -ne 0) { throw "[q3] $name failed (exit $LASTEXITCODE), see $log" }
}

# The default w4a16 group (the one a candidate at that group would not change): r4dx_convert's
# kW4A16Group.
$defaultGroup = 64

# ---- the recipe's --keep-bf16 ------------------------------------------------------------------
# r4dx-convert's --keep-bf16 is single-valued (the last one wins), so the recipe's effective keep regex
# is the LAST --keep-bf16 in -Recipe + -ExtraArgs, and a keep / unkeep candidate replaces every one of
# them with its merged regex (header comment).
$recipeAll = @($Recipe + $ExtraArgs)
$recipeRest = @()
for ($i = 0; $i -lt $recipeAll.Count; $i++) {
  if ($recipeAll[$i] -eq '--keep-bf16') {
    if ($i + 1 -ge $recipeAll.Count) { throw '[q3] -Recipe/-ExtraArgs end with --keep-bf16 and no value' }
    $i++
  } else { $recipeRest += $recipeAll[$i] }
}
# The value of the LAST `$flag` in an argument list ('' without one): r4dx-convert's single-valued
# flags keep the last.
function Get-LastArg($list, [string]$flag) {
  $v = ''
  $a = @($list)
  for ($i = 0; $i -lt $a.Count - 1; $i++) { if ("$($a[$i])" -ceq $flag) { $v = "$($a[$i + 1])"; $i++ } }
  return $v
}
$recipeKeep = Get-LastArg $recipeAll '--keep-bf16'
function Join-KeepRegex([string]$add, [string]$remove) {
  if ($add) {
    if ($recipeKeep) { return "(?:$recipeKeep)|(?:$add)" }
    return $add
  }
  if (-not $recipeKeep) { throw '[q3] an unkeep candidate needs a --keep-bf16 in -Recipe/-ExtraArgs' }
  return "^(?![\s\S]*?(?:$remove))[\s\S]*?(?:$recipeKeep)"
}

# ---- candidates --------------------------------------------------------------------------------
# Default: 7 body classes x 2 depth halves x -Groups, plus lm_head (one tensor, no halves) x -Groups;
# then the bf16 keep spenders and the attn.k/v unkeep savers of docs/quant2.md 5.3. attn.qg / attn.o
# exist on the full-attention layers only (every 4th); attn.k / attn.v are bf16 in the recipe
# (--keep-bf16), so they are not group candidates; mtp.* keeps the default group and is never kept.
$halves = [ordered]@{ 'L0-31' = '([0-9]|[12][0-9]|3[01])'; 'L32-63' = '(3[2-9]|[45][0-9]|6[0-3])' }
$candidates = @()
if ($CandidateFile) {
  # The parentheses matter: Windows PowerShell 5.1's ConvertFrom-Json emits a JSON array as ONE
  # pipeline object, so without them ForEach-Object would see the whole array as $_.
  $candidates = @((Get-Content $CandidateFile -Raw | ConvertFrom-Json) | ForEach-Object {
      $kind = if ($_.kind) { "$($_.kind)" } else { 'group' }
      $g = if ($null -ne $_.group) { [int]$_.group } elseif ($kind -eq 'unkeep') { $defaultGroup } else { 0 }
      [pscustomobject]@{ name = $_.name; kind = $kind; bases_regex = $_.bases_regex; group = $g } })
} else {
  $classes = [ordered]@{
    'gdn.in_proj_qkv' = 'gdn\.in_proj_qkv'; 'gdn.in_proj_z' = 'gdn\.in_proj_z'
    'gdn.out_proj' = 'gdn\.out_proj'; 'attn.qg' = 'attn\.qg'; 'attn.o' = 'attn\.o'
    'mlp.gate_up' = 'mlp\.gate_up'; 'mlp.down' = 'mlp\.down'
  }
  foreach ($g in $Groups) {
    foreach ($c in $classes.Keys) {
      foreach ($h in $halves.Keys) {
        $candidates += [pscustomobject]@{ name = "$c.$h.g$g"; kind = 'group'
          bases_regex = "^text\.layers\.$($halves[$h])\.$($classes[$c])$"; group = $g }
      }
    }
    $candidates += [pscustomobject]@{ name = "lm_head.g$g"; kind = 'group'; bases_regex = '^lm_head$'; group = $g }
  }
  # docs/quant2.md 5.3: ~0.34-0.51 GiB each, on the depth ends Milestone 11 found most sensitive, and
  # attn.o by half (the class it measured cheapest after k/v). lm_head bf16 (+1.70 GiB) is left out:
  # lm_head.g32 is its affordable probe.
  $keeps = [ordered]@{
    'mlp.down.L0-3'         = '^text\.layers\.[0-3]\.mlp\.down$'
    'mlp.down.L60-63'       = '^text\.layers\.6[0-3]\.mlp\.down$'
    'mlp.gate_up.L62-63'    = '^text\.layers\.6[23]\.mlp\.gate_up$'
    'gdn.in_proj_qkv.L0-7'  = '^text\.layers\.[0-7]\.gdn\.in_proj_qkv$'
    'gdn.in_proj_qkv.L56-63' = '^text\.layers\.(5[6-9]|6[0-3])\.gdn\.in_proj_qkv$'
    'gdn.out_proj.L0-15'    = '^text\.layers\.([0-9]|1[0-5])\.gdn\.out_proj$'
    'gdn.out_proj.L48-63'   = '^text\.layers\.(4[89]|5[0-9]|6[0-3])\.gdn\.out_proj$'
    'attn.o.L0-31'          = "^text\.layers\.$($halves['L0-31'])\.attn\.o$"
    'attn.o.L32-63'         = "^text\.layers\.$($halves['L32-63'])\.attn\.o$"
  }
  foreach ($k in $keeps.Keys) {
    $candidates += [pscustomobject]@{ name = "$k.bf16"; kind = 'keep'; bases_regex = $keeps[$k]; group = 0 }
  }
  if ($recipeKeep) {
    foreach ($h in $halves.Keys) {
      $candidates += [pscustomobject]@{ name = "attn.kv.$h.g$defaultGroup"; kind = 'unkeep'
        bases_regex = "^text\.layers\.$($halves[$h])\.attn\.[kv]$"; group = $defaultGroup }
    }
  }
}
$seen = @{}
foreach ($c in $candidates) {
  if (-not $c.name -or $c.name -eq 'baseline' -or $seen.ContainsKey($c.name)) { throw "[q3] candidate name '$($c.name)' is empty, 'baseline' or a duplicate" }
  $seen[$c.name] = $true
  if (-not $c.bases_regex) { throw "[q3] candidate $($c.name) has no bases_regex" }
  switch ($c.kind) {
    'group' { if (@(32, 64) -notcontains $c.group) { throw "[q3] candidate $($c.name): group $($c.group) is not 32/64" } }
    'keep' { }
    'unkeep' { if (@(32, 64) -notcontains $c.group) { throw "[q3] candidate $($c.name): group $($c.group) is not 32/64" } }
    default { throw "[q3] candidate $($c.name): kind '$($c.kind)' is not group, keep or unkeep" }
  }
}
$skipped = @($candidates | Where-Object { $_.kind -eq 'group' -and $_.group -eq $defaultGroup })
if ($skipped) { Write-Host "[q3] skipping $($skipped.Count) group candidate(s) at the build default group $defaultGroup" }
$candidates = @($candidates | Where-Object { -not ($_.kind -eq 'group' -and $_.group -eq $defaultGroup) })
if ($Only) { $candidates = @($candidates | Where-Object { $_.name -match $Only }) }
$all = @([pscustomobject]@{ name = 'baseline'; kind = 'baseline'; bases_regex = ''; group = $defaultGroup }) + $candidates
$recipeRot = Get-LastArg $recipeAll '--rotate'
$recipeLdlq = Get-LastArg $recipeAll '--ldlq'
Write-Host ("[q3] default group $defaultGroup; $($candidates.Count) candidate(s) + baseline; recipe --keep-bf16 '$recipeKeep', " +
            "--rotate $(if ($recipeRot) { $recipeRot } else { 'none' }), --ldlq $(if ($recipeLdlq) { "'$recipeLdlq'" } else { 'none' })")

# The whole header of a container: 8-byte little-endian length, then the JSON.
function Read-Header([string]$path) {
  $fs = [IO.File]::OpenRead($path)
  try {
    $b = New-Object byte[] 8
    if ($fs.Read($b, 0, 8) -ne 8) { throw "[q3] $path is too short" }
    $n = [int][BitConverter]::ToUInt64($b, 0)
    $h = New-Object byte[] $n
    $got = 0
    while ($got -lt $n) { $got += $fs.Read($h, $got, $n - $got) }
    return ([Text.Encoding]::UTF8.GetString($h) | ConvertFrom-Json)
  } finally { $fs.Close() }
}
function Read-Metadata([string]$path) { return (Read-Header $path).__metadata__ }

function Free-GB([string]$dir) {
  $root = [IO.Path]::GetPathRoot((Resolve-Path $dir).Path)
  return (New-Object IO.DriveInfo $root).AvailableFreeSpace / 1GB
}

# Ordinal sort / set equality of base names (PowerShell's own sort is culture-aware). Sort-Ordinal
# writes the names to the pipeline one by one: call it inside @(...).
function Sort-Ordinal($a) {
  $x = [string[]]@($a | Where-Object { $null -ne $_ })
  [Array]::Sort($x, [StringComparer]::Ordinal)
  return $x
}
function Test-SameSet($a, $b) { return ((Sort-Ordinal $a) -join "`n") -ceq ((Sort-Ordinal $b) -join "`n") }
function New-OrdinalTable { return New-Object System.Collections.Hashtable ([StringComparer]::Ordinal) }

# Every linear of a container as a `--layout w4a16` load sees it, from the tensor directory alone:
# base -> {p = 'w4a16.g<g>' | 'bf16' | 'none', bytes = what that load reads for it, nk = N*K}. A base
# with a w4a16 form loads it (wq + its wsz, at the group its wsz name says); one without loads its
# .bf16.w. `single`: every linear has exactly one of the two forms and nothing else (w4a8 / mxfp4 /
# a bf16 companion would make on-disk and loaded bytes differ).
function Get-LinearLayouts($header) {
  $defG = [int]$header.__metadata__.quant.w4a16.group
  $lin = New-OrdinalTable
  foreach ($prop in $header.PSObject.Properties) {
    if ($prop.Name -eq '__metadata__') { continue }
    $m = [regex]::Match($prop.Name, '^(.+)\.(bf16\.w|w4a16\.wq|w4a16\.wsz(?:\.g(\d+))?|w4a8\.wq|w4a8\.ws|mxfp4\.wq|mxfp4\.ws|mxfp4\.wref)$')
    if (-not $m.Success) { continue }
    $base = $m.Groups[1].Value
    if (-not $lin.ContainsKey($base)) { $lin[$base] = @{ bf16 = -1L; wq = -1L; wsz = -1L; g = 0; other = 0 } }
    $e = $lin[$base]
    $size = [int64]$prop.Value.data_offsets[1] - [int64]$prop.Value.data_offsets[0]
    $what = $m.Groups[2].Value
    if ($what -eq 'bf16.w') { $e.bf16 = $size }
    elseif ($what -eq 'w4a16.wq') { $e.wq = $size }
    elseif ($what.StartsWith('w4a16.wsz')) {
      $e.wsz = $size
      $e.g = if ($m.Groups[3].Success) { [int]$m.Groups[3].Value } else { $defG }
    } else { $e.other++ }
  }
  $map = New-OrdinalTable
  $single = $true
  foreach ($base in $lin.Keys) {
    $e = $lin[$base]
    if ($e.wq -ge 0) {
      if ($e.wsz -lt 0) { throw "[q3] $base has .w4a16.wq but no .w4a16.wsz" }
      $map[$base] = [pscustomobject]@{ p = "w4a16.g$($e.g)"; bytes = $e.wq + $e.wsz; nk = 2 * $e.wq }
      if ($e.bf16 -ge 0 -or $e.other -gt 0) { $single = $false }
    } elseif ($e.bf16 -ge 0) {
      $map[$base] = [pscustomobject]@{ p = 'bf16'; bytes = $e.bf16; nk = [int64]($e.bf16 / 2) }
      if ($e.other -gt 0) { $single = $false }
    } else {
      $map[$base] = [pscustomobject]@{ p = 'none'; bytes = 0L; nk = 0L }
      $single = $false
    }
  }
  return [pscustomobject]@{ map = $map; single = $single }
}

# What a `--layout w4a16` load reads for a linear of N*K = $nk at precision $p (Get-LinearLayouts' p).
function Get-LoadBytes([string]$p, [int64]$nk) {
  if ($p -eq 'bf16') { return 2 * $nk }
  if ($p -match '^w4a16\.g(\d+)$') { return [int64]($nk / 2) + [int64]($nk / [int]$Matches[1]) * 4 }
  throw "[q3] no w4a16-load byte count for layout '$p'"
}

# The precision a candidate moves its linears to, in Get-LinearLayouts' terms.
function Get-TargetLayout($c) {
  switch ($c.kind) {
    'group' { return "w4a16.g$($c.group)" }
    'keep' { return 'bf16' }
    'unkeep' { return "w4a16.g$($c.group)" }
  }
  throw "[q3] no target layout for kind '$($c.kind)'"
}

# The baseline's per-linear layouts (linear_layouts: base -> {p, nk}), keep list, the converter's
# byte accounting and the data digest baseline.meta.json recorded (null without a guard): from
# baseline.meta.json, else from the baseline container itself.
$script:baseInfo = $null
function ConvertTo-BaseInfo($ll, $kept, $grpExtra, $keepExtra, $single, $dataSha) {
  $map = New-OrdinalTable
  foreach ($p in $ll.PSObject.Properties) { $map[$p.Name] = [pscustomobject]@{ p = "$($p.Value.p)"; nk = [int64]$p.Value.nk } }
  return [pscustomobject]@{ map = $map; kept = @(Sort-Ordinal @($kept)); group_extra = [int64]$grpExtra
    keep_extra = [int64]$keepExtra; single = [bool]$single; data_sha256 = $(if ($dataSha) { "$dataSha" } else { $null }) }
}
function Get-BaselineInfo {
  if ($script:baseInfo) { return $script:baseInfo }
  $bmPath = Join-Path $OutDir 'baseline.meta.json'
  $bm = $null
  if (Test-Path $bmPath) {
    $bm = Get-Content $bmPath -Raw | ConvertFrom-Json
    if ($bm.linear_layouts) {
      $script:baseInfo = ConvertTo-BaseInfo $bm.linear_layouts $bm.keep_bf16_linears $bm.w4a16_group_extra_bytes $bm.keep_bf16_extra_bytes $bm.single_layout $bm.data_sha256
      return $script:baseInfo
    }
  }
  $bc = Join-Path $ContainerDir 'baseline.r4dx'
  if ((Test-Path $bc) -and $bm) {
    $hdr = Read-Header $bc
    $lay = Get-LinearLayouts $hdr
    $ll = [pscustomobject]@{}
    foreach ($b in $lay.map.Keys) { $ll | Add-Member -NotePropertyName $b -NotePropertyValue ([pscustomobject]@{ p = $lay.map[$b].p; nk = $lay.map[$b].nk }) }
    $run = $hdr.__metadata__.r4dx_convert_run
    $script:baseInfo = ConvertTo-BaseInfo $ll $run.keep_bf16_linears $run.w4a16_group_extra_bytes $run.keep_bf16_extra_bytes $lay.single $bm.data_sha256
    return $script:baseInfo
  }
  throw ("[q3] the baseline's per-linear layouts are unknown: $bmPath predates keep candidates (no linear_layouts) and $bc is gone. " +
         "Convert the sweep again in a new -OutDir.")
}

# The candidate's intended linear set: the baseline's linears its bases_regex matches (ECMAScript,
# search -- like r4dx-convert), each in the state the kind needs.
function Get-IntendedSet($c, $bi) {
  $re = [regex]::new($c.bases_regex, [Text.RegularExpressions.RegexOptions]::ECMAScript)
  $matched = @(Sort-Ordinal @($bi.map.Keys | Where-Object { $re.IsMatch($_) }))
  if ($matched.Count -eq 0) { throw "[q3] candidate $($c.name): '$($c.bases_regex)' matches none of the baseline's linears" }
  $target = Get-TargetLayout $c
  foreach ($b in $matched) {
    $p = $bi.map[$b].p
    $ok = switch ($c.kind) {
      'group' { $p -match '^w4a16\.g' -and $p -ne $target }
      'keep' { $p -match '^w4a16\.g' }
      'unkeep' { $p -eq 'bf16' -and ($bi.kept -ccontains $b) }
    }
    if (-not $ok) {
      throw ("[q3] candidate $($c.name) ($($c.kind) -> $target): '$b' is $p in the baseline" +
             $(if ($c.kind -eq 'unkeep') { ' (only a linear the recipe''s --keep-bf16 keeps can be un-kept)' }
               elseif ($p -eq 'bf16') { ' (--keep-bf16 wins over a rule; un-keep it instead)' } else { '' }))
    }
  }
  return ,$matched
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
# the guard without its two completion fields and its per-linear record (linears: the one thing the
# candidates change), compact JSON. Every container of one sweep must share it; $null without a guard.
function Get-GuardIdentity($g) {
  if ($null -eq $g) { return $null }
  return ($g | Select-Object -Property * -ExcludeProperty emit_complete, data_sha256, linears |
    ConvertTo-Json -Depth 30 -Compress)
}

if ($ListCandidates) {
  $bi = $null
  try { $bi = Get-BaselineInfo } catch { Write-Host "[q3] (no converted baseline in $OutDir yet: linear counts and byte deltas unknown)" }
  Write-Host ("{0,-30} {1,-7} {2,-10} {3,8} {4,10}  {5}" -f 'name', 'kind', 'to', 'linears', 'dGiB', 'bases_regex')
  foreach ($c in $candidates) {
    $cnt = ''; $gib = ''
    if ($bi) {
      $set = Get-IntendedSet $c $bi
      $target = Get-TargetLayout $c
      $d = 0L
      foreach ($b in $set) { $d += (Get-LoadBytes $target $bi.map[$b].nk) - (Get-LoadBytes $bi.map[$b].p $bi.map[$b].nk) }
      $cnt = $set.Count; $gib = '{0:+0.0000;-0.0000}' -f ($d / 1GB)
    }
    Write-Host ("{0,-30} {1,-7} {2,-10} {3,8} {4,10}  {5}" -f $c.name, $c.kind, (Get-TargetLayout $c), $cnt, $gib, $c.bases_regex)
  }
  if (-not ($Convert -or $Kl)) { return }
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

# A MEASURED baseline fixes the recipe's --keep-bf16 for its -OutDir: every keep / unkeep
# candidate's merged regex, the unkeep list and the allocator's flags are built from it, and the
# guard identity does not carry it. Checked before any work, and again when collecting.
function Assert-BaselineKeep {
  $bmPath = Join-Path $OutDir 'baseline.meta.json'
  if (-not ((Test-Path $bmPath) -and (Test-Path (Join-Path $OutDir 'kl_baseline.json')))) { return }
  $bm0 = Get-Content $bmPath -Raw | ConvertFrom-Json
  if ($null -eq $bm0.convert_args) { return }
  $k = Get-LastArg @($bm0.convert_args) '--keep-bf16'
  if ($k -cne $recipeKeep) {
    throw ("[q3] the measured baseline was converted with --keep-bf16 '$k' (baseline.meta.json convert_args), " +
           "this run's -Recipe/-ExtraArgs give '$recipeKeep': pass the -Recipe/-ExtraArgs the sweep was converted with, " +
           "or start a new -OutDir")
  }
}
if ($Convert -or $Kl) { Assert-BaselineKeep }
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
    # The candidate's arguments: the recipe as given, plus a rule (group), or the recipe with its
    # --keep-bf16 replaced by ONE merged regex (keep / unkeep).
    $convArgs = $recipeAll
    $rule = @()
    $keepRe = $null
    $intended = $null
    $bi = $null
    if ($c.kind -ne 'baseline') {
      $bi = Get-BaselineInfo
      $intended = Get-IntendedSet $c $bi
      switch ($c.kind) {
        'group' { $rule = @('--w4a16-group-rule', "$($c.bases_regex)=$($c.group)") }
        'keep' { $keepRe = Join-KeepRegex $c.bases_regex '' }
        'unkeep' { $keepRe = Join-KeepRegex '' $c.bases_regex }
      }
      if ($keepRe) { $convArgs = @($recipeRest + @('--keep-bf16', $keepRe)) }
      Write-Host "[q3] $n ($($c.kind) -> $(Get-TargetLayout $c)): $($intended.Count) linear(s) intended$(if ($keepRe) { "; --keep-bf16 '$keepRe'" })"
    }
    # The source records its guard; everything else copies from the source when there is one.
    $reuse = @()
    if ($isSource) { $reuse = @('--record-reuse-guard') }
    elseif ($reuseSrc) {
      if (Test-ReuseGuard $reuseSrc) { $reuse = @('--reuse-tensors-from', $reuseSrc) }
      else { Write-Host "[q3] reuse source $reuseSrc missing or without a completed guard -- $n converts in full" }
    }
    Run "convert_$n" { & $conv --input $Checkpoint --output $partial @convArgs @rule @reuse }
    $hdr = Read-Header $partial
    $md = $hdr.__metadata__
    $run = $md.r4dx_convert_run
    $lay = Get-LinearLayouts $hdr
    $keptNow = @(Sort-Ordinal @($run.keep_bf16_linears))
    $grpExtra = if ($null -ne $run.w4a16_group_extra_bytes) { [int64]$run.w4a16_group_extra_bytes } else { 0L }
    $keepExtra = if ($null -ne $run.keep_bf16_extra_bytes) { [int64]$run.keep_bf16_extra_bytes } else { 0L }
    $extra = 0L
    $changed = @()
    $failed = $null
    if ($c.kind -ne 'baseline') {
      # Verify the converter's result against the intent, from its own output (header comment).
      if (-not (Test-SameSet @($lay.map.Keys) @($bi.map.Keys))) { $failed = "its linears are not the baseline's" }
      else {
        $changed = @(Sort-Ordinal @($lay.map.Keys | Where-Object { $lay.map[$_].p -cne $bi.map[$_].p }))
        $target = Get-TargetLayout $c
        $wrong = @($changed | Where-Object { $lay.map[$_].p -cne $target })
        $extraSet = @($changed | Where-Object { $intended -cnotcontains $_ })
        $missing = @($intended | Where-Object { $changed -cnotcontains $_ })
        if ($extraSet.Count -or $missing.Count -or $wrong.Count) {
          $failed = "changed $($changed.Count) linear(s), intended $($intended.Count): unintended [$($extraSet -join ', ')], " +
                    "unchanged [$($missing -join ', ')], not at $target [$(($wrong | ForEach-Object { "$_=$($lay.map[$_].p)" }) -join ', ')]"
        }
      }
      if (-not $failed) {
        $keptWant = switch ($c.kind) {
          'group' { $bi.kept }
          'keep' { @($bi.kept + $changed) }
          'unkeep' { @($bi.kept | Where-Object { $changed -cnotcontains $_ }) }
        }
        if (-not (Test-SameSet $keptNow $keptWant)) {
          $failed = "the converter's keep_bf16_linears ($($keptNow.Count)) is not the baseline's $(if ($c.kind -eq 'keep') { 'plus' } elseif ($c.kind -eq 'unkeep') { 'minus' } else { 'unchanged by' }) the candidate's set ($($keptWant.Count))"
        }
      }
      if (-not $failed) {
        foreach ($b in $changed) {
          $pred = (Get-LoadBytes $lay.map[$b].p $bi.map[$b].nk) - (Get-LoadBytes $bi.map[$b].p $bi.map[$b].nk)
          $meas = $lay.map[$b].bytes - (Get-LoadBytes $bi.map[$b].p $lay.map[$b].nk)
          if ($pred -ne $meas -or $lay.map[$b].nk -ne $bi.map[$b].nk) { $failed = "$b's bytes ($meas) differ from its shape's ($pred)"; break }
          $extra += $meas
        }
      }
      if (-not $failed) {
        # The converter's own accounting of the same change.
        if ($c.kind -eq 'group' -and $extra -ne ($grpExtra - $bi.group_extra)) {
          $failed = "the w4a16-load delta $extra B differs from the converter's w4a16_group_extra_bytes delta $($grpExtra - $bi.group_extra) B"
        }
        if ($c.kind -ne 'group' -and $lay.single -and $bi.single -and $extra -ne ($keepExtra - $bi.keep_extra)) {
          $failed = "the w4a16-load delta $extra B differs from the converter's keep_bf16_extra_bytes delta $($keepExtra - $bi.keep_extra) B"
        }
      }
    }
    # reused_from: the source and what the converter recomputed (null for a full conversion).
    $rf = $run.reused_from
    if (($reuse -contains '--reuse-tensors-from') -and -not $rf) { $failed = "no r4dx_convert_run.reused_from" }
    $reusedFrom = $null
    if ($rf) {
      $reusedFrom = [ordered]@{ path = $rf.path; header_sha256 = $rf.header_sha256; data_sha256 = $rf.data_sha256
        tensors_copied = $rf.tensors_copied; tensors_recomputed = $rf.tensors_recomputed
        linears_recomputed = @($rf.linears_recomputed) }
      # A reuse from the sweep's own baseline recomputes exactly the changed set.
      if (-not $failed -and $c.kind -ne 'baseline' -and $autoReuse -and -not (Test-SameSet @($rf.linears_recomputed) $changed)) {
        $failed = "reused_from.linears_recomputed ($(@($rf.linears_recomputed).Count)) is not the changed set ($($changed.Count))"
      }
    }
    if ($failed) {
      throw "[q3] $n FAILED verification: $failed -- see convert_$n.log; $partial left for inspection"
    }
    if ($c.kind -ne 'baseline') {
      Write-Host ("[q3] $n verified: {0} linear(s) -> {1}, {2:+0.0000;-0.0000} GiB (w4a16 load)" -f $changed.Count, (Get-TargetLayout $c), ($extra / 1GB))
    }
    # The container's identity for later runs: its data digest and its guard identity (null without
    # a guard, i.e. -NoReuse or a -ReuseFrom-less full conversion).
    $guard = $run.reuse_guard
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
      $meta = [ordered]@{ name = $n; kind = $c.kind; bases_regex = $c.bases_regex
        group = $(if ($c.kind -eq 'keep' -or $c.kind -eq 'baseline') { $null } else { $c.group })
        precision = $(if ($c.kind -eq 'baseline') { $null } else { Get-TargetLayout $c })
        extra_bytes = $extra; n_linears = $changed.Count; linears = @($changed)
        keep_bf16 = $keepRe; keep_bf16_linears = @($keptNow)
        w4a16_group_extra_bytes = $grpExtra; keep_bf16_extra_bytes = $keepExtra; single_layout = $lay.single
        convert_args = @($convArgs + $rule); reused_from = $reusedFrom
        data_sha256 = $dataSha; guard_json = $guardId
        # The baseline this candidate was planned and verified against (-Kl: it must still be the
        # measured one).
        baseline_data_sha256 = $(if ($bi) { $bi.data_sha256 } else { $null }) }
      if ($c.kind -eq 'baseline') {
        # Every linear's layout and N*K: what each candidate is planned and verified against.
        $ll = [ordered]@{}
        foreach ($b in (Sort-Ordinal @($lay.map.Keys))) { $ll[$b] = [ordered]@{ p = $lay.map[$b].p; nk = $lay.map[$b].nk } }
        $meta.linear_layouts = $ll
      }
      $meta | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $metaPath
      if ($c.kind -eq 'baseline') { $script:baseInfo = $null }  # re-read from the new meta.json
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
  # candidates.json for alloc_groups.py: exact byte deltas (a --layout w4a16 load, from the tensor
  # directories), the exact linear sets (for the allocator's overlap exclusion), KL from kl_report.py,
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
  # The recipe is the one the measured baseline was converted with (its convert_args), not this
  # run's -Recipe/-ExtraArgs: alloc_groups.py builds its merged --keep-bf16 from it. This run's
  # candidate list depends on the recipe's keep too (the unkeep savers exist only with one), so a
  # different keep is refused rather than collected under the wrong list.
  Assert-BaselineKeep
  $baseRecipe = $recipeAll
  if ($null -ne $bm.convert_args) {
    $baseRecipe = @($bm.convert_args)
  } else {
    Write-Host "[q3] baseline.meta.json has no convert_args (an older sweep's): candidates.json takes this run's -Recipe/-ExtraArgs"
  }
  $bmKept = if ($null -ne $bm.keep_bf16_linears) { @($bm.keep_bf16_linears) } else { $null }
  $rows = @()
  $unmeasured = @()
  $stale = @()
  foreach ($c in $candidates) {
    $klPath = Join-Path $OutDir "kl_$($c.name).json"
    $metaPath = Join-Path $OutDir "$($c.name).meta.json"
    if (-not (Test-Path $klPath) -or -not (Test-Path $metaPath)) { $unmeasured += $c.name; continue }
    $k = Get-Content $klPath -Raw | ConvertFrom-Json
    $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
    # A candidate that is not this ONE change against the measured baseline is left out, named:
    # converted by another binary / CPU / checkpoint / flags / input set (both guarded, identities
    # differ); under another definition (kind, regex, group); planned and verified against another
    # baseline container (baseline_data_sha256); or with a keep set that is not the baseline's plus /
    # minus / unchanged by its own set -- the guard identity no longer carries the recipe's
    # --keep-bf16 (keep candidates change it), so a recipe whose keep changed is caught here.
    $metaKind = if ($meta.kind) { "$($meta.kind)" } else { 'group' }
    $why = $null
    if ($bm.guard_json -and $meta.guard_json -and $meta.guard_json -ne $bm.guard_json) { $why = 'another guard identity' }
    elseif ($metaKind -ne $c.kind -or "$($meta.bases_regex)" -cne "$($c.bases_regex)" -or
            ($c.kind -ne 'keep' -and [int]$meta.group -ne $c.group)) { $why = 'another candidate definition' }
    elseif ($meta.baseline_data_sha256 -and $bm.data_sha256 -and "$($meta.baseline_data_sha256)" -ne "$($bm.data_sha256)") {
      $why = "verified against another baseline (data_sha256 $($meta.baseline_data_sha256))"
    } elseif ($null -ne $bmKept -and $null -ne $meta.keep_bf16_linears -and $meta.linears -is [array]) {
      $lin = @($meta.linears)
      $want = switch ($c.kind) {
        'group' { $bmKept }
        'keep' { @($bmKept + $lin) }
        'unkeep' { @($bmKept | Where-Object { $lin -cnotcontains $_ }) }
      }
      if (-not (Test-SameSet @($meta.keep_bf16_linears) @($want))) {
        $why = "its keep_bf16_linears ($(@($meta.keep_bf16_linears).Count)) is not the baseline's ($($bmKept.Count)) $(if ($c.kind -eq 'keep') { 'plus' } elseif ($c.kind -eq 'unkeep') { 'minus' } else { 'unchanged by' }) its set"
      }
    }
    if ($why) { $stale += "$($c.name) ($why)"; continue }
    $row = [ordered]@{ name = $c.name; kind = $c.kind; bases_regex = $c.bases_regex }
    if ($c.kind -ne 'keep') { $row.group = $c.group }
    $row.precision = Get-TargetLayout $c
    $row.delta_gib = [double]$meta.extra_bytes / 1GB
    $row.kl = $k.overall.mean_kl
    $row.top1 = $k.overall.top1_agreement_pct
    # An older sweep's meta.json has only a count here; the allocator then falls back to bases_regex.
    if ($meta.linears -is [array]) { $row.linears = @($meta.linears); $row.n_linears = @($meta.linears).Count }
    elseif ($null -ne $meta.linears) { $row.n_linears = [int]$meta.linears }
    $rows += $row
  }
  $baseRow = [ordered]@{ kl = $bj.overall.mean_kl; top1 = $bj.overall.top1_agreement_pct; weights_gib = $weights }
  # The recipe's resolved keep set: the allocator prints what its merged --keep-bf16 must resolve to,
  # and checks the regex matches exactly that among these and the baseline's w4a16 linears.
  if ($null -ne $bm.keep_bf16_linears) { $baseRow.keep_bf16_linears = @($bm.keep_bf16_linears) }
  if ($bm.linear_layouts) {
    $baseRow.quantized_linears = @(Sort-Ordinal @($bm.linear_layouts.PSObject.Properties |
        Where-Object { "$($_.Value.p)" -like 'w4a16.*' } | ForEach-Object { $_.Name }))
  }
  $out = [ordered]@{
    baseline = $baseRow
    default_group = $defaultGroup
    recipe = @($baseRecipe)
    candidates = $rows
  }
  if ($BudgetGib -gt 0) { $out.budget_gib = $BudgetGib }
  $cj = Join-Path $OutDir 'candidates.json'
  $out | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $cj
  Write-Host "[q3] $($rows.Count) measured candidate(s) -> $cj"
  if ($unmeasured) { Write-Host "[q3] $($unmeasured.Count) not measured, left out: $($unmeasured -join ', ')" }
  if ($stale) {
    Write-Host ("[q3] $($stale.Count) not a delta against the measured baseline (or not this list's definition), left out -- " +
                "delete their <name>.meta.json and kl_<name>.json to convert and measure them again: $($stale -join '; ')")
  }
  Run 'alloc' { & $Python tools\quant2\alloc_groups.py $cj --json-out (Join-Path $OutDir 'picks.json') }
  Get-Content (Join-Path $OutDir 'alloc.log')
}
