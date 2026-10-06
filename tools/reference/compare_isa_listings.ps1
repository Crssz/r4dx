# compare_isa_listings.ps1 <before.s> <after.s> -- are two `hipcc --cuda-device-only -S` listings the same code?
#
# For checking that a change to a libr4d source left the kernels it was not meant to touch compiling to
# the same ISA (docs/int8-prefill.md, "OFF path"; third_party/check_qwen_attn_isa.cmake does it for the Qwen
# attention by hash). Both listings are normalised: comments, mangled symbol names (_Z...), per-function
# block labels (.LBB<fn>_<n>), .Lfunc_begin/_end numbers, the __hip_cuid_ hash and trailing blanks. Then:
#   1. every function body (label to .Lfunc_end), hashed, as a multiset -- the instructions of each kernel;
#   2. the whole listing as a sorted multiset of lines -- that also covers the code-object metadata
#      (.vgpr_count, .private_segment_fixed_size, kernarg sizes, ...).
# Exit 0 and "IDENTICAL" when both agree. -Pattern restricts (1) to function names matching a regex.
# A name change alone (a new defaulted template parameter changes every instantiation's mangled name) does
# not count as a difference.
param(
  [Parameter(Mandatory = $true)][string]$Before,
  [Parameter(Mandatory = $true)][string]$After,
  [string]$Pattern = '.'
)
$ErrorActionPreference = 'Stop'
$Before = (Resolve-Path $Before).Path
$After = (Resolve-Path $After).Path

function Normalize([string]$line) {
  (($line -replace ';.*$', '') -replace '_Z[0-9A-Za-z_$.]+', 'SYM' -replace '\.LBB\d+_', '.LBB_' `
     -replace '\.Lfunc_(end|begin)\d+', '.Lf' -replace '__hip_cuid_[0-9a-f]+', 'CUID' -replace 'BB\d+_', 'BB_').Trim()
}

function Summarize([string]$path) {
  $sha = [Security.Cryptography.SHA256]::Create()
  $bodies = New-Object Collections.Generic.List[string]
  $lines = New-Object Collections.Generic.List[string]
  $name = $null
  $sb = $null
  foreach ($raw in [IO.File]::ReadLines($path)) {
    $l = Normalize $raw
    if ($l -ne '') { $lines.Add($l) }
    if ($null -eq $name) {
      if ($raw -match '^(_Z[0-9A-Za-z_$.]+):') {
        $cand = $Matches[1]
        if ($cand -match $Pattern) {
          $name = $cand
          $sb = New-Object Text.StringBuilder
        }
      }
      continue
    }
    if ($raw -match '^\.Lfunc_end') {
      $bodies.Add([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($sb.ToString()))))
      $name = $null
      continue
    }
    if ($l -ne '') { [void]$sb.AppendLine($l) }
  }
  $bodies.Sort([StringComparer]::Ordinal)
  $lines.Sort([StringComparer]::Ordinal)
  [pscustomobject]@{ Bodies = $bodies; Lines = $lines }
}

$a = Summarize $Before
$b = Summarize $After
$bodiesSame = ($a.Bodies.Count -eq $b.Bodies.Count) -and -not (Compare-Object $a.Bodies $b.Bodies)
$linesSame = ($a.Lines.Count -eq $b.Lines.Count) -and -not (Compare-Object $a.Lines $b.Lines)
"function bodies matching '$Pattern': $($a.Bodies.Count) before, $($b.Bodies.Count) after, multiset identical: $bodiesSame"
"whole listing: $($a.Lines.Count) normalised lines before, $($b.Lines.Count) after, multiset identical: $linesSame"
if ($bodiesSame -and $linesSame) { 'IDENTICAL'; exit 0 }
'DIFFERENT'
exit 1
