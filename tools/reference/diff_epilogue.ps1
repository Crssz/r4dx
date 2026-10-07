# diff_epilogue.ps1 -- is the int8 GEMM's epilogue still the f16 M = 256 kernel's?
#
#   powershell -NoProfile -File tools\reference\diff_epilogue.ps1 [-SelfTest]
#
# docs/int8-prefill.md "Production path": r4d_trellis_i8.h's epilogue (the LDS reduction over the K slices, the fp32
# partials in ws, the ticket protocol, the last block's y-sum, the FWHT-128 stages, svh, out_scale and the one bf16
# rounding) is a COPY of third_party/libr4d/r4d_gemm_trellis_nt_m256.hip's, not a shared function: moving the shipped
# kernel's code into a helper changes its ISA (docs/int8-prefill.md "OFF path"), and the shipped kernels must not move.
# A copy can drift. This script compares the two as text, after removing comments and whitespace and mapping the
# two files' function prefixes (r4d_t256_ / i8g_) together, on exactly the parts that define the arithmetic:
#   * the helper functions bf16_rn, lane_stage and bfly, whole bodies;
#   * the ordered trace of FWHT stages of the NP = 1 path (the lane stages 1 2 4, the bit-3 butterfly loop, lane stage
#     8, then the explicit butterflies of bits 5 and 6): the i8 kernel has no NP, so the m256 side is read at NP = 1;
#   * the ticket protocol, literally (contributors, the fences, the atomicAdd, the self-reset, the flag);
#   * the last block's accumulation (the y loop, the ws unit address, the v += unit sum) and svh load;
#   * the LDS reduction's slice rules (owner of tile i, writer index, the slice-order sum);
#   * the output expression: bf16_rn((v * sv) * out_scale).
# Exit 0 and "EPILOGUE EQUAL" when they agree, 1 and the first difference otherwise. -SelfTest also edits one fact in
# memory and requires the comparison to fail (the check is known to see a change).
param([switch]$SelfTest)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$f16 = [IO.File]::ReadAllText((Join-Path $root 'third_party\libr4d\r4d_gemm_trellis_nt_m256.hip'))
$i8 = [IO.File]::ReadAllText((Join-Path $root 'third_party\libr4d\r4d_trellis_i8.h'))

function Normalize([string]$t) {
  $t = $t -replace "`r", ''
  $t = [regex]::Replace($t, '//[^\n]*', '')
  $t = $t -replace 'r4d_t256_', 'X_' -replace 'i8g_', 'X_'
  $t = [regex]::Replace($t, '\s+', '')
  $t
}
# the text of a function / kernel whose definition contains `anchor` (compact form), by brace matching
function Body([string]$t, [string]$anchor) {
  $i = $t.IndexOf($anchor)
  if ($i -lt 0) { throw "anchor not found: $anchor" }
  $open = $t.IndexOf('{', $i)
  $depth = 0
  for ($k = $open; $k -lt $t.Length; ++$k) {
    if ($t[$k] -eq '{') { ++$depth } elseif ($t[$k] -eq '}') { --$depth; if ($depth -eq 0) { return $t.Substring($i, $k - $i + 1) } }
  }
  throw "unbalanced braces after: $anchor"
}

function Facts([string]$text, [string]$kernel_anchor) {
  $t = Normalize $text
  $facts = [ordered]@{}
  $facts['bf16_rn'] = Body $t 'unsignedshortX_bf16_rn(floatf)'
  $facts['bfly'] = Body $t 'voidX_bfly(v8f&lo,v8f&hi)'
  $facts['lane_stage'] = Body $t 'voidX_lane_stage(v8f(&v)[N],intlane)'
  $kernel = Body $t $kernel_anchor
  # the final phase: from `constintg0=n0/128*128;` to the end of the kernel
  $g0 = $kernel.IndexOf('constintg0=n0/128*128;')
  if ($g0 -lt 0) { throw 'g0 marker not found' }
  $tail = $kernel.Substring($g0)
  # ticket protocol, literally
  $m = [regex]::Match($tail, 'constunsignedcontributors=\(unsigned\)\(SKG\*NBLK\);__threadfence\(\);__syncthreads\(\);float\*flag=\(float\*\)lds;if\(tid==0\)\{[^}]*\}__syncthreads\(\);if\(flag\[0\]==0\.f\)return;__threadfence\(\);')
  $facts['ticket'] = $m.Value
  # FWHT stage trace (NP = 1 reading): lane stages, the bit-3 loop, the plain explicit butterflies
  $trace = New-Object System.Collections.Generic.List[string]
  foreach ($mm in [regex]::Matches($tail, 'X_lane_stage<(\d+)>\(v,lane\)|for\(intbp=0;bp<NBLK(?:\*NP)?;\+\+bp\)X_bfly\(v\[bp\*2\],v\[bp\*2\+1\]\)|X_bfly\(v\[(\d)\*2\+f\],v\[(\d)\*2\+f\]\)')) {
    if ($mm.Groups[1].Success) { $trace.Add("L$($mm.Groups[1].Value)") }
    elseif ($mm.Groups[2].Success) { $trace.Add("B$($mm.Groups[2].Value)$($mm.Groups[3].Value)") }
    else { $trace.Add('B3loop') }
  }
  $facts['fwht_trace'] = ($trace -join ' ')
  # the y loop and the ws unit address, the accumulation
  $facts['y_sum'] = ([regex]::Matches($tail, 'for\(inty=0;y<SKG;\+\+y\)\{|ws\+\(size_t\)y\*M\*N\+\(\(size_t\)R\*\(N/Wc\)\+cb0\+b\)\*\(16\*Wc\)|\+__builtin_shufflevector\(lo,hi,0,1,2,3,4,5,6,7\);') | ForEach-Object { $_.Value }) -join '|'
  $facts['svh'] = ([regex]::Matches($tail, 'svh\[g0\+b\*Wc\+') | ForEach-Object { $_.Value }) -join '|'
  # the output expression: (v * sv) * out_scale under bf16_rn, whatever the indices are called
  $out = [regex]::Match($tail, 'X_bf16_rn\((?:FWHT\?)?\(v\[[^\]]+\]\[[^\]]+\]\[e\]\*sv\[[^;]*?\]\)\*out_scale|X_bf16_rn\((?:FWHT\?)?\(v\[[^\]]+\]\[e\]\*sv\[[^;]*?\]\)\*out_scale')
  $facts['output'] = if ($out.Success) { 'bf16_rn((v*sv)*out_scale)' } else { 'NOT FOUND' }
  # the SK reduction's slice rules, in the kernel before g0 (the part it shares: owner, writer index, the sum)
  $head = $kernel.Substring(0, $g0)
  $facts['sk_rules'] = ([regex]::Matches($head, 'consto=i%SKW;|if\(ks!=o\)\{constintkw=ks>o\?ks-1:ks;|for\(intsl=0;sl<SKW;\+\+sl\)\{|constintkw=sl>o\?sl-1:sl;|s=s\+term;|if\(ks==o\)\{') | ForEach-Object { $_.Value }) -join '|'
  $facts
}

function Compare-Facts($a, $b) {
  $bad = 0
  foreach ($k in $a.Keys) {
    if ($a[$k] -ceq $b[$k]) { continue }
    ++$bad
    Write-Host "DIFFERENT: $k"
    Write-Host "  f16 : $($a[$k])"
    Write-Host "  int8: $($b[$k])"
  }
  $bad
}

$fa = Facts $f16 'voidr4d_gemm_trellis_nt_m256_kernel('
$fb = Facts $i8 'voidX_kernel('
foreach ($k in $fa.Keys) { if ([string]::IsNullOrEmpty($fa[$k]) -or $fa[$k] -eq 'NOT FOUND') { Write-Host "f16 side: fact '$k' is empty (the extraction no longer matches the source)"; exit 1 } }
foreach ($k in $fb.Keys) { if ([string]::IsNullOrEmpty($fb[$k]) -or $fb[$k] -eq 'NOT FOUND') { Write-Host "int8 side: fact '$k' is empty (the extraction no longer matches the source)"; exit 1 } }
$bad = Compare-Facts $fa $fb
if ($bad -ne 0) { Write-Host "EPILOGUE DIFFERS ($bad facts)"; exit 1 }
Write-Host "EPILOGUE EQUAL ($($fa.Count) facts: $($fa.Keys -join ', '))"
if ($SelfTest) {
  # one fact edited in memory (the first FWHT stage's mask): the comparison must now fail
  $mut = $i8.Replace('i8g_lane_stage<1>(v, lane);', 'i8g_lane_stage<8>(v, lane);')
  if ($mut -ceq $i8) { Write-Host 'self test: the mutation did not apply'; exit 1 }
  $fm = Facts $mut 'voidX_kernel('
  $n = (Compare-Facts $fa $fm)
  if ($n -eq 0) { Write-Host 'SELF TEST FAILED: an edited FWHT stage was not seen'; exit 1 }
  Write-Host "self test: an edited FWHT stage is seen ($n facts differ)"
}
exit 0
