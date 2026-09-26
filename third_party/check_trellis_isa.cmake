# check_trellis_isa.cmake -- build-time register and schedule check of libr4d's trellis GEMM
# (docs/trellis-kernel.md 4.3 "every instantiation's VGPR count <= 190, checked at build time", and
# 4.4's one-block decode).
#
#   cmake -DISA=<r4d_gemm_trellis_nt_m64-gfx1201.s> -DSTAMP=<file> -P check_trellis_isa.cmake
#
# Reads a device-only -S listing and fails the build unless
#
#   - every r4d_gemm_trellis_nt_m64_raw_kernel and r4d_trellis_reconstruct_f16_kernel instantiation
#     has .vgpr_count <= 190 (8 waves per SIMD on gfx1201, docs/r9700.md C14), no scratch
#     (.private_segment_fixed_size 0) and no VGPR/SGPR spills (code-object metadata). The
#     instantiated (NP, U, MT) set is the kernel's own r4d_tq_max_mt table; a compiler that allocates
#     more registers than the table assumed fails here instead of silently running at fewer waves or
#     through scratch;
#
#   - no VALU instruction of a GEMM kernel reads a VGPR written by an inline-asm VALU instruction
#     among the 3 VALU before it with no s_delay_alu in between (a "near dependency"). LLVM puts an
#     s_delay_alu in front of a VALU that reads a recent VALU result only when it generated the
#     producer itself; behind inline asm it cannot, and the consumer then stalls the SIMD's VALU for
#     every wave on it. The first M1 kernel (one asm per hash) had 14-275 of them per K loop and
#     decoded at ~0.52 VALU per clock; r4d_trellis_k4_decode is one asm block whose dependencies are
#     all >= 8 VALU apart and which ends in an s_delay_alu, so the count must be 0. Only the asm
#     blocks and the 3 VALU after each are parsed (a near dependency on an asm producer can only be
#     there), and the check fails if it finds no asm VALU at all, so it cannot pass vacuously.
#
# On success it writes STAMP, so the check re-runs only when the listing changes.

if(NOT DEFINED ISA OR NOT EXISTS "${ISA}")
  message(FATAL_ERROR "check_trellis_isa: ISA file '${ISA}' does not exist")
endif()
if(NOT DEFINED STAMP)
  message(FATAL_ERROR "check_trellis_isa: STAMP not set")
endif()

set(max_vgpr 190)

# One list element per line; `;` and square brackets are list syntax in CMake, so the listing's
# comments start with `#` here and VGPR ranges read v<a:b>.
file(READ "${ISA}" text)
string(REPLACE ";" "#" text "${text}")
string(REPLACE "[" "<" text "${text}")
string(REPLACE "]" ">" text "${text}")
string(REPLACE "\n" ";" lines "${text}")

# The VGPR numbers named in `text` (vN and v<a:b> ranges, expanded).
function(tq_vregs out text)
  string(REGEX MATCHALL "v[0-9]+|v<[0-9]+:[0-9]+>" toks "${text}")
  set(r "")
  foreach(t IN LISTS toks)
    if(t MATCHES "^v<([0-9]+):([0-9]+)>$")
      foreach(i RANGE ${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
        list(APPEND r ${i})
      endforeach()
    elseif(t MATCHES "^v([0-9]+)$")
      list(APPEND r ${CMAKE_MATCH_1})
    endif()
  endforeach()
  set(${out} "${r}" PARENT_SCOPE)
endfunction()

# ---- near dependencies behind inline asm (function bodies, label to .Lfunc_end) -----------------
set(fn "")
set(in_asm OFF)
set(tail 0)                      # VALU still to check after the last asm block ended
set(asm_valu 0)
set(near 0)
set(near_msgs "")
foreach(line IN LISTS lines)
  if(line MATCHES "^(_Z[0-9A-Za-z_]*r4d_gemm_trellis_nt_m64_raw_kernel[0-9A-Za-z_]*):")
    set(fn "${CMAKE_MATCH_1}")
    set(in_asm OFF)
    set(tail 0)
    set(w1_r "")
    set(w2_r "")
    set(w3_r "")
    set(w1_a OFF)
    set(w2_a OFF)
    set(w3_a OFF)
    continue()
  endif()
  if(fn STREQUAL "")
    continue()
  endif()
  if(line MATCHES "^\\.Lfunc_end")
    set(fn "")
    continue()
  endif()
  if(line MATCHES "##ASMSTART")
    set(in_asm ON)
    continue()
  elseif(line MATCHES "##ASMEND")
    set(in_asm OFF)
    set(tail 3)
    continue()
  endif()
  if(NOT in_asm AND tail EQUAL 0)
    continue()
  endif()
  if(line MATCHES "^[ \t]+s_delay_alu")
    set(w1_r "")
    set(w2_r "")
    set(w3_r "")
    if(NOT in_asm)
      set(tail 0)
    endif()
    continue()
  endif()
  if(NOT line MATCHES "^[ \t]+(v_[a-z0-9_]+)[ \t]+([^#]*)")
    continue()
  endif()
  set(op "${CMAKE_MATCH_1}")
  set(args "${CMAKE_MATCH_2}")
  # Destinations: the first operand of each half (a VOPD pair is `a d, s :: b d, s`).
  set(dst_text "")
  set(src_text "")
  string(REPLACE "::" "|" args "${args}")
  string(REPLACE "|" ";" halves "${args}")
  foreach(h IN LISTS halves)
    if(h MATCHES "^[ \t]*(v_[a-z0-9_]+[ \t]+)?([^,]*)(,(.*))?$")
      string(APPEND dst_text " ${CMAKE_MATCH_2}")
      string(APPEND src_text " ${CMAKE_MATCH_4}")
    endif()
  endforeach()
  tq_vregs(dst "${dst_text}")
  tq_vregs(src "${src_text}")
  if(op MATCHES "^v_wmma")
    list(APPEND src ${dst})          # the accumulator is read too
  endif()
  if(in_asm)
    math(EXPR asm_valu "${asm_valu} + 1")
  endif()
  foreach(w w1 w2 w3)
    if(${w}_a AND NOT "${${w}_r}" STREQUAL "")
      set(hit OFF)
      foreach(r IN LISTS src)
        list(FIND ${w}_r ${r} idx)
        if(idx GREATER -1)
          set(hit ON)
          break()
        endif()
      endforeach()
      if(hit)
        math(EXPR near "${near} + 1")
        list(LENGTH near_msgs nm)
        if(nm LESS 8)
          string(STRIP "${line}" l)
          list(APPEND near_msgs "${fn}: ${l} (${w} back)")
        endif()
        break()
      endif()
    endif()
  endforeach()
  set(w3_r "${w2_r}")
  set(w3_a ${w2_a})
  set(w2_r "${w1_r}")
  set(w2_a ${w1_a})
  set(w1_r "${dst}")
  set(w1_a ${in_asm})
  if(NOT in_asm)
    math(EXPR tail "${tail} - 1")
  endif()
endforeach()

# ---- registers (amdhsa.kernels metadata) ------------------------------------------------------
# In each kernel's metadata map the keys are sorted, so .name precedes .private_segment_fixed_size,
# the spill counts and .vgpr_count of the same kernel.
set(name "")
set(count 0)
set(worst 0)
set(failures "")
foreach(line IN LISTS lines)
  if(line MATCHES "^[ \t-]*\\.name:[ \t]+([^ \t]+)")
    set(name "${CMAKE_MATCH_1}")
    if(NOT name MATCHES "r4d_gemm_trellis_nt_m64_raw_kernel|r4d_trellis_reconstruct_f16_kernel")
      set(name "")
    else()
      math(EXPR count "${count} + 1")
    endif()
  elseif(NOT name STREQUAL "")
    if(line MATCHES "^[ \t-]*\\.private_segment_fixed_size:[ \t]+([0-9]+)")
      if(NOT CMAKE_MATCH_1 EQUAL 0)
        list(APPEND failures "${name}: scratch ${CMAKE_MATCH_1} bytes")
      endif()
    elseif(line MATCHES "^[ \t-]*\\.(vgpr|sgpr)_spill_count:[ \t]+([0-9]+)")
      if(NOT CMAKE_MATCH_2 EQUAL 0)
        list(APPEND failures "${name}: ${CMAKE_MATCH_2} ${CMAKE_MATCH_1} spills")
      endif()
    elseif(line MATCHES "^[ \t-]*\\.vgpr_count:[ \t]+([0-9]+)")
      set(v "${CMAKE_MATCH_1}")
      if(v GREATER worst)
        set(worst ${v})
      endif()
      if(v GREATER max_vgpr)
        list(APPEND failures "${name}: ${v} VGPRs > ${max_vgpr}")
      endif()
    endif()
  endif()
endforeach()

if(count EQUAL 0)
  message(FATAL_ERROR "check_trellis_isa: no trellis kernel metadata found in ${ISA}")
endif()
if(asm_valu EQUAL 0)
  message(FATAL_ERROR "check_trellis_isa: no inline-asm VALU found in the GEMM kernels of ${ISA} "
                      "(the decode is expected to be one asm block; update this check with it)")
endif()
if(failures)
  string(REPLACE ";" "\n  " msg "${failures}")
  message(FATAL_ERROR "check_trellis_isa: ${ISA}:\n  ${msg}\n(every instantiation must fit "
                      "${max_vgpr} VGPRs with no scratch -- shrink r4d_tq_max_mt in "
                      "third_party/libr4d/r4d_gemm_trellis_nt_m64.hip)")
endif()
if(near GREATER 0)
  string(REPLACE ";" "\n  " msg "${near_msgs}")
  message(FATAL_ERROR "check_trellis_isa: ${ISA}: ${near} VALU instructions read an inline-asm VALU "
                      "result 1-3 VALU later with no s_delay_alu between (they stall the SIMD), e.g.\n"
                      "  ${msg}\n(keep r4d_trellis_k4_decode's dependencies >= 4 VALU apart inside "
                      "the block and its trailing s_delay_alu)")
endif()
message(STATUS "check_trellis_isa: ${count} trellis kernels OK (max ${worst} VGPRs <= ${max_vgpr}, "
               "no scratch, no spills); ${asm_valu} asm VALU in the GEMM kernels, 0 near "
               "dependencies on them")
file(WRITE "${STAMP}" "ok\n")
