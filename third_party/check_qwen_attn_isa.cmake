# check_qwen_attn_isa.cmake -- build-time guard that extending libr4d's attention templates for
# Gemma 4's sliding window (docs/gemma4-plan.md 3.3, task M1-32) did not change the code the Qwen
# instantiations compile to.
#
#   cmake -DISA=<r4d_attn_paged_h256_gqa6-gfx1201.s> -DBASELINE=<qwen_attn_isa.sha256> \
#         -DSTAMP=<file> [-DGQA2_ISA=<r4d_attn_paged_h256_gqa2-gfx1201.s>] -P check_qwen_attn_isa.cmake
#   cmake -DISA=<listing> -DBASELINE=<file> -DUPDATE_BASELINE=ON -P check_qwen_attn_isa.cmake
#
# ISA is a device-only -S listing of the Qwen attention unit (r4d_attn_paged_h256_gqa6.hip, the
# unit's own flags). The check normalises the two things that legitimately move between builds of the
# SAME code -- the mangled kernel names (the new defaulted WIN / EXT template parameters and the
# R4DKernelArgs<> argument type are part of every instantiation's name) and the per-compilation
# __hip_cuid_<hash> -- hashes the rest (every instruction, label, resource and metadata line,
# including .vgpr_count, scratch, spills and kernarg sizes) and compares it with BASELINE, the hash
# of the listing built from the sources BEFORE the window parameters existed. Any difference in code
# generation fails the build.
#
# When the baseline has to move on purpose (a ROCm / compiler update, or a deliberate change to a
# Qwen kernel): build the listing from the previous sources with the NEW toolchain (git stash / a
# worktree at the last good commit), run this script with -DUPDATE_BASELINE=ON on it, then rebuild
# from the current sources, confirm the check passes, and commit the new hash with the reason.
#
# GQA2_ISA (optional) is the windowed unit's listing: its kernels' resource usage is only REPORTED
# (a status line), not gated -- like the Qwen decode kernels they are tuned for speed, not for zero
# spills (r4d_attn_decode_h256_gqa6 spills at this compiler version too).

if(NOT DEFINED ISA OR NOT EXISTS "${ISA}")
  message(FATAL_ERROR "check_qwen_attn_isa: ISA file '${ISA}' does not exist")
endif()
if(NOT DEFINED BASELINE)
  message(FATAL_ERROR "check_qwen_attn_isa: BASELINE not set")
endif()

function(qa_normalized_hash out isa)
  file(READ "${isa}" text)
  string(REGEX REPLACE "_Z[0-9A-Za-z_$.]+" "SYM" text "${text}")
  string(REGEX REPLACE "__hip_cuid_[0-9a-f]+" "CUID" text "${text}")
  string(SHA256 h "${text}")
  set(${out} "${h}" PARENT_SCOPE)
endfunction()

qa_normalized_hash(hash "${ISA}")

if(UPDATE_BASELINE)
  file(WRITE "${BASELINE}" "${hash}\n")
  message(STATUS "check_qwen_attn_isa: baseline ${BASELINE} <- ${hash}")
  return()
endif()

if(NOT DEFINED STAMP)
  message(FATAL_ERROR "check_qwen_attn_isa: STAMP not set")
endif()
if(NOT EXISTS "${BASELINE}")
  message(FATAL_ERROR "check_qwen_attn_isa: baseline ${BASELINE} is missing")
endif()
file(READ "${BASELINE}" want)
string(STRIP "${want}" want)
if(NOT hash STREQUAL want)
  message(FATAL_ERROR "check_qwen_attn_isa: the Qwen attention listing ${ISA} no longer matches the "
                      "baseline (normalised sha256 ${hash}, baseline ${want}): a change to "
                      "third_party/libr4d's attention templates altered what the Qwen instantiations "
                      "compile to. See the header of this script for how to compare and, if the change is "
                      "intended, to move the baseline.")
endif()

set(extra "")
if(DEFINED GQA2_ISA AND EXISTS "${GQA2_ISA}")
  file(READ "${GQA2_ISA}" g2)
  string(REPLACE ";" "#" g2 "${g2}")
  string(REPLACE "[" "<" g2 "${g2}")
  string(REPLACE "]" ">" g2 "${g2}")
  string(REPLACE "\n" ";" g2lines "${g2}")
  set(name "")
  set(kernels 0)
  set(max_vgpr 0)
  set(max_scratch 0)
  foreach(line IN LISTS g2lines)
    if(line MATCHES "^[ \t-]*\\.name:[ \t]+(_Z[^ \t]+)")
      set(name "${CMAKE_MATCH_1}")
      if(name MATCHES "r4d_attn_(prefill|decode)_kernel")
        math(EXPR kernels "${kernels} + 1")
      else()
        set(name "")
      endif()
    elseif(NOT name STREQUAL "")
      if(line MATCHES "^[ \t-]*\\.vgpr_count:[ \t]+([0-9]+)" AND CMAKE_MATCH_1 GREATER max_vgpr)
        set(max_vgpr ${CMAKE_MATCH_1})
      elseif(line MATCHES "^[ \t-]*\\.private_segment_fixed_size:[ \t]+([0-9]+)" AND CMAKE_MATCH_1 GREATER max_scratch)
        set(max_scratch ${CMAKE_MATCH_1})
      endif()
    endif()
  endforeach()
  set(extra "; windowed gqa2 unit: ${kernels} attention kernels, max ${max_vgpr} VGPRs, max scratch ${max_scratch} B (reported, not gated)")
endif()
message(STATUS "check_qwen_attn_isa: Qwen attention instantiations unchanged (normalised sha256 ${hash})${extra}")
file(WRITE "${STAMP}" "ok\n")
