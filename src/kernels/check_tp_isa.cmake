# check_tp_isa.cmake -- build-time codegen check of the tensor-parallel all-reduce (docs/tp.md 6.3.5).
#
#   cmake -DISA=<r4dx_tp_kernels-gfx1201.s> -DFLAGS=<hipcc flags, one string> -DSTAMP=<file>
#         -P check_tp_isa.cmake
#
# The production protocol is tp_bench's flag(drain 3, acq 0). Its correctness argument
# (tools/tp_bench/README.md "drain 3 note") is a property of the emitted ISA, not of the source:
# in WGP mode __syncthreads' workgroup release already waits for every wave's stores, and tid0's
# system-scope release store of the flag starts with `global_wb scope:SCOPE_SYS`, which is L2-wide
# and writes back the other waves' pushed lines. So this fails the build unless, in the function
# body of r4dx_tp_ar_flag_bf16_kernel:
#   1. the kernel is compiled in WGP mode (.amdhsa_workgroup_processor_mode 1) and the flags do not
#      contain -mcumode;
#   2. the flag store is found STRUCTURALLY, not by its scope: the post-push barrier is the first
#      s_barrier_wait after the first 16-byte global push store, and the flag store is the first
#      global/flat/buffer store or atomic after that barrier (LDS and scratch stores excluded). It
#      must be a 32-bit SCOPE_SYS store, immediately preceded by `global_wb scope:SCOPE_SYS` (no
#      other store, label or branch in between). Anchoring on "the first SCOPE_SYS store" instead
#      would, after a scope downgrade of the flag's release, latch onto a later system-scope store
#      (note_failure's ABORT word) and pass (docs/tp.md Appendix B N43);
#   3. acquire side: between the flag store and the next barrier there is a relaxed SCOPE_SYS 32-bit
#      poll load and a STANDALONE `global_inv scope:SCOPE_SYS` -- the one acquire fence after a
#      successful spin (spin_acq = 0), i.e. one not directly after a load (the spin_acq = 1 acquire
#      loads carry their own global_inv).
# On success it writes STAMP, so the check re-runs only when the ISA changes.

if(NOT DEFINED ISA OR NOT EXISTS "${ISA}")
  message(FATAL_ERROR "check_tp_isa: ISA file '${ISA}' does not exist")
endif()
if(NOT DEFINED STAMP)
  message(FATAL_ERROR "check_tp_isa: STAMP not set")
endif()
if("${FLAGS}" MATCHES "-mcumode")
  message(FATAL_ERROR "check_tp_isa: r4dx_tp_kernels.hip must be built in WGP mode, but the hipcc "
                      "flags contain -mcumode (docs/tp.md 6.3.1; tools/tp_bench/README.md \"drain 3 note\")")
endif()

set(kernel "r4dx_tp_ar_flag_bf16_kernel")

# One list element per line. `;` (asm comments) and square brackets (register ranges) are list
# syntax in CMake, so they are replaced first.
file(READ "${ISA}" text)
string(REPLACE ";" "#" text "${text}")
string(REPLACE "[" "<" text "${text}")
string(REPLACE "]" ">" text "${text}")
string(REPLACE "\n" ";" lines "${text}")

set(in_body FALSE)
set(in_desc FALSE)
set(found_body FALSE)
set(wgp_mode "")
set(body "")
foreach(line IN LISTS lines)
  if(in_body)
    if(line MATCHES "^\\.Lfunc_end[0-9]+:")
      set(in_body FALSE)
    else()
      list(APPEND body "${line}")
    endif()
  elseif(NOT found_body AND line MATCHES "^[_A-Za-z0-9.$]*${kernel}[_A-Za-z0-9.$]*:")
    set(in_body TRUE)
    set(found_body TRUE)
  endif()
  if(line MATCHES "^[ \t]*\\.amdhsa_kernel [_A-Za-z0-9.$]*${kernel}")
    set(in_desc TRUE)
  elseif(in_desc)
    if(line MATCHES "^[ \t]*\\.end_amdhsa_kernel")
      set(in_desc FALSE)
    elseif(line MATCHES "^[ \t]*\\.amdhsa_workgroup_processor_mode[ \t]+([0-9]+)")
      set(wgp_mode "${CMAKE_MATCH_1}")
    endif()
  endif()
endforeach()

if(NOT found_body)
  message(FATAL_ERROR "check_tp_isa: no function body for ${kernel} in ${ISA}")
endif()
if(NOT wgp_mode STREQUAL "1")
  message(FATAL_ERROR "check_tp_isa: ${kernel} is not compiled in WGP mode "
                      "(.amdhsa_workgroup_processor_mode '${wgp_mode}', need 1)")
endif()

list(LENGTH body n)
# Memory stores and atomics that are not LDS (ds_) or scratch: global_/flat_/buffer_ store or atomic.
set(mem_store_re "^[ \t]*(global|flat|buffer)_(store|atomic)")

# 2a. the push: the first 16-byte global/flat store; the post-push barrier: the first s_barrier_wait
#     after it (the __syncthreads that follows the drain).
set(push_idx -1)
set(i 0)
while(i LESS n)
  list(GET body ${i} line)
  if(line MATCHES "^[ \t]*(global|flat)_store_b128")
    set(push_idx ${i})
    break()
  endif()
  math(EXPR i "${i} + 1")
endwhile()
if(push_idx LESS 0)
  message(FATAL_ERROR "check_tp_isa: ${kernel} has no 16-byte global store (the push)")
endif()
set(bar_idx -1)
math(EXPR i "${push_idx} + 1")
while(i LESS n)
  list(GET body ${i} line)
  if(line MATCHES "s_barrier_wait|s_barrier$|s_barrier[ \t]")
    set(bar_idx ${i})
    break()
  endif()
  math(EXPR i "${i} + 1")
endwhile()
if(bar_idx LESS 0)
  message(FATAL_ERROR "check_tp_isa: ${kernel}: no workgroup barrier after the push stores (body line ${push_idx})")
endif()

# 2b. the flag store: the first memory store/atomic after the post-push barrier. It must be the
#     system-scope 32-bit release store -- whatever its scope, it is the flag (tid0's first store).
set(flag_idx -1)
math(EXPR i "${bar_idx} + 1")
while(i LESS n)
  list(GET body ${i} line)
  if(line MATCHES "${mem_store_re}")
    set(flag_idx ${i})
    break()
  endif()
  math(EXPR i "${i} + 1")
endwhile()
if(flag_idx LESS 0)
  message(FATAL_ERROR "check_tp_isa: ${kernel}: no store after the post-push barrier (the flag release)")
endif()
list(GET body ${flag_idx} flag_line)
if(NOT flag_line MATCHES "^[ \t]*(global|flat)_store_b32 .*scope:SCOPE_SYS")
  message(FATAL_ERROR "check_tp_isa: the first store after the post-push barrier in ${kernel} ('${flag_line}') "
                      "is not a 32-bit system-scope store: the flag's release must be SCOPE_SYS "
                      "(tools/tp_bench/README.md \"drain 3 note\")")
endif()

# 2c. walk back from the flag store to its global_wb
set(wb_idx -1)
math(EXPR i "${flag_idx} - 1")
while(i GREATER_EQUAL 0)
  list(GET body ${i} line)
  if(line MATCHES "global_wb scope:SCOPE_SYS")
    set(wb_idx ${i})
    break()
  endif()
  if(line MATCHES "_store_|_atomic_|^\\.LBB|s_branch|s_cbranch|s_setpc|s_endpgm")
    break()
  endif()
  math(EXPR i "${i} - 1")
endwhile()
if(wb_idx LESS 0 OR NOT wb_idx GREATER bar_idx)
  message(FATAL_ERROR "check_tp_isa: the flag's release store in ${kernel} ('${flag_line}') is not "
                      "immediately preceded by `global_wb scope:SCOPE_SYS` after the post-push barrier; "
                      "the drain-3 protocol is not safe with this codegen (tools/tp_bench/README.md "
                      "\"drain 3 note\")")
endif()

# 3. acquire side, between the flag store and the next barrier (the one before the reduce): the
#    relaxed system-scope poll, and a standalone system-scope invalidate (the acquire fence).
set(acq_end ${n})
math(EXPR i "${flag_idx} + 1")
while(i LESS n)
  list(GET body ${i} line)
  if(line MATCHES "s_barrier_wait|s_barrier$|s_barrier[ \t]")
    set(acq_end ${i})
    break()
  endif()
  math(EXPR i "${i} + 1")
endwhile()
set(poll_idx -1)
set(fence_idx -1)
math(EXPR i "${flag_idx} + 1")
while(i LESS acq_end)
  list(GET body ${i} line)
  if(poll_idx LESS 0 AND line MATCHES "^[ \t]*(global|flat)_load_b32 .*scope:SCOPE_SYS")
    set(poll_idx ${i})
  endif()
  if(fence_idx LESS 0 AND line MATCHES "^[ \t]*global_inv scope:SCOPE_SYS")
    # standalone = the nearest preceding instruction other than s_wait_* / s_nop is not a load
    math(EXPR j "${i} - 1")
    set(prev "")
    while(j GREATER_EQUAL 0)
      list(GET body ${j} prev)
      if(NOT prev MATCHES "^[ \t]*(s_wait_|s_nop)")
        break()
      endif()
      math(EXPR j "${j} - 1")
    endwhile()
    if(NOT prev MATCHES "^[ \t]*(global|flat|buffer)_load")
      set(fence_idx ${i})
    endif()
  endif()
  math(EXPR i "${i} + 1")
endwhile()
if(poll_idx LESS 0 OR fence_idx LESS 0)
  message(FATAL_ERROR "check_tp_isa: ${kernel}: after the flag store (body line ${flag_idx}) expected a "
                      "system-scope 32-bit poll load and a standalone `global_inv scope:SCOPE_SYS` (the "
                      "acquire fence after the relaxed spin) before the next barrier; found poll at "
                      "${poll_idx}, fence at ${fence_idx}")
endif()

message(STATUS "check_tp_isa: ${kernel} OK (WGP mode; push -> barrier -> global_wb scope:SCOPE_SYS -> "
               "flag store -> SYS poll -> SYS acquire fence at body lines "
               "${push_idx}/${bar_idx}/${wb_idx}/${flag_idx}/${poll_idx}/${fence_idx})")
file(WRITE "${STAMP}" "ok\n")
