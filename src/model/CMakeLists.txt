# src/model: the layer graph (docs/architecture.md) -- container loader, linear apply, GDN layer,
# MLP, final norm + lm_head, embedding gather+upload. Owned by this component: every file directly
# under src/model/ and src/model/kernels/. src/model/attention/** (the full-attention layer) is a
# different agent's; guarded below so this target's configure/build never blocks on it existing.

# ---- src/model/kernels/model_kernels.hip: two tiny device kernels (bf16->f16 cast, bf16->fp32
# widen) this component needs and neither r4d.h nor src/kernels provides -- see
# src/model/kernels/model_kernels.h. Built by hipcc directly, the same recipe
# src/kernels/CMakeLists.txt uses (CMake's HIP language fights clang-cl on this toolchain --
# docs/build-windows.md); R4D_ROCM_ROOT/R4D_GFX_ARCH are already-set CACHE variables by the time
# this subdirectory runs (root CMakeLists.txt adds third_party before src/model).
set(R4DX_MODEL_HIPCC "${R4D_ROCM_ROOT}/bin/hipcc.exe")
set(R4DX_MODEL_ROCM_BIN "${R4D_ROCM_ROOT}/bin")
set(R4DX_MODEL_ROCM_LLVM_BIN "${R4D_ROCM_ROOT}/lib/llvm/bin")
set(R4DX_MODEL_DEVICE_LIB_PATH "${R4D_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")

if(NOT EXISTS "${R4DX_MODEL_HIPCC}")
  message(FATAL_ERROR "hipcc.exe not found at ${R4DX_MODEL_HIPCC}; set -DR4D_ROCM_ROOT=<path to ROCm SDK>")
endif()

set(R4DX_MODEL_KERNELS_FLAGS
  -O3 -std=c++17 --offload-arch=${R4D_GFX_ARCH}
  -Wno-unused-result -ffp-contract=off
  --rocm-path=${R4D_ROCM_ROOT}
  -DNDEBUG
  -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
  -I "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  -I "${CMAKE_CURRENT_SOURCE_DIR}/../core/include"
)
if(EXISTS "${R4DX_MODEL_DEVICE_LIB_PATH}")
  list(APPEND R4DX_MODEL_KERNELS_FLAGS --rocm-device-lib-path=${R4DX_MODEL_DEVICE_LIB_PATH})
endif()

set(R4DX_MODEL_KERNELS_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/objs")
file(MAKE_DIRECTORY "${R4DX_MODEL_KERNELS_OBJ_DIR}")
set(R4DX_MODEL_KERNELS_HIPCC_PATH "${R4DX_MODEL_ROCM_BIN};${R4DX_MODEL_ROCM_LLVM_BIN};$ENV{PATH}")

set(R4DX_MODEL_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernels/model_kernels.hip")
set(R4DX_MODEL_KERNELS_OBJ "${R4DX_MODEL_KERNELS_OBJ_DIR}/model_kernels.obj")

add_custom_command(
  OUTPUT "${R4DX_MODEL_KERNELS_OBJ}"
  COMMAND "${CMAKE_COMMAND}" -E env "PATH=${R4DX_MODEL_KERNELS_HIPCC_PATH}"
          "${R4DX_MODEL_HIPCC}" ${R4DX_MODEL_KERNELS_FLAGS} -c "${R4DX_MODEL_KERNELS_SRC}" -o "${R4DX_MODEL_KERNELS_OBJ}"
  DEPENDS "${R4DX_MODEL_KERNELS_SRC}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels/model_kernels.h"
          "${CMAKE_CURRENT_SOURCE_DIR}/../core/include/r4dx/core/dtype.hpp"
  COMMENT "hipcc (${R4D_GFX_ARCH}) model_kernels.obj"
  VERBATIM
)
set_source_files_properties("${R4DX_MODEL_KERNELS_OBJ}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(r4dx_model_kernels STATIC "${R4DX_MODEL_KERNELS_OBJ}")
set_target_properties(r4dx_model_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(r4dx_model_kernels PUBLIC r4dx_hip_runtime)

# ---- r4dx_model_linear: Layout/QuantLinear (quant_linear.h/.cpp) + ApplyLinear (linear.h/.cpp) --
# Split out (decode-perf pass, 2026-09-19) so r4dx_model_attention (AttentionLayer's qg/o linears,
# see attention/CMakeLists.txt) can dispatch through the SAME ApplyLinear GDN/MLP/lm_head already
# use, without linking the rest of r4dx_model (Container/GdnLayer/Mlp/Model) -- that would create a
# link cycle, since r4dx_model itself links r4dx_model_attention (below).
add_library(r4dx_model_linear STATIC
  quant_linear.cpp
  linear.cpp
)
target_include_directories(r4dx_model_linear PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_model_linear PUBLIC
  r4dx_core
  r4dx_kernels
  r4d_core
  r4dx_hip_runtime
  r4dx_model_kernels
)

# ---- r4dx_tp_shard: tensor-parallel sharding rules and byte plans (docs/tp.md 5.2) ---------------
# CPU-only on purpose (no HIP include, no HIP call): tests/model/test_tp_shard.cpp and
# test_tp_config.cpp run it against the converter's real packers without a device. Needs only
# model_config.h (this directory) and nlohmann/json (third_party).
add_library(r4dx_tp_shard STATIC tp/tp_shard.cpp)
target_include_directories(r4dx_tp_shard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${CMAKE_SOURCE_DIR}/third_party)

# ---- r4dx_tp: the TpComm implementations (docs/tp.md 6.1, Appendix A) ----------------------------
# NoopComm (tp/tp_comm_noop.cpp, docs/tp.md 6.4); P3: TpGroup (tp/tp_group.cpp -- the pinned host
# mailbox, seq base, recovery) and HostMailboxComm (tp/tp_comm_host_mailbox.cpp -- the two-GPU
# all-reduce of docs/tp.md 6.3); P2b: EmulatedComm (tp/tp_comm_emulated.cpp -- both ranks on one
# device, docs/tp.md 6.5). The interface itself is header-only in src/core
# (r4dx/core/tp_comm.hpp) so the header-only AttentionLayer can call through it.
add_library(r4dx_tp STATIC tp/tp_comm_noop.cpp tp/tp_group.cpp tp/tp_comm_host_mailbox.cpp
                           tp/tp_comm_emulated.cpp)
target_include_directories(r4dx_tp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_tp PUBLIC r4dx_core r4dx_kernels)
# /EHc-: the r4dx_tp_* entry points are extern "C" but report bad arguments and HIP errors by
# throwing (src/kernels/include/r4dx/kernels/tp_kernels.h). Under the default /EHsc a caller assumes
# extern "C" never throws and such a throw kills the process instead of reaching TpModel's
# rank-error path (see tests/kernels/CMakeLists.txt's /EHc- comment).
set_source_files_properties(tp/tp_group.cpp tp/tp_comm_host_mailbox.cpp tp/tp_comm_emulated.cpp
                            PROPERTIES COMPILE_OPTIONS "/EHc-")

# ---- r4dx_model: the layer graph itself -------------------------------------------------------
add_library(r4dx_model STATIC
  container.cpp
  gdn_layer.cpp
  mlp.cpp
  final_lm_head.cpp
  mtp_head.cpp
  dflash_draft.cpp
  model.cpp
  # docs/tp.md 2.8: the TextModel surface src/cli and src/server call -- LocalTextModel (TP=1, a
  # one-line forward per method onto Model) and TpModel (the tensor-parallel facade, 2.2-2.9).
  local_text_model.cpp
  tp_model.cpp
)
target_include_directories(r4dx_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_model PUBLIC
  r4dx_model_linear
  r4dx_core
  r4dx_kernels
  r4d_core
  r4dx_hip_runtime
  r4dx_model_kernels
  r4dx_convert   # header-only: SafetensorsReader (container.cpp) + nlohmann/json (model_config.h)
  # The vision tower (docs/vision.md): Container owns the loaded vision.* weights and Model owns
  # the VisionTower built on them, so the dependency edge runs src/model -> src/vision, never back.
  r4dx_vision_tower
  # Tensor parallel (docs/tp.md): the sharded Container::Load path plans every rank slice with
  # r4dx_tp_shard; Model reaches its communicator through r4dx_tp.
  r4dx_tp_shard
  r4dx_tp
)

# src/model/attention/** is another agent's -- guard so this configure never blocks on it existing
# yet, per the task's own instruction.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/attention/CMakeLists.txt")
  add_subdirectory(attention)
  if(TARGET r4dx_model_attention)
    target_link_libraries(r4dx_model PUBLIC r4dx_model_attention)
  endif()
endif()
