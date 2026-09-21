# src/vision: the HOST-side half of the vision tower (docs/vision.md) -- image decode, the
# preprocessing pipeline that produces `pixel_values`/`image_grid_thw`, the text-side mrope
# position-id builder, and the grid_thw-derived index tables (pos-embed interpolation taps, vision
# rope positions, cu_seqlens) the tower's device code will consume.
#
# Deliberately CPU-only and dependency-light: no HIP, no r4dx_core, no r4dx_model. Everything here
# is plain integer/float arithmetic over std::vector, which is what lets tests/vision/** run as
# ordinary always-on unit tests (no GPU, no container) and lets a future server/CLI path call the
# preprocessing without dragging a device context along. The only third-party dependency is the
# vendored stb_image.h, pulled in through third_party's include root.
add_library(r4dx_vision STATIC
  image_decode.cpp
  preprocess.cpp
  position_ids.cpp
  vision_index.cpp
)
target_include_directories(r4dx_vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(r4dx_vision PRIVATE ${CMAKE_SOURCE_DIR}/third_party)

# r4dx_vision_tower: the DEVICE half (docs/vision.md, Milestone 8 stage 3) -- the 333 vision.*
# tensors in VRAM and `Qwen3_5VisionModel.forward` on top of them. Kept a separate target from
# r4dx_vision above precisely so that library's "no HIP, no device context" property survives:
# tests/vision's three preprocessing/index tests still link only r4dx_vision and still run on a
# machine with no GPU at all.
#
# It links r4dx_core (DeviceBuffer/Arena/Stream + the typed r4d wrappers), r4dx_kernels (the five
# vision primitives) and r4d_core (r4d_gemm_bf16_nt_m64, r4d_attn_vit_h72_bf16), plus
# r4dx_convert for its header-only SafetensorsReader -- the same reader src/model/container.cpp
# uses, rather than a second mmap implementation. It deliberately does NOT link r4dx_model:
# src/model depends on THIS target (Container owns the loaded weights), so the edge only runs one
# way.
add_library(r4dx_vision_tower STATIC
  vision_weights.cpp
  vision_tower.cpp
)
target_include_directories(r4dx_vision_tower PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(r4dx_vision_tower PUBLIC r4dx_vision r4dx_core r4dx_kernels r4d_core
                                                r4dx_convert)
