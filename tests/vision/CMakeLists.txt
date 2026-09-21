# tests/vision/: CPU-only golden tests for src/vision (docs/vision.md). No HIP device, no r4dx
# container -- every one of these is plain host arithmetic compared against
# tools/reference/golden_out/*.safetensors, so they are registered as ordinary always-on tests with
# no ENVIRONMENT setting, the same way tests/server's HIP-free tests are. They still exit 77
# (CTest SKIPPED, not FAILED) when the golden files are absent, since those come from the real
# checkpoint and are not vendored into the repo -- the convention tests/model and tests/tokenizer
# already use.
#
# r4dx_convert is linked for its header-only SafetensorsReader (the golden reader) and, through it,
# r4dx_core's bf16 conversions; r4dx_vision is the component under test.
# test_image_prompt is the one entry here that needs no golden either -- ExpandImagePlaceholders is
# pure integer bookkeeping over a token vector, so it is built from hand-written inputs and never
# takes the exit-77 path. It is in the same list anyway: it links the same targets and the extra
# SKIP_RETURN_CODE property is simply never exercised.
set(R4DX_VISION_TESTS
  test_image_prompt
  test_preprocess
  test_position_ids
  test_vision_index
)

foreach(t ${R4DX_VISION_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE r4dx_vision r4dx_convert r4dx_core)
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# test_vision_tower is the one test in this directory that DOES need a GPU and the real container:
# it runs r4dx::vision::VisionTower (src/vision/vision_tower.cpp) against all three of
# tools/reference/vision_golden.py's cases. It loads only the container's vision.* tensors
# (~0.90 GiB), not the 27B text model. Same HIP_VISIBLE_DEVICES=1 + SKIP_RETURN_CODE 77 convention
# every other real-container GPU test in this repo uses.
add_executable(test_vision_tower test_vision_tower.cpp)
target_link_libraries(test_vision_tower PRIVATE r4dx_vision_tower r4dx_convert r4dx_core)
add_test(NAME test_vision_tower COMMAND test_vision_tower)
set_tests_properties(test_vision_tower PROPERTIES
  ENVIRONMENT "HIP_VISIBLE_DEVICES=1"
  SKIP_RETURN_CODE 77
)

# tool_vision_bench: encode time + VRAM high-water for 448/1024/1536-pixel images, with the real
# 27B text model loaded alongside (docs/vision.md "Large images"). Built but deliberately NOT
# add_test()'d -- it prints numbers for a human and for the docs, it asserts no pass/fail contract,
# so it must never affect the ctest count. Same convention as tests/model/tool_hseed_drift and
# tests/kernels/tool_sampler_bench. Links r4dx_model because "does a 1536x1536 image fit NEXT TO
# the loaded model" is the question it exists to answer.
add_executable(tool_vision_bench tool_vision_bench.cpp)
target_link_libraries(tool_vision_bench PRIVATE r4dx_model r4dx_vision_tower r4dx_convert r4dx_core)

# tool_vision_chat: the end-to-end splice/mrope driver (docs/vision.md "Text-side splicing") --
# real container, real tokenizer, real image, greedy answer. Same "built but never add_test()'d"
# convention as tool_vision_bench above. The explicit third_party include ahead of the link list is
# the same nlohmann-v3.11.3-vs-ROCm-v3.12.0 ABI ordering fix src/cli/CMakeLists.txt documents --
# this is the second target in the repo to link r4dx_tokenizer and a HIP-including library at once.
add_executable(tool_vision_chat tool_vision_chat.cpp)
target_include_directories(tool_vision_chat PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(tool_vision_chat PRIVATE r4dx_model r4dx_vision_tower r4dx_tokenizer
                                                r4dx_convert r4dx_core)
