# tests/model/qwen4exp: one executable + one ctest per file, picked up by glob (CONFIGURE_DEPENDS), so
# parallel authors never edit this file.
#   test_*.cpp        link r4dx_model_qwen4exp_kernels only (kernels, host helpers; no Container/Model)
#   test_model_*.cpp  link r4dx_model (Container / Model / TextModel of the arch)
# Every test gets the label q4exp; files named test_gpu_* or test_model_* additionally get q4exp_gpu and
# run on device 1 (HIP_VISIBLE_DEVICES=1; the test may set both devices itself). Exit code 77 = SKIP
# (fixture missing). Fixtures are found through env vars the test documents; tests never download.

file(GLOB Q4_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
foreach(src ${Q4_TEST_SOURCES})
  get_filename_component(name "${src}" NAME_WE)
  add_executable(${name} "${src}")
  if(name MATCHES "^test_model_")
    target_link_libraries(${name} PRIVATE r4dx_model r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
  else()
    target_link_libraries(${name} PRIVATE r4dx_model_qwen4exp_kernels r4dx_core r4d_core)
  endif()
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../kernels")
  add_test(NAME ${name} COMMAND ${name})
  set(labels q4exp)
  if(name MATCHES "^test_gpu_" OR name MATCHES "^test_model_")
    list(APPEND labels q4exp_gpu)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "${labels}" SKIP_RETURN_CODE 77)
endforeach()
