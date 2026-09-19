add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE r4dx_core)
add_test(NAME test_core COMMAND test_core)
# See tests/kernels/CMakeLists.txt's comment on this property: pins the test to HIP device 1
# (the project's only permitted GPU) regardless of how ctest itself was invoked.
set_tests_properties(test_core PROPERTIES ENVIRONMENT "HIP_VISIBLE_DEVICES=1")
