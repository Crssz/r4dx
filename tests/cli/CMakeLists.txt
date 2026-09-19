# tests/cli/: r4dx-cli argument-parsing unit test (src/cli/cli_args.h). Pure CPU, no HIP device
# needed -- unlike tests/model/**, this is a plain add_test() with no ENVIRONMENT/SKIP_RETURN_CODE.
add_executable(test_cli_args test_args.cpp)
target_link_libraries(test_cli_args PRIVATE r4dx_cli)
add_test(NAME test_cli_args COMMAND test_cli_args)
