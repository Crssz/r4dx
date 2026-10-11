Public headers of the qwen4_exp kernels and host helpers (docs/qwen4exp-plan.md). Layout contract in
src/model/qwen4exp/CMakeLists.txt: `kernels/*.hip`, `host/*.cpp`, and the headers in this directory are
picked up by glob; do not edit the CMake files to add one.
