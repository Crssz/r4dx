// dev/w4mod.hip - a one-kernel pybind module, so a variant of the 4-bit GEMM builds in seconds and
// several of them can be loaded into ONE process and timed round robin. The same kernel drifts
// 216-246 us between docker runs on this box, so a variant compared across runs is not compared.
#include <pybind11/pybind11.h>
#include "../r4d.h"
namespace py = pybind11;
PYBIND11_MODULE(TU_NAME, m) {
  m.def("gemm", &r4d_gemm_w4a16_nt_m64, py::arg("a"), py::arg("wq"), py::arg("wsz"), py::arg("c"),
        py::arg("M"), py::arg("K"), py::arg("N"), py::arg("WV"), py::arg("SK"), py::arg("MB"),
        py::arg("NPW"), py::arg("NT"), py::arg("stream"));
  m.attr("MAX_M") = r4d_gemm_w4a16_nt_m64_max_m();
  m.attr("GROUP") = r4d_gemm_w4a16_nt_m64_group();
  m.attr("TAG") = R4D_W4_TAG;
}
