// dev/w4a8mod.hip - one-kernel pybind module for the W4A8 GEMM, same shape as w4mod.hip.
#include <pybind11/pybind11.h>
#include "../r4d.h"
namespace py = pybind11;
PYBIND11_MODULE(TU_NAME, m) {
  m.def("gemm", &r4d_gemm_w4a8_nt_m64, py::arg("a"), py::arg("ascale"), py::arg("wq"),
        py::arg("ws"), py::arg("c"), py::arg("M"), py::arg("K"), py::arg("N"), py::arg("WV"),
        py::arg("SK"), py::arg("MB"), py::arg("NPW"), py::arg("NT"), py::arg("stream"));
  m.def("quant_act", &r4d_quant_act_i8, py::arg("a"), py::arg("q"), py::arg("s"),
        py::arg("M"), py::arg("K"), py::arg("stream"));
  m.attr("MAX_M") = r4d_gemm_w4a8_nt_m64_max_m();
  m.attr("GROUP") = r4d_gemm_w4a8_nt_m64_group();
  m.attr("APERM") = r4d_gemm_w4a8_nt_m64_aperm();
  m.attr("TAG") = R4D_W4_TAG;
}
