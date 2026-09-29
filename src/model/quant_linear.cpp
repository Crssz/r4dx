#include "quant_linear.h"

#include <stdexcept>

namespace r4dx::model {

const char* LayoutName(Layout l) {
  switch (l) {
    case Layout::kBf16: return "bf16";
    case Layout::kW4a16: return "w4a16";
    case Layout::kTrellis: return "trellis";
  }
  return "?";
}

Layout LayoutFromName(const std::string& name) {
  if (name == "bf16") return Layout::kBf16;
  if (name == "w4a16") return Layout::kW4a16;
  if (name == "trellis") return Layout::kTrellis;
  if (name == "mxfp4" || name == "w4a8") {
    throw std::runtime_error("r4dx::model::LayoutFromName: layout '" + name +
                             "' was retired (r4dx supports bf16, w4a16 and trellis only); "
                             "use a bf16, w4a16 or trellis container");
  }
  throw std::runtime_error("r4dx::model::LayoutFromName: unrecognized layout '" + name + "'");
}

}  // namespace r4dx::model
