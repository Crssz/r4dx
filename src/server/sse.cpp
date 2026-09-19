#include "sse.h"

namespace r4dx::server {

std::string FormatSseEvent(const nlohmann::json& payload) {
  return "data: " + payload.dump() + "\n\n";
}

std::string FormatSseDone() { return "data: [DONE]\n\n"; }

}  // namespace r4dx::server
