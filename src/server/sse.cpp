#include "sse.h"

namespace r4dx::server {

std::string FormatSseEvent(const nlohmann::json& payload) {
  return "data: " + payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n\n";
}

std::string FormatSseDone() { return "data: [DONE]\n\n"; }

}  // namespace r4dx::server
