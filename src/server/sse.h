// r4dx::server SSE (Server-Sent Events) chunk formatting -- the wire format
// /v1/chat/completions and /v1/completions stream in when `"stream": true` (task point 1: "SSE
// streaming with 'data: ... [DONE]'"). Pure string formatting, no HIP/network dependency, so it's
// unit-testable on its own (tests/server/test_sse.cpp).
#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace r4dx::server {

// "data: <compact JSON>\n\n" -- the standard SSE event framing every OpenAI-compatible client
// expects (a blank line terminates the event; no explicit "event:" field, matching OpenAI's own
// server, which only ever sends unnamed "message" events on this endpoint).
std::string FormatSseEvent(const nlohmann::json& payload);

// The sentinel OpenAI's streaming API sends after the last real chunk, literally "data: [DONE]"
// (not a JSON value) -- clients match on this exact string to know the stream is finished.
std::string FormatSseDone();

}  // namespace r4dx::server
