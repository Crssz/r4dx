#include "request_log.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <utility>

#include "nlohmann/json.hpp"

#ifdef _WIN32
#include <share.h>
#endif

namespace r4dx::server {

namespace {

using OJson = nlohmann::ordered_json;

// Seconds since 1970-01-01 of `t` read as if it were UTC (Howard Hinnant's days_from_civil). The
// difference of this for a local and a UTC breakdown of one instant is the UTC offset, without
// tm_gmtoff (not available with the MSVC runtime).
int64_t AsUtcSeconds(const std::tm& t) {
  int64_t y = static_cast<int64_t>(t.tm_year) + 1900;
  const int64_t m = static_cast<int64_t>(t.tm_mon) + 1;
  const int64_t d = t.tm_mday;
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const int64_t yoe = y - era * 400;
  const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  const int64_t days = era * 146097 + doe - 719468;
  return days * 86400 + t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec;
}

double Round(double v, double scale) {
  if (!std::isfinite(v)) return 0.0;
  return std::round(v * scale) / scale;
}

// JSON null for an unset optional, else the value.
template <typename T>
OJson Opt(const std::optional<T>& v) {
  return v ? OJson(*v) : OJson(nullptr);
}
OJson OptMs(const std::optional<double>& v) { return v ? OJson(Round(*v, 1000.0)) : OJson(nullptr); }

std::FILE* OpenAppend(const std::string& path) {
  // The path arrives as argv does: narrow, in the process's own code page on Windows (which is what
  // path(std::string) assumes there too), so a non-ASCII folder name survives where u8path would
  // misread it.
  const std::filesystem::path p(path);
#ifdef _WIN32
  // "ab": binary, so a "\n" stays one byte (no CRLF translation), and append, so the file is created
  // when missing and never truncated. _SH_DENYNO (what fopen itself uses, unlike fopen_s's deny-write)
  // lets any reader tail or copy the file while the server runs.
  return _wfsopen(p.c_str(), L"ab", _SH_DENYNO);
#else
  return std::fopen(p.string().c_str(), "ab");
#endif
}

std::string ErrnoText(int err) {
  char buf[128] = {};
#ifdef _WIN32
  if (strerror_s(buf, sizeof(buf), err) != 0) return "error " + std::to_string(err);
  return buf;
#else
  return std::strerror(err);
#endif
}

}  // namespace

std::string FormatRequestLogLine(const RequestLogRecord& rec) {
  OJson o = OJson::object();
  o["ts"] = rec.timestamp;
  o["request_id"] = rec.request_id.empty() ? OJson(nullptr) : OJson(rec.request_id);
  o["endpoint"] = rec.endpoint;
  o["stream"] = Opt(rec.stream);
  o["http_status"] = rec.http_status;
  o["error_status"] = Opt(rec.error_status);
  o["finish_reason"] = Opt(rec.finish_reason);
  o["cancelled"] = rec.cancelled;
  o["thinking"] = Opt(rec.thinking);
  o["max_tokens"] = Opt(rec.max_tokens);
  o["temperature"] = rec.temperature ? OJson(Round(*rec.temperature, 1e6)) : OJson(nullptr);
  o["tools_present"] = rec.tools_count ? OJson(*rec.tools_count > 0) : OJson(nullptr);
  o["tools_count"] = Opt(rec.tools_count);
  o["image_count"] = Opt(rec.image_count);
  o["prompt_tokens"] = Opt(rec.prompt_tokens);
  o["prompt_n"] = Opt(rec.prompt_n);
  o["cached_tokens"] = Opt(rec.cached_tokens);
  o["completion_tokens"] = Opt(rec.completion_tokens);
  o["reasoning_tokens"] = Opt(rec.reasoning_tokens);
  o["queue_wait_ms"] = OptMs(rec.queue_wait_ms);
  o["prompt_ms"] = OptMs(rec.prompt_ms);
  o["predicted_ms"] = OptMs(rec.predicted_ms);
  // Same arithmetic as BuildTimingsJson (openai_types.cpp): tokens over the matching seconds, 0 when
  // the time is 0. Unset when the pieces are not known.
  if (rec.prompt_n && rec.prompt_ms) {
    o["prompt_per_second"] = Round(*rec.prompt_ms > 0.0 ? *rec.prompt_n / (*rec.prompt_ms / 1000.0) : 0.0, 100.0);
  } else {
    o["prompt_per_second"] = nullptr;
  }
  if (rec.completion_tokens && rec.predicted_ms) {
    o["predicted_per_second"] =
        Round(*rec.predicted_ms > 0.0 ? *rec.completion_tokens / (*rec.predicted_ms / 1000.0) : 0.0, 100.0);
  } else {
    o["predicted_per_second"] = nullptr;
  }
  o["speculative"] = Opt(rec.speculative);
  o["draft_n"] = Opt(rec.draft_n);
  o["draft_n_accepted"] = Opt(rec.draft_n_accepted);
  o["full_reset"] = Opt(rec.full_reset);
  o["checkpoint_restore"] = Opt(rec.checkpoint_restore);
  o["reset_ms"] = OptMs(rec.reset_ms);
  o["restore_ms"] = OptMs(rec.restore_ms);
  o["image_n"] = Opt(rec.image_n);
  o["image_ms"] = OptMs(rec.image_ms);
  // `replace`: nothing written here is caller text, but a dump must never throw on a stray byte.
  return o.dump(-1, ' ', false, OJson::error_handler_t::replace);
}

std::string FormatIso8601(const std::tm& local, int utc_offset_minutes, int millis) {
  const int off = utc_offset_minutes < 0 ? -utc_offset_minutes : utc_offset_minutes;
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03d%c%02d:%02d", local.tm_year + 1900,
                local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec, millis,
                utc_offset_minutes < 0 ? '-' : '+', off / 60, off % 60);
  return buf;
}

std::string LocalIso8601Now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const int millis = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
  std::tm local{}, utc{};
#ifdef _WIN32
  localtime_s(&local, &t);
  gmtime_s(&utc, &t);
#else
  localtime_r(&t, &local);
  gmtime_r(&t, &utc);
#endif
  const int offset_minutes = static_cast<int>((AsUtcSeconds(local) - AsUtcSeconds(utc)) / 60);
  return FormatIso8601(local, offset_minutes, millis < 0 ? 0 : millis);
}

std::unique_ptr<RequestLog> RequestLog::Open(const std::string& path, std::string* error) {
  std::FILE* f = nullptr;
  int err = 0;
  try {
    errno = 0;
    f = OpenAppend(path);
    err = errno;
  } catch (const std::exception& e) {  // a path the filesystem layer refuses to convert
    if (error) *error = std::string("cannot open request log '") + path + "' for append: " + e.what();
    return nullptr;
  }
  if (f == nullptr) {
    if (error) {
      *error = std::string("cannot open request log '") + path + "' for append: " + ErrnoText(err);
    }
    return nullptr;
  }
  return std::unique_ptr<RequestLog>(new RequestLog(f, path));
}

std::unique_ptr<RequestLog> RequestLog::FromStream(std::FILE* file, std::string path) {
  return std::unique_ptr<RequestLog>(new RequestLog(file, std::move(path)));
}

RequestLog::~RequestLog() {
  if (file_ != nullptr) std::fclose(file_);
}

void RequestLog::Write(RequestLogRecord rec) noexcept {
  try {
    if (rec.timestamp.empty()) rec.timestamp = LocalIso8601Now();
    std::string line = FormatRequestLogLine(rec);
    line += '\n';
    std::lock_guard<std::mutex> lock(mu_);
    const bool ok = std::fwrite(line.data(), 1, line.size(), file_) == line.size() && std::fflush(file_) == 0;
    if (ok) {
      ++lines_written_;
      return;
    }
    if (++write_failures_ == 1) {
      std::fprintf(stderr,
                   "[r4dx-server][warn] request log '%s': write failed (%s); requests are unaffected, "
                   "further failures are counted silently\n",
                   path_.c_str(), ErrnoText(errno).c_str());
    }
  } catch (...) {
    // A formatting or allocation failure: drop this line, never the request.
    if (++write_failures_ == 1) {
      std::fprintf(stderr, "[r4dx-server][warn] request log '%s': could not format a line; dropped\n",
                   path_.c_str());
    }
  }
}

}  // namespace r4dx::server
