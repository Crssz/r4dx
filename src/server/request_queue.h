// r4dx::server::BoundedQueue -- a small thread-safe bounded FIFO used in two places: (1) the
// top-level request queue (HTTP handler threads -> the single Engine worker thread that owns the
// GPU/Model, task point 2's "a request queue with a worker thread ... a --max-queue and 429
// beyond"), where TryPush's non-blocking "false means full" return is exactly the 429 signal; and
// (2) per-request SSE delivery (the worker thread -> the HTTP handler thread driving httplib's
// chunked content provider, response_sink.h's StreamingSink), where the blocking Push is used
// instead so a slow client throttles generation via backpressure rather than dropping tokens.
// `Close()` makes every subsequent Push/TryPush a no-op and unblocks any waiting Pop() once
// drained, so a producer or consumer can shut the queue down without a sentinel value.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace r4dx::server {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  // Returns false (without blocking) if the queue is already at max_size or closed -- the queue's
  // "reject the caller" signal (e.g. the HTTP handler answering 429).
  bool TryPush(T item) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_ || items_.size() >= max_size_) return false;
    items_.push_back(std::move(item));
    cv_.notify_all();
    return true;
  }

  // Blocks until there is room in the queue (or it is closed, in which case this silently
  // no-ops -- a producer racing a consumer's Close() should not deadlock or throw).
  void Push(T item) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return closed_ || items_.size() < max_size_; });
    if (closed_) return;
    items_.push_back(std::move(item));
    cv_.notify_all();
  }

  // Blocks until an item is available or the queue is closed and drained. Returns std::nullopt in
  // the latter case.
  std::optional<T> Pop() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return !items_.empty() || closed_; });
    if (items_.empty()) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    cv_.notify_all();
    return item;
  }

  // Idempotent: safe to call more than once (e.g. both a client-disconnect path and a normal
  // end-of-stream path racing to close the same StreamingSink queue).
  void Close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.size();
  }

  bool Closed() const {
    std::lock_guard<std::mutex> lock(mu_);
    return closed_;
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> items_;
  size_t max_size_;
  bool closed_ = false;
};

}  // namespace r4dx::server
