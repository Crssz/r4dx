// tests/server/test_request_queue.cpp -- pure CPU unit test for src/server/request_queue.h's
// BoundedQueue: capacity/TryPush-rejects-when-full (the 429 signal), FIFO order, and Close()
// unblocking a waiting Pop() from another thread.
#include <chrono>
#include <cstdio>
#include <thread>

#include "request_queue.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                   #cond);                                                    \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

void TestTryPushRejectsWhenFull() {
  r4dx::server::BoundedQueue<int> q(2);
  CHECK(q.TryPush(1));
  CHECK(q.TryPush(2));
  CHECK(!q.TryPush(3));  // at capacity -- caller's 429 signal
  CHECK(q.Size() == 2);
}

void TestFifoOrder() {
  r4dx::server::BoundedQueue<int> q(8);
  for (int i = 0; i < 5; ++i) CHECK(q.TryPush(i));
  for (int i = 0; i < 5; ++i) {
    auto v = q.Pop();
    CHECK(v.has_value());
    CHECK(*v == i);
  }
}

void TestCloseUnblocksPop() {
  r4dx::server::BoundedQueue<int> q(4);
  std::thread closer([&q] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.Close();
  });
  const auto v = q.Pop();  // blocks until Close() above
  CHECK(!v.has_value());
  closer.join();
}

void TestTryPushAfterCloseFails() {
  r4dx::server::BoundedQueue<int> q(4);
  q.Close();
  CHECK(!q.TryPush(1));
  CHECK(!q.Pop().has_value());
}

void TestBlockingPushWaitsForRoom() {
  r4dx::server::BoundedQueue<int> q(1);
  CHECK(q.TryPush(1));
  std::thread pusher([&q] { q.Push(2); });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  CHECK(q.Size() == 1);  // pusher should still be blocked (queue full)
  const auto first = q.Pop();
  CHECK(first.has_value() && *first == 1);
  pusher.join();  // now unblocked
  const auto second = q.Pop();
  CHECK(second.has_value() && *second == 2);
}

void TestProducerConsumerThreaded() {
  r4dx::server::BoundedQueue<int> q(4);
  const int kCount = 1000;
  std::thread producer([&q] {
    for (int i = 0; i < kCount; ++i) q.Push(i);
    q.Close();
  });
  int received = 0;
  int expected = 0;
  bool order_ok = true;
  while (true) {
    auto v = q.Pop();
    if (!v) break;
    if (*v != expected) order_ok = false;
    ++expected;
    ++received;
  }
  producer.join();
  CHECK(received == kCount);
  CHECK(order_ok);
}

}  // namespace

int main() {
  TestTryPushRejectsWhenFull();
  TestFifoOrder();
  TestCloseUnblocksPop();
  TestTryPushAfterCloseFails();
  TestBlockingPushWaitsForRoom();
  TestProducerConsumerThreaded();

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all request-queue checks passed\n");
  return 0;
}
