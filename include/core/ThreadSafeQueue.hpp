#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>

namespace asdun {

template <typename T>
class ThreadSafeQueue {
 public:
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(item));
    }
    cv_.notify_one();
  }

  std::optional<T> popWithTimeout(std::uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue_.empty(); })) {
      return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<T> empty;
    std::swap(queue_, empty);
  }

 private:
  std::mutex mutex_{};
  std::condition_variable cv_{};
  std::queue<T> queue_{};
};

}  // namespace asdun
