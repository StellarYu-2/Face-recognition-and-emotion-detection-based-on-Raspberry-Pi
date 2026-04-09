#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>

#include "core/Types.hpp"

namespace asdun {

class CameraManager {
 public:
  CameraManager(std::string source, int width, int height, int fps);
  ~CameraManager();

  bool start();
  void stop();
  bool getLatestFrame(FramePacket& out, std::uint32_t timeout_ms = 30);

 private:
  bool openCapture();
  void captureLoop();

  std::string source_;
  int width_{640};
  int height_{480};
  int fps_{30};

  std::atomic<bool> running_{false};
  std::thread capture_thread_{};

  cv::VideoCapture cap_{};
  std::mutex frame_mutex_{};
  std::condition_variable frame_cv_{};
  FramePacket latest_frame_{};
  bool has_frame_{false};
  std::uint64_t frame_counter_{0};
  std::uint64_t last_delivered_frame_id_{0};
};

}  // namespace asdun
