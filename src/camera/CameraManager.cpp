#include "camera/CameraManager.hpp"

#include <chrono>
#include <iostream>

namespace asdun {

namespace {

bool isDigitsOnly(const std::string& s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

}  // namespace

CameraManager::CameraManager(std::string source, int width, int height, int fps)
    : source_(std::move(source)), width_(width), height_(height), fps_(fps) {}

CameraManager::~CameraManager() { stop(); }

bool CameraManager::openCapture() {
  if (cap_.isOpened()) {
    cap_.release();
  }

  bool opened = false;
  if (source_.rfind("gst:", 0) == 0) {
    const std::string pipeline = source_.substr(4);
    opened = cap_.open(pipeline, cv::CAP_GSTREAMER);
  } else if (isDigitsOnly(source_)) {
    opened = cap_.open(std::stoi(source_));
  } else {
    opened = cap_.open(source_, cv::CAP_ANY);
  }

  if (!opened) {
    return false;
  }
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width_));
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height_));
  cap_.set(cv::CAP_PROP_FPS, static_cast<double>(fps_));
  return true;
}

bool CameraManager::start() {
  if (running_.load()) {
    return true;
  }
  if (!openCapture()) {
    std::cerr << "[Camera] Failed to open source: " << source_ << std::endl;
    return false;
  }
  running_.store(true);
  capture_thread_ = std::thread(&CameraManager::captureLoop, this);
  return true;
}

void CameraManager::stop() {
  running_.store(false);
  frame_cv_.notify_all();
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  if (cap_.isOpened()) {
    cap_.release();
  }
}

bool CameraManager::getLatestFrame(FramePacket& out, std::uint32_t timeout_ms) {
  std::unique_lock<std::mutex> lock(frame_mutex_);
  const bool got = frame_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return has_frame_ || !running_.load(); });
  if (!got || !has_frame_) {
    return false;
  }
  out = latest_frame_;
  return true;
}

void CameraManager::captureLoop() {
  while (running_.load()) {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    FramePacket pkt{};
    pkt.bgr = frame;
    pkt.frame_id = ++frame_counter_;
    const auto now = std::chrono::system_clock::now();
    pkt.ts_ms =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_frame_ = std::move(pkt);
      has_frame_ = true;
    }
    frame_cv_.notify_one();
  }
}

}  // namespace asdun

