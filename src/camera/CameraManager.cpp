#include "camera/CameraManager.hpp"

#include <chrono>
#include <iostream>
#include <vector>

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

bool configureAndProbe(cv::VideoCapture& cap, int width, int height, int fps) {
  if (!cap.isOpened()) {
    return false;
  }

  if (width > 0) {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width));
  }
  if (height > 0) {
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));
  }
  if (fps > 0) {
    cap.set(cv::CAP_PROP_FPS, static_cast<double>(fps));
  }
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1.0);

  cv::Mat probe;
  for (int i = 0; i < 8; ++i) {
    if (cap.read(probe) && !probe.empty()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  return false;
}

}  // namespace

CameraManager::CameraManager(std::string source, int width, int height, int fps)
    : source_(std::move(source)), width_(width), height_(height), fps_(fps) {}

CameraManager::~CameraManager() { stop(); }

bool CameraManager::openCapture() {
  if (cap_.isOpened()) {
    cap_.release();
  }

  auto tryOpenDevice = [this](int index, int backend, int width, int height) {
    if (cap_.isOpened()) {
      cap_.release();
    }
    if (!cap_.open(index, backend)) {
      return false;
    }
    if (!configureAndProbe(cap_, width, height, fps_)) {
      cap_.release();
      return false;
    }
    width_ = width;
    height_ = height;
    return true;
  };

  if (source_.rfind("gst:", 0) == 0) {
    const std::string pipeline = source_.substr(4);
    if (!cap_.open(pipeline, cv::CAP_GSTREAMER)) {
      return false;
    }
    return configureAndProbe(cap_, width_, height_, fps_);
  } else if (isDigitsOnly(source_)) {
    const int index = std::stoi(source_);
    std::vector<std::pair<int, int>> sizes;
    sizes.emplace_back(width_, height_);
    if (width_ != 640 || height_ != 480) {
      sizes.emplace_back(640, 480);
    }
    if (width_ != 320 || height_ != 240) {
      sizes.emplace_back(320, 240);
    }

    for (const int backend : {cv::CAP_V4L2, cv::CAP_ANY}) {
      for (const auto& size : sizes) {
        if (tryOpenDevice(index, backend, size.first, size.second)) {
          if (size.first != sizes.front().first || size.second != sizes.front().second) {
            std::cerr << "[Camera] Fallback to " << size.first << "x" << size.second << " for source " << source_
                      << std::endl;
          }
          return true;
        }
      }
    }
    return false;
  } else {
    if (!cap_.open(source_, cv::CAP_ANY)) {
      return false;
    }
    return configureAndProbe(cap_, width_, height_, fps_);
  }
}

bool CameraManager::start() {
  if (running_.load()) {
    return true;
  }
  if (!openCapture()) {
    std::cerr << "[Camera] Failed to open source: " << source_ << std::endl;
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    has_frame_ = false;
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
