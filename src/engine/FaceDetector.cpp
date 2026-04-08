#include "engine/FaceDetector.hpp"

#include <array>

#include <opencv2/imgproc.hpp>

namespace asdun {

FaceDetector::FaceDetector(std::string cascade_path) : cascade_path_(std::move(cascade_path)) {}

bool FaceDetector::init() {
  std::array<std::string, 3> candidate_paths = {
      cascade_path_,
      "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"};

  for (const auto& p : candidate_paths) {
    if (p.empty()) {
      continue;
    }
    if (cascade_.load(p)) {
      initialized_ = true;
      return true;
    }
  }
  initialized_ = false;
  return false;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame_bgr) const {
  std::vector<Detection> out;
  if (!initialized_ || frame_bgr.empty()) {
    return out;
  }

  cv::Mat gray{};
  cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  std::vector<cv::Rect> faces;
  cascade_.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(80, 80));
  out.reserve(faces.size());
  for (const auto& r : faces) {
    out.push_back(Detection{r, 1.0F});
  }
  return out;
}

}  // namespace asdun

