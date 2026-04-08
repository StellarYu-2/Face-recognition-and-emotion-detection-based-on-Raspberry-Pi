#pragma once

#include <string>
#include <vector>

#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include "core/Types.hpp"

namespace asdun {

class FaceDetector {
 public:
  explicit FaceDetector(std::string cascade_path = "");
  bool init();
  std::vector<Detection> detect(const cv::Mat& frame_bgr) const;

 private:
  std::string cascade_path_{};
  mutable cv::CascadeClassifier cascade_{};
  bool initialized_{false};
};

}  // namespace asdun

