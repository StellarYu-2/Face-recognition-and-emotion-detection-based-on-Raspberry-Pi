#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include "core/Types.hpp"

namespace asdun {

class EmotionRecognizer {
 public:
  EmotionRecognizer() = default;
  bool init(const std::string& model_param_path = "", const std::string& model_bin_path = "");
  EmotionResult infer(const cv::Mat& face_bgr) const;

 private:
  static cv::Mat normalizeFace(const cv::Mat& face_bgr);
  bool detectSmile(const cv::Mat& normalized_gray, float* smile_strength) const;

  bool initialized_{false};
  bool smile_ready_{false};
  mutable cv::CascadeClassifier smile_cascade_{};
};

}  // namespace asdun
