#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "core/Types.hpp"

namespace asdun {

class EmotionRecognizer {
 public:
  EmotionRecognizer() = default;
  bool init(const std::string& model_param_path = "", const std::string& model_bin_path = "");
  EmotionResult infer(const cv::Mat& face_bgr) const;

 private:
  bool initialized_{false};
};

}  // namespace asdun

