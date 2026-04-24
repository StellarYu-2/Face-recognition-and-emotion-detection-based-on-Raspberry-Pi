#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#ifdef USE_NCNN
#include <net.h>
#endif

#include "core/Types.hpp"

namespace asdun {

class EmotionRecognizer {
 public:
  EmotionRecognizer() = default;
  bool init(const std::string& model_param_path = "",
            const std::string& model_bin_path = "",
            int input_size = 64,
            std::string input_blob_name = "input",
            std::string output_blob_name = "output");
  void setDecisionPolicy(float non_calm_floor, float handoff_margin);
  EmotionResult infer(const cv::Mat& face_bgr) const;

 private:
  static cv::Mat normalizeFace(const cv::Mat& face_bgr);
  EmotionResult inferWithHeuristics(const cv::Mat& face_bgr) const;
  EmotionResult inferWithNcnn(const cv::Mat& face_bgr) const;
  bool detectSmile(const cv::Mat& normalized_gray, float* smile_strength) const;

  bool initialized_{false};
  bool smile_ready_{false};
  bool ncnn_ready_{false};
  int input_size_{64};
  float non_calm_floor_{0.22F};
  float handoff_margin_{0.08F};
  std::string model_param_path_{};
  std::string model_bin_path_{};
  std::string input_blob_name_{"input"};
  std::string output_blob_name_{"output"};
  mutable cv::CascadeClassifier smile_cascade_{};

#ifdef USE_NCNN
  ncnn::Net net_{};
#endif
};

}  // namespace asdun
