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
  void detectWithCascade(cv::CascadeClassifier& cascade,
                         const cv::Mat& gray,
                         bool mirror,
                         std::vector<Detection>& out) const;
  void detectWithRotation(cv::CascadeClassifier& cascade,
                          const cv::Mat& gray,
                          double angle_deg,
                          std::vector<Detection>& out) const;
  static std::vector<Detection> suppressDuplicates(const std::vector<Detection>& detections);
  static cv::Point2f applyAffine(const cv::Mat& affine, const cv::Point2f& pt);
  static float iou(const cv::Rect& a, const cv::Rect& b);

  std::string cascade_path_{};
  std::string profile_cascade_path_{};
  mutable cv::CascadeClassifier frontal_cascade_{};
  mutable cv::CascadeClassifier profile_cascade_{};
  bool initialized_{false};
  bool profile_initialized_{false};
};

}  // namespace asdun
