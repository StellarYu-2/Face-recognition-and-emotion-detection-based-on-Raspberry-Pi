#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace asdun {

struct QualityResult {
  bool valid{false};
  bool large_enough{false};
  bool sharp_enough{false};
  float area_ratio{0.0F};
  float blur_score{0.0F};
  std::string reason{"unknown"};
};

class FaceQualityGate {
 public:
  FaceQualityGate(float min_area_ratio, float blur_threshold, int stable_frames_required);

  QualityResult evaluate(const cv::Mat& frame_bgr, const cv::Rect& face_box);

 private:
  float min_area_ratio_{0.08F};
  float blur_threshold_{100.0F};
  int stable_frames_required_{3};
  int stable_counter_{0};
};

}  // namespace asdun

