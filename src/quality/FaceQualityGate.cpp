#include "quality/FaceQualityGate.hpp"

#include <opencv2/imgproc.hpp>

namespace asdun {

FaceQualityGate::FaceQualityGate(float min_area_ratio, float blur_threshold, int stable_frames_required)
    : min_area_ratio_(min_area_ratio),
      blur_threshold_(blur_threshold),
      stable_frames_required_(stable_frames_required) {}

QualityResult FaceQualityGate::evaluate(const cv::Mat& frame_bgr, const cv::Rect& face_box) {
  QualityResult result{};
  if (frame_bgr.empty() || face_box.width <= 0 || face_box.height <= 0) {
    stable_counter_ = 0;
    result.reason = "no_face";
    return result;
  }

  const float frame_area = static_cast<float>(frame_bgr.cols * frame_bgr.rows);
  const float face_area = static_cast<float>(face_box.width * face_box.height);
  result.area_ratio = (frame_area > 0.0F) ? (face_area / frame_area) : 0.0F;
  result.large_enough = (result.area_ratio >= min_area_ratio_);

  const cv::Rect bounded = face_box & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
  cv::Mat face_roi = frame_bgr(bounded).clone();
  cv::Mat gray{};
  cv::cvtColor(face_roi, gray, cv::COLOR_BGR2GRAY);
  cv::Mat lap{};
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean{}, stddev{};
  cv::meanStdDev(lap, mean, stddev);
  result.blur_score = static_cast<float>(stddev[0] * stddev[0]);
  result.sharp_enough = (result.blur_score >= blur_threshold_);

  if (result.large_enough && result.sharp_enough) {
    stable_counter_++;
  } else {
    stable_counter_ = 0;
  }

  result.valid = (stable_counter_ >= stable_frames_required_);
  if (!result.large_enough) {
    result.reason = "face_too_small";
  } else if (!result.sharp_enough) {
    result.reason = "blur_too_high";
  } else if (!result.valid) {
    result.reason = "stabilizing";
  } else {
    result.reason = "ready";
  }
  return result;
}

}  // namespace asdun

