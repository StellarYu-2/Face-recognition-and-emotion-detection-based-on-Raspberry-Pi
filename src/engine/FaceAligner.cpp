#include "engine/FaceAligner.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

const std::array<cv::Point2f, 5> kArcFaceReference112 = {
    cv::Point2f(38.2946F, 51.6963F),
    cv::Point2f(73.5318F, 51.5014F),
    cv::Point2f(56.0252F, 71.7366F),
    cv::Point2f(41.5493F, 92.3655F),
    cv::Point2f(70.7299F, 92.2041F)};

cv::Rect clampRect(const cv::Rect& rect, const cv::Size& image_size) {
  return rect & cv::Rect(0, 0, image_size.width, image_size.height);
}

}  // namespace

FaceAligner::FaceAligner(int output_size) { setOutputSize(output_size); }

void FaceAligner::setOutputSize(int output_size) { output_size_ = output_size > 0 ? output_size : 112; }

cv::Mat FaceAligner::align(const cv::Mat& frame_bgr, const FivePointLandmarks& landmarks) const {
  if (frame_bgr.empty() || !landmarks.valid) {
    return {};
  }

  std::vector<cv::Point2f> src_points(landmarks.points.begin(), landmarks.points.end());
  std::vector<cv::Point2f> dst_points;
  dst_points.reserve(kArcFaceReference112.size());
  const float scale = static_cast<float>(output_size_) / 112.0F;
  for (const auto& pt : kArcFaceReference112) {
    dst_points.emplace_back(pt.x * scale, pt.y * scale);
  }

  cv::Mat inliers{};
  const cv::Mat affine = cv::estimateAffinePartial2D(src_points, dst_points, inliers, cv::LMEDS);
  if (affine.empty()) {
    return {};
  }

  cv::Mat aligned{};
  cv::warpAffine(frame_bgr,
                 aligned,
                 affine,
                 cv::Size(output_size_, output_size_),
                 cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  return aligned;
}

cv::Rect FaceAligner::refineBox(const FivePointLandmarks& landmarks, const cv::Size& image_size) const {
  if (!landmarks.valid || image_size.width <= 0 || image_size.height <= 0) {
    return {};
  }

  const cv::Rect landmarks_bounds =
      cv::boundingRect(std::vector<cv::Point2f>(landmarks.points.begin(), landmarks.points.end()));
  if (landmarks_bounds.width <= 0 || landmarks_bounds.height <= 0) {
    return {};
  }

  const auto& left_eye = landmarks.points[0];
  const auto& right_eye = landmarks.points[1];
  const float eye_center_y = (left_eye.y + right_eye.y) * 0.5F;
  const auto& nose = landmarks.points[2];
  const auto& left_mouth = landmarks.points[3];
  const auto& right_mouth = landmarks.points[4];
  const float eye_distance = cv::norm(right_eye - left_eye);
  const float mouth_distance = cv::norm(right_mouth - left_mouth);
  const float width = std::max({static_cast<float>(landmarks_bounds.width) * 2.35F,
                                eye_distance * 2.70F,
                                mouth_distance * 2.40F,
                                static_cast<float>(landmarks_bounds.height) * 2.60F});
  const float height = width * 1.24F;
  const float cx = nose.x;
  const float cy = eye_center_y + height * 0.22F;

  cv::Rect refined(static_cast<int>(std::round(cx - width * 0.5F)),
                   static_cast<int>(std::round(cy - height * 0.46F)),
                   static_cast<int>(std::round(width)),
                   static_cast<int>(std::round(height)));
  return clampRect(refined, image_size);
}

}  // namespace asdun
