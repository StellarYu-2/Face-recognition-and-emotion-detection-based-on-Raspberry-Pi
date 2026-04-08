#include "engine/FaceDetector.hpp"

#include <algorithm>
#include <array>

#include <opencv2/imgproc.hpp>

namespace asdun {

FaceDetector::FaceDetector(std::string cascade_path) : cascade_path_(std::move(cascade_path)) {}

bool FaceDetector::init() {
  std::array<std::string, 3> frontal_paths = {
      cascade_path_,
      "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"};
  std::array<std::string, 2> profile_paths = {
      "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
      "/usr/share/opencv/haarcascades/haarcascade_profileface.xml"};

  for (const auto& p : frontal_paths) {
    if (p.empty()) {
      continue;
    }
    if (frontal_cascade_.load(p)) {
      initialized_ = true;
      break;
    }
  }

  for (const auto& p : profile_paths) {
    if (p.empty()) {
      continue;
    }
    if (profile_cascade_.load(p)) {
      profile_initialized_ = true;
      profile_cascade_path_ = p;
      break;
    }
  }

  return initialized_;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame_bgr) const {
  std::vector<Detection> out;
  if (!initialized_ || frame_bgr.empty()) {
    return out;
  }

  cv::Mat gray{};
  cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  detectWithCascade(frontal_cascade_, gray, false, out);
  if (profile_initialized_) {
    detectWithCascade(profile_cascade_, gray, false, out);
    detectWithCascade(profile_cascade_, gray, true, out);
  }

  // 只有在标准角度没有命中时，才回退到轻度旋转检测，避免 CPU 压力持续飙升。
  if (out.empty()) {
    detectWithRotation(frontal_cascade_, gray, -18.0, out);
    detectWithRotation(frontal_cascade_, gray, 18.0, out);
  }

  return suppressDuplicates(out);
}

void FaceDetector::detectWithCascade(cv::CascadeClassifier& cascade,
                                     const cv::Mat& gray,
                                     bool mirror,
                                     std::vector<Detection>& out) const {
  cv::Mat work = gray;
  if (mirror) {
    cv::flip(gray, work, 1);
  }

  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(work, faces, 1.1, 4, 0, cv::Size(72, 72));
  for (auto rect : faces) {
    if (mirror) {
      rect.x = gray.cols - rect.x - rect.width;
    }
    rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (rect.width > 0 && rect.height > 0) {
      out.push_back(Detection{rect, 1.0F});
    }
  }
}

void FaceDetector::detectWithRotation(cv::CascadeClassifier& cascade,
                                      const cv::Mat& gray,
                                      double angle_deg,
                                      std::vector<Detection>& out) const {
  const cv::Point2f center(static_cast<float>(gray.cols) * 0.5F, static_cast<float>(gray.rows) * 0.5F);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);
  const cv::Rect2f bbox = cv::RotatedRect(center, gray.size(), angle_deg).boundingRect2f();
  rot.at<double>(0, 2) += bbox.width * 0.5 - center.x;
  rot.at<double>(1, 2) += bbox.height * 0.5 - center.y;

  cv::Mat rotated{};
  cv::warpAffine(gray,
                 rotated,
                 rot,
                 cv::Size(cvRound(bbox.width), cvRound(bbox.height)),
                 cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT,
                 cv::Scalar(0));

  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(rotated, faces, 1.1, 4, 0, cv::Size(72, 72));
  if (faces.empty()) {
    return;
  }

  cv::Mat inv{};
  cv::invertAffineTransform(rot, inv);
  for (const auto& rect : faces) {
    const std::array<cv::Point2f, 4> corners = {
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height))};

    std::vector<cv::Point2f> mapped;
    mapped.reserve(corners.size());
    for (const auto& corner : corners) {
      mapped.push_back(applyAffine(inv, corner));
    }

    cv::Rect mapped_rect = cv::boundingRect(mapped);
    mapped_rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (mapped_rect.width > 0 && mapped_rect.height > 0) {
      out.push_back(Detection{mapped_rect, 0.95F});
    }
  }
}

std::vector<Detection> FaceDetector::suppressDuplicates(const std::vector<Detection>& detections) {
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b) {
    return a.box.area() > b.box.area();
  });

  std::vector<Detection> kept;
  kept.reserve(sorted.size());
  for (const auto& det : sorted) {
    bool overlapped = false;
    for (const auto& existing : kept) {
      if (iou(det.box, existing.box) > 0.35F) {
        overlapped = true;
        break;
      }
    }
    if (!overlapped) {
      kept.push_back(det);
    }
  }
  return kept;
}

cv::Point2f FaceDetector::applyAffine(const cv::Mat& affine, const cv::Point2f& pt) {
  const double x = affine.at<double>(0, 0) * pt.x + affine.at<double>(0, 1) * pt.y + affine.at<double>(0, 2);
  const double y = affine.at<double>(1, 0) * pt.x + affine.at<double>(1, 1) * pt.y + affine.at<double>(1, 2);
  return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

float FaceDetector::iou(const cv::Rect& a, const cv::Rect& b) {
  const int x_left = std::max(a.x, b.x);
  const int y_top = std::max(a.y, b.y);
  const int x_right = std::min(a.x + a.width, b.x + b.width);
  const int y_bottom = std::min(a.y + a.height, b.y + b.height);
  if (x_right <= x_left || y_bottom <= y_top) {
    return 0.0F;
  }

  const float inter = static_cast<float>((x_right - x_left) * (y_bottom - y_top));
  const float union_area = static_cast<float>(a.area() + b.area()) - inter;
  return (union_area > 1e-6F) ? (inter / union_area) : 0.0F;
}

}  // namespace asdun
