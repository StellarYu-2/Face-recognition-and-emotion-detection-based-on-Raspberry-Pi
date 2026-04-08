#include "engine/FaceRecognizer.hpp"

#include <array>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

cv::Mat rotateFace(const cv::Mat& face_bgr, double angle_deg) {
  const cv::Point2f center(static_cast<float>(face_bgr.cols) * 0.5F, static_cast<float>(face_bgr.rows) * 0.5F);
  const cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);

  cv::Mat rotated{};
  cv::warpAffine(face_bgr, rotated, rot, face_bgr.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  return rotated;
}

std::vector<float> buildSingleEmbedding(const cv::Mat& face_bgr) {
  cv::Mat gray{};
  cv::cvtColor(face_bgr, gray, cv::COLOR_BGR2GRAY);

  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat enhanced{};
  clahe->apply(gray, enhanced);

  cv::Mat resized{};
  cv::resize(enhanced, resized, cv::Size(16, 8), 0.0, 0.0, cv::INTER_LINEAR);
  resized.convertTo(resized, CV_32F, 1.0 / 255.0);

  std::vector<float> embedding;
  embedding.reserve(static_cast<std::size_t>(resized.rows * resized.cols));
  float sum = 0.0F;
  for (int r = 0; r < resized.rows; ++r) {
    for (int c = 0; c < resized.cols; ++c) {
      const float v = resized.at<float>(r, c);
      embedding.push_back(v);
      sum += v;
    }
  }

  const float mean = sum / static_cast<float>(embedding.size());
  for (float& v : embedding) {
    v -= mean;
  }
  return embedding;
}

}  // namespace

std::vector<float> FaceRecognizer::extractEmbedding(const cv::Mat& face_bgr) const {
  if (face_bgr.empty()) {
    return {};
  }

  // 通过原图和轻度旋转图像求平均特征，提升对小角度歪头的鲁棒性。
  const std::array<double, 3> angles = {-12.0, 0.0, 12.0};
  std::vector<float> embedding;
  int valid_views = 0;

  for (const double angle : angles) {
    const cv::Mat view = (std::abs(angle) < 1e-3) ? face_bgr : rotateFace(face_bgr, angle);
    auto single = buildSingleEmbedding(view);
    if (single.empty()) {
      continue;
    }

    if (embedding.empty()) {
      embedding.assign(single.size(), 0.0F);
    }
    for (std::size_t i = 0; i < single.size(); ++i) {
      embedding[i] += single[i];
    }
    valid_views++;
  }

  if (valid_views <= 0) {
    return {};
  }
  for (float& v : embedding) {
    v /= static_cast<float>(valid_views);
  }
  l2Normalize(embedding);
  return embedding;
}

float FaceRecognizer::l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return std::numeric_limits<float>::max();
  }
  float sum = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

void FaceRecognizer::l2Normalize(std::vector<float>& v) {
  float norm2 = 0.0F;
  for (const float x : v) {
    norm2 += x * x;
  }
  const float norm = std::sqrt(norm2);
  if (norm <= 1e-6F) {
    return;
  }
  for (float& x : v) {
    x /= norm;
  }
}

}  // namespace asdun

