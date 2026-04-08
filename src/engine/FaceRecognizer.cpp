#include "engine/FaceRecognizer.hpp"

#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace asdun {

std::vector<float> FaceRecognizer::extractEmbedding(const cv::Mat& face_bgr) const {
  if (face_bgr.empty()) {
    return {};
  }

  cv::Mat gray{};
  cv::cvtColor(face_bgr, gray, cv::COLOR_BGR2GRAY);

  // 这里使用轻量级占位特征（16x8=128维），后续可直接替换成 NCNN 正式 embedding 模型输出。
  cv::Mat resized{};
  cv::resize(gray, resized, cv::Size(16, 8), 0.0, 0.0, cv::INTER_LINEAR);
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

