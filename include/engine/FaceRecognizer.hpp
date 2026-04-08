#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace asdun {

class FaceRecognizer {
 public:
  std::vector<float> extractEmbedding(const cv::Mat& face_bgr) const;

  static float l2Distance(const std::vector<float>& a, const std::vector<float>& b);

 private:
  static void l2Normalize(std::vector<float>& v);
};

}  // namespace asdun

