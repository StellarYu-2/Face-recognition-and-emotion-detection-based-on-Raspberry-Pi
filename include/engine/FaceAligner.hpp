#pragma once

#include <opencv2/core.hpp>

#include "core/Types.hpp"

namespace asdun {

class FaceAligner {
 public:
  explicit FaceAligner(int output_size = 112);

  void setOutputSize(int output_size);
  int outputSize() const { return output_size_; }

  cv::Mat align(const cv::Mat& frame_bgr, const FivePointLandmarks& landmarks) const;
  cv::Rect refineBox(const FivePointLandmarks& landmarks, const cv::Size& image_size) const;

 private:
  int output_size_{112};
};

}  // namespace asdun
