#pragma once

#include <string>

#include <opencv2/core.hpp>

#ifdef USE_NCNN
#include <net.h>
#endif

#include "core/Types.hpp"

namespace asdun {

class FaceLandmarkEstimator {
 public:
  enum class CoordMode {
    ZeroOne = 0,
    MinusOneOne,
    Pixel
  };

  FaceLandmarkEstimator() = default;

  bool init(const std::string& model_param_path,
            const std::string& model_bin_path,
            int input_width,
            int input_height,
            std::string input_blob_name = "input",
            std::string output_blob_name = "output",
            float crop_scale = 1.45F,
            CoordMode coord_mode = CoordMode::ZeroOne,
            bool use_rgb_input = true,
            float input_mean = 127.5F,
            float input_norm = 1.0F / 128.0F);

  bool ready() const { return ncnn_ready_; }
  FivePointLandmarks estimate(const cv::Mat& frame_bgr, const cv::Rect& face_box) const;

  static CoordMode parseCoordMode(const std::string& value);

 private:
  static cv::Rect buildCropRect(const cv::Rect& face_box, const cv::Size& image_size, float crop_scale);
  static bool isReasonableLandmarkSet(const FivePointLandmarks& landmarks, const cv::Rect& face_box);

  std::string model_param_path_{};
  std::string model_bin_path_{};
  int input_width_{112};
  int input_height_{112};
  std::string input_blob_name_{"input"};
  std::string output_blob_name_{"output"};
  float crop_scale_{1.45F};
  CoordMode coord_mode_{CoordMode::ZeroOne};
  bool use_rgb_input_{true};
  float input_mean_{127.5F};
  float input_norm_{1.0F / 128.0F};
  bool ncnn_ready_{false};

#ifdef USE_NCNN
  ncnn::Net net_{};
#endif
};

}  // namespace asdun
