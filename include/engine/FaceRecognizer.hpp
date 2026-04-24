#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#ifdef USE_NCNN
#include <net.h>
#endif

namespace asdun {

class FaceRecognizer {
 public:
  FaceRecognizer() = default;
  bool init(const std::string& model_param_path,
            const std::string& model_bin_path,
            int input_size,
            std::string input_blob_name = "input",
            std::string output_blob_name = "fc1",
            bool use_rgb_input = true);

  std::vector<float> extractEmbedding(const cv::Mat& face_bgr) const;
  const std::string& modelTag() const { return model_tag_; }

  static float l2Distance(const std::vector<float>& a, const std::vector<float>& b);

 private:
  std::vector<float> extractEmbeddingSingle(const cv::Mat& face_bgr) const;
  static cv::Mat squarePad(const cv::Mat& face_bgr);
  static cv::Mat mirrorFace(const cv::Mat& face_bgr);
  static void l2Normalize(std::vector<float>& v);

  bool ncnn_ready_{false};
  int input_size_{112};
  std::string model_param_path_{};
  std::string model_bin_path_{};
  std::string input_blob_name_{"input"};
  std::string output_blob_name_{"fc1"};
  std::string model_tag_{};
  bool use_rgb_input_{true};

#ifdef USE_NCNN
  ncnn::Net net_{};
#endif
};

}  // namespace asdun
