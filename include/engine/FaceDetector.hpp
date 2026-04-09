#pragma once

#include <string>
#include <vector>

#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#ifdef USE_NCNN
#include <net.h>
#endif

#include "core/Types.hpp"

namespace asdun {

class FaceDetector {
 public:
  FaceDetector(std::string cascade_path = "",
               std::string model_param_path = "",
               std::string model_bin_path = "",
               int input_width = 320,
               int input_height = 240,
               float score_threshold = 0.7F,
               float nms_threshold = 0.3F,
               std::string input_blob_name = "input",
               std::string score_blob_name = "scores",
               std::string bbox_blob_name = "boxes");
  bool init();
  std::vector<Detection> detect(const cv::Mat& frame_bgr) const;

 private:
  struct PriorBox {
    float center_x{0.0F};
    float center_y{0.0F};
    float width{0.0F};
    float height{0.0F};
  };

  std::vector<Detection> detectWithNcnn(const cv::Mat& frame_bgr) const;
  std::vector<PriorBox> generatePriors() const;
  static std::vector<Detection> nms(const std::vector<Detection>& detections, float nms_threshold);
  void detectWithCascade(cv::CascadeClassifier& cascade,
                         const cv::Mat& gray,
                         bool mirror,
                         std::vector<Detection>& out) const;
  void detectWithRotation(cv::CascadeClassifier& cascade,
                          const cv::Mat& gray,
                          double angle_deg,
                          std::vector<Detection>& out) const;
  static std::vector<Detection> suppressDuplicates(const std::vector<Detection>& detections);
  static cv::Point2f applyAffine(const cv::Mat& affine, const cv::Point2f& pt);
  static float iou(const cv::Rect& a, const cv::Rect& b);

  std::string cascade_path_{};
  std::string model_param_path_{};
  std::string model_bin_path_{};
  int input_width_{320};
  int input_height_{240};
  float score_threshold_{0.7F};
  float nms_threshold_{0.3F};
  std::string input_blob_name_{"input"};
  std::string score_blob_name_{"scores"};
  std::string bbox_blob_name_{"boxes"};
  std::string profile_cascade_path_{};
  mutable cv::CascadeClassifier frontal_cascade_{};
  mutable cv::CascadeClassifier profile_cascade_{};
  std::vector<PriorBox> priors_cache_{};
  bool ncnn_ready_{false};
  bool initialized_{false};
  bool profile_initialized_{false};

#ifdef USE_NCNN
  ncnn::Net net_{};
#endif
};

}  // namespace asdun
