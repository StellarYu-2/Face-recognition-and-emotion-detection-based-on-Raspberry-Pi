#include "engine/EmotionRecognizer.hpp"

#include <opencv2/imgproc.hpp>

namespace asdun {

bool EmotionRecognizer::init(const std::string& /*model_param_path*/, const std::string& /*model_bin_path*/) {
  // 当前提供可运行占位逻辑，后续可在此处接入 NCNN 情绪模型。
  initialized_ = true;
  return true;
}

EmotionResult EmotionRecognizer::infer(const cv::Mat& face_bgr) const {
  EmotionResult out{};
  if (!initialized_ || face_bgr.empty()) {
    out.label = EmotionLabel::Unknown;
    out.conf_pct = 0.0F;
    return out;
  }

  cv::Mat gray{};
  cv::cvtColor(face_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Scalar mean{}, stddev{};
  cv::meanStdDev(gray, mean, stddev);

  const float mean_light = static_cast<float>(mean[0]);
  const float contrast = static_cast<float>(stddev[0]);

  // 轻量级启发式情绪占位输出，便于先跑通系统与 UI 流程。
  if (mean_light > 150.0F && contrast > 45.0F) {
    out.label = EmotionLabel::Happy;
    out.conf_pct = 72.0F;
  } else if (mean_light < 80.0F) {
    out.label = EmotionLabel::Sad;
    out.conf_pct = 68.0F;
  } else if (contrast > 60.0F) {
    out.label = EmotionLabel::Surprise;
    out.conf_pct = 61.0F;
  } else {
    out.label = EmotionLabel::Neutral;
    out.conf_pct = 65.0F;
  }
  return out;
}

}  // namespace asdun

