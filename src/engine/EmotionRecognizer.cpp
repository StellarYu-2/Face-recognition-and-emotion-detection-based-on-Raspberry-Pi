#include "engine/EmotionRecognizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

float clampPct(float v) {
  return std::clamp(v, 0.0F, 99.0F);
}

float edgeDensity(const cv::Mat& gray) {
  cv::Mat edges{};
  cv::Canny(gray, edges, 60.0, 140.0);
  return static_cast<float>(cv::countNonZero(edges)) / static_cast<float>(edges.total());
}

float darkPixelRatio(const cv::Mat& gray, std::uint8_t threshold) {
  cv::Mat mask{};
  cv::threshold(gray, mask, static_cast<double>(threshold), 255.0, cv::THRESH_BINARY_INV);
  return static_cast<float>(cv::countNonZero(mask)) / static_cast<float>(mask.total());
}

#ifdef USE_NCNN
int matElementCount(const ncnn::Mat& mat) {
  if (mat.dims == 1) {
    return mat.w;
  }
  if (mat.dims == 2) {
    return mat.w * mat.h;
  }
  if (mat.dims == 3) {
    return mat.w * mat.h * mat.c;
  }
  return 0;
}
#endif

EmotionLabel indexToEmotion(int idx) {
  switch (idx) {
    case 0:
      return EmotionLabel::Neutral;
    case 1:
      return EmotionLabel::Happy;
    case 2:
      return EmotionLabel::Surprise;
    case 3:
      return EmotionLabel::Sad;
    case 4:
      return EmotionLabel::Angry;
    case 5:
      return EmotionLabel::Disgust;
    case 6:
      return EmotionLabel::Fear;
    case 7:
      return EmotionLabel::Contempt;
    default:
      return EmotionLabel::Unknown;
  }
}

}  // namespace

bool EmotionRecognizer::init(const std::string& model_param_path,
                             const std::string& model_bin_path,
                             int input_size,
                             std::string input_blob_name,
                             std::string output_blob_name) {
  model_param_path_ = model_param_path;
  model_bin_path_ = model_bin_path;
  input_size_ = (input_size > 0) ? input_size : 64;
  input_blob_name_ = std::move(input_blob_name);
  output_blob_name_ = std::move(output_blob_name);
  ncnn_ready_ = false;

  std::array<std::string, 2> smile_paths = {
      "/usr/share/opencv4/haarcascades/haarcascade_smile.xml",
      "/usr/share/opencv/haarcascades/haarcascade_smile.xml"};

  for (const auto& p : smile_paths) {
    if (smile_cascade_.load(p)) {
      smile_ready_ = true;
      break;
    }
  }

#ifdef USE_NCNN
  if (std::filesystem::exists(model_param_path_) && std::filesystem::exists(model_bin_path_)) {
    net_.clear();
    net_.opt.use_vulkan_compute = false;
    net_.opt.lightmode = true;
    net_.opt.num_threads = 4;
    if (net_.load_param(model_param_path_.c_str()) == 0 && net_.load_model(model_bin_path_.c_str()) == 0) {
      ncnn_ready_ = true;
    }
  }
#endif

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

#ifdef USE_NCNN
  if (ncnn_ready_) {
    return inferWithNcnn(face_bgr);
  }
#endif

  const cv::Mat normalized = normalizeFace(face_bgr);
  const int width = normalized.cols;
  const int height = normalized.rows;

  const cv::Rect upper_roi(0, 0, width, height / 2);
  const cv::Rect lower_roi(0, height / 2, width, height - height / 2);
  const cv::Rect mouth_roi(width / 4, (height * 5) / 8, width / 2, height / 4);

  const cv::Mat upper = normalized(upper_roi);
  const cv::Mat lower = normalized(lower_roi);
  const cv::Mat mouth = normalized(mouth_roi);

  cv::Scalar mean{}, stddev{};
  cv::meanStdDev(normalized, mean, stddev);

  const float global_mean = static_cast<float>(mean[0]);
  const float global_contrast = static_cast<float>(stddev[0]);
  const float upper_edges = edgeDensity(upper);
  const float lower_edges = edgeDensity(lower);
  const float mouth_dark = darkPixelRatio(mouth, 55);

  float smile_strength = 0.0F;
  const bool has_smile = detectSmile(normalized, &smile_strength);

  if (has_smile) {
    out.label = EmotionLabel::Happy;
    out.conf_pct = clampPct(72.0F + 20.0F * smile_strength);
    return out;
  }

  if (mouth_dark > 0.16F && lower_edges > upper_edges * 1.18F && global_contrast > 42.0F) {
    out.label = EmotionLabel::Surprise;
    out.conf_pct = clampPct(60.0F + mouth_dark * 120.0F);
    return out;
  }

  if (global_mean < 100.0F && global_contrast < 36.0F && lower_edges < 0.11F) {
    out.label = EmotionLabel::Sad;
    out.conf_pct = clampPct(55.0F + (100.0F - global_mean) * 0.25F);
    return out;
  }

  if (global_contrast > 55.0F && upper_edges > 0.14F && mouth_dark < 0.10F) {
    out.label = EmotionLabel::Angry;
    out.conf_pct = clampPct(58.0F + upper_edges * 150.0F);
    return out;
  }

  out.label = EmotionLabel::Neutral;
  out.conf_pct = clampPct(58.0F + std::max(0.0F, 30.0F - std::abs(global_mean - 118.0F) * 0.35F));
  return out;
}

EmotionResult EmotionRecognizer::inferWithNcnn(const cv::Mat& face_bgr) const {
  EmotionResult out{};
#ifdef USE_NCNN
  const cv::Mat normalized = normalizeFace(face_bgr);
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(normalized.data,
                                               ncnn::Mat::PIXEL_GRAY,
                                               normalized.cols,
                                               normalized.rows,
                                               input_size_,
                                               input_size_);

  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  ex.input(input_blob_name_.c_str(), in);

  ncnn::Mat prob{};
  int rc = ex.extract(output_blob_name_.c_str(), prob);
  if (rc != 0) {
    rc = ex.extract(0, prob);
  }
  if (rc != 0) {
    out.label = EmotionLabel::Unknown;
    out.conf_pct = 0.0F;
    return out;
  }

  const int total = matElementCount(prob);
  if (total <= 0) {
    out.label = EmotionLabel::Unknown;
    out.conf_pct = 0.0F;
    return out;
  }

  const float* ptr = static_cast<const float*>(prob.data);
  int best_idx = 0;
  float best_score = ptr[0];
  float sum_exp = 0.0F;
  std::vector<float> logits(total, 0.0F);
  for (int i = 0; i < total; ++i) {
    logits[static_cast<std::size_t>(i)] = ptr[i];
    if (ptr[i] > best_score) {
      best_score = ptr[i];
      best_idx = i;
    }
  }

  for (const float x : logits) {
    sum_exp += std::exp(x - best_score);
  }
  const float prob_pct = (sum_exp > 1e-6F) ? (100.0F / sum_exp) : 0.0F;
  out.label = indexToEmotion(best_idx);
  out.conf_pct = clampPct(prob_pct);
  return out;
#else
  (void)face_bgr;
  out.label = EmotionLabel::Unknown;
  out.conf_pct = 0.0F;
  return out;
#endif
}

cv::Mat EmotionRecognizer::normalizeFace(const cv::Mat& face_bgr) {
  cv::Mat gray{};
  cv::cvtColor(face_bgr, gray, cv::COLOR_BGR2GRAY);

  cv::Mat resized{};
  cv::resize(gray, resized, cv::Size(128, 128), 0.0, 0.0, cv::INTER_LINEAR);

  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat equalized{};
  clahe->apply(resized, equalized);
  cv::GaussianBlur(equalized, equalized, cv::Size(3, 3), 0.0);
  return equalized;
}

bool EmotionRecognizer::detectSmile(const cv::Mat& normalized_gray, float* smile_strength) const {
  if (smile_strength != nullptr) {
    *smile_strength = 0.0F;
  }
  if (!smile_ready_) {
    return false;
  }

  const cv::Rect lower_half(0, normalized_gray.rows / 2, normalized_gray.cols, normalized_gray.rows / 2);
  cv::Mat lower = normalized_gray(lower_half);

  std::vector<cv::Rect> smiles;
  smile_cascade_.detectMultiScale(lower, smiles, 1.7, 18, 0, cv::Size(24, 16));
  if (smiles.empty()) {
    return false;
  }

  const auto best = *std::max_element(smiles.begin(), smiles.end(), [](const cv::Rect& a, const cv::Rect& b) {
    return a.area() < b.area();
  });
  const float lower_area = static_cast<float>(lower.rows * lower.cols);
  const float ratio = (lower_area > 1e-6F) ? (static_cast<float>(best.area()) / lower_area) : 0.0F;
  if (smile_strength != nullptr) {
    *smile_strength = std::clamp(ratio * 12.0F, 0.0F, 1.0F);
  }
  return true;
}

}  // namespace asdun
