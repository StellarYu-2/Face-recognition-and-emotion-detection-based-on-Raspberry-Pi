#include "engine/FaceRecognizer.hpp"

#include <array>
#include <cmath>
#include <filesystem>
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

std::vector<float> buildFallbackEmbedding(const cv::Mat& face_bgr) {
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

}  // namespace

bool FaceRecognizer::init(const std::string& model_param_path,
                          const std::string& model_bin_path,
                          int input_size,
                          std::string input_blob_name,
                          std::string output_blob_name,
                          bool use_rgb_input) {
  model_param_path_ = model_param_path;
  model_bin_path_ = model_bin_path;
  input_size_ = (input_size > 0) ? input_size : 112;
  input_blob_name_ = std::move(input_blob_name);
  output_blob_name_ = std::move(output_blob_name);
  use_rgb_input_ = use_rgb_input;
  model_tag_ = std::filesystem::path(model_param_path_).stem().string();
  if (model_tag_.empty()) {
    model_tag_ = "recognizer_default";
  }
  ncnn_ready_ = false;

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

  return true;
}

std::vector<float> FaceRecognizer::extractEmbedding(const cv::Mat& face_bgr) const {
  if (face_bgr.empty()) {
    return {};
  }

#ifdef USE_NCNN
  if (ncnn_ready_) {
    const cv::Mat square = squarePad(face_bgr);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(square.data,
                                                 use_rgb_input_ ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR,
                                                 square.cols,
                                                 square.rows,
                                                 input_size_,
                                                 input_size_);

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    ex.input(input_blob_name_.c_str(), in);

    ncnn::Mat out{};
    int rc = ex.extract(output_blob_name_.c_str(), out);
    if (rc != 0) {
      rc = ex.extract(0, out);
    }
    if (rc == 0) {
      const int total = matElementCount(out);
      if (total > 0) {
        const float* ptr = static_cast<const float*>(out.data);
        std::vector<float> embedding(ptr, ptr + total);
        l2Normalize(embedding);
        return embedding;
      }
    }
  }
#endif

  // 模型不可用时回退到占位特征，但保留轻度旋转增强，避免现有功能直接失效。
  const std::array<double, 3> angles = {-12.0, 0.0, 12.0};
  std::vector<float> embedding;
  int valid_views = 0;

  for (const double angle : angles) {
    const cv::Mat view = (std::abs(angle) < 1e-3) ? face_bgr : rotateFace(face_bgr, angle);
    auto single = buildFallbackEmbedding(view);
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

cv::Mat FaceRecognizer::squarePad(const cv::Mat& face_bgr) {
  const int size = std::max(face_bgr.cols, face_bgr.rows);
  cv::Mat square(size, size, face_bgr.type(), cv::Scalar(0, 0, 0));
  const int x = (size - face_bgr.cols) / 2;
  const int y = (size - face_bgr.rows) / 2;
  face_bgr.copyTo(square(cv::Rect(x, y, face_bgr.cols, face_bgr.rows)));
  return square;
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
