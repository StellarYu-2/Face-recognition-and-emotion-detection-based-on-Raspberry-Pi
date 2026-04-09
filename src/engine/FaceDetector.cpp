#include "engine/FaceDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

#ifdef USE_NCNN
int matRows(const ncnn::Mat& mat) {
  if (mat.dims == 1) {
    return 1;
  }
  if (mat.dims == 2) {
    return mat.h;
  }
  if (mat.dims == 3) {
    return mat.c;
  }
  return 0;
}

const float kCenterVariance = 0.1F;
const float kSizeVariance = 0.2F;
#endif

float centerDistanceRatio(const cv::Rect& a, const cv::Rect& b) {
  const float ax = static_cast<float>(a.x) + static_cast<float>(a.width) * 0.5F;
  const float ay = static_cast<float>(a.y) + static_cast<float>(a.height) * 0.5F;
  const float bx = static_cast<float>(b.x) + static_cast<float>(b.width) * 0.5F;
  const float by = static_cast<float>(b.y) + static_cast<float>(b.height) * 0.5F;
  const float dx = ax - bx;
  const float dy = ay - by;
  const float scale = static_cast<float>(std::max({a.width, a.height, b.width, b.height}));
  if (scale <= 1e-6F) {
    return std::numeric_limits<float>::max();
  }
  return std::sqrt(dx * dx + dy * dy) / scale;
}

float containmentRatio(const cv::Rect& a, const cv::Rect& b) {
  const int x_left = std::max(a.x, b.x);
  const int y_top = std::max(a.y, b.y);
  const int x_right = std::min(a.x + a.width, b.x + b.width);
  const int y_bottom = std::min(a.y + a.height, b.y + b.height);
  if (x_right <= x_left || y_bottom <= y_top) {
    return 0.0F;
  }
  const float inter = static_cast<float>((x_right - x_left) * (y_bottom - y_top));
  const float min_area = static_cast<float>(std::min(a.area(), b.area()));
  if (min_area <= 1e-6F) {
    return 0.0F;
  }
  return inter / min_area;
}

bool shouldMergeDuplicate(const cv::Rect& a, const cv::Rect& b, float iou_value, float iou_threshold) {
  if (iou_value > iou_threshold) {
    return true;
  }
  return containmentRatio(a, b) > 0.82F && centerDistanceRatio(a, b) < 0.18F;
}

}  // namespace

FaceDetector::FaceDetector(std::string cascade_path,
                           std::string model_param_path,
                           std::string model_bin_path,
                           int input_width,
                           int input_height,
                           float score_threshold,
                           float nms_threshold,
                           std::string input_blob_name,
                           std::string score_blob_name,
                           std::string bbox_blob_name)
    : cascade_path_(std::move(cascade_path)),
      model_param_path_(std::move(model_param_path)),
      model_bin_path_(std::move(model_bin_path)),
      input_width_(input_width > 0 ? input_width : 320),
      input_height_(input_height > 0 ? input_height : 240),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      input_blob_name_(std::move(input_blob_name)),
      score_blob_name_(std::move(score_blob_name)),
      bbox_blob_name_(std::move(bbox_blob_name)) {}

bool FaceDetector::init() {
#ifdef USE_NCNN
  if (std::filesystem::exists(model_param_path_) && std::filesystem::exists(model_bin_path_)) {
    net_.clear();
    net_.opt.use_vulkan_compute = false;
    net_.opt.lightmode = true;
    net_.opt.num_threads = 4;
    if (net_.load_param(model_param_path_.c_str()) == 0 && net_.load_model(model_bin_path_.c_str()) == 0) {
      priors_cache_ = generatePriors();
      ncnn_ready_ = true;
    }
  }
#endif

  std::array<std::string, 3> frontal_paths = {
      cascade_path_,
      "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"};
  std::array<std::string, 2> profile_paths = {
      "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
      "/usr/share/opencv/haarcascades/haarcascade_profileface.xml"};

  for (const auto& p : frontal_paths) {
    if (p.empty()) {
      continue;
    }
    if (frontal_cascade_.load(p)) {
      initialized_ = true;
      break;
    }
  }

  for (const auto& p : profile_paths) {
    if (p.empty()) {
      continue;
    }
    if (profile_cascade_.load(p)) {
      profile_initialized_ = true;
      profile_cascade_path_ = p;
      break;
    }
  }

  return ncnn_ready_ || initialized_;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame_bgr) const {
#ifdef USE_NCNN
  if (ncnn_ready_) {
    const auto dets = detectWithNcnn(frame_bgr);
    if (!dets.empty()) {
      return dets;
    }
  }
#endif

  std::vector<Detection> out;
  if (!initialized_ || frame_bgr.empty()) {
    return out;
  }

  cv::Mat gray{};
  cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  detectWithCascade(frontal_cascade_, gray, false, out);
  if (profile_initialized_) {
    detectWithCascade(profile_cascade_, gray, false, out);
    detectWithCascade(profile_cascade_, gray, true, out);
  }

  if (out.empty()) {
    detectWithRotation(frontal_cascade_, gray, -18.0, out);
    detectWithRotation(frontal_cascade_, gray, 18.0, out);
  }

  return suppressDuplicates(out);
}

std::vector<Detection> FaceDetector::detectWithNcnn(const cv::Mat& frame_bgr) const {
  std::vector<Detection> out;
#ifdef USE_NCNN
  if (!ncnn_ready_ || frame_bgr.empty()) {
    return out;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame_bgr.data,
                                               ncnn::Mat::PIXEL_BGR2RGB,
                                               frame_bgr.cols,
                                               frame_bgr.rows,
                                               input_width_,
                                               input_height_);
  const float mean_vals[3] = {127.0F, 127.0F, 127.0F};
  const float norm_vals[3] = {1.0F / 128.0F, 1.0F / 128.0F, 1.0F / 128.0F};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  ex.input(input_blob_name_.c_str(), in);

  ncnn::Mat scores{};
  ncnn::Mat boxes{};
  int rc_scores = ex.extract(score_blob_name_.c_str(), scores);
  if (rc_scores != 0) {
    rc_scores = ex.extract(0, scores);
  }
  int rc_boxes = ex.extract(bbox_blob_name_.c_str(), boxes);
  if (rc_boxes != 0) {
    rc_boxes = ex.extract(1, boxes);
  }
  if (rc_scores != 0 || rc_boxes != 0) {
    return out;
  }

  const int rows = std::min({matRows(scores), matRows(boxes), static_cast<int>(priors_cache_.size())});
  for (int i = 0; i < rows; ++i) {
    const float* score_ptr = scores.row(i);
    const float* box_ptr = boxes.row(i);
    const float score = (scores.w >= 2) ? score_ptr[1] : score_ptr[0];
    if (score < score_threshold_) {
      continue;
    }

    const auto& prior = priors_cache_[static_cast<std::size_t>(i)];
    const float x_center = box_ptr[0] * kCenterVariance * prior.width + prior.center_x;
    const float y_center = box_ptr[1] * kCenterVariance * prior.height + prior.center_y;
    const float w = std::exp(box_ptr[2] * kSizeVariance) * prior.width;
    const float h = std::exp(box_ptr[3] * kSizeVariance) * prior.height;

    const float x_min = (x_center - w * 0.5F) * static_cast<float>(frame_bgr.cols);
    const float y_min = (y_center - h * 0.5F) * static_cast<float>(frame_bgr.rows);
    const float x_max = (x_center + w * 0.5F) * static_cast<float>(frame_bgr.cols);
    const float y_max = (y_center + h * 0.5F) * static_cast<float>(frame_bgr.rows);

    cv::Rect rect(static_cast<int>(std::round(x_min)),
                  static_cast<int>(std::round(y_min)),
                  static_cast<int>(std::round(x_max - x_min)),
                  static_cast<int>(std::round(y_max - y_min)));
    rect &= cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
    if (rect.width > 0 && rect.height > 0) {
      out.push_back(Detection{rect, score});
    }
  }

  return nms(out, nms_threshold_);
#else
  (void)frame_bgr;
  return out;
#endif
}

std::vector<FaceDetector::PriorBox> FaceDetector::generatePriors() const {
  std::vector<PriorBox> priors;

  const std::array<std::vector<int>, 4> min_boxes = {
      std::vector<int>{10, 16, 24},
      std::vector<int>{32, 48},
      std::vector<int>{64, 96},
      std::vector<int>{128, 192, 256}};
  const std::array<int, 4> strides = {8, 16, 32, 64};

  for (std::size_t idx = 0; idx < strides.size(); ++idx) {
    const int stride = strides[idx];
    const int feature_map_w = static_cast<int>(std::ceil(static_cast<float>(input_width_) / static_cast<float>(stride)));
    const int feature_map_h = static_cast<int>(std::ceil(static_cast<float>(input_height_) / static_cast<float>(stride)));
    for (int h = 0; h < feature_map_h; ++h) {
      for (int w = 0; w < feature_map_w; ++w) {
        for (const int min_box : min_boxes[idx]) {
          PriorBox prior{};
          prior.center_x = (static_cast<float>(w) + 0.5F) / static_cast<float>(feature_map_w);
          prior.center_y = (static_cast<float>(h) + 0.5F) / static_cast<float>(feature_map_h);
          prior.width = static_cast<float>(min_box) / static_cast<float>(input_width_);
          prior.height = static_cast<float>(min_box) / static_cast<float>(input_height_);
          priors.push_back(prior);
        }
      }
    }
  }
  return priors;
}

std::vector<Detection> FaceDetector::nms(const std::vector<Detection>& detections, float nms_threshold) {
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b) {
    return a.det_score > b.det_score;
  });

  std::vector<Detection> kept;
  kept.reserve(sorted.size());
  for (const auto& det : sorted) {
    bool overlapped = false;
    for (const auto& existing : kept) {
      const float overlap = iou(det.box, existing.box);
      if (shouldMergeDuplicate(det.box, existing.box, overlap, nms_threshold)) {
        overlapped = true;
        break;
      }
    }
    if (!overlapped) {
      kept.push_back(det);
    }
  }
  return kept;
}

void FaceDetector::detectWithCascade(cv::CascadeClassifier& cascade,
                                     const cv::Mat& gray,
                                     bool mirror,
                                     std::vector<Detection>& out) const {
  cv::Mat work = gray;
  if (mirror) {
    cv::flip(gray, work, 1);
  }

  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(work, faces, 1.1, 4, 0, cv::Size(72, 72));
  for (auto rect : faces) {
    if (mirror) {
      rect.x = gray.cols - rect.x - rect.width;
    }
    rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (rect.width > 0 && rect.height > 0) {
      out.push_back(Detection{rect, 1.0F});
    }
  }
}

void FaceDetector::detectWithRotation(cv::CascadeClassifier& cascade,
                                      const cv::Mat& gray,
                                      double angle_deg,
                                      std::vector<Detection>& out) const {
  const cv::Point2f center(static_cast<float>(gray.cols) * 0.5F, static_cast<float>(gray.rows) * 0.5F);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);
  const cv::Rect2f bbox = cv::RotatedRect(center, gray.size(), angle_deg).boundingRect2f();
  rot.at<double>(0, 2) += bbox.width * 0.5 - center.x;
  rot.at<double>(1, 2) += bbox.height * 0.5 - center.y;

  cv::Mat rotated{};
  cv::warpAffine(gray,
                 rotated,
                 rot,
                 cv::Size(cvRound(bbox.width), cvRound(bbox.height)),
                 cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT,
                 cv::Scalar(0));

  std::vector<cv::Rect> faces;
  cascade.detectMultiScale(rotated, faces, 1.1, 4, 0, cv::Size(72, 72));
  if (faces.empty()) {
    return;
  }

  cv::Mat inv{};
  cv::invertAffineTransform(rot, inv);
  for (const auto& rect : faces) {
    const std::array<cv::Point2f, 4> corners = {
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height))};

    std::vector<cv::Point2f> mapped;
    mapped.reserve(corners.size());
    for (const auto& corner : corners) {
      mapped.push_back(applyAffine(inv, corner));
    }

    cv::Rect mapped_rect = cv::boundingRect(mapped);
    mapped_rect &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (mapped_rect.width > 0 && mapped_rect.height > 0) {
      out.push_back(Detection{mapped_rect, 0.95F});
    }
  }
}

std::vector<Detection> FaceDetector::suppressDuplicates(const std::vector<Detection>& detections) {
  std::vector<Detection> sorted = detections;
  std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b) {
    return a.box.area() > b.box.area();
  });

  std::vector<Detection> kept;
  kept.reserve(sorted.size());
  for (const auto& det : sorted) {
    bool overlapped = false;
    for (const auto& existing : kept) {
      const float overlap = iou(det.box, existing.box);
      if (shouldMergeDuplicate(det.box, existing.box, overlap, 0.35F)) {
        overlapped = true;
        break;
      }
    }
    if (!overlapped) {
      kept.push_back(det);
    }
  }
  return kept;
}

cv::Point2f FaceDetector::applyAffine(const cv::Mat& affine, const cv::Point2f& pt) {
  const double x = affine.at<double>(0, 0) * pt.x + affine.at<double>(0, 1) * pt.y + affine.at<double>(0, 2);
  const double y = affine.at<double>(1, 0) * pt.x + affine.at<double>(1, 1) * pt.y + affine.at<double>(1, 2);
  return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

float FaceDetector::iou(const cv::Rect& a, const cv::Rect& b) {
  const int x_left = std::max(a.x, b.x);
  const int y_top = std::max(a.y, b.y);
  const int x_right = std::min(a.x + a.width, b.x + b.width);
  const int y_bottom = std::min(a.y + a.height, b.y + b.height);
  if (x_right <= x_left || y_bottom <= y_top) {
    return 0.0F;
  }

  const float inter = static_cast<float>((x_right - x_left) * (y_bottom - y_top));
  const float union_area = static_cast<float>(a.area() + b.area()) - inter;
  return (union_area > 1e-6F) ? (inter / union_area) : 0.0F;
}

}  // namespace asdun
