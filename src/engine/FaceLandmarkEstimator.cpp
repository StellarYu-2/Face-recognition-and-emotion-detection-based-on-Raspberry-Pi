#include "engine/FaceLandmarkEstimator.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

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

float decodeCoordinate(float value, FaceLandmarkEstimator::CoordMode coord_mode, int crop_axis, int input_axis) {
  const float axis = static_cast<float>(std::max(crop_axis, 1));
  switch (coord_mode) {
    case FaceLandmarkEstimator::CoordMode::MinusOneOne:
      return (std::clamp(value, -1.0F, 1.0F) + 1.0F) * 0.5F * axis;
    case FaceLandmarkEstimator::CoordMode::Pixel:
      return value * axis / static_cast<float>(std::max(input_axis, 1));
    case FaceLandmarkEstimator::CoordMode::ZeroOne:
    default:
      return std::clamp(value, 0.0F, 1.0F) * axis;
  }
}

}  // namespace

bool FaceLandmarkEstimator::init(const std::string& model_param_path,
                                 const std::string& model_bin_path,
                                 int input_width,
                                 int input_height,
                                 std::string input_blob_name,
                                 std::string output_blob_name,
                                 float crop_scale,
                                 CoordMode coord_mode,
                                 bool use_rgb_input,
                                 float input_mean,
                                 float input_norm) {
  model_param_path_ = model_param_path;
  model_bin_path_ = model_bin_path;
  input_width_ = input_width > 0 ? input_width : 112;
  input_height_ = input_height > 0 ? input_height : 112;
  input_blob_name_ = std::move(input_blob_name);
  output_blob_name_ = std::move(output_blob_name);
  crop_scale_ = crop_scale > 1.0F ? crop_scale : 1.45F;
  coord_mode_ = coord_mode;
  use_rgb_input_ = use_rgb_input;
  input_mean_ = input_mean;
  input_norm_ = input_norm;
  ncnn_ready_ = false;

#ifdef USE_NCNN
  if (std::filesystem::exists(model_param_path_) && std::filesystem::exists(model_bin_path_)) {
    net_.clear();
    net_.opt.use_vulkan_compute = false;
    net_.opt.lightmode = true;
    net_.opt.use_packing_layout = true;
    net_.opt.num_threads = 4;
    if (net_.load_param(model_param_path_.c_str()) == 0 && net_.load_model(model_bin_path_.c_str()) == 0) {
      ncnn_ready_ = true;
    }
  }
#endif

  return true;
}

FivePointLandmarks FaceLandmarkEstimator::estimate(const cv::Mat& frame_bgr, const cv::Rect& face_box) const {
  FivePointLandmarks result{};
#ifdef USE_NCNN
  if (!ncnn_ready_ || frame_bgr.empty() || face_box.width <= 0 || face_box.height <= 0) {
    return result;
  }

  const cv::Rect crop_rect = buildCropRect(face_box, frame_bgr.size(), crop_scale_);
  if (crop_rect.width <= 0 || crop_rect.height <= 0) {
    return result;
  }

  const cv::Mat crop = frame_bgr(crop_rect);
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(crop.data,
                                               use_rgb_input_ ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR,
                                               crop.cols,
                                               crop.rows,
                                               input_width_,
                                               input_height_);
  if (std::abs(input_mean_) > 1e-6F || std::abs(input_norm_ - 1.0F) > 1e-6F) {
    const float mean_vals[3] = {input_mean_, input_mean_, input_mean_};
    const float norm_vals[3] = {input_norm_, input_norm_, input_norm_};
    in.substract_mean_normalize(mean_vals, norm_vals);
  }

  ncnn::Extractor ex = net_.create_extractor();
  ex.set_light_mode(true);
  ex.input(input_blob_name_.c_str(), in);

  ncnn::Mat out{};
  int rc = ex.extract(output_blob_name_.c_str(), out);
  if (rc != 0) {
    rc = ex.extract(0, out);
  }
  if (rc != 0) {
    return result;
  }

  const int total = matElementCount(out);
  if (total < 10) {
    return result;
  }

  const float* ptr = static_cast<const float*>(out.data);
  for (int i = 0; i < 5; ++i) {
    const float local_x = decodeCoordinate(ptr[i * 2], coord_mode_, crop_rect.width, input_width_);
    const float local_y = decodeCoordinate(ptr[i * 2 + 1], coord_mode_, crop_rect.height, input_height_);
    result.points[static_cast<std::size_t>(i)] =
        cv::Point2f(static_cast<float>(crop_rect.x) + local_x, static_cast<float>(crop_rect.y) + local_y);
  }
  result.valid = isReasonableLandmarkSet(result, face_box);
  return result;
#else
  (void)frame_bgr;
  (void)face_box;
  return result;
#endif
}

FaceLandmarkEstimator::CoordMode FaceLandmarkEstimator::parseCoordMode(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(),
                 lowered.end(),
                 lowered.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lowered == "minus_one_one" || lowered == "neg1_1" || lowered == "-1_1") {
    return CoordMode::MinusOneOne;
  }
  if (lowered == "pixel" || lowered == "pixels") {
    return CoordMode::Pixel;
  }
  return CoordMode::ZeroOne;
}

cv::Rect FaceLandmarkEstimator::buildCropRect(const cv::Rect& face_box, const cv::Size& image_size, float crop_scale) {
  if (face_box.width <= 0 || face_box.height <= 0 || image_size.width <= 0 || image_size.height <= 0) {
    return {};
  }

  const float side = static_cast<float>(std::max(face_box.width, face_box.height)) * crop_scale;
  const float cx = static_cast<float>(face_box.x) + static_cast<float>(face_box.width) * 0.5F;
  const float cy = static_cast<float>(face_box.y) + static_cast<float>(face_box.height) * 0.52F;
  cv::Rect crop(static_cast<int>(std::round(cx - side * 0.5F)),
                static_cast<int>(std::round(cy - side * 0.5F)),
                static_cast<int>(std::round(side)),
                static_cast<int>(std::round(side)));
  crop &= cv::Rect(0, 0, image_size.width, image_size.height);
  return crop;
}

bool FaceLandmarkEstimator::isReasonableLandmarkSet(const FivePointLandmarks& landmarks, const cv::Rect& face_box) {
  if (face_box.area() <= 0) {
    return false;
  }

  const auto& left_eye = landmarks.points[0];
  const auto& right_eye = landmarks.points[1];
  const auto& nose = landmarks.points[2];
  const auto& left_mouth = landmarks.points[3];
  const auto& right_mouth = landmarks.points[4];

  if (!(left_eye.x < right_eye.x && left_mouth.x < right_mouth.x)) {
    return false;
  }
  if (!(nose.y > std::min(left_eye.y, right_eye.y) && nose.y < std::max(left_mouth.y, right_mouth.y))) {
    return false;
  }
  if (!(left_mouth.y > left_eye.y && right_mouth.y > right_eye.y)) {
    return false;
  }

  const float eye_dx = right_eye.x - left_eye.x;
  const float eye_dy = right_eye.y - left_eye.y;
  const float eye_distance = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy);
  const float mouth_dx = right_mouth.x - left_mouth.x;
  const float mouth_dy = right_mouth.y - left_mouth.y;
  const float mouth_distance = std::sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy);
  const float face_scale = static_cast<float>(std::max(face_box.width, face_box.height));

  if (eye_distance < face_scale * 0.18F || eye_distance > face_scale * 0.8F) {
    return false;
  }
  if (mouth_distance < face_scale * 0.12F || mouth_distance > face_scale * 0.75F) {
    return false;
  }

  const cv::Rect bounds = cv::boundingRect(std::vector<cv::Point2f>(landmarks.points.begin(), landmarks.points.end()));
  const float overlap = static_cast<float>((bounds & face_box).area()) / static_cast<float>(std::max(1, bounds.area()));
  return overlap > 0.45F;
}

}  // namespace asdun
