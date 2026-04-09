#pragma once

#include <memory>
#include <string>

#include "camera/CameraManager.hpp"
#include "core/StateMachine.hpp"
#include "engine/EmotionRecognizer.hpp"
#include "engine/FaceDetector.hpp"
#include "engine/FaceRecognizer.hpp"
#include "engine/InferencePipeline.hpp"
#include "quality/FaceQualityGate.hpp"
#include "storage/Database.hpp"
#include "storage/EmbeddingStore.hpp"
#include "storage/FileStore.hpp"
#include "tracking/TrackManager.hpp"
#include "ui/Renderer.hpp"

namespace asdun {

struct AppConfig {
  std::string camera_source{"0"};
  int frame_width{480};
  int frame_height{360};
  int frame_fps{30};
  int enroll_target_images{4};
  int detect_interval{5};
  int recognition_interval{20};
  int emotion_interval{15};
  int max_inference_faces{1};
  float recognition_crop_scale{1.15F};
  bool debug_recognition{true};
  float min_face_area_ratio{0.08F};
  float blur_threshold{100.0F};
  int quality_stable_frames{3};
  float match_threshold{0.8F};
  float sigmoid_tau{0.08F};
  int track_ttl{10};
  float track_iou_threshold{0.2F};
  std::string db_path{"./data/db/face_access.db"};
  std::string images_root{"./data/images"};
  std::string face_cascade_path{"/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"};

  std::string detector_param_path{"./models/face_detector.param"};
  std::string detector_bin_path{"./models/face_detector.bin"};
  int detector_input_width{320};
  int detector_input_height{240};
  float detector_score_threshold{0.7F};
  float detector_nms_threshold{0.3F};
  std::string detector_input_blob{"input"};
  std::string detector_score_blob{"scores"};
  std::string detector_bbox_blob{"boxes"};

  std::string recognizer_param_path{"./models/onnx_to_ncnn/mobilefacenet.param"};
  std::string recognizer_bin_path{"./models/onnx_to_ncnn/mobilefacenet.bin"};
  int recognizer_input_size{112};
  std::string recognizer_color_order{"rgb"};
  std::string recognizer_input_blob{"data"};
  std::string recognizer_output_blob{"fc1"};

  std::string emotion_param_path{"./models/emotion.param"};
  std::string emotion_bin_path{"./models/emotion.bin"};
  int emotion_input_size{64};
  std::string emotion_input_blob{"input"};
  std::string emotion_output_blob{"output"};
};

class App {
 public:
  explicit App(std::string config_path);
  int run();

 private:
  bool loadConfig();
  bool initComponents();
  void printMainMenu() const;
  void handleEnrollment();
  void handleRecognition();
  static std::string trim(const std::string& s);

  std::string config_path_;
  AppConfig config_{};
  StateMachine sm_{};

  std::unique_ptr<CameraManager> camera_;
  std::unique_ptr<Database> database_;
  std::unique_ptr<FileStore> file_store_;
  std::unique_ptr<EmbeddingStore> embedding_store_;
  std::unique_ptr<FaceDetector> detector_;
  FaceRecognizer recognizer_{};
  EmotionRecognizer emotion_recognizer_{};
  std::unique_ptr<TrackManager> track_manager_;
  std::unique_ptr<FaceQualityGate> quality_gate_;
  std::unique_ptr<InferencePipeline> pipeline_;
  std::unique_ptr<Renderer> renderer_;
};

}  // namespace asdun
