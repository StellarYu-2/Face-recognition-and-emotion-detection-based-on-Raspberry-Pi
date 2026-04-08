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
  int frame_width{640};
  int frame_height{480};
  int frame_fps{30};
  int detect_interval{3};
  int emotion_interval{2};
  float min_face_area_ratio{0.08F};
  float blur_threshold{100.0F};
  int quality_stable_frames{3};
  float match_threshold{0.8F};
  float sigmoid_tau{0.08F};
  int track_ttl{10};
  float track_iou_threshold{0.3F};
  std::string db_path{"./data/db/face_access.db"};
  std::string images_root{"./data/images"};
  std::string face_cascade_path{"/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"};
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
