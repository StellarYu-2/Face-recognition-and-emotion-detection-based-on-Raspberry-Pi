#include "core/App.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace asdun {

App::App(std::string config_path) : config_path_(std::move(config_path)), sm_(AppState::MainMenu) {}

int App::run() {
  if (!loadConfig()) {
    std::cerr << "[App] 配置加载失败: " << config_path_ << std::endl;
    return 1;
  }
  if (!initComponents()) {
    std::cerr << "[App] 组件初始化失败" << std::endl;
    return 1;
  }

  while (sm_.getState() != AppState::Exit) {
    printMainMenu();
    std::string cmd;
    std::getline(std::cin, cmd);
    cmd = trim(cmd);
    if (cmd == "1") {
      sm_.setState(AppState::EnrollInputName);
      handleEnrollment();
      if (sm_.getState() != AppState::Exit) {
        sm_.setState(AppState::MainMenu);
      }
    } else if (cmd == "2") {
      sm_.setState(AppState::Recognize);
      handleRecognition();
      if (sm_.getState() != AppState::Exit) {
        sm_.setState(AppState::MainMenu);
      }
    } else if (cmd == "0") {
      if (renderer_ != nullptr) {
        renderer_->closeWindow();
      }
      if (camera_ != nullptr) {
        camera_->stop();
      }
      sm_.setState(AppState::Exit);
    } else {
      std::cout << "[提示] 无效指令，请重新输入。" << std::endl;
    }
  }

  if (camera_ != nullptr) {
    camera_->stop();
  }
  return 0;
}

bool App::loadConfig() {
  std::ifstream fin(config_path_);
  if (!fin.is_open()) {
    std::cout << "[App] 未找到配置文件，使用默认配置: " << config_path_ << std::endl;
    return true;
  }

  std::string line;
  while (std::getline(fin, line)) {
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos) {
      line = line.substr(0, hash_pos);
    }
    if (line.find(':') == std::string::npos) {
      continue;
    }

    const std::size_t colon = line.find(':');
    std::string key = trim(line.substr(0, colon));
    std::string value = trim(line.substr(colon + 1));
    if (!value.empty() && value.front() == '"' && value.back() == '"' && value.size() >= 2) {
      value = value.substr(1, value.size() - 2);
    }
    if (key.empty() || value.empty()) {
      continue;
    }

    try {
      if (key == "camera_source") {
        config_.camera_source = value;
      } else if (key == "frame_width") {
        config_.frame_width = std::stoi(value);
      } else if (key == "frame_height") {
        config_.frame_height = std::stoi(value);
      } else if (key == "frame_fps") {
        config_.frame_fps = std::stoi(value);
      } else if (key == "detect_interval") {
        config_.detect_interval = std::stoi(value);
      } else if (key == "emotion_interval") {
        config_.emotion_interval = std::stoi(value);
      } else if (key == "min_face_area_ratio") {
        config_.min_face_area_ratio = std::stof(value);
      } else if (key == "blur_threshold") {
        config_.blur_threshold = std::stof(value);
      } else if (key == "quality_stable_frames") {
        config_.quality_stable_frames = std::stoi(value);
      } else if (key == "match_threshold") {
        config_.match_threshold = std::stof(value);
      } else if (key == "sigmoid_tau") {
        config_.sigmoid_tau = std::stof(value);
      } else if (key == "track_ttl") {
        config_.track_ttl = std::stoi(value);
      } else if (key == "track_iou_threshold") {
        config_.track_iou_threshold = std::stof(value);
      } else if (key == "db_path") {
        config_.db_path = value;
      } else if (key == "images_root") {
        config_.images_root = value;
      } else if (key == "face_cascade_path") {
        config_.face_cascade_path = value;
      }
    } catch (...) {
      std::cerr << "[App] 忽略错误配置项: " << key << "=" << value << std::endl;
    }
  }
  return true;
}

bool App::initComponents() {
  camera_ = std::make_unique<CameraManager>(config_.camera_source, config_.frame_width, config_.frame_height, config_.frame_fps);
  database_ = std::make_unique<Database>();
  file_store_ = std::make_unique<FileStore>(config_.images_root);
  detector_ = std::make_unique<FaceDetector>(config_.face_cascade_path);
  track_manager_ = std::make_unique<TrackManager>(config_.track_ttl, config_.track_iou_threshold);
  quality_gate_ =
      std::make_unique<FaceQualityGate>(config_.min_face_area_ratio, config_.blur_threshold, config_.quality_stable_frames);
  renderer_ = std::make_unique<Renderer>("asdun_access");

  if (!file_store_->ensureBaseDirs()) {
    std::cerr << "[FileStore] 创建目录失败: " << config_.images_root << std::endl;
    return false;
  }
  {
    std::error_code ec;
    const auto db_parent = std::filesystem::path(config_.db_path).parent_path();
    if (!db_parent.empty()) {
      std::filesystem::create_directories(db_parent, ec);
      if (ec) {
        std::cerr << "[Database] 创建数据库目录失败: " << db_parent << std::endl;
        return false;
      }
    }
  }
  if (!database_->open(config_.db_path) || !database_->initSchema()) {
    std::cerr << "[Database] 打开或初始化失败: " << config_.db_path << std::endl;
    return false;
  }
  embedding_store_ = std::make_unique<EmbeddingStore>(*database_);
  embedding_store_->reload();

  if (!detector_->init()) {
    std::cerr << "[FaceDetector] 初始化失败，请检查 face_cascade_path 配置。" << std::endl;
    return false;
  }
  if (!emotion_recognizer_.init()) {
    std::cerr << "[EmotionRecognizer] 初始化失败。" << std::endl;
    return false;
  }

  pipeline_ = std::make_unique<InferencePipeline>(*detector_,
                                                  recognizer_,
                                                  emotion_recognizer_,
                                                  *embedding_store_,
                                                  *track_manager_,
                                                  config_.detect_interval,
                                                  config_.emotion_interval,
                                                  config_.match_threshold,
                                                  config_.sigmoid_tau);
  return true;
}

void App::printMainMenu() const {
  std::cout << "\n========================\n";
  std::cout << "树莓派门禁系统主菜单\n";
  std::cout << "1) 录入新人员（2张）\n";
  std::cout << "2) 实时识别 + 情绪检测\n";
  std::cout << "0) 退出\n";
  std::cout << "请输入指令: ";
}

void App::handleEnrollment() {
  if (!camera_->start()) {
    std::cerr << "[Enrollment] 摄像头启动失败" << std::endl;
    return;
  }

  std::cout << "请输入人员姓名（如 zhangsan）: ";
  std::string raw_name;
  std::getline(std::cin, raw_name);
  const std::string name = file_store_->sanitizeName(trim(raw_name));
  if (name.empty()) {
    std::cout << "[Enrollment] 姓名无效，已返回主菜单。" << std::endl;
    camera_->stop();
    renderer_->closeWindow();
    return;
  }

  int person_id = 0;
  if (database_->personExists(name, &person_id)) {
    std::cout << "该人员已有记录，是否清空旧数据并重新录入？(y/n): ";
    std::string yn;
    std::getline(std::cin, yn);
    yn = trim(yn);
    std::transform(yn.begin(), yn.end(), yn.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (yn != "y") {
      std::cout << "[Enrollment] 已取消覆盖，返回主菜单。" << std::endl;
      camera_->stop();
      renderer_->closeWindow();
      return;
    }

    const auto old_paths = database_->listImagePathsByPerson(name);
    if (!database_->deletePersonAndEmbeddings(name)) {
      std::cerr << "[Enrollment] 清理旧数据库记录失败，已中止。" << std::endl;
      camera_->stop();
      renderer_->closeWindow();
      return;
    }
    file_store_->removeFiles(old_paths);
    file_store_->removePersonDir(name);
    std::cout << "[Enrollment] 已清理旧数据，开始重新录入。" << std::endl;
  }

  if (!database_->upsertPerson(name, &person_id)) {
    std::cerr << "[Enrollment] 创建人员记录失败。" << std::endl;
    camera_->stop();
    renderer_->closeWindow();
    return;
  }

  FaceQualityGate gate(config_.min_face_area_ratio, config_.blur_threshold, config_.quality_stable_frames);
  int captured = 0;
  std::cout << "[Enrollment] 按 s 抓拍，按 q 退出。需要有效样本 2 张。" << std::endl;

  while (captured < 2) {
    FramePacket frame{};
    if (!camera_->getLatestFrame(frame, 200)) {
      continue;
    }

    const auto dets = detector_->detect(frame.bgr);
    cv::Rect best_face{};
    if (!dets.empty()) {
      best_face = dets.front().box;
      for (const auto& d : dets) {
        if (d.box.area() > best_face.area()) {
          best_face = d.box;
        }
      }
    }

    bool ready = false;
    std::string status = "No face";
    QualityResult qres{};
    if (best_face.area() > 0) {
      qres = gate.evaluate(frame.bgr, best_face);
      ready = qres.valid;
      std::ostringstream oss;
      oss << "Captured " << captured << "/2 | " << qres.reason << " | area=" << qres.area_ratio
          << " blur=" << qres.blur_score;
      status = oss.str();
    }

    renderer_->drawEnrollment(frame.bgr, best_face, status, ready);
    const int key = renderer_->waitKey(1);
    if (key == 'q' || key == 'Q') {
      std::cout << "[Enrollment] 用户中止录入。" << std::endl;
      break;
    }
    if (key == '0') {
      sm_.setState(AppState::Exit);
      break;
    }
    if ((key == 's' || key == 'S') && ready && best_face.area() > 0) {
      const cv::Rect bounded = best_face & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
      const cv::Mat face = frame.bgr(bounded).clone();

      std::string image_path;
      if (!file_store_->saveFaceImage(name, face, &image_path)) {
        std::cerr << "[Enrollment] 图片保存失败。" << std::endl;
        continue;
      }

      const auto emb = recognizer_.extractEmbedding(face);
      if (emb.empty()) {
        std::cerr << "[Enrollment] 特征提取失败，跳过本次样本。" << std::endl;
        std::error_code ec;
        std::filesystem::remove(image_path, ec);
        continue;
      }
      if (!database_->insertEmbedding(person_id, emb, image_path, qres.blur_score)) {
        std::cerr << "[Enrollment] 特征入库失败。" << std::endl;
        std::error_code ec;
        std::filesystem::remove(image_path, ec);
        continue;
      }
      captured++;
      std::cout << "[Enrollment] 抓拍成功: " << captured << "/2" << std::endl;
    }
  }

  embedding_store_->reload();
  camera_->stop();
  renderer_->closeWindow();
  std::cout << "[Enrollment] 完成，返回主菜单。" << std::endl;
}

void App::handleRecognition() {
  if (!camera_->start()) {
    std::cerr << "[Recognition] 摄像头启动失败" << std::endl;
    return;
  }
  embedding_store_->reload();
  std::cout << "[Recognition] 按 q 退出识别，按 r 重新加载本地特征库。" << std::endl;

  while (true) {
    FramePacket frame{};
    if (!camera_->getLatestFrame(frame, 200)) {
      continue;
    }
    const RecognitionResult result = pipeline_->process(frame);
    renderer_->drawRecognition(frame.bgr, result.tracks);

    const int key = renderer_->waitKey(1);
    if (key == 'q' || key == 'Q') {
      break;
    }
    if (key == '0') {
      sm_.setState(AppState::Exit);
      break;
    }
    if (key == 'r' || key == 'R') {
      embedding_store_->reload();
      std::cout << "[Recognition] 特征库已重新加载。" << std::endl;
    }
  }

  camera_->stop();
  renderer_->closeWindow();
}

std::string App::trim(const std::string& s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  auto begin = std::find_if_not(s.begin(), s.end(), is_space);
  if (begin == s.end()) {
    return "";
  }
  auto end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
  return std::string(begin, end);
}

}  // namespace asdun
