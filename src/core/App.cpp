#include "core/App.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

/** @brief 将字符串解析为布尔值，支持 1/true/yes/on */
bool parseBoolValue(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string trimValue(const std::string& value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  const auto begin = std::find_if_not(value.begin(), value.end(), is_space);
  if (begin == value.end()) {
    return "";
  }
  const auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
  return std::string(begin, end);
}

std::string stripOptionalQuotes(std::string value) {
  value = trimValue(value);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

/**
 * @brief 根据已采集数量计算当前录入阶段（0~3）
 * @param captured      已采集的照片数
 * @param target_images 目标采集总数
 * @return 阶段索引：0=正面, 1=左转, 2=右转, 3=俯仰
 */
int enrollmentStageIndex(int captured, int target_images) {
  if (target_images <= 0) {
    return 0;
  }
  const int progress = captured * 4;
  return std::min(3, progress / target_images);
}

/**
 * @brief 根据当前录入阶段返回姿势提示语
 * 引导用户变换角度，提高注册特征的多样性。
 */
std::string enrollmentPoseHint(int captured, int target_images) {
  switch (enrollmentStageIndex(captured, target_images)) {
    case 0:
      return "Look forward";
    case 1:
      return "Turn slightly left";
    case 2:
      return "Turn slightly right";
    default:
      return "Raise or lower chin";
  }
}

}  // namespace

/** @brief 构造时传入配置文件路径，初始状态设为主菜单 */
App::App(std::string config_path) : config_path_(std::move(config_path)), sm_(AppState::MainMenu) {}

/**
 * @brief 应用程序主入口
 *
 * 流程：加载配置 -> 初始化组件 -> 进入主循环等待用户选择：
 *  1) 录入/更新人员
 *  2) 实时识别+情绪检测
 *  3) 删除人员
 *  0) 退出
 */
int App::run() {
  if (!loadConfig()) {
    std::cerr << "[App] failed to load config: " << config_path_ << std::endl;
    return 1;
  }
  if (!initComponents()) {
    std::cerr << "[App] failed to initialize components." << std::endl;
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
    } else if (cmd == "3") {
      handleDeletePerson();
    } else if (cmd == "0") {
      // 退出前关闭渲染窗口和相机
      if (renderer_ != nullptr) {
        renderer_->closeWindow();
      }
      if (camera_ != nullptr) {
        camera_->stop();
      }
      sm_.setState(AppState::Exit);
    } else {
      std::cout << "[Menu] invalid command, please retry." << std::endl;
    }
  }

  if (camera_ != nullptr) {
    camera_->stop();
  }
  return 0;
}

/**
 * @brief 从 YAML/类 YAML 配置文件加载参数
 *
 * 支持 `#` 开头的注释，冒号分隔的键值对。若文件不存在则使用 AppConfig 中的默认值。
 * 读取失败/非法的条目会被忽略并打印警告。
 */
bool App::loadConfig() {
  std::ifstream fin(config_path_);
  if (!fin.is_open()) {
    std::cout << "[App] config not found, using defaults: " << config_path_ << std::endl;
    return true;
  }

  std::string line;
  bool reading_cloud_server_urls = false;
  while (std::getline(fin, line)) {
    // 截去行内注释
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos) {
      line = line.substr(0, hash_pos);
    }
    const std::string trimmed_line = trim(line);
    if (trimmed_line.empty()) {
      continue;
    }

    if (reading_cloud_server_urls) {
      if (trimmed_line.rfind("-", 0) == 0) {
        const std::string url = stripOptionalQuotes(trim(trimmed_line.substr(1)));
        if (!url.empty()) {
          config_.cloud.server_urls.push_back(url);
        }
        continue;
      }
      reading_cloud_server_urls = false;
    }

    if (trimmed_line == "cloud_server_urls:") {
      config_.cloud.server_urls.clear();
      reading_cloud_server_urls = true;
      continue;
    }

    if (line.find(':') == std::string::npos) {
      continue;
    }

    const std::size_t colon = line.find(':');
    std::string key = trim(line.substr(0, colon));
    std::string value = stripOptionalQuotes(trim(line.substr(colon + 1)));
    if (key.empty() || value.empty()) {
      continue;
    }

    try {
      // 相机与通用参数
      if (key == "camera_source") {
        config_.camera_source = value;
      } else if (key == "frame_width") {
        config_.frame_width = std::stoi(value);
      } else if (key == "frame_height") {
        config_.frame_height = std::stoi(value);
      } else if (key == "frame_fps") {
        config_.frame_fps = std::stoi(value);
      } else if (key == "opencv_num_threads") {
        config_.opencv_num_threads = std::stoi(value);
      } else if (key == "enroll_target_images") {
        config_.enroll_target_images = std::stoi(value);
      }
      // 推理间隔
      else if (key == "detect_interval") {
        config_.detect_interval = std::stoi(value);
      } else if (key == "recognition_interval") {
        config_.recognition_interval = std::stoi(value);
      } else if (key == "emotion_interval") {
        config_.emotion_interval = std::stoi(value);
      }
      // 每帧处理人脸数与质量阈值
      else if (key == "max_inference_faces") {
        config_.max_inference_faces = std::stoi(value);
      } else if (key == "max_emotion_faces") {
        config_.max_emotion_faces = std::stoi(value);
      } else if (key == "recognition_crop_scale") {
        config_.recognition_crop_scale = std::stof(value);
      } else if (key == "recognition_min_face_size") {
        config_.recognition_min_face_size = std::stoi(value);
      } else if (key == "recognition_blur_threshold") {
        config_.recognition_blur_threshold = std::stof(value);
      } else if (key == "recognition_margin_threshold") {
        config_.recognition_margin_threshold = std::stof(value);
      }
      // 冷却与重触发参数
      else if (key == "known_identity_cooldown_ms") {
        config_.known_identity_cooldown_ms = std::stoi(value);
      } else if (key == "unknown_identity_cooldown_ms") {
        config_.unknown_identity_cooldown_ms = std::stoi(value);
      } else if (key == "recognition_retrigger_blur_gain") {
        config_.recognition_retrigger_blur_gain = std::stof(value);
      } else if (key == "recognition_retrigger_size_gain") {
        config_.recognition_retrigger_size_gain = std::stoi(value);
      }
      // 情绪识别参数
      else if (key == "emotion_crop_scale") {
        config_.emotion_crop_scale = std::stof(value);
      } else if (key == "emotion_min_face_size") {
        config_.emotion_min_face_size = std::stoi(value);
      } else if (key == "emotion_cooldown_ms") {
        config_.emotion_cooldown_ms = std::stoi(value);
      } else if (key == "emotion_require_known_identity") {
        config_.emotion_require_known_identity = parseBoolValue(value);
      } else if (key == "emotion_non_calm_floor") {
        config_.emotion_non_calm_floor = std::stof(value);
      } else if (key == "emotion_handoff_margin") {
        config_.emotion_handoff_margin = std::stof(value);
      }
      // 调试与质量门控
      else if (key == "debug_recognition") {
        config_.debug_recognition = parseBoolValue(value);
      } else if (key == "debug_emotion") {
        config_.debug_emotion = parseBoolValue(value);
      } else if (key == "min_face_area_ratio") {
        config_.min_face_area_ratio = std::stof(value);
      } else if (key == "blur_threshold") {
        config_.blur_threshold = std::stof(value);
      } else if (key == "quality_stable_frames") {
        config_.quality_stable_frames = std::stoi(value);
      }
      // 匹配与跟踪参数
      else if (key == "match_threshold") {
        config_.match_threshold = std::stof(value);
      } else if (key == "sigmoid_tau") {
        config_.sigmoid_tau = std::stof(value);
      } else if (key == "track_ttl") {
        config_.track_ttl = std::stoi(value);
      } else if (key == "track_iou_threshold") {
        config_.track_iou_threshold = std::stof(value);
      }
      // 路径配置
      else if (key == "db_path") {
        config_.db_path = value;
      } else if (key == "images_root") {
        config_.images_root = value;
      } else if (key == "face_cascade_path") {
        config_.face_cascade_path = value;
      }
      // 检测模型参数
      else if (key == "detector_param_path") {
        config_.detector_param_path = value;
      } else if (key == "detector_bin_path") {
        config_.detector_bin_path = value;
      } else if (key == "detector_input_width") {
        config_.detector_input_width = std::stoi(value);
      } else if (key == "detector_input_height") {
        config_.detector_input_height = std::stoi(value);
      } else if (key == "detector_score_threshold") {
        config_.detector_score_threshold = std::stof(value);
      } else if (key == "detector_nms_threshold") {
        config_.detector_nms_threshold = std::stof(value);
      } else if (key == "detector_enable_cascade_fallback") {
        config_.detector_enable_cascade_fallback = parseBoolValue(value);
      } else if (key == "detector_input_blob") {
        config_.detector_input_blob = value;
      } else if (key == "detector_score_blob") {
        config_.detector_score_blob = value;
      } else if (key == "detector_bbox_blob") {
        config_.detector_bbox_blob = value;
      }
      // 识别模型参数
      else if (key == "recognizer_param_path") {
        config_.recognizer_param_path = value;
      } else if (key == "recognizer_bin_path") {
        config_.recognizer_bin_path = value;
      } else if (key == "recognizer_input_size") {
        config_.recognizer_input_size = std::stoi(value);
      } else if (key == "recognizer_color_order") {
        config_.recognizer_color_order = value;
      } else if (key == "recognizer_input_blob") {
        config_.recognizer_input_blob = value;
      } else if (key == "recognizer_output_blob") {
        config_.recognizer_output_blob = value;
      }
      // 关键点模型参数
      else if (key == "landmark_param_path") {
        config_.landmark_param_path = value;
      } else if (key == "landmark_bin_path") {
        config_.landmark_bin_path = value;
      } else if (key == "landmark_input_width") {
        config_.landmark_input_width = std::stoi(value);
      } else if (key == "landmark_input_height") {
        config_.landmark_input_height = std::stoi(value);
      } else if (key == "landmark_crop_scale") {
        config_.landmark_crop_scale = std::stof(value);
      } else if (key == "landmark_coord_mode") {
        config_.landmark_coord_mode = value;
      } else if (key == "landmark_color_order") {
        config_.landmark_color_order = value;
      } else if (key == "landmark_mean") {
        config_.landmark_mean = std::stof(value);
      } else if (key == "landmark_norm") {
        config_.landmark_norm = std::stof(value);
      } else if (key == "landmark_input_blob") {
        config_.landmark_input_blob = value;
      } else if (key == "landmark_output_blob") {
        config_.landmark_output_blob = value;
      } else if (key == "aligned_face_size") {
        config_.aligned_face_size = std::stoi(value);
      }
      // 情绪模型参数
      else if (key == "emotion_param_path") {
        config_.emotion_param_path = value;
      } else if (key == "emotion_bin_path") {
        config_.emotion_bin_path = value;
      } else if (key == "emotion_input_size") {
        config_.emotion_input_size = std::stoi(value);
      } else if (key == "emotion_input_blob") {
        config_.emotion_input_blob = value;
      } else if (key == "emotion_output_blob") {
        config_.emotion_output_blob = value;
      }
      // 云端混合推理参数
      else if (key == "inference_mode") {
        config_.inference_mode = value;
      } else if (key == "cloud_server_url") {
        config_.cloud.server_url = value;
      } else if (key == "cloud_health_check_path") {
        config_.cloud.health_check_path = value;
      } else if (key == "cloud_timeout_ms") {
        config_.cloud.timeout_ms = std::stoi(value);
      } else if (key == "cloud_connect_timeout_ms") {
        config_.cloud.connect_timeout_ms = std::stoi(value);
      } else if (key == "cloud_min_interval_ms") {
        config_.cloud.min_interval_ms = std::stoi(value);
      } else if (key == "cloud_max_queue_size") {
        config_.cloud.max_queue_size = std::stoi(value);
      } else if (key == "cloud_jpeg_quality") {
        config_.cloud.jpeg_quality = std::stoi(value);
      } else if (key == "cloud_crop_size") {
        config_.cloud.crop_size = std::stoi(value);
      } else if (key == "cloud_crop_scale") {
        config_.cloud.crop_scale = std::stof(value);
      } else if (key == "cloud_result_ttl_ms") {
        config_.cloud.result_ttl_ms = std::stoi(value);
      } else if (key == "cloud_identity_min_confidence") {
        config_.cloud.identity_min_confidence = std::stof(value);
      } else if (key == "cloud_identity_apply_unknown") {
        config_.cloud.identity_apply_unknown = parseBoolValue(value);
      } else if (key == "cloud_emotion_min_confidence") {
        config_.cloud.emotion_min_confidence = std::stof(value);
      } else if (key == "cloud_emotion_min_gap") {
        config_.cloud.emotion_min_gap = std::stof(value);
      } else if (key == "cloud_emotion_sad_min_confidence") {
        config_.cloud.emotion_sad_min_confidence = std::stof(value);
      } else if (key == "cloud_emotion_sad_min_gap") {
        config_.cloud.emotion_sad_min_gap = std::stof(value);
      } else if (key == "cloud_apply_identity") {
        config_.cloud.apply_identity = parseBoolValue(value);
      } else if (key == "cloud_apply_emotion") {
        config_.cloud.apply_emotion = parseBoolValue(value);
      } else if (key == "cloud_debug") {
        config_.cloud.debug = parseBoolValue(value);
      }
    } catch (...) {
      std::cerr << "[App] ignoring invalid config entry: " << key << "=" << value << std::endl;
    }
  }

  std::string mode = config_.inference_mode;
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  config_.cloud.enabled = (mode == "hybrid" || mode == "cloud");
  return true;
}

/**
 * @brief 初始化所有子模块
 *
 * 按顺序创建并启动：
 *  1. CameraManager（相机）
 *  2. Database / FileStore（数据库与文件存储）
 *  3. FaceDetector（人脸检测器）
 *  4. TrackManager（跟踪器）
 *  5. FaceQualityGate（质量门控）
 *  6. FaceRecognizer / FaceLandmarkEstimator / FaceAligner（识别链路）
 *  7. EmotionRecognizer（情绪识别）
 *  8. InferencePipeline（推理调度管线）
 *  9. Renderer（UI 渲染）
 *
 * @return 全部初始化成功返回 true，任一失败返回 false
 */
bool App::initComponents() {
  cv::setUseOptimized(true);
  cv::setNumThreads(std::max(1, config_.opencv_num_threads));

  // 1. 相机
  camera_ = std::make_unique<CameraManager>(config_.camera_source, config_.frame_width, config_.frame_height, config_.frame_fps);
  // 2. 数据持久化
  database_ = std::make_unique<Database>();
  file_store_ = std::make_unique<FileStore>(config_.images_root);
  // 3. 检测器
  detector_ = std::make_unique<FaceDetector>(config_.face_cascade_path,
                                             config_.detector_param_path,
                                             config_.detector_bin_path,
                                             config_.detector_input_width,
                                             config_.detector_input_height,
                                             config_.detector_score_threshold,
                                             config_.detector_nms_threshold,
                                             config_.detector_enable_cascade_fallback,
                                             config_.detector_input_blob,
                                             config_.detector_score_blob,
                                             config_.detector_bbox_blob);
  // 4. 跟踪与质量门控
  track_manager_ = std::make_unique<TrackManager>(config_.track_ttl, config_.track_iou_threshold);
  quality_gate_ =
      std::make_unique<FaceQualityGate>(config_.min_face_area_ratio, config_.blur_threshold, config_.quality_stable_frames);
  // 9. 渲染器（先创建，后续录 enrollment 需要用到）
  renderer_ = std::make_unique<Renderer>("asdun_access");

  // 确保图片根目录存在
  if (!file_store_->ensureBaseDirs()) {
    std::cerr << "[FileStore] failed to create base directories: " << config_.images_root << std::endl;
    return false;
  }

  // 确保数据库目录存在
  {
    std::error_code ec;
    const auto db_parent = std::filesystem::path(config_.db_path).parent_path();
    if (!db_parent.empty()) {
      std::filesystem::create_directories(db_parent, ec);
      if (ec) {
        std::cerr << "[Database] failed to create database directory: " << db_parent << std::endl;
        return false;
      }
    }
  }

  // 打开数据库并建表
  if (!database_->open(config_.db_path) || !database_->initSchema()) {
    std::cerr << "[Database] failed to open or initialize: " << config_.db_path << std::endl;
    return false;
  }
  embedding_store_ = std::make_unique<EmbeddingStore>(*database_);

  // 初始化 ncnn 模型
  if (!detector_->init()) {
    std::cerr << "[FaceDetector] init failed." << std::endl;
    return false;
  }
  if (!recognizer_.init(config_.recognizer_param_path,
                        config_.recognizer_bin_path,
                        config_.recognizer_input_size,
                        config_.recognizer_input_blob,
                        config_.recognizer_output_blob,
                        parseBoolValue(config_.recognizer_color_order == "bgr" ? "false" : "true"))) {
    std::cerr << "[FaceRecognizer] init failed." << std::endl;
    return false;
  }
  if (!landmark_estimator_.init(config_.landmark_param_path,
                                config_.landmark_bin_path,
                                config_.landmark_input_width,
                                config_.landmark_input_height,
                                config_.landmark_input_blob,
                                config_.landmark_output_blob,
                                config_.landmark_crop_scale,
                                FaceLandmarkEstimator::parseCoordMode(config_.landmark_coord_mode),
                                parseBoolValue(config_.landmark_color_order == "bgr" ? "false" : "true"),
                                config_.landmark_mean,
                                config_.landmark_norm)) {
    std::cerr << "[FaceLandmarkEstimator] init failed." << std::endl;
    return false;
  }

  // 设置对齐输出尺寸，并判断关键点来源（检测器自带 or 独立模型）
  face_aligner_.setOutputSize(config_.aligned_face_size);
  if (detector_->providesLandmarks()) {
    std::cout << "[Landmark] detector keypoints enabled." << std::endl;
  } else if (landmark_estimator_.ready()) {
    std::cout << "[Landmark] standalone landmark model enabled." << std::endl;
  } else {
    std::cout << "[Landmark] no alignment source enabled yet." << std::endl;
  }

  // 加载底库特征，用于 1:N 比对
  embedding_store_->setActiveModelTag(recognizer_.modelTag());
  embedding_store_->reload();

  // 情绪模型
  if (!emotion_recognizer_.init(config_.emotion_param_path,
                                config_.emotion_bin_path,
                                config_.emotion_input_size,
                                config_.emotion_input_blob,
                                config_.emotion_output_blob)) {
    std::cerr << "[EmotionRecognizer] init failed." << std::endl;
    return false;
  }
  emotion_recognizer_.setDecisionPolicy(config_.emotion_non_calm_floor, config_.emotion_handoff_margin);

  // 8. 组装推理管线：把检测、跟踪、识别、情绪串成一个 pipeline
  pipeline_ = std::make_unique<InferencePipeline>(*detector_,
                                                  landmark_estimator_,
                                                  face_aligner_,
                                                  recognizer_,
                                                  emotion_recognizer_,
                                                  *embedding_store_,
                                                  *track_manager_,
                                                  config_.detect_interval,
                                                  config_.recognition_interval,
                                                  config_.emotion_interval,
                                                  config_.max_inference_faces,
                                                  config_.max_emotion_faces,
                                                  config_.recognition_crop_scale,
                                                  config_.recognition_min_face_size,
                                                  config_.recognition_blur_threshold,
                                                  config_.recognition_margin_threshold,
                                                  config_.known_identity_cooldown_ms,
                                                  config_.unknown_identity_cooldown_ms,
                                                  config_.recognition_retrigger_blur_gain,
                                                  config_.recognition_retrigger_size_gain,
                                                  config_.emotion_crop_scale,
                                                  config_.emotion_min_face_size,
                                                  config_.emotion_cooldown_ms,
                                                  config_.emotion_require_known_identity,
                                                  config_.debug_recognition,
                                                  config_.debug_emotion,
                                                  config_.match_threshold,
                                                  config_.sigmoid_tau);

  if (config_.cloud.enabled) {
    cloud_client_ = std::make_unique<CloudClient>(config_.cloud);
    if (!cloud_client_->start()) {
      std::cerr << "[CloudClient] disabled; local pipeline will continue." << std::endl;
      cloud_client_.reset();
    }
  }
  return true;
}

/** @brief 打印主菜单到控制台 */
void App::printMainMenu() const {
  std::cout << "\n========================\n";
  std::cout << "ASDUN Main Menu\n";
  std::cout << "1) Enroll or update person\n";
  std::cout << "2) Live recognition + emotion\n";
  std::cout << "3) Delete person\n";
  std::cout << "0) Exit\n";
  std::cout << "Enter command: ";
}

/**
 * @brief 处理【录入/更新人员】流程
 *
 * 步骤：
 *  1. 启动相机
 *  2. 输入姓名（已存在则提示覆盖）
 *  3. 引导用户变换姿势，按 's' 采集
 *  4. 对每张照片做人脸检测 -> 质量评估 -> 提取特征 -> 存入数据库
 *  5. 完成后重载底库
 */
void App::handleEnrollment() {
  if (!camera_->start()) {
    std::cerr << "[Enrollment] failed to start camera." << std::endl;
    return;
  }

  std::cout << "Enter person name: ";
  std::string raw_name;
  std::getline(std::cin, raw_name);
  const std::string name = file_store_->sanitizeName(trim(raw_name));
  if (name.empty()) {
    std::cout << "[Enrollment] invalid name, returning to menu." << std::endl;
    camera_->stop();
    renderer_->closeWindow();
    return;
  }

  // 若人员已存在，提示是否覆盖
  int person_id = 0;
  if (database_->personExists(name, &person_id)) {
    std::cout << "Person already exists. Re-enroll and replace old data? (y/n): ";
    std::string yn;
    std::getline(std::cin, yn);
    yn = trim(yn);
    std::transform(yn.begin(), yn.end(), yn.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (yn != "y") {
      std::cout << "[Enrollment] cancelled." << std::endl;
      camera_->stop();
      renderer_->closeWindow();
      return;
    }

    // 删除旧数据（数据库记录 + 本地图片）
    const auto old_paths = database_->listImagePathsByPerson(name);
    if (!database_->deletePersonAndEmbeddings(name)) {
      std::cerr << "[Enrollment] failed to clear old database records." << std::endl;
      camera_->stop();
      renderer_->closeWindow();
      return;
    }
    file_store_->removeFiles(old_paths);
    file_store_->removePersonDir(name);
    std::cout << "[Enrollment] old data cleared, starting new enrollment." << std::endl;
  }

  // 插入或更新人员记录，获取 person_id
  if (!database_->upsertPerson(name, &person_id)) {
    std::cerr << "[Enrollment] failed to create person record." << std::endl;
    camera_->stop();
    renderer_->closeWindow();
    return;
  }

  // 录入主循环
  FaceQualityGate gate(config_.min_face_area_ratio, config_.blur_threshold, config_.quality_stable_frames);
  const int target_images = std::max(1, config_.enroll_target_images);
  int captured = 0;
  int last_stage = -1;
  std::vector<std::string> enrolled_image_paths;
  std::cout << "[Enrollment] press s to capture, q to quit. Target images: " << target_images << std::endl;

  while (captured < target_images) {
    FramePacket frame{};
    if (!camera_->getLatestFrame(frame, 200)) {
      continue;
    }

    // ---- 人脸检测：选最大的人脸 ----
    const auto dets = detector_->detect(frame.bgr);
    cv::Rect best_face{};
    FivePointLandmarks best_landmarks{};
    if (!dets.empty()) {
      best_face = dets.front().box;
      best_landmarks = dets.front().landmarks;
      for (const auto& d : dets) {
        if (d.box.area() > best_face.area()) {
          best_face = d.box;
          best_landmarks = d.landmarks;
        }
      }
      // 若检测器自带关键点，用其细化人脸框
      if (best_landmarks.valid) {
        const cv::Rect refined_face = face_aligner_.refineBox(best_landmarks, frame.bgr.size());
        if (refined_face.width > 0 && refined_face.height > 0) {
          best_face = (best_face | refined_face) & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
        }
      }
      // 否则尝试用独立关键点模型估计
      else if (landmark_estimator_.ready()) {
        const auto landmarks = landmark_estimator_.estimate(frame.bgr, best_face);
        if (landmarks.valid) {
          best_landmarks = landmarks;
          const cv::Rect refined_face = face_aligner_.refineBox(landmarks, frame.bgr.size());
          if (refined_face.width > 0 && refined_face.height > 0) {
            best_face = (best_face | refined_face) & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
          }
        }
      }
    }

    // ---- 质量评估与 UI 状态更新 ----
    bool ready = false;
    std::string status = "No face";
    QualityResult qres{};
    const std::string pose_hint = enrollmentPoseHint(captured, target_images);
    const int stage = enrollmentStageIndex(captured, target_images);
    if (stage != last_stage) {
      std::cout << "[Enrollment] pose hint: " << pose_hint << std::endl;
      last_stage = stage;
    }
    if (best_face.area() > 0) {
      qres = gate.evaluate(frame.bgr, best_face);
      ready = qres.valid;
      std::ostringstream oss;
      oss << pose_hint << " | " << captured << "/" << target_images << " | " << qres.reason << " | area="
          << qres.area_ratio << " blur=" << qres.blur_score;
      status = oss.str();
    } else {
      status = pose_hint + " | No face";
    }

    renderer_->drawEnrollment(frame.bgr, best_face, status, ready);

    // ---- 按键处理 ----
    const int key = renderer_->waitKey(1);
    if (key == 'q' || key == 'Q') {
      std::cout << "[Enrollment] cancelled by user." << std::endl;
      break;
    }
    if (key == '0') {
      sm_.setState(AppState::Exit);
      break;
    }
    // 按 's' 且质量合格时保存
    if ((key == 's' || key == 'S') && ready && best_face.area() > 0) {
      const cv::Rect bounded = best_face & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
      if (bounded.width <= 0 || bounded.height <= 0) {
        std::cerr << "[Enrollment] invalid face region, please retry." << std::endl;
        continue;
      }
      cv::Mat face = frame.bgr(bounded).clone();
      cv::Rect saved_face_rect = bounded;
      if (config_.cloud.enabled) {
        const cv::Rect expanded_for_cloud = expandRect(best_face, frame.bgr.size(), std::max(1.0F, config_.cloud.crop_scale));
        if (expanded_for_cloud.width > 0 && expanded_for_cloud.height > 0) {
          saved_face_rect = expanded_for_cloud;
        }
      }
      cv::Mat saved_face = frame.bgr(saved_face_rect).clone();

      // 保存图片到本地
      std::string image_path;
      if (!file_store_->saveFaceImage(name, saved_face, &image_path)) {
        std::cerr << "[Enrollment] failed to save image." << std::endl;
        continue;
      }

      // 提取特征向量
      const auto emb = recognizer_.extractEmbedding(face);
      if (emb.empty()) {
        std::cerr << "[Enrollment] failed to extract embedding, skipping sample." << std::endl;
        std::error_code ec;
        std::filesystem::remove(image_path, ec);
        continue;
      }
      // 存入数据库
      if (!database_->insertEmbedding(person_id, emb, image_path, qres.blur_score, recognizer_.modelTag())) {
        std::cerr << "[Enrollment] failed to store embedding." << std::endl;
        std::error_code ec;
        std::filesystem::remove(image_path, ec);
        continue;
      }

      enrolled_image_paths.push_back(image_path);
      captured++;
      std::cout << "[Enrollment] captured: " << captured << "/" << target_images << std::endl;
    }
  }

  // 录入结束，重载底库并释放资源
  embedding_store_->reload();
  camera_->stop();
  renderer_->closeWindow();
  if (cloud_client_ && cloud_client_->enabled() && !enrolled_image_paths.empty()) {
    std::cout << "[Enrollment] uploading " << enrolled_image_paths.size() << " samples to cloud gallery..." << std::endl;
    if (cloud_client_->enrollPerson(name, enrolled_image_paths, true)) {
      std::cout << "[Enrollment] cloud gallery updated for: " << name << std::endl;
    } else {
      std::cerr << "[Enrollment] cloud gallery update failed; local enrollment is still saved." << std::endl;
    }
  }
  std::cout << "[Enrollment] finished." << std::endl;
}

cv::Rect App::expandRect(const cv::Rect& rect, const cv::Size& image_size, float scale) {
  if (rect.width <= 0 || rect.height <= 0 || image_size.width <= 0 || image_size.height <= 0) {
    return {};
  }
  const float cx = static_cast<float>(rect.x) + static_cast<float>(rect.width) * 0.5F;
  const float cy = static_cast<float>(rect.y) + static_cast<float>(rect.height) * 0.5F;
  const float w = static_cast<float>(rect.width) * scale;
  const float h = static_cast<float>(rect.height) * scale;
  cv::Rect expanded(static_cast<int>(std::round(cx - w * 0.5F)),
                    static_cast<int>(std::round(cy - h * 0.5F)),
                    static_cast<int>(std::round(w)),
                    static_cast<int>(std::round(h)));
  expanded &= cv::Rect(0, 0, image_size.width, image_size.height);
  return expanded;
}

void App::submitCloudRequests(const FramePacket& frame, const std::vector<TrackState>& tracks) {
  if (!cloud_client_ || !cloud_client_->enabled() || frame.bgr.empty()) {
    return;
  }

  for (const auto& track : tracks) {
    const cv::Rect crop_rect = expandRect(track.box, frame.bgr.size(), config_.cloud.crop_scale);
    if (crop_rect.width <= 0 || crop_rect.height <= 0) {
      continue;
    }

    CloudAnalysisRequest request{};
    request.track_id = track.track_id;
    request.frame_id = frame.frame_id;
    request.ts_ms = frame.ts_ms;
    request.face_bgr = frame.bgr(crop_rect);
    request.local_known = track.identity.known;
    request.local_name = track.identity.name;
    request.local_conf_pct = track.identity.conf_pct;
    cloud_client_->submit(request);
  }
}

bool App::applyCloudResults(std::uint64_t now_ms) {
  if (!cloud_client_ || !cloud_client_->enabled()) {
    return false;
  }

  auto completed = cloud_client_->pollCompleted();
  if (completed.empty()) {
    return false;
  }

  auto emotion_gap = [](const EmotionResult& emotion) {
    std::array<float, 4> probs = emotion.grouped_probs;
    std::sort(probs.begin(), probs.end(), std::greater<float>());
    return probs[0] - probs[1];
  };

  for (auto& result : completed) {
    if (result.has_identity) {
      const bool known_enough = result.identity.known && result.identity.conf_pct >= config_.cloud.identity_min_confidence;
      const bool acceptable_unknown = !result.identity.known && config_.cloud.identity_apply_unknown;
      if (!known_enough && !acceptable_unknown) {
        if (config_.cloud.debug) {
          std::cout << "[CloudClient] reject identity track=" << result.track_id
                    << " name=" << result.identity.name
                    << " known=" << (result.identity.known ? 1 : 0)
                    << " conf=" << result.identity.conf_pct
                    << " min_conf=" << config_.cloud.identity_min_confidence << std::endl;
        }
        result.has_identity = false;
      }
    }

    if (!result.has_emotion) {
      continue;
    }
    const float gap = emotion_gap(result.emotion);
    const bool is_sad = result.emotion.label == EmotionLabel::Sad;
    const float min_confidence = is_sad ? config_.cloud.emotion_sad_min_confidence : config_.cloud.emotion_min_confidence;
    const float min_gap = is_sad ? config_.cloud.emotion_sad_min_gap : config_.cloud.emotion_min_gap;
    const bool confident = result.emotion.conf_pct >= min_confidence;
    const bool separated = gap >= min_gap;
    if (!confident || !separated) {
      if (config_.cloud.debug) {
        std::cout << "[CloudClient] reject emotion track=" << result.track_id
                  << " label=" << emotionToString(result.emotion.label)
                  << " conf=" << result.emotion.conf_pct
                  << " gap=" << gap
                  << " min_conf=" << min_confidence
                  << " min_gap=" << min_gap << std::endl;
      }
      result.has_emotion = false;
    }
  }

  if (config_.cloud.debug) {
    for (const auto& result : completed) {
      std::cout << "[CloudClient] completed track=" << result.track_id
                << " frame=" << result.frame_id
                << " identity=" << (result.has_identity ? result.identity.name : "-")
                << " emotion=" << (result.has_emotion ? emotionToString(result.emotion.label) : "-")
                << " apply_id=" << (config_.cloud.apply_identity ? 1 : 0)
                << " apply_emo=" << (config_.cloud.apply_emotion ? 1 : 0) << std::endl;
    }
  }

  if (!config_.cloud.apply_identity && !config_.cloud.apply_emotion) {
    return false;
  }

  track_manager_->applyExternalAnalyses(completed,
                                        now_ms,
                                        static_cast<std::uint64_t>(std::max(0, config_.cloud.result_ttl_ms)),
                                        config_.cloud.apply_identity,
                                        config_.cloud.apply_emotion);
  return true;
}

/**
 * @brief 处理【实时识别 + 情绪检测】流程
 *
 * 步骤：
 *  1. 启动相机
 *  2. 循环取帧 -> 交给 InferencePipeline 处理 -> 渲染结果
 *  3. 支持按 'q' 退出，'r' 重载底库
 */
void App::handleRecognition() {
  if (!camera_->start()) {
    std::cerr << "[Recognition] failed to start camera." << std::endl;
    return;
  }

  embedding_store_->reload();
  std::cout << "[Recognition] press q to quit, press r to reload local gallery." << std::endl;

  while (true) {
    FramePacket frame{};
    if (!camera_->getLatestFrame(frame, 200)) {
      continue;
    }

    applyCloudResults(frame.ts_ms);

    // 整帧送入推理管线，返回所有人脸的跟踪/识别/情绪结果
    RecognitionResult result = pipeline_->process(frame);
    submitCloudRequests(frame, result.tracks);
    if (applyCloudResults(frame.ts_ms)) {
      result.tracks = track_manager_->snapshot();
    }
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
      std::cout << "[Recognition] gallery reloaded." << std::endl;
    }
  }

  camera_->stop();
  renderer_->closeWindow();
}

/**
 * @brief 处理【删除人员】流程
 *
 * 列出数据库中所有人员，用户输入编号后确认删除，同时清理本地图片与底库缓存。
 */
void App::handleDeletePerson() {
  std::cout << "[Delete] choose delete scope:" << std::endl;
  std::cout << "1) Local only" << std::endl;
  std::cout << "2) Cloud only" << std::endl;
  std::cout << "3) Local + cloud" << std::endl;
  std::cout << "0) Cancel" << std::endl;
  std::cout << "Enter choice: ";

  std::string scope_text;
  std::getline(std::cin, scope_text);
  scope_text = trim(scope_text);
  if (scope_text.empty() || scope_text == "0") {
    std::cout << "[Delete] cancelled." << std::endl;
    return;
  }

  int scope = 0;
  try {
    scope = std::stoi(scope_text);
  } catch (...) {
    std::cout << "[Delete] invalid selection." << std::endl;
    return;
  }
  if (scope < 1 || scope > 3) {
    std::cout << "[Delete] invalid selection." << std::endl;
    return;
  }

  const bool delete_local = (scope == 1 || scope == 3);
  const bool delete_cloud = (scope == 2 || scope == 3);
  if (delete_cloud && (!cloud_client_ || !cloud_client_->enabled())) {
    if (!delete_local) {
      std::cout << "[Delete] cloud client is not enabled; cloud-only delete cannot continue." << std::endl;
      return;
    }
    std::cout << "[Delete] cloud client is not enabled; local delete can continue, cloud delete will be skipped." << std::endl;
  }

  std::string name;
  auto chooseNameFromList = [this](const std::vector<std::string>& people, const std::string& title, std::string* out_name) {
    if (people.empty() || out_name == nullptr) {
      return -1;
    }

    std::cout << title << std::endl;
    for (std::size_t i = 0; i < people.size(); ++i) {
      std::cout << (i + 1) << ") " << people[i] << std::endl;
    }
    std::cout << "Enter the number to delete (0 to cancel): ";

    std::string choice_text;
    std::getline(std::cin, choice_text);
    choice_text = trim(choice_text);
    if (choice_text.empty() || choice_text == "0") {
      std::cout << "[Delete] cancelled." << std::endl;
      return 0;
    }

    int choice = 0;
    try {
      choice = std::stoi(choice_text);
    } catch (...) {
      std::cout << "[Delete] invalid selection." << std::endl;
      return 0;
    }
    if (choice < 1 || choice > static_cast<int>(people.size())) {
      std::cout << "[Delete] invalid selection." << std::endl;
      return 0;
    }

    *out_name = people[static_cast<std::size_t>(choice - 1)];
    return 1;
  };

  if (delete_local) {
    const auto persons = database_->listPersons();
    if (persons.empty()) {
      std::cout << "[Delete] no person in local database." << std::endl;
      return;
    }

    const int selected = chooseNameFromList(persons, "[Delete] people in local database:", &name);
    if (selected != 1) {
      return;
    }
  } else {
    const auto cloud_people = (cloud_client_ && cloud_client_->enabled()) ? cloud_client_->listPeople() : std::vector<std::string>{};
    const int selected = chooseNameFromList(cloud_people, "[Delete] people in cloud gallery:", &name);
    if (selected == 0) {
      return;
    }
    if (selected < 0) {
      std::cout << "[Delete] cloud gallery list is empty or unavailable; enter cloud person name to delete: ";
      std::string raw_name;
      std::getline(std::cin, raw_name);
      name = file_store_->sanitizeName(trim(raw_name));
      if (name.empty()) {
        std::cout << "[Delete] invalid name." << std::endl;
        return;
      }
    }
  }

  const auto image_paths = delete_local ? file_store_->listPersonImages(name) : std::vector<std::string>{};

  std::cout << "[Delete] remove " << name << " from ";
  if (delete_local && delete_cloud) {
    std::cout << "local + cloud";
  } else if (delete_local) {
    std::cout << "local";
  } else {
    std::cout << "cloud";
  }
  if (!image_paths.empty()) {
    std::cout << " and " << image_paths.size() << " saved image(s)";
  }
  std::cout << "? (y/n): ";

  std::string yn;
  std::getline(std::cin, yn);
  yn = trim(yn);
  std::transform(yn.begin(), yn.end(), yn.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (yn != "y") {
    std::cout << "[Delete] cancelled." << std::endl;
    return;
  }

  if (delete_local) {
    // 先删数据库记录，再删本地文件
    if (!database_->deletePersonAndEmbeddings(name)) {
      std::cerr << "[Delete] failed to remove local database records for " << name << std::endl;
      return;
    }

    file_store_->removeFiles(image_paths);
    file_store_->removePersonDir(name);
    embedding_store_->reload();
    std::cout << "[Delete] removed local data for " << name << std::endl;
  }

  if (delete_cloud) {
    if (cloud_client_ && cloud_client_->enabled()) {
      if (cloud_client_->deletePerson(name)) {
        std::cout << "[Delete] cloud delete requested for " << name << std::endl;
      } else {
        std::cerr << "[Delete] failed to delete cloud data for " << name << std::endl;
      }
    } else {
      std::cerr << "[Delete] cloud client is disabled; cloud data was not changed." << std::endl;
    }
  }
}

/** @brief 去除字符串首尾空白字符（空格、制表、换行等） */
std::string App::trim(const std::string& s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  const auto begin = std::find_if_not(s.begin(), s.end(), is_space);
  if (begin == s.end()) {
    return "";
  }
  const auto end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
  return std::string(begin, end);
}

}  // namespace asdun
