#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace asdun {

/**
 * @brief 应用程序运行状态枚举
 * 用于主循环中的状态机切换，区分菜单/录入/识别等不同场景。
 */
enum class AppState {
  MainMenu = 0,       /**< 主菜单 */
  EnrollInputName,    /**< 录入流程：等待输入姓名 */
  EnrollCapture,      /**< 录入流程：采集人脸并提取特征 */
  Recognize,          /**< 正常识别模式 */
  Exit                /**< 退出应用 */
};

/**
 * @brief 图像帧数据包
 * 从摄像头采集的一帧原始图像，附带时间戳与帧序号。
 */
struct FramePacket {
  cv::Mat bgr;              /**< BGR 格式图像矩阵 */
  std::uint64_t ts_ms{0};   /**< 采集时间戳（毫秒） */
  std::uint64_t frame_id{0};/**< 全局帧序号，用于同步与去重 */
};

/**
 * @brief 五点人脸关键点
 * 通常对应双眼中心、鼻尖、左右嘴角，用于人脸对齐与质量评估。
 */
struct FivePointLandmarks {
  std::array<cv::Point2f, 5> points{}; /**< 5 个二维关键点坐标 */
  bool valid{false};                   /**< 关键点是否有效/检测成功 */
};

/**
 * @brief 单个人脸检测结果
 * 包含人脸位置、置信度及关键点信息。
 */
struct Detection {
  cv::Rect box;              /**< 人脸包围盒（像素坐标） */
  float det_score{0.0F};     /**< 检测置信度，值越大越可信 */
  FivePointLandmarks landmarks{}; /**< 关联的 5 点人脸关键点 */
};

/**
 * @brief 身份识别结果
 * 描述单张人脸与底库比对后的匹配信息，以及输入质量指标。
 */
struct IdentityResult {
  std::string name{"Unknown"}; /**< 匹配到的姓名，未匹配则为 Unknown */
  float distance{1.0F};        /**< 特征空间距离（如余弦距离），越小越相似 */
  float margin{0.0F};          /**< 与次优候选的距离差，越大区分度越高 */
  float conf_pct{0.0F};        /**< 综合置信度百分比（0~100），用于 UI 展示 */
  bool known{false};           /**< 是否成功匹配到底库已知人员 */
  bool measured{false};        /**< 是否已完成特征比对计算 */
  bool attempted{false};       /**< 是否已尝试执行识别（防止重复计算） */
  float input_blur_score{0.0F};/**< 输入人脸模糊度评分，值大表示模糊/质量差 */
  int input_min_face_size{0};  /**< 检测框最小边长（像素），用于判断人脸是否过小 */
  int matched_sample_count{0}; /**< 该人员底库中参与匹配的样本数量 */
  std::string debug_summary{}; /**< 调试信息摘要，供开发排查使用 */
};

/**
 * @brief 情绪标签枚举
 * 模型输出 8 种基本情绪，Unknown 表示未识别或失败。
 */
enum class EmotionLabel {
  Neutral = 0, /**< 中性 */
  Happy,       /**< 开心 */
  Surprise,    /**< 惊讶 */
  Sad,         /**< 悲伤 */
  Angry,       /**< 愤怒 */
  Fear,        /**< 恐惧 */
  Disgust,     /**< 厌恶 */
  Contempt,    /**< 轻蔑 */
  Unknown      /**< 未知/未识别 */
};

/**
 * @brief 将情绪标签映射为 UI 显示的字符串
 *
 * 为简化前端展示，将 8 类情绪归并为 4 类：
 * Calm（平静）、Happy（开心）、Sad（悲伤）、Angry（愤怒）。
 * @param label 情绪标签
 * @return 对应的 UI 显示名称
 */
inline std::string emotionToString(EmotionLabel label) {
  switch (label) {
    case EmotionLabel::Neutral:
      return "Calm";
    case EmotionLabel::Happy:
      return "Happy";
    case EmotionLabel::Sad:
      return "Sad";
    case EmotionLabel::Angry:
      return "Angry";
    case EmotionLabel::Surprise:
      return "Happy";   /**< 惊讶映射为开心，用于简化展示 */
    case EmotionLabel::Fear:
      return "Sad";     /**< 恐惧映射为悲伤 */
    case EmotionLabel::Disgust:
      return "Angry";   /**< 厌恶映射为愤怒 */
    case EmotionLabel::Contempt:
      return "Angry";   /**< 轻蔑映射为愤怒 */
    default:
      return "Unknown";
  }
}

/**
 * @brief 情绪分析结果
 * 包含单个人脸的情绪识别信息及分组概率。
 */
struct EmotionResult {
  EmotionLabel label{EmotionLabel::Unknown}; /**< 识别到的情绪标签 */
  float conf_pct{0.0F};                      /**< 情绪置信度百分比 */
  bool attempted{false};                     /**< 是否已尝试执行情绪识别 */
  std::array<float, 4> grouped_probs{{0.0F, 0.0F, 0.0F, 0.0F}};
  /**< 归并后的 4 类情绪概率，顺序对应 Calm/Happy/Sad/Angry */
  std::string debug_summary{};               /**< 调试信息摘要 */
};

/**
 * @brief 单条跟踪状态
 * 将检测框、身份与情绪绑定到同一个跟踪 ID，用于跨帧关联同一人脸。
 */
struct TrackState {
  int track_id{-1};            /**< 跟踪器分配的唯一 ID，-1 表示未跟踪 */
  cv::Rect box;                /**< 当前帧人脸包围盒 */
  IdentityResult identity{};   /**< 关联的身份识别结果 */
  EmotionResult emotion{};     /**< 关联的情绪分析结果 */
  std::uint64_t last_update_ms{0}; /**< 最后一次更新的时间戳（毫秒） */
};

/**
 * @brief 外部分析结果
 * 用于承接云端或其他异步推理源返回的身份/情绪结果。Tracking 层只关心
 * “某个 track 的新分析结果”，不需要知道 HTTP、云端模型等实现细节。
 */
struct ExternalTrackAnalysis {
  int track_id{-1};                /**< 目标跟踪 ID */
  std::uint64_t frame_id{0};       /**< 结果对应的源帧 */
  std::uint64_t ts_ms{0};          /**< 结果对应的源时间戳 */
  bool has_identity{false};        /**< 是否携带身份结果 */
  bool has_emotion{false};         /**< 是否携带情绪结果 */
  IdentityResult identity{};       /**< 外部身份结果 */
  EmotionResult emotion{};         /**< 外部情绪结果 */
  std::string source{"external"};  /**< 结果来源，便于调试 */
};

/**
 * @brief 整帧识别结果
 * 汇总一帧图像中所有被跟踪人脸的状态。
 */
struct RecognitionResult {
  std::vector<TrackState> tracks; /**< 当前帧所有人脸的跟踪状态列表 */
  std::uint64_t frame_id{0};      /**< 对应帧序号，与 FramePacket.frame_id 对齐 */
  std::uint64_t ts_ms{0};         /**< 对应时间戳（毫秒） */
};

/**
 * @brief 已持久化的人脸特征
 * 对应底库中一条注册记录，用于 1:N 比对。
 */
struct StoredEmbedding {
  int person_id{0};            /**< 人员唯一编号 */
  std::string person_name;     /**< 人员姓名 */
  std::vector<float> embedding;/**< 人脸特征向量（如 128/512 维浮点数组） */
  std::string image_path;      /**< 注册源图像路径，用于管理与回显 */
  float quality_score{0.0F};   /**< 人脸质量评分，用于过滤低质量注册 */
  std::string model_tag;       /**< 提取该特征所用模型的版本标识 */
};

}  // namespace asdun
