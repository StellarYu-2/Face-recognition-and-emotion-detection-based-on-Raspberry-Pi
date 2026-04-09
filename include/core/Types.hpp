#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace asdun {

enum class AppState {
  MainMenu = 0,
  EnrollInputName,
  EnrollCapture,
  Recognize,
  Exit
};

struct FramePacket {
  cv::Mat bgr;
  std::uint64_t ts_ms{0};
  std::uint64_t frame_id{0};
};

struct Detection {
  cv::Rect box;
  float det_score{0.0F};
};

struct IdentityResult {
  std::string name{"Unknown"};
  float distance{1.0F};
  float conf_pct{0.0F};
  bool known{false};
  int matched_sample_count{0};
  std::string debug_summary{};
};

enum class EmotionLabel {
  Neutral = 0,
  Happy,
  Surprise,
  Sad,
  Angry,
  Fear,
  Disgust,
  Contempt,
  Unknown
};

inline std::string emotionToString(EmotionLabel label) {
  switch (label) {
    case EmotionLabel::Neutral:
      return "Neutral";
    case EmotionLabel::Happy:
      return "Happy";
    case EmotionLabel::Surprise:
      return "Surprise";
    case EmotionLabel::Sad:
      return "Sad";
    case EmotionLabel::Angry:
      return "Angry";
    case EmotionLabel::Fear:
      return "Fear";
    case EmotionLabel::Disgust:
      return "Disgust";
    case EmotionLabel::Contempt:
      return "Contempt";
    default:
      return "Unknown";
  }
}

struct EmotionResult {
  EmotionLabel label{EmotionLabel::Unknown};
  float conf_pct{0.0F};
};

struct TrackState {
  int track_id{-1};
  cv::Rect box;
  IdentityResult identity{};
  EmotionResult emotion{};
  std::uint64_t last_update_ms{0};
};

struct RecognitionResult {
  std::vector<TrackState> tracks;
  std::uint64_t frame_id{0};
  std::uint64_t ts_ms{0};
};

struct StoredEmbedding {
  int person_id{0};
  std::string person_name;
  std::vector<float> embedding;
  std::string image_path;
  float quality_score{0.0F};
  std::string model_tag;
};

}  // namespace asdun
