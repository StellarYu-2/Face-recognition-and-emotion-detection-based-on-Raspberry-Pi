#pragma once

#include <array>
#include <string>

#include "core/Types.hpp"

namespace asdun {

struct Track {
  int id{-1};
  cv::Rect box{};
  IdentityResult identity{};
  IdentityResult pending_identity{};
  EmotionResult emotion{};
  EmotionResult pending_emotion{};
  std::array<float, 4> smoothed_emotion_probs{{0.0F, 0.0F, 0.0F, 0.0F}};
  bool emotion_initialized{false};
  int ttl{0};
  int detection_hits{0};
  int pending_identity_hits{0};
  int pending_emotion_hits{0};
  int unknown_identity_streak{0};
  float velocity_x{0.0F};
  float velocity_y{0.0F};
  float velocity_w{0.0F};
  float velocity_h{0.0F};
  std::uint64_t last_confirmed_identity_ms{0};
  std::uint64_t last_recognition_frame_id{0};
  std::uint64_t last_recognition_ms{0};
  float last_recognition_blur_score{0.0F};
  int last_recognition_face_size{0};
  std::uint64_t last_emotion_ms{0};
  std::uint64_t last_frame_id{0};
  std::uint64_t last_update_ms{0};
};

}  // namespace asdun
