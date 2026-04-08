#pragma once

#include <string>

#include "core/Types.hpp"

namespace asdun {

struct Track {
  int id{-1};
  cv::Rect box{};
  IdentityResult identity{};
  EmotionResult emotion{};
  int ttl{0};
  std::uint64_t last_frame_id{0};
  std::uint64_t last_update_ms{0};
};

}  // namespace asdun

