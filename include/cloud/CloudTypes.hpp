#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/Types.hpp"

namespace asdun {

struct CloudClientConfig {
  bool enabled{false};
  std::string server_url{"http://127.0.0.1:8000"};
  std::vector<std::string> server_urls{};
  std::string health_check_path{"/health"};
  int connect_timeout_ms{100};
  int timeout_ms{300};
  int min_interval_ms{600};
  int max_queue_size{2};
  int jpeg_quality{85};
  int crop_size{256};
  float crop_scale{1.25F};
  int result_ttl_ms{1200};
  float identity_min_confidence{55.0F};
  bool identity_apply_unknown{false};
  float emotion_min_confidence{55.0F};
  float emotion_min_gap{0.12F};
  float emotion_sad_min_confidence{35.0F};
  float emotion_sad_min_gap{0.05F};
  bool apply_identity{false};
  bool apply_emotion{false};
  bool debug{false};
};

struct CloudAnalysisRequest {
  int track_id{-1};
  std::uint64_t frame_id{0};
  std::uint64_t ts_ms{0};
  cv::Mat face_bgr{};
  bool local_known{false};
  std::string local_name{"Unknown"};
  float local_conf_pct{0.0F};
};

}  // namespace asdun
