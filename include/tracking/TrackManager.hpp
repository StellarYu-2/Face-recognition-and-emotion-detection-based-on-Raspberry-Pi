#pragma once

#include <unordered_map>
#include <vector>

#include "core/Types.hpp"
#include "tracking/Track.hpp"

namespace asdun {

class TrackManager {
 public:
  TrackManager(int max_ttl, float iou_match_threshold);

  void updateWithDetections(const std::vector<Detection>& detections,
                            const std::vector<IdentityResult>& identities,
                            const std::vector<EmotionResult>& emotions,
                            std::uint64_t frame_id,
                            std::uint64_t ts_ms);

  void tickWithoutDetections(std::uint64_t frame_id, std::uint64_t ts_ms);
  void updateEmotionsByTrackOrder(const std::vector<EmotionResult>& emotions, std::uint64_t ts_ms);

  std::vector<TrackState> snapshot() const;

 private:
  static float iou(const cv::Rect& a, const cv::Rect& b);
  static float ema(float old_value, float new_value, float alpha);
  void pruneDeadTracks();

  int next_track_id_{1};
  int max_ttl_{10};
  float iou_match_threshold_{0.3F};
  std::vector<Track> tracks_{};
};

}  // namespace asdun

