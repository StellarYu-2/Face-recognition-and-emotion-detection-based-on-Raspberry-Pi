#include "tracking/TrackManager.hpp"

#include <algorithm>

namespace asdun {

TrackManager::TrackManager(int max_ttl, float iou_match_threshold)
    : max_ttl_(max_ttl), iou_match_threshold_(iou_match_threshold) {}

void TrackManager::updateWithDetections(const std::vector<Detection>& detections,
                                        const std::vector<IdentityResult>& identities,
                                        const std::vector<EmotionResult>& emotions,
                                        std::uint64_t frame_id,
                                        std::uint64_t ts_ms) {
  const std::size_t old_track_count = tracks_.size();
  std::vector<int> matched_track_for_det(detections.size(), -1);
  std::vector<bool> track_taken(old_track_count, false);

  for (std::size_t d = 0; d < detections.size(); ++d) {
    float best_iou = 0.0F;
    int best_track = -1;
    for (std::size_t t = 0; t < old_track_count; ++t) {
      if (track_taken[t]) {
        continue;
      }
      const float v = iou(detections[d].box, tracks_[t].box);
      if (v > iou_match_threshold_ && v > best_iou) {
        best_iou = v;
        best_track = static_cast<int>(t);
      }
    }
    if (best_track >= 0) {
      matched_track_for_det[d] = best_track;
      track_taken[static_cast<std::size_t>(best_track)] = true;
    }
  }

  for (std::size_t d = 0; d < detections.size(); ++d) {
    const int track_idx = matched_track_for_det[d];
    const IdentityResult new_id = (d < identities.size()) ? identities[d] : IdentityResult{};
    const EmotionResult new_emo = (d < emotions.size()) ? emotions[d] : EmotionResult{};

    if (track_idx >= 0) {
      Track& tr = tracks_[static_cast<std::size_t>(track_idx)];
      tr.box = detections[d].box;
      tr.last_frame_id = frame_id;
      tr.last_update_ms = ts_ms;
      tr.ttl = max_ttl_;

      if (new_id.known) {
        if (tr.identity.known && tr.identity.name == new_id.name) {
          tr.identity.conf_pct = ema(tr.identity.conf_pct, new_id.conf_pct, 0.35F);
          tr.identity.distance = ema(tr.identity.distance, new_id.distance, 0.35F);
        } else if (!tr.identity.known || (new_id.conf_pct > tr.identity.conf_pct + 8.0F)) {
          tr.identity = new_id;
        }
      } else if (!tr.identity.known) {
        tr.identity = new_id;
      }

      if (tr.emotion.label == new_emo.label) {
        tr.emotion.conf_pct = ema(tr.emotion.conf_pct, new_emo.conf_pct, 0.35F);
      } else if (new_emo.conf_pct > tr.emotion.conf_pct + 10.0F) {
        tr.emotion = new_emo;
      }
    } else {
      Track tr{};
      tr.id = next_track_id_++;
      tr.box = detections[d].box;
      tr.identity = new_id;
      tr.emotion = new_emo;
      tr.ttl = max_ttl_;
      tr.last_frame_id = frame_id;
      tr.last_update_ms = ts_ms;
      tracks_.push_back(std::move(tr));
    }
  }

  for (std::size_t t = 0; t < old_track_count; ++t) {
    if (!track_taken[t]) {
      tracks_[t].ttl -= 1;
      tracks_[t].last_frame_id = frame_id;
      tracks_[t].last_update_ms = ts_ms;
    }
  }

  pruneDeadTracks();
}

void TrackManager::tickWithoutDetections(std::uint64_t frame_id, std::uint64_t ts_ms) {
  for (auto& tr : tracks_) {
    tr.ttl -= 1;
    tr.last_frame_id = frame_id;
    tr.last_update_ms = ts_ms;
  }
  pruneDeadTracks();
}

void TrackManager::updateEmotionsByTrackOrder(const std::vector<EmotionResult>& emotions, std::uint64_t ts_ms) {
  const std::size_t n = std::min(emotions.size(), tracks_.size());
  for (std::size_t i = 0; i < n; ++i) {
    auto& tr = tracks_[i];
    const auto& e = emotions[i];
    if (tr.emotion.label == e.label) {
      tr.emotion.conf_pct = ema(tr.emotion.conf_pct, e.conf_pct, 0.35F);
    } else if (e.conf_pct > tr.emotion.conf_pct + 8.0F) {
      tr.emotion = e;
    }
    tr.last_update_ms = ts_ms;
  }
}

std::vector<TrackState> TrackManager::snapshot() const {
  std::vector<TrackState> out;
  out.reserve(tracks_.size());
  for (const auto& tr : tracks_) {
    TrackState s{};
    s.track_id = tr.id;
    s.box = tr.box;
    s.identity = tr.identity;
    s.emotion = tr.emotion;
    s.last_update_ms = tr.last_update_ms;
    out.push_back(std::move(s));
  }
  return out;
}

float TrackManager::iou(const cv::Rect& a, const cv::Rect& b) {
  const int x_left = std::max(a.x, b.x);
  const int y_top = std::max(a.y, b.y);
  const int x_right = std::min(a.x + a.width, b.x + b.width);
  const int y_bottom = std::min(a.y + a.height, b.y + b.height);
  if (x_right <= x_left || y_bottom <= y_top) {
    return 0.0F;
  }
  const float inter = static_cast<float>((x_right - x_left) * (y_bottom - y_top));
  const float union_area = static_cast<float>(a.area() + b.area()) - inter;
  if (union_area <= 1e-6F) {
    return 0.0F;
  }
  return inter / union_area;
}

float TrackManager::ema(float old_value, float new_value, float alpha) {
  const float a = std::clamp(alpha, 0.0F, 1.0F);
  return (1.0F - a) * old_value + a * new_value;
}

void TrackManager::pruneDeadTracks() {
  tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& tr) { return tr.ttl <= 0; }),
      tracks_.end());
}

}  // namespace asdun
