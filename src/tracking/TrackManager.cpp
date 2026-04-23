#include "tracking/TrackManager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace asdun {

namespace {

constexpr int kIdentityConfirmHits = 2;
constexpr int kIdentitySwitchConfirmHits = 4;
constexpr int kIdentityUnknownClearHits = 8;
constexpr int kTrackConfirmHits = 3;
constexpr int kTrackMaxHits = 6;
constexpr std::uint64_t kIdentityHoldMs = 2600;
constexpr float kIdentityHoldStrongConf = 82.0F;
constexpr float kIdentitySwitchMinMargin = 0.09F;
constexpr float kIdentitySwitchMinDistanceGain = 0.05F;
constexpr float kIdentitySwitchConfBoost = 10.0F;
constexpr float kIdentityOutlierDistance = 1.28F;
constexpr float kIdentityOutlierDistanceGain = 0.22F;
constexpr float kBoxSmoothAlpha = 0.38F;
constexpr float kBoxFastFollowAlpha = 0.62F;
constexpr float kVelocityAlpha = 0.42F;
constexpr float kVelocityDecay = 0.74F;
constexpr float kEmotionStableAlpha = 0.34F;
constexpr float kEmotionSwitchAlpha = 0.62F;
constexpr float kEmotionStrongSwitchAlpha = 0.74F;
constexpr float kEmotionKeepMargin = 0.035F;

cv::Rect smoothRect(const cv::Rect& old_rect, const cv::Rect& new_rect, float alpha) {
  if (old_rect.width <= 0 || old_rect.height <= 0) {
    return new_rect;
  }
  const float a = std::clamp(alpha, 0.0F, 1.0F);
  auto blend = [a](int old_v, int new_v) {
    return static_cast<int>(std::lround((1.0F - a) * static_cast<float>(old_v) + a * static_cast<float>(new_v)));
  };

  cv::Rect out(blend(old_rect.x, new_rect.x),
               blend(old_rect.y, new_rect.y),
               blend(old_rect.width, new_rect.width),
               blend(old_rect.height, new_rect.height));
  if (out.width <= 0 || out.height <= 0) {
    return new_rect;
  }
  return out;
}

cv::Rect predictRect(const cv::Rect& rect, float dx, float dy, float dw, float dh) {
  if (rect.width <= 0 || rect.height <= 0) {
    return rect;
  }
  cv::Rect out(static_cast<int>(std::lround(static_cast<float>(rect.x) + dx)),
               static_cast<int>(std::lround(static_cast<float>(rect.y) + dy)),
               static_cast<int>(std::lround(static_cast<float>(rect.width) + dw)),
               static_cast<int>(std::lround(static_cast<float>(rect.height) + dh)));
  out.width = std::max(out.width, 12);
  out.height = std::max(out.height, 12);
  return out;
}

float clampStep(float value, float limit) {
  return std::clamp(value, -limit, limit);
}

float blendValue(float old_value, float new_value, float alpha) {
  const float a = std::clamp(alpha, 0.0F, 1.0F);
  return (1.0F - a) * old_value + a * new_value;
}

float motionRatio(const cv::Rect& old_rect, const cv::Rect& new_rect) {
  if (old_rect.width <= 0 || old_rect.height <= 0 || new_rect.width <= 0 || new_rect.height <= 0) {
    return 0.0F;
  }
  const float old_cx = static_cast<float>(old_rect.x) + static_cast<float>(old_rect.width) * 0.5F;
  const float old_cy = static_cast<float>(old_rect.y) + static_cast<float>(old_rect.height) * 0.5F;
  const float new_cx = static_cast<float>(new_rect.x) + static_cast<float>(new_rect.width) * 0.5F;
  const float new_cy = static_cast<float>(new_rect.y) + static_cast<float>(new_rect.height) * 0.5F;
  const float dx = new_cx - old_cx;
  const float dy = new_cy - old_cy;
  const float scale = static_cast<float>(std::max({old_rect.width, old_rect.height, new_rect.width, new_rect.height}));
  if (scale <= 1e-6F) {
    return 0.0F;
  }
  return std::sqrt(dx * dx + dy * dy) / scale;
}

void updateTrackVelocity(Track& tr, const cv::Rect& old_rect, const cv::Rect& new_rect) {
  const float motion_limit = static_cast<float>(std::max(old_rect.width, old_rect.height)) * 0.22F;
  const float size_limit = static_cast<float>(std::max(old_rect.width, old_rect.height)) * 0.10F;
  tr.velocity_x =
      blendValue(tr.velocity_x, clampStep(static_cast<float>(new_rect.x - old_rect.x), motion_limit), kVelocityAlpha);
  tr.velocity_y =
      blendValue(tr.velocity_y, clampStep(static_cast<float>(new_rect.y - old_rect.y), motion_limit), kVelocityAlpha);
  tr.velocity_w =
      blendValue(tr.velocity_w, clampStep(static_cast<float>(new_rect.width - old_rect.width), size_limit), 0.20F);
  tr.velocity_h =
      blendValue(tr.velocity_h, clampStep(static_cast<float>(new_rect.height - old_rect.height), size_limit), 0.20F);
}

void predictTrack(Track& tr) {
  if (tr.detection_hits < kTrackConfirmHits) {
    return;
  }
  const float motion_limit = static_cast<float>(std::max(tr.box.width, tr.box.height)) * 0.18F;
  const float size_limit = static_cast<float>(std::max(tr.box.width, tr.box.height)) * 0.08F;
  tr.box = predictRect(tr.box,
                       clampStep(tr.velocity_x, motion_limit),
                       clampStep(tr.velocity_y, motion_limit),
                       clampStep(tr.velocity_w, size_limit),
                       clampStep(tr.velocity_h, size_limit));
  tr.velocity_x *= kVelocityDecay;
  tr.velocity_y *= kVelocityDecay;
  tr.velocity_w *= kVelocityDecay;
  tr.velocity_h *= kVelocityDecay;
}

float sizeSimilarity(const cv::Rect& a, const cv::Rect& b) {
  const float a_area = static_cast<float>(a.area());
  const float b_area = static_cast<float>(b.area());
  if (a_area <= 1e-6F || b_area <= 1e-6F) {
    return 0.0F;
  }
  return std::min(a_area, b_area) / std::max(a_area, b_area);
}

int emotionIndex(EmotionLabel label) {
  switch (label) {
    case EmotionLabel::Neutral:
      return 0;
    case EmotionLabel::Happy:
    case EmotionLabel::Surprise:
      return 1;
    case EmotionLabel::Sad:
    case EmotionLabel::Fear:
      return 2;
    case EmotionLabel::Angry:
    case EmotionLabel::Disgust:
    case EmotionLabel::Contempt:
      return 3;
    default:
      return -1;
  }
}

EmotionLabel indexToEmotionLabel(int idx) {
  switch (idx) {
    case 0:
      return EmotionLabel::Neutral;
    case 1:
      return EmotionLabel::Happy;
    case 2:
      return EmotionLabel::Sad;
    case 3:
      return EmotionLabel::Angry;
    default:
      return EmotionLabel::Unknown;
  }
}

std::array<float, 4> normalizeEmotionScores(std::array<float, 4> scores) {
  for (float& score : scores) {
    score = std::max(0.0F, score);
  }
  const float sum = std::accumulate(scores.begin(), scores.end(), 0.0F);
  if (sum <= 1e-6F) {
    return {{0.0F, 0.0F, 0.0F, 0.0F}};
  }
  for (float& score : scores) {
    score /= sum;
  }
  return scores;
}

std::array<float, 4> emotionScoresFromResult(const EmotionResult& emo) {
  auto scores = normalizeEmotionScores(emo.grouped_probs);
  const float sum = std::accumulate(scores.begin(), scores.end(), 0.0F);
  if (sum > 1e-6F) {
    return scores;
  }

  const int idx = emotionIndex(emo.label);
  if (idx < 0) {
    return {{0.0F, 0.0F, 0.0F, 0.0F}};
  }

  const float winner = std::clamp(emo.conf_pct / 100.0F, 0.55F, 0.99F);
  const float rest = (1.0F - winner) / 3.0F;
  scores.fill(rest);
  scores[static_cast<std::size_t>(idx)] = winner;
  return normalizeEmotionScores(scores);
}

int dominantEmotionIndex(const std::array<float, 4>& scores) {
  const auto it = std::max_element(scores.begin(), scores.end());
  if (it == scores.end() || *it <= 1e-6F) {
    return -1;
  }
  return static_cast<int>(std::distance(scores.begin(), it));
}

float dominantEmotionScore(const std::array<float, 4>& scores) {
  const int idx = dominantEmotionIndex(scores);
  return (idx >= 0) ? scores[static_cast<std::size_t>(idx)] : 0.0F;
}

void updateEmotionState(Track& tr, const EmotionResult& new_emo) {
  const auto raw_scores = emotionScoresFromResult(new_emo);
  if (dominantEmotionIndex(raw_scores) < 0) {
    return;
  }

  if (!tr.emotion_initialized) {
    tr.smoothed_emotion_probs = raw_scores;
    tr.emotion = new_emo;
    tr.emotion.label = indexToEmotionLabel(dominantEmotionIndex(raw_scores));
    tr.emotion.conf_pct = std::clamp(dominantEmotionScore(raw_scores) * 100.0F, 0.0F, 99.0F);
    tr.emotion.grouped_probs = raw_scores;
    tr.emotion_initialized = true;
    tr.pending_emotion = EmotionResult{};
    tr.pending_emotion_hits = 0;
    return;
  }

  const int current_idx = dominantEmotionIndex(tr.smoothed_emotion_probs);
  const int raw_idx = dominantEmotionIndex(raw_scores);
  float alpha = (raw_idx == current_idx) ? kEmotionStableAlpha : kEmotionSwitchAlpha;
  if (raw_idx != current_idx && raw_idx >= 0 && raw_scores[static_cast<std::size_t>(raw_idx)] >= 0.55F) {
    alpha = kEmotionStrongSwitchAlpha;
  }

  for (std::size_t i = 0; i < tr.smoothed_emotion_probs.size(); ++i) {
    tr.smoothed_emotion_probs[i] = blendValue(tr.smoothed_emotion_probs[i], raw_scores[i], alpha);
  }
  tr.smoothed_emotion_probs = normalizeEmotionScores(tr.smoothed_emotion_probs);

  int display_idx = dominantEmotionIndex(tr.smoothed_emotion_probs);
  if (display_idx < 0) {
    return;
  }

  if (current_idx >= 0 && display_idx != current_idx) {
    const float best_score = tr.smoothed_emotion_probs[static_cast<std::size_t>(display_idx)];
    const float current_score = tr.smoothed_emotion_probs[static_cast<std::size_t>(current_idx)];
    if (best_score < current_score + kEmotionKeepMargin && raw_idx != display_idx) {
      display_idx = current_idx;
    }
  }

  tr.emotion.label = indexToEmotionLabel(display_idx);
  tr.emotion.conf_pct =
      std::clamp(tr.smoothed_emotion_probs[static_cast<std::size_t>(display_idx)] * 100.0F, 0.0F, 99.0F);
  tr.emotion.grouped_probs = tr.smoothed_emotion_probs;

  std::ostringstream debug;
  if (!new_emo.debug_summary.empty()) {
    debug << new_emo.debug_summary << " ";
  }
  debug << "smooth_calm=" << tr.smoothed_emotion_probs[0] << " smooth_happy=" << tr.smoothed_emotion_probs[1]
        << " smooth_sad=" << tr.smoothed_emotion_probs[2] << " smooth_angry=" << tr.smoothed_emotion_probs[3]
        << " alpha=" << alpha;
  tr.emotion.debug_summary = debug.str();
}

}  // namespace

TrackManager::TrackManager(int max_ttl, float iou_match_threshold)
    : max_ttl_(max_ttl), iou_match_threshold_(iou_match_threshold) {}

std::vector<int> TrackManager::matchDetectionsToTracks(const std::vector<Detection>& detections,
                                                       const std::vector<Track>& tracks,
                                                       float iou_match_threshold) {
  std::vector<int> matched_track_for_det(detections.size(), -1);
  std::vector<bool> track_taken(tracks.size(), false);

  for (std::size_t d = 0; d < detections.size(); ++d) {
    float best_iou = 0.0F;
    float best_center_ratio = std::numeric_limits<float>::max();
    int best_track = -1;
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      if (track_taken[t]) {
        continue;
      }
      const float v = iou(detections[d].box, tracks[t].box);
      const float center_ratio = centerDistanceRatio(detections[d].box, tracks[t].box);
      const float size_sim = sizeSimilarity(detections[d].box, tracks[t].box);
      const bool iou_match = (v > iou_match_threshold);
      const bool center_match = (center_ratio < 0.24F) && (size_sim > 0.55F);
      if (iou_match && v > best_iou) {
        best_iou = v;
        best_center_ratio = center_ratio;
        best_track = static_cast<int>(t);
      } else if (!iou_match && center_match && best_track < 0 && center_ratio < best_center_ratio) {
        best_center_ratio = center_ratio;
        best_track = static_cast<int>(t);
      }
    }
    if (best_track >= 0) {
      matched_track_for_det[d] = best_track;
      track_taken[static_cast<std::size_t>(best_track)] = true;
    }
  }

  return matched_track_for_det;
}

std::vector<int> TrackManager::previewMatches(const std::vector<Detection>& detections) const {
  return matchDetectionsToTracks(detections, tracks_, iou_match_threshold_);
}

const Track* TrackManager::getTrackByIndex(int index) const {
  if (index < 0 || index >= static_cast<int>(tracks_.size())) {
    return nullptr;
  }
  return &tracks_[static_cast<std::size_t>(index)];
}

void TrackManager::updateWithDetections(const std::vector<Detection>& detections,
                                        const std::vector<IdentityResult>& identities,
                                        const std::vector<EmotionResult>& emotions,
                                        std::uint64_t frame_id,
                                        std::uint64_t ts_ms) {
  const std::size_t old_track_count = tracks_.size();
  const std::vector<int> matched_track_for_det = matchDetectionsToTracks(detections, tracks_, iou_match_threshold_);
  std::vector<bool> track_taken(old_track_count, false);
  for (const int matched_track : matched_track_for_det) {
    if (matched_track >= 0) {
      track_taken[static_cast<std::size_t>(matched_track)] = true;
    }
  }

  for (std::size_t d = 0; d < detections.size(); ++d) {
    const int track_idx = matched_track_for_det[d];
    const IdentityResult new_id = (d < identities.size()) ? identities[d] : IdentityResult{};
    const EmotionResult new_emo = (d < emotions.size()) ? emotions[d] : EmotionResult{};

    if (track_idx >= 0) {
      Track& tr = tracks_[static_cast<std::size_t>(track_idx)];
      const cv::Rect old_box = tr.box;
      const float alpha = motionRatio(tr.box, detections[d].box) > 0.12F ? kBoxFastFollowAlpha : kBoxSmoothAlpha;
      tr.box = smoothRect(tr.box, detections[d].box, alpha);
      updateTrackVelocity(tr, old_box, tr.box);
      tr.last_frame_id = frame_id;
      tr.last_update_ms = ts_ms;
      tr.ttl = max_ttl_;
      tr.detection_hits = std::min(tr.detection_hits + 1, kTrackMaxHits);
      if (new_id.attempted) {
        tr.last_recognition_frame_id = frame_id;
        tr.last_recognition_ms = ts_ms;
        tr.last_recognition_blur_score = new_id.input_blur_score;
        tr.last_recognition_face_size = new_id.input_min_face_size;
      }
      if (new_emo.attempted) {
        tr.last_emotion_ms = ts_ms;
      }

      if (!new_id.measured) {
        tr.unknown_identity_streak = 0;
      } else if (new_id.known) {
        tr.unknown_identity_streak = 0;
        tr.last_confirmed_identity_ms = ts_ms;
        if (tr.identity.known && tr.identity.name == new_id.name) {
          tr.identity.conf_pct = ema(tr.identity.conf_pct, new_id.conf_pct, 0.35F);
          tr.identity.distance = ema(tr.identity.distance, new_id.distance, 0.35F);
          tr.identity.margin = ema(tr.identity.margin, new_id.margin, 0.35F);
          tr.identity.measured = true;
          tr.identity.matched_sample_count = new_id.matched_sample_count;
          tr.identity.debug_summary = new_id.debug_summary;
          tr.pending_identity = IdentityResult{};
          tr.pending_identity_hits = 0;
        } else {
          if (tr.pending_identity.known && tr.pending_identity.name == new_id.name) {
            tr.pending_identity.conf_pct = ema(tr.pending_identity.conf_pct, new_id.conf_pct, 0.35F);
            tr.pending_identity.distance = ema(tr.pending_identity.distance, new_id.distance, 0.35F);
            tr.pending_identity.margin = ema(tr.pending_identity.margin, new_id.margin, 0.35F);
            tr.pending_identity.matched_sample_count = new_id.matched_sample_count;
            tr.pending_identity.debug_summary = new_id.debug_summary;
            tr.pending_identity_hits += 1;
          } else {
            tr.pending_identity = new_id;
            tr.pending_identity_hits = 1;
          }

          bool accept_pending = false;
          if (!tr.identity.known) {
            accept_pending = (tr.pending_identity_hits >= kIdentityConfirmHits);
          } else {
            const bool enough_hits = tr.pending_identity_hits >= kIdentitySwitchConfirmHits;
            const bool margin_ok = tr.pending_identity.margin >= kIdentitySwitchMinMargin;
            const bool distance_gain_ok = (tr.identity.distance - tr.pending_identity.distance) >= kIdentitySwitchMinDistanceGain;
            const bool conf_boost_ok = tr.pending_identity.conf_pct >= tr.identity.conf_pct + kIdentitySwitchConfBoost;
            accept_pending = enough_hits && (margin_ok || distance_gain_ok || conf_boost_ok);
          }

          if (accept_pending) {
            tr.identity = tr.pending_identity;
            tr.last_confirmed_identity_ms = ts_ms;
            tr.pending_identity = IdentityResult{};
            tr.pending_identity_hits = 0;
          }
        }
      } else if (!tr.identity.known) {
        tr.identity = new_id;
        tr.pending_identity = IdentityResult{};
        tr.pending_identity_hits = 0;
      } else {
        tr.pending_identity = IdentityResult{};
        tr.pending_identity_hits = 0;
        const bool hold_recent_identity =
            tr.last_confirmed_identity_ms > 0 && ts_ms > tr.last_confirmed_identity_ms &&
            (ts_ms - tr.last_confirmed_identity_ms) <= kIdentityHoldMs && tr.identity.conf_pct >= kIdentityHoldStrongConf;
        const bool catastrophic_outlier =
            new_id.measured && !new_id.known &&
            (new_id.distance >= kIdentityOutlierDistance ||
             new_id.distance >= (tr.identity.distance + kIdentityOutlierDistanceGain));
        if (hold_recent_identity || catastrophic_outlier) {
          tr.unknown_identity_streak = std::max(0, tr.unknown_identity_streak - 1);
        } else {
          tr.unknown_identity_streak += 1;
        }
        if (tr.unknown_identity_streak >= kIdentityUnknownClearHits) {
          tr.identity = new_id;
          tr.unknown_identity_streak = 0;
        }
      }

      updateEmotionState(tr, new_emo);
    } else {
      Track tr{};
      tr.id = next_track_id_++;
      tr.box = detections[d].box;
      tr.identity = new_id;
      tr.emotion = EmotionResult{};
      tr.ttl = max_ttl_;
      tr.pending_identity = IdentityResult{};
      tr.pending_emotion = EmotionResult{};
      tr.smoothed_emotion_probs = {{0.0F, 0.0F, 0.0F, 0.0F}};
      tr.emotion_initialized = false;
      tr.detection_hits = 1;
      tr.pending_identity_hits = 0;
      tr.pending_emotion_hits = 0;
      tr.unknown_identity_streak = 0;
      tr.velocity_x = 0.0F;
      tr.velocity_y = 0.0F;
      tr.velocity_w = 0.0F;
      tr.velocity_h = 0.0F;
      tr.last_confirmed_identity_ms = new_id.known ? ts_ms : 0;
      if (new_id.attempted) {
        tr.last_recognition_frame_id = frame_id;
        tr.last_recognition_ms = ts_ms;
        tr.last_recognition_blur_score = new_id.input_blur_score;
        tr.last_recognition_face_size = new_id.input_min_face_size;
      }
      if (new_emo.attempted) {
        tr.last_emotion_ms = ts_ms;
      }
      tr.last_frame_id = frame_id;
      tr.last_update_ms = ts_ms;
      updateEmotionState(tr, new_emo);
      tracks_.push_back(std::move(tr));
    }
  }

  for (std::size_t t = 0; t < old_track_count; ++t) {
    if (!track_taken[t]) {
      predictTrack(tracks_[t]);
      tracks_[t].ttl -= 1;
      tracks_[t].detection_hits = std::max(0, tracks_[t].detection_hits - 1);
      tracks_[t].last_frame_id = frame_id;
      tracks_[t].last_update_ms = ts_ms;
    }
  }

  pruneDeadTracks();
}

void TrackManager::tickWithoutDetections(std::uint64_t frame_id, std::uint64_t ts_ms) {
  for (auto& tr : tracks_) {
    predictTrack(tr);
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
    updateEmotionState(tr, e);
    if (e.attempted) {
      tr.last_emotion_ms = ts_ms;
    }
    tr.last_update_ms = ts_ms;
  }
}

std::vector<TrackState> TrackManager::snapshot() const {
  std::vector<TrackState> out;
  out.reserve(tracks_.size());
  for (const auto& tr : tracks_) {
    if (tr.detection_hits < kTrackConfirmHits) {
      continue;
    }
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

float TrackManager::centerDistanceRatio(const cv::Rect& a, const cv::Rect& b) {
  const float ax = static_cast<float>(a.x) + static_cast<float>(a.width) * 0.5F;
  const float ay = static_cast<float>(a.y) + static_cast<float>(a.height) * 0.5F;
  const float bx = static_cast<float>(b.x) + static_cast<float>(b.width) * 0.5F;
  const float by = static_cast<float>(b.y) + static_cast<float>(b.height) * 0.5F;

  const float dx = ax - bx;
  const float dy = ay - by;
  const float dist = std::sqrt(dx * dx + dy * dy);
  const float scale = static_cast<float>(std::max({a.width, a.height, b.width, b.height}));
  if (scale <= 1e-6F) {
    return std::numeric_limits<float>::max();
  }
  return dist / scale;
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
