#include "engine/InferencePipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

constexpr float kRecognitionDetScoreFloor = 0.68F;
constexpr float kEmotionDetScoreFloor = 0.72F;
constexpr float kValidationScoreBypass = 0.94F;

float laplacianVariance(const cv::Mat& bgr) {
  if (bgr.empty()) {
    return 0.0F;
  }
  cv::Mat gray{};
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Mat lap{};
  cv::Laplacian(gray, lap, CV_32F, 3);
  cv::Scalar mean{}, stddev{};
  cv::meanStdDev(lap, mean, stddev);
  return static_cast<float>(stddev[0] * stddev[0]);
}

float adjustMatchThreshold(float base_threshold,
                           int min_face_side,
                           int min_face_size,
                           float blur_score,
                           float blur_threshold) {
  float adjusted = base_threshold;

  if (min_face_side < (min_face_size + 20)) {
    adjusted -= 0.05F;
  }
  if (min_face_side < (min_face_size + 8)) {
    adjusted -= 0.03F;
  }
  if (blur_score < (blur_threshold * 1.8F)) {
    adjusted -= 0.03F;
  }
  if (blur_score < (blur_threshold * 1.2F)) {
    adjusted -= 0.02F;
  }

  return std::clamp(adjusted, 0.88F, base_threshold);
}

float adjustMarginThreshold(float base_margin_threshold,
                            int min_face_side,
                            int min_face_size,
                            float blur_score,
                            float blur_threshold) {
  float adjusted = base_margin_threshold;

  if (min_face_side < (min_face_size + 20)) {
    adjusted += 0.02F;
  }
  if (blur_score < (blur_threshold * 1.8F)) {
    adjusted += 0.015F;
  }

  return std::clamp(adjusted, base_margin_threshold, base_margin_threshold + 0.05F);
}

std::uint64_t elapsedMs(std::uint64_t now_ms, std::uint64_t last_ms) {
  if (last_ms == 0 || now_ms <= last_ms) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return now_ms - last_ms;
}

bool qualityImproved(const Track* track, float blur_score, int min_face_side, float blur_gain, int size_gain) {
  if (track == nullptr || track->last_recognition_ms == 0) {
    return true;
  }
  return blur_score >= (track->last_recognition_blur_score + blur_gain) ||
         min_face_side >= (track->last_recognition_face_size + size_gain);
}

bool shouldAttemptRecognition(const Track* track,
                              bool selected,
                              bool periodic_slot,
                              std::uint64_t ts_ms,
                              float blur_score,
                              int min_face_side,
                              int known_identity_cooldown_ms,
                              int unknown_identity_cooldown_ms,
                              float recognition_retrigger_blur_gain,
                              int recognition_retrigger_size_gain) {
  if (!selected) {
    return false;
  }
  if (track == nullptr) {
    return true;
  }

  const std::uint64_t since_last = elapsedMs(ts_ms, track->last_recognition_ms);
  const bool improved = qualityImproved(track,
                                        blur_score,
                                        min_face_side,
                                        recognition_retrigger_blur_gain,
                                        recognition_retrigger_size_gain);
  if (!track->identity.known) {
    return improved || since_last >= static_cast<std::uint64_t>(std::max(unknown_identity_cooldown_ms, 0));
  }

  const bool weak_identity = track->identity.conf_pct < 78.0F || track->identity.margin < 0.12F;
  if (improved && since_last >= static_cast<std::uint64_t>(std::max(known_identity_cooldown_ms / 2, 1))) {
    return true;
  }
  if (weak_identity && since_last >= static_cast<std::uint64_t>(std::max(unknown_identity_cooldown_ms, 0))) {
    return true;
  }
  return periodic_slot && since_last >= static_cast<std::uint64_t>(std::max(known_identity_cooldown_ms, 0));
}

bool shouldAttemptEmotion(const Track* track,
                          bool selected,
                          bool periodic_slot,
                          std::uint64_t ts_ms,
                          int emotion_cooldown_ms) {
  if (!selected || !periodic_slot || track == nullptr) {
    return false;
  }
  if (track->detection_hits < 3) {
    return false;
  }
  const std::uint64_t since_last = elapsedMs(ts_ms, track->last_emotion_ms);
  return since_last >= static_cast<std::uint64_t>(std::max(emotion_cooldown_ms, 0));
}

}  // namespace

InferencePipeline::InferencePipeline(FaceDetector& detector,
                                     FaceLandmarkEstimator& landmark_estimator,
                                     FaceAligner& face_aligner,
                                     FaceRecognizer& recognizer,
                                     EmotionRecognizer& emotion_recognizer,
                                     EmbeddingStore& embedding_store,
                                     TrackManager& track_manager,
                                     int detect_interval,
                                     int recognition_interval,
                                     int emotion_interval,
                                     int max_inference_faces,
                                     int max_emotion_faces,
                                     float recognition_crop_scale,
                                     int recognition_min_face_size,
                                     float recognition_blur_threshold,
                                     float recognition_margin_threshold,
                                     int known_identity_cooldown_ms,
                                     int unknown_identity_cooldown_ms,
                                     float recognition_retrigger_blur_gain,
                                     int recognition_retrigger_size_gain,
                                     float emotion_crop_scale,
                                     int emotion_min_face_size,
                                     int emotion_cooldown_ms,
                                     bool emotion_require_known_identity,
                                     bool debug_recognition,
                                     bool debug_emotion,
                                     float match_threshold,
                                     float sigmoid_tau)
    : detector_(detector),
      landmark_estimator_(landmark_estimator),
      face_aligner_(face_aligner),
      recognizer_(recognizer),
      emotion_recognizer_(emotion_recognizer),
      embedding_store_(embedding_store),
      track_manager_(track_manager),
      detect_interval_(detect_interval > 0 ? detect_interval : 1),
      recognition_interval_(recognition_interval > 0 ? recognition_interval : 1),
      emotion_interval_(emotion_interval > 0 ? emotion_interval : 1),
      max_inference_faces_(max_inference_faces > 0 ? max_inference_faces : 1),
      max_emotion_faces_(max_emotion_faces > 0 ? max_emotion_faces : 1),
      recognition_crop_scale_(recognition_crop_scale > 1.0F ? recognition_crop_scale : 1.0F),
      recognition_min_face_size_(recognition_min_face_size > 0 ? recognition_min_face_size : 96),
      recognition_blur_threshold_(recognition_blur_threshold > 0.0F ? recognition_blur_threshold : 35.0F),
      recognition_margin_threshold_(recognition_margin_threshold > 0.0F ? recognition_margin_threshold : 0.05F),
      known_identity_cooldown_ms_(known_identity_cooldown_ms > 0 ? known_identity_cooldown_ms : 900),
      unknown_identity_cooldown_ms_(unknown_identity_cooldown_ms > 0 ? unknown_identity_cooldown_ms : 250),
      recognition_retrigger_blur_gain_(recognition_retrigger_blur_gain > 0.0F ? recognition_retrigger_blur_gain : 18.0F),
      recognition_retrigger_size_gain_(recognition_retrigger_size_gain > 0 ? recognition_retrigger_size_gain : 20),
      emotion_crop_scale_(emotion_crop_scale > 1.0F ? emotion_crop_scale : recognition_crop_scale_),
      emotion_min_face_size_(emotion_min_face_size > 0 ? emotion_min_face_size : 96),
      emotion_cooldown_ms_(emotion_cooldown_ms > 0 ? emotion_cooldown_ms : 600),
      emotion_require_known_identity_(emotion_require_known_identity),
      debug_recognition_(debug_recognition),
      debug_emotion_(debug_emotion),
      match_threshold_(match_threshold),
      sigmoid_tau_(sigmoid_tau) {}

cv::Rect InferencePipeline::expandRect(const cv::Rect& rect, int image_width, int image_height, float scale) {
  if (rect.width <= 0 || rect.height <= 0) {
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
  expanded &= cv::Rect(0, 0, image_width, image_height);
  return expanded;
}

RecognitionResult InferencePipeline::process(const FramePacket& frame) {
  RecognitionResult result{};
  result.frame_id = frame.frame_id;
  result.ts_ms = frame.ts_ms;

  if (frame.bgr.empty()) {
    return result;
  }

  const bool do_detect = ((frame.frame_id % static_cast<std::uint64_t>(detect_interval_)) == 0);
  const bool do_recognize = ((frame.frame_id % static_cast<std::uint64_t>(recognition_interval_)) == 0);
  const bool do_emotion = ((frame.frame_id % static_cast<std::uint64_t>(emotion_interval_)) == 0);

  if (do_detect) {
    auto detections = detector_.detect(frame.bgr);
    if (debug_recognition_ && detections.empty()) {
      std::cout << "[Detect] frame=" << frame.frame_id << " dets=0" << '\n';
    }
    const std::vector<int> preview_matches = track_manager_.previewMatches(detections);
    std::vector<IdentityResult> identities;
    std::vector<EmotionResult> emotions;
    identities.reserve(detections.size());
    emotions.reserve(detections.size());

    std::vector<std::size_t> ranked_indices(detections.size());
    for (std::size_t i = 0; i < detections.size(); ++i) {
      ranked_indices[i] = i;
    }
    std::sort(ranked_indices.begin(), ranked_indices.end(), [&detections](std::size_t a, std::size_t b) {
      return detections[a].box.area() > detections[b].box.area();
    });

    std::vector<bool> selected(detections.size(), false);
    const std::size_t active_count = std::min<std::size_t>(ranked_indices.size(), static_cast<std::size_t>(max_inference_faces_));
    for (std::size_t i = 0; i < active_count; ++i) {
      selected[ranked_indices[i]] = true;
    }

    std::vector<bool> emotion_selected(detections.size(), false);
    std::vector<std::size_t> emotion_ranked_indices = ranked_indices;
    std::sort(emotion_ranked_indices.begin(), emotion_ranked_indices.end(), [&](std::size_t a, std::size_t b) {
      const Track* track_a =
          (a < preview_matches.size()) ? track_manager_.getTrackByIndex(preview_matches[a]) : nullptr;
      const Track* track_b =
          (b < preview_matches.size()) ? track_manager_.getTrackByIndex(preview_matches[b]) : nullptr;
      const bool known_a = track_a != nullptr && track_a->identity.known;
      const bool known_b = track_b != nullptr && track_b->identity.known;
      if (known_a != known_b) {
        return known_a > known_b;
      }

      const int hits_a = (track_a != nullptr) ? track_a->detection_hits : 0;
      const int hits_b = (track_b != nullptr) ? track_b->detection_hits : 0;
      if (hits_a != hits_b) {
        return hits_a > hits_b;
      }

      if (std::abs(detections[a].det_score - detections[b].det_score) > 1e-6F) {
        return detections[a].det_score > detections[b].det_score;
      }
      return detections[a].box.area() > detections[b].box.area();
    });

    const std::size_t emotion_count =
        std::min<std::size_t>(emotion_ranked_indices.size(), static_cast<std::size_t>(max_emotion_faces_));
    for (std::size_t i = 0; i < emotion_count; ++i) {
      emotion_selected[emotion_ranked_indices[i]] = true;
    }

    for (std::size_t idx = 0; idx < detections.size(); ++idx) {
      auto& det = detections[idx];
      const Track* matched_track =
          (idx < preview_matches.size()) ? track_manager_.getTrackByIndex(preview_matches[idx]) : nullptr;
      const bool has_detector_landmarks = det.landmarks.valid;
      const bool alignment_required = selected[idx] && (has_detector_landmarks || landmark_estimator_.ready());
      if (alignment_required) {
        const auto landmarks = has_detector_landmarks ? det.landmarks : landmark_estimator_.estimate(frame.bgr, det.box);
        if (landmarks.valid) {
          const cv::Rect refined_box = face_aligner_.refineBox(landmarks, frame.bgr.size());
          if (refined_box.width > 0 && refined_box.height > 0) {
            det.box = (det.box | refined_box) & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
          }
        }
      }
      const cv::Rect recognition_rect = expandRect(det.box, frame.bgr.cols, frame.bgr.rows, recognition_crop_scale_);
      const cv::Rect emotion_rect = expandRect(det.box, frame.bgr.cols, frame.bgr.rows, emotion_crop_scale_);
      if (recognition_rect.width <= 0 || recognition_rect.height <= 0) {
        identities.push_back(IdentityResult{});
        emotions.push_back(EmotionResult{});
        continue;
      }

      const cv::Mat recognition_face = frame.bgr(recognition_rect);
      const float blur_score = (selected[idx] || emotion_selected[idx]) ? laplacianVariance(recognition_face) : 0.0F;
      const int min_face_side = std::min(recognition_rect.width, recognition_rect.height);
      const bool is_new_or_unknown_track = (matched_track == nullptr) || !matched_track->identity.known;
      const bool require_face_validation =
          selected[idx] && is_new_or_unknown_track && det.det_score < kValidationScoreBypass;
      const bool face_validation_ok = !require_face_validation || detector_.validateFaceRegion(frame.bgr, det.box);
      const bool recognition_quality_ok =
          recognition_rect.width >= recognition_min_face_size_ && recognition_rect.height >= recognition_min_face_size_ &&
          blur_score >= recognition_blur_threshold_ && det.det_score >= kRecognitionDetScoreFloor && face_validation_ok;
      const bool attempt_recognition =
          recognition_quality_ok &&
          shouldAttemptRecognition(matched_track,
                                   selected[idx],
                                   do_recognize,
                                   frame.ts_ms,
                                   blur_score,
                                   min_face_side,
                                   known_identity_cooldown_ms_,
                                   unknown_identity_cooldown_ms_,
                                   recognition_retrigger_blur_gain_,
                                   recognition_retrigger_size_gain_);

      if (attempt_recognition) {
        IdentityResult id{};
        const float dynamic_match_threshold =
            adjustMatchThreshold(match_threshold_, min_face_side, recognition_min_face_size_, blur_score, recognition_blur_threshold_);
        const float dynamic_margin_threshold =
            adjustMarginThreshold(recognition_margin_threshold_,
                                  min_face_side,
                                  recognition_min_face_size_,
                                  blur_score,
                                  recognition_blur_threshold_);
        const auto embedding = recognizer_.extractEmbedding(recognition_face);
        id.attempted = true;
        id.input_blur_score = blur_score;
        id.input_min_face_size = min_face_side;
        if (embedding.empty()) {
          id.measured = true;
          id.known = false;
          id.distance = std::numeric_limits<float>::max();
          std::ostringstream skip;
          skip << std::fixed << std::setprecision(3);
          skip << "decision=skip_embedding shown=Unknown area=" << det.box.area() << " blur=" << blur_score
               << " det=" << det.det_score << " src=crop";
          id.debug_summary = skip.str();
        } else {
          id = embedding_store_.match(embedding, dynamic_match_threshold, sigmoid_tau_, dynamic_margin_threshold);
          id.attempted = true;
          id.input_blur_score = blur_score;
          id.input_min_face_size = min_face_side;
          if (!id.debug_summary.empty()) {
            std::ostringstream extra;
            extra << std::fixed << std::setprecision(3);
            extra << id.debug_summary << " thr=" << dynamic_match_threshold << " gap_thr=" << dynamic_margin_threshold
                  << " det=" << det.det_score << " src=crop";
            id.debug_summary = extra.str();
          }
        }
        if (debug_recognition_) {
          std::cout << "[Recog] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area() << " "
                    << id.debug_summary << '\n';
        }
        identities.push_back(std::move(id));
      } else {
        IdentityResult id{};
        if (selected[idx] && !face_validation_ok) {
          id.attempted = true;
          id.measured = true;
          id.known = false;
          id.input_blur_score = blur_score;
          id.input_min_face_size = min_face_side;
          std::ostringstream reject;
          reject << std::fixed << std::setprecision(3);
          reject << "decision=reject_validation shown=Unknown area=" << det.box.area() << " blur=" << blur_score
                 << " det=" << det.det_score;
          id.debug_summary = reject.str();
          if (debug_recognition_) {
            std::cout << "[Recog] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area() << " "
                      << id.debug_summary << '\n';
          }
        } else if (selected[idx] && !recognition_quality_ok) {
          const bool size_ok =
              recognition_rect.width >= recognition_min_face_size_ && recognition_rect.height >= recognition_min_face_size_;
          const bool blur_ok = blur_score >= recognition_blur_threshold_;
          const bool det_ok = det.det_score >= kRecognitionDetScoreFloor;
          std::ostringstream skip;
          skip << std::fixed << std::setprecision(3);
          skip << "decision=skip_quality shown=Unknown area=" << det.box.area() << " blur=" << blur_score
               << " size_ok=" << (size_ok ? 1 : 0) << " blur_ok=" << (blur_ok ? 1 : 0) << " det_ok=" << (det_ok ? 1 : 0)
               << " det=" << det.det_score;
          id.debug_summary = skip.str();
          if (debug_recognition_) {
            std::cout << "[Recog] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area() << " "
                      << id.debug_summary << '\n';
          }
        }
        identities.push_back(std::move(id));
      }

      const IdentityResult& emotion_identity = identities.back();
      const bool known_for_emotion =
          emotion_identity.known || (matched_track != nullptr && matched_track->identity.known);
      const bool identity_ok_for_emotion = !emotion_require_known_identity_ || known_for_emotion;
      const cv::Mat emotion_face = frame.bgr(emotion_rect);
      const float emotion_blur_score = emotion_selected[idx] ? laplacianVariance(emotion_face) : 0.0F;
      const bool emotion_quality_ok = emotion_rect.width >= emotion_min_face_size_ &&
                                      emotion_rect.height >= emotion_min_face_size_ &&
                                      det.det_score >= kEmotionDetScoreFloor &&
                                      emotion_blur_score >= (recognition_blur_threshold_ * 0.75F) &&
                                      face_validation_ok && identity_ok_for_emotion;
      const bool attempt_emotion =
          emotion_quality_ok &&
          shouldAttemptEmotion(matched_track, emotion_selected[idx], do_emotion, frame.ts_ms, emotion_cooldown_ms_);
      if (attempt_emotion) {
        auto emotion = emotion_recognizer_.infer(emotion_face);
        emotion.attempted = true;
        if (debug_emotion_) {
          std::cout << "[Emotion] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area()
                    << " label=" << emotionToString(emotion.label) << " conf=" << emotion.conf_pct
                    << " known=" << (known_for_emotion ? 1 : 0)
                    << " blur=" << emotion_blur_score
                    << " " << emotion.debug_summary << '\n';
        }
        emotions.push_back(std::move(emotion));
      } else {
        if (debug_emotion_ && emotion_selected[idx]) {
          std::cout << "[Emotion] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area()
                    << " skipped known=" << (known_for_emotion ? 1 : 0)
                    << " require_known=" << (emotion_require_known_identity_ ? 1 : 0)
                    << " blur=" << emotion_blur_score
                    << " det_ok=" << (det.det_score >= kEmotionDetScoreFloor ? 1 : 0)
                    << " size_ok=" << ((emotion_rect.width >= emotion_min_face_size_ &&
                                         emotion_rect.height >= emotion_min_face_size_) ? 1 : 0)
                    << " face_ok=" << (face_validation_ok ? 1 : 0) << '\n';
        }
        emotions.push_back(EmotionResult{});
      }
    }

    track_manager_.updateWithDetections(detections, identities, emotions, frame.frame_id, frame.ts_ms);
  } else {
    track_manager_.tickWithoutDetections(frame.frame_id, frame.ts_ms);
  }

  result.tracks = track_manager_.snapshot();
  return result;
}

}  // namespace asdun
