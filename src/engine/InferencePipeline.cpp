#include "engine/InferencePipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

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

}  // namespace

InferencePipeline::InferencePipeline(FaceDetector& detector,
                                     FaceRecognizer& recognizer,
                                     EmotionRecognizer& emotion_recognizer,
                                     EmbeddingStore& embedding_store,
                                     TrackManager& track_manager,
                                     int detect_interval,
                                     int recognition_interval,
                                     int emotion_interval,
                                     int max_inference_faces,
                                     float recognition_crop_scale,
                                     int recognition_min_face_size,
                                     float recognition_blur_threshold,
                                     float recognition_margin_threshold,
                                     float emotion_crop_scale,
                                     int emotion_min_face_size,
                                     bool debug_recognition,
                                     bool debug_emotion,
                                     float match_threshold,
                                     float sigmoid_tau)
    : detector_(detector),
      recognizer_(recognizer),
      emotion_recognizer_(emotion_recognizer),
      embedding_store_(embedding_store),
      track_manager_(track_manager),
      detect_interval_(detect_interval > 0 ? detect_interval : 1),
      recognition_interval_(recognition_interval > 0 ? recognition_interval : 1),
      emotion_interval_(emotion_interval > 0 ? emotion_interval : 1),
      max_inference_faces_(max_inference_faces > 0 ? max_inference_faces : 1),
      recognition_crop_scale_(recognition_crop_scale > 1.0F ? recognition_crop_scale : 1.0F),
      recognition_min_face_size_(recognition_min_face_size > 0 ? recognition_min_face_size : 96),
      recognition_blur_threshold_(recognition_blur_threshold > 0.0F ? recognition_blur_threshold : 35.0F),
      recognition_margin_threshold_(recognition_margin_threshold > 0.0F ? recognition_margin_threshold : 0.05F),
      emotion_crop_scale_(emotion_crop_scale > 1.0F ? emotion_crop_scale : recognition_crop_scale_),
      emotion_min_face_size_(emotion_min_face_size > 0 ? emotion_min_face_size : 96),
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

    for (std::size_t idx = 0; idx < detections.size(); ++idx) {
      const auto& det = detections[idx];
      const cv::Rect recognition_rect = expandRect(det.box, frame.bgr.cols, frame.bgr.rows, recognition_crop_scale_);
      const cv::Rect emotion_rect = expandRect(det.box, frame.bgr.cols, frame.bgr.rows, emotion_crop_scale_);
      if (recognition_rect.width <= 0 || recognition_rect.height <= 0) {
        identities.push_back(IdentityResult{});
        emotions.push_back(EmotionResult{});
        continue;
      }

      if (!selected[idx]) {
        identities.push_back(IdentityResult{});
        emotions.push_back(EmotionResult{});
        continue;
      }

      const cv::Mat recognition_face = frame.bgr(recognition_rect).clone();
      if (do_recognize) {
        IdentityResult id{};
        const float blur_score = laplacianVariance(recognition_face);
        const bool recognition_quality_ok =
            recognition_rect.width >= recognition_min_face_size_ && recognition_rect.height >= recognition_min_face_size_ &&
            blur_score >= recognition_blur_threshold_ && det.det_score >= 0.50F;
        if (recognition_quality_ok) {
          id = embedding_store_.match(recognizer_.extractEmbedding(recognition_face),
                                      match_threshold_,
                                      sigmoid_tau_,
                                      recognition_margin_threshold_);
        } else {
          id.debug_summary = "skipped_quality area=" + std::to_string(det.box.area()) + " blur=" + std::to_string(blur_score);
        }
        if (debug_recognition_) {
          std::cout << "[Recog] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area()
                    << " name=" << id.name << " dist=" << id.distance << " conf=" << id.conf_pct
                    << " " << id.debug_summary << std::endl;
        }
        identities.push_back(std::move(id));
      } else {
        identities.push_back(IdentityResult{});
      }

      if (do_emotion && emotion_rect.width >= emotion_min_face_size_ && emotion_rect.height >= emotion_min_face_size_ &&
          det.det_score >= 0.55F) {
        auto emotion = emotion_recognizer_.infer(frame.bgr(emotion_rect).clone());
        if (debug_emotion_) {
          std::cout << "[Emotion] frame=" << frame.frame_id << " det=" << idx << " area=" << det.box.area()
                    << " label=" << emotionToString(emotion.label) << " conf=" << emotion.conf_pct
                    << " " << emotion.debug_summary << std::endl;
        }
        emotions.push_back(std::move(emotion));
      } else {
        emotions.push_back(EmotionResult{EmotionLabel::Unknown, 0.0F});
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
