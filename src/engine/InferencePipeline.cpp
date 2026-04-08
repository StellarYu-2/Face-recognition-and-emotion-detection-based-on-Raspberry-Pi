#include "engine/InferencePipeline.hpp"

namespace asdun {

InferencePipeline::InferencePipeline(FaceDetector& detector,
                                     FaceRecognizer& recognizer,
                                     EmotionRecognizer& emotion_recognizer,
                                     EmbeddingStore& embedding_store,
                                     TrackManager& track_manager,
                                     int detect_interval,
                                     int emotion_interval,
                                     float match_threshold,
                                     float sigmoid_tau)
    : detector_(detector),
      recognizer_(recognizer),
      emotion_recognizer_(emotion_recognizer),
      embedding_store_(embedding_store),
      track_manager_(track_manager),
      detect_interval_(detect_interval > 0 ? detect_interval : 1),
      emotion_interval_(emotion_interval > 0 ? emotion_interval : 1),
      match_threshold_(match_threshold),
      sigmoid_tau_(sigmoid_tau) {}

RecognitionResult InferencePipeline::process(const FramePacket& frame) {
  RecognitionResult result{};
  result.frame_id = frame.frame_id;
  result.ts_ms = frame.ts_ms;

  if (frame.bgr.empty()) {
    return result;
  }

  const bool do_detect = ((frame.frame_id % static_cast<std::uint64_t>(detect_interval_)) == 0);
  const bool do_emotion = ((frame.frame_id % static_cast<std::uint64_t>(emotion_interval_)) == 0);

  if (do_detect) {
    const auto detections = detector_.detect(frame.bgr);
    std::vector<IdentityResult> identities;
    std::vector<EmotionResult> emotions;
    identities.reserve(detections.size());
    emotions.reserve(detections.size());

    for (const auto& det : detections) {
      const cv::Rect bounded = det.box & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
      if (bounded.width <= 0 || bounded.height <= 0) {
        identities.push_back(IdentityResult{});
        emotions.push_back(EmotionResult{});
        continue;
      }
      const cv::Mat face = frame.bgr(bounded).clone();
      const auto emb = recognizer_.extractEmbedding(face);
      identities.push_back(embedding_store_.match(emb, match_threshold_, sigmoid_tau_));

      if (do_emotion) {
        emotions.push_back(emotion_recognizer_.infer(face));
      } else {
        emotions.push_back(EmotionResult{EmotionLabel::Unknown, 0.0F});
      }
    }

    track_manager_.updateWithDetections(detections, identities, emotions, frame.frame_id, frame.ts_ms);
  } else {
    track_manager_.tickWithoutDetections(frame.frame_id, frame.ts_ms);

    if (do_emotion) {
      const auto tracks = track_manager_.snapshot();
      std::vector<EmotionResult> emotions;
      emotions.reserve(tracks.size());
      for (const auto& tr : tracks) {
        const cv::Rect bounded = tr.box & cv::Rect(0, 0, frame.bgr.cols, frame.bgr.rows);
        if (bounded.width <= 0 || bounded.height <= 0) {
          emotions.push_back(EmotionResult{EmotionLabel::Unknown, 0.0F});
          continue;
        }
        emotions.push_back(emotion_recognizer_.infer(frame.bgr(bounded)));
      }
      track_manager_.updateEmotionsByTrackOrder(emotions, frame.ts_ms);
    }
  }

  result.tracks = track_manager_.snapshot();
  return result;
}

}  // namespace asdun

