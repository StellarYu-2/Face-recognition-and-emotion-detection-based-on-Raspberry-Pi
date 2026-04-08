#pragma once

#include "core/Types.hpp"
#include "engine/EmotionRecognizer.hpp"
#include "engine/FaceDetector.hpp"
#include "engine/FaceRecognizer.hpp"
#include "storage/EmbeddingStore.hpp"
#include "tracking/TrackManager.hpp"

namespace asdun {

class InferencePipeline {
 public:
  InferencePipeline(FaceDetector& detector,
                    FaceRecognizer& recognizer,
                    EmotionRecognizer& emotion_recognizer,
                    EmbeddingStore& embedding_store,
                    TrackManager& track_manager,
                    int detect_interval,
                    int emotion_interval,
                    float match_threshold,
                    float sigmoid_tau);

  RecognitionResult process(const FramePacket& frame);

 private:
  FaceDetector& detector_;
  FaceRecognizer& recognizer_;
  EmotionRecognizer& emotion_recognizer_;
  EmbeddingStore& embedding_store_;
  TrackManager& track_manager_;

  int detect_interval_{3};
  int emotion_interval_{2};
  float match_threshold_{0.8F};
  float sigmoid_tau_{0.08F};
};

}  // namespace asdun

