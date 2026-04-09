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
                    float sigmoid_tau);

  RecognitionResult process(const FramePacket& frame);

 private:
  static cv::Rect expandRect(const cv::Rect& rect, int image_width, int image_height, float scale);

  FaceDetector& detector_;
  FaceRecognizer& recognizer_;
  EmotionRecognizer& emotion_recognizer_;
  EmbeddingStore& embedding_store_;
  TrackManager& track_manager_;

  int detect_interval_{5};
  int recognition_interval_{20};
  int emotion_interval_{15};
  int max_inference_faces_{1};
  float recognition_crop_scale_{1.15F};
  int recognition_min_face_size_{96};
  float recognition_blur_threshold_{35.0F};
  float recognition_margin_threshold_{0.05F};
  float emotion_crop_scale_{1.14F};
  int emotion_min_face_size_{96};
  bool debug_recognition_{false};
  bool debug_emotion_{false};
  float match_threshold_{0.8F};
  float sigmoid_tau_{0.08F};
};

}  // namespace asdun
