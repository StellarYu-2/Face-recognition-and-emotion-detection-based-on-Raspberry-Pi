#pragma once

#include <string>
#include <vector>

#include <opencv2/highgui.hpp>

#include "core/Types.hpp"

namespace asdun {

class Renderer {
 public:
  explicit Renderer(std::string window_name);
  ~Renderer();

  void drawRecognition(const cv::Mat& frame_bgr, const std::vector<TrackState>& tracks) const;
  void drawEnrollment(const cv::Mat& frame_bgr,
                      const cv::Rect& face_box,
                      const std::string& status_text,
                      bool ready) const;
  int waitKey(int delay_ms) const;
  void closeWindow() const;

 private:
  void ensureWindow() const;

  std::string window_name_;
  mutable bool window_created_{false};
};

}  // namespace asdun
