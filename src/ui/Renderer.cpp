#include "ui/Renderer.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace asdun {

Renderer::Renderer(std::string window_name) : window_name_(std::move(window_name)) { ensureWindow(); }

Renderer::~Renderer() { closeWindow(); }

void Renderer::ensureWindow() const {
  if (!window_created_) {
    cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    window_created_ = true;
  }
}

void Renderer::drawRecognition(const cv::Mat& frame_bgr, const std::vector<TrackState>& tracks) const {
  ensureWindow();
  cv::Mat canvas = frame_bgr.clone();
  for (const auto& tr : tracks) {
    const cv::Scalar color = tr.identity.known ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 220);
    cv::rectangle(canvas, tr.box, color, 2);

    std::ostringstream oss;
    oss << tr.identity.name << " (" << std::fixed << std::setprecision(1) << tr.identity.conf_pct << "%)";
    if (tr.emotion.label != EmotionLabel::Unknown && tr.emotion.conf_pct > 0.1F) {
      oss << " | " << emotionToString(tr.emotion.label) << " (" << std::fixed << std::setprecision(1)
          << tr.emotion.conf_pct << "%)";
    }
    cv::putText(canvas, oss.str(), cv::Point(tr.box.x, std::max(0, tr.box.y - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
  }
  cv::imshow(window_name_, canvas);
}

void Renderer::drawEnrollment(const cv::Mat& frame_bgr,
                              const cv::Rect& face_box,
                              const std::string& status_text,
                              bool ready) const {
  ensureWindow();
  cv::Mat canvas = frame_bgr.clone();
  if (face_box.width > 0 && face_box.height > 0) {
    cv::rectangle(canvas, face_box, ready ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255), 2);
  }
  cv::putText(canvas,
              status_text,
              cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX,
              0.7,
              ready ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 220),
              2);
  cv::imshow(window_name_, canvas);
}

int Renderer::waitKey(int delay_ms) const { return cv::waitKey(delay_ms); }

void Renderer::closeWindow() const {
  if (window_created_) {
    cv::destroyWindow(window_name_);
    cv::waitKey(1);
    window_created_ = false;
  }
}

}  // namespace asdun
