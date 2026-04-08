#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace asdun {

class FileStore {
 public:
  explicit FileStore(std::string images_root);

  bool ensureBaseDirs() const;
  std::string sanitizeName(const std::string& name) const;
  bool saveFaceImage(const std::string& person_name, const cv::Mat& image_bgr, std::string* out_path) const;
  bool removePersonDir(const std::string& person_name) const;
  bool removeFiles(const std::vector<std::string>& paths) const;

 private:
  std::string images_root_;
};

}  // namespace asdun

