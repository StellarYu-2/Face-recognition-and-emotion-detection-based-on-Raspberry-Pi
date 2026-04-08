#include "storage/FileStore.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace asdun {

FileStore::FileStore(std::string images_root) : images_root_(std::move(images_root)) {}

bool FileStore::ensureBaseDirs() const {
  std::error_code ec;
  fs::create_directories(images_root_, ec);
  return !ec;
}

std::string FileStore::sanitizeName(const std::string& name) const {
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  if (out.empty()) {
    out = "user";
  }
  return out;
}

bool FileStore::saveFaceImage(const std::string& person_name, const cv::Mat& image_bgr, std::string* out_path) const {
  const std::string safe_name = sanitizeName(person_name);
  const fs::path person_dir = fs::path(images_root_) / safe_name;
  std::error_code ec;
  fs::create_directories(person_dir, ec);
  if (ec) {
    return false;
  }

  const auto now = std::chrono::system_clock::now();
  const auto sec = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf{};
#ifdef _WIN32
  localtime_s(&tm_buf, &sec);
#else
  localtime_r(&sec, &tm_buf);
#endif

  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_" << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000
      << ".jpg";
  const fs::path img_path = person_dir / oss.str();
  if (!cv::imwrite(img_path.string(), image_bgr)) {
    return false;
  }
  if (out_path != nullptr) {
    *out_path = img_path.string();
  }
  return true;
}

bool FileStore::removePersonDir(const std::string& person_name) const {
  const std::string safe_name = sanitizeName(person_name);
  const fs::path person_dir = fs::path(images_root_) / safe_name;
  std::error_code ec;
  fs::remove_all(person_dir, ec);
  return !ec;
}

bool FileStore::removeFiles(const std::vector<std::string>& paths) const {
  bool ok = true;
  for (const auto& p : paths) {
    std::error_code ec;
    fs::remove(fs::path(p), ec);
    if (ec) {
      ok = false;
    }
  }
  return ok;
}

}  // namespace asdun

