#include "platform/PlatformClient.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#ifdef USE_CLOUD_CLIENT
#include <curl/curl.h>
#endif

namespace asdun {

namespace {

std::string trimTrailingSlash(std::string url) {
  while (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  return url;
}

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

std::string jsonBool(bool value) { return value ? "true" : "false"; }

std::string buildPayload(const PlatformClientConfig& config, bool online, const PlatformStatus& status) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  out << "{";
  out << "\"device_id\":\"" << jsonEscape(config.device_id) << "\",";
  out << "\"role\":\"" << jsonEscape(config.role) << "\",";
  out << "\"display_name\":\"" << jsonEscape(config.display_name) << "\",";
  out << "\"online\":true,";
  out << "\"merge_status\":true,";
  out << "\"status\":{";
  out << "\"app\":\"asdun_access\",";
  out << "\"app_online\":" << jsonBool(online) << ",";
  out << "\"mode\":\"" << jsonEscape(status.mode) << "\",";
  out << "\"inference_mode\":\"" << jsonEscape(status.inference_mode) << "\",";
  out << "\"cloud_connected\":" << jsonBool(status.cloud_connected) << ",";
  out << "\"active_tracks\":" << std::max(0, status.active_tracks) << ",";
  out << "\"fps\":" << std::max(0.0, status.fps);
  out << "},";
  out << "\"metadata\":{";
  out << "\"login\":\"asdun@asdun\"";
  out << "}";
  out << "}";
  return out.str();
}

#ifdef USE_CLOUD_CLIENT
std::size_t discardResponseCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  (void)ptr;
  (void)userdata;
  return size * nmemb;
}
#endif

}  // namespace

PlatformClient::PlatformClient(PlatformClientConfig config) : config_(std::move(config)) {}

PlatformClient::~PlatformClient() { stop(); }

bool PlatformClient::start() {
  if (!config_.enabled) {
    return false;
  }

#ifndef USE_CLOUD_CLIENT
  std::cerr << "[PlatformClient] libcurl is not available; platform reporting disabled." << std::endl;
  return false;
#else
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return true;
  }
  curl_global_init(CURL_GLOBAL_DEFAULT);
  running_ = true;
  worker_ = std::thread(&PlatformClient::workerLoop, this);
  std::cout << "[PlatformClient] enabled: " << statusUrl() << std::endl;
  return true;
#endif
}

void PlatformClient::stop() {
#ifdef USE_CLOUD_CLIENT
  PlatformStatus last_status{};
  bool should_send_offline = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    should_send_offline = running_;
    running_ = false;
    last_status = status_;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (should_send_offline) {
    postStatus(false, last_status);
  }
#else
  std::lock_guard<std::mutex> lock(mutex_);
  running_ = false;
#endif
}

bool PlatformClient::enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.enabled && running_;
}

void PlatformClient::updateStatus(const PlatformStatus& status) {
  std::lock_guard<std::mutex> lock(mutex_);
  status_ = status;
}

std::string PlatformClient::statusUrl() const {
  return trimTrailingSlash(config_.base_url) + "/api/status";
}

void PlatformClient::workerLoop() {
  while (true) {
    PlatformStatus snapshot{};
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!running_) {
        break;
      }
      snapshot = status_;
    }

    postStatus(true, snapshot);

    std::unique_lock<std::mutex> lock(mutex_);
    if (cv_.wait_for(lock,
                     std::chrono::milliseconds(std::max(1000, config_.status_interval_ms)),
                     [this] { return !running_; })) {
      break;
    }
  }
}

bool PlatformClient::postStatus(bool online, const PlatformStatus& status) const {
#ifndef USE_CLOUD_CLIENT
  (void)online;
  (void)status;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  const std::string payload = buildPayload(config_, online, status);
  const std::string url = statusUrl();
  curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(500, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discardResponseCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  if (!ok && config_.debug) {
    std::cerr << "[PlatformClient] status post failed url=" << url
              << " curl=" << curl_easy_strerror(code)
              << " http=" << http_code << std::endl;
  }
  return ok;
#endif
}

}  // namespace asdun
