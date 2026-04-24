#include "cloud/CloudClient.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

std::string joinUrl(const std::string& server_url, const std::string& path) {
  std::string url = trimTrailingSlash(server_url);
  if (path.empty()) {
    return url;
  }
  if (path.front() == '/') {
    return url + path;
  }
  return url + "/" + path;
}

std::string extractObject(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const std::size_t open = json.find('{', key_pos + needle.size());
  if (open == std::string::npos) {
    return {};
  }

  int depth = 0;
  for (std::size_t i = open; i < json.size(); ++i) {
    if (json[i] == '{') {
      ++depth;
    } else if (json[i] == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(open, i - open + 1);
      }
    }
  }
  return {};
}

std::optional<std::string> extractString(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t first_quote = json.find('"', colon + 1);
  if (first_quote == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t second_quote = json.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return std::nullopt;
  }
  return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::vector<std::string> extractPeopleNames(const std::string& json) {
  std::vector<std::string> names;
  const std::string needle = "\"people\"";
  const std::size_t people_pos = json.find(needle);
  if (people_pos == std::string::npos) {
    return names;
  }
  const std::size_t array_open = json.find('[', people_pos + needle.size());
  const std::size_t array_close = json.find(']', array_open == std::string::npos ? people_pos : array_open);
  if (array_open == std::string::npos || array_close == std::string::npos || array_close <= array_open) {
    return names;
  }

  std::size_t cursor = array_open + 1;
  while (cursor < array_close) {
    const std::size_t object_open = json.find('{', cursor);
    if (object_open == std::string::npos || object_open >= array_close) {
      break;
    }
    int depth = 0;
    std::size_t object_close = std::string::npos;
    for (std::size_t i = object_open; i < array_close; ++i) {
      if (json[i] == '{') {
        ++depth;
      } else if (json[i] == '}') {
        --depth;
        if (depth == 0) {
          object_close = i;
          break;
        }
      }
    }
    if (object_close == std::string::npos) {
      break;
    }
    const std::string object = json.substr(object_open, object_close - object_open + 1);
    if (auto name = extractString(object, "name"); name.has_value() && !name->empty()) {
      names.push_back(*name);
    }
    cursor = object_close + 1;
  }

  return names;
}

std::optional<bool> extractBool(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return std::nullopt;
  }
  if (json.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

std::optional<float> extractFloat(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos || json.compare(value_pos, 4, "null") == 0) {
    return std::nullopt;
  }

  char* end = nullptr;
  const float value = std::strtof(json.c_str() + value_pos, &end);
  if (end == json.c_str() + value_pos) {
    return std::nullopt;
  }
  return value;
}

EmotionLabel parseEmotionLabel(const std::string& label) {
  if (label == "Calm" || label == "Neutral") {
    return EmotionLabel::Neutral;
  }
  if (label == "Happy" || label == "Surprise") {
    return EmotionLabel::Happy;
  }
  if (label == "Sad" || label == "Fear") {
    return EmotionLabel::Sad;
  }
  if (label == "Angry" || label == "Disgust" || label == "Contempt") {
    return EmotionLabel::Angry;
  }
  return EmotionLabel::Unknown;
}

void fillGroupedEmotionProbs(const std::string& emotion_json, EmotionResult& emotion) {
  const std::string probs = extractObject(emotion_json, "probs");
  if (probs.empty()) {
    return;
  }
  emotion.grouped_probs[0] = extractFloat(probs, "Calm").value_or(0.0F);
  emotion.grouped_probs[1] = extractFloat(probs, "Happy").value_or(0.0F);
  emotion.grouped_probs[2] = extractFloat(probs, "Sad").value_or(0.0F);
  emotion.grouped_probs[3] = extractFloat(probs, "Angry").value_or(0.0F);
}

ExternalTrackAnalysis parseAnalysisResponse(const CloudAnalysisRequest& request,
                                            const std::string& response,
                                            double roundtrip_ms) {
  ExternalTrackAnalysis out{};
  out.track_id = request.track_id;
  out.frame_id = request.frame_id;
  out.ts_ms = request.ts_ms;
  out.source = "cloud";

  const auto ok = extractBool(response, "ok").value_or(false);
  if (!ok) {
    return out;
  }

  const std::string identity_json = extractObject(response, "identity");
  if (!identity_json.empty()) {
    IdentityResult identity{};
    identity.name = extractString(identity_json, "name").value_or("Unknown");
    identity.known = extractBool(identity_json, "known").value_or(false);
    identity.conf_pct = extractFloat(identity_json, "confidence").value_or(0.0F);
    identity.distance = extractFloat(identity_json, "distance").value_or(1.0F);
    identity.margin = extractFloat(identity_json, "gap").value_or(0.0F);
    identity.matched_sample_count = static_cast<int>(extractFloat(identity_json, "samples").value_or(0.0F));
    identity.measured = true;
    identity.attempted = true;
    std::ostringstream identity_candidates;
    if (auto top1 = extractString(identity_json, "top1"); top1.has_value() && !top1->empty()) {
      identity_candidates << " top1=" << *top1;
    }
    if (auto top2 = extractString(identity_json, "top2"); top2.has_value() && !top2->empty()) {
      identity_candidates << " top2=" << *top2;
    }
    identity.debug_summary = identity_candidates.str();
    out.identity = std::move(identity);
    out.has_identity = true;
  }

  const std::string emotion_json = extractObject(response, "emotion");
  if (!emotion_json.empty()) {
    EmotionResult emotion{};
    emotion.label = parseEmotionLabel(extractString(emotion_json, "label").value_or("Unknown"));
    emotion.conf_pct = extractFloat(emotion_json, "confidence").value_or(0.0F);
    emotion.attempted = true;
    fillGroupedEmotionProbs(emotion_json, emotion);
    emotion.debug_summary = "source=cloud";
    out.emotion = std::move(emotion);
    out.has_emotion = true;
  }

  std::ostringstream debug;
  debug << "source=cloud rtt_ms=" << std::fixed << std::setprecision(1) << roundtrip_ms;
  if (out.has_identity) {
    std::ostringstream identity_debug;
    identity_debug << debug.str()
                   << " dist=" << std::fixed << std::setprecision(3) << out.identity.distance
                   << " gap=" << out.identity.margin
                   << out.identity.debug_summary;
    out.identity.debug_summary = identity_debug.str();
  }
  if (out.has_emotion) {
    out.emotion.debug_summary = debug.str();
  }
  return out;
}

#ifdef USE_CLOUD_CLIENT
std::size_t writeStringCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  auto* out = static_cast<std::string*>(userdata);
  const std::size_t bytes = size * nmemb;
  out->append(ptr, bytes);
  return bytes;
}
#endif

}  // namespace

CloudClient::CloudClient(CloudClientConfig config) : config_(std::move(config)) {}

CloudClient::~CloudClient() { stop(); }

bool CloudClient::start() {
  if (!config_.enabled) {
    return false;
  }

#ifndef USE_CLOUD_CLIENT
  std::cerr << "[CloudClient] libcurl is not available; cloud client disabled." << std::endl;
  return false;
#else
  if (running_) {
    return true;
  }
  curl_global_init(CURL_GLOBAL_DEFAULT);
  if (!selectActiveServer()) {
    curl_global_cleanup();
    return false;
  }
  running_ = true;
  worker_ = std::thread(&CloudClient::workerLoop, this);
  std::cout << "[CloudClient] enabled: " << analyzeUrl() << std::endl;
  return true;
#endif
}

void CloudClient::stop() {
#ifdef USE_CLOUD_CLIENT
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    running_ = false;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  curl_global_cleanup();
#else
  running_ = false;
#endif
}

bool CloudClient::enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.enabled && running_;
}

bool CloudClient::submit(const CloudAnalysisRequest& request) {
  if (!enabled() || request.track_id < 0 || request.face_bgr.empty()) {
    return false;
  }

  CloudAnalysisRequest queued = request;
  queued.face_bgr = request.face_bgr.clone();
  if (config_.crop_size > 0 &&
      (queued.face_bgr.cols != config_.crop_size || queued.face_bgr.rows != config_.crop_size)) {
    const int interpolation = (queued.face_bgr.cols > config_.crop_size || queued.face_bgr.rows > config_.crop_size)
                                  ? cv::INTER_AREA
                                  : cv::INTER_LINEAR;
    cv::resize(queued.face_bgr, queued.face_bgr, cv::Size(config_.crop_size, config_.crop_size), 0.0, 0.0, interpolation);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    return false;
  }
  const auto last_it = last_submit_ms_.find(request.track_id);
  if (last_it != last_submit_ms_.end() &&
      request.ts_ms >= last_it->second &&
      request.ts_ms - last_it->second < static_cast<std::uint64_t>(std::max(0, config_.min_interval_ms))) {
    return false;
  }
  if (inflight_tracks_.count(request.track_id) > 0 || queue_.size() >= static_cast<std::size_t>(std::max(1, config_.max_queue_size))) {
    return false;
  }

  last_submit_ms_[request.track_id] = request.ts_ms;
  inflight_tracks_.insert(request.track_id);
  queue_.push_back(std::move(queued));
  cv_.notify_one();
  return true;
}

std::vector<ExternalTrackAnalysis> CloudClient::pollCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ExternalTrackAnalysis> out;
  out.reserve(completed_.size());
  while (!completed_.empty()) {
    out.push_back(std::move(completed_.front()));
    completed_.pop_front();
  }
  return out;
}

std::vector<std::string> CloudClient::candidateServerUrls() const {
  std::vector<std::string> configured_urls = config_.server_urls;
  if (configured_urls.empty()) {
    configured_urls.push_back(config_.server_url);
  }
  std::vector<std::string> urls;
  urls.reserve(configured_urls.size());
  for (const auto& raw_url : configured_urls) {
    const std::string url = trimTrailingSlash(raw_url);
    if (url.empty() || std::find(urls.begin(), urls.end(), url) != urls.end()) {
      continue;
    }
    urls.push_back(url);
  }
  return urls;
}

bool CloudClient::probeServer(const std::string& server_url, std::string* response) const {
  if (server_url.empty()) {
    return false;
  }
  if (config_.health_check_path.empty()) {
    return true;
  }

#ifndef USE_CLOUD_CLIENT
  (void)response;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::string body;
  const std::string url = joinUrl(server_url, config_.health_check_path);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(500, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);

  if (response != nullptr) {
    *response = std::move(body);
  }
  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  if (!ok && config_.debug) {
    std::cerr << "[CloudClient] health check failed url=" << url
              << " curl=" << curl_easy_strerror(code)
              << " http=" << http_code << std::endl;
  }
  return ok;
#endif
}

bool CloudClient::selectActiveServer() {
  const auto urls = candidateServerUrls();
  if (urls.empty()) {
    std::cerr << "[CloudClient] no cloud server URL configured." << std::endl;
    return false;
  }

  for (const auto& url : urls) {
    if (config_.debug) {
      std::cout << "[CloudClient] probing: " << joinUrl(url, config_.health_check_path) << std::endl;
    }
    std::string response;
    if (!probeServer(url, &response)) {
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_server_url_ = trimTrailingSlash(url);
    }
    std::cout << "[CloudClient] selected server: " << url << std::endl;
    if (config_.debug && !response.empty()) {
      std::cout << "[CloudClient] health response: " << response << std::endl;
    }
    return true;
  }

  std::cerr << "[CloudClient] no healthy cloud server found; local pipeline will continue." << std::endl;
  return false;
}

std::string CloudClient::baseUrl() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_server_url_.empty()) {
    return active_server_url_;
  }
  return trimTrailingSlash(config_.server_url);
}

std::string CloudClient::analyzeUrl() const { return baseUrl() + "/analyze"; }

std::string CloudClient::enrollUrl() const { return baseUrl() + "/gallery/enroll"; }

std::string CloudClient::deleteUrl() const { return baseUrl() + "/gallery/delete"; }

std::string CloudClient::galleryUrl() const { return baseUrl() + "/gallery"; }

bool CloudClient::enrollPerson(const std::string& name, const std::vector<std::string>& image_paths, bool replace) const {
  if (!enabled() || name.empty() || image_paths.empty()) {
    return false;
  }

#ifndef USE_CLOUD_CLIENT
  (void)name;
  (void)image_paths;
  (void)replace;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::string response;
  curl_mime* mime = curl_mime_init(curl);
  auto addTextPart = [mime](const char* field_name, const std::string& value) {
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, field_name);
    curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
  };

  addTextPart("name", name);
  addTextPart("replace", replace ? "true" : "false");

  int attached = 0;
  for (const auto& path : image_paths) {
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "images");
    curl_mime_type(part, "image/jpeg");
    if (curl_mime_filedata(part, path.c_str()) == CURLE_OK) {
      ++attached;
    }
  }

  if (attached == 0) {
    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    return false;
  }

  const std::string url = enrollUrl();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(5000, config_.timeout_ms * std::max(1, attached))));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  if (config_.debug) {
    if (ok) {
      std::cout << "[CloudClient] cloud enrollment uploaded name=" << name
                << " images=" << attached
                << " response=" << response << std::endl;
    } else {
      std::cerr << "[CloudClient] cloud enrollment failed name=" << name
                << " images=" << attached
                << " curl=" << curl_easy_strerror(code)
                << " http=" << http_code
                << " response=" << response << std::endl;
    }
  }
  return ok;
#endif
}

std::vector<std::string> CloudClient::listPeople() const {
  if (!enabled()) {
    return {};
  }

#ifndef USE_CLOUD_CLIENT
  return {};
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return {};
  }

  std::string response;
  const std::string url = galleryUrl();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1000, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);

  if (code != CURLE_OK || http_code < 200 || http_code >= 300) {
    if (config_.debug) {
      std::cerr << "[CloudClient] cloud gallery list failed curl=" << curl_easy_strerror(code)
                << " http=" << http_code << std::endl;
    }
    return {};
  }
  return extractPeopleNames(response);
#endif
}

bool CloudClient::deletePerson(const std::string& name) const {
  if (!enabled() || name.empty()) {
    return false;
  }

#ifndef USE_CLOUD_CLIENT
  (void)name;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::string response;
  curl_mime* mime = curl_mime_init(curl);
  curl_mimepart* part = curl_mime_addpart(mime);
  curl_mime_name(part, "name");
  curl_mime_data(part, name.c_str(), CURL_ZERO_TERMINATED);

  const std::string url = deleteUrl();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1000, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  if (config_.debug) {
    if (ok) {
      std::cout << "[CloudClient] cloud person delete requested name=" << name
                << " response=" << response << std::endl;
    } else {
      std::cerr << "[CloudClient] cloud person delete failed name=" << name
                << " curl=" << curl_easy_strerror(code)
                << " http=" << http_code << std::endl;
    }
  }
  return ok;
#endif
}

void CloudClient::workerLoop() {
  while (true) {
    CloudAnalysisRequest request{};
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !running_ || !queue_.empty(); });
      if (!running_ && queue_.empty()) {
        break;
      }
      request = std::move(queue_.front());
      queue_.pop_front();
    }

    ExternalTrackAnalysis analysis{};
    const bool ok = analyze(request, &analysis);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      inflight_tracks_.erase(request.track_id);
      if (ok) {
        completed_.push_back(std::move(analysis));
      }
    }
  }
}

bool CloudClient::analyze(const CloudAnalysisRequest& request, ExternalTrackAnalysis* out) const {
  if (out == nullptr) {
    return false;
  }

#ifndef USE_CLOUD_CLIENT
  (void)request;
  return false;
#else
  std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, std::clamp(config_.jpeg_quality, 30, 100)};
  std::vector<uchar> jpg;
  if (!cv::imencode(".jpg", request.face_bgr, jpg, params) || jpg.empty()) {
    return false;
  }

  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::string response;
  curl_mime* mime = curl_mime_init(curl);
  auto addTextPart = [mime](const char* name, const std::string& value) {
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, name);
    curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
  };

  addTextPart("track_id", std::to_string(request.track_id));
  addTextPart("frame_id", std::to_string(request.frame_id));
  addTextPart("ts_ms", std::to_string(request.ts_ms));
  addTextPart("source", "raspberry-pi");
  addTextPart("local_name", request.local_known ? request.local_name : "Unknown");
  addTextPart("local_conf", std::to_string(request.local_conf_pct));

  curl_mimepart* image_part = curl_mime_addpart(mime);
  curl_mime_name(image_part, "image");
  curl_mime_filename(image_part, "face.jpg");
  curl_mime_type(image_part, "image/jpeg");
  curl_mime_data(image_part, reinterpret_cast<const char*>(jpg.data()), jpg.size());

  const std::string url = analyzeUrl();
  const auto start = std::chrono::steady_clock::now();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  const auto end = std::chrono::steady_clock::now();
  const double roundtrip_ms = std::chrono::duration<double, std::milli>(end - start).count();
  if (code != CURLE_OK || http_code < 200 || http_code >= 300) {
    if (config_.debug) {
      std::cerr << "[CloudClient] analyze failed track=" << request.track_id
                << " curl=" << curl_easy_strerror(code)
                << " http=" << http_code << std::endl;
    }
    return false;
  }

  *out = parseAnalysisResponse(request, response, roundtrip_ms);
  if (config_.debug) {
    std::cout << "[CloudClient] result track=" << request.track_id
              << " frame=" << request.frame_id
              << " identity=" << (out->has_identity ? out->identity.name : "-");
    if (out->has_identity) {
      std::cout << " id_conf=" << std::fixed << std::setprecision(1) << out->identity.conf_pct
                << " id_dist=" << std::setprecision(3) << out->identity.distance
                << " id_gap=" << out->identity.margin
                << " " << out->identity.debug_summary;
    }
    std::cout
              << " emotion=" << (out->has_emotion ? emotionToString(out->emotion.label) : "-")
              << " rtt_ms=" << std::fixed << std::setprecision(1) << roundtrip_ms << std::endl;
  }
  return out->has_identity || out->has_emotion;
#endif
}

}  // namespace asdun
