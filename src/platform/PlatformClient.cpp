#include "platform/PlatformClient.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
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

std::string jsonObjectFromMap(const std::map<std::string, std::string>& values) {
  std::ostringstream out;
  out << "{";
  bool first = true;
  for (const auto& [key, value] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << jsonEscape(key) << "\":\"" << jsonEscape(value) << "\"";
  }
  out << "}";
  return out.str();
}

std::string lowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string trimCopy(const std::string& value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  const auto begin = std::find_if_not(value.begin(), value.end(), is_space);
  if (begin == value.end()) {
    return "";
  }
  const auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
  return std::string(begin, end);
}

std::string jsonUnescape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  bool escaped = false;
  for (const char c : value) {
    if (escaped) {
      switch (c) {
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        default:
          out += c;
          break;
      }
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    out += c;
  }
  return out;
}

std::size_t findJsonKey(const std::string& json, const std::string& key, std::size_t start = 0) {
  return json.find("\"" + key + "\"", start);
}

std::size_t skipWhitespace(const std::string& json, std::size_t pos) {
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) {
    ++pos;
  }
  return pos;
}

std::string extractStringValue(const std::string& json, const std::string& key) {
  const std::size_t key_pos = findJsonKey(json, key);
  if (key_pos == std::string::npos) {
    return "";
  }
  const std::size_t colon = json.find(':', key_pos + key.size() + 2);
  if (colon == std::string::npos) {
    return "";
  }
  std::size_t quote = skipWhitespace(json, colon + 1);
  if (quote >= json.size() || json[quote] != '"') {
    return "";
  }
  ++quote;
  std::string raw;
  bool escaped = false;
  for (std::size_t i = quote; i < json.size(); ++i) {
    const char c = json[i];
    if (escaped) {
      raw += '\\';
      raw += c;
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"') {
      return jsonUnescape(raw);
    }
    raw += c;
  }
  return "";
}

std::string extractJsonContainer(const std::string& json, const std::string& key, char open_char, char close_char) {
  const std::size_t key_pos = findJsonKey(json, key);
  if (key_pos == std::string::npos) {
    return "";
  }
  const std::size_t colon = json.find(':', key_pos + key.size() + 2);
  if (colon == std::string::npos) {
    return "";
  }
  std::size_t open = skipWhitespace(json, colon + 1);
  if (open >= json.size() || json[open] != open_char) {
    return "";
  }

  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = open; i < json.size(); ++i) {
    const char c = json[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open_char) {
      ++depth;
    } else if (c == close_char) {
      --depth;
      if (depth == 0) {
        return json.substr(open, i - open + 1);
      }
    }
  }
  return "";
}

std::string extractObjectValue(const std::string& json, const std::string& key) {
  return extractJsonContainer(json, key, '{', '}');
}

std::string extractArrayValue(const std::string& json, const std::string& key) {
  return extractJsonContainer(json, key, '[', ']');
}

std::vector<std::string> splitTopLevelObjects(const std::string& array_json) {
  std::vector<std::string> objects;
  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  std::size_t object_start = std::string::npos;
  for (std::size_t i = 0; i < array_json.size(); ++i) {
    const char c = array_json[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') {
      if (depth == 0) {
        object_start = i;
      }
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0 && object_start != std::string::npos) {
        objects.push_back(array_json.substr(object_start, i - object_start + 1));
        object_start = std::string::npos;
      }
    }
  }
  return objects;
}

std::map<std::string, std::string> parseFlatObject(const std::string& object_json) {
  std::map<std::string, std::string> values;
  std::size_t pos = 0;
  while (pos < object_json.size()) {
    const std::size_t key_start = object_json.find('"', pos);
    if (key_start == std::string::npos) {
      break;
    }
    std::size_t key_end = key_start + 1;
    bool escaped = false;
    for (; key_end < object_json.size(); ++key_end) {
      const char c = object_json[key_end];
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        break;
      }
    }
    if (key_end >= object_json.size()) {
      break;
    }
    const std::string key = jsonUnescape(object_json.substr(key_start + 1, key_end - key_start - 1));
    const std::size_t colon = object_json.find(':', key_end + 1);
    if (colon == std::string::npos) {
      break;
    }
    std::size_t value_start = skipWhitespace(object_json, colon + 1);
    if (value_start >= object_json.size()) {
      break;
    }

    std::string value;
    if (object_json[value_start] == '"') {
      ++value_start;
      std::size_t value_end = value_start;
      escaped = false;
      std::string raw;
      for (; value_end < object_json.size(); ++value_end) {
        const char c = object_json[value_end];
        if (escaped) {
          raw += '\\';
          raw += c;
          escaped = false;
        } else if (c == '\\') {
          escaped = true;
        } else if (c == '"') {
          break;
        } else {
          raw += c;
        }
      }
      value = jsonUnescape(raw);
      pos = value_end + 1;
    } else {
      std::size_t value_end = value_start;
      while (value_end < object_json.size() && object_json[value_end] != ',' && object_json[value_end] != '}') {
        ++value_end;
      }
      value = trimCopy(object_json.substr(value_start, value_end - value_start));
      pos = value_end + 1;
    }
    values[key] = value;
  }
  return values;
}

std::vector<PlatformCommand> parsePendingCommands(const std::string& response_json) {
  std::vector<PlatformCommand> commands;
  const std::string commands_array = extractArrayValue(response_json, "commands");
  if (commands_array.empty()) {
    return commands;
  }
  for (const std::string& object_json : splitTopLevelObjects(commands_array)) {
    PlatformCommand command{};
    command.command_id = extractStringValue(object_json, "command_id");
    command.device_id = extractStringValue(object_json, "device_id");
    command.command = extractStringValue(object_json, "command");
    command.raw_payload = extractObjectValue(object_json, "payload");
    if (!command.raw_payload.empty()) {
      command.payload = parseFlatObject(command.raw_payload);
    }
    if (!command.command_id.empty() && !command.command.empty()) {
      commands.push_back(std::move(command));
    }
  }
  return commands;
}

std::string buildPayload(const PlatformClientConfig& config, bool online, const PlatformStatus& status) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  out << "{";
  out << "\"device_id\":\"" << jsonEscape(config.device_id) << "\",";
  out << "\"role\":\"" << jsonEscape(config.role) << "\",";
  out << "\"display_name\":\"" << jsonEscape(config.display_name) << "\",";
  out << "\"online\":" << jsonBool(online) << ",";
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

std::size_t writeResponseCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  if (userdata == nullptr) {
    return 0;
  }
  auto* response = static_cast<std::string*>(userdata);
  response->append(ptr, size * nmemb);
  return size * nmemb;
}

std::string curlEscape(CURL* curl, const std::string& value) {
  char* escaped = curl_easy_escape(curl, value.c_str(), static_cast<int>(value.size()));
  if (escaped == nullptr) {
    return value;
  }
  std::string out = escaped;
  curl_free(escaped);
  return out;
}

curl_slist* appendAuthHeaders(curl_slist* headers, const PlatformClientConfig& config) {
  const std::string device_id_header = "X-ASDUN-Device-Id: " + config.device_id;
  const std::string device_token_header = "X-ASDUN-Device-Token: " + config.device_token;
  headers = curl_slist_append(headers, device_id_header.c_str());
  if (!config.device_token.empty()) {
    headers = curl_slist_append(headers, device_token_header.c_str());
  }
  return headers;
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

void PlatformClient::setCommandHandler(PlatformCommandHandler handler) {
  std::lock_guard<std::mutex> lock(mutex_);
  command_handler_ = std::move(handler);
}

std::string PlatformClient::statusUrl() const {
  return trimTrailingSlash(config_.base_url) + "/api/status";
}

std::string PlatformClient::pendingCommandsUrl() const {
  return trimTrailingSlash(config_.base_url) + "/api/commands/pending";
}

std::string PlatformClient::commandResultUrl(const std::string& command_id) const {
#ifdef USE_CLOUD_CLIENT
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return trimTrailingSlash(config_.base_url) + "/api/commands/" + command_id + "/result";
  }
  const std::string escaped = curlEscape(curl, command_id);
  curl_easy_cleanup(curl);
  return trimTrailingSlash(config_.base_url) + "/api/commands/" + escaped + "/result";
#else
  return trimTrailingSlash(config_.base_url) + "/api/commands/" + command_id + "/result";
#endif
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
    pollAndHandleCommands();

    std::unique_lock<std::mutex> lock(mutex_);
    if (cv_.wait_for(lock,
                     std::chrono::milliseconds(std::max(1000, config_.status_interval_ms)),
                     [this] { return !running_; })) {
      break;
    }
  }
}

bool PlatformClient::pollAndHandleCommands() {
#ifndef USE_CLOUD_CLIENT
  return false;
#else
  bool should_poll = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || !config_.command_poll_enabled) {
      return true;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto interval = std::chrono::milliseconds(std::max(1000, config_.command_poll_interval_ms));
    if (last_command_poll_time_.time_since_epoch().count() == 0 || now - last_command_poll_time_ >= interval) {
      last_command_poll_time_ = now;
      should_poll = true;
    }
  }
  if (!should_poll) {
    return true;
  }

  const auto commands = fetchPendingCommands();
  bool all_ok = true;
  for (const auto& command : commands) {
    const PlatformCommandResult result = executeCommand(command);
    all_ok = postCommandResult(command, result) && all_ok;
  }
  return all_ok;
#endif
}

std::vector<PlatformCommand> PlatformClient::fetchPendingCommands() const {
#ifndef USE_CLOUD_CLIENT
  return {};
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return {};
  }

  const int limit = std::max(1, std::min(100, config_.command_poll_limit));
  const std::string url = pendingCommandsUrl() + "?device_id=" + curlEscape(curl, config_.device_id) +
                          "&limit=" + std::to_string(limit);
  std::string response;
  curl_slist* headers = nullptr;
  headers = appendAuthHeaders(headers, config_);

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(500, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeResponseCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  bool should_log = false;
  if (!ok && config_.debug) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (last_command_error_log_time_.time_since_epoch().count() == 0 ||
        now - last_command_error_log_time_ >= std::chrono::seconds(30)) {
      last_command_error_log_time_ = now;
      should_log = true;
    }
  }
  if (should_log) {
    std::cerr << "[PlatformClient] command poll failed url=" << url
              << " curl=" << curl_easy_strerror(code)
              << " http=" << http_code << std::endl;
  }
  if (!ok) {
    return {};
  }
  return parsePendingCommands(response);
#endif
}

PlatformCommandResult PlatformClient::executeCommand(const PlatformCommand& command) {
  const std::string command_name = lowerCopy(trimCopy(command.command));
  if (command_name == "ping") {
    PlatformCommandResult result{};
    result.ok = true;
    result.message = "pong";
    result.result["handled_by"] = "PlatformClient";
    return result;
  }

  PlatformCommandHandler handler;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    handler = command_handler_;
  }
  if (handler) {
    return handler(command);
  }

  PlatformCommandResult result{};
  result.ok = false;
  result.message = "unsupported command: " + command.command;
  result.result["handled_by"] = "PlatformClient";
  return result;
}

bool PlatformClient::postCommandResult(const PlatformCommand& command, const PlatformCommandResult& result) const {
#ifndef USE_CLOUD_CLIENT
  (void)command;
  (void)result;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::map<std::string, std::string> result_values = result.result;
  result_values["command"] = command.command;
  result_values["command_id"] = command.command_id;
  if (result_values.find("handled_by") == result_values.end()) {
    result_values["handled_by"] = "PlatformClient";
  }

  std::ostringstream body;
  body << "{";
  body << "\"device_id\":\"" << jsonEscape(config_.device_id) << "\",";
  body << "\"ok\":" << jsonBool(result.ok) << ",";
  body << "\"message\":\"" << jsonEscape(result.message) << "\",";
  if (!result.ok) {
    body << "\"error\":\"" << jsonEscape(result.message) << "\",";
  }
  body << "\"result\":" << jsonObjectFromMap(result_values);
  body << "}";
  const std::string payload = body.str();
  const std::string url = commandResultUrl(command.command_id);

  curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = appendAuthHeaders(headers, config_);

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
    std::cerr << "[PlatformClient] command result post failed id=" << command.command_id
              << " curl=" << curl_easy_strerror(code)
              << " http=" << http_code << std::endl;
  } else if (ok && config_.debug) {
    std::cout << "[PlatformClient] command handled id=" << command.command_id
              << " command=" << command.command
              << " ok=" << (result.ok ? 1 : 0) << std::endl;
  }
  return ok;
#endif
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
  const std::string device_id_header = "X-ASDUN-Device-Id: " + config_.device_id;
  const std::string device_token_header = "X-ASDUN-Device-Token: " + config_.device_token;
  curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, device_id_header.c_str());
  if (!config_.device_token.empty()) {
    headers = curl_slist_append(headers, device_token_header.c_str());
  }

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
  bool should_log = false;
  if (!ok && config_.debug) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (last_error_log_time_.time_since_epoch().count() == 0 ||
        now - last_error_log_time_ >= std::chrono::seconds(30)) {
      last_error_log_time_ = now;
      should_log = true;
    }
  }
  if (should_log) {
    std::cerr << "[PlatformClient] status post failed url=" << url
              << " curl=" << curl_easy_strerror(code)
              << " http=" << http_code << std::endl;
  }
  return ok;
#endif
}

}  // namespace asdun
