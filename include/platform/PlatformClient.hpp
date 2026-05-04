#pragma once

#include <condition_variable>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace asdun {

struct PlatformClientConfig {
  bool enabled{false};
  std::string base_url{"http://127.0.0.1:9000"};
  std::string device_id{"pi-01"};
  std::string device_token{};
  std::string role{"raspberry_pi"};
  std::string display_name{"asdun@asdun"};
  int status_interval_ms{5000};
  bool command_poll_enabled{true};
  int command_poll_interval_ms{5000};
  int command_poll_limit{5};
  int connect_timeout_ms{1000};
  int timeout_ms{2000};
  bool debug{false};
};

struct PlatformStatus {
  std::string mode{"idle"};
  std::string inference_mode{"local"};
  bool cloud_connected{false};
  int active_tracks{0};
  double fps{0.0};
};

struct PlatformCommand {
  std::string command_id;
  std::string device_id;
  std::string command;
  std::map<std::string, std::string> payload;
  std::string raw_payload;
};

struct PlatformCommandResult {
  bool ok{true};
  std::string message{"ok"};
  std::map<std::string, std::string> result;
};

using PlatformCommandHandler = std::function<PlatformCommandResult(const PlatformCommand&)>;

class PlatformClient {
 public:
  explicit PlatformClient(PlatformClientConfig config);
  ~PlatformClient();

  PlatformClient(const PlatformClient&) = delete;
  PlatformClient& operator=(const PlatformClient&) = delete;

  bool start();
  void stop();
  bool enabled() const;
  void updateStatus(const PlatformStatus& status);
  void setCommandHandler(PlatformCommandHandler handler);

 private:
  bool postStatus(bool online, const PlatformStatus& status) const;
  bool pollAndHandleCommands();
  std::vector<PlatformCommand> fetchPendingCommands() const;
  PlatformCommandResult executeCommand(const PlatformCommand& command);
  bool postCommandResult(const PlatformCommand& command, const PlatformCommandResult& result) const;
  std::string statusUrl() const;
  std::string pendingCommandsUrl() const;
  std::string commandResultUrl(const std::string& command_id) const;
  void workerLoop();

  PlatformClientConfig config_{};
  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  PlatformStatus status_{};
  PlatformCommandHandler command_handler_{};
  std::thread worker_{};
  mutable std::chrono::steady_clock::time_point last_error_log_time_{};
  mutable std::chrono::steady_clock::time_point last_command_error_log_time_{};
  std::chrono::steady_clock::time_point last_command_poll_time_{};
  bool running_{false};
};

}  // namespace asdun
