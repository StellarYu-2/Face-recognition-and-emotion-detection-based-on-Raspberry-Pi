#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace asdun {

struct PlatformClientConfig {
  bool enabled{false};
  std::string base_url{"http://127.0.0.1:9000"};
  std::string device_id{"pi-01"};
  std::string role{"raspberry_pi"};
  std::string display_name{"asdun@asdun"};
  int status_interval_ms{5000};
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

 private:
  bool postStatus(bool online, const PlatformStatus& status) const;
  std::string statusUrl() const;
  void workerLoop();

  PlatformClientConfig config_{};
  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  PlatformStatus status_{};
  std::thread worker_{};
  bool running_{false};
};

}  // namespace asdun
