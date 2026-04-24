#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cloud/CloudTypes.hpp"

namespace asdun {

class CloudClient {
 public:
  explicit CloudClient(CloudClientConfig config);
  ~CloudClient();

  CloudClient(const CloudClient&) = delete;
  CloudClient& operator=(const CloudClient&) = delete;

  bool start();
  void stop();
  bool enabled() const;
  bool submit(const CloudAnalysisRequest& request);
  bool enrollPerson(const std::string& name, const std::vector<std::string>& image_paths, bool replace) const;
  bool deletePerson(const std::string& name) const;
  std::vector<std::string> listPeople() const;
  std::vector<ExternalTrackAnalysis> pollCompleted();

 private:
  bool analyze(const CloudAnalysisRequest& request, ExternalTrackAnalysis* out) const;
  bool selectActiveServer();
  bool probeServer(const std::string& server_url, std::string* response) const;
  std::vector<std::string> candidateServerUrls() const;
  std::string baseUrl() const;
  std::string analyzeUrl() const;
  std::string enrollUrl() const;
  std::string deleteUrl() const;
  std::string galleryUrl() const;
  void workerLoop();

  CloudClientConfig config_{};
  std::string active_server_url_{};
  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  std::deque<CloudAnalysisRequest> queue_{};
  std::deque<ExternalTrackAnalysis> completed_{};
  std::unordered_map<int, std::uint64_t> last_submit_ms_{};
  std::unordered_set<int> inflight_tracks_{};
  std::thread worker_{};
  bool running_{false};
};

}  // namespace asdun
