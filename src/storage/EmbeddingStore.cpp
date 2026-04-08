#include "storage/EmbeddingStore.hpp"

#include <limits>

#include "engine/ConfidenceMapper.hpp"
#include "engine/FaceRecognizer.hpp"

namespace asdun {

EmbeddingStore::EmbeddingStore(Database& db) : db_(db) {}

bool EmbeddingStore::reload() {
  gallery_ = db_.loadAllEmbeddings();
  return true;
}

const std::vector<StoredEmbedding>& EmbeddingStore::gallery() const { return gallery_; }

IdentityResult EmbeddingStore::match(const std::vector<float>& query_embedding, float threshold, float tau) const {
  IdentityResult result{};
  if (query_embedding.empty() || gallery_.empty()) {
    result.name = "Unknown";
    result.known = false;
    result.distance = 1.0F;
    result.conf_pct = 0.0F;
    return result;
  }

  float best_distance = std::numeric_limits<float>::max();
  std::string best_name = "Unknown";
  for (const auto& item : gallery_) {
    const float d = FaceRecognizer::l2Distance(query_embedding, item.embedding);
    if (d < best_distance) {
      best_distance = d;
      best_name = item.person_name;
    }
  }

  const float known_conf = ConfidenceMapper::distanceToPercent(best_distance, threshold, tau);
  const bool is_known = best_distance <= threshold;
  result.name = is_known ? best_name : "Unknown";
  result.known = is_known;
  result.distance = best_distance;
  result.conf_pct = is_known ? known_conf : (100.0F - known_conf);
  return result;
}

}  // namespace asdun

