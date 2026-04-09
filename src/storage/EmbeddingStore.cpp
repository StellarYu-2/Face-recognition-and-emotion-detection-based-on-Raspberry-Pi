#include "storage/EmbeddingStore.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>

#include "engine/ConfidenceMapper.hpp"
#include "engine/FaceRecognizer.hpp"

namespace asdun {

EmbeddingStore::EmbeddingStore(Database& db) : db_(db) {}

void EmbeddingStore::setActiveModelTag(std::string model_tag) { active_model_tag_ = std::move(model_tag); }

bool EmbeddingStore::reload() {
  gallery_ = db_.loadAllEmbeddings();
  rebuildTemplates();
  return true;
}

const std::vector<StoredEmbedding>& EmbeddingStore::gallery() const { return gallery_; }

IdentityResult EmbeddingStore::match(const std::vector<float>& query_embedding, float threshold, float tau) const {
  IdentityResult result{};
  if (query_embedding.empty() || templates_.empty()) {
    result.name = "Unknown";
    result.known = false;
    result.distance = 1.0F;
    result.conf_pct = 0.0F;
    result.debug_summary = active_model_tag_.empty() ? "no_templates" : ("no_templates_for_model=" + active_model_tag_);
    return result;
  }

  struct RankedDistance {
    std::string person_name;
    float distance{std::numeric_limits<float>::max()};
    int sample_count{0};
  };
  std::vector<RankedDistance> ranked;
  ranked.reserve(templates_.size());

  for (const auto& item : templates_) {
    const float d = FaceRecognizer::l2Distance(query_embedding, item.mean_embedding);
    ranked.push_back(RankedDistance{item.person_name, d, item.sample_count});
  }
  std::sort(ranked.begin(), ranked.end(), [](const RankedDistance& a, const RankedDistance& b) {
    return a.distance < b.distance;
  });

  if (ranked.empty()) {
    result.name = "Unknown";
    result.known = false;
    result.distance = 1.0F;
    result.conf_pct = 0.0F;
    result.debug_summary = "ranked_empty";
    return result;
  }

  const float best_distance = ranked.front().distance;
  const std::string& best_name = ranked.front().person_name;
  const float known_conf = ConfidenceMapper::distanceToPercent(best_distance, threshold, tau);
  const bool is_known = best_distance <= threshold;
  result.name = is_known ? best_name : "Unknown";
  result.known = is_known;
  result.distance = best_distance;
  result.conf_pct = is_known ? known_conf : (100.0F - known_conf);
  result.matched_sample_count = ranked.front().sample_count;

  std::ostringstream oss;
  oss << "model=" << (active_model_tag_.empty() ? "any" : active_model_tag_) << " best=" << ranked.front().person_name << ":"
      << ranked.front().distance << " samples=" << ranked.front().sample_count;
  if (ranked.size() > 1) {
    oss << " next=" << ranked[1].person_name << ":" << ranked[1].distance;
  }
  result.debug_summary = oss.str();
  return result;
}

void EmbeddingStore::rebuildTemplates() {
  templates_.clear();
  struct Aggregate {
    int person_id{0};
    std::string person_name;
    std::string model_tag;
    std::vector<float> sum_embedding;
    int sample_count{0};
  };

  std::unordered_map<std::string, Aggregate> grouped;
  for (const auto& item : gallery_) {
    if (!active_model_tag_.empty() && item.model_tag != active_model_tag_) {
      continue;
    }
    if (item.embedding.empty()) {
      continue;
    }

    const std::string key = std::to_string(item.person_id) + "|" + item.model_tag;
    auto& agg = grouped[key];
    if (agg.sample_count == 0) {
      agg.person_id = item.person_id;
      agg.person_name = item.person_name;
      agg.model_tag = item.model_tag;
      agg.sum_embedding.assign(item.embedding.size(), 0.0F);
    }
    if (agg.sum_embedding.size() != item.embedding.size()) {
      continue;
    }
    for (std::size_t i = 0; i < item.embedding.size(); ++i) {
      agg.sum_embedding[i] += item.embedding[i];
    }
    agg.sample_count += 1;
  }

  templates_.reserve(grouped.size());
  for (auto& [_, agg] : grouped) {
    if (agg.sample_count <= 0 || agg.sum_embedding.empty()) {
      continue;
    }
    for (float& v : agg.sum_embedding) {
      v /= static_cast<float>(agg.sample_count);
    }
    l2Normalize(agg.sum_embedding);
    templates_.push_back(PersonTemplate{
        agg.person_id, agg.person_name, agg.model_tag, std::move(agg.sum_embedding), agg.sample_count});
  }
}

void EmbeddingStore::l2Normalize(std::vector<float>& v) {
  float norm2 = 0.0F;
  for (const float x : v) {
    norm2 += x * x;
  }
  const float norm = std::sqrt(norm2);
  if (norm <= 1e-6F) {
    return;
  }
  for (float& x : v) {
    x /= norm;
  }
}

}  // namespace asdun
