#include "storage/EmbeddingStore.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
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

IdentityResult EmbeddingStore::match(const std::vector<float>& query_embedding,
                                     float threshold,
                                     float tau,
                                     float margin_threshold) const {
  IdentityResult result{};
  if (query_embedding.empty() || gallery_.empty()) {
    result.name = "Unknown";
    result.known = false;
    result.distance = 1.0F;
    result.conf_pct = 0.0F;
    result.measured = false;
    result.debug_summary =
        "decision=no_templates shown=Unknown model=" + (active_model_tag_.empty() ? std::string("any") : active_model_tag_);
    return result;
  }

  struct RankedDistance {
    std::string person_name;
    float distance{std::numeric_limits<float>::max()};
    int sample_count{0};
  };
  std::unordered_map<std::string, RankedDistance> best_by_person;
  for (const auto& item : gallery_) {
    if (!active_model_tag_.empty() && item.model_tag != active_model_tag_) {
      continue;
    }
    if (item.embedding.empty()) {
      continue;
    }

    const float d = FaceRecognizer::l2Distance(query_embedding, item.embedding);
    auto [it, inserted] =
        best_by_person.emplace(item.person_name, RankedDistance{item.person_name, d, 1});
    if (!inserted) {
      it->second.sample_count += 1;
      if (d < it->second.distance) {
        it->second.distance = d;
      }
    }
  }

  std::vector<RankedDistance> ranked;
  ranked.reserve(best_by_person.size());
  for (const auto& [_, item] : best_by_person) {
    ranked.push_back(item);
  }
  std::sort(ranked.begin(), ranked.end(), [](const RankedDistance& a, const RankedDistance& b) {
    return a.distance < b.distance;
  });

  if (ranked.empty()) {
    result.name = "Unknown";
    result.known = false;
    result.distance = 1.0F;
    result.conf_pct = 0.0F;
    result.measured = false;
    result.debug_summary = "decision=empty_gallery shown=Unknown";
    return result;
  }

  const float best_distance = ranked.front().distance;
  const std::string& best_name = ranked.front().person_name;
  const float known_conf = ConfidenceMapper::distanceToPercent(best_distance, threshold, tau);
  const float next_distance = (ranked.size() > 1) ? ranked[1].distance : std::numeric_limits<float>::max();
  const float margin = (ranked.size() > 1) ? (next_distance - best_distance) : std::numeric_limits<float>::max();
  const bool pass_threshold = best_distance <= threshold;
  const bool pass_margin = (ranked.size() <= 1) || (margin >= margin_threshold);
  const bool is_known = pass_threshold && pass_margin;
  const bool is_ambiguous = pass_threshold && !pass_margin;
  const std::string decision = is_known ? "accept" : (is_ambiguous ? "ambiguous" : "reject");
  const std::string reason = is_known ? "passed" : (is_ambiguous ? "gap_too_small" : "distance_too_high");
  result.name = is_known ? best_name : "Unknown";
  result.known = is_known;
  result.distance = best_distance;
  result.margin = std::isfinite(margin) ? margin : 999.0F;
  result.measured = !is_ambiguous;
  result.conf_pct = is_known ? known_conf : (is_ambiguous ? 0.0F : (100.0F - known_conf));
  result.matched_sample_count = ranked.front().sample_count;

  std::ostringstream oss;
  oss << std::fixed;
  oss << "decision=" << decision << " shown=" << result.name << " model="
      << (active_model_tag_.empty() ? "any" : active_model_tag_) << " top1=" << ranked.front().person_name
      << " sim=" << std::setprecision(1) << known_conf << " dist=" << std::setprecision(3) << ranked.front().distance;
  if (ranked.size() > 1) {
    oss << " top2=" << ranked[1].person_name << " dist2=" << ranked[1].distance << " gap=" << margin;
  } else {
    oss << " top2=- dist2=- gap=-";
  }
  oss << " samples=" << ranked.front().sample_count << " reason=" << reason << " dist_ok=" << (pass_threshold ? 1 : 0)
      << " gap_ok=" << (pass_margin ? 1 : 0);
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
