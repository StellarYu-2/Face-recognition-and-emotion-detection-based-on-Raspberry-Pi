#pragma once

#include <string>
#include <vector>

#include "core/Types.hpp"
#include "storage/Database.hpp"

namespace asdun {

class EmbeddingStore {
 public:
  explicit EmbeddingStore(Database& db);

  void setActiveModelTag(std::string model_tag);
  bool reload();
  const std::vector<StoredEmbedding>& gallery() const;

  IdentityResult match(const std::vector<float>& query_embedding, float threshold, float tau, float margin_threshold) const;

 private:
  struct PersonTemplate {
    int person_id{0};
    std::string person_name;
    std::string model_tag;
    std::vector<float> mean_embedding;
    int sample_count{0};
  };

  void rebuildTemplates();
  static void l2Normalize(std::vector<float>& v);

  Database& db_;
  std::string active_model_tag_{};
  std::vector<StoredEmbedding> gallery_{};
  std::vector<PersonTemplate> templates_{};
};

}  // namespace asdun
