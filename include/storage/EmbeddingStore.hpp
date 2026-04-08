#pragma once

#include <vector>

#include "core/Types.hpp"
#include "storage/Database.hpp"

namespace asdun {

class EmbeddingStore {
 public:
  explicit EmbeddingStore(Database& db);

  bool reload();
  const std::vector<StoredEmbedding>& gallery() const;

  IdentityResult match(const std::vector<float>& query_embedding, float threshold, float tau) const;

 private:
  Database& db_;
  std::vector<StoredEmbedding> gallery_{};
};

}  // namespace asdun

