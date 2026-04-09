#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "core/Types.hpp"

namespace asdun {

class Database {
 public:
  Database() = default;
  ~Database();

  bool open(const std::string& db_path);
  void close();
  bool initSchema();

  bool personExists(const std::string& name, int* person_id = nullptr) const;
  bool upsertPerson(const std::string& name, int* out_person_id);
  bool deletePersonAndEmbeddings(const std::string& name);

  bool insertEmbedding(int person_id,
                       const std::vector<float>& embedding,
                       const std::string& image_path,
                       float quality_score,
                       const std::string& model_tag);

  std::vector<std::string> listImagePathsByPerson(const std::string& name) const;
  std::vector<StoredEmbedding> loadAllEmbeddings() const;

 private:
  bool ensureEmbeddingsColumn(const std::string& column_name, const std::string& column_def);
  static std::vector<std::uint8_t> floatsToBlob(const std::vector<float>& vec);
  static std::vector<float> blobToFloats(const void* data, int bytes);
  static std::int64_t nowEpochSeconds();

  sqlite3* db_{nullptr};
};

}  // namespace asdun
