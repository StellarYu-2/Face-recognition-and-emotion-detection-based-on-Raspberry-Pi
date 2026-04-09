#include "storage/Database.hpp"

#include <chrono>
#include <cstring>
#include <iostream>

namespace asdun {

namespace {

bool execSql(sqlite3* db, const char* sql) {
  char* err_msg = nullptr;
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
  if (rc != SQLITE_OK) {
    if (err_msg != nullptr) {
      std::cerr << "[DB] SQL error: " << err_msg << std::endl;
      sqlite3_free(err_msg);
    }
    return false;
  }
  return true;
}

bool tableHasColumn(sqlite3* db, const char* table_name, const std::string& column_name) {
  if (db == nullptr) {
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  const std::string sql = "PRAGMA table_info(" + std::string(table_name) + ");";
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
    return false;
  }

  bool found = false;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const unsigned char* text = sqlite3_column_text(stmt, 1);
    if (text != nullptr && column_name == reinterpret_cast<const char*>(text)) {
      found = true;
      break;
    }
  }
  sqlite3_finalize(stmt);
  return found;
}

}  // namespace

Database::~Database() { close(); }

bool Database::open(const std::string& db_path) {
  close();
  const int rc = sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  if (rc != SQLITE_OK || db_ == nullptr) {
    std::cerr << "[DB] Failed to open db: " << db_path << std::endl;
    close();
    return false;
  }
  return execSql(db_, "PRAGMA foreign_keys = ON;");
}

void Database::close() {
  if (db_ != nullptr) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
}

bool Database::initSchema() {
  if (db_ == nullptr) {
    return false;
  }
  constexpr const char* kSql = R"SQL(
CREATE TABLE IF NOT EXISTS persons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  image_path TEXT,
  quality_score REAL DEFAULT 0,
  model_tag TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
);
  )SQL";
  if (!execSql(db_, kSql)) {
    return false;
  }
  return ensureEmbeddingsColumn("model_tag", "TEXT NOT NULL DEFAULT ''");
}

bool Database::personExists(const std::string& name, int* person_id) const {
  if (db_ == nullptr) {
    return false;
  }
  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql = "SELECT id FROM persons WHERE name = ? LIMIT 1;";
  if (sqlite3_prepare_v2(db_, kSql, -1, &stmt, nullptr) != SQLITE_OK) {
    return false;
  }
  sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
  const int rc = sqlite3_step(stmt);
  bool exists = false;
  if (rc == SQLITE_ROW) {
    exists = true;
    if (person_id != nullptr) {
      *person_id = sqlite3_column_int(stmt, 0);
    }
  }
  sqlite3_finalize(stmt);
  return exists;
}

bool Database::upsertPerson(const std::string& name, int* out_person_id) {
  if (db_ == nullptr) {
    return false;
  }
  int existing_id = 0;
  if (personExists(name, &existing_id)) {
    if (out_person_id != nullptr) {
      *out_person_id = existing_id;
    }
    return true;
  }

  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql = "INSERT INTO persons(name, created_at, updated_at) VALUES(?, ?, ?);";
  if (sqlite3_prepare_v2(db_, kSql, -1, &stmt, nullptr) != SQLITE_OK) {
    return false;
  }
  const auto now = nowEpochSeconds();
  sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 2, now);
  sqlite3_bind_int64(stmt, 3, now);
  const int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    return false;
  }
  if (out_person_id != nullptr) {
    *out_person_id = static_cast<int>(sqlite3_last_insert_rowid(db_));
  }
  return true;
}

bool Database::deletePersonAndEmbeddings(const std::string& name) {
  if (db_ == nullptr) {
    return false;
  }
  if (!execSql(db_, "BEGIN TRANSACTION;")) {
    return false;
  }

  sqlite3_stmt* find_stmt = nullptr;
  constexpr const char* kFindSql = "SELECT id FROM persons WHERE name = ? LIMIT 1;";
  if (sqlite3_prepare_v2(db_, kFindSql, -1, &find_stmt, nullptr) != SQLITE_OK) {
    execSql(db_, "ROLLBACK;");
    return false;
  }
  sqlite3_bind_text(find_stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
  const int find_rc = sqlite3_step(find_stmt);
  if (find_rc != SQLITE_ROW) {
    sqlite3_finalize(find_stmt);
    execSql(db_, "COMMIT;");
    return true;
  }
  const int person_id = sqlite3_column_int(find_stmt, 0);
  sqlite3_finalize(find_stmt);

  sqlite3_stmt* del_emb_stmt = nullptr;
  constexpr const char* kDelEmbSql = "DELETE FROM embeddings WHERE person_id = ?;";
  if (sqlite3_prepare_v2(db_, kDelEmbSql, -1, &del_emb_stmt, nullptr) != SQLITE_OK) {
    execSql(db_, "ROLLBACK;");
    return false;
  }
  sqlite3_bind_int(del_emb_stmt, 1, person_id);
  const int del_emb_rc = sqlite3_step(del_emb_stmt);
  sqlite3_finalize(del_emb_stmt);
  if (del_emb_rc != SQLITE_DONE) {
    execSql(db_, "ROLLBACK;");
    return false;
  }

  sqlite3_stmt* del_person_stmt = nullptr;
  constexpr const char* kDelPersonSql = "DELETE FROM persons WHERE id = ?;";
  if (sqlite3_prepare_v2(db_, kDelPersonSql, -1, &del_person_stmt, nullptr) != SQLITE_OK) {
    execSql(db_, "ROLLBACK;");
    return false;
  }
  sqlite3_bind_int(del_person_stmt, 1, person_id);
  const int del_person_rc = sqlite3_step(del_person_stmt);
  sqlite3_finalize(del_person_stmt);
  if (del_person_rc != SQLITE_DONE) {
    execSql(db_, "ROLLBACK;");
    return false;
  }

  return execSql(db_, "COMMIT;");
}

bool Database::insertEmbedding(int person_id,
                               const std::vector<float>& embedding,
                               const std::string& image_path,
                               float quality_score,
                               const std::string& model_tag) {
  if (db_ == nullptr) {
    return false;
  }
  const auto blob = floatsToBlob(embedding);

  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql =
      "INSERT INTO embeddings(person_id, embedding, image_path, quality_score, model_tag, created_at) "
      "VALUES(?, ?, ?, ?, ?, ?);";
  if (sqlite3_prepare_v2(db_, kSql, -1, &stmt, nullptr) != SQLITE_OK) {
    return false;
  }
  sqlite3_bind_int(stmt, 1, person_id);
  sqlite3_bind_blob(stmt, 2, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
  sqlite3_bind_text(stmt, 3, image_path.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, 4, static_cast<double>(quality_score));
  sqlite3_bind_text(stmt, 5, model_tag.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64(stmt, 6, nowEpochSeconds());
  const int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE;
}

std::vector<std::string> Database::listImagePathsByPerson(const std::string& name) const {
  std::vector<std::string> paths;
  if (db_ == nullptr) {
    return paths;
  }
  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql = R"SQL(
SELECT e.image_path
FROM embeddings e
JOIN persons p ON p.id = e.person_id
WHERE p.name = ?;
  )SQL";
  if (sqlite3_prepare_v2(db_, kSql, -1, &stmt, nullptr) != SQLITE_OK) {
    return paths;
  }
  sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const unsigned char* text = sqlite3_column_text(stmt, 0);
    if (text != nullptr) {
      paths.emplace_back(reinterpret_cast<const char*>(text));
    }
  }
  sqlite3_finalize(stmt);
  return paths;
}

std::vector<StoredEmbedding> Database::loadAllEmbeddings() const {
  std::vector<StoredEmbedding> out;
  if (db_ == nullptr) {
    return out;
  }
  sqlite3_stmt* stmt = nullptr;
  constexpr const char* kSql = R"SQL(
SELECT p.id, p.name, e.embedding, e.image_path, e.quality_score
     , e.model_tag
FROM embeddings e
JOIN persons p ON p.id = e.person_id;
  )SQL";
  if (sqlite3_prepare_v2(db_, kSql, -1, &stmt, nullptr) != SQLITE_OK) {
    return out;
  }
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    StoredEmbedding row{};
    row.person_id = sqlite3_column_int(stmt, 0);
    const unsigned char* name = sqlite3_column_text(stmt, 1);
    row.person_name = (name != nullptr) ? reinterpret_cast<const char*>(name) : "";

    const void* blob_data = sqlite3_column_blob(stmt, 2);
    const int blob_bytes = sqlite3_column_bytes(stmt, 2);
    row.embedding = blobToFloats(blob_data, blob_bytes);

    const unsigned char* path = sqlite3_column_text(stmt, 3);
    row.image_path = (path != nullptr) ? reinterpret_cast<const char*>(path) : "";
    row.quality_score = static_cast<float>(sqlite3_column_double(stmt, 4));
    const unsigned char* model_tag = sqlite3_column_text(stmt, 5);
    row.model_tag = (model_tag != nullptr) ? reinterpret_cast<const char*>(model_tag) : "";
    out.push_back(std::move(row));
  }
  sqlite3_finalize(stmt);
  return out;
}

bool Database::ensureEmbeddingsColumn(const std::string& column_name, const std::string& column_def) {
  if (db_ == nullptr) {
    return false;
  }
  if (tableHasColumn(db_, "embeddings", column_name)) {
    return true;
  }
  const std::string sql = "ALTER TABLE embeddings ADD COLUMN " + column_name + " " + column_def + ";";
  return execSql(db_, sql.c_str());
}

std::vector<std::uint8_t> Database::floatsToBlob(const std::vector<float>& vec) {
  std::vector<std::uint8_t> blob(vec.size() * sizeof(float));
  if (!vec.empty()) {
    std::memcpy(blob.data(), vec.data(), blob.size());
  }
  return blob;
}

std::vector<float> Database::blobToFloats(const void* data, int bytes) {
  std::vector<float> out;
  if (data == nullptr || bytes <= 0 || (bytes % static_cast<int>(sizeof(float)) != 0)) {
    return out;
  }
  out.resize(static_cast<std::size_t>(bytes) / sizeof(float));
  std::memcpy(out.data(), data, static_cast<std::size_t>(bytes));
  return out;
}

std::int64_t Database::nowEpochSeconds() {
  const auto now = std::chrono::system_clock::now();
  return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
}

}  // namespace asdun
