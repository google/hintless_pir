// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hintless_simplepir/database_hwy.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/testing.h"
#include "hintless_simplepir/utils.h"
#include "lwe/lwe_symmetric_encryption.h"
#include "lwe/types.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"
#include "shell_encryption/testing/testing_prng.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

using rlwe::testing::StatusIs;
using ::testing::HasSubstr;
using Prng = rlwe::testing::TestingPrng;

const Parameters kParameters{
    .db_rows = 130,
    .db_cols = 32,
    .db_record_bit_size = 16,
    .lwe_secret_dim = 32,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 7,
    .lwe_error_variance = 8,
};

class DatabaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prng_ = std::make_unique<Prng>(0);
    lwe::Matrix lwe_query_pad =
        lwe::ExpandPad(kParameters.db_cols, kParameters.lwe_secret_dim,
                       prng_.get())
            .value();
    lwe_query_pad_ = std::make_unique<lwe::Matrix>(std::move(lwe_query_pad));
  }

  std::unique_ptr<Prng> prng_;
  std::unique_ptr<lwe::Matrix> lwe_query_pad_;
};

TEST(Database, Create) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  int num_shards = DivAndRoundUp(kParameters.db_record_bit_size,
                                 kParameters.lwe_plaintext_bit_size);
  ASSERT_EQ(database->NumShards(), num_shards);
  ASSERT_EQ(database->NumRecords(), 0);
}

TEST(Database, SetLweQueryPadFailsWithNullPointer) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  EXPECT_THAT(database->UpdateLweQueryPad(/*lwe_query_pad=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`lwe_query_pad` must not be null")));
}

TEST_F(DatabaseTest, AppendFailsIfRecordHasIncorrectSize) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  int record_size = DivAndRoundUp(kParameters.db_record_bit_size, 8);
  std::string short_record(record_size - 1, 0);
  EXPECT_THAT(database->Append(short_record),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`record` has incorrect size")));
  std::string long_record(record_size + 1, 0);
  EXPECT_THAT(database->Append(long_record),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`record` has incorrect size")));
}

TEST_F(DatabaseTest, AppendFailsIfDatabaseIsFull) {
  // Allows just two records in the database.
  const Parameters params{
      .db_rows = 1,
      .db_cols = 2,
      .db_record_bit_size = 8,
      .lwe_secret_dim = 2,
      .lwe_modulus_bit_size = 32,
      .lwe_plaintext_bit_size = 8,
      .lwe_error_variance = 8,
  };

  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(params));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  for (int64_t i = 0; i < params.db_rows * params.db_cols; ++i) {
    std::string record = testing::GenerateRandomRecord(params);
    ASSERT_OK(database->Append(record));
  }
  ASSERT_EQ(database->NumRecords(), params.db_rows * params.db_cols);

  int record_size = DivAndRoundUp(params.db_record_bit_size, 8);
  std::string dummy_record(record_size, 0);
  EXPECT_THAT(database->Append(dummy_record),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Database is full")));
}

TEST_F(DatabaseTest, AppendRecords) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  ASSERT_EQ(database->NumRecords(), 0);

  // Append a row of records to the database.
  for (int i = 0; i < kParameters.db_cols; ++i) {
    std::string record = testing::GenerateRandomRecord(kParameters);
    ASSERT_OK(database->Append(record));
    EXPECT_EQ(database->NumRecords(), i + 1);
    ASSERT_OK_AND_ASSIGN(std::string retrieved, database->Record(i));
    EXPECT_EQ(retrieved, record);
  }

  // Append another row of records to the database.
  for (int i = 0; i < kParameters.db_cols; ++i) {
    std::string record = testing::GenerateRandomRecord(kParameters);
    ASSERT_OK(database->Append(record));
    int index = kParameters.db_cols + i;
    EXPECT_EQ(database->NumRecords(), index + 1);
    ASSERT_OK_AND_ASSIGN(std::string retrieved, database->Record(index));
    EXPECT_EQ(retrieved, record);
  }

  int num_shards = DivAndRoundUp(kParameters.db_record_bit_size,
                                 kParameters.lwe_plaintext_bit_size);
  ASSERT_EQ(database->Data().size(), num_shards);
}

TEST_F(DatabaseTest, UpdateHintsFailsIfLweQueryPadIsNotSet) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  EXPECT_THAT(database->UpdateHints(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("LWE query pad not set")));
}

TEST_F(DatabaseTest, UpdateHints) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::CreateRandom(kParameters));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  ASSERT_EQ(database->NumRecords(), kParameters.db_cols * kParameters.db_rows);

  // Update the hint matrices and check.
  ASSERT_OK(database->UpdateHints());
  absl::Span<const Database::RawMatrix> data_matrices = database->Data();
  absl::Span<const Database::LweMatrix> hint_matrices = database->Hints();
  ASSERT_EQ(data_matrices.size(), hint_matrices.size());
  for (int i = 0; i < data_matrices.size(); ++i) {
    lwe::Matrix data_matrix =
        ExportRawMatrix(data_matrices[i], kParameters.db_rows,
                        kParameters.lwe_plaintext_bit_size);
    lwe::Matrix hint_matrix = ExportLweMatrix(hint_matrices[i]).transpose();
    lwe::Matrix expected_hint = data_matrix * (*this->lwe_query_pad_);
    EXPECT_EQ(hint_matrix, expected_hint);
  }
}

TEST_F(DatabaseTest, AccessRecordWithInvalidIndex) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::Create(kParameters));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  ASSERT_EQ(database->NumRecords(), 0);

  // The database is empty now, so accessing record will fail for any index
  EXPECT_THAT(database->Record(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`index` is out of range")));
  EXPECT_THAT(database->Record(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`index` is out of range")));

  // Append a record.
  std::string record = testing::GenerateRandomRecord(kParameters);
  ASSERT_OK(database->Append(record));
  EXPECT_THAT(database->Record(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`index` is out of range")));
  EXPECT_THAT(database->Record(1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`index` is out of range")));
}

TEST_F(DatabaseTest, InnerProductWith) {
  ASSERT_OK_AND_ASSIGN(auto database, Database::CreateRandom(kParameters));
  ASSERT_OK(database->UpdateLweQueryPad(this->lwe_query_pad_.get()));
  ASSERT_OK(database->UpdateHints());

  // Get the second column.
  std::vector<lwe::Integer> query(kParameters.db_cols, 0);
  query[1] = 1;
  ASSERT_OK_AND_ASSIGN(std::vector<Database::LweVector> product,
                       database->InnerProductWith(query));
  absl::Span<const Database::RawMatrix> data_matrices = database->Data();
  ASSERT_EQ(product.size(), data_matrices.size());

  Database::BlockType mask =
      (Database::BlockType{1} << kParameters.lwe_plaintext_bit_size) - 1;
  int num_values_per_block =
      sizeof(Database::BlockType) / sizeof(lwe::PlainInteger);
  for (int i = 0; i < product.size(); ++i) {
    ASSERT_EQ(product[i].size(), kParameters.db_rows);
    for (int j = 0; j < product[i].size(); ++j) {
      int block_idx = j / num_values_per_block;
      int block_pos = j % num_values_per_block;
      int base_bits = block_pos * 8 * sizeof(lwe::PlainInteger);
      Database::BlockType block = data_matrices[i][1][block_idx];
      lwe::Integer expected =
          static_cast<lwe::Integer>((block >> base_bits) & mask);
      EXPECT_EQ(product[i][j], expected);
    }
  }
}

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir
