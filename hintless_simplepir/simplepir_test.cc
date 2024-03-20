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

#include "hintless_simplepir/simplepir.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"

namespace hintless_pir {
namespace simplepir {
namespace {

using rlwe::testing::StatusIs;
using Prng = rlwe::SingleThreadHkdfPrng;

// Generates a random database with entries in [0, 2^record_bit_size).
// Uses an insecure RNG under the hood, does not impact security though.
absl::StatusOr<lwe::Matrix> GenerateDatabase(const Parems<Prng>* parems) {
  if (parems->DbRows() <= 0 || parems->DbCols() <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The number of rows, ", parems->DbRows(), ", and the number of cols, ",
        parems->DbCols(), " must be non-negative."));
  }
  if (parems->RecordBitSize() <= 0 ||
      parems->RecordBitSize() > lwe::kIntBitwidth) {
    return absl::InvalidArgumentError("Invalid record bit size.");
  }
  lwe::Matrix database =
      lwe::Matrix::Random(parems->DbRows(), parems->DbCols());
  database = database.array().unaryExpr(
      [&](lwe::Integer x) { return x % (1 << (parems->RecordBitSize())); });
  return database.eval();
}

class SimplePirTest : public testing::Test {
 protected:
  using Parems = simplepir::Parems<Prng>;
  // Samples a database and set of parems once for all tests
  void SetUp() override {
    constexpr int db_rows = 40;
    constexpr int db_cols = 40;
    constexpr int record_bit_size = 8;
    constexpr int lwe_secret_dim = 20;
    auto prng_seed = Prng::GenerateSeed().value();
    parems_ = std::make_unique<Parems>(Parems::Create(
        prng_seed, lwe_secret_dim, record_bit_size, db_rows, db_cols));
  }

  std::unique_ptr<Parems> parems_;
};

// Tests functional correctness of SimplePIR end-to-end.
TEST_F(SimplePirTest, EndToEndTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));

  // Arbitrary query value.
  int client_idx = 111;
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));

  // Client: Generates the client state and the PIR query.
  ClientState client_state;
  lwe::Vector query;
  ASSERT_OK_AND_ASSIGN(std::tie(client_state, query),
                       parems_->ClientQuery(client_idx, hint));

  // Server: Handles the query and returns the PIR response.
  ASSERT_OK_AND_ASSIGN(lwe::Vector response,
                       parems_->ServerResponse(database, query));

  // Client: Recover the required database record from the response.
  ASSERT_OK_AND_ASSIGN(lwe::Integer output,
                       parems_->ClientRecovery(client_state, response));
  EXPECT_EQ(output, database(client_state.row_idx, client_state.col_idx));
}

// Testing the various error-handling checks are all properly triggered
// for code coverage.
TEST_F(SimplePirTest, DatabaseGenNegRowsTest) {
  auto parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                               parems_->RecordBitSize(), -1, parems_->DbCols());
  auto database = GenerateDatabase(&parems);
  EXPECT_THAT(database, StatusIs(absl::StatusCode::kInvalidArgument,
                                 testing::HasSubstr("must be non-negative.")));
}

TEST_F(SimplePirTest, DatabaseGenNegColsTest) {
  auto parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                               parems_->RecordBitSize(), parems_->DbRows(), -1);
  auto database = GenerateDatabase(&parems);
  EXPECT_THAT(database, StatusIs(absl::StatusCode::kInvalidArgument,
                                 testing::HasSubstr("must be non-negative.")));
}

TEST_F(SimplePirTest, DatabaseGenNegBitSizeTest) {
  auto parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(), -1,
                               parems_->DbRows(), parems_->DbCols());
  auto database = GenerateDatabase(&parems);
  EXPECT_THAT(database,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Invalid record bit size")));
}

TEST_F(SimplePirTest, DatabaseGenTooLargeBitSizeTest) {
  auto parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                               lwe::kIntBitwidth + 1, parems_->DbRows(),
                               parems_->DbCols());
  auto database = GenerateDatabase(&parems);
  EXPECT_THAT(database,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Invalid record bit size")));
}

TEST_F(SimplePirTest, ServerPreprocessPubParemsRowsTest) {
  auto wrong_parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                                     parems_->RecordBitSize(),
                                     parems_->DbRows() + 1, parems_->DbCols());
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(&wrong_parems));
  auto status = parems_->ServerPreprocess(database);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("rows in the database")));
}

TEST_F(SimplePirTest, ServerPreprocessPubParemsColsTest) {
  auto wrong_parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                                     parems_->RecordBitSize(),
                                     parems_->DbRows(), parems_->DbCols() + 1);
  ASSERT_OK_AND_ASSIGN(lwe::Matrix wrong_database,
                       GenerateDatabase(&wrong_parems));
  auto status = parems_->ServerPreprocess(wrong_database);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("cols in the database")));
}

TEST_F(SimplePirTest, ClientQueryNegIdxTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));
  auto status = parems_->ClientQuery(-1, hint);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("out of range")));
}

TEST_F(SimplePirTest, ClientQueryLargeIdxTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));
  auto status =
      parems_->ClientQuery(parems_->DbCols() * parems_->DbRows(), hint);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("out of range")));
}

TEST_F(SimplePirTest, ClientQueryHintWrongColsTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));
  auto wrong_parems = Parems::Create(
      parems_->Seed(), parems_->LweSecretDim() - 1, parems_->RecordBitSize(),
      parems_->DbRows(), parems_->DbCols());
  auto status = wrong_parems.ClientQuery(parems_->DbCols() + 1, hint);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("number of cols in the hint")));
}

TEST_F(SimplePirTest, ClientQueryHintWrongRowsTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));
  auto wrong_parems = Parems::Create(parems_->Seed(), parems_->LweSecretDim(),
                                     parems_->RecordBitSize(),
                                     parems_->DbRows() + 1, parems_->DbCols());
  auto status = wrong_parems.ClientQuery(parems_->DbCols() + 1, hint);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("number of rows in the hint")));
}

// We only check if the server response has the wrong number of rows
// because it is a Eigen `Vector`, e.g. Eigen enforces that it will
// have a single column.
TEST_F(SimplePirTest, ClientRecoveryServerResponseWrongRowsTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));

  ClientState client_state;
  lwe::Vector query;
  ASSERT_OK_AND_ASSIGN(std::tie(client_state, query),
                       parems_->ClientQuery(parems_->DbCols() + 1, hint));

  ASSERT_OK_AND_ASSIGN(lwe::Vector response,
                       parems_->ServerResponse(database, query));
  // Creating a malformed_query that has too many rows
  lwe::Matrix malformed_query =
      lwe::Matrix::Zero(2 * response.rows(), response.cols());
  auto status = parems_->ClientRecovery(client_state, malformed_query);
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("number of rows in the server response")));
}

TEST_F(SimplePirTest, ClientRecoveryServerResponseWrongDecPartTest) {
  ASSERT_OK_AND_ASSIGN(lwe::Matrix database, GenerateDatabase(parems_.get()));
  ASSERT_OK_AND_ASSIGN(lwe::Matrix hint, parems_->ServerPreprocess(database));

  ClientState client_state;
  lwe::Vector query;
  ASSERT_OK_AND_ASSIGN(std::tie(client_state, query),
                       parems_->ClientQuery(parems_->DbCols() + 1, hint));

  ClientState malformed_client_state =
      ClientState{.row_idx = client_state.row_idx,
                  .col_idx = client_state.col_idx,
                  .decryption_part = std::move(lwe::Matrix::Zero(2, 1))};
  ASSERT_OK_AND_ASSIGN(lwe::Vector response,
                       parems_->ServerResponse(database, query));
  auto status = parems_->ClientRecovery(malformed_client_state, response);
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("The decryption part is not 1 x 1.")));
}

}  // namespace
}  // namespace simplepir
}  // namespace hintless_pir
