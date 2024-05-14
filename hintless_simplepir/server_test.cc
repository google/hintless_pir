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

#include "hintless_simplepir/server.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/database_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/testing.h"
#include "hintless_simplepir/utils.h"
#include "linpir/parameters.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

using rlwe::testing::StatusIs;
using ::testing::HasSubstr;
using RlweInteger = Parameters::RlweInteger;
using Prng = rlwe::SingleThreadHkdfPrng;

const Parameters kParameters{
    .db_rows = 128,
    .db_cols = 32,
    .db_record_bit_size = 8,
    .lwe_secret_dim = 32,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 8,
    .lwe_error_variance = 8,
    .linpir_params =
        linpir::RlweParameters<RlweInteger>{
            .log_n = 10,
            .qs = {536813569ULL},
            .ts = {12289, 65537},
            .gadget_log_bs = {8},
            .error_variance = 8,
            .prng_type = rlwe::PRNG_TYPE_HKDF,
            .rows_per_block = 512,
        },
    .prng_type = rlwe::PRNG_TYPE_HKDF,
};

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Creates a server and fill in the database with random records.
    server_ = Server::Create(kParameters).value();
    auto database = server_->GetDatabase();
    for (int64_t i = 0; i < kParameters.db_rows * kParameters.db_cols; ++i) {
      CHECK_OK(database->Append(testing::GenerateRandomRecord(kParameters)));
    }
  }

  std::unique_ptr<Server> server_;
};

TEST(Server, CreateFailsIfInvalidPrngType) {
  const Parameters params{
      .prng_type = rlwe::PRNG_TYPE_INVALID,
  };
  EXPECT_THAT(Server::Create(params),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid PRNG type")));
}

TEST(Server, Create) {
  ASSERT_OK_AND_ASSIGN(auto server, Server::Create(kParameters));
  auto database = server->GetDatabase();
  ASSERT_NE(database, nullptr);

  int num_shards = DivAndRoundUp(kParameters.db_record_bit_size,
                                 kParameters.lwe_plaintext_bit_size);
  ASSERT_EQ(database->NumShards(), num_shards);
  ASSERT_EQ(database->NumRecords(), 0);
}

TEST_F(ServerTest, Preprocess) {
  // Check that the server's public parameters are generated and the database
  // structures are preprocessed.
  ASSERT_OK(this->server_->Preprocess());
  HintlessPirServerPublicParams pub_params = this->server_->GetPublicParams();
  int expected_prng_seed_length = Prng::SeedLength();
  EXPECT_EQ(pub_params.prng_seed_lwe_query_pad().size(),
            expected_prng_seed_length);
  EXPECT_EQ(pub_params.prng_seed_linpir_gk_pad().size(),
            expected_prng_seed_length);
  int num_linpir_instances = kParameters.linpir_params.ts.size();
  ASSERT_EQ(pub_params.prng_seed_linpir_ct_pads_size(), num_linpir_instances);
  for (int i = 0; i < num_linpir_instances; ++i) {
    EXPECT_EQ(pub_params.prng_seed_linpir_ct_pads(i).size(),
              expected_prng_seed_length);
  }

  const lwe::Matrix* lwe_matrix = this->server_->LweQueryPad();
  ASSERT_NE(lwe_matrix, nullptr);
  ASSERT_EQ(lwe_matrix->rows(), kParameters.db_cols);
  ASSERT_EQ(lwe_matrix->cols(), kParameters.lwe_secret_dim);

  Database* database = this->server_->GetDatabase();
  ASSERT_NE(database, nullptr);
  int num_shards = DivAndRoundUp(kParameters.db_record_bit_size,
                                 kParameters.lwe_plaintext_bit_size);
  ASSERT_EQ(database->Data().size(), num_shards);
  int num_values_per_block =
      sizeof(Database::BlockType) / sizeof(lwe::PlainInteger);
  int expected_num_blocks_per_column = DivAndRoundUp(
      static_cast<int>(kParameters.db_rows), num_values_per_block);
  for (const Database::RawMatrix& data_matrix : database->Data()) {
    EXPECT_EQ(data_matrix.size(), kParameters.db_cols);
    EXPECT_EQ(data_matrix[0].size(), expected_num_blocks_per_column);
  }
  ASSERT_EQ(database->Hints().size(), num_shards);
  for (const Database::LweMatrix& hint_matrix : database->Hints()) {
    EXPECT_EQ(hint_matrix.size(), kParameters.db_rows);
    EXPECT_EQ(hint_matrix[0].size(), kParameters.lwe_secret_dim);
  }
}

TEST_F(ServerTest, HandleRequestFailsIfNotPreprocessed) {
  // Handle a request without preprocessing the server.
  HintlessPirRequest request;
  *request.mutable_ct_query_vector() =
      SerializeLweCiphertext(lwe::Vector::Zero(kParameters.db_cols));
  EXPECT_THAT(this->server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Server has not been preprocessed")));
}

TEST_F(ServerTest, HandleRequestFailsIfIncorrectLinPirRequest) {
  ASSERT_OK(this->server_->Preprocess());
  HintlessPirRequest request;
  *request.mutable_ct_query_vector() =
      SerializeLweCiphertext(lwe::Vector::Zero(kParameters.db_cols));

  // No LinPIR request
  EXPECT_THAT(this->server_->HandleRequest(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unexpected number of LinPir requests")));
}

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir
