/*
 * Copyright 2024 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "linpir/client.h"
#include "linpir/database.h"
#include "linpir/parameters.h"
#include "linpir/server.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/testing/status_testing.h"

ABSL_FLAG(int, num_rows, 1024, "Number of rows");
ABSL_FLAG(int, num_cols, 1400, "Number of cols");
ABSL_FLAG(bool, debug, false, "debug mode");

namespace hintless_pir {
namespace linpir {
namespace {

using Integer = Uint64;
using ModularInt = rlwe::MontgomeryInt<Integer>;
using RnsContext = rlwe::RnsContext<ModularInt>;
using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
using Prng = rlwe::SingleThreadHkdfPrng;

const RlweParameters<Integer> kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .ts = {4169729, 4120577},
    .gadget_log_bs = {18, 18},
    .error_variance = 8,
    .prng_type = rlwe::PRNG_TYPE_HKDF,
    .rows_per_block = 1024,
};

class LinPirTest : public ::testing::Test {
 protected:
  void SetUp() override {
    params_ = std::make_unique<RlweParameters<Integer>>(kRlweParameters);
    auto params = params_.get();
    auto rns_context = RnsContext::CreateForBfvFiniteFieldEncoding(
        params->log_n, params->qs, /*ps=*/{}, params->ts[0]);
    rns_context_ =
        std::make_unique<const RnsContext>(std::move(rns_context.value()));
    moduli_ = rns_context_->MainPrimeModuli();
  }

  // Returns a vector of `num_values` many random integers in [0, max_value).
  std::vector<Integer> SampleValues(int num_values, Integer max_value) const {
    absl::BitGen bitgen;
    std::vector<Integer> values;
    for (int i = 0; i < num_values; ++i) {
      values.push_back(absl::Uniform<Integer>(bitgen, 0, max_value));
    }
    return values;
  }

  // Returns a matrix of dimension `num_rows` * `num_cols`, with values in
  // the range [0, max_value).
  std::vector<std::vector<Integer>> SampleMatrix(int num_rows, int num_cols,
                                                 Integer max_value) const {
    std::vector<std::vector<Integer>> matrix(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      matrix[i] = SampleValues(num_cols, max_value);
    }
    return matrix;
  }

  std::unique_ptr<const RlweParameters<Integer>> params_;
  std::unique_ptr<const RnsContext> rns_context_;
  std::vector<const rlwe::PrimeModulus<ModularInt>*> moduli_;
};

TEST_F(LinPirTest, EndToEndTest) {
  int num_rows = absl::GetFlag(FLAGS_num_rows);
  int num_cols = absl::GetFlag(FLAGS_num_cols);

  ASSERT_OK_AND_ASSIGN(std::string prng_seed_ct_pad, Prng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_gk_pad, Prng::GenerateSeed());

  // Create a database and a server
  auto data = SampleMatrix(num_rows, num_cols, 8);
  ASSERT_OK_AND_ASSIGN(
      auto database, Database<Integer>::Create(*this->params_,
                                               this->rns_context_.get(), data));
  ASSERT_OK_AND_ASSIGN(
      auto server, Server<Integer>::Create(
                       *this->params_, this->rns_context_.get(),
                       {database.get()}, prng_seed_ct_pad, prng_seed_gk_pad));
  ASSERT_OK(server->Preprocess());

  // Create a client
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(*this->params_, this->rns_context_.get(),
                              prng_seed_ct_pad, prng_seed_gk_pad));

  // Encrypt a random query vector
  std::vector<Integer> query = SampleValues(num_cols, 8);
  ASSERT_OK_AND_ASSIGN(auto request, client->GenerateRequest(query));

  ASSERT_OK_AND_ASSIGN(auto response, server->HandleRequest(request));
  int num_blocks = ceil(num_rows * 1.0 / this->params_->rows_per_block);
  ASSERT_EQ(response.ct_inner_products_size(), 1);
  ASSERT_EQ(response.ct_inner_products(0).ct_blocks_size(), num_blocks);

  // Recover the results
  ASSERT_OK_AND_ASSIGN(auto results, client->Recover(response));
  ASSERT_GE(results.size(), 1);
  ASSERT_GE(results[0].size(), num_rows);
  std::vector<Integer> expected(num_rows, 0);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      expected[i] += (data[i][j] * query[j]) % this->params_->ts[0];
      expected[i] = expected[i] % this->params_->ts[0];
    }
  }
  for (int i = 0; i < num_rows; ++i) {
    EXPECT_EQ(results[0][i], expected[i]);
  }
}

}  // namespace
}  // namespace linpir
}  // namespace hintless_pir

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);

  return RUN_ALL_TESTS();
}
