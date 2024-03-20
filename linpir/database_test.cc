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

#include "linpir/database.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "linpir/parameters.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_error_params.h"
#include "shell_encryption/rns/rns_gadget.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/rns/rns_secret_key.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"

namespace hintless_pir {
namespace linpir {
namespace {

using Integer = Uint64;
using ModularInt = rlwe::MontgomeryInt<Integer>;
using RnsContext = rlwe::RnsContext<ModularInt>;
using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
using RnsSecretKey = rlwe::RnsRlweSecretKey<ModularInt>;
using RnsCiphertext = rlwe::RnsBfvCiphertext<ModularInt>;
using RnsGaloisKey = rlwe::RnsGaloisKey<ModularInt>;
using RnsGadget = rlwe::RnsGadget<ModularInt>;
using RnsErrorParams = rlwe::RnsErrorParams<ModularInt>;
using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
using Prng = rlwe::SingleThreadHkdfPrng;
using ::rlwe::testing::StatusIs;
using ::testing::HasSubstr;

constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;
constexpr absl::string_view kPrngSeed =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr int kNumRows = 32;
constexpr int kNumCols = 1200;
const RlweParameters<Integer> kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .ts = {4169729, 4120577},
    .gadget_log_bs = {18, 18},
    .error_variance = 8,
    .prng_type = kPrngType,
    .rows_per_block = 1024,
};

class DatabaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    params_ = kRlweParameters;
    auto rns_context = RnsContext::CreateForBfvFiniteFieldEncoding(
                           params_.log_n, params_.qs, /*ps=*/{}, params_.ts[0])
                           .value();
    rns_context_ = std::make_unique<const RnsContext>(std::move(rns_context));
    moduli_ = rns_context_->MainPrimeModuli();
    auto error_params =
        RnsErrorParams::Create(
            params_.log_n, moduli_, {},
            std::log2(static_cast<double>(rns_context_->PlaintextModulus())),
            std::sqrt(params_.error_variance))
            .value();
    error_params_ =
        std::make_unique<const RnsErrorParams>(std::move(error_params));
    auto encoder = Encoder::Create(rns_context_.get()).value();
    encoder_ = std::make_unique<const Encoder>(std::move(encoder));
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

  RlweParameters<Integer> params_;
  std::unique_ptr<const RnsContext> rns_context_;
  std::vector<const rlwe::PrimeModulus<ModularInt>*> moduli_;
  std::unique_ptr<const RnsErrorParams> error_params_;
  std::unique_ptr<const Encoder> encoder_;
};

TEST(Database, CreateFailsIfRnsContextIsNull) {
  std::vector<Integer> row(1, 0);
  EXPECT_THAT(Database<Integer>::Create(kRlweParameters,
                                        /*rns_context=*/nullptr, {row}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`rns_context` must not be null")));
}

TEST_F(DatabaseTest, CreateFailsIfDataIsEmpty) {
  EXPECT_THAT(Database<Integer>::Create(kRlweParameters,
                                        this->rns_context_.get(), /*data=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`data` must not be empty")));
}

TEST_F(DatabaseTest, CreateFailsIfDataHasTooManyColumns) {
  int num_slots_per_group = 1 << (kRlweParameters.log_n - 1);
  std::vector<Integer> row(num_slots_per_group + 1, 0);
  EXPECT_THAT(Database<Integer>::Create(kRlweParameters,
                                        this->rns_context_.get(), {row}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`data` has more columns than")));
}

TEST_F(DatabaseTest, CreateDatabase) {
  auto data = SampleMatrix(kNumRows, kNumCols, 16);
  ASSERT_OK_AND_ASSIGN(
      auto database,
      Database<Integer>::Create(this->params_, this->rns_context_.get(), data));

  int expected_num_blocks = ceil(kNumRows * 1.0 / this->params_.rows_per_block);
  int expected_num_diags_per_block = this->params_.rows_per_block / 2;
  EXPECT_EQ(database->NumBlocks(), expected_num_blocks);
  EXPECT_EQ(database->NumDiagonalsPerBlock(), expected_num_diags_per_block);
}

TEST_F(DatabaseTest, InnerProductFailsIfIncorrectNumberOfQueryCiphertexts) {
  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));
  EXPECT_THAT(database->InnerProductWith(/*ct_rotated_queries=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ct_rotated_queries")));
}

TEST_F(DatabaseTest, InnerProduct) {
  auto data = SampleMatrix(kNumRows, kNumCols, 16);
  ASSERT_OK_AND_ASSIGN(
      auto database,
      Database<Integer>::Create(this->params_, this->rns_context_.get(), data));

  ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
  ASSERT_OK_AND_ASSIGN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                           this->moduli_, prng.get()));

  // Encrypt rotations of a unit vector "u" selecting the third column.
  constexpr int index = 2;
  int num_rotations = this->params_.rows_per_block / 2;
  int num_slots_per_group = 1 << (this->params_.log_n - 1);
  std::vector<RnsCiphertext> ct_rotated_queries;
  for (int i = 0; i < num_rotations; ++i) {
    std::vector<Integer> slots(num_slots_per_group * 2, 0);
    // Encode the unit vector rotated by i positions in the first group
    int fst_idx = (num_slots_per_group + index - i) % num_slots_per_group;
    // Encode the unit vector rotated by (num_rotations + i) in the second group
    int snd_idx =
        (num_slots_per_group - num_rotations + index - i) % num_slots_per_group;
    slots[fst_idx] = 1;
    slots[snd_idx + num_slots_per_group] = 1;
    ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query,
                         secret_key.template EncryptBfv<Encoder>(
                             slots, this->encoder_.get(),
                             this->error_params_.get(), prng.get()));
    ct_rotated_queries.push_back(std::move(ct_query));
  }
  ASSERT_OK_AND_ASSIGN(auto ct_inner_products,
                       database->InnerProductWith(ct_rotated_queries));
  ASSERT_EQ(ct_inner_products.size(), 1);
  ASSERT_OK_AND_ASSIGN(auto decrypted,
                       secret_key.template DecryptBfv<Encoder>(
                           ct_inner_products[0], this->encoder_.get()));
  ASSERT_EQ(decrypted.size(), num_slots_per_group * 2);

  std::vector<Integer> results(this->params_.rows_per_block, 0);
  for (int i = 0; i < num_slots_per_group; ++i) {
    results[i % this->params_.rows_per_block] += decrypted[i];
    results[i % this->params_.rows_per_block] +=
        decrypted[num_slots_per_group + i];
  }
  for (int i = 0; i < kNumRows; ++i) {
    EXPECT_EQ(results[i] % this->rns_context_->PlaintextModulus(),
              data[i][index]);
  }
}

TEST_F(DatabaseTest, PreprocessFailsIfIncorrectNumberOfRandomPads) {
  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));
  EXPECT_THAT(database->Preprocess(/*pad_rotated_queries=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("pad_rotated_queries")));
}

TEST_F(DatabaseTest,
       InnerProductWithPreprocessingFailsIfDatabaseIsNotPreprocessed) {
  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));

  // Call `InnerProductWithPreprocessedPads` without preprocessing database.
  ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
  ASSERT_OK_AND_ASSIGN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                           this->moduli_, prng.get()));
  int num_slots = 1 << this->params_.log_n;
  std::vector<Integer> slots(num_slots, 0);
  ASSERT_OK_AND_ASSIGN(
      RnsCiphertext ct_query,
      secret_key.template EncryptBfv<Encoder>(
          slots, this->encoder_.get(), this->error_params_.get(), prng.get()));
  EXPECT_THAT(database->InnerProductWithPreprocessedPads({ct_query}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("There is no preprocessed data")));
}

TEST_F(DatabaseTest,
       InnerProductWithPreprocessingFailsIfIncorrectNumberOfQueryCiphertexts) {
  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));

  // Preprocess the database first.
  int num_diagonals = database->NumDiagonalsPerBlock();
  ASSERT_OK_AND_ASSIGN(
      RnsPolynomial fake_pad,
      RnsPolynomial::CreateZero(this->params_.log_n, this->moduli_));
  std::vector<RnsPolynomial> pads(num_diagonals, fake_pad);
  ASSERT_OK(database->Preprocess(pads));

  EXPECT_THAT(
      database->InnerProductWithPreprocessedPads(/*ct_rotated_queries=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("ct_rotated_queries")));
}

TEST_F(DatabaseTest, InnerProductWithPreprocessing) {
  auto data = SampleMatrix(kNumRows, kNumCols, 16);
  ASSERT_OK_AND_ASSIGN(
      auto database,
      Database<Integer>::Create(this->params_, this->rns_context_.get(), data));

  ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
  ASSERT_OK_AND_ASSIGN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                           this->moduli_, prng.get()));

  // Create a Galois key
  constexpr int power = 5;  // rotate by 1 position
  int level = this->moduli_.size() - 1;
  ASSERT_OK_AND_ASSIGN(auto q_hats,
                       this->rns_context_->MainPrimeModulusComplements(level));
  ASSERT_OK_AND_ASSIGN(auto q_hat_invs,
                       this->rns_context_->MainPrimeModulusCrtFactors(level));
  ASSERT_OK_AND_ASSIGN(
      RnsGadget gadget,
      RnsGadget::Create(this->params_.log_n, this->params_.gadget_log_bs,
                        q_hats, q_hat_invs, this->moduli_));
  ASSERT_OK_AND_ASSIGN(
      RnsGaloisKey gk,
      RnsGaloisKey::CreateForBfv(
          secret_key, power, this->params_.error_variance, &gadget, kPrngType));

  // Encrypt a unit vector "u" selecting the third column.
  constexpr int index = 2;
  int num_rotations = this->params_.rows_per_block / 2;
  int num_slots_per_group = 1 << (this->params_.log_n - 1);
  int snd_index =
      (num_slots_per_group - num_rotations + index) % num_slots_per_group;
  std::vector<Integer> slots(num_slots_per_group * 2, 0);
  slots[index] = 1;
  slots[num_slots_per_group + snd_index] = 1;
  ASSERT_OK_AND_ASSIGN(
      RnsCiphertext ct_query,
      secret_key.template EncryptBfv<Encoder>(
          slots, this->encoder_.get(), this->error_params_.get(), prng.get()));

  ASSERT_OK_AND_ASSIGN(RnsPolynomial pad_query, ct_query.Component(1));
  std::vector<RnsPolynomial> pad_rotated_queries;
  pad_rotated_queries.push_back(std::move(pad_query));
  std::vector<RnsCiphertext> ct_rotated_queries;
  ct_rotated_queries.push_back(std::move(ct_query));
  for (int i = 1; i < num_rotations; ++i) {
    ASSERT_OK_AND_ASSIGN(auto ct_sub_query,
                         ct_rotated_queries[i - 1].Substitute(power));
    ASSERT_OK_AND_ASSIGN(auto ct_rotated_query, gk.ApplyTo(ct_sub_query));
    ASSERT_OK_AND_ASSIGN(auto pad_rotated_query, ct_rotated_query.Component(1));
    pad_rotated_queries.push_back(std::move(pad_rotated_query));
    ct_rotated_queries.push_back(std::move(ct_rotated_query));
  }

  // Preprocess the database
  ASSERT_OK(database->Preprocess(pad_rotated_queries));
  ASSERT_TRUE(database->IsPreprocessed());
  ASSERT_OK_AND_ASSIGN(
      auto ct_inner_products,
      database->InnerProductWithPreprocessedPads(ct_rotated_queries));
  ASSERT_EQ(ct_inner_products.size(), 1);
  ASSERT_OK_AND_ASSIGN(auto decrypted,
                       secret_key.template DecryptBfv<Encoder>(
                           ct_inner_products[0], this->encoder_.get()));
  ASSERT_EQ(decrypted.size(), num_slots_per_group * 2);

  std::vector<Integer> results(this->params_.rows_per_block, 0);
  for (int i = 0; i < num_slots_per_group; ++i) {
    results[i % this->params_.rows_per_block] += decrypted[i];
    results[i % this->params_.rows_per_block] +=
        decrypted[num_slots_per_group + i];
  }
  for (int i = 0; i < kNumRows; ++i) {
    EXPECT_EQ(results[i] % this->rns_context_->PlaintextModulus(),
              data[i][index]);
  }
}

}  // namespace
}  // namespace linpir
}  // namespace hintless_pir
