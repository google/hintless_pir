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

#include "linpir/server.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "linpir/database.h"
#include "linpir/parameters.h"
#include "linpir/serialization.pb.h"
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
#include "shell_encryption/serialization.pb.h"
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

constexpr int kNumRows = 32;
constexpr int kNumCols = 1200;
constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;
constexpr absl::string_view kPrngSeed =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const RlweParameters<Integer> kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .ts = {4169729},
    .gadget_log_bs = {18, 18},
    .error_variance = 8,
    .prng_type = kPrngType,
    .rows_per_block = 1024,
};

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    params_ = kRlweParameters;
    auto rns_context = RnsContext::CreateForBfvFiniteFieldEncoding(
                           params_.log_n, params_.qs, /*ps=*/{}, params_.ts[0])
                           .value();
    rns_context_ = std::make_unique<const RnsContext>(std::move(rns_context));
    moduli_ = rns_context_->MainPrimeModuli();

    int level = moduli_.size() - 1;
    auto q_hats = rns_context_->MainPrimeModulusComplements(level).value();
    auto q_hat_invs = rns_context_->MainPrimeModulusCrtFactors(level).value();
    RnsGadget gadget = RnsGadget::Create(params_.log_n, params_.gadget_log_bs,
                                         q_hats, q_hat_invs, moduli_)
                           .value();
    gadget_ = std::make_unique<const RnsGadget>(std::move(gadget));

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

    prng_ = Prng::Create(kPrngSeed).value();
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

  RnsSecretKey GenerateSecretKey() {
    RnsSecretKey secret_key =
        RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                             this->moduli_, prng_.get())
            .value();
    return secret_key;
  }

  RnsGaloisKey GenerateGaloisKey(const RnsSecretKey& secret_key) {
    constexpr int power = 5;  // rotate by 1 position
    RnsGaloisKey gk =
        RnsGaloisKey::CreateForBfv(secret_key, power, params_.error_variance,
                                   gadget_.get(), kPrngType)
            .value();
    return gk;
  }

  RnsGaloisKey GenerateGaloisKey(const RnsSecretKey& secret_key,
                                 absl::string_view prng_seed_gk_pad) {
    constexpr int power = 5;  // rotate by 1 position
    std::vector<RnsPolynomial> gk_pads =
        RnsGaloisKey::SampleRandomPad(gadget_->Dimension(), params_.log_n,
                                      moduli_, prng_seed_gk_pad,
                                      params_.prng_type)
            .value();
    RnsGaloisKey gk =
        RnsGaloisKey::CreateWithRandomPadForBfv(
            std::move(gk_pads), secret_key, power, params_.error_variance,
            gadget_.get(), prng_seed_gk_pad, params_.prng_type)
            .value();
    return gk;
  }

  LinPirRequest SerializeLinPirRequest(const RnsCiphertext& ct_query,
                                       const RnsGaloisKey& gk) {
    LinPirRequest request;
    RnsPolynomial ct_query_b = ct_query.Component(0).value();
    *request.mutable_ct_query_b() = ct_query_b.Serialize(moduli_).value();
    for (const auto& gk_key_b : gk.GetKeyB()) {
      *request.add_gk_key_bs() = gk_key_b.Serialize(moduli_).value();
    }
    return request;
  }

  RlweParameters<Integer> params_;
  std::unique_ptr<const RnsContext> rns_context_;
  std::vector<const rlwe::PrimeModulus<ModularInt>*> moduli_;
  std::unique_ptr<const RnsGadget> gadget_;
  std::unique_ptr<const RnsErrorParams> error_params_;
  std::unique_ptr<const Encoder> encoder_;
  std::unique_ptr<Prng> prng_;
};

TEST_F(ServerTest, CreateFailsIfRnsContextIsNull) {
  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));
  EXPECT_THAT(
      Server<Integer>::Create(kRlweParameters,
                              /*rns_context=*/nullptr, {database.get()}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("`rns_context` must not be null")));
}

TEST_F(ServerTest, CreateFailsIfInvalidPrngType) {
  RlweParameters<Integer> invalid_params = kRlweParameters;
  invalid_params.prng_type = rlwe::PRNG_TYPE_INVALID;

  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));
  EXPECT_THAT(Server<Integer>::Create(invalid_params, this->rns_context_.get(),
                                      {database.get()}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid `prng_type`")));
}

TEST_F(ServerTest, CreateWithPrngSeedsFailsIfInvalidPrngType) {
  RlweParameters<Integer> invalid_params = kRlweParameters;
  invalid_params.prng_type = rlwe::PRNG_TYPE_INVALID;

  std::vector<Integer> row(1, 0);
  ASSERT_OK_AND_ASSIGN(auto database,
                       Database<Integer>::Create(
                           this->params_, this->rns_context_.get(), {row}));
  EXPECT_THAT(
      Server<Integer>::Create(invalid_params, this->rns_context_.get(),
                              {database.get()}, /*prng_seed_ct_pad=*/kPrngSeed,
                              /*prng_seed_gk_pad=*/kPrngSeed),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid `prng_type`")));
}

TEST_F(ServerTest, HandleRequestWithoutPreprocessing) {
  auto data =
      SampleMatrix(kNumRows, kNumCols, rns_context_->PlaintextModulus());
  ASSERT_OK_AND_ASSIGN(
      auto database,
      Database<Integer>::Create(this->params_, this->rns_context_.get(), data));
  ASSERT_OK_AND_ASSIGN(auto server, Server<Integer>::Create(
                                        this->params_, this->rns_context_.get(),
                                        {database.get()}));

  RnsSecretKey secret_key = this->GenerateSecretKey();
  RnsGaloisKey gk = this->GenerateGaloisKey(secret_key);

  // Encrypt rotations of a unit vector "u" selecting the third column.
  constexpr int index = 2;
  int num_rotations = this->params_.rows_per_block / 2;
  int num_slots_per_group = 1 << (this->params_.log_n - 1);
  std::vector<Integer> slots(num_slots_per_group * 2, 0);
  int snd_index =
      (num_slots_per_group - num_rotations + index) % num_slots_per_group;
  slots[index] = 1;
  slots[num_slots_per_group + snd_index] = 1;
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query,
                       secret_key.template EncryptBfv<Encoder>(
                           slots, this->encoder_.get(),
                           this->error_params_.get(), this->prng_.get()));

  ASSERT_OK_AND_ASSIGN(LinPirResponse response,
                       server->HandleRequest(ct_query, gk));

  // The response should contain one inner product as there is one database
  // held by the server, and the inner product contains one ciphertext, as
  // kNumRows <= kParameters.num_rows_per_block and hence just one block.
  ASSERT_EQ(response.ct_inner_products_size(), 1);  // only one database
  ASSERT_EQ(response.ct_inner_products(0).ct_blocks_size(), 1);

  ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      RnsCiphertext::Deserialize(response.ct_inner_products(0).ct_blocks(0),
                                 this->moduli_, this->error_params_.get()));
  RnsCiphertext ct_response(deserialized);
  ASSERT_OK_AND_ASSIGN(auto decrypted, secret_key.template DecryptBfv<Encoder>(
                                           ct_response, this->encoder_.get()));
  ASSERT_EQ(decrypted.size(), num_slots_per_group * 2);

  // Recover the plaintext inner product from the decrypted values.
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

TEST_F(ServerTest, HandleRequestWithPreprocessing) {
  auto data =
      SampleMatrix(kNumRows, kNumCols, rns_context_->PlaintextModulus());
  ASSERT_OK_AND_ASSIGN(
      auto database,
      Database<Integer>::Create(this->params_, this->rns_context_.get(), data));
  ASSERT_OK_AND_ASSIGN(auto server, Server<Integer>::Create(
                                        this->params_, this->rns_context_.get(),
                                        {database.get()}));
  // Get the PRNG seeds for generating Galois key and for encrypting query.
  absl::string_view prng_seed_ct_pad =
      server->PrngSeedForCiphertextRandomPads();
  absl::string_view prng_seed_gk_pad = server->PrngSeedForGaloisKeyRandomPads();

  RnsSecretKey secret_key = this->GenerateSecretKey();
  RnsGaloisKey gk = this->GenerateGaloisKey(secret_key, prng_seed_gk_pad);

  // Encrypt rotations of a unit vector "u" selecting the third column.
  constexpr int index = 2;
  int num_rotations = this->params_.rows_per_block / 2;
  int num_slots_per_group = 1 << (this->params_.log_n - 1);
  std::vector<Integer> slots(num_slots_per_group * 2, 0);
  int snd_index =
      (num_slots_per_group - num_rotations + index) % num_slots_per_group;
  slots[index] = 1;
  slots[num_slots_per_group + snd_index] = 1;
  ASSERT_OK_AND_ASSIGN(auto prng_pad, Prng::Create(prng_seed_ct_pad));
  ASSERT_OK_AND_ASSIGN(
      RnsCiphertext ct_query,
      secret_key.template EncryptBfv<Encoder>(
          slots, this->encoder_.get(), this->error_params_.get(),
          this->prng_.get(), prng_pad.get()));

  // Preprocess the database and server computation.
  ASSERT_OK(server->Preprocess());

  // Let the server handle the request with preprocessed data.
  LinPirRequest request = this->SerializeLinPirRequest(ct_query, gk);
  ASSERT_OK_AND_ASSIGN(LinPirResponse response, server->HandleRequest(request));

  // The response should contain one inner product as there is one database
  // held by the server, and the inner product contains one ciphertext, as
  // kNumRows <= kParameters.num_rows_per_block and hence just one block.
  ASSERT_EQ(response.ct_inner_products_size(), 1);  // only one database
  ASSERT_EQ(response.ct_inner_products(0).ct_blocks_size(), 1);

  ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      RnsCiphertext::Deserialize(response.ct_inner_products(0).ct_blocks(0),
                                 this->moduli_, this->error_params_.get()));
  RnsCiphertext ct_response(deserialized);
  ASSERT_OK_AND_ASSIGN(auto decrypted, secret_key.template DecryptBfv<Encoder>(
                                           ct_response, this->encoder_.get()));
  ASSERT_EQ(decrypted.size(), num_slots_per_group * 2);
  // Recover the plaintext inner product from the decrypted values.
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
