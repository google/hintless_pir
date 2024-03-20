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

#include "linpir/client.h"

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

constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;
constexpr absl::string_view kPrngSeed0 =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr absl::string_view kPrngSeed1 =
    "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0";
const RlweParameters<Integer> kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .ts = {4169729},
    .gadget_log_bs = {18, 18},
    .error_variance = 8,
    .prng_type = kPrngType,
    .rows_per_block = 1024,
};

class ClientTest : public ::testing::Test {
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

    prng_ = Prng::Create(kPrngSeed0).value();
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

  RlweParameters<Integer> params_;
  std::unique_ptr<const RnsContext> rns_context_;
  std::vector<const rlwe::PrimeModulus<ModularInt>*> moduli_;
  std::unique_ptr<const RnsGadget> gadget_;
  std::unique_ptr<const RnsErrorParams> error_params_;
  std::unique_ptr<const Encoder> encoder_;
  std::unique_ptr<Prng> prng_;
};

TEST(Client, CreateFailsIfNullRnsContext) {
  EXPECT_THAT(
      Client<Integer>::Create(kRlweParameters,
                              /*rns_context=*/nullptr, kPrngSeed0, kPrngSeed1),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("`rns_context` must not be null")));
}

TEST_F(ClientTest, CreateFailsIfInvalidPrngType) {
  RlweParameters<Integer> invalid_params = kRlweParameters;
  invalid_params.prng_type = rlwe::PRNG_TYPE_INVALID;
  EXPECT_THAT(Client<Integer>::Create(invalid_params, this->rns_context_.get(),
                                      kPrngSeed0, kPrngSeed1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid `prng_type`")));
}

TEST_F(ClientTest, CreateSucceeds) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));
  ASSERT_EQ(client->PrngSeedForCiphertextRandomPads(), kPrngSeed0);
  ASSERT_EQ(client->PrngSeedForGaloisKeyRandomPads(), kPrngSeed1);
}

TEST_F(ClientTest, EncryptQueryFailsIfQueryIsTooLong) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));
  // The query vector can contain at most `num_slots_per_group` elements.
  int num_slots_per_group = 1 << (kRlweParameters.log_n - 1);
  std::vector<Integer> long_query(num_slots_per_group + 1, 0);
  EXPECT_THAT(client->EncryptQuery(long_query, kPrngSeed0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`query_vector` can contain at most")));
}

TEST_F(ClientTest, EncryptQueryHasCorrectEncoding) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));

  int num_slots_per_group = 1 << (kRlweParameters.log_n - 1);
  ASSERT_LE(kRlweParameters.rows_per_block, num_slots_per_group);

  std::vector<Integer> query(kRlweParameters.rows_per_block, 0);
  for (int i = 0; i < kRlweParameters.rows_per_block; ++i) {
    query[i] = i;
  }
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query,
                       client->EncryptQuery(query, kPrngSeed0));

  // Generate the same secret key used by the client to encrypt `query`.
  ASSERT_OK_AND_ASSIGN(auto prng_sk, Prng::Create(kPrngSeed0));
  ASSERT_OK_AND_ASSIGN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                           this->moduli_, prng_sk.get()));

  // Decrypt the encrypted query and check the encoding.
  int num_rotations = this->params_.rows_per_block / 2;
  ASSERT_OK_AND_ASSIGN(
      std::vector<Integer> slots,
      secret_key.template DecryptBfv<Encoder>(ct_query, this->encoder_.get()));

  for (int i = 0; i < kRlweParameters.rows_per_block; ++i) {
    // The first group of slots contains `query` without rotation.
    EXPECT_EQ(slots[i], query[i]);
    // The second group of slots contains `query` rotated to the left by
    // `num_rotations` positions.
    int snd = (num_slots_per_group + i - num_rotations) % num_slots_per_group;
    EXPECT_EQ(slots[num_slots_per_group + snd], query[i]);
  }
}

TEST_F(ClientTest, GenerateGaloisKeyFailsIfSecretKeyIsNotSet) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));

  // Call `GenerateGaloisKey` when secret key is not set (either via explicit
  // PRNG seed or by calling `EncryptQuery` beforehand) will result in an error.
  EXPECT_THAT(client->GenerateGaloisKey(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Secret key not found")));
}

TEST_F(ClientTest, GenerateGaloisKeySucceeds) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));

  // Call `GenerateGaloisKey` with a PRNG seed that expands to the secret key.
  ASSERT_OK_AND_ASSIGN(RnsGaloisKey gk, client->GenerateGaloisKey(kPrngSeed0));
  EXPECT_EQ(gk.Dimension(), this->gadget_->Dimension());
  EXPECT_EQ(gk.SubstitutionPower(), 5);
}

TEST_F(ClientTest, RecoverFailsIfSecretKeyIsNotSet) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));

  // `Recover` requires that the secret key is set (either via explicit
  // PRNG seed or by calling `EncryptQuery` beforehand).
  LinPirResponse response;
  EXPECT_THAT(client->Recover(response),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Secret key not found")));
}

TEST_F(ClientTest, RecoverSucceeds) {
  ASSERT_OK_AND_ASSIGN(
      auto client,
      Client<Integer>::Create(kRlweParameters, this->rns_context_.get(),
                              /*prng_seed_ct_pad=*/kPrngSeed0,
                              /*prng_seed_gk_pad=*/kPrngSeed1));

  // Set the secret key by encrypting a dummy query.
  std::vector<Integer> query(kRlweParameters.rows_per_block, 0);
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query,
                       client->EncryptQuery(query, kPrngSeed0));

  // Generate the same secret key used by the client to encrypt `query`.
  ASSERT_OK_AND_ASSIGN(auto prng_sk, Prng::Create(kPrngSeed0));
  ASSERT_OK_AND_ASSIGN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(this->params_.log_n, this->params_.error_variance,
                           this->moduli_, prng_sk.get()));

  // Each ciphertext in a LinPirResponse encrypts a vector in slots such that
  // the sum of every `rows_per_block` slots is the desired matrix-vector
  // product and is the output of `Recover`.
  int num_slots = 1 << kRlweParameters.log_n;
  std::vector<Integer> slots(num_slots, 0);
  for (int i = 0; i < num_slots; ++i) {
    slots[i] = i;
  }
  std::vector<Integer> expected(kRlweParameters.rows_per_block, 0);
  for (int i = 0; i < num_slots; ++i) {
    expected[i % kRlweParameters.rows_per_block] += slots[i];
  }
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_block,
                       secret_key.template EncryptBfv<Encoder>(
                           slots, this->encoder_.get(),
                           this->error_params_.get(), this->prng_.get()));
  LinPirResponse::EncryptedInnerProduct inner_product;
  ASSERT_OK_AND_ASSIGN(*inner_product.add_ct_blocks(), ct_block.Serialize());
  LinPirResponse response;
  *response.add_ct_inner_products() = std::move(inner_product);
  ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Integer>> results,
                       client->Recover(response));
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], expected);
}

}  // namespace
}  // namespace linpir
}  // namespace hintless_pir
