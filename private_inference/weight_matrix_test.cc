/*
 * Copyright 2026 Google LLC.
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

#include "private_inference/weight_matrix.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"  // NOLINT(misc-include-cleaner)
#include "gtest/gtest.h"
#include "private_inference/parameters.h"
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
#include "shell_encryption/testing/status_testing.h"

ABSL_FLAG(int, num_queries, 2, "Number of query vectors");
ABSL_FLAG(int, num_iterations, 1, "Number of benchmark iterations");

namespace private_inference {
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

constexpr rlwe::PrngType kPrngType = rlwe::PRNG_TYPE_HKDF;
constexpr absl::string_view kPrngSeed =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr absl::string_view kPrngSeedForGk =
    "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr absl::string_view kPrngSeed0 =
    "x123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr absl::string_view kPrngSeed1 =
    "y123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// PRNG seeds for sampling the pad "a" element
constexpr absl::string_view kPrngSeedForPad =
    "a123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

constexpr int kNumRows = 1 << 12;
constexpr int kNumCols = 1 << 12;

const RlweParameters kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .gadget_log_bs = {18, 18},
    .plaintext_modulus = 67239937,  // 27 bits
    .error_variance = 8,
    .prng_type = kPrngType,
};

class WeightMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    rlwe_params_ = kRlweParameters;
    auto rns_context = RnsContext::CreateForBfvFiniteFieldEncoding(
                           rlwe_params_.log_n, rlwe_params_.qs, /*ps=*/{},
                           rlwe_params_.plaintext_modulus)
                           .value();
    rns_context_ = std::make_unique<const RnsContext>(std::move(rns_context));
    moduli_ = rns_context_->MainPrimeModuli();
    auto error_params =
        RnsErrorParams::Create(
            rlwe_params_.log_n, moduli_, {},
            std::log2(static_cast<double>(rns_context_->PlaintextModulus())),
            std::sqrt(rlwe_params_.error_variance))
            .value();
    error_params_ =
        std::make_unique<const RnsErrorParams>(std::move(error_params));
    auto encoder = Encoder::Create(rns_context_.get()).value();
    encoder_ = std::make_unique<const Encoder>(std::move(encoder));

    int level = moduli_.size() - 1;
    auto q_hats = rns_context_->MainPrimeModulusComplements(level).value();
    auto q_hat_invs =
        this->rns_context_->MainPrimeModulusCrtFactors(level).value();
    RnsGadget gadget =
        RnsGadget::Create(rlwe_params_.log_n, rlwe_params_.gadget_log_bs,
                          q_hats, q_hat_invs, moduli_)
            .value();
    gadget_ = std::make_unique<const RnsGadget>(std::move(gadget));
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

  // Returns a matrix of dimension `num_rows` * `num_cols`.
  std::vector<std::vector<Integer>> SampleMatrix(int num_rows, int num_cols,
                                                 Integer max_value) const {
    std::vector<std::vector<Integer>> matrix(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      matrix[i] = SampleValues(num_cols, max_value);
    }
    return matrix;
  }

  RnsSecretKey GenerateSecretKey(Prng* prng) const {
    auto secret_key =
        RnsSecretKey::Sample(rlwe_params_.log_n, rlwe_params_.error_variance,
                             moduli_, prng)
            .value();
    return secret_key;
  }

  RnsGaloisKey GenerateGaloisKey(const RnsSecretKey& secret_key,
                                 int power) const {
    auto gk_pads = RnsGaloisKey::SampleRandomPad(
                       gadget_->Dimension(), secret_key.LogN(),
                       secret_key.Moduli(), kPrngSeedForGk, kPrngType)
                       .value();
    auto gk = RnsGaloisKey::CreateWithRandomPadForBfv(
                  gk_pads, secret_key, power, rlwe_params_.error_variance,
                  gadget_.get(), kPrngSeedForGk, kPrngType)
                  .value();
    return gk;
  }

  std::pair<std::vector<RnsPolynomial>, std::vector<RnsCiphertext>>
  GenerateRotatedQueries(const RnsCiphertext& ct_query, const RnsGaloisKey& gk,
                         int num_rotations) const {
    constexpr int power = 5;
    RnsPolynomial pad_query = ct_query.Component(1).value();
    std::vector<RnsPolynomial> pad_rotated_queries;
    pad_rotated_queries.push_back(std::move(pad_query));
    std::vector<RnsCiphertext> ct_rotated_queries;
    ct_rotated_queries.push_back(std::move(ct_query));
    for (int i = 1; i < num_rotations; ++i) {
      auto ct_sub_query = ct_rotated_queries[i - 1].Substitute(power).value();
      auto ct_rotated_query = gk.ApplyTo(ct_sub_query).value();
      auto pad_rotated_query = ct_rotated_query.Component(1).value();
      pad_rotated_queries.push_back(std::move(pad_rotated_query));
      ct_rotated_queries.push_back(std::move(ct_rotated_query));
    }
    return std::make_pair(pad_rotated_queries, ct_rotated_queries);
  }

  std::vector<RnsCiphertext> GenerateRotatedQueriesWithPreprocessedPads(
      const RnsCiphertext& ct_query, const RnsGaloisKey& gk,
      const std::vector<RnsPolynomial>& ct_pads,
      const std::vector<std::vector<RnsPolynomial>>& ct_sub_pad_digits,
      int num_rotations) const {
    constexpr int power = 5;
    std::vector<RnsCiphertext> ct_rotated_queries;
    ct_rotated_queries.push_back(std::move(ct_query));
    for (int i = 1; i < num_rotations; ++i) {
      auto ct_sub_query = ct_rotated_queries[i - 1].Substitute(power).value();
      auto ct_rotated_query =
          gk.ApplyToWithRandomPad(ct_sub_query, ct_sub_pad_digits[i - 1],
                                  ct_pads[i])
              .value();
      ct_rotated_queries.push_back(std::move(ct_rotated_query));
    }
    return ct_rotated_queries;
  }

  RlweParameters rlwe_params_;
  std::unique_ptr<const RnsContext> rns_context_;
  std::vector<const rlwe::PrimeModulus<ModularInt>*> moduli_;
  std::unique_ptr<const RnsErrorParams> error_params_;
  std::unique_ptr<const Encoder> encoder_;
  std::unique_ptr<const RnsGadget> gadget_;
};

TEST_F(WeightMatrixTest, InnerProductWithPreprocessing) {
  constexpr int power = 5;  // rotate by 1 position

  auto data = SampleMatrix(kNumRows, kNumCols, 8);
  ASSERT_OK_AND_ASSIGN(
      auto matrix,
      WeightMatrix::Create(this->rns_context_.get(), this->gadget_.get(),
                           kPrngSeedForPad, kPrngSeedForGk));
  ASSERT_OK(matrix->SetData(data));

  // Preprocess the pads elements for ciphertext rotations
  int num_rotations = kNumCols / 2 / 2;
  int num_slots_per_group = 1 << (this->rlwe_params_.log_n - 1);
  ASSERT_OK(matrix->PreprocessPad(num_rotations));
  ASSERT_TRUE(matrix->IsPreprocessedPad());

  // Preprocess the matrix
  auto pad_rotated_ct_v = matrix->PreprocessedRotatedCiphertextPads();
  ASSERT_OK(matrix->PreprocessDatabase(pad_rotated_ct_v, pad_rotated_ct_v));
  ASSERT_TRUE(matrix->IsPreprocessedDatabase());

  auto sub_pad_digits = matrix->PreprocessedSubCiphertextPadDigits();

  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  absl::Duration time_online = absl::ZeroDuration();
  for (int it = 0; it < num_iterations; ++it) {
    ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
    RnsSecretKey secret_key = GenerateSecretKey(prng.get());
    RnsGaloisKey gk = GenerateGaloisKey(secret_key, power);

    // Encrypt a vector "u" using two ciphertexts, each encrypting half of "u".
    // For simplicity, we choose u to have two nonzero entries: u[index0] = 1
    // and u[kNumCols / 2 + index1] = 1.
    // We assume that kNumCols <= 2 * num_slots_per_group; so if u0 and u1 are
    // the two halves of u, then we can encrypt two copies of u0 in the first
    // ciphertext occupying the two cyclic subgroups of the slots, and similarly
    // for u1 in the second ciphertext. The first copy is the vector u0 (and u1)
    // and the second copy is the vector u0 rotated by kNumCols / 4 positions
    // (and similarly for u1). With this encoding, we can generate all needed
    // rotations of u0 (and u1) using kNumCols / 4 rotations.
    constexpr int index0 = 2, index1 = 5;
    const int index0_rot =
        num_slots_per_group +
        (index0 - kNumCols / 4 + num_slots_per_group) % num_slots_per_group;
    const int index1_rot =
        num_slots_per_group +
        (index1 - kNumCols / 4 + num_slots_per_group) % num_slots_per_group;
    std::vector<Integer> slots0(num_slots_per_group * 2, 0);
    std::vector<Integer> slots1(num_slots_per_group * 2, 0);
    slots0[index0] = 1;
    slots0[index0_rot] = 1;
    slots1[index1] = 1;
    slots1[index1_rot] = 1;
    ASSERT_OK_AND_ASSIGN(auto prng0, Prng::Create(kPrngSeed0));
    ASSERT_OK_AND_ASSIGN(auto prng1, Prng::Create(kPrngSeed1));
    ASSERT_OK_AND_ASSIGN(auto prng_pad, Prng::Create(kPrngSeedForPad));
    ASSERT_OK_AND_ASSIGN(
        RnsCiphertext ct_query0,
        secret_key.template EncryptBfv<Encoder>(slots0, this->encoder_.get(),
                                                this->error_params_.get(),
                                                prng0.get(), prng_pad.get()));
    ASSERT_OK_AND_ASSIGN(prng_pad, Prng::Create(kPrngSeedForPad));
    ASSERT_OK_AND_ASSIGN(
        RnsCiphertext ct_query1,
        secret_key.template EncryptBfv<Encoder>(slots1, this->encoder_.get(),
                                                this->error_params_.get(),
                                                prng1.get(), prng_pad.get()));

    auto time_online_s = absl::Now();
    auto ct_rotated_queries0 = GenerateRotatedQueriesWithPreprocessedPads(
        ct_query0, gk, pad_rotated_ct_v, sub_pad_digits, num_rotations);
    auto ct_rotated_queries1 = GenerateRotatedQueriesWithPreprocessedPads(
        ct_query1, gk, pad_rotated_ct_v, sub_pad_digits, num_rotations);

    ASSERT_OK_AND_ASSIGN(auto ct_inner_products,
                         matrix->InnerProductWithPreprocessedPads(
                             ct_rotated_queries0, ct_rotated_queries1));
    auto time_online_e = absl::Now();
    time_online += (time_online_e - time_online_s);
    std::cout << "Online time: " << time_online_e - time_online_s << std::endl;

    ASSERT_EQ(ct_inner_products.size(), 2);
    ASSERT_OK_AND_ASSIGN(auto decrypted0,
                         secret_key.template DecryptBfv<Encoder>(
                             ct_inner_products[0], this->encoder_.get()));
    ASSERT_OK_AND_ASSIGN(auto decrypted1,
                         secret_key.template DecryptBfv<Encoder>(
                             ct_inner_products[1], this->encoder_.get()));
    ASSERT_EQ(decrypted0.size(), num_slots_per_group * 2);
    ASSERT_EQ(decrypted1.size(), num_slots_per_group * 2);

    int rows_per_block = kNumRows / 2;

    std::vector<Integer> results0(rows_per_block, 0);
    std::vector<Integer> results1(rows_per_block, 0);
    for (int i = 0; i < rows_per_block; ++i) {
      results0[i] += decrypted0[i];
      results0[i] += decrypted0[num_slots_per_group + i];
      results1[i] += decrypted1[i];
      results1[i] += decrypted1[num_slots_per_group + i];
    }
    for (int i = 0; i < rows_per_block; ++i) {
      EXPECT_EQ(results0[i] % this->rns_context_->PlaintextModulus(),
                data[i][index0] + data[i][kNumCols / 2 + index1]);
      EXPECT_EQ(results1[i] % this->rns_context_->PlaintextModulus(),
                data[rows_per_block + i][index0] +
                    data[rows_per_block + i][kNumCols / 2 + index1]);
    }
  }
  std::cout << "Average time (total): " << time_online / num_iterations
            << std::endl;
}

TEST_F(WeightMatrixTest, InnerProductWithMany) {
  constexpr int power = 5;  // rotate by 1 position

  auto data = SampleMatrix(kNumRows, kNumCols, 8);
  ASSERT_OK_AND_ASSIGN(auto matrix,
                       WeightMatrix::Create(this->rns_context_.get()));
  ASSERT_OK(matrix->SetData(data));
  ASSERT_OK_AND_ASSIGN(auto prng0, Prng::Create(kPrngSeed0));
  ASSERT_OK_AND_ASSIGN(auto prng1, Prng::Create(kPrngSeed1));
  RnsSecretKey secret_key0 = GenerateSecretKey(prng0.get());
  RnsSecretKey secret_key1 = GenerateSecretKey(prng1.get());
  RnsGaloisKey gk0 = GenerateGaloisKey(secret_key0, power);
  RnsGaloisKey gk1 = GenerateGaloisKey(secret_key1, power);

  // Encrypt a unit vector "u" using two ciphertexts, each encrypting half of u.
  constexpr int index00 = 2, index01 = 5, index10 = 6, index11 = 10;
  int num_rotations = kNumCols / 2 / 2;
  int num_slots_per_group = 1 << (this->rlwe_params_.log_n - 1);
  std::vector<Integer> slots00(num_slots_per_group * 2, 0);
  std::vector<Integer> slots01(num_slots_per_group * 2, 0);
  std::vector<Integer> slots10(num_slots_per_group * 2, 0);
  std::vector<Integer> slots11(num_slots_per_group * 2, 0);
  slots00[index00] = 1;
  slots00[num_slots_per_group + (index00 - kNumCols / 4 + num_slots_per_group) %
                                    num_slots_per_group] = 1;
  slots01[index01] = 1;
  slots01[num_slots_per_group + (index01 - kNumCols / 4 + num_slots_per_group) %
                                    num_slots_per_group] = 1;
  slots10[index10] = 1;
  slots10[num_slots_per_group + (index10 - kNumCols / 4 + num_slots_per_group) %
                                    num_slots_per_group] = 1;
  slots11[index11] = 1;
  slots11[num_slots_per_group + (index11 - kNumCols / 4 + num_slots_per_group) %
                                    num_slots_per_group] = 1;

  ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query00,
                       secret_key0.template EncryptBfv<Encoder>(
                           slots00, this->encoder_.get(),
                           this->error_params_.get(), prng0.get(), prng.get()));
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query01,
                       secret_key0.template EncryptBfv<Encoder>(
                           slots01, this->encoder_.get(),
                           this->error_params_.get(), prng0.get(), prng.get()));

  ASSERT_OK_AND_ASSIGN(prng, Prng::Create(kPrngSeed));
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query10,
                       secret_key1.template EncryptBfv<Encoder>(
                           slots10, this->encoder_.get(),
                           this->error_params_.get(), prng1.get(), prng.get()));
  ASSERT_OK_AND_ASSIGN(RnsCiphertext ct_query11,
                       secret_key1.template EncryptBfv<Encoder>(
                           slots11, this->encoder_.get(),
                           this->error_params_.get(), prng1.get(), prng.get()));

  auto [pad_rotated_queries00, ct_rotated_queries00] =
      GenerateRotatedQueries(ct_query00, gk0, num_rotations);
  auto [pad_rotated_queries01, ct_rotated_queries01] =
      GenerateRotatedQueries(ct_query01, gk0, num_rotations);
  auto [pad_rotated_queries10, ct_rotated_queries10] =
      GenerateRotatedQueries(ct_query10, gk1, num_rotations);
  auto [pad_rotated_queries11, ct_rotated_queries11] =
      GenerateRotatedQueries(ct_query11, gk1, num_rotations);
  std::vector<std::vector<std::vector<RnsCiphertext>>> ct_rotated_queries(2);
  ct_rotated_queries[0] = {std::move(ct_rotated_queries00),
                           std::move(ct_rotated_queries01)};
  ct_rotated_queries[1] = {std::move(ct_rotated_queries10),
                           std::move(ct_rotated_queries11)};

  // Preprocess the matrix
  ASSERT_OK(
      matrix->PreprocessDatabase(pad_rotated_queries00, pad_rotated_queries01));
  ASSERT_TRUE(matrix->IsPreprocessedDatabase());

  auto time_online_s = absl::Now();
  ASSERT_OK_AND_ASSIGN(auto ct_inner_products,
                       matrix->InnerProductWithMany(ct_rotated_queries));
  auto time_online_e = absl::Now();
  std::cout << "Online time: " << time_online_e - time_online_s << std::endl;

  ASSERT_EQ(ct_inner_products.size(), 2);
  ASSERT_EQ(ct_inner_products[0].size(), 2);
  ASSERT_EQ(ct_inner_products[1].size(), 2);
  ASSERT_OK_AND_ASSIGN(auto decrypted00,
                       secret_key0.template DecryptBfv<Encoder>(
                           ct_inner_products[0][0], this->encoder_.get()));
  ASSERT_OK_AND_ASSIGN(auto decrypted01,
                       secret_key0.template DecryptBfv<Encoder>(
                           ct_inner_products[0][1], this->encoder_.get()));
  ASSERT_OK_AND_ASSIGN(auto decrypted10,
                       secret_key1.template DecryptBfv<Encoder>(
                           ct_inner_products[1][0], this->encoder_.get()));
  ASSERT_OK_AND_ASSIGN(auto decrypted11,
                       secret_key1.template DecryptBfv<Encoder>(
                           ct_inner_products[1][1], this->encoder_.get()));
  ASSERT_EQ(decrypted00.size(), num_slots_per_group * 2);
  ASSERT_EQ(decrypted01.size(), num_slots_per_group * 2);
  ASSERT_EQ(decrypted10.size(), num_slots_per_group * 2);
  ASSERT_EQ(decrypted11.size(), num_slots_per_group * 2);

  int rows_per_block = kNumRows / 2;
  std::vector<Integer> results00(rows_per_block, 0);
  std::vector<Integer> results01(rows_per_block, 0);
  std::vector<Integer> results10(rows_per_block, 0);
  std::vector<Integer> results11(rows_per_block, 0);
  for (int i = 0; i < rows_per_block; ++i) {
    results00[i] += decrypted00[i];
    results00[i] += decrypted00[num_slots_per_group + i];
    results01[i] += decrypted01[i];
    results01[i] += decrypted01[num_slots_per_group + i];
    results10[i] += decrypted10[i];
    results10[i] += decrypted10[num_slots_per_group + i];
    results11[i] += decrypted11[i];
    results11[i] += decrypted11[num_slots_per_group + i];
  }
  for (int i = 0; i < rows_per_block; ++i) {
    EXPECT_EQ(results00[i] % this->rns_context_->PlaintextModulus(),
              data[i][index00] + data[i][kNumCols / 2 + index01]);
    EXPECT_EQ(results01[i] % this->rns_context_->PlaintextModulus(),
              data[rows_per_block + i][index00] +
                  data[rows_per_block + i][kNumCols / 2 + index01]);
    EXPECT_EQ(results10[i] % this->rns_context_->PlaintextModulus(),
              data[i][index10] + data[i][kNumCols / 2 + index11]);
    EXPECT_EQ(results11[i] % this->rns_context_->PlaintextModulus(),
              data[rows_per_block + i][index10] +
                  data[rows_per_block + i][kNumCols / 2 + index11]);
  }
}

TEST_F(WeightMatrixTest, StrassenWithLargerK) {
  constexpr int power = 5;

  auto data = SampleMatrix(kNumRows, kNumCols, 8);
  ASSERT_OK_AND_ASSIGN(
      auto matrix,
      WeightMatrix::Create(this->rns_context_.get(), this->gadget_.get(),
                           kPrngSeedForPad, kPrngSeedForGk));
  ASSERT_OK(matrix->SetData(data));

  // Preprocess the matrix
  int num_rotations = kNumCols / 2 / 2;
  ASSERT_OK(matrix->PreprocessPad(num_rotations));

  auto pad_rotated_ct_v = matrix->PreprocessedRotatedCiphertextPads();
  ASSERT_OK(matrix->PreprocessDatabase(pad_rotated_ct_v, pad_rotated_ct_v));

  ASSERT_OK(matrix->PreprocessBlocksForStrassen());
  ASSERT_TRUE(matrix->IsPreprocessedBlocksForStrassen());

  // Warm-up
  {
    std::vector<std::vector<std::string>> prng_seeds(2);
    std::vector<std::vector<RnsSecretKey>> sks(2);

    for (int j = 0; j < 2; ++j) {
      ASSERT_OK_AND_ASSIGN(auto seed, Prng::GenerateSeed());
      prng_seeds[0].push_back(seed);
      ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(seed));
      sks[0].push_back(GenerateSecretKey(prng.get()));

      ASSERT_OK_AND_ASSIGN(seed, Prng::GenerateSeed());
      prng_seeds[1].push_back(seed);
      ASSERT_OK_AND_ASSIGN(prng, Prng::Create(seed));
      sks[1].push_back(GenerateSecretKey(prng.get()));
    }

    std::vector<std::vector<RnsCiphertext>> ct_vs(2);
    ct_vs[0].reserve(2);
    ct_vs[1].reserve(2);
    std::vector<std::vector<RnsGaloisKey>> gks(2);
    gks[0].reserve(2);
    gks[1].reserve(2);

    ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
    int num_slots_per_group = 1 << (this->rlwe_params_.log_n - 1);
    std::vector<Integer> zeros(num_slots_per_group * 2, 0);
    for (int j = 0; j < 2; ++j) {
      ASSERT_OK_AND_ASSIGN(auto prng_pad, Prng::Create(kPrngSeedForPad));
      ASSERT_OK_AND_ASSIGN(
          RnsCiphertext ct00,
          sks[0][j].template EncryptBfv<Encoder>(zeros, this->encoder_.get(),
                                                 this->error_params_.get(),
                                                 prng.get(), prng_pad.get()));
      ct_vs[0].push_back(std::move(ct00));

      ASSERT_OK_AND_ASSIGN(prng_pad, Prng::Create(kPrngSeedForPad));
      ASSERT_OK_AND_ASSIGN(
          RnsCiphertext ct10,
          sks[1][j].template EncryptBfv<Encoder>(zeros, this->encoder_.get(),
                                                 this->error_params_.get(),
                                                 prng.get(), prng_pad.get()));
      ct_vs[1].push_back(std::move(ct10));

      gks[0].push_back(GenerateGaloisKey(sks[0][j], power));
      gks[1].push_back(GenerateGaloisKey(sks[1][j], power));
    }

    auto time_online_s = absl::Now();
    ASSERT_OK_AND_ASSIGN(auto result,
                         matrix->Strassen(ct_vs, gks, num_rotations));
    auto time_online_e = absl::Now();
    std::cout << "Warm-up average time: " << (time_online_e - time_online_s) / 2
              << std::endl;
  }

  int num_queries = absl::GetFlag(FLAGS_num_queries);
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  int k = num_queries >> 1;
  std::cout << "num_queries = " << k * 2 << std::endl;
  absl::Duration time_online = absl::ZeroDuration();
  for (int it = 0; it < num_iterations; ++it) {
    std::vector<std::vector<std::string>> prng_seeds(2);
    std::vector<std::vector<RnsSecretKey>> sks(2);

    for (int j = 0; j < k * 2; ++j) {
      ASSERT_OK_AND_ASSIGN(auto seed, Prng::GenerateSeed());
      prng_seeds[0].push_back(seed);
      ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(seed));
      sks[0].push_back(GenerateSecretKey(prng.get()));

      ASSERT_OK_AND_ASSIGN(seed, Prng::GenerateSeed());
      prng_seeds[1].push_back(seed);
      ASSERT_OK_AND_ASSIGN(prng, Prng::Create(seed));
      sks[1].push_back(GenerateSecretKey(prng.get()));
    }

    std::vector<std::vector<RnsCiphertext>> ct_vs(2);
    ct_vs[0].reserve(2 * k);
    ct_vs[1].reserve(2 * k);
    std::vector<std::vector<RnsGaloisKey>> gks(2);
    gks[0].reserve(2 * k);
    gks[1].reserve(2 * k);

    ASSERT_OK_AND_ASSIGN(auto prng, Prng::Create(kPrngSeed));
    int num_slots_per_group = 1 << (this->rlwe_params_.log_n - 1);
    std::vector<Integer> zeros(num_slots_per_group * 2, 0);
    for (int j = 0; j < k * 2; ++j) {
      ASSERT_OK_AND_ASSIGN(auto prng_pad, Prng::Create(kPrngSeedForPad));
      ASSERT_OK_AND_ASSIGN(
          RnsCiphertext ct00,
          sks[0][j].template EncryptBfv<Encoder>(zeros, this->encoder_.get(),
                                                 this->error_params_.get(),
                                                 prng.get(), prng_pad.get()));
      ct_vs[0].push_back(std::move(ct00));

      ASSERT_OK_AND_ASSIGN(prng_pad, Prng::Create(kPrngSeedForPad));
      ASSERT_OK_AND_ASSIGN(
          RnsCiphertext ct10,
          sks[1][j].template EncryptBfv<Encoder>(zeros, this->encoder_.get(),
                                                 this->error_params_.get(),
                                                 prng.get(), prng_pad.get()));
      ct_vs[1].push_back(std::move(ct10));

      gks[0].push_back(GenerateGaloisKey(sks[0][j], power));
      gks[1].push_back(GenerateGaloisKey(sks[1][j], power));
    }

    auto time_online_s = absl::Now();
    ASSERT_OK_AND_ASSIGN(auto result,
                         matrix->Strassen(ct_vs, gks, num_rotations));
    auto time_online_e = absl::Now();
    time_online += (time_online_e - time_online_s);
    std::cout << "Online time (k = " << k
              << "): " << time_online_e - time_online_s << std::endl;
    std::cout << "Average time: " << (time_online_e - time_online_s) / (2 * k)
              << std::endl;

    // Check results
    auto moduli = this->moduli_;
    auto raw_u = result.u00[0];
    ASSERT_OK_AND_ASSIGN(auto a00_sk00,
                         result.pad00->Mul(sks[0][0].Key(), moduli));
    ASSERT_OK_AND_ASSIGN(auto a01_sk10,
                         result.pad01->Mul(sks[1][0].Key(), moduli));
    ASSERT_OK(raw_u.AddInPlace(a00_sk00, moduli));
    ASSERT_OK(raw_u.AddInPlace(a01_sk10, moduli));
    ASSERT_OK_AND_ASSIGN(auto u, this->encoder_->DecodeBfv(raw_u, moduli));
    EXPECT_EQ(u.size(), kNumRows);

    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(u[i], 0);
    }
  }
  std::cout << "Average time (total): "
            << time_online / (2 * k) / num_iterations << std::endl;
}

}  // namespace
}  // namespace private_inference

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);

  return RUN_ALL_TESTS();
}
