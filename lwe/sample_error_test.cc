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

// Testing using the tests in
//
// /third_party/rlwe/sample_error_test.cc
//
// Only the first test makes sense in our context, as we are hard-coding
// the variance to be 8.

#include "lwe/sample_error.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lwe/types.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"
#include "shell_encryption/testing/testing_prng.h"

namespace hintless_pir {
namespace lwe {
namespace {

using rlwe::testing::StatusIs;
using rlwe::testing::TestingPrng;

constexpr int kTestingRounds = 10;

TEST(SampleErrorTest, CheckUpperBoundOnNoise) {
  const std::vector<int> k_num_coeffs = {1200, 1024, 2};

  auto prng = std::make_unique<TestingPrng>(0);

  for (int i = 0; i < kTestingRounds; ++i) {
    for (auto num_coeffs : k_num_coeffs) {
      ASSERT_OK_AND_ASSIGN(Vector error,
                           SampleCenteredBinomial(num_coeffs, prng.get()));
      // Check that each coefficient is in [-16, 16]
      // As we are sampling with Variance = 8 via
      // \sum_{i = 1}^16 B_i - B_i', this is a (theoretical) worst-case
      // bound on the size of the output.
      for (int k = 0; k < num_coeffs; k++) {
        // Checking if each coefficient is in [-16, 16]
        // by checking if coeff + 16 is in [0, 32]
        EXPECT_LT(error[k] + 16, 32 + 1);
      }
    }
  }
}

TEST(SampleErrorTest, CheckUniformTernary) {
  const std::vector<int> k_num_coeffs = {1200, 1024, 2};
  constexpr Integer k_lwe_modulus_half = 1 << (kIntBitwidth / 2);
  constexpr Integer plus = 1;
  constexpr Integer minus = -plus;

  auto prng = std::make_unique<TestingPrng>(0);

  for (int i = 0; i < kTestingRounds; ++i) {
    for (auto num_coeffs : k_num_coeffs) {
      ASSERT_OK_AND_ASSIGN(Vector error,
                           SampleUniformTernary(num_coeffs, prng.get()));
      // Check that each coefficient is in {-1, 0, 1} mod 2^32.
      for (int k = 0; k < num_coeffs; k++) {
        if (error[k] > k_lwe_modulus_half) {
          EXPECT_EQ(error[k], minus);
        } else {
          EXPECT_LE(error[k], plus);
        }
      }
    }
  }
}

TEST(SampleErrorTest, BinomialNegCoeffsTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleCenteredBinomial(-1, prng.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("non-negative")));
}

TEST(SampleErrorTest, BinomialNullPrngTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleCenteredBinomial(10, static_cast<TestingPrng*>(nullptr));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("null")));
}

TEST(SampleErrorTest, UniformVecNegCoeffsTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleUniformVector(-1, prng.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("non-negative")));
}

TEST(SampleErrorTest, UniformVecNullPrngTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleUniformVector(10, static_cast<TestingPrng*>(nullptr));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("null")));
}

TEST(SampleErrorTest, UniformMatNegRowsTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleUniformMatrix(-1, 2, prng.get());
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("num_rows must be non-negative.")));
}

TEST(SampleErrorTest, UniformMatNegColsTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleUniformMatrix(2, -1, prng.get());
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("num_cols must be non-negative.")));
}

TEST(SampleErrorTest, UniformMatNullPrngTest) {
  auto prng = std::make_unique<TestingPrng>(0);
  auto status = SampleUniformMatrix(10, 10, static_cast<TestingPrng*>(nullptr));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("null")));
}

}  // namespace
}  // namespace lwe
}  // namespace hintless_pir
