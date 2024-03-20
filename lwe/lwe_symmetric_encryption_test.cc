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

#include "lwe/lwe_symmetric_encryption.h"

#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lwe/encode.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"

namespace hintless_pir {
namespace lwe {
namespace {

using rlwe::testing::StatusIs;
using Prng = rlwe::SingleThreadHkdfPrng;

constexpr int kNumRows = 20;
constexpr int kNumCols = 20;
constexpr int kLogScalingFactor = 8;

// Tests symmetric-key encryption scheme, including the following homomorphic
// operations:
//   * (plaintext) matrix (ciphertext) vector multiplication.
class SymmetricLweEncryptionTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string prng_seed = Prng::GenerateSeed().value();
    prng_ = Prng::Create(prng_seed).value();
    num_rows_ = kNumRows;
    num_cols_ = kNumCols;
    log_scaling_factor_ = kLogScalingFactor;
  }

  std::unique_ptr<Prng> prng_;
  int num_rows_;
  int num_cols_;
  int log_scaling_factor_;
};

// Things to test
// Error correction works
// Encrypt -> Decrypt
// Encrypt -> Lin Trans (selection) -> Decrypt

// Tests that the pad is created with the correct size
TEST_F(SymmetricLweEncryptionTest, ExpansionTest) {
  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  EXPECT_EQ(pad.rows(), num_rows_);
  EXPECT_EQ(pad.cols(), num_cols_);
}

// Tests that Encoding + adding noise + Decoding gets the plaintext back
TEST_F(SymmetricLweEncryptionTest, ErrorCorrectionTest) {
  Vector actual_ptxt = Vector::Zero(num_rows_);
  // First size/2 entries are small (mod q / 2^log_scaling_factor)
  // next size/2 entires are large
  for (int i = 0; i < num_rows_; ++i) {
    if (i < num_rows_ / 2) {
      actual_ptxt[i] = i;
    } else {
      actual_ptxt[i] = (1 << (kIntBitwidth - log_scaling_factor_)) - i;
    }
  }
  Vector m = actual_ptxt;
  ASSERT_OK(EncodeMessageInPlace(m, log_scaling_factor_))
  Vector e = SymmetricLweKey::Sample(num_rows_, prng_.get()).value().Key();
  m += e;
  ASSERT_OK(RemoveErrorInPlace(m, log_scaling_factor_));
  EXPECT_EQ(actual_ptxt, m);
}

// Tests that Encryption -> Decryption works
TEST_F(SymmetricLweEncryptionTest, EncryptThenDecryptTest) {
  std::vector<Integer> actual_ptxt(num_rows_, 0);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  auto c = SymmetricLweCiphertext(pad, b, log_scaling_factor_);
  ASSERT_OK_AND_ASSIGN(Vector plaintext, key.Decrypt(c));
  EXPECT_EQ(actual_plaintext.size(), plaintext.size());
  for (int i = 0; i < actual_plaintext.size(); ++i) {
    EXPECT_EQ(actual_plaintext[i], plaintext[i]);
  }
}

// Tests that Linear transformations work.
// We test it in the boring way of letting T be a row-vector that
// is the ith basis vector.
// It should yield a ciphertext that decrypts to the ith component of the
// plaintext, and is used to optimize decryption's running time in SimplePIR.
TEST_F(SymmetricLweEncryptionTest, LinearTransformationTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  auto c = SymmetricLweCiphertext(pad, b, log_scaling_factor_);
  // Creating a row-vector that will select the 5th element
  // from the plaintext
  Matrix LinTrans = Matrix::Zero(1, num_cols_);
  LinTrans(0, 5) = 1;
  auto status = c.HomLinTransInPlace(LinTrans);
  ASSERT_OK(status);
  ASSERT_OK_AND_ASSIGN(Vector plaintext, key.Decrypt(c));
  ASSERT_EQ(plaintext[0], 5);
}

// Input validation tests below here.

// Checks if applying a linear transformation to the ciphertext [pad, b]
// where the # rows of pad != the # of rows of b causes an error.
TEST_F(SymmetricLweEncryptionTest, LinTransRowsMalformedCtxtTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  // Pad with wrong number of rows
  ASSERT_OK_AND_ASSIGN(Matrix wrong_pad,
                       ExpandPad(num_rows_ + 1, num_cols_, prng_.get()));
  auto c = SymmetricLweCiphertext(wrong_pad, b, log_scaling_factor_);
  Matrix LinTrans = Matrix::Zero(1, num_cols_);
  LinTrans(0, 5) = 1;
  auto status = c.HomLinTransInPlace(LinTrans);
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("does not match the number of rows of b")));
}

// Checks if applying a linear transformation T to the ciphertext [pad, b]
// where the # cols of T does not match the # of ros of pad, b, leads to an
// error.
TEST_F(SymmetricLweEncryptionTest, LinTransWrongLinTransTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  auto c = SymmetricLweCiphertext(pad, b, log_scaling_factor_);
  // Wrong num_cols here
  Matrix LinTrans = Matrix::Zero(1, num_cols_ + 1);
  LinTrans(0, 5) = 1;
  auto status = c.HomLinTransInPlace(LinTrans);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "does not match the number of cols of lin_trans")));
}

// Checks if passing a nullptr prng to EncryptFromPadInPlace is caught
TEST_F(SymmetricLweEncryptionTest, EncryptFromPadInPlaceNullPrngTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  auto status = key.EncryptFromPadInPlace(
      actual_plaintext, pad, log_scaling_factor_, static_cast<Prng*>(nullptr));
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("The prng must not be null")));
}

// Checks if passing a negative log scaling factor to EncryptFromPadInPlace is
// caught
TEST_F(SymmetricLweEncryptionTest,
       EncryptFromPadInPlaceNegLogScalingFactorTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  auto status =
      key.EncryptFromPadInPlace(actual_plaintext, pad, -1, prng_.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("The log scaling factor,")));
}

// Checks if passing a log scaling factor above kIntBitwidth (=32 currently)
// to EncryptFromPadInPlace is caught
TEST_F(SymmetricLweEncryptionTest,
       EncryptFromPadInPlaceLargeLogScalingFactorTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  auto status = key.EncryptFromPadInPlace(actual_plaintext, pad,
                                          kIntBitwidth + 1, prng_.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("The log scaling factor,")));
}

// Checks if passing a mismatched plaintext to EncryptFromPadInPlace is caught
TEST_F(SymmetricLweEncryptionTest,
       EncryptFromPadInPlaceMismatchedPlaintextTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  // Wrong number of rows
  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_ + 1, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  auto status = key.EncryptFromPadInPlace(actual_plaintext, pad,
                                          log_scaling_factor_, prng_.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("The plaintext size, ")));
}

// Checks if passing a incorrectly-sized key to EncryptFromPadInPlace is caught
TEST_F(SymmetricLweEncryptionTest, EncryptFromPadInPlaceMismatchedKeyTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  // Wrong number of cols
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_ + 1, prng_.get()));
  auto status = key.EncryptFromPadInPlace(actual_plaintext, pad,
                                          log_scaling_factor_, prng_.get());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("The key length, ")));
}

// Checks if passing in a malformed ciphertext to ExtractErrorAndMessage is
// caught
TEST_F(SymmetricLweEncryptionTest,
       ExtractErrorAndMessageMalformedCiphertextTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  // Pad with wrong number of rows
  ASSERT_OK_AND_ASSIGN(Matrix wrong_pad,
                       ExpandPad(num_rows_ + 1, num_cols_, prng_.get()));
  auto c = SymmetricLweCiphertext(wrong_pad, b, log_scaling_factor_);
  auto status = key.ExtractErrorAndMessage(c);
  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("does not match the dimension of b")));
}

// Checks if passing in a incorrect dimension key to ExtractErrorAndMessage is
// caught
TEST_F(SymmetricLweEncryptionTest, ExtractErrorAndMessageWrongDimKeyTest) {
  auto actual_ptxt = std::vector<Integer>(num_rows_);
  for (int i = 0; i < num_rows_; ++i) {
    actual_ptxt[i] = i;
  }
  Vector actual_plaintext =
      Eigen::Map<Vector>(actual_ptxt.data(), actual_ptxt.size());

  ASSERT_OK_AND_ASSIGN(Matrix pad,
                       ExpandPad(num_rows_, num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey key,
                       SymmetricLweKey::Sample(num_cols_, prng_.get()));
  ASSERT_OK_AND_ASSIGN(
      Vector b, key.EncryptFromPad(actual_plaintext, pad, log_scaling_factor_,
                                   prng_.get()));
  auto c = SymmetricLweCiphertext(pad, b, log_scaling_factor_);
  // Wrong dimension key here
  ASSERT_OK_AND_ASSIGN(SymmetricLweKey wrong_key,
                       SymmetricLweKey::Sample(num_cols_ + 1, prng_.get()));
  auto status = wrong_key.ExtractErrorAndMessage(c);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("The key length, ")));
}

TEST_F(SymmetricLweEncryptionTest, SampleTooSmallKeyTest) {
  EXPECT_THAT(SymmetricLweKey::Sample(0, prng_.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("The number of coefficients")));
}

TEST_F(SymmetricLweEncryptionTest, SampleNullPrngTest) {
  EXPECT_THAT(SymmetricLweKey::template Sample<Prng>(num_cols_, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("The prng must not be null")));
}

}  // namespace
}  // namespace lwe
}  // namespace hintless_pir
