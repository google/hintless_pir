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

#ifndef HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_H_
#define HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_H_

#include <utility>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "lwe/encode.h"
#include "lwe/sample_error.h"
#include "lwe/types.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace lwe {

// Expands the pad from a prng.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::StatusOr<Matrix> ExpandPad(int num_rows, int num_cols,
                                        Prng* encryption_prng) {
  if (num_rows < 1) {
    return absl::InvalidArgumentError("The number of rows must be positive.");
  } else if (num_cols < 1) {
    return absl::InvalidArgumentError("The number of cols must be positive.");
  } else if (encryption_prng == nullptr) {
    return absl::InvalidArgumentError("The prng must not be null.");
  }
  return SampleUniformMatrix(num_rows, num_cols, encryption_prng);
}

// This file implements the somewhat homomorphic symmetric-key encryption scheme
// used in SimplePIR
// https://eprint.iacr.org/2022/949
//
// Which itself dates back to Regev's paper introducing
// Learning with Errors-based encryption
// https://cims.nyu.edu/~regev/papers/qcrypto.pdf
//
// Only homomorphic operations required for HintlessSimplePIR are implemented.
// Moreover, priority is given to simplicity of implementation (vs generality).
//
// Each ciphertext comprises a pair [pad, b], where
// * b \in Z_q^m, and
// * pad \in \Z_q^{m \times n}
// * for q = 2^32.
// and
// b := pad*s + e + \Delta * m for
// * s, e centered binomial vectors (see `hintless_simplepir/sample_error.h`)
// * \Delta >0 an integer scaling factor.

// This implementation supports the following homomorphic operation:
//  - Multiplying an encrypted vector by a scalar matrix
//
// This is the only homomorphic operation required for HintlessSimplePIR.
class SymmetricLweCiphertext {
 public:
  // Create a ciphertext by supplying the pair of components, and
  // the scaling factor used during encryption
  explicit SymmetricLweCiphertext(Matrix pad, Vector b, int log_scaling_factor)
      : pad_(std::move(pad)),
        b_(std::move(b)),
        log_scaling_factor_(log_scaling_factor) {}

  absl::Status HomLinTransInPlace(RefMatrix lin_trans) {
    auto num_rows_pad = this->pad_.rows();
    auto num_rows_b = this->b_.rows();
    auto num_cols_lt = lin_trans.cols();

    if (num_rows_pad != num_rows_b) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The number of rows of the pad, ", num_rows_pad,
          ", does not match the number of rows of b, ", num_rows_b));
    } else if (num_rows_pad != num_cols_lt) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The number of rows of the ciphertext, ", num_rows_pad,
          ", does not match the number of cols of lin_trans, ", num_cols_lt));
    }
    this->pad_ = lin_trans * this->pad_;
    this->b_ = lin_trans * this->b_;
    return absl::OkStatus();
  }

  // Accessors.
  unsigned int Size() const { return pad_.size() + b_.size(); }
  const Matrix& Pad() const { return pad_; }
  const Vector& B() const { return b_; }
  int LogScalingFactor() const { return log_scaling_factor_; }

 private:
  // The ciphertext "A" component.
  Matrix pad_;
  // The ciphertext "b" component.
  Vector b_;
  // Plaintext space scaling factor.
  int log_scaling_factor_;
};

// Holds a key that can be used to encrypt messages using the LWE-based
// encryption scheme.
class SymmetricLweKey {
 public:
  // Static factory that samples a key from the error distribution.
  template <typename Prng = rlwe::SingleThreadHkdfPrng>
  static absl::StatusOr<SymmetricLweKey> Sample(int num_coeffs, Prng* prng) {
    if (num_coeffs < 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of coefficients of the key, ", num_coeffs,
                       ", should be positive."));
    }
    if (prng == nullptr) {
      return absl::InvalidArgumentError("The prng must not be null.");
    }
    RLWE_ASSIGN_OR_RETURN(Vector key, SampleUniformTernary(num_coeffs, prng));
    return SymmetricLweKey(std::move(key));
  }

  // Encrypts the plaintext using learning-with-errors (LWE) encryption.
  // Takes the matrix `pad` as input, and the output ciphertext is stored in
  // `plaintext` in-place.
  template <typename Prng = rlwe::SingleThreadHkdfPrng>
  absl::Status EncryptFromPadInPlace(Vector& plaintext, const Matrix& pad,
                                     const int log_scaling_factor,
                                     Prng* prng) const {
    if (prng == nullptr) {
      return absl::InvalidArgumentError("The prng must not be null");
    } else if (log_scaling_factor < 0 || log_scaling_factor > kIntBitwidth) {
      return absl::InvalidArgumentError(
          absl::StrCat("The log scaling factor, ", log_scaling_factor,
                       ", should be >= 0 and <= ", kIntBitwidth));
    } else if (plaintext.size() != pad.rows()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The plaintext size, ", plaintext.size(),
          ", does not match the number of rows of the pad, ", pad.rows()));
    } else if (Len() != pad.cols()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The key length, ", Len(),
          ", does not match the number of cols of the pad, ", pad.cols()));
    }
    // Encodes the vector
    RLWE_RETURN_IF_ERROR(EncodeMessageInPlace(plaintext, log_scaling_factor));
    // Samples the Centered binomial and adds it to the (encoded) plaintext
    RLWE_RETURN_IF_ERROR(SampleAndAddCenteredBinomialInPlace(plaintext, prng));
    // Adds pad * s to the encoded vector \Delta * m + e
    plaintext += pad * key_;
    return absl::OkStatus();
  }

  // Encrypts the plaintext using learning-with-errors (LWE) encryption.
  // Takes the matrix `pad` as input, and returns the ciphertext.
  // Defers validating inputs to EncryptFromPadInPlace.
  template <typename Prng = rlwe::SingleThreadHkdfPrng>
  absl::StatusOr<Vector> EncryptFromPad(const Vector& plaintext,
                                        const Matrix& pad,
                                        const int log_scaling_factor,
                                        Prng* prng) const {
    Vector output = plaintext;
    RLWE_RETURN_IF_ERROR(
        EncryptFromPadInPlace(output, pad, log_scaling_factor, prng));
    return output;
  }

  // Extracts the error and message \Delta * m + e from a LWE ciphertext.
  absl::StatusOr<Vector> ExtractErrorAndMessage(
      const SymmetricLweCiphertext& ciphertext) const {
    const Matrix& pad = ciphertext.Pad();
    Vector b = ciphertext.B();
    if (pad.rows() != b.rows()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of rows of `pad`, ", pad.rows(),
                       " does not match the dimension of b, ", b.size()));
    } else if (Len() != pad.cols()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The key length, ", Len(),
          ", does not match the number of cols of A, ", pad.cols()));
    }
    b -= pad * key_;
    return b;
  }

  // Decrypts a LWE ciphertext and returns the plaintext.
  absl::StatusOr<Vector> Decrypt(
      const SymmetricLweCiphertext& ciphertext) const {
    int log_scaling_factor = ciphertext.LogScalingFactor();
    RLWE_ASSIGN_OR_RETURN(Vector noisy_m, ExtractErrorAndMessage(ciphertext));
    RLWE_RETURN_IF_ERROR(RemoveErrorInPlace(noisy_m, log_scaling_factor));
    return noisy_m;
  }

  // Accessors.
  int Len() const { return key_.size(); }
  const Vector& Key() { return key_; }

 private:
  // A constructor. Does not take ownership of params.
  explicit SymmetricLweKey(Vector key) : key_(std::move(key)) {}

  // The contents of the key itself.
  Vector key_;
};

}  // namespace lwe
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_H_
