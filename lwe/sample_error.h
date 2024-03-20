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

#ifndef HINTLESS_PIR_LWE_SAMPLE_ERROR_H_
#define HINTLESS_PIR_LWE_SAMPLE_ERROR_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "Eigen/Core"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "lwe/types.h"
#include "shell_encryption/bits_util.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace lwe {

// Takes as input a uint32_t buffer, and adds an i.i.d. Centered Binomial
// (of Variance 8) to each coordinate of the buffer.
//
// These are distributed according to
// \sum_{i=1}^16 B_i-B_i' for i.i.d. random coinflips B_i, B_i'.
//
// We sample (B_i, B_i') during each loop iteration for efficiency.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::Status SampleAndAddCenteredBinomialInPlace(Vector& buffer,
                                                        Prng* prng) {
  int num_coeffs = buffer.size();
  // Always holds in practice, so do not bother handling num_coeffs % 2 = 1
  if (num_coeffs % 2 != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The number of coefficients, ", num_coeffs, ", must be even."));
  }
  if (prng == nullptr) {
    return absl::InvalidArgumentError("prng must not be null.");
  }

  // Optimizes that the variance = 8 exactly
  // so we need one PRNG call per pair of two coefficients (32 bits each).
  // Masks for extracting 4 groups of 16 bits from 64 bits.
  constexpr uint64_t mask = 0xFFFF;
  for (unsigned int i = 0; i < num_coeffs; i += 2) {
    // Computing \sum_i=1^16 B_i - B_i' for i and i+1 in parallel
    RLWE_ASSIGN_OR_RETURN(uint64_t r64, prng->Rand64());
    buffer[i] += rlwe::internal::CountOnes64(r64 & mask);
    buffer[i] -= rlwe::internal::CountOnes64(r64 & (mask << 16));
    buffer[i + 1] += rlwe::internal::CountOnes64(r64 & mask << 32);
    buffer[i + 1] -= rlwe::internal::CountOnes64(r64 & (mask << 48));
  }
  return absl::OkStatus();
}

// Samples a centered binomial, allocating and returning the Vector it is
// contained in.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::StatusOr<Vector> SampleCenteredBinomial(int num_coeffs,
                                                     Prng* prng) {
  if (num_coeffs < 0) {
    return absl::InvalidArgumentError("num_coeffs must be non-negative.");
  }
  // To handle an odd number of coefficients ---
  // round up to an even number, then resize the vector back down
  // afterwards.
  Vector output = Vector::Zero(num_coeffs + (num_coeffs % 2));
  RLWE_RETURN_IF_ERROR(SampleAndAddCenteredBinomialInPlace(output, prng));
  output.resize(num_coeffs);
  return output;
}

// Samples a vector whose coefficients are uniformly random I.I.D. ternary,
// i.e. uniformly random over {-1, 0, 1}, represented modulo 2^32.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::StatusOr<Vector> SampleUniformTernary(int num_coeffs, Prng* prng) {
  if (num_coeffs <= 0) {
    return absl::InvalidArgumentError("`num_coeffs` must be positive.");
  }
  if (prng == nullptr) {
    return absl::InvalidArgumentError("`prng` must not be null.");
  }

  // Below implements a parallel rejection sampling algorithm as suggested in
  // "A constant-time sampler for close-to-uniform bitsliced ternary vectors"
  // by Pierre Karpman, https://hal.archives-ouvertes.fr/hal-03777885
  // An element from {-1, 0, 1} is represented using two bits: 0 as (0,0), 1 as
  // (1,0), and -1 as (1,1). The algorithm samples in batches two uniformly
  // random 8-bit integers, r0 and r1, and uses a mask to indicate if a certain
  // bit in r0 and r1 is the invalid representation (0,1) and needs re-sample.
  Integer plus = 1;
  Integer minus = -plus;
  std::vector<Integer> coeffs;
  coeffs.reserve(num_coeffs);
  while (num_coeffs > 0) {
    int num_filled_coeffs = std::min(num_coeffs, 8);
    // The mask to indicate if a bit index requires re-sampling.
    uint8_t missing_bits =
        num_filled_coeffs < 8 ? (1 << num_filled_coeffs) - 1 : ~0;
    uint8_t encoding_bits0 = 0;
    uint8_t encoding_bits1 = 0;
    while (missing_bits != 0) {
      RLWE_ASSIGN_OR_RETURN(uint8_t rand_bits0, prng->Rand8());
      RLWE_ASSIGN_OR_RETURN(uint8_t rand_bits1, prng->Rand8());
      encoding_bits0 ^= (rand_bits0 & missing_bits);
      encoding_bits1 ^= (rand_bits1 & missing_bits);
      missing_bits = ~encoding_bits0 & encoding_bits1;
    }

    for (int i = 0; i < num_filled_coeffs; ++i) {
      uint8_t mask = 1 << i;
      bool bit0 = ((encoding_bits0 & mask) > 0);
      bool bit1 = ((encoding_bits1 & mask) > 0);
      Integer is_plus = -static_cast<Integer>(bit0 && !bit1);
      Integer is_minus = -static_cast<Integer>(bit0 && bit1);
      Integer value = (is_plus & plus) | (is_minus & minus);
      coeffs.push_back(value);
    }
    num_coeffs -= num_filled_coeffs;
  }
  Eigen::Map<Vector> map(coeffs.data(), coeffs.size());
  return map;
}

// Samples a vector of uniforms without allocating
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::Status SampleUniformVectorInPlace(Vector& buffer, Prng* prng) {
  int num_coeffs = buffer.size();
  if (num_coeffs % 2 != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("The size of `buffer`, ", num_coeffs, ", must be even."));
  }
  if (prng == nullptr) {
    return absl::InvalidArgumentError("prng must not be null.");
  }
  constexpr uint64_t low_mask = 0x00000000ffffffff;
  for (int i = 0; i < num_coeffs; i += 2) {
    RLWE_ASSIGN_OR_RETURN(uint64_t sample, prng->Rand64());
    buffer[i] = (static_cast<Integer>(sample & low_mask));
    buffer[i + 1] = (static_cast<Integer>(sample >> 32));
  }
  return absl::OkStatus();
}

// Samples a vector of uniforms via allocating
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::StatusOr<Vector> SampleUniformVector(int num_coeffs, Prng* prng) {
  if (num_coeffs < 0) {
    return absl::InvalidArgumentError("num_coeffs must be non-negative.");
  }
  Vector buffer = Vector::Zero(num_coeffs);
  RLWE_RETURN_IF_ERROR(SampleUniformVectorInPlace(buffer, prng));
  return buffer;
}

// Samples a matrix of uniforms, allocating and returning the matrix it is
// contained in.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
static absl::StatusOr<Matrix> SampleUniformMatrix(int num_rows, int num_cols,
                                                  Prng* prng) {
  if (num_rows < 0) {
    return absl::InvalidArgumentError("num_rows must be non-negative.");
  }
  if (num_cols < 0) {
    return absl::InvalidArgumentError("num_cols must be non-negative.");
  }
  if (prng == nullptr) {
    return absl::InvalidArgumentError("prng must not be null.");
  }
  Matrix output = Matrix::Zero(num_rows, num_cols);
  for (int i = 0; i < num_rows; ++i) {
    Vector buffer = Vector::Zero(num_cols);
    RLWE_RETURN_IF_ERROR(SampleUniformVectorInPlace(buffer, prng));
    output.row(i) += buffer;
  }
  return output;
}

}  // namespace lwe
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LWE_SAMPLE_ERROR_H_
