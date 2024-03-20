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

#ifndef HINTLESS_PIR_LINPIR_PARAMETERS_H_
#define HINTLESS_PIR_LINPIR_PARAMETERS_H_

#include <cstddef>
#include <vector>

#include "shell_encryption/integral_types.h"
#include "shell_encryption/serialization.pb.h"

namespace hintless_pir {
namespace linpir {

using Uint32 = rlwe::Uint32;
using Uint64 = rlwe::Uint64;

// Parameters of the RLWE-based LinPIR protocol.
// We instantiate a RLWE-based BFV homomorphic encryption scheme, which requires
// the following parameters:
// - log_n: the base-2 log of the polynomial ring degree.
// - qs:    the RNS moduli whose product is the ciphertext modulus. Member of
//          `qs` must be distinct primes q such that 2^(log_n + 1) divides q-1.
// - ts:    the RNS moduli whose product is the ambient plaintext modulus. The
//          product of `ts` is used as the plaintext computation modulus when
//          using multiple LinPir instances, where each LinPir is instantiated
//          with a member of `ts`. Members of `ts` must be distinct primes t
//          such that 2^(log_n + 1) divides t-1.
// - gadget_log_bs:  the base-2 log of the gadget bases, one per ciphertext
//          modulus in `qs`.
// - error_variance: the variance of a centered binomial distribution, which
//          is used as the RLWE error distribution.
// - prng_type: The type of PRNG to sample random polynomials.
// - rows_per_block: the number of rows of the database matrix in every block
//          of the database encoding.
template <typename RlweInteger>
struct RlweParameters {
  int log_n;
  std::vector<RlweInteger> qs;
  std::vector<RlweInteger> ts;
  std::vector<size_t> gadget_log_bs;
  double error_variance;
  rlwe::PrngType prng_type;

  // Encoding a matrix into blocks.
  int rows_per_block;
};

}  // namespace linpir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LINPIR_PARAMETERS_H_
