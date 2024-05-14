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

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_

#include <cstdint>

#include "linpir/parameters.h"
#include "lwe/types.h"
#include "shell_encryption/serialization.pb.h"

namespace hintless_pir {
namespace hintless_simplepir {

// Parameters of the hintless SimplePIR protocol.
struct Parameters {
  using LweInteger = lwe::Integer;
  using RlweInteger = linpir::Uint64;

  int64_t db_rows;
  int64_t db_cols;
  int db_record_bit_size;

  int lwe_secret_dim;
  int lwe_modulus_bit_size;  // 32 or 64
  int lwe_plaintext_bit_size;
  double lwe_error_variance;

  linpir::RlweParameters<RlweInteger> linpir_params;

  rlwe::PrngType prng_type;
};

}  // namespace hintless_simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_PARAMETERS_H_
