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

#ifndef HINTLESS_PIR_PRIVATE_INFERENCE_PARAMETERS_H_
#define HINTLESS_PIR_PRIVATE_INFERENCE_PARAMETERS_H_
// This file defines the parameters to instantiate RLWE and LinPIR systems

#include <cstddef>
#include <vector>

#include "shell_encryption/integral_types.h"
#include "shell_encryption/serialization.pb.h"

namespace private_inference {

using Uint64 = rlwe::Uint64;

using Integer = Uint64;

struct ModelParameters {
  int input_dimension;
  int output_dimension;
  int batch_size;
  int num_rounds;
};

struct RlweParameters {
  int log_n;                // log2(N)
  std::vector<Integer> qs;  // RNS moduli
  std::vector<size_t> gadget_log_bs;
  Integer plaintext_modulus;
  double error_variance;
  rlwe::PrngType prng_type;

  // Encoding a matrix into blocks.
  int rows_per_block;
};

}  // namespace private_inference

#endif  // HINTLESS_PIR_PRIVATE_INFERENCE_PARAMETERS_H_
