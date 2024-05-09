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

#ifndef HINTLESS_PIR_LWE_TYPES_H_
#define HINTLESS_PIR_LWE_TYPES_H_

// Certain basic types used in LWE-based encryption scheme.

#include <cstdint>

#include "Eigen/Core"

namespace hintless_pir {
namespace lwe {

// Unsigned integer type to store an LWE ciphertext element. Either uint32_t or
// uint64_t for practical LWE parameters.
using Integer = uint32_t;
using Matrix = Eigen::Matrix<Integer, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Vector<Integer, Eigen::Dynamic>;

// Unsigned integer type to store an LWE plaintext element. This will be the
// type of the database element. Either uint8_t or uint16_t for practical LWE
// parameters.
using PlainInteger = uint8_t;

// Required to use Eigen without templates, see
// https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
using RefMatrix = Eigen::Ref<Matrix>;
using RefVector = Eigen::Ref<Vector>;

constexpr int kIntBitwidth = 8 * sizeof(Integer);

}  // namespace lwe
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LWE_TYPES_H_
