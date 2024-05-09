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

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_UTILS_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/serialization.pb.h"
#include "lwe/types.h"

namespace hintless_pir {
namespace hintless_simplepir {

// Returns ceil(x / y).
template <typename T>
inline T DivAndRoundUp(T x, T y) {
  return (x + y - 1) / y;
}

// Splits `record` per `params.db_record_bit_size` bits, and returns the vector
// that contains the resulting chunks of bits.
inline std::vector<lwe::Integer> SplitRecord(absl::string_view record,
                                             const Parameters& params) {
  int num_shards =
      DivAndRoundUp(params.db_record_bit_size, params.lwe_plaintext_bit_size);
  std::vector<lwe::Integer> values(num_shards, 0);
  lwe::Integer curr_bits{0};  // Buffer of bits for the current shard.
  int shard_idx = 0;
  int num_filled_ptxt_bits = 0;  // # plaintext bits currently filled in.
  int num_remaining_bits = params.db_record_bit_size;
  for (auto it = record.begin(); it != record.end(); ++it) {
    // Fill in the remaining plaintext bits from the byte referenced by `it`.
    int num_available_ptxt_bits =
        std::min(8, params.lwe_plaintext_bit_size - num_filled_ptxt_bits);
    int num_fill_bits = std::min(num_available_ptxt_bits, num_remaining_bits);
    lwe::Integer mask = (lwe::Integer{1} << num_fill_bits) - 1;
    curr_bits |= (static_cast<lwe::Integer>(*it) & mask)
                 << num_filled_ptxt_bits;
    num_filled_ptxt_bits += num_fill_bits;
    num_remaining_bits -= num_fill_bits;

    // We have used up all the plaintext bits for the current shard, or all the
    // record bits are used.
    if (num_filled_ptxt_bits >= params.lwe_plaintext_bit_size ||
        num_remaining_bits == 0) {
      values[shard_idx] = curr_bits;
      shard_idx++;

      // Extract the remaining bits in `*it` and use them for the next shard.
      mask = (lwe::Integer{1} << 8) - mask;
      curr_bits = (static_cast<lwe::Integer>(*it) & mask) >> num_fill_bits;
      num_filled_ptxt_bits = 8 - num_fill_bits;
    }
  }
  if (num_remaining_bits > 0) {
    // This happens when the record size is not a multiple of plaintext space.
    values[shard_idx] = curr_bits;
  }
  return values;
}

// Returns the record that was splitted into `values`, where each value contains
// `params.db_record_bit_size` bits of the record.
inline std::string ReconstructRecord(absl::Span<const lwe::Integer> values,
                                     const Parameters& params) {
  int num_bytes = DivAndRoundUp(params.db_record_bit_size, 8);
  std::string record(num_bytes, 0);
  auto curr_byte = record.begin();
  int num_remaining_bits = params.db_record_bit_size;
  for (int i = 0; i < values.size(); ++i) {
    // We use the bits in `values[i]` in two steps: first we fill in the current
    // byte of `record` with the "head" bits, and then use the remaining "body"
    // bits to fill in `record` starting from the next byte.
    lwe::Integer value = values[i];
    int num_filled_bits = (params.db_record_bit_size - num_remaining_bits) % 8;
    int num_shard_bits =
        std::min(num_remaining_bits, params.lwe_plaintext_bit_size);
    int num_head_bits = std::min(8 - num_filled_bits, num_shard_bits);
    lwe::Integer head_mask = (lwe::Integer{1} << num_head_bits) - 1;
    *curr_byte |= static_cast<char>(value & head_mask) << num_filled_bits;
    if (num_filled_bits + num_head_bits == 8) {
      curr_byte++;
    }
    value >>= num_head_bits;

    int num_body_bits = num_shard_bits - num_head_bits;
    int num_next_bits = std::min(8, num_body_bits);
    while (num_next_bits > 0) {
      lwe::Integer mask = (lwe::Integer{1} << num_next_bits) - 1;
      *curr_byte |= static_cast<char>(value & mask);
      if (num_next_bits == 8) {
        curr_byte++;
      }
      value >>= num_next_bits;
      num_body_bits -= num_next_bits;
      num_next_bits = std::min(8, num_body_bits);
    }

    num_remaining_bits -= num_shard_bits;
  }
  return record;
}

inline SerializedLweCiphertext SerializeLweCiphertext(
    const lwe::Vector& ct_vector) {
  SerializedLweCiphertext serialized;
  serialized.mutable_b_coeffs()->Resize(ct_vector.size(), 0);
  Eigen::Map<lwe::Vector> map(serialized.mutable_b_coeffs()->mutable_data(),
                              ct_vector.size());
  map = ct_vector;
  return serialized;
}

inline SerializedLweCiphertext SerializeLweCiphertext(
    const std::vector<lwe::Integer>& ct_vector) {
  SerializedLweCiphertext serialized;
  serialized.mutable_b_coeffs()->Reserve(ct_vector.size());
  serialized.mutable_b_coeffs()->Add(ct_vector.begin(), ct_vector.end());
  return serialized;
}

inline std::vector<lwe::Integer> DeserializeLweCiphertext(
    const SerializedLweCiphertext& serialized) {
  std::vector<lwe::Integer> vec(serialized.b_coeffs().begin(),
                                serialized.b_coeffs().end());
  return vec;
}

// Given an integer `x` representing a mod-q number, returns `x` mod p, where
// modular numbers are in balanced representation.
template <typename Integer>
inline Integer ConvertModulus(const Integer& x, const Integer& q,
                              const Integer& p, const Integer& q_half) {
  if (x > q_half) {
    Integer diff = q - x;
    return p - diff % p;
  } else {
    return x % p;
  }
}

}  // namespace hintless_simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_UTILS_H_
