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

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_TESTING_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_TESTING_H_

#include <string>
#include <vector>

#include "absl/random/random.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/utils.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace testing {

// Returns a random record of a database with the given parameters.
inline static std::string GenerateRandomRecord(const Parameters& params) {
  int num_bytes = DivAndRoundUp(params.db_record_bit_size, 8);
  std::string record(num_bytes, 0);
  absl::BitGen bitgen;
  for (int i = 0; i < num_bytes; ++i) {
    record[i] = absl::Uniform<unsigned char>(bitgen);
  }
  if (params.db_record_bit_size % 8 != 0) {
    unsigned char mask = (1 << (params.db_record_bit_size % 8)) - 1;
    record[num_bytes - 1] = record[num_bytes - 1] & mask;
  }
  return record;
}

inline static std::vector<Parameters::LweInteger> GenerateRandomQuery(
    int num_values) {
  absl::BitGen bitgen;
  std::vector<Parameters::LweInteger> query(num_values, 0);
  for (int i = 0; i < num_values; ++i) {
    query[i] = absl::Uniform<Parameters::LweInteger>(bitgen);
  }
  return query;
}

}  // namespace testing
}  // namespace hintless_simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_TESTING_H_
