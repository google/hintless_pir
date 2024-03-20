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

#include "hintless_simplepir/utils.h"

#include <string>
#include <vector>

#include "absl/random/random.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/parameters.h"
#include "lwe/types.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

const std::vector<Parameters> kTestParameters{
    Parameters{
        .db_record_bit_size = 8,
        .lwe_plaintext_bit_size = 8,
    },
    Parameters{
        .db_record_bit_size = 7,
        .lwe_plaintext_bit_size = 8,
    },
    Parameters{
        .db_record_bit_size = 16,
        .lwe_plaintext_bit_size = 8,
    },
    Parameters{
        .db_record_bit_size = 15,
        .lwe_plaintext_bit_size = 7,
    },
    Parameters{
        .db_record_bit_size = 128,
        .lwe_plaintext_bit_size = 8,
    },
};

static std::string GenerateRandomRecord(const Parameters& params) {
  int num_bytes = DivAndRoundUp(params.db_record_bit_size, 8);
  std::string record(num_bytes, 0);
  absl::BitGen bitgen;
  for (int i = 0; i < num_bytes; ++i) {
    record[i] = absl::Uniform<unsigned char>(bitgen);
  }
  char mask = (1 << (params.db_record_bit_size % 8)) - 1;
  record[num_bytes - 1] = record[num_bytes - 1] & mask;
  return record;
}

TEST(UtilsTest, SplitAndReconstruct) {
  for (auto const& params : kTestParameters) {
    std::string record = GenerateRandomRecord(params);

    int expected_num_shards =
        DivAndRoundUp(params.db_record_bit_size, params.lwe_plaintext_bit_size);
    std::vector<lwe::Integer> values = SplitRecord(record, params);
    EXPECT_EQ(values.size(), expected_num_shards);

    std::string reconstruced = ReconstructRecord(values, params);
    EXPECT_EQ(reconstruced, record);
  }
}

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir
