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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/database_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/testing.h"
#include "lwe/types.h"

ABSL_FLAG(int, num_rows, 1024, "Number of rows");
ABSL_FLAG(int, num_cols, 1024, "Number of cols");

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

using RlweInteger = Parameters::RlweInteger;

const Parameters kParameters{
    .db_rows = 1024,
    .db_cols = 1024,
    .db_record_bit_size = 8,
    .lwe_secret_dim = 1400,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 8,
    .lwe_error_variance = 8,
    .prng_type = rlwe::PRNG_TYPE_HKDF,
};

void BM_InnerProductWith(benchmark::State& state) {
  int64_t num_rows = absl::GetFlag(FLAGS_num_rows);
  int64_t num_cols = absl::GetFlag(FLAGS_num_cols);
  Parameters params = kParameters;
  params.db_rows = num_rows;
  params.db_cols = num_cols;

  // Create a database and fill in random database records.
  const auto database = Database::CreateRandom(params).value();
  ASSERT_EQ(database->NumRecords(), num_rows * num_cols);

  std::vector<lwe::Integer> query = testing::GenerateRandomQuery(num_cols);

  for (auto _ : state) {
    auto results = database->InnerProductWith(query);
    benchmark::DoNotOptimize(results);
  }
}
BENCHMARK(BM_InnerProductWith);

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir

// Declare benchmark_filter flag, which will be defined by benchmark library.
// Use it to check if any benchmarks were specified explicitly.
//
namespace benchmark {
extern std::string FLAGS_benchmark_filter;
}
using benchmark::FLAGS_benchmark_filter;

int main(int argc, char* argv[]) {
  FLAGS_benchmark_filter = "";
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  if (!FLAGS_benchmark_filter.empty()) {
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  return 0;
}
