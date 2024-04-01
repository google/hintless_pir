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

#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "linpir/client.h"
#include "linpir/database.h"
#include "linpir/parameters.h"
#include "linpir/server.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/testing/status_testing.h"

ABSL_FLAG(int, num_rows, 1024, "Number of rows");
ABSL_FLAG(int, num_cols, 1400, "Number of cols");

namespace hintless_pir {
namespace linpir {
namespace {

using Integer = Uint64;
using ModularInt = rlwe::MontgomeryInt<Integer>;
using RnsContext = rlwe::RnsContext<ModularInt>;
using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
using Prng = rlwe::SingleThreadHkdfPrng;

const RlweParameters<Integer> kRlweParameters{
    .log_n = 12,
    .qs = {18014398509309953ULL, 18014398509293569ULL},  // 108 bits
    .ts = {4169729, 4120577},
    .gadget_log_bs = {18, 18},
    .error_variance = 8,
    .prng_type = rlwe::PRNG_TYPE_HKDF,
    .rows_per_block = 1024,
};

// Returns a vector of `num_values` many random integers in [0, max_value).
inline std::vector<Integer> SampleValues(int num_values, Integer max_value) {
  absl::BitGen bitgen;
  std::vector<Integer> values;
  values.reserve(num_values);
  for (int i = 0; i < num_values; ++i) {
    values.push_back(absl::Uniform<Integer>(bitgen, 0, max_value));
  }
  return values;
}

// Returns a matrix of dimension `num_rows` * `num_cols`, with values in
// the range [0, max_value).
inline std::vector<std::vector<Integer>> SampleMatrix(int num_rows,
                                                      int num_cols,
                                                      Integer max_value) {
  std::vector<std::vector<Integer>> matrix(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    matrix[i] = SampleValues(num_cols, max_value);
  }
  return matrix;
}

// Simple benchmark: There is a single database holding a matrix mod t, and
// the LinPIR server homomorphically computes matrix * vector (mod t).
void BM_SingleDatabase(benchmark::State& state) {
  int num_rows = absl::GetFlag(FLAGS_num_rows);
  int num_cols = absl::GetFlag(FLAGS_num_cols);
  // Setup the RLWE contexts for client and server computation.
  auto rns_context = RnsContext::CreateForBfvFiniteFieldEncoding(
                         kRlweParameters.log_n, kRlweParameters.qs, /*ps=*/{},
                         kRlweParameters.ts[0])
                         .value();
  auto moduli = rns_context.MainPrimeModuli();

  // Create a PRNG seed for encrypting query vector and another for Galois key.
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_ct_pad, Prng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_gk_pad, Prng::GenerateSeed());

  // Create the database and the server.
  auto data = SampleMatrix(num_rows, num_cols, 8);
  ASSERT_OK_AND_ASSIGN(auto database, Database<Integer>::Create(
                                          kRlweParameters, &rns_context, data));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      Server<Integer>::Create(kRlweParameters, &rns_context, {database.get()},
                              prng_seed_ct_pad, prng_seed_gk_pad));
  ASSERT_OK(server->Preprocess());

  // Create a client.
  ASSERT_OK_AND_ASSIGN(
      auto client, Client<Integer>::Create(kRlweParameters, &rns_context,
                                           prng_seed_ct_pad, prng_seed_gk_pad));

  // Random query vector.
  std::vector<Integer> query = SampleValues(num_cols, 8);
  ASSERT_OK_AND_ASSIGN(auto request, client->GenerateRequest(query));

  for (auto _ : state) {
    auto response = server->HandleRequest(request);
    benchmark::DoNotOptimize(response);
  }
}
BENCHMARK(BM_SingleDatabase);

// There are two instances of LinPIR protocols, where the two servers share
// the same Galois key for generating rotations of encrypted query vector.
// The two clients share the same RLWE secret key, but they use different seeds
// for encrypting query vectors.
void BM_TwoServersWithSharedGaloisKey(benchmark::State& state) {
  int num_rows = absl::GetFlag(FLAGS_num_rows);
  int num_cols = absl::GetFlag(FLAGS_num_cols);

  // Setup the RLWE contexts for client and server computation.
  auto rns_context0 = RnsContext::CreateForBfvFiniteFieldEncoding(
                          kRlweParameters.log_n, kRlweParameters.qs, /*ps=*/{},
                          kRlweParameters.ts[0])
                          .value();
  auto rns_context1 = RnsContext::CreateForBfvFiniteFieldEncoding(
                          kRlweParameters.log_n, kRlweParameters.qs,
                          /*ps=*/{}, kRlweParameters.ts[1])
                          .value();
  auto moduli = rns_context0.MainPrimeModuli();

  // We create two seeds for ciphertexts and one seed for the Galois key.
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_ct_pad0, Prng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_ct_pad1, Prng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_gk_pad, Prng::GenerateSeed());

  // First pair of database and server.
  auto matrix = SampleMatrix(num_rows, num_cols, 8);
  ASSERT_OK_AND_ASSIGN(
      auto database0,
      Database<Integer>::Create(kRlweParameters, &rns_context0, matrix));
  ASSERT_OK_AND_ASSIGN(
      auto server0,
      Server<Integer>::Create(kRlweParameters, &rns_context0, {database0.get()},
                              prng_seed_ct_pad0, prng_seed_gk_pad));
  ASSERT_OK(server0->Preprocess());

  // Second pair of database and server, with a different plaintext modulus,
  // but the same PRNG seed for the Galois key.
  ASSERT_OK_AND_ASSIGN(
      auto database1,
      Database<Integer>::Create(kRlweParameters, &rns_context1, matrix));
  ASSERT_OK_AND_ASSIGN(
      auto server1,
      Server<Integer>::Create(kRlweParameters, &rns_context1, {database1.get()},
                              prng_seed_ct_pad1, prng_seed_gk_pad));
  ASSERT_OK(server1->Preprocess());

  // Create two clients
  ASSERT_OK_AND_ASSIGN(auto client0, Client<Integer>::Create(
                                         kRlweParameters, &rns_context0,
                                         prng_seed_ct_pad0, prng_seed_gk_pad));
  ASSERT_OK_AND_ASSIGN(auto client1, Client<Integer>::Create(
                                         kRlweParameters, &rns_context1,
                                         prng_seed_ct_pad1, prng_seed_gk_pad));

  // Random query vector.
  std::vector<Integer> query = SampleValues(num_cols, 8);

  // The two clients share the same secret key. Since they use independent seeds
  // for sampling random "a" components and error terms, the generated requests
  // are still pseudorandom.
  ASSERT_OK_AND_ASSIGN(std::string prng_seed_sk, Prng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto ct_query0,
                       client0->EncryptQuery(query, prng_seed_sk));
  ASSERT_OK_AND_ASSIGN(auto ct_query1,
                       client1->EncryptQuery(query, prng_seed_sk));
  ASSERT_OK_AND_ASSIGN(auto gk, client0->GenerateGaloisKey(prng_seed_sk));
  ASSERT_OK_AND_ASSIGN(auto request0, client0->GenerateRequest(ct_query0, gk));
  ASSERT_OK_AND_ASSIGN(auto request1, client1->GenerateRequest(ct_query1, gk));

  for (auto _ : state) {
    auto response0 = server0->HandleRequest(request0);
    benchmark::DoNotOptimize(response0);
    auto response1 = server1->HandleRequest(request1);
    benchmark::DoNotOptimize(response1);
  }
}
BENCHMARK(BM_TwoServersWithSharedGaloisKey);

}  // namespace
}  // namespace linpir
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
