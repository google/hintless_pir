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

#include <memory>
#include <string>

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hintless_simplepir/client.h"
#include "hintless_simplepir/database.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/server.h"
#include "hintless_simplepir/utils.h"
#include "linpir/parameters.h"
#include "shell_encryption/testing/status_testing.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

using RlweInteger = Parameters::RlweInteger;

const Parameters kParameters{
    .db_rows = 8,
    .db_cols = 1024,
    .db_record_bit_size = 16,
    .lwe_secret_dim = 1400,
    .lwe_modulus_bit_size = 32,
    .lwe_plaintext_bit_size = 8,
    .lwe_error_variance = 8,
    .linpir_params =
        linpir::RlweParameters<RlweInteger>{
            .log_n = 12,
            .qs = {35184371884033ULL, 35184371703809ULL},  // 90 bits
            .ts = {2056193, 1990657},                      // 42 bits
            .gadget_log_bs = {16, 16},
            .error_variance = 8,
            .prng_type = rlwe::PRNG_TYPE_HKDF,
            .rows_per_block = 1024,
        },
    .prng_type = rlwe::PRNG_TYPE_HKDF,
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

TEST(HintlessSimplePir, EndToEndTest) {
  // Create server and fill in random database records.
  ASSERT_OK_AND_ASSIGN(auto server, Server::Create(kParameters));

  Database* database = server->GetDatabase();
  for (int i = 0; i < kParameters.db_rows * kParameters.db_cols; ++i) {
    ASSERT_OK(database->Append(GenerateRandomRecord(kParameters)));
  }

  // Preprocess the server and get public parameters.
  ASSERT_OK(server->Preprocess());
  auto public_params = server->GetPublicParams();

  // Create a client and issue request.
  ASSERT_OK_AND_ASSIGN(auto client, Client::Create(kParameters, public_params));
  ASSERT_OK_AND_ASSIGN(auto request, client->GenerateRequest(1));

  // Handle the request
  ASSERT_OK_AND_ASSIGN(auto response, server->HandleRequest(request));
  ASSERT_OK_AND_ASSIGN(auto record, client->RecoverRecord(response));

  ASSERT_OK_AND_ASSIGN(auto expected, database->Record(1));
  EXPECT_EQ(record, expected);
}

TEST(HintlessSimplePir, EndToEndTestWithChaChaPrng) {
  // Use ChaCha PRNG in both LinPIR and SimplePIR sub-protocols.
  Parameters params = kParameters;
  params.linpir_params.prng_type = rlwe::PRNG_TYPE_CHACHA;
  params.prng_type = rlwe::PRNG_TYPE_CHACHA;

  // Create server and fill in random database records.
  ASSERT_OK_AND_ASSIGN(auto server, Server::Create(params));

  Database* database = server->GetDatabase();
  for (int i = 0; i < params.db_rows * params.db_cols; ++i) {
    ASSERT_OK(database->Append(GenerateRandomRecord(params)));
  }

  // Preprocess the server and get public parameters.
  ASSERT_OK(server->Preprocess());
  auto public_params = server->GetPublicParams();

  // Create a client and issue request.
  ASSERT_OK_AND_ASSIGN(auto client, Client::Create(params, public_params));
  ASSERT_OK_AND_ASSIGN(auto request, client->GenerateRequest(1));

  // Handle the request
  ASSERT_OK_AND_ASSIGN(auto response, server->HandleRequest(request));
  ASSERT_OK_AND_ASSIGN(auto record, client->RecoverRecord(response));

  ASSERT_OK_AND_ASSIGN(auto expected, database->Record(1));
  EXPECT_EQ(record, expected);
}

}  // namespace
}  // namespace hintless_simplepir
}  // namespace hintless_pir