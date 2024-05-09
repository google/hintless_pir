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

#include "hintless_simplepir/server.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "hintless_simplepir/database_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/serialization.pb.h"
#include "hintless_simplepir/utils.h"
#include "lwe/lwe_symmetric_encryption.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace hintless_simplepir {

namespace {

// Returns an error if `params` uses an invalid PRNG type.
inline absl::Status CheckForValidPrngType(const Parameters& params) {
  if (!(params.prng_type == rlwe::PRNG_TYPE_HKDF ||
        params.prng_type == rlwe::PRNG_TYPE_CHACHA)) {
    return absl::InvalidArgumentError("Invalid PRNG type in `params`.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<Server>> Server::Create(
    const Parameters& params) {
  RLWE_RETURN_IF_ERROR(CheckForValidPrngType(params));

  // Create RLWE contexts, one per plaintext modulus in `ts`.
  auto const& rlwe_params = params.linpir_params;
  int num_linpir_instances = rlwe_params.ts.size();
  std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts;
  rlwe_contexts.reserve(num_linpir_instances);
  for (int i = 0; i < num_linpir_instances; ++i) {
    RLWE_ASSIGN_OR_RETURN(
        auto rlwe_context,
        RlweRnsContext::CreateForBfvFiniteFieldEncoding(
            rlwe_params.log_n, rlwe_params.qs, /*ps=*/{}, rlwe_params.ts[i]));
    rlwe_contexts.push_back(
        std::make_unique<const RlweRnsContext>(std::move(rlwe_context)));
  }

  // Create a Database object holding the database and hint matrices.
  RLWE_ASSIGN_OR_RETURN(auto database, Database::Create(params));

  return absl::WrapUnique(
      new Server(params, std::move(database), std::move(rlwe_contexts)));
}

absl::StatusOr<std::unique_ptr<Server>> Server::CreateWithRandomDatabaseRecords(
    const Parameters& params) {
  RLWE_RETURN_IF_ERROR(CheckForValidPrngType(params));

  // Create RLWE contexts, one per plaintext modulus in `ts`.
  auto const& rlwe_params = params.linpir_params;
  int num_linpir_instances = rlwe_params.ts.size();
  std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts;
  rlwe_contexts.reserve(num_linpir_instances);
  for (int i = 0; i < num_linpir_instances; ++i) {
    RLWE_ASSIGN_OR_RETURN(
        auto rlwe_context,
        RlweRnsContext::CreateForBfvFiniteFieldEncoding(
            rlwe_params.log_n, rlwe_params.qs, /*ps=*/{}, rlwe_params.ts[i]));
    rlwe_contexts.push_back(
        std::make_unique<const RlweRnsContext>(std::move(rlwe_context)));
  }

  // Create a Databas holding random records.
  RLWE_ASSIGN_OR_RETURN(auto database, Database::CreateRandom(params));

  return absl::WrapUnique(
      new Server(params, std::move(database), std::move(rlwe_contexts)));
}

absl::Status Server::GeneratePublicParams() {
  int num_linpir_instances = params_.linpir_params.ts.size();
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    // Sample PRNG seeds for LWE "A" matrix and LinPIR.
    RLWE_ASSIGN_OR_RETURN(prng_seed_lwe_query_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    prng_seed_linpir_ct_pads_.clear();
    prng_seed_linpir_ct_pads_.resize(num_linpir_instances);
    for (int i = 0; i < num_linpir_instances; ++i) {
      RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_ct_pads_[i],
                            rlwe::SingleThreadHkdfPrng::GenerateSeed());
    }
    RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_gk_pad_,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    // Generate the LWE "A" matrix.
    RLWE_ASSIGN_OR_RETURN(auto prng, rlwe::SingleThreadHkdfPrng::Create(
                                         prng_seed_lwe_query_pad_));
    RLWE_ASSIGN_OR_RETURN(
        auto pad,
        lwe::ExpandPad(params_.db_cols, params_.lwe_secret_dim, prng.get()));
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
  } else {
    RLWE_ASSIGN_OR_RETURN(prng_seed_lwe_query_pad_,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
    prng_seed_linpir_ct_pads_.clear();
    prng_seed_linpir_ct_pads_.resize(num_linpir_instances);
    for (int i = 0; i < num_linpir_instances; ++i) {
      RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_ct_pads_[i],
                            rlwe::SingleThreadChaChaPrng::GenerateSeed());
    }
    RLWE_ASSIGN_OR_RETURN(prng_seed_linpir_gk_pad_,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());

    RLWE_ASSIGN_OR_RETURN(auto prng, rlwe::SingleThreadChaChaPrng::Create(
                                         prng_seed_lwe_query_pad_));
    RLWE_ASSIGN_OR_RETURN(
        auto pad,
        lwe::ExpandPad(params_.db_cols, params_.lwe_secret_dim, prng.get()));
    lwe_query_pad_ = std::make_unique<const lwe::Matrix>(std::move(pad));
  }
  return absl::OkStatus();
}

namespace {

// Given `matrix` with mod-q entries, returns `matrix` mod p, where modular
// numbers are in balanced representation.
template <typename Integer>
std::vector<std::vector<Integer>> EncodeLweMatrix(
    const Database::LweMatrix& matrix, Integer q, Integer p) {
  Integer q_half = q >> 1;
  int num_rows = matrix.size();
  int num_cols = matrix[0].size();
  std::vector<std::vector<Integer>> matrix_mod_p(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    matrix_mod_p[i].reserve(num_cols);
    for (int j = 0; j < num_cols; ++j) {
      Integer x = static_cast<Integer>(matrix[i][j]);
      matrix_mod_p[i].push_back(ConvertModulus(x, q, p, q_half));
    }
  }
  return matrix_mod_p;
}

}  // namespace

absl::Status Server::Preprocess() {
  // Refresh the PRNG seeds.
  RLWE_RETURN_IF_ERROR(GeneratePublicParams());

  // Make sure the hint is up to date.
  RLWE_RETURN_IF_ERROR(database_->UpdateLweQueryPad(lwe_query_pad_.get()));
  RLWE_RETURN_IF_ERROR(database_->UpdateHints());

  RlweInteger lwe_modulus = RlweInteger{1} << params_.lwe_modulus_bit_size;
  size_t num_shards = database_->NumShards();

  // Create LinPir databases (holding the preprocessed hints) and servers.
  linpir_servers_.clear();
  linpir_servers_.resize(rlwe_contexts_.size());
  linpir_databases_.clear();
  linpir_databases_.resize(rlwe_contexts_.size());
  for (int k = 0; k < rlwe_contexts_.size(); ++k) {
    RlweInteger plaintext_modulus = rlwe_contexts_[k]->PlaintextModulus();

    // One LinPir database per shard, for the current plaintext modulus.
    std::vector<std::unique_ptr<LinPirDatabase>> linpir_databases_mod_tk;
    linpir_databases_mod_tk.reserve(num_shards);
    for (const Database::LweMatrix& hint : database_->Hints()) {
      std::vector<std::vector<RlweInteger>> hint_mod_tk =
          EncodeLweMatrix(hint, lwe_modulus, plaintext_modulus);
      RLWE_ASSIGN_OR_RETURN(
          auto linpir_database,
          LinPirDatabase::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                                 hint_mod_tk));
      linpir_databases_mod_tk.push_back(std::move(linpir_database));
    }
    std::vector<LinPirDatabase*> linpir_databases_ptrs;
    std::transform(linpir_databases_mod_tk.begin(),
                   linpir_databases_mod_tk.end(),
                   std::back_inserter(linpir_databases_ptrs),
                   [](auto& ptr) { return ptr.get(); });
    RLWE_ASSIGN_OR_RETURN(
        auto linpir_server_mod_tk,
        LinPirServer::Create(params_.linpir_params, rlwe_contexts_[k].get(),
                             linpir_databases_ptrs,
                             prng_seed_linpir_ct_pads_[k],
                             prng_seed_linpir_gk_pad_));
    RLWE_RETURN_IF_ERROR(linpir_server_mod_tk->Preprocess());

    linpir_databases_[k] = std::move(linpir_databases_mod_tk);
    linpir_servers_[k] = std::move(linpir_server_mod_tk);
  }

  return absl::OkStatus();
}

absl::StatusOr<HintlessPirResponse> Server::HandleRequest(
    const HintlessPirRequest& request) {
  if (!IsPreprocessed()) {
    return absl::FailedPreconditionError("Server has not been preprocessed.");
  }

  HintlessPirResponse response;
  // Handle the LWE part of the request.
  Database::LweVector ct_query_vector =
      DeserializeLweCiphertext(request.ct_query_vector());
  RLWE_ASSIGN_OR_RETURN(std::vector<Database::LweVector> ct_records,
                        database_->InnerProductWith(ct_query_vector));
  for (auto& ct_record : ct_records) {
    *response.add_ct_records() = SerializeLweCiphertext(ct_record);
  }

  // Handle the LinPIR requests.
  int num_linpir_requests = request.linpir_ct_bs_size();
  if (num_linpir_requests != linpir_servers_.size()) {
    return absl::InvalidArgumentError(
        "`request` contains unexpected number of LinPir requests.");
  }
  for (int k = 0; k < num_linpir_requests; ++k) {
    RLWE_ASSIGN_OR_RETURN(LinPirResponse linpir_response,
                          linpir_servers_[k]->HandleRequest(
                              request.linpir_ct_bs(k), request.linpir_gk_bs()));
    *response.add_linpir_responses() = std::move(linpir_response);
  }

  return response;
}

HintlessPirServerPublicParams Server::GetPublicParams() const {
  HintlessPirServerPublicParams output;
  output.set_prng_seed_lwe_query_pad(prng_seed_lwe_query_pad_);
  for (auto const& prng_seed : prng_seed_linpir_ct_pads_) {
    *output.add_prng_seed_linpir_ct_pads() = prng_seed;
  }
  output.set_prng_seed_linpir_gk_pad(prng_seed_linpir_gk_pad_);
  return output;
}

}  // namespace hintless_simplepir
}  // namespace hintless_pir
