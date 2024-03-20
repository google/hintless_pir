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

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_SERVER_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_SERVER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "hintless_simplepir/database.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/serialization.pb.h"
#include "linpir/database.h"
#include "linpir/server.h"
#include "lwe/types.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/rns_context.h"

namespace hintless_pir {
namespace hintless_simplepir {

// The server part of the HintlessPir protocol.
class Server {
 public:
  static absl::StatusOr<std::unique_ptr<Server>> Create(
      const Parameters& params);

  // Refreshes the server's public parameters and preprocess the database and
  // LinPir servers. The server's public parameters are used by the clients to
  // generate their requests, accessible via `GetPublicParams()`. This should
  // be called before accepting client requests.
  absl::Status Preprocess();

  absl::StatusOr<HintlessPirResponse> HandleRequest(
      const HintlessPirRequest& request);

  // Returns the server's public parameters that are sent to the client.
  HintlessPirServerPublicParams GetPublicParams() const;

  Database* GetDatabase() const { return database_.get(); }

  const lwe::Matrix* LweQueryPad() const { return lwe_query_pad_.get(); }

 private:
  using RlweInteger = Parameters::RlweInteger;
  using RlweModularInt = rlwe::MontgomeryInt<RlweInteger>;
  using RlweRnsContext = rlwe::RnsContext<RlweModularInt>;
  using LinPirServer = linpir::Server<RlweInteger>;
  using LinPirDatabase = linpir::Database<RlweInteger>;

  explicit Server(
      Parameters params, std::unique_ptr<Database> database,
      std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts)
      : params_(std::move(params)),
        database_(std::move(database)),
        rlwe_contexts_(std::move(rlwe_contexts)) {}

  // Refreshes the server's public parameters.
  // This is part of the preprocess steps.
  absl::Status GeneratePublicParams();

  // Returns if the server has been preprocessed to accept requests.
  bool IsPreprocessed() const { return lwe_query_pad_ != nullptr; }

  // The parameters of the SimplePIR protocol.
  const Parameters params_;

  // Holding the database matrices and the hint matrices.
  std::unique_ptr<Database> database_;

  std::vector<std::unique_ptr<const RlweRnsContext>> rlwe_contexts_;

  std::string prng_seed_lwe_query_pad_;
  std::unique_ptr<const lwe::Matrix> lwe_query_pad_;

  std::vector<std::string> prng_seed_linpir_ct_pads_;
  std::string prng_seed_linpir_gk_pad_;

  std::vector<std::vector<std::unique_ptr<LinPirDatabase>>> linpir_databases_;
  std::vector<std::unique_ptr<LinPirServer>> linpir_servers_;
};

}  // namespace hintless_simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_SERVER_H_
