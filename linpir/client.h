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

#ifndef HINTLESS_PIR_LINPIR_CLIENT_H_
#define HINTLESS_PIR_LINPIR_CLIENT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "linpir/parameters.h"
#include "linpir/serialization.pb.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_error_params.h"
#include "shell_encryption/rns/rns_gadget.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"
#include "shell_encryption/rns/rns_secret_key.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace linpir {

// This class implements the client component of the LinPIR scheme, which takes
// a plaintext vector and outsources the database * vector to the LinPIR server.
template <typename RlweInteger>
class Client {
 public:
  using ModularInt = rlwe::MontgomeryInt<RlweInteger>;
  using RnsContext = rlwe::RnsContext<ModularInt>;
  using RnsGadget = rlwe::RnsGadget<ModularInt>;
  using RnsGaloisKey = rlwe::RnsGaloisKey<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using RnsCiphertext = rlwe::RnsBfvCiphertext<ModularInt>;
  using RnsSecretKey = rlwe::RnsRlweSecretKey<ModularInt>;
  using RnsErrorParams = rlwe::RnsErrorParams<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;

  // Creates a Client given the PRNG seeds for sampling random polynomials
  // in the query ciphertext and the Galois key.
  static absl::StatusOr<std::unique_ptr<Client>> Create(
      const RlweParameters<RlweInteger>& parameters,
      const RnsContext* rns_context, absl::string_view prng_seed_ct_pad,
      absl::string_view prng_seed_gk_pad);

  // Samples a fresh RLWE secret key which is cached in `secret_key_`, and
  // returns a ciphertext encrypting the vector under `secret_key_`.
  absl::StatusOr<RnsCiphertext> EncryptQuery(
      absl::Span<const RlweInteger> query_vector);

  // This variant samples a RLWE secret key using the given PRNG seed.
  absl::StatusOr<RnsCiphertext> EncryptQuery(
      absl::Span<const RlweInteger> query_vector,
      absl::string_view prng_seed_sk);

  // Returns a Galois key based on the secret key that is sampled using the
  // given PRNG seed.
  absl::StatusOr<RnsGaloisKey> GenerateGaloisKey(
      absl::string_view prng_seed_sk) const;

  // Returns a Galois key based on the cached `secret_key_`.
  absl::StatusOr<RnsGaloisKey> GenerateGaloisKey() const;

  // Returns a LinPIR request including the given ciphertext and Galois key.
  absl::StatusOr<LinPirRequest> GenerateRequest(const RnsCiphertext& ct_query,
                                                const RnsGaloisKey& gk) const {
    LinPirRequest request;
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial ct_query_b, ct_query.Component(0));
    RLWE_ASSIGN_OR_RETURN(*request.mutable_ct_query_b(),
                          ct_query_b.Serialize(rns_moduli_));
    for (auto const& gk_key_b : gk.GetKeyB()) {
      RLWE_ASSIGN_OR_RETURN(*request.add_gk_key_bs(),
                            gk_key_b.Serialize(rns_moduli_));
    }
    return request;
  }

  // Returns a LinPIR request including ciphertext that encrypts `query_vector`
  // under a fresh RLWE secret key and a corresponding Galois key.
  absl::StatusOr<LinPirRequest> GenerateRequest(
      absl::Span<const RlweInteger> query_vector) {
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_query, EncryptQuery(query_vector));
    RLWE_ASSIGN_OR_RETURN(RnsGaloisKey gk, GenerateGaloisKey());
    return GenerateRequest(ct_query, gk);
  }

  // Recovers the inner products from `response`, one per database matrix.
  absl::StatusOr<std::vector<std::vector<RlweInteger>>> Recover(
      const LinPirResponse& response);

  absl::string_view PrngSeedForCiphertextRandomPads() const {
    return prng_seed_ct_pad_;
  }
  absl::string_view PrngSeedForGaloisKeyRandomPads() const {
    return prng_seed_gk_pad_;
  }

 private:
  explicit Client(RlweParameters<RlweInteger> params,
                  std::string prng_seed_ct_pad, std::string prng_seed_gk_pad,
                  const RnsContext* rns_context,
                  std::vector<const PrimeModulus*> rns_moduli,
                  RnsGadget rns_gadget, RnsErrorParams rns_error_params,
                  Encoder encoder)
      : params_(std::move(params)),
        prng_seed_ct_pad_(std::move(prng_seed_ct_pad)),
        prng_seed_gk_pad_(std::move(prng_seed_gk_pad)),
        rns_context_(rns_context),
        rns_moduli_(std::move(rns_moduli)),
        rns_gadget_(std::move(rns_gadget)),
        rns_error_params_(std::move(rns_error_params)),
        encoder_(std::move(encoder)) {}

  const RlweParameters<RlweInteger> params_;

  // PRNG seeds generated by the server to sample the "a" polynomials.
  std::string prng_seed_ct_pad_;
  std::string prng_seed_gk_pad_;

  const RnsContext* rns_context_;
  const std::vector<const PrimeModulus*> rns_moduli_;
  const RnsGadget rns_gadget_;
  const RnsErrorParams rns_error_params_;

  const Encoder encoder_;

  // The RLWE secret key generated for every request. This is cached until the
  // response is received.
  std::unique_ptr<RnsSecretKey> secret_key_;
};

}  // namespace linpir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LINPIR_CLIENT_H_
