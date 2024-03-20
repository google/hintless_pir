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

#include "linpir/server.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "linpir/database.h"
#include "linpir/parameters.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace linpir {

template <typename RlweInteger>
absl::StatusOr<std::unique_ptr<Server<RlweInteger>>>
Server<RlweInteger>::Create(
    const RlweParameters<RlweInteger>& parameters,
    const RnsContext* rns_context,
    const std::vector<Database<RlweInteger>*>& databases,
    absl::string_view prng_seed_ct_pad, absl::string_view prng_seed_gk_pad) {
  if (!(parameters.prng_type == rlwe::PRNG_TYPE_HKDF ||
        parameters.prng_type == rlwe::PRNG_TYPE_CHACHA)) {
    return absl::InvalidArgumentError("Invalid `prng_type`.");
  }
  if (rns_context == nullptr) {
    return absl::InvalidArgumentError("`rns_context` must not be null.");
  }

  auto rns_moduli = rns_context->MainPrimeModuli();
  int level = rns_moduli.size() - 1;
  RLWE_ASSIGN_OR_RETURN(auto q_hats,
                        rns_context->MainPrimeModulusComplements(level));
  RLWE_ASSIGN_OR_RETURN(auto q_hat_invs,
                        rns_context->MainPrimeModulusCrtFactors(level));
  RLWE_ASSIGN_OR_RETURN(
      RnsGadget rns_gadget,
      RnsGadget::Create(parameters.log_n, parameters.gadget_log_bs, q_hats,
                        q_hat_invs, rns_moduli));
  RLWE_ASSIGN_OR_RETURN(
      auto rns_error_params,
      RnsErrorParams::Create(
          parameters.log_n, rns_moduli, /*aux_moduli=*/{},
          std::log2(static_cast<double>(rns_context->PlaintextModulus())),
          std::sqrt(parameters.error_variance)));

  return absl::WrapUnique(new Server<RlweInteger>(
      parameters, std::string(prng_seed_ct_pad), std::string(prng_seed_gk_pad),
      rns_context, std::move(rns_moduli), std::move(rns_gadget),
      std::move(rns_error_params), databases));
}

template <typename RlweInteger>
absl::StatusOr<std::unique_ptr<Server<RlweInteger>>>
Server<RlweInteger>::Create(
    const RlweParameters<RlweInteger>& parameters,
    const RnsContext* rns_context,
    const std::vector<Database<RlweInteger>*>& databases) {
  // Sample PRNG seeds for the query vector and the Galois key.
  std::string prng_seed_ct_pad, prng_seed_gk_pad;
  if (parameters.prng_type == rlwe::PRNG_TYPE_HKDF) {
    RLWE_ASSIGN_OR_RETURN(prng_seed_ct_pad,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    RLWE_ASSIGN_OR_RETURN(prng_seed_gk_pad,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
  } else if (parameters.prng_type == rlwe::PRNG_TYPE_CHACHA) {
    RLWE_ASSIGN_OR_RETURN(prng_seed_ct_pad,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
    RLWE_ASSIGN_OR_RETURN(prng_seed_gk_pad,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
  } else {
    return absl::InvalidArgumentError("Invalid `prng_type`.");
  }

  return Server<RlweInteger>::Create(parameters, rns_context, databases,
                                     prng_seed_ct_pad, prng_seed_gk_pad);
}

template <typename RlweInteger>
absl::Status Server<RlweInteger>::Preprocess() {
  ct_pads_.clear();
  ct_sub_pad_digits_.clear();
  gk_pads_.clear();

  // Create PRNGs.
  std::unique_ptr<rlwe::SecurePrng> prng_ct, prng_gk;
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    RLWE_ASSIGN_OR_RETURN(
        prng_ct, rlwe::SingleThreadHkdfPrng::Create(prng_seed_ct_pad_));
    RLWE_ASSIGN_OR_RETURN(
        prng_gk, rlwe::SingleThreadHkdfPrng::Create(prng_seed_gk_pad_));
  } else {
    RLWE_ASSIGN_OR_RETURN(
        prng_ct, rlwe::SingleThreadChaChaPrng::Create(prng_seed_ct_pad_));
    RLWE_ASSIGN_OR_RETURN(
        prng_gk, rlwe::SingleThreadChaChaPrng::Create(prng_seed_gk_pad_));
  }

  // Expand seed to the "a" part of Enc(query vector)
  int log_n = rns_context_->LogN();
  int gadget_dim = rns_gadget_.Dimension();
  RLWE_ASSIGN_OR_RETURN(auto ct_pad, RnsPolynomial::SampleUniform(
                                         log_n, prng_ct.get(), rns_moduli_));
  RLWE_RETURN_IF_ERROR(ct_pad.NegateInPlace(rns_moduli_));

  // Create the "a" part of Galois key
  RLWE_ASSIGN_OR_RETURN(gk_pads_, RnsGaloisKey::SampleRandomPad(
                                      gadget_dim, log_n, rns_moduli_,
                                      prng_seed_gk_pad_, params_.prng_type));

  // Precompute the "a" part of Enc(s << i) and the digits used to generate
  // Enc(s << i).
  int num_rotations = params_.rows_per_block / 2;
  ct_pads_.reserve(num_rotations);
  ct_pads_.push_back(std::move(ct_pad));
  ct_sub_pad_digits_.reserve(num_rotations);

  int curr_power = 1;
  int cyclotomic_order = 1 << (log_n + 1);
  for (int i = 1; i < num_rotations; ++i) {
    curr_power = (curr_power * 5) % cyclotomic_order;
    // ct[i-1].a(X^5)
    RLWE_ASSIGN_OR_RETURN(RnsPolynomial prev_sub_a,
                          ct_pads_[i - 1].Substitute(5, rns_moduli_));

    // g^-1(ct[i-1].a(X^5))
    if (prev_sub_a.IsNttForm()) {
      RLWE_RETURN_IF_ERROR(prev_sub_a.ConvertToCoeffForm(rns_moduli_));
    }
    RLWE_ASSIGN_OR_RETURN(auto prev_sub_a_digits,
                          rns_gadget_.Decompose(prev_sub_a, rns_moduli_));
    for (auto& digit : prev_sub_a_digits) {
      RLWE_RETURN_IF_ERROR(digit.ConvertToNttForm(rns_moduli_));
    }

    // g^-1(ct[i-1].a(X^5))^T * gk.a
    RLWE_ASSIGN_OR_RETURN(
        auto curr_a,
        RnsPolynomial::CreateZero(log_n, rns_moduli_, /*is_ntt=*/true));
    for (int i = 0; i < prev_sub_a_digits.size(); ++i) {
      RLWE_RETURN_IF_ERROR(curr_a.FusedMulAddInPlace(prev_sub_a_digits[i],
                                                     gk_pads_[i], rns_moduli_));
    }
    ct_pads_.push_back(std::move(curr_a));
    ct_sub_pad_digits_.push_back(std::move(prev_sub_a_digits));
  }

  // Preprocess the databases using the "a" part of Enc(s << i).
  for (auto const& database : databases_) {
    RLWE_RETURN_IF_ERROR(database->Preprocess(ct_pads_));
  }
  return absl::OkStatus();
}

template <typename RlweInteger>
absl::StatusOr<LinPirResponse> Server<RlweInteger>::HandleRequest(
    const RnsCiphertext& ct_query, const RnsGaloisKey& gk) const {
  // Compute all rotations of the query vector.
  int num_rotations = params_.rows_per_block / 2;
  std::vector<RnsCiphertext> ct_rotated_queries;
  ct_rotated_queries.reserve(num_rotations);
  ct_rotated_queries.push_back(std::move(ct_query));
  for (int i = 1; i < num_rotations; ++i) {
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_sub,
                          ct_rotated_queries[i - 1].Substitute(5));
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_rot, gk.ApplyTo(ct_sub));
    ct_rotated_queries.push_back(std::move(ct_rot));
  }
  // Compute inner products with the databases and serialize.
  LinPirResponse response;
  for (auto const& database : databases_) {
    LinPirResponse::EncryptedInnerProduct inner_product;
    RLWE_ASSIGN_OR_RETURN(std::vector<RnsCiphertext> ct_blocks,
                          database->InnerProductWith(ct_rotated_queries));
    for (auto const& ct : ct_blocks) {
      RLWE_ASSIGN_OR_RETURN(*inner_product.add_ct_blocks(), ct.Serialize());
    }
    *response.add_ct_inner_products() = std::move(inner_product);
  }
  return response;
}

template <typename RlweInteger>
absl::StatusOr<LinPirResponse> Server<RlweInteger>::HandleRequest(
    const ::rlwe::SerializedRnsPolynomial& proto_ct_query_b,
    const google::protobuf::RepeatedPtrField<::rlwe::SerializedRnsPolynomial>&
        proto_gk_key_bs) const {
  // Deserialize the "b" components from request and build the query ciphertext
  // and the Galois key.
  RLWE_ASSIGN_OR_RETURN(
      RnsPolynomial ct_query_b,
      RnsPolynomial::Deserialize(proto_ct_query_b, rns_moduli_));
  RnsCiphertext ct_query({std::move(ct_query_b), ct_pads_[0]}, rns_moduli_,
                         /*power_of_s=*/1, /*error=*/0, &rns_error_params_,
                         rns_context_);

  std::vector<RnsPolynomial> gk_key_bs;
  gk_key_bs.reserve(proto_gk_key_bs.size());
  for (int i = 0; i < proto_gk_key_bs.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial gk_key_b,
        RnsPolynomial::Deserialize(proto_gk_key_bs[i], rns_moduli_));
    gk_key_bs.push_back(std::move(gk_key_b));
  }
  RLWE_ASSIGN_OR_RETURN(
      RnsGaloisKey gk,
      RnsGaloisKey::CreateFromKeyComponents(
          gk_pads_, std::move(gk_key_bs), /*power=*/5, &rns_gadget_,
          rns_moduli_, prng_seed_gk_pad_, params_.prng_type));

  // Compute all rotations of the query vector.
  int num_rotations = params_.rows_per_block / 2;
  std::vector<RnsCiphertext> ct_rotated_queries;
  ct_rotated_queries.reserve(num_rotations);
  ct_rotated_queries.push_back(std::move(ct_query));
  for (int i = 1; i < num_rotations; ++i) {
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_sub,
                          ct_rotated_queries[i - 1].Substitute(5));
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_rot,
                          gk.ApplyToWithRandomPad(
                              ct_sub, ct_sub_pad_digits_[i - 1], ct_pads_[i]));
    ct_rotated_queries.push_back(std::move(ct_rot));
  }

  // Compute inner products with the databases and serialize them.
  LinPirResponse response;
  response.mutable_ct_inner_products()->Reserve(databases_.size());
  for (auto const& database : databases_) {
    RLWE_ASSIGN_OR_RETURN(
        std::vector<RnsCiphertext> ct_blocks,
        database->InnerProductWithPreprocessedPads(ct_rotated_queries));
    LinPirResponse::EncryptedInnerProduct inner_product;
    inner_product.mutable_ct_blocks()->Reserve(ct_blocks.size());
    for (auto const& ct : ct_blocks) {
      RLWE_ASSIGN_OR_RETURN(*inner_product.add_ct_blocks(), ct.Serialize());
    }
    *response.add_ct_inner_products() = std::move(inner_product);
  }
  return response;
}

template class Server<Uint32>;
template class Server<Uint64>;

}  // namespace linpir
}  // namespace hintless_pir
