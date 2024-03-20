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

#include "linpir/client.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "linpir/parameters.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/rns/rns_galois_key.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace linpir {

template <typename RlweInteger>
absl::StatusOr<std::unique_ptr<Client<RlweInteger>>>
Client<RlweInteger>::Create(const RlweParameters<RlweInteger>& parameters,
                            const RnsContext* rns_context,
                            absl::string_view prng_seed_ct_pad,
                            absl::string_view prng_seed_gk_pad) {
  if (!(parameters.prng_type == rlwe::PRNG_TYPE_HKDF ||
        parameters.prng_type == rlwe::PRNG_TYPE_CHACHA)) {
    return absl::InvalidArgumentError("Invalid `prng_type`.");
  }
  if (rns_context == nullptr) {
    return absl::InvalidArgumentError("`rns_context` must not be null.");
  }

  auto rns_moduli = rns_context->MainPrimeModuli();
  RLWE_ASSIGN_OR_RETURN(Encoder encoder, Encoder::Create(rns_context));
  RLWE_ASSIGN_OR_RETURN(
      auto rns_error_params,
      RnsErrorParams::Create(
          parameters.log_n, rns_moduli, {},
          std::log2(static_cast<double>(rns_context->PlaintextModulus())),
          std::sqrt(parameters.error_variance)));
  int level = rns_moduli.size() - 1;
  RLWE_ASSIGN_OR_RETURN(auto q_hats,
                        rns_context->MainPrimeModulusComplements(level));
  RLWE_ASSIGN_OR_RETURN(auto q_hat_invs,
                        rns_context->MainPrimeModulusCrtFactors(level));
  RLWE_ASSIGN_OR_RETURN(
      RnsGadget rns_gadget,
      RnsGadget::Create(parameters.log_n, parameters.gadget_log_bs, q_hats,
                        q_hat_invs, rns_moduli));
  return absl::WrapUnique(new Client<RlweInteger>(
      parameters, std::string(prng_seed_ct_pad), std::string(prng_seed_gk_pad),
      rns_context, std::move(rns_moduli), std::move(rns_gadget),
      std::move(rns_error_params), std::move(encoder)));
}

template <typename RlweInteger>
absl::StatusOr<rlwe::RnsBfvCiphertext<rlwe::MontgomeryInt<RlweInteger>>>
Client<RlweInteger>::EncryptQuery(absl::Span<const RlweInteger> query_vector,
                                  absl::string_view prng_seed_sk) {
  int num_slots_per_group = 1 << (params_.log_n - 1);
  if (query_vector.size() > num_slots_per_group) {
    return absl::InvalidArgumentError(
        absl::StrCat("`query_vector` can contain at most ", num_slots_per_group,
                     " element."));
  }

  // Encode the query vector.
  int num_rotations = params_.rows_per_block / 2;
  std::vector<RlweInteger> slots(num_slots_per_group * 2, 0);
  for (int i = 0; i < num_slots_per_group; ++i) {
    if (i < query_vector.size()) {
      slots[i] = query_vector[i];
    }
    int snd_idx = (num_rotations + i) % num_slots_per_group;
    if (snd_idx < query_vector.size()) {
      slots[num_slots_per_group + i] = query_vector[snd_idx];
    }
  }

  // Create PRNGs for sampling RLWE secret key and encryption.
  std::unique_ptr<rlwe::SecurePrng> prng_sk, prng_enc, prng_pad;
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    RLWE_ASSIGN_OR_RETURN(prng_sk,
                          rlwe::SingleThreadHkdfPrng::Create(prng_seed_sk));
    RLWE_ASSIGN_OR_RETURN(std::string prng_seed_enc,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
    RLWE_ASSIGN_OR_RETURN(prng_enc,
                          rlwe::SingleThreadHkdfPrng::Create(prng_seed_enc));
    RLWE_ASSIGN_OR_RETURN(
        prng_pad, rlwe::SingleThreadHkdfPrng::Create(prng_seed_ct_pad_));
  } else {
    RLWE_ASSIGN_OR_RETURN(prng_sk,
                          rlwe::SingleThreadChaChaPrng::Create(prng_seed_sk));
    RLWE_ASSIGN_OR_RETURN(std::string prng_seed_enc,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
    RLWE_ASSIGN_OR_RETURN(prng_enc,
                          rlwe::SingleThreadChaChaPrng::Create(prng_seed_enc));
    RLWE_ASSIGN_OR_RETURN(
        prng_pad, rlwe::SingleThreadChaChaPrng::Create(prng_seed_ct_pad_));
  }

  // Sample RLWE secret key
  RLWE_ASSIGN_OR_RETURN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(params_.log_n, params_.error_variance, rns_moduli_,
                           prng_sk.get()));

  // Encrypt the query vector
  RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_query,
                        secret_key.template EncryptBfv<Encoder>(
                            slots, &encoder_, &rns_error_params_,
                            prng_enc.get(), prng_pad.get()));

  // Store the secret key
  secret_key_ = std::make_unique<RnsSecretKey>(std::move(secret_key));

  return ct_query;
}

template <typename RlweInteger>
absl::StatusOr<rlwe::RnsBfvCiphertext<rlwe::MontgomeryInt<RlweInteger>>>
Client<RlweInteger>::EncryptQuery(absl::Span<const RlweInteger> query_vector) {
  // Create PRNG for sampling RLWE secret key.
  std::string prng_seed_sk;
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    RLWE_ASSIGN_OR_RETURN(prng_seed_sk,
                          rlwe::SingleThreadHkdfPrng::GenerateSeed());
  } else {
    RLWE_ASSIGN_OR_RETURN(prng_seed_sk,
                          rlwe::SingleThreadChaChaPrng::GenerateSeed());
  }
  return EncryptQuery(query_vector, prng_seed_sk);
}

template <typename RlweInteger>
absl::StatusOr<rlwe::RnsGaloisKey<rlwe::MontgomeryInt<RlweInteger>>>
Client<RlweInteger>::GenerateGaloisKey(absl::string_view prng_seed_sk) const {
  // Sample RLWE secret key
  std::unique_ptr<rlwe::SecurePrng> prng_sk;
  if (params_.prng_type == rlwe::PRNG_TYPE_HKDF) {
    RLWE_ASSIGN_OR_RETURN(prng_sk,
                          rlwe::SingleThreadHkdfPrng::Create(prng_seed_sk));
  } else {
    RLWE_ASSIGN_OR_RETURN(prng_sk,
                          rlwe::SingleThreadChaChaPrng::Create(prng_seed_sk));
  }
  RLWE_ASSIGN_OR_RETURN(
      RnsSecretKey secret_key,
      RnsSecretKey::Sample(params_.log_n, params_.error_variance, rns_moduli_,
                           prng_sk.get()));

  // Create a Galois key with the given random pads.
  RLWE_ASSIGN_OR_RETURN(std::vector<RnsPolynomial> gk_pads,
                        RnsGaloisKey::SampleRandomPad(
                            rns_gadget_.Dimension(), params_.log_n, rns_moduli_,
                            prng_seed_gk_pad_, params_.prng_type));

  RLWE_ASSIGN_OR_RETURN(
      RnsGaloisKey gk,
      RnsGaloisKey::CreateWithRandomPadForBfv(
          std::move(gk_pads), secret_key, /*power=*/5, params_.error_variance,
          &rns_gadget_, prng_seed_gk_pad_, params_.prng_type));

  return gk;
}

template <typename RlweInteger>
absl::StatusOr<rlwe::RnsGaloisKey<rlwe::MontgomeryInt<RlweInteger>>>
Client<RlweInteger>::GenerateGaloisKey() const {
  if (secret_key_ == nullptr) {
    return absl::InvalidArgumentError("Secret key not found.");
  }

  // Create a Galois key from the stored secret key and random pads.
  RLWE_ASSIGN_OR_RETURN(std::vector<RnsPolynomial> gk_pads,
                        RnsGaloisKey::SampleRandomPad(
                            rns_gadget_.Dimension(), params_.log_n, rns_moduli_,
                            prng_seed_gk_pad_, params_.prng_type));
  RLWE_ASSIGN_OR_RETURN(
      RnsGaloisKey gk,
      RnsGaloisKey::CreateWithRandomPadForBfv(
          std::move(gk_pads), *secret_key_, /*power=*/5, params_.error_variance,
          &rns_gadget_, prng_seed_gk_pad_, params_.prng_type));
  return gk;
}

template <typename RlweInteger>
absl::StatusOr<std::vector<std::vector<RlweInteger>>>
Client<RlweInteger>::Recover(const LinPirResponse& response) {
  if (secret_key_ == nullptr) {
    return absl::InvalidArgumentError("Secret key not found.");
  }

  RlweInteger plaintext_modulus = rns_context_->PlaintextModulus();

  std::vector<std::vector<RlweInteger>> results(
      response.ct_inner_products_size());
  for (int i = 0; i < response.ct_inner_products_size(); ++i) {
    const auto& ct_inner_products = response.ct_inner_products(i);
    int num_slots_per_group = 1 << (params_.log_n - 1);
    int num_blocks = ct_inner_products.ct_blocks_size();
    results[i].reserve(num_blocks * params_.rows_per_block);

    for (int j = 0; j < num_blocks; ++j) {
      RLWE_ASSIGN_OR_RETURN(
          auto ct_deserialized,
          RnsCiphertext::Deserialize(ct_inner_products.ct_blocks(j),
                                     rns_moduli_, &rns_error_params_));
      RnsCiphertext ct_block(std::move(ct_deserialized));
      RLWE_ASSIGN_OR_RETURN(
          auto slots,
          secret_key_->template DecryptBfv<Encoder>(ct_block, &encoder_));
      std::vector<RlweInteger> values(params_.rows_per_block, 0);
      // First half of the block
      for (int k = 0; k < num_slots_per_group; ++k) {
        values[k % params_.rows_per_block] += slots[k];
      }
      // Second half of the block
      for (int k = 0; k < num_slots_per_group; ++k) {
        values[k % params_.rows_per_block] += slots[num_slots_per_group + k];
      }
      for (auto const& value : values) {
        results[i].push_back(value % plaintext_modulus);
      }
    }
  }

  // Clear the cached RLWE secret key.
  secret_key_ = nullptr;

  return results;
}

template class Client<Uint32>;
template class Client<Uint64>;

}  // namespace linpir
}  // namespace hintless_pir
