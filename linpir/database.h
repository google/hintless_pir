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

#ifndef HINTLESS_PIR_LINPIR_DATABASE_H_
#define HINTLESS_PIR_LINPIR_DATABASE_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "linpir/parameters.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/finite_field_encoder.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/rns/rns_context.h"
#include "shell_encryption/rns/rns_modulus.h"
#include "shell_encryption/rns/rns_polynomial.h"

namespace hintless_pir {
namespace linpir {

// The database to the LinPIR scheme is a matrix arranged into blocks of
// diagonals, such that the matrix-vector product with a query vector is
// computed as the inner products between the diagonals and rotations of
// the query vector.
template <typename RlweInteger>
class Database {
 public:
  using ModularInt = rlwe::MontgomeryInt<RlweInteger>;
  using RnsContext = rlwe::RnsContext<ModularInt>;
  using RnsPolynomial = rlwe::RnsPolynomial<ModularInt>;
  using RnsCiphertext = rlwe::RnsBfvCiphertext<ModularInt>;
  using PrimeModulus = rlwe::PrimeModulus<ModularInt>;
  using Encoder = rlwe::FiniteFieldEncoder<ModularInt>;

  static absl::StatusOr<std::unique_ptr<Database>> Create(
      const RlweParameters<RlweInteger>& rlwe_params,
      const RnsContext* rns_context,
      const std::vector<std::vector<RlweInteger>>& data);

  // Preprocess the database with the given random pads to speedup inner product
  // computation when query is available.
  absl::Status Preprocess(absl::Span<const RnsPolynomial> pad_rotated_queries);

  // Compute the matrix-vector product with the encrypted query vector.
  absl::StatusOr<std::vector<RnsCiphertext>> InnerProductWith(
      absl::Span<const RnsCiphertext> ct_rotated_queries) const;

  // Compute the matrix-vector product with the encrypted query vector when the
  // database has been preprocessed.
  // Returns error if `Preprocess` has not been called.
  absl::StatusOr<std::vector<RnsCiphertext>> InnerProductWithPreprocessedPads(
      absl::Span<const RnsCiphertext> ct_rotated_queries) const;

  // Accessors
  int NumBlocks() const { return diagonals_.size(); }
  int NumDiagonalsPerBlock() const { return diagonals_[0].size(); }
  bool IsPreprocessed() const { return !pad_inner_products_.empty(); }

 private:
  explicit Database(const RnsContext* rns_context,
                    std::vector<const PrimeModulus*> moduli, Encoder encoder,
                    std::vector<std::vector<RnsPolynomial>> diagonals)
      : rns_context_(rns_context),
        moduli_(std::move(moduli)),
        encoder_(std::move(encoder)),
        diagonals_(std::move(diagonals)) {}

  const RnsContext* rns_context_;

  const std::vector<const PrimeModulus*> moduli_;

  const Encoder encoder_;

  // Database matrix arranged into blocks of sub-matrices, where each sub-matrix
  // is stored as a vector of diagonals packed in RnsPolynomial.
  std::vector<std::vector<RnsPolynomial>> diagonals_;

  // The random pads, i.e. the "a" parts, of the ciphertexts encrypting the
  // matrix-vector products between the blocks of diagonals and the query vector
  std::vector<RnsPolynomial> pad_inner_products_;
};

}  // namespace linpir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LINPIR_DATABASE_H_
