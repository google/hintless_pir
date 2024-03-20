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

#include "linpir/database.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "linpir/parameters.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/rns/rns_bfv_ciphertext.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace linpir {

namespace {

inline int DivAndRoundUp(int x, int y) { return (x + y - 1) / y; }

}  // namespace

template <typename RlweInteger>
absl::StatusOr<std::unique_ptr<Database<RlweInteger>>>
Database<RlweInteger>::Create(
    const RlweParameters<RlweInteger>& rlwe_params,
    const RnsContext* rns_context,
    const std::vector<std::vector<RlweInteger>>& data) {
  if (rns_context == nullptr) {
    return absl::InvalidArgumentError("`rns_context` must not be null.");
  }
  if (data.empty()) {
    return absl::InvalidArgumentError("`data` must not be empty.");
  }

  std::vector<const PrimeModulus*> moduli = rns_context->MainPrimeModuli();
  RLWE_ASSIGN_OR_RETURN(Encoder encoder, Encoder::Create(rns_context));

  int num_rows = data.size();
  int num_cols = data[0].size();
  int num_slots_per_group = 1 << (rlwe_params.log_n - 1);
  if (num_cols > num_slots_per_group) {
    return absl::InvalidArgumentError(
        "`data` has more columns than supported by RLWE parameters.");
  }

  int num_slots = num_slots_per_group * 2;
  int num_polynomials_per_block = rlwe_params.rows_per_block / 2;
  int num_blocks = DivAndRoundUp(num_rows, rlwe_params.rows_per_block);
  std::vector<std::vector<RnsPolynomial>> diagonals(num_blocks);
  for (int i = 0; i < num_blocks; ++i) {
    diagonals[i].reserve(num_polynomials_per_block);

    // Each block is a rectangle matrix divided into square submatrices of
    // dimension rows_per_block * rows_per_block, and there are rows_per_block
    // many diagonals. Since we assume data has number of columns < number of
    // slots per group, we pack diagonals 0..(rows_per_block/2 - 1) in the first
    // slot group, and rows_per_block/2..rows_per_block in the second group.
    //
    // *--@--*--@-. <- first group starts with the diagonal *, and the second
    // -*--@--*--@.    group starts with the diagonal @, where . means empty
    // --*--@--*--.    positions when extending the block into multiple square
    // @--*--@--*-.    matrices.
    // -@--*--@--*.
    int row_idx_begin = i * rlwe_params.rows_per_block;
    // The j'th and rows_per_block/2 + j'th diagonals.
    for (int j = 0; j < num_polynomials_per_block; ++j) {
      std::vector<RlweInteger> diag_values(num_slots, 0);
      // first group of slots
      for (int k = 0; k < num_slots_per_group; ++k) {
        int row_idx = row_idx_begin + (k % rlwe_params.rows_per_block);
        int col_idx = (k + j) % num_slots_per_group;
        if (row_idx < num_rows && col_idx < num_cols) {  // valid indices
          diag_values[k] = data[row_idx][col_idx];
        }
      }
      // second group of slots
      for (int k = 0; k < num_slots_per_group; ++k) {
        int row_idx = row_idx_begin + (k % rlwe_params.rows_per_block);
        int col_idx =
            (rlwe_params.rows_per_block / 2 + k + j) % num_slots_per_group;
        if (row_idx < num_rows && col_idx < num_cols) {  // valid indices
          diag_values[num_slots_per_group + k] = data[row_idx][col_idx];
        }
      }
      RLWE_ASSIGN_OR_RETURN(
          RnsPolynomial diagonal,
          encoder.EncodeBfv(diag_values, moduli, /*is_scaled=*/false));
      diagonals[i].push_back(std::move(diagonal));
    }
  }
  return absl::WrapUnique(
      new Database<RlweInteger>(rns_context, std::move(moduli),
                                std::move(encoder), std::move(diagonals)));
}

template <typename RlweInteger>
absl::StatusOr<
    std::vector<rlwe::RnsBfvCiphertext<rlwe::MontgomeryInt<RlweInteger>>>>
Database<RlweInteger>::InnerProductWith(
    absl::Span<const RnsCiphertext> ct_rotated_queries) const {
  if (ct_rotated_queries.size() != diagonals_[0].size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries` does not contain correct number of ciphertexts.");
  }

  std::vector<RnsCiphertext> ct_inner_products;
  ct_inner_products.reserve(diagonals_.size());
  for (int i = 0; i < diagonals_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_inner_product,
                          ct_rotated_queries[0].AbsorbSimple(diagonals_[i][0]));
    for (int j = 1; j < ct_rotated_queries.size(); ++j) {
      RLWE_RETURN_IF_ERROR(ct_inner_product.FusedAbsorbAddInPlace(
          ct_rotated_queries[j], diagonals_[i][j]));
    }
    ct_inner_products.push_back(std::move(ct_inner_product));
  }
  return ct_inner_products;
}

template <typename RlweInteger>
absl::Status Database<RlweInteger>::Preprocess(
    absl::Span<const RnsPolynomial> pad_rotated_queries) {
  if (pad_rotated_queries.size() != diagonals_[0].size()) {
    return absl::InvalidArgumentError(
        "`pad_rotated_queries` does not contain correct number of "
        "polynomials.");
  }

  pad_inner_products_.clear();
  pad_inner_products_.reserve(diagonals_.size());
  for (int i = 0; i < diagonals_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(
        RnsPolynomial pad_inner_product,
        pad_rotated_queries[0].Mul(diagonals_[i][0], moduli_));
    for (int j = 1; j < pad_rotated_queries.size(); ++j) {
      RLWE_RETURN_IF_ERROR(pad_inner_product.FusedMulAddInPlace(
          pad_rotated_queries[j], diagonals_[i][j], moduli_));
    }
    pad_inner_products_.push_back(std::move(pad_inner_product));
  }
  return absl::OkStatus();
}

template <typename RlweInteger>
absl::StatusOr<
    std::vector<rlwe::RnsBfvCiphertext<rlwe::MontgomeryInt<RlweInteger>>>>
Database<RlweInteger>::InnerProductWithPreprocessedPads(
    absl::Span<const RnsCiphertext> ct_rotated_queries) const {
  if (pad_inner_products_.size() != diagonals_.size()) {
    return absl::FailedPreconditionError("There is no preprocessed data.");
  }
  if (ct_rotated_queries.size() != diagonals_[0].size()) {
    return absl::InvalidArgumentError(
        "`ct_rotated_queries` does not contain correct number of ciphertexts.");
  }

  std::vector<RnsCiphertext> ct_inner_products;
  ct_inner_products.reserve(diagonals_.size());
  for (int i = 0; i < diagonals_.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(RnsCiphertext ct_inner_product,
                          ct_rotated_queries[0].AbsorbSimple(diagonals_[i][0]));
    for (int j = 1; j < ct_rotated_queries.size(); ++j) {
      RLWE_RETURN_IF_ERROR(ct_inner_product.FusedAbsorbAddInPlaceWithoutPad(
          ct_rotated_queries[j], diagonals_[i][j]));
    }
    RLWE_RETURN_IF_ERROR(
        ct_inner_product.SetPadComponent(pad_inner_products_[i]));
    ct_inner_products.push_back(std::move(ct_inner_product));
  }

  return ct_inner_products;
}

template class Database<Uint32>;
template class Database<Uint64>;

}  // namespace linpir
}  // namespace hintless_pir
