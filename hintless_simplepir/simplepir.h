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

#ifndef HINTLESS_PIR_HINTLESS_SIMPLEPIR_SIMPLEPIR_H_
#define HINTLESS_PIR_HINTLESS_SIMPLEPIR_SIMPLEPIR_H_

#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "lwe/encode.h"
#include "lwe/lwe_symmetric_encryption.h"
#include "lwe/sample_error.h"
#include "lwe/types.h"
#include "shell_encryption/prng/single_thread_hkdf_prng.h"
#include "shell_encryption/status_macros.h"

// A standard implementation of the SimplePIR library, e.g. with the large
// database-dependent hint.
//
// SimplePIR [1] is a private-information retrieval scheme that is both
// * simple to specify (and implement), and
// * rather performant.
//
// The "downside" of SimplePIR is that it requires all clients to download a
// (large) database-dependent hint in a preprocessing step.
// We will later remove this downside using homomorphic computation.
//
// [1]: https://eprint.iacr.org/2022/949

namespace hintless_pir {
namespace simplepir {

// i0 = col_idx
// i1 = row_idx
//
// The clients state, meaning a pair of integers (row_idx, col_idx) such that
// for the client's desired query i in to the entire database (of size rows *
// cols) that
//
//  i = (col_idx * rows) + row_idx.
//
// In other words, when Database is presented as a matrix with r rows and c
// cols, this retrieves the value corresponding to the (col_idx, row_idx) entry.
//
// The "Decryption part" is (implicitly) a 1 x 1 matrix.
// It is kept this way (rather than converted to a scalar) so symmetric LWE
// decryption may be used with no modifications.
struct ClientState {
  int row_idx;
  int col_idx;
  lwe::Vector decryption_part;
};

// The public parameters required by both the Client and Server to execute
// the SimplePIR protocol.
template <typename Prng = rlwe::SingleThreadHkdfPrng>
class Parems {
 public:
  // Create a new set of Parems
  static Parems Create(absl::string_view seed, int lwe_secret_dim,
                       int record_bit_size, int db_rows, int db_cols) {
    return Parems(seed, lwe_secret_dim, record_bit_size, db_rows, db_cols);
  }

  // Preprocesses the database-dependent hint for the client.
  absl::StatusOr<lwe::Matrix> ServerPreprocess(
      const lwe::Matrix& database) const {
    if (database.cols() != DbCols()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of cols in the database, ", database.cols(),
                       " does not match the number in the public parameters, ",
                       DbCols(), "."));
    }
    if (database.rows() != DbRows()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of rows in the database, ", database.rows(),
                       " does not match the number in the public parameters, ",
                       DbRows(), "."));
    }
    RLWE_ASSIGN_OR_RETURN(auto pad_prng, Prng::Create(Seed()));
    RLWE_ASSIGN_OR_RETURN(
        lwe::Matrix pad,
        lwe::SampleUniformMatrix(DbCols(), LweSecretDim(), pad_prng.get()));
    lwe::Matrix hint = database * pad;
    return hint;
  }

  absl::StatusOr<std::pair<ClientState, lwe::Vector>> ClientQuery(
      int query_idx, const lwe::Matrix& hint) const {
    int db_size = DbRows() * DbCols();
    if (query_idx < 0 || query_idx >= db_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("The query index, ", query_idx, " is out of range."));
    }
    if (hint.cols() != LweSecretDim()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of cols in the hint, ", hint.cols(),
                       " does not match the number in the public parameters, ",
                       LweSecretDim(), "."));
    }
    if (hint.rows() != DbRows()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The number of rows in the hint, ", hint.rows(),
                       " does not match the number in the public parameters, ",
                       DbRows(), "."));
    }

    RLWE_ASSIGN_OR_RETURN(auto pad_prng, Prng::Create(Seed()));
    RLWE_ASSIGN_OR_RETURN(
        lwe::Matrix pad,
        lwe::SampleUniformMatrix(DbCols(), LweSecretDim(), pad_prng.get()));
    int col_idx = query_idx / DbRows();
    int row_idx = query_idx % DbRows();
    RLWE_ASSIGN_OR_RETURN(std::string client_seed, Prng::GenerateSeed());
    RLWE_ASSIGN_OR_RETURN(auto client_prng, Prng::Create(client_seed));
    RLWE_ASSIGN_OR_RETURN(
        lwe::SymmetricLweKey key,
        lwe::SymmetricLweKey::Sample(LweSecretDim(), client_prng.get()));
    // Choosing the largest scaling factor that supports our plaintext space
    int log_scaling_factor = lwe::kIntBitwidth - RecordBitSize();

    // Plaintext is a selection vector for col_idx
    lwe::Vector ptxt = lwe::Vector::Zero(DbCols());
    ptxt[col_idx] = 1;
    // Computing b = pad * s + \Delta ptxt + e
    RLWE_RETURN_IF_ERROR(key.EncryptFromPadInPlace(
        ptxt, pad, log_scaling_factor, client_prng.get()));

    // Also need H_i1 * s, where H_i1 is the i1-th row of the hint H
    // From above there are DimLweSecretKey() columns of the hint,
    // so this will be 1 x 1.
    lwe::Vector decryption_part = hint.row(row_idx) * key.Key();
    ClientState client_state = ClientState{.row_idx = row_idx,
                                           .col_idx = col_idx,
                                           .decryption_part = decryption_part};
    return std::make_pair(client_state, ptxt);
  }

  // Computes and returns the ServerResponse, via the map
  // (client_query, database) -> database * client_query.
  absl::StatusOr<lwe::Vector> ServerResponse(
      const lwe::Matrix& database, const lwe::Vector& client_query) const {
    return database * client_query;
  }

  absl::StatusOr<lwe::Integer> ClientRecovery(
      const ClientState& state, const lwe::Vector& server_response) const {
    // Note that as server_response is a RefVector,
    // eigen ensures at compile-time that it only has a single column.
    if (server_response.rows() != DbRows()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The number of rows in the server response, ", server_response.cols(),
          " does not match the number in the public parameters, ", DbRows(),
          "."));
    }
    if (state.decryption_part.size() != 1) {
      return absl::InvalidArgumentError("The decryption part is not 1 x 1.");
    }

    // Remove hint * s from the server response, which gives us \Delta * m + e.
    lwe::Vector noisy_plaintext = server_response.row(state.row_idx);
    noisy_plaintext -= state.decryption_part;

    // Remove the error e.
    int log_scaling_factor = lwe::kIntBitwidth - RecordBitSize();
    RLWE_RETURN_IF_ERROR(
        lwe::RemoveErrorInPlace(noisy_plaintext, log_scaling_factor));

    // Extracting the coefficient from the 1 x 1 matrix noisy_plaintext.
    return noisy_plaintext.eval()(0);
  }

  // Accessors.
  absl::string_view Seed() const { return seed_; }
  int LweSecretDim() const { return lwe_secret_dim_; }
  int RecordBitSize() const { return record_bit_size_; }
  int DbRows() const { return db_rows_; }
  int DbCols() const { return db_cols_; }

 private:
  explicit Parems(absl::string_view seed, int lwe_secret_dim,
                  int record_bit_size, int db_rows, int db_cols)
      : seed_(std::string{seed}),
        lwe_secret_dim_(lwe_secret_dim),
        record_bit_size_(record_bit_size),
        db_rows_(db_rows),
        db_cols_(db_cols) {}

  std::string seed_;
  int lwe_secret_dim_;
  int record_bit_size_;
  int db_rows_;
  int db_cols_;
};

}  // namespace simplepir
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_HINTLESS_SIMPLEPIR_SIMPLEPIR_H_
