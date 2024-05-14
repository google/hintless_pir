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

#include "hintless_simplepir/database.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/utils.h"
#include "lwe/types.h"

namespace hintless_pir {
namespace hintless_simplepir {

absl::StatusOr<std::unique_ptr<Database>> Database::Create(
    const Parameters& parameters) {
  // Initialize the data and the hint matrices for all shards.
  int num_shards = DivAndRoundUp(parameters.db_record_bit_size,
                                 parameters.lwe_plaintext_bit_size);
  std::vector<lwe::Matrix> data_matrices(num_shards);
  std::vector<lwe::Matrix> hint_matrices(num_shards);
  for (int i = 0; i < num_shards; ++i) {
    data_matrices[i] =
        lwe::Matrix::Zero(parameters.db_rows, parameters.db_cols);
    hint_matrices[i] =
        lwe::Matrix::Zero(parameters.db_rows, parameters.lwe_secret_dim);
  }
  return absl::WrapUnique(
      new Database(parameters, /*lwe_query_pad=*/nullptr, /*num_records=*/0,
                   std::move(data_matrices), std::move(hint_matrices)));
}

namespace {

// Function object to generate random data matrix entries.
struct RandomValueOp {
  int num_bits;

  inline lwe::Integer operator()() const {
    lwe::Integer r = std::rand();
    lwe::Integer mask = (lwe::Integer{1} << num_bits) - 1;
    return r & mask;
  }
};

}  // namespace

absl::StatusOr<std::unique_ptr<Database>> Database::CreateRandom(
    const Parameters& parameters) {
  // Initialize the data and the hint matrices for all shards.
  int num_shards = DivAndRoundUp(parameters.db_record_bit_size,
                                 parameters.lwe_plaintext_bit_size);
  std::vector<lwe::Matrix> data_matrices(num_shards);
  std::vector<lwe::Matrix> hint_matrices(num_shards);
  for (int i = 0; i < num_shards; ++i) {
    // Generate a random matrix.
    data_matrices[i] = lwe::Matrix::NullaryExpr(
        parameters.db_rows, parameters.db_cols,
        RandomValueOp{parameters.lwe_plaintext_bit_size});
    // Hint cannot be computed until the LWE pad matrix is set.
    hint_matrices[i] =
        lwe::Matrix::Zero(parameters.db_rows, parameters.lwe_secret_dim);
  }
  int64_t num_records = parameters.db_rows * parameters.db_cols;
  return absl::WrapUnique(new Database(parameters, /*lwe_query_pad=*/nullptr,
                                       num_records, std::move(data_matrices),
                                       std::move(hint_matrices)));
}

absl::Status Database::UpdateLweQueryPad(const lwe::Matrix* lwe_query_pad) {
  if (lwe_query_pad == nullptr) {
    return absl::InvalidArgumentError("`lwe_query_pad` must not be null.");
  }
  lwe_query_pad_ = lwe_query_pad;
  return absl::OkStatus();
}

absl::Status Database::Append(absl::string_view record) {
  if (record.size() * 8 >= params_.db_record_bit_size + 8 ||
      record.size() * 8 < params_.db_record_bit_size) {
    return absl::InvalidArgumentError("`record` has incorrect size.");
  }
  if (num_records_ >= params_.db_rows * params_.db_cols) {
    return absl::InvalidArgumentError("Database is full.");
  }
  int64_t row_idx, col_idx;
  std::tie(row_idx, col_idx) = MatrixCoordinate(num_records_);
  num_records_++;
  std::vector<lwe::Integer> values = SplitRecord(record, params_);
  for (int i = 0; i < values.size(); ++i) {
    data_matrices_[i](row_idx, col_idx) = values[i];
  }
  return absl::OkStatus();
}

absl::Status Database::UpdateHints() {
  if (lwe_query_pad_ == nullptr) {
    return absl::FailedPreconditionError("LWE query pad not set.");
  }
  for (int i = 0; i < data_matrices_.size(); ++i) {
    hint_matrices_[i] = data_matrices_[i] * (*lwe_query_pad_);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<lwe::Vector>> Database::InnerProductWith(
    const lwe::Vector& query) const {
  std::vector<lwe::Vector> results;
  results.reserve(data_matrices_.size());
  for (auto const& matrix : data_matrices_) {
    results.push_back(matrix * query);
  }
  return results;
}

absl::StatusOr<std::string> Database::Record(int64_t index) const {
  if (index < 0 || index >= num_records_) {
    return absl::InvalidArgumentError("`index` is out of range.");
  }
  int64_t row_idx, col_idx;
  std::tie(row_idx, col_idx) = MatrixCoordinate(index);
  std::vector<lwe::Integer> values;
  values.reserve(data_matrices_.size());
  for (auto const& data_matrix : data_matrices_) {
    values.push_back(data_matrix(row_idx, col_idx));
  }
  return ReconstructRecord(values, params_);
}

}  // namespace hintless_simplepir
}  // namespace hintless_pir
