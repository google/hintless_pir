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

#include "hintless_simplepir/database_hwy.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "hintless_simplepir/inner_product_hwy.h"
#include "hintless_simplepir/parameters.h"
#include "hintless_simplepir/utils.h"
#include "lwe/types.h"
#include "shell_encryption/status_macros.h"

namespace hintless_pir {
namespace hintless_simplepir {
namespace {

static inline Database::RawMatrix CreateZeroRawMatrix(size_t num_rows,
                                                      size_t num_cols) {
  size_t num_values_per_block =
      sizeof(internal::BlockType) / sizeof(lwe::PlainInteger);
  size_t num_blocks_per_col = DivAndRoundUp(num_rows, num_values_per_block);
  Database::RawMatrix matrix(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    matrix[i].resize(num_blocks_per_col, 0);
  }
  return matrix;
}

static inline Database::RawMatrix CreateRandomRawMatrix(size_t num_rows,
                                                        size_t num_cols,
                                                        size_t plain_bits) {
  size_t num_values_per_block =
      sizeof(internal::BlockType) / sizeof(lwe::PlainInteger);
  size_t num_blocks_per_col = DivAndRoundUp(num_rows, num_values_per_block);
  lwe::Integer mask = (lwe::Integer{1} << plain_bits) - 1;
  Database::RawMatrix matrix(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    matrix[i].resize(num_blocks_per_col, 0);
    for (int j = 0; j < num_blocks_per_col; ++j) {
      for (int k = 0, b = 0; k < num_values_per_block;
           ++k, b += 8 * sizeof(lwe::PlainInteger)) {
        lwe::Integer r = std::rand();
        matrix[i][j] |= static_cast<internal::BlockType>(r & mask) << b;
      }
    }
  }
  return matrix;
}

static inline Database::LweMatrix CreateZeroMatrix(size_t num_rows,
                                                   size_t num_cols) {
  Database::LweMatrix matrix(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    matrix[i].resize(num_cols, 0);
  }
  return matrix;
}

// Assume both `plain_matrix` and `lwe_matrix` are stored by columns.
static inline absl::StatusOr<Database::LweMatrix> MatrixProduct(
    const Database::RawMatrix& plain_matrix,
    const Database::LweMatrix& lwe_matrix, size_t num_rows) {
  Database::LweMatrix cols;
  cols.reserve(lwe_matrix.size());
  for (int i = 0; i < lwe_matrix.size(); ++i) {
    RLWE_ASSIGN_OR_RETURN(
        Database::LweVector col,
        internal::InnerProduct<lwe::PlainInteger>(plain_matrix, lwe_matrix[i]));
    cols.push_back(col);
  }
  // return `matrix` organized by rows.
  Database::LweMatrix matrix(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    matrix[i].resize(lwe_matrix.size());
    for (int j = 0; j < lwe_matrix.size(); ++j) {
      matrix[i][j] = cols[j][i];
    }
  }
  return matrix;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Database>> Database::Create(
    const Parameters& parameters) {
  // Initialize the data and the hint matrices for all shards.
  int num_shards = DivAndRoundUp(parameters.db_record_bit_size,
                                 parameters.lwe_plaintext_bit_size);
  std::vector<RawMatrix> data_matrices(num_shards);
  std::vector<LweMatrix> hint_matrices(num_shards);
  for (int i = 0; i < num_shards; ++i) {
    data_matrices[i] =
        CreateZeroRawMatrix(parameters.db_rows, parameters.db_cols);
    hint_matrices[i] =
        CreateZeroMatrix(parameters.db_rows, parameters.lwe_secret_dim);
  }
  return absl::WrapUnique(
      new Database(parameters, /*lwe_query_pad=*/nullptr, /*num_records=*/0,
                   std::move(data_matrices), std::move(hint_matrices)));
}

absl::StatusOr<std::unique_ptr<Database>> Database::CreateRandom(
    const Parameters& parameters) {
  // Initialize the data and the hint matrices for all shards.
  int num_shards = DivAndRoundUp(parameters.db_record_bit_size,
                                 parameters.lwe_plaintext_bit_size);
  std::vector<RawMatrix> data_matrices(num_shards);
  std::vector<LweMatrix> hint_matrices(num_shards);
  for (int i = 0; i < num_shards; ++i) {
    data_matrices[i] =
        CreateRandomRawMatrix(parameters.db_rows, parameters.db_cols,
                              parameters.lwe_plaintext_bit_size);
    hint_matrices[i] =
        CreateZeroMatrix(parameters.db_rows, parameters.lwe_secret_dim);
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
  int64_t num_values_per_block = sizeof(BlockType) / sizeof(lwe::PlainInteger);
  int64_t block_idx = row_idx / num_values_per_block;
  int64_t block_pos = row_idx % num_values_per_block;
  int64_t base_bits = block_pos * 8 * sizeof(lwe::PlainInteger);

  num_records_++;
  std::vector<lwe::Integer> values = SplitRecord(record, params_);
  for (int i = 0; i < values.size(); ++i) {
    BlockType block = static_cast<BlockType>(values[i]) << base_bits;
    data_matrices_[i][col_idx][block_idx] |= block;
  }
  return absl::OkStatus();
}

absl::Status Database::UpdateHints() {
  if (lwe_query_pad_ == nullptr) {
    return absl::FailedPreconditionError("LWE query pad not set.");
  }
  for (int i = 0; i < data_matrices_.size(); ++i) {
    LweMatrix lwe_matrix = ImportLweMatrix(*lwe_query_pad_);
    RLWE_ASSIGN_OR_RETURN(
        hint_matrices_[i],
        MatrixProduct(data_matrices_[i], lwe_matrix, params_.db_rows));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Database::LweVector>> Database::InnerProductWith(
    const LweVector& query) const {
  std::vector<LweVector> results;
  results.reserve(data_matrices_.size());
  for (auto const& matrix : data_matrices_) {
    RLWE_ASSIGN_OR_RETURN(
        LweVector result,
        internal::InnerProduct<lwe::PlainInteger>(matrix, query));
    result.resize(params_.db_rows);
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<std::string> Database::Record(int64_t index) const {
  if (index < 0 || index >= num_records_) {
    return absl::InvalidArgumentError("`index` is out of range.");
  }
  int64_t row_idx, col_idx;
  std::tie(row_idx, col_idx) = MatrixCoordinate(index);
  int64_t num_values_per_block = sizeof(BlockType) / sizeof(lwe::PlainInteger);
  int64_t block_idx = row_idx / num_values_per_block;
  int64_t block_pos = row_idx % num_values_per_block;
  int64_t base_bits = block_pos * 8 * sizeof(lwe::PlainInteger);

  BlockType mask = (BlockType{1} << params_.lwe_plaintext_bit_size) - 1;
  std::vector<lwe::Integer> values;
  values.reserve(data_matrices_.size());
  for (auto const& data_matrix : data_matrices_) {
    BlockType block = data_matrix[col_idx][block_idx] >> base_bits;
    values.push_back(static_cast<lwe::Integer>(block & mask));
  }
  return ReconstructRecord(values, params_);
}

Database::LweMatrix ImportLweMatrix(const lwe::Matrix& matrix) {
  // `results` organized by columns.
  Database::LweMatrix results(matrix.cols());
  for (int64_t j = 0; j < matrix.cols(); ++j) {
    results[j].resize(matrix.rows(), 0);
    for (int64_t i = 0; i < matrix.rows(); ++i) {
      results[j][i] = matrix(i, j);
    }
  }
  return results;
}

lwe::Matrix ExportLweMatrix(const Database::LweMatrix& matrix) {
  // Assume `matrix` organized by columns.
  int64_t num_cols = matrix.size();
  int64_t num_rows = matrix[0].size();
  lwe::Matrix results = lwe::Matrix::Zero(num_rows, num_cols);
  for (int64_t j = 0; j < num_cols; ++j) {
    for (int64_t i = 0; i < num_rows; ++i) {
      results(i, j) = matrix[j][i];
    }
  }
  return results;
}

lwe::Matrix ExportRawMatrix(const Database::RawMatrix& matrix, size_t num_rows,
                            size_t num_bits_per_value) {
  // Assume `matrix` organized by columns.
  int64_t num_cols = matrix.size();
  int64_t num_values_per_block =
      Database::kBlockBits / sizeof(lwe::PlainInteger);
  lwe::Integer mask = (lwe::Integer{1} << num_bits_per_value) - 1;
  lwe::Matrix results = lwe::Matrix::Zero(num_rows, num_cols);
  for (int64_t col_idx = 0; col_idx < num_cols; ++col_idx) {
    for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      int64_t block_idx = row_idx / num_values_per_block;
      int64_t block_pos = row_idx % num_values_per_block;
      int64_t base_bits = block_pos * 8 * sizeof(lwe::PlainInteger);
      auto raw =
          static_cast<lwe::Integer>(matrix[col_idx][block_idx] >> base_bits);
      results(row_idx, col_idx) = raw & mask;
    }
  }
  return results;
}

}  // namespace hintless_simplepir
}  // namespace hintless_pir
