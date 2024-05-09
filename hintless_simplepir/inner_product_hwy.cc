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

#include "hintless_simplepir/inner_product_hwy.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "hwy/detect_targets.h"
#include "lwe/types.h"

// Highway implementations.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hintless_simplepir/inner_product_hwy.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// clang-format on

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hintless_pir::hintless_simplepir::internal {
namespace HWY_NAMESPACE {

#if HWY_TARGET == HWY_SCALAR

template <typename PlainInteger>
absl::StatusOr<std::vector<lwe::Integer>> InnerProductHwy(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  return InnerProductNoHwy<PlainInteger>(matrix, vec);
}

#else

namespace hn = hwy::HWY_NAMESPACE;

template <typename PlainInteger>
absl::StatusOr<std::vector<lwe::Integer>> InnerProductHwy(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  if (matrix.size() != vec.size()) {
    return absl::InvalidArgumentError(
        "`matrix` and `vec` must have matching dimensions.");
  }

  // Vector type used throughout this function: Largest byte vector
  // available.
  const hn::ScalableTag<lwe::Integer> d32;
  const hn::Rebind<PlainInteger, hn::ScalableTag<lwe::Integer>> d_plain;
  const int N = hn::Lanes(d32);

  // Do not run the highway version if
  // - the number of bytes in a hwy vector is less than 16, or
  // - the number of bytes in a hwy vector is not a multiple of 16.
  if (ABSL_PREDICT_FALSE(N < 4 || N % 4 != 0)) {
    return InnerProductNoHwy<PlainInteger>(matrix, vec);
  }

  // Assume all columns have the same size.
  int num_blocks = matrix[0].size();
  int num_values_per_block = sizeof(BlockType) / sizeof(PlainInteger);
  int num_rows = num_blocks * num_values_per_block;

  // Allocate aligned buffers to hold the intermediate values of inner
  // products.
  hwy::AlignedFreeUniquePtr<lwe::Integer[]> aligned_results =
      hwy::AllocateAligned<lwe::Integer>(num_rows);
  std::fill_n(aligned_results.get(), num_rows, 0);

  for (int j = 0; j < vec.size(); ++j) {
    int row_idx = 0;
    const PlainInteger* value_ptr = 0;
    // First, run 4x SIMD multiplication in each iteration.
    for (; row_idx + N * 4 <= num_rows; row_idx += N * 4) {
      if (row_idx % num_values_per_block == 0) {
        int block_idx = row_idx / num_values_per_block;
        value_ptr =
            reinterpret_cast<const PlainInteger*>(&matrix[j][block_idx]);
      } else {
        value_ptr += N;
      }
      lwe::Integer* result_ptr = &aligned_results[row_idx];
      auto add32_0 = hn::Load(d32, result_ptr);
      auto add32_1 = hn::Load(d32, result_ptr + N);
      auto add32_2 = hn::Load(d32, result_ptr + 2 * N);
      auto add32_3 = hn::Load(d32, result_ptr + 3 * N);

      auto left0 = hn::LoadU(d_plain, value_ptr);
      auto left1 = hn::LoadU(d_plain, value_ptr + N);
      auto left2 = hn::LoadU(d_plain, value_ptr + 2 * N);
      auto left3 = hn::LoadU(d_plain, value_ptr + 3 * N);

      auto left32_0 = hn::PromoteTo(d32, left0);
      auto left32_1 = hn::PromoteTo(d32, left1);
      auto left32_2 = hn::PromoteTo(d32, left2);
      auto left32_3 = hn::PromoteTo(d32, left3);

      auto right32 = hn::Set(d32, vec[j]);

      auto mul32_0 = hn::MulAdd(left32_0, right32, add32_0);
      auto mul32_1 = hn::MulAdd(left32_1, right32, add32_1);
      auto mul32_2 = hn::MulAdd(left32_2, right32, add32_2);
      auto mul32_3 = hn::MulAdd(left32_3, right32, add32_3);

      hn::Store(mul32_0, d32, result_ptr);
      hn::Store(mul32_1, d32, result_ptr + N);
      hn::Store(mul32_2, d32, result_ptr + 2 * N);
      hn::Store(mul32_3, d32, result_ptr + 3 * N);
    }

    // Next, run 1x per iteration.
    for (; row_idx + N <= num_rows; row_idx += N) {
      if (row_idx % num_values_per_block == 0) {
        int block_idx = row_idx / num_values_per_block;
        value_ptr =
            reinterpret_cast<const PlainInteger*>(&matrix[j][block_idx]);
      } else {
        value_ptr += N;
      }
      lwe::Integer* result_ptr = &aligned_results[row_idx];
      auto add32 = hn::Load(d32, result_ptr);
      auto left = hn::LoadU(d_plain, value_ptr);
      auto left32 = hn::PromoteTo(d32, left);
      auto right32 = hn::Set(d32, vec[j]);
      auto mul32 = hn::MulAdd(left32, right32, add32);
      hn::Store(mul32, d32, result_ptr);
    }

    // Handle the remaining rows that didn't take a full lane.
    if (row_idx < num_rows) {
      int block_idx = row_idx / num_values_per_block;
      const lwe::PlainInteger* block_as_values =
          reinterpret_cast<const lwe::PlainInteger*>(&matrix[j][block_idx]);
      for (; row_idx < num_rows; ++row_idx) {
        if (row_idx % num_values_per_block == 0) {
          // update the block pointer, which should be rate
          block_idx = row_idx / num_values_per_block;
          block_as_values =
              reinterpret_cast<const lwe::PlainInteger*>(&matrix[j][block_idx]);
        }
        int block_pos = row_idx % num_values_per_block;
        aligned_results.get()[row_idx] +=
            static_cast<lwe::Integer>(block_as_values[block_pos]) * vec[j];
      }
    }
  }

  return std::vector<lwe::Integer>(aligned_results.get(),
                                   aligned_results.get() + num_rows);
}

#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace HWY_NAMESPACE
}  // namespace hintless_pir::hintless_simplepir::internal
HWY_AFTER_NAMESPACE();

#if HWY_ONCE || HWY_IDE
namespace hintless_pir::hintless_simplepir::internal {

template <typename PlainInteger>
absl::StatusOr<std::vector<lwe::Integer>> InnerProductNoHwy(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  if (matrix.size() != vec.size()) {
    return absl::InvalidArgumentError(
        "`matrix` and `vec` must have matching dimensions.");
  }

  constexpr int num_values_per_block =
      sizeof(BlockType) / sizeof(lwe::PlainInteger);

  // Assume all columns have the same size.
  int num_blocks = matrix[0].size();
  int num_rows = num_blocks * sizeof(BlockType);

  std::vector<lwe::Integer> result(num_rows, 0);
  for (int j = 0; j < vec.size(); ++j) {
    int i = 0;
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
      BlockType block = matrix[j][block_idx];
      const lwe::PlainInteger* block_as_values =
          reinterpret_cast<const lwe::PlainInteger*>(&block);
      for (int block_pos = 0; block_pos < num_values_per_block && i < num_rows;
           ++block_pos, ++i) {
        result[i] +=
            static_cast<lwe::Integer>(block_as_values[block_pos]) * vec[j];
      }
    }
  }
  return result;
}

// Only instantiate the 8-bit and 16-bit versions, which are the choices of
// LWE plaintext integer types we support.
HWY_EXPORT_T(InnerProductHwy8, InnerProductHwy<uint8_t>);
HWY_EXPORT_T(InnerProductHwy16, InnerProductHwy<uint16_t>);

template <typename PlainInteger>
absl::StatusOr<std::vector<lwe::Integer>> InnerProduct(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  return InnerProductNoHwy<PlainInteger>(matrix, vec);
}

template <>
absl::StatusOr<std::vector<lwe::Integer>> InnerProduct<uint8_t>(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  return HWY_DYNAMIC_DISPATCH_T(InnerProductHwy8)(matrix, vec);
}

template <>
absl::StatusOr<std::vector<lwe::Integer>> InnerProduct<uint16_t>(
    absl::Span<const BlockVector> matrix, absl::Span<const lwe::Integer> vec) {
  return HWY_DYNAMIC_DISPATCH_T(InnerProductHwy16)(matrix, vec);
}

}  // namespace hintless_pir::hintless_simplepir::internal
#endif  // HWY_ONCE || HWY_IDE
