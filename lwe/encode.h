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

#ifndef HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_INTERNAL_H_
#define HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_INTERNAL_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "lwe/types.h"

namespace hintless_pir {
namespace lwe {

// Encodes the message in place using a scaling factor.
inline absl::Status EncodeMessageInPlace(Vector& message,
                                         int log_scaling_factor) {
  if (log_scaling_factor < 0 || log_scaling_factor > kIntBitwidth) {
    return absl::InvalidArgumentError(
        absl::StrCat("The log scaling factor, ", log_scaling_factor,
                     ", should be >= 0 and <= ", kIntBitwidth));
  }
  message *= (1 << log_scaling_factor);
  return absl::OkStatus();
}

// Removes the error in `noisy_message` in-place, where the message is scaled up
// with a scaling factor.
inline absl::Status RemoveErrorInPlace(Vector& noisy_message,
                                       int log_scaling_factor) {
  if (log_scaling_factor < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The log scaling factor, ", log_scaling_factor, ", should be >= 0"));
  }
  // noisy_message := \Delta m + e + (\Delta/2)
  noisy_message.array() += (1 << (log_scaling_factor - 1));
  // = floor(m + 1/2 + e/\Delta) = nearest_int(m + e/\Delta)
  noisy_message.array() /= (1 << log_scaling_factor);
  // Result may be large, reduce back to the ptxt space
  noisy_message = noisy_message.array().unaryExpr([&](Integer x) {
    return x % (1 << (kIntBitwidth - log_scaling_factor));
  });
  return absl::OkStatus();
}

}  // namespace lwe
}  // namespace hintless_pir

#endif  // HINTLESS_PIR_LWE_SYMMETRIC_ENCRYPTION_INTERNAL_H_
