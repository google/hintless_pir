/*
 * Copyright 2026 Google LLC.
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

#ifndef HINTLESS_PIR_PRIVATE_INFERENCE_UTILS_H_
#define HINTLESS_PIR_PRIVATE_INFERENCE_UTILS_H_
// Utility functions

namespace private_inference {
namespace utils {

// Returns ceil(x / y).
template <typename T>
inline T DivAndRoundUp(T x, T y) {
  return (x + y - 1) / y;
}

}  // namespace utils
}  // namespace private_inference

#endif  // HINTLESS_PIR_PRIVATE_INFERENCE_UTILS_H_
