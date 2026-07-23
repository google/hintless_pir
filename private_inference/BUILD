# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Batched private inference protocol

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "parameters",
    hdrs = ["parameters.h"],
    deps = [
        "@com_github_google_shell-encryption//shell_encryption:integral_types",
        "@com_github_google_shell-encryption//shell_encryption:serialization_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "weight_matrix",
    srcs = ["weight_matrix.cc"],
    hdrs = ["weight_matrix.h"],
    deps = [
        ":parameters",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:finite_field_encoder",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_bfv_ciphertext",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_galois_key",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_polynomial",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "weight_matrix_test",
    srcs = ["weight_matrix_test.cc"],
    deps = [
        ":parameters",
        ":weight_matrix",
        "@com_github_google_googletest//:gtest",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:finite_field_encoder",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_bfv_ciphertext",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_error_params",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_gadget",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_galois_key",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_polynomial",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_secret_key",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)
