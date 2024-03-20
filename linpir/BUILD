# Copyright 2024 Google LLC.
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

# RLWE-based linear Private Information Retrieval

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

proto_library(
    name = "serialization_proto",
    srcs = ["serialization.proto"],
    deps = [
        "@com_github_google_shell-encryption//shell_encryption/rns:serialization_proto",
    ],
)

cc_proto_library(
    name = "serialization_cc_proto",
    deps = [":serialization_proto"],
)

# Parameters
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

# LinPIR database
cc_library(
    name = "database",
    srcs = ["database.cc"],
    hdrs = ["database.h"],
    deps = [
        ":parameters",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/rns:finite_field_encoder",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_bfv_ciphertext",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_ciphertext",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_polynomial",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "database_test",
    srcs = ["database_test.cc"],
    deps = [
        ":database",
        ":parameters",
        "@com_github_google_googletest//:gtest_main",
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
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# LinPIR server
cc_library(
    name = "server",
    srcs = ["server.cc"],
    hdrs = ["server.h"],
    deps = [
        ":database",
        ":parameters",
        ":serialization_cc_proto",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_chacha_prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_bfv_ciphertext",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_error_params",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_gadget",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_galois_key",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_polynomial",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

cc_test(
    name = "server_test",
    srcs = ["server_test.cc"],
    deps = [
        ":database",
        ":parameters",
        ":serialization_cc_proto",
        ":server",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:serialization_cc_proto",
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
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# LinPIR client
cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":parameters",
        ":serialization_cc_proto",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_chacha_prng",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client",
        ":parameters",
        ":serialization_cc_proto",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:serialization_cc_proto",
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
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# End-to-end test
cc_test(
    name = "linpir_test",
    srcs = ["linpir_test.cc"],
    deps = [
        ":client",
        ":database",
        ":parameters",
        ":server",
        "@com_github_google_googletest//:gtest",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:finite_field_encoder",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_polynomial",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/time",
    ],
)
