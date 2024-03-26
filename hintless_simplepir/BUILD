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

# An implementation of the Hintless SimplePIR scheme that uses RLWE-based
# homomorphic encryption to remove the database-dependent "hint" in SimplePIR.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# SimplePIR implementation.
cc_library(
    name = "simplepir",
    hdrs = [
        "simplepir.h",
    ],
    deps = [
        "//lwe:encode",
        "//lwe:lwe_symmetric_encryption",
        "//lwe:sample_error",
        "//lwe:types",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "simplepir_test",
    srcs = ["simplepir_test.cc"],
    deps = [
        ":simplepir",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

# Hintless SimplePIR request and response types.
proto_library(
    name = "serialization_proto",
    srcs = ["serialization.proto"],
    deps = [
        "//linpir:serialization_proto",
        "@com_github_google_shell-encryption//shell_encryption/rns:serialization_proto",
    ],
)

cc_proto_library(
    name = "serialization_cc_proto",
    deps = [":serialization_proto"],
)

# Hintless SimplePIR parameters.
cc_library(
    name = "parameters",
    hdrs = ["parameters.h"],
    deps = [
        "//linpir:parameters",
        "//lwe:types",
        "@com_github_google_shell-encryption//shell_encryption:serialization_cc_proto",
    ],
)

# Test helpers.
cc_library(
    name = "testing",
    hdrs = ["testing.h"],
    deps = [
        ":parameters",
        ":utils",
        "@com_google_absl//absl/random",
    ],
)

# Utilities
cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        ":parameters",
        ":serialization_cc_proto",
        "//lwe:types",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":parameters",
        ":testing",
        ":utils",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# Hintless SimplePIR database and the preprocessed hint states
cc_library(
    name = "database",
    srcs = ["database.cc"],
    hdrs = ["database.h"],
    deps = [
        ":parameters",
        ":utils",
        "//lwe:types",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "database_test",
    srcs = ["database_test.cc"],
    deps = [
        ":database",
        ":parameters",
        ":testing",
        ":utils",
        "//lwe:lwe_symmetric_encryption",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_github_google_shell-encryption//shell_encryption/testing:testing_prng",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

# Hintless SimplePIR server.
cc_library(
    name = "server",
    srcs = ["server.cc"],
    hdrs = ["server.h"],
    deps = [
        ":database",
        ":parameters",
        ":serialization_cc_proto",
        ":utils",
        "//linpir:database",
        "//linpir:server",
        "//lwe:lwe_symmetric_encryption",
        "//lwe:types",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_chacha_prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "server_test",
    srcs = ["server_test.cc"],
    deps = [
        ":database",
        ":parameters",
        ":server",
        ":testing",
        ":utils",
        "//linpir:parameters",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

# Hintless SimplePIR client.
cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":parameters",
        ":serialization_cc_proto",
        ":utils",
        "//linpir:client",
        "//lwe:encode",
        "//lwe:lwe_symmetric_encryption",
        "//lwe:types",
        "@com_github_google_shell-encryption//shell_encryption:int256",
        "@com_github_google_shell-encryption//shell_encryption:montgomery",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_chacha_prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/rns:crt_interpolation",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_context",
        "@com_github_google_shell-encryption//shell_encryption/rns:rns_modulus",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "client_test",
    srcs = ["client_test.cc"],
    deps = [
        ":client",
        ":parameters",
        ":utils",
        "//linpir:parameters",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# End-to-end test.
cc_test(
    name = "hintless_simplepir_test",
    srcs = ["hintless_simplepir_test.cc"],
    deps = [
        ":client",
        ":database",
        ":parameters",
        ":server",
        ":utils",
        "//linpir:parameters",
        "//lwe:types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
