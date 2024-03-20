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

# Symmetric-key encryption scheme based on Learning With Errors.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        "@com_gitlab_libeigen-eigen//:eigen3",
    ],
)

cc_library(
    name = "sample_error",
    hdrs = [
        "sample_error.h",
    ],
    deps = [
        ":types",
        "@com_github_google_shell-encryption//shell_encryption:bits_util",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "sample_error_test",
    srcs = ["sample_error_test.cc"],
    deps = [
        ":sample_error",
        ":types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_github_google_shell-encryption//shell_encryption/testing:testing_prng",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "encode",
    hdrs = [
        "encode.h",
    ],
    deps = [
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "lwe_symmetric_encryption",
    hdrs = ["lwe_symmetric_encryption.h"],
    deps = [
        ":encode",
        ":sample_error",
        ":types",
        "@com_github_google_shell-encryption//shell_encryption:statusor_fork",
        "@com_github_google_shell-encryption//shell_encryption/prng",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "lwe_symmetric_encryption_test",
    srcs = ["lwe_symmetric_encryption_test.cc"],
    deps = [
        ":encode",
        ":lwe_symmetric_encryption",
        ":types",
        "@com_github_google_googletest//:gtest_main",
        "@com_github_google_shell-encryption//shell_encryption/prng:single_thread_hkdf_prng",
        "@com_github_google_shell-encryption//shell_encryption/testing:matchers",
        "@com_github_google_shell-encryption//shell_encryption/testing:status_testing",
        "@com_gitlab_libeigen-eigen//:eigen3",
        "@com_google_absl//absl/status",
    ],
)
