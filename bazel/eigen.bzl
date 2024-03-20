# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bazel build file for Eigen3
"""

package(
    default_visibility = ["//visibility:public"],
)

all_files_with_extensions = glob(["**/*.*"])

eigen_hdrs = glob(
    [
        "Eigen/*",
        "unsupported/Eigen/*",
        "unsupported/Eigen/CXX11/*",
    ],
    exclude = [
        "**/src/**",
    ] + all_files_with_extensions,
)

eigen_srcs = glob(
    [
        "Eigen/**/src/**/*.h",
        "Eigen/**/src/**/*.inc",
        "unsupported/Eigen/**/src/**/*.h",
        "unsupported/Eigen/**/src/**/*.inc",
    ],
)

cc_library(
    name = "eigen3",
    srcs = eigen_srcs,
    hdrs = eigen_hdrs,
    deps = [],
)
