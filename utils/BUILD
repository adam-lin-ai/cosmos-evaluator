# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "bazel",
    srcs = ["bazel.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks:__subpackages__",
        "//services:__subpackages__",
        "//utils/tests:__pkg__",
    ],
    deps = ["@rules_python//python/runfiles"],
)

# Generate image labels file with dynamic values
# NOTE: Uses the MODULE.bazel file to determine the workspace directory
genrule(
    name = "git_sha",
    outs = ["git_sha.txt"],
    cmd = """
        # Extract STABLE_GIT_SHA from the stable-status.txt file
        # This ensures the git SHA matches the build's workspace status
        GIT_SHA=$$(grep '^STABLE_GIT_SHA ' bazel-out/stable-status.txt | cut -d' ' -f2-)
        echo "GIT_SHA=$$GIT_SHA" > $@
    """,
    stamp = True,
    tags = ["manual"],
    visibility = [
        "//services:__subpackages__",
    ],
)
