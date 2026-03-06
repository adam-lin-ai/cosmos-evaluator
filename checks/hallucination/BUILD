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

load("@pypi//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "processor",
    srcs = ["processor.py"],
    data = ["//checks:config"],
    visibility = [
        "//checks/hallucination/tests:__pkg__",
        "//services/hallucination:__pkg__",
    ],
    deps = [
        ":frame_processing",
        "//checks/utils:config_manager",
        "//checks/utils:frame_processing",
        "//checks/utils:multistorage",
        "//checks/utils:types",
        requirement("opencv_python_headless"),
    ],
)

py_library(
    name = "frame_processing",
    srcs = ["frame_processing.py"],
    visibility = [
        "//checks/hallucination/tests:__pkg__",
    ],
    deps = [
        requirement("opencv_python_headless"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = ["//checks:config"] + glob(
        [".env"],
        allow_empty = True,
    ),
    visibility = ["//checks/hallucination/tests:__pkg__"],
    deps = [
        ":processor",
        "//checks/utils:config_manager",
        "//checks/utils:types",
        requirement("python-dotenv"),
        requirement("pyyaml"),
    ],
)
