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
    name = "common",
    srcs = ["common.py"],
    visibility = ["//checks/attribute_verification/tests:__pkg__"],
)

py_library(
    name = "question_generator",
    srcs = ["question_generator.py"],
    visibility = ["//checks/attribute_verification/tests:__pkg__"],
    deps = [
        "//checks/attribute_verification:common",
        requirement("openai"),
    ],
)

py_library(
    name = "vlm_verifier",
    srcs = ["vlm_verifier.py"],
    visibility = ["//checks/attribute_verification/tests:__pkg__"],
    deps = [
        "//checks/attribute_verification:common",
        requirement("openai"),
        requirement("multi-storage-client"),
        requirement("opencv-python-headless"),
    ],
)

py_library(
    name = "processor",
    srcs = ["processor.py"],
    data = ["//checks:config"],
    visibility = [
        "//checks/attribute_verification/tests:__pkg__",
        "//services/attribute_verification:__pkg__",
    ],
    deps = [
        "//checks/attribute_verification:question_generator",
        "//checks/attribute_verification:vlm_verifier",
        "//checks/utils:config_manager",
        requirement("fastapi"),
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = ["//checks:config"] + glob(
        [".env"],
        allow_empty = True,
    ),
    visibility = ["//checks/attribute_verification/tests:__pkg__"],
    deps = [
        ":processor",
        "//checks/utils:config_manager",
        "//checks/utils:multistorage",
        requirement("python-dotenv"),
        requirement("pyyaml"),
    ],
)
