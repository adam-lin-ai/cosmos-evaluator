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
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "json_response_formatter",
    srcs = ["json_response_formatter.py"],
    tags = ["no-mypy"],
    visibility = [
        "//services:__subpackages__",
    ],
    deps = [
        requirement("fastapi"),
    ],
)

py_library(
    name = "streaming_response_formatter",
    srcs = ["streaming_response_formatter.py"],
    tags = ["no-mypy"],
    visibility = [
        "//services:__subpackages__",
    ],
    deps = [
        ":json_response_formatter",
        requirement("aiofiles"),
        requirement("fastapi"),
    ],
)
