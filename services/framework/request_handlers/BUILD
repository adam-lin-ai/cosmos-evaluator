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
    name = "json_request_handler",
    srcs = ["json_request_handler.py"],
    tags = ["no-mypy"],
    visibility = [
        "//services/framework/request_handlers/tests:__pkg__",
        "//services/obstacle_correspondence:__pkg__",
    ],
    deps = [
        requirement("fastapi"),
        requirement("pydantic"),
    ],
)
