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
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("//services:local_ports.bzl", "LOCAL_PORTS")

py_library(
    name = "image_test_mixin",
    srcs = ["image_test_mixin.py"],
    deps = [
        "//utils:bazel",
        requirement("testcontainers"),
    ],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = ["//services/obstacle_correspondence:image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["obstacle_correspondence"])},
    tags = [
        "exclusive",
        "gpu",
        "no-coverage",
    ],
    deps = [":image_test_mixin"],
)

py_test(
    name = "obstacle_correspondence_service_test",
    srcs = ["obstacle_correspondence_service_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//services/obstacle_correspondence:models",
        "//services/obstacle_correspondence:obstacle_correspondence_service",
        requirement("pydantic"),
    ],
)

py_test(
    name = "rest_api_common_test",
    srcs = ["rest_api_common_test.py"],
    deps = [
        "//services/framework/protocols:storage_provider",
        "//services/obstacle_correspondence:models",
        "//services/obstacle_correspondence:rest_api_common",
        requirement("fastapi"),
        requirement("httpx"),
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//services/obstacle_correspondence:rest_api",
        requirement("fastapi"),
        requirement("httpx"),
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/obstacle_correspondence:settings",
        requirement("pydantic"),
    ],
)
