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
load("@rules_python//python:py_test.bzl", "py_test")
load("//services:local_ports.bzl", "LOCAL_PORTS")

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = ["//services/vlm:image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["vlm"])},
    tags = [
        "exclusive",  # Run in isolation to avoid resource contention
        "no-coverage",  # No meaningful Python coverage is collected so exclude from coverage CI job to improve runtime
    ],
    deps = [
        "//utils:bazel",
        requirement("testcontainers"),
    ],
)

py_test(
    name = "service_test",
    srcs = ["service_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//services/vlm:service",
        requirement("pydantic"),
        requirement("pytest"),
        requirement("pytest-asyncio"),
    ],
)

py_test(
    name = "rest_api_common_test",
    srcs = ["rest_api_common_test.py"],
    deps = [
        "//services:utils",
        "//services/vlm:rest_api_common",
        "//services/vlm:service",
        requirement("fastapi"),
        requirement("httpx"),
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//services/vlm:rest_api",
        requirement("fastapi"),
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/vlm:settings",
        requirement("pydantic"),
    ],
)
