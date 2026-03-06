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
    name = "exceptions",
    srcs = ["exceptions.py"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [requirement("pydantic")],
)

py_library(
    name = "s3_url_utils",
    srcs = ["s3_url_utils.py"],
    visibility = ["//services:__subpackages__"],
)

py_library(
    name = "s3_storage_provider",
    srcs = ["s3_storage_provider.py"],
    visibility = ["//services:__subpackages__"],
    deps = [
        ":config",
        ":exceptions",
        ":s3_url_utils",
        "//services:settings_base",
        "//services/framework:url_security",
        "//services/framework/protocols:storage_provider",
        requirement("aioboto3"),
        requirement("aiohttp"),
        requirement("boto3"),
        requirement("botocore"),
        requirement("certifi"),
    ],
)

py_library(
    name = "local_storage_provider",
    srcs = ["local_storage_provider.py"],
    visibility = ["//services:__subpackages__"],
    deps = [
        "//services/framework/protocols:storage_provider",
        requirement("aiohttp"),
    ],
)

py_library(
    name = "factory",
    srcs = ["factory.py"],
    visibility = ["//services:__subpackages__"],
    deps = [
        ":config",
        ":local_storage_provider",
        ":s3_storage_provider",
        "//services:settings_base",
        "//services/framework/protocols:storage_provider",
    ],
)
