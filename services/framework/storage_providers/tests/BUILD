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

py_test(
    name = "s3_storage_provider_test",
    srcs = ["s3_storage_provider_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//services/framework/protocols:storage_provider",
        "//services/framework/storage_providers:s3_storage_provider",
        "//services/framework/storage_providers:s3_url_utils",
        requirement("aiohttp"),
        requirement("botocore"),
        requirement("pytest-asyncio"),
    ],
)

py_test(
    name = "factory_test",
    srcs = ["factory_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//services:settings_base",
        "//services/framework/storage_providers:factory",
        "//services/framework/storage_providers:local_storage_provider",
        "//services/framework/storage_providers:s3_storage_provider",
    ],
)
