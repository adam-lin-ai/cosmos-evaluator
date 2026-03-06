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

py_library(
    name = "bev_render",
    srcs = ["bev_render.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/utils/tests:__pkg__",
        "//checks/vlm:__pkg__",
    ],
    deps = [
        "//checks/utils:rds_data_loader",
        requirement("numpy"),
        requirement("opencv_python_headless"),
    ],
)

py_library(
    name = "camera_render",
    srcs = ["camera_render.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils:__pkg__",
        "//checks/vlm:__subpackages__",
    ],
    deps = [
        "@cosmos_drive_dreams_toolkits//:render_from_rds_hq",
    ],
)
