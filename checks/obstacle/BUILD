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
    name = "importance_filter",
    srcs = ["importance_filter.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        "//checks/utils:coord_transforms",
    ],
)

py_library(
    name = "hallucination_detector",
    srcs = ["hallucination_detector.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
    ],
)

py_library(
    name = "hallucination_tracking",
    srcs = ["hallucination_tracking.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        "//checks/utils:coord_transforms",
    ],
)

py_library(
    name = "overlap_detector",
    srcs = ["overlap_detector.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        ":importance_filter",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
    ],
)

py_library(
    name = "seg_helper",
    srcs = ["seg_helper.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        "//checks/utils:segformer",
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_library(
    name = "object_processor_base",
    srcs = ["object_processor_base.py"],
    data = [
        "//checks:config",
    ],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tests:__pkg__",
        "//services/obstacle_correspondence:__pkg__",
    ],
    deps = [
        ":importance_filter",
        ":results",
        ":visualization_helper",
        "//checks/utils:config_manager",
        "//checks/utils:coord_transforms",
        "//checks/utils:cuboid",
        "//checks/utils:profiler",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils:segformer",
    ],
)

py_library(
    name = "object_processor_dynamic",
    srcs = ["object_processor_dynamic.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tests:__pkg__",
        "//services/obstacle_correspondence:__pkg__",
    ],
    deps = [
        ":hallucination_detector",
        ":hallucination_tracking",
        ":object_processor_base",
        ":overlap_detector",
        ":seg_helper",
        "//checks/utils:scene_rasterizer",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tests:__pkg__",
        "//services/obstacle_correspondence:__pkg__",
    ],
    deps = [
        ":object_processor_base",
        ":object_processor_dynamic",
    ],
)

py_library(
    name = "results",
    srcs = ["results.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tests:__pkg__",
        "//checks/obstacle/tools:__pkg__",
    ],
    deps = [
        requirement("numpy"),
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [
        "//checks:config",
    ],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tests:__pkg__",
    ],
    deps = [
        ":api",
        ":results",
        "//checks/utils:cli_common",
        "//checks/utils:config_manager",
        requirement("boto3"),
        requirement("numpy"),
    ],
)

py_library(
    name = "visualization_helper",
    srcs = ["visualization_helper.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/obstacle/tests:__pkg__"],
    deps = [
        "//checks/utils:overlay_video_helper",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils:video",
        requirement("opencv_python_headless"),
        requirement("numpy"),
        requirement("torch"),
    ],
)
