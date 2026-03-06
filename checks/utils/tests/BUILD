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
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "cli_common_test",
    srcs = ["cli_common_test.py"],
    tags = ["no-mypy"],
    deps = ["//checks/utils:cli_common"],
)

py_test(
    name = "cuboid_test",
    srcs = ["cuboid_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:cuboid",
        requirement("numpy"),
    ],
)

py_test(
    name = "polyline_test",
    srcs = ["polyline_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:polyline",
        requirement("numpy"),
    ],
)

py_test(
    name = "surface_test",
    srcs = ["surface_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:surface",
        requirement("numpy"),
    ],
)

py_test(
    name = "scene_rasterizer_test",
    srcs = ["scene_rasterizer_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:rds_geometry",
        "//checks/utils:scene_rasterizer",
        requirement("numpy"),
    ],
)

py_test(
    name = "rasterization_test",
    srcs = ["rasterization_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:coord_transforms",
        "//checks/utils:rasterization",
        requirement("numpy"),
    ],
)

py_test(
    name = "config_manager_test",
    srcs = ["config_manager_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:config_manager",
        requirement("pyyaml"),
    ],
)

py_test(
    name = "coord_transforms_test",
    srcs = ["coord_transforms_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:coord_transforms",
        "//checks/utils:cuboid",
        requirement("numpy"),
    ],
)

py_test(
    name = "cv2_video_dataset_test",
    srcs = ["cv2_video_dataset_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:cv2_video_dataset",
        requirement("opencv_python_headless"),
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_test(
    name = "video_helper_test",
    srcs = ["video_helper_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:overlay_video_helper",
        "//checks/utils:video",
    ],
)

py_test(
    name = "onnx_test",
    srcs = ["onnx_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:onnx",
        requirement("onnxruntime_gpu"),
    ],
)

py_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:profiler",
    ],
)

py_test(
    name = "rds_data_loader_test",
    srcs = ["rds_data_loader_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:rds_data_loader",
        requirement("numpy"),
    ],
)

py_test(
    name = "segformer_test",
    srcs = ["segformer_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:segformer",
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_test(
    name = "frame_processing_test",
    srcs = ["frame_processing_test.py"],
    data = ["//checks:config"],
    deps = [
        "//checks/utils:config_manager",
        "//checks/utils:frame_processing",
        requirement("numpy"),
    ],
)

py_test(
    name = "wm_bev_render_test",
    srcs = ["wm_bev_render_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils/world_model:bev_render",
        requirement("numpy"),
        requirement("opencv_python_headless"),
    ],
)

py_test(
    name = "multistorage_test",
    srcs = ["multistorage_test.py"],
    deps = [
        "//checks/utils:multistorage",
    ],
)
