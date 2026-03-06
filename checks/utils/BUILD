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
    name = "rasterization",
    srcs = ["rasterization.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/utils:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        requirement("numpy"),
        requirement("numba"),
        requirement("opencv_python_headless"),
    ],
)

py_library(
    name = "rds_geometry",
    srcs = ["rds_geometry.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        "@cosmos_drive_dreams_toolkits//:camera_base",
        requirement("numpy"),
    ],
)

py_library(
    name = "scene_rasterizer",
    srcs = ["scene_rasterizer.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        requirement("numpy"),
    ],
)

py_library(
    name = "cuboid",
    srcs = ["cuboid.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        ":coord_transforms",
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
    ],
)

py_library(
    name = "polyline",
    srcs = ["polyline.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/utils:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        ":coord_transforms",
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
    ],
)

py_library(
    name = "surface",
    srcs = ["surface.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/utils:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        ":coord_transforms",
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
    ],
)

filegroup(
    name = "cityformer_onnx",
    srcs = glob(
        ["citysemsegformer.onnx"],
        allow_empty = True,
    ),
)

py_library(
    name = "coord_transforms",
    srcs = ["coord_transforms.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [requirement("numpy")],
)

py_library(
    name = "cli_common",
    srcs = ["cli_common.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks:__subpackages__",
    ],
    deps = [
        "//checks/utils/world_model:camera_render",
        requirement("python-dotenv"),
    ],
)

py_library(
    name = "config_manager",
    srcs = ["config_manager.py"],
    tags = ["no-mypy"],
    visibility = ["//checks:__subpackages__"],
    deps = [
        "//utils:bazel",
        requirement("pyyaml"),
    ],
)

py_library(
    name = "cv2_video_dataset",
    srcs = ["cv2_video_dataset.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle/tools:__pkg__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        requirement("opencv_python_headless"),
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_library(
    name = "onnx",
    srcs = ["onnx.py"],
    tags = ["no-mypy"],
    visibility = ["//checks/utils/tests:__pkg__"],
    deps = [
        requirement("onnxruntime_gpu"),
        requirement("torch"),
    ],
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
)

py_library(
    name = "rds_data_loader",
    srcs = ["rds_data_loader.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
        "//checks/utils/world_model:__pkg__",
        "//checks/vlm:__subpackages__",
    ],
    deps = [
        ":coord_transforms",
        ":cuboid",
        ":polyline",
        ":surface",
        "@cosmos_drive_dreams_toolkits//:camera_ftheta",
        requirement("webdataset"),
    ],
)

py_library(
    name = "segformer",
    srcs = ["segformer.py"],
    data = [":cityformer_onnx"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils:__subpackages__",
    ],
    deps = [
        ":cv2_video_dataset",
        ":onnx",
        "//utils:bazel",
        requirement("numpy"),
        requirement("opencv_python_headless"),
        requirement("torch"),
        requirement("torchvision"),
    ],
)

py_library(
    name = "video",
    srcs = ["video.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
        "//checks/vlm:__subpackages__",
    ],
    deps = [
        requirement("opencv_python_headless"),
        requirement("numpy"),
    ],
)

py_library(
    name = "overlay_video_helper",
    srcs = ["overlay_video_helper.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/obstacle:__subpackages__",
        "//checks/utils/tests:__pkg__",
    ],
    deps = [
        ":video",
        requirement("opencv_python_headless"),
        requirement("numpy"),
    ],
)

py_library(
    name = "frame_processing",
    srcs = ["frame_processing.py"],
    visibility = [
        "//checks:__subpackages__",
    ],
    deps = [
        ":types",
        requirement("numpy"),
        requirement("opencv_python_headless"),
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    visibility = [
        "//checks:__subpackages__",
        "//services/hallucination:__pkg__",
    ],
    deps = [requirement("pydantic")],
)

py_library(
    name = "multistorage",
    srcs = ["multistorage.py"],
    visibility = [
        "//checks:__subpackages__",
        "//services:__subpackages__",
    ],
    deps = [
        requirement("multi-storage-client"),
        requirement("boto3"),
    ],
)
