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
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "analyze_results",
    srcs = [
        "analyze_results.py",
    ],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:results",
        requirement("matplotlib"),
    ],
)

py_binary(
    name = "prepare_batch_results",
    srcs = [
        "prepare_batch_results.py",
    ],
    tags = ["no-mypy"],
    deps = [
        requirement("boto3"),
        requirement("python-dotenv"),
    ],
)

py_binary(
    name = "export_objects",
    srcs = [
        "export_objects.py",
        "export_viz_helper.py",
    ],
    main = "export_objects.py",
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:cuboid",
        "//checks/utils:cv2_video_dataset",
        "//checks/utils:rds_data_loader",
        requirement("numpy"),
        requirement("opencv_python_headless"),
        requirement("webdataset"),
    ],
)
