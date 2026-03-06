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
    name = "results_test",
    srcs = ["results_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:results",
        requirement("numpy"),
    ],
)

py_test(
    name = "importance_filter_test",
    srcs = ["importance_filter_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:importance_filter",
        "//checks/utils:cuboid",
        requirement("numpy"),
    ],
)

py_test(
    name = "object_processor_base_test",
    srcs = ["object_processor_base_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:object_processor_base",
        requirement("numpy"),
    ],
)

py_test(
    name = "object_processor_dynamic_test",
    srcs = ["object_processor_dynamic_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:object_processor_dynamic",
        requirement("numpy"),
    ],
)

py_test(
    name = "run_test",
    timeout = "long",
    srcs = ["run_test.py"],
    data = [
        "//checks/sample_data:0d21d408_ceca_4af6_9c4b_4f6f78ee7459_1727936028200000_0_mp4",
        "//checks/sample_data:camera_front_wide_120fov_mp4",
        "//checks/sample_data:rds_hq_data_zip",
    ],
    tags = [
        "gpu",
        "no-mypy",
    ],
    deps = [
        "//checks/obstacle:run",
        "//utils:bazel",
    ],
)

py_test(
    name = "seg_helper_test",
    srcs = ["seg_helper_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:seg_helper",
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_test(
    name = "visualization_helper_test",
    srcs = ["visualization_helper_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:visualization_helper",
        requirement("numpy"),
        requirement("opencv_python_headless"),
    ],
)

py_test(
    name = "hallucination_detector_test",
    srcs = ["hallucination_detector_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:hallucination_detector",
        "//checks/obstacle:seg_helper",
        "//checks/utils:cuboid",
        requirement("numpy"),
        requirement("opencv_python_headless"),
    ],
)

py_test(
    name = "hallucination_tracking_test",
    srcs = ["hallucination_tracking_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:hallucination_tracking",
        "//checks/obstacle:seg_helper",
        requirement("numpy"),
        requirement("torch"),
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/obstacle:api",
    ],
)
