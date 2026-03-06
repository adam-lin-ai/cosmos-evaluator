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
    name = "processor_test",
    srcs = ["processor_test.py"],
    data = [
        "//checks/sample_data:augmented_mp4",
        "//checks/sample_data:original_mp4",
    ],
    deps = [
        "//checks/hallucination:processor",
    ],
)

py_test(
    name = "frame_processing_test",
    srcs = ["frame_processing_test.py"],
    deps = [
        "//checks/hallucination:frame_processing",
        requirement("numpy"),
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    data = [
        "//checks/hallucination/tests/data:config.yaml",
        "//checks/sample_data:augmented_mp4",
        "//checks/sample_data:original_mp4",
    ],
    deps = [
        "//checks/hallucination:run",
        requirement("pyyaml"),
    ],
)
