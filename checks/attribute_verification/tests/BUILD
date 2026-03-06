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
    ],
    deps = [
        "//checks/attribute_verification:processor",
        requirement("numpy"),
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    data = [
        "//checks/attribute_verification/tests/data:config.yaml",
        "//checks/sample_data:augmented_mp4",
    ],
    deps = [
        "//checks/attribute_verification:processor",
        "//checks/attribute_verification:run",
        requirement("pyyaml"),
    ],
)

py_test(
    name = "common_test",
    srcs = ["common_test.py"],
    deps = [
        "//checks/attribute_verification:common",
    ],
)

py_test(
    name = "question_generator_test",
    srcs = ["question_generator_test.py"],
    deps = [
        "//checks/attribute_verification:question_generator",
    ],
)

py_test(
    name = "vlm_verifier_test",
    srcs = ["vlm_verifier_test.py"],
    deps = [
        "//checks/attribute_verification:vlm_verifier",
    ],
)
