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
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "settings_base",
    srcs = ["settings_base.py"],
    visibility = [":__subpackages__"],
    deps = [requirement("pydantic-settings")],
)

# Separate layer for git SHA so it doesn't invalidate the app_layer cache on every commit
pkg_tar(
    name = "git_sha_layer",
    srcs = ["//utils:git_sha"],
    package_dir = "/app",
    tags = ["manual"],
    visibility = [":__subpackages__"],
)

# Pre-downloaded NVIDIA CUDA .deb files (installed via dpkg -i at startup).
# These come from NVIDIA's flat apt repo which rules_distroless cannot resolve.
pkg_tar(
    name = "cuda_debs_layer",
    srcs = [
        "@cuda_cupti_12_8_deb//file",
        "@cusparselt_cuda_12_deb//file",
        "@libcusparselt0_cuda_12_deb//file",
        "@libcusparselt0_dev_cuda_12_deb//file",
        "@libnccl2_deb//file",
        "@libnvshmem3_cuda_12_deb//file",
    ],
    package_dir = "/tmp/cuda-debs",
    tags = ["manual"],
    visibility = [
        "//services/obstacle_correspondence:__pkg__",
    ],
)

# Pre-built LGPL FFmpeg 8.0.1 with NVENC/CUVID hardware acceleration and OpenH264.
# Provides both CLI tools (ffmpeg, ffprobe) and shared libraries (libavutil, libavcodec, libopenh264, etc.)
# as an OCI image layer tar. See third_party/ffmpeg/tar.sh to rebuild.
alias(
    name = "ffmpeg_layer",
    actual = "//third_party/ffmpeg:ffmpeg_lgpl_tar",
    tags = ["manual"],
    visibility = [":__subpackages__"],
)

# Symlink /usr/bin/python3 to the hermetic python3.13 bundled in the py_binary runfiles.
# The Bazel py_binary wrapper uses #!/usr/bin/env python3 as its shebang.
pkg_tar(
    name = "python3_symlink_layer",
    symlinks = {
        "/usr/bin/python3": "/app/rest_api_deployment.runfiles/rules_python++python+python_3_13_x86_64-unknown-linux-gnu/bin/python3",
    },
    tags = ["manual"],
    visibility = [":__subpackages__"],
)

py_binary(
    name = "merge_dpkg_status",
    srcs = ["merge_dpkg_status.py"],
    tags = ["manual"],
    visibility = [":__subpackages__"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = [":__subpackages__"],
    deps = [
        "//utils:bazel",
    ],
)
