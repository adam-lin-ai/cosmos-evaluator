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
load("@rules_distroless//distroless:defs.bzl", "flatten")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pex//pex:defs.bzl", "py_pex_binary")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load(":version.bzl", "VERSION")

# ======================================
# Application packages
# ======================================

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//services/hallucination/tests:__pkg__"],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = [
        "//services/hallucination:__pkg__",
        "//services/hallucination/tests:__pkg__",
    ],
    deps = [
        "//services:settings_base",
        requirement("pydantic-settings"),
    ],
)

py_library(
    name = "hallucination_service",
    srcs = ["hallucination_service.py"],
    visibility = ["//services/hallucination/tests:__pkg__"],
    deps = [
        "//checks/hallucination:processor",
        "//checks/utils:types",
        "//services/framework:service_base",
        requirement("pydantic"),
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = [
        ":generate_version_txt",
        "//utils:git_sha",
    ],
    main = "rest_api.py",
    visibility = ["//services/hallucination/tests:__pkg__"],
    deps = [
        ":hallucination_service",
        ":settings",
        "//checks/utils:multistorage",
        "//services:settings_base",
        "//services:utils",
        "//services/framework/response_formatters:json_response_formatter",
        requirement("pydantic"),
        requirement("pydantic-settings"),
        requirement("fastapi"),
        requirement("uvicorn"),
    ],
)

# Package the Rest API code and its dependency tree into a single executable for deployment
py_pex_binary(
    name = "rest_api_pex",
    binary = ":rest_api",
)

# ======================================
# Container Image Build
# ======================================

# Merge overlay package metadata into the base image's /var/lib/dpkg/status.
# This ensures dpkg -l and security scanners (e.g. nSpect) see all packages —
# both from the base image and from packages.yaml — in a single database.
genrule(
    name = "dpkg_status_merged",
    srcs = [
        "@nvidia_distroless_python_313_image_linux_amd64",
        "@hallucination_packages//:dpkg_status",
    ],
    outs = ["dpkg_status_merged.tar"],
    cmd = "$(execpath //services:merge_dpkg_status) $(location @nvidia_distroless_python_313_image_linux_amd64//:nvidia_distroless_python_313_image_linux_amd64) $(location @hallucination_packages//:dpkg_status) $@",
    tags = ["manual"],
    tools = ["//services:merge_dpkg_status"],
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@hallucination_packages//:packages",
        ":dpkg_status_merged",
    ],
)

# License layer adds the third party notices and license files to the container image
pkg_tar(
    name = "license_layer",
    srcs = ["//services/hallucination:THIRD-PARTY-NOTICES.txt"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app",
    tags = ["manual"],
)

# App layer adds a python executable with all python code and dependencies to the container image
pkg_tar(
    name = "app_layer",
    srcs = [":rest_api_pex"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app",
    tags = ["manual"],
)

# Configuration layer adds the configuration files to the container image
pkg_tar(
    name = "config_layer",
    srcs = ["//checks:config"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app/config",
    tags = ["manual"],
)

oci_image(
    name = "image",
    base = "@nvidia_distroless_python_313_image",
    entrypoint = [
        "/usr/local/bin/python3.13",
        "/app/rest_api_pex.pex",
    ],
    env = {
        "CONFIG_DIR": "/app/config",
    },
    exposed_ports = ["8000"],
    labels = "//utils:git_sha",
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        ":license_layer",
        ":app_layer",
        ":config_layer",
    ],
    workdir = "/app",
)

# ======================================
# Container Image Publishing and Testing
# ======================================

# Load the container image for testing
oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["hallucination-checker:" + VERSION],
    tags = ["manual"],
)
