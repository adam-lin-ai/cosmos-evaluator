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
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//services:image_layers.bzl", "py_image_layers")
load(":version.bzl", "VERSION")

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = [
        "//services/obstacle_correspondence/tests:__pkg__",
    ],
    deps = [requirement("pydantic")],
)

py_library(
    name = "obstacle_correspondence_service",
    srcs = ["obstacle_correspondence_service.py"],
    visibility = ["//services/obstacle_correspondence/tests:__pkg__"],
    deps = [
        ":models",
        "//checks/obstacle:api",
        "//services/framework:service_base",
    ],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = [
        "//services/obstacle_correspondence:__pkg__",
        "//services/obstacle_correspondence/tests:__pkg__",
    ],
    deps = [
        "//services:settings_base",
        requirement("pydantic-settings"),
    ],
)

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//services/obstacle_correspondence/tests:__pkg__"],
)

py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    visibility = ["//services/obstacle_correspondence/tests:__pkg__"],
    deps = [
        ":models",
        ":obstacle_correspondence_service",
        ":settings",
        "//services:settings_base",
        "//services:utils",
        "//services/framework/protocols:storage_provider",
        "//services/framework/request_handlers:json_request_handler",
        "//services/framework/response_formatters:json_response_formatter",
        "//services/framework/response_formatters:streaming_response_formatter",
        "//services/framework/storage_providers:factory",
        requirement("fastapi"),
        requirement("uvicorn"),
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = ["//utils:git_sha"],
    visibility = ["//services/obstacle_correspondence/tests:__pkg__"],
    deps = [":rest_api_common"],
)

# //utils:git_sha excluded from the deployed rest_api to avoid invalidating the cache for app_layer on every commit
py_binary(
    name = "rest_api_deployment",
    srcs = ["rest_api.py"],
    main = "rest_api.py",
    tags = ["manual"],
    deps = [":rest_api_common"],
)

# Create a startup script that installs packages and starts the app:
# - System packages are pre-installed via the @obstacle_correspondence_packages deb layer
# - NVIDIA CUDA .deb files are pre-downloaded at build time and installed via dpkg -i
#   to avoid apt-get update/install network overhead at container startup
genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

dpkg -i /tmp/cuda-debs/*.deb

# Create symlinks that are normally set up by postinst scripts (via update-alternatives)
# but aren't created by rules_distroless since it only extracts package files.
ln -sf /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 /usr/lib/x86_64-linux-gnu/libblas.so.3
ln -sf /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3

# Register libnvshmem (installed to non-standard path) with the dynamic linker.
echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem-cuda-12.conf
ldconfig

# Remove CUDA compat libraries to disable forward compatibility when host system
# has an older version of CUDA installed because this container already pins the
# CUDA version to 12.8.1
if ls /usr/local/cuda*/compat 1> /dev/null 2>&1; then rm -rf /usr/local/cuda*/compat; fi

# Create non-root user for running the application
useradd --create-home --shell /bin/bash --uid 1001 appuser

# Ensure /tmp and any bind-mounted subdirectories are writable by appuser
chmod 1777 /tmp
chown -R appuser:appuser /tmp

# Switch to non-root user and run the application
cd /app
exec su appuser -c "./rest_api_deployment --host $${UVICORN_HOST:-0.0.0.0} --port $${UVICORN_PORT:-8000}"
EOF
    """,
)

pkg_tar(
    name = "startup_layer",
    srcs = [":startup_script"],
    mode = "0755",
    package_dir = "/usr/local/bin",
    tags = ["manual"],
)

# Split the Python binary into separate pip deps, app source, and model layers.
# Pip packages (~6-7 GiB, stable) are separated from app source (~100-300 MB, changes frequently)
# to improve build cache hit rates and docker load times.
py_image_layers(
    name = "app",
    binary = ":rest_api_deployment",
    owner = "1001",
    ownername = "appuser",
    package_dir = "/app",
    tags = ["manual"],
)

# Merge overlay package metadata into the base image's /var/lib/dpkg/status.
# This ensures dpkg -l and security scanners (e.g. nSpect) see all packages —
# both from the base image and from packages.yaml — in a single database.
genrule(
    name = "dpkg_status_merged",
    srcs = [
        "@cuda_cudnn_image_linux_amd64",
        "@obstacle_correspondence_packages//:dpkg_status",
    ],
    outs = ["dpkg_status_merged.tar"],
    cmd = "$(execpath //services:merge_dpkg_status) $(location @cuda_cudnn_image_linux_amd64//:cuda_cudnn_image_linux_amd64) $(location @obstacle_correspondence_packages//:dpkg_status) $@",
    tags = ["manual"],
    tools = ["//services:merge_dpkg_status"],
)

pkg_tar(
    name = "third_party_attributions_layer",
    srcs = ["//services/obstacle_correspondence:THIRD-PARTY-NOTICES.txt"],
    package_dir = "/app",
    tags = ["manual"],
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@obstacle_correspondence_packages//:packages",
        ":dpkg_status_merged",
    ],
)

oci_image(
    name = "image",
    base = "@cuda_cudnn_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_COMPAT_DISABLED": "1",
    },
    exposed_ports = ["8000"],
    labels = "//utils:git_sha",
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        "//services:cuda_debs_layer",
        "//services:ffmpeg_layer",
        "//services:python3_symlink_layer",
        ":startup_layer",
        ":app_pip_deps_layer",
        ":third_party_attributions_layer",
        ":app_source_layer",
        "//services:git_sha_layer",
    ],
    workdir = "/app",
)

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["obstacle-correspondence:" + VERSION],
    tags = ["manual"],
    visibility = ["//services/obstacle_correspondence/tests:__pkg__"],
)
