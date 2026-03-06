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

"""Module extension for http_archive with dynamic netrc path support.

Uses a custom repository rule to resolve ~ at fetch time (not module resolution time),
making the extension reproducible and excluding it from MODULE.bazel.lock.
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "read_netrc", "use_netrc")

def _resolve_netrc_path(netrc, repository_ctx):
    """Resolve netrc path at fetch time, expanding ~ to home directory."""
    home = repository_ctx.os.environ.get("HOME", "")
    if not home:
        fail("HOME environment variable is not set")

    if not netrc:
        return "{}/.netrc".format(home)  # Default to $HOME/.netrc

    return netrc.replace("~", home)  # Replace ~ with home directory

def _http_archive_with_netrc_impl(repository_ctx):
    """Repository rule that resolves netrc paths at fetch time."""

    netrc_path = _resolve_netrc_path(repository_ctx.attr.netrc, repository_ctx)

    auth = {}
    if netrc_path:
        netrc_content = read_netrc(repository_ctx, netrc_path)
        auth = use_netrc(netrc_content, [repository_ctx.attr.url], repository_ctx.attr.auth_patterns)

    output_dir = repository_ctx.attr.add_prefix if repository_ctx.attr.add_prefix else ""
    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url,
        output = output_dir,
        integrity = repository_ctx.attr.integrity if repository_ctx.attr.integrity else None,
        stripPrefix = repository_ctx.attr.strip_prefix,
        auth = auth,
    )

    if repository_ctx.attr.build_file:
        repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")

_http_archive_with_netrc = repository_rule(
    implementation = _http_archive_with_netrc_impl,
    attrs = {
        "url": attr.string(mandatory = True),
        "integrity": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
        "add_prefix": attr.string(default = ""),
        "build_file": attr.label(allow_single_file = True),
        "netrc": attr.string(default = "", doc = "Path to netrc file. Supports ~/ for home directory."),
        "auth_patterns": attr.string_dict(default = {}),
    },
    environ = ["HOME"],  # Declare env dependency for proper cache invalidation
)

def _http_archive_impl(module_ctx):
    """Implementation of the http_archive module extension."""
    for mod in module_ctx.modules:
        for archive in mod.tags.archive:
            # Pass through config - netrc resolution happens in repository rule
            _http_archive_with_netrc(
                name = archive.name,
                url = archive.url,
                integrity = archive.integrity,
                strip_prefix = archive.strip_prefix,
                add_prefix = archive.add_prefix,
                build_file = archive.build_file,
                netrc = archive.netrc,
                auth_patterns = archive.auth_patterns,
            )

    return module_ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

_archive_tag = tag_class(
    attrs = {
        "name": attr.string(mandatory = True),
        "add_prefix": attr.string(default = ""),
        "auth_patterns": attr.string_dict(default = {}),
        "build_file": attr.string(default = ""),
        "integrity": attr.string(default = ""),
        "netrc": attr.string(default = "", doc = "Path to netrc file. Supports ~/ for home directory."),
        "strip_prefix": attr.string(default = ""),
        "url": attr.string(mandatory = True),
    },
)

http_archive = module_extension(
    implementation = _http_archive_impl,
    tag_classes = {"archive": _archive_tag},
)
