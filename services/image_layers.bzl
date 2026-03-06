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

"""Macro to split a py_binary into separate OCI image layers for pip dependencies and application source code.

Splitting the app layer improves build and container load times by separating large, stable pip
packages (~6-7 GiB for services with PyTorch/CUDA) from small, frequently changing application
source code (~100-300 MB). On a typical code-only change, only the small source layer is rebuilt.
"""

load("@aspect_bazel_lib//lib:tar.bzl", "mtree_mutate", "mtree_spec", "tar")

def py_image_layers(name, binary, package_dir, owner, ownername, tags, model_file_pattern = None):
    """Split a py_binary's runfiles into separate layers for pip dependencies and application source.

    Produces tar layer targets suitable for use in oci_image(tars = [...]):
      {name}_pip_deps_layer: Pip packages (site-packages), large and rarely changing
      {name}_source_layer: Application source code and non-pip deps, small and frequently changing
      {name}_model_layer: (only when model_file_pattern is set) Model files matching the pattern

    When model_file_pattern is provided, files matching the pattern are excluded from source_layer
    and placed into model_layer instead. This enables building image variants with or without model
    weights while sharing all other layers.

    Args:
        name: Base name for generated targets.
        binary: Label of the py_binary to split.
        package_dir: Directory prefix in the tar archive.
        owner: UID for file ownership.
        ownername: Username for file ownership.
        tags: Bazel tags for generated targets.
        model_file_pattern: Optional grep pattern for model files to split into a separate layer.
            When set, matching files are excluded from source_layer and placed in model_layer.
            Example: "model\\.safetensors" to separate model weights.
    """

    mtree_spec(
        name = name + "_mtree",
        srcs = [binary],
    )

    native.genrule(
        name = name + "_pip_mtree_filter",
        srcs = [name + "_mtree"],
        outs = [name + "_pip_deps.mtree"],
        cmd = "grep 'site-packages' $< > $@ || true",
        tags = tags,
    )

    native.genrule(
        name = name + "_app_mtree_filter",
        srcs = [name + "_mtree"],
        outs = [name + "_app_source.mtree"],
        cmd = "grep -v 'site-packages' $< > $@ || true",
        tags = tags,
    )

    mtree_mutate(
        name = name + "_pip_mtree",
        mtree = name + "_pip_mtree_filter",
        strip_prefix = native.package_name(),
        owner = owner,
        ownername = ownername,
        package_dir = package_dir,
        tags = tags,
    )

    mtree_mutate(
        name = name + "_app_mtree",
        mtree = name + "_app_mtree_filter",
        strip_prefix = native.package_name(),
        owner = owner,
        ownername = ownername,
        package_dir = package_dir,
        tags = tags,
    )

    tar(
        name = name + "_pip_deps_layer",
        srcs = [binary],
        mtree = name + "_pip_mtree",
        tags = tags,
    )

    if model_file_pattern:
        native.genrule(
            name = name + "_source_mtree_filter",
            srcs = [name + "_app_mtree"],
            outs = [name + "_source.mtree"],
            cmd = "grep -v '{}' $< > $@ || true".format(model_file_pattern),
            tags = tags,
        )

        native.genrule(
            name = name + "_model_mtree_filter",
            srcs = [name + "_app_mtree"],
            outs = [name + "_model.mtree"],
            cmd = "grep '{}' $< > $@ || true".format(model_file_pattern),
            tags = tags,
        )

        tar(
            name = name + "_source_layer",
            srcs = [binary],
            mtree = name + "_source_mtree_filter",
            tags = tags,
        )

        tar(
            name = name + "_model_layer",
            srcs = [binary],
            mtree = name + "_model_mtree_filter",
            tags = tags,
        )
    else:
        tar(
            name = name + "_source_layer",
            srcs = [binary],
            mtree = name + "_app_mtree",
            tags = tags,
        )
