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

"""Utility functions for services."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.bazel import get_runfiles_path

logger = logging.getLogger(__name__)


def extract_clip_id(rds_hq_dir_path: Path) -> str:
    """Extract clip ID from the tar files in the extracted RDS HQ directory.

    Args:
        rds_hq_dir_path: Path to the extracted RDS HQ directory

    Returns:
        Clip ID
    """
    # Look for .tar files in any subdirectory
    tar_files = list(rds_hq_dir_path.rglob("*.tar"))

    if not tar_files:
        raise ValueError(f"No .tar files found in extracted RDS HQ directory: {rds_hq_dir_path}")

    # Extract clip ID from the first .tar file name
    # Format: 1408ad50-5d9c-463d-a6c9-9e042c7cbcb3_17280960000.tar
    clip_id = tar_files[0].stem  # Remove .tar extension

    logger.info(f"Extracted clip_id: {clip_id}")
    return clip_id


def get_contents_from_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file

    Returns:
        Contents of the file

    Raises:
        ValueError: If the file is empty
    """
    with open(file_path, "r") as file:
        contents = file.read().strip()

    if not contents:
        raise ValueError(f"File is empty: {file_path}")

    return contents


def get_contents_from_runfile(repo_path: str) -> str:
    """Read the contents of a file in the Bazel runfiles directory.

    Args:
        repo_path: Path to the file in the repository (e.g., "services/obstacle_correspondence/version.txt")

    Returns:
        Contents of the file

    Raises:
        FileNotFoundError: If the file is not found
    """
    file_path = get_runfiles_path(repo_path)
    if file_path:
        return get_contents_from_file(file_path)

    # PEX/container execution often has no Bazel runfiles manifest. In that case,
    # resolve repo-relative paths from the code root (works for bundled data files).
    repo_root = Path(__file__).resolve().parents[1]
    fallback_path = repo_root / repo_path
    if fallback_path.exists():
        return get_contents_from_file(str(fallback_path))

    raise FileNotFoundError(f"File not found: {repo_path}")


def get_git_sha() -> str:
    """Get the current Git commit SHA.

    Tries multiple sources in order:
    1. Container path (/app/git_sha.txt) - available in Docker containers via git_sha_layer
    2. Bazel runfiles - available when //utils:git_sha is in data deps (local dev, tests)

    Returns:
        Git SHA
    """
    # Container path (available for :rest_api_deployment via git_sha_layer)
    container_path = "/app/git_sha.txt"
    if Path(container_path).exists():
        return get_contents_from_file(container_path).split("=")[1].strip()
    else:
        # Bazel runfiles (available for :rest_api which has //utils:git_sha in data)
        return get_contents_from_runfile("utils/git_sha.txt").split("=")[1].strip()
