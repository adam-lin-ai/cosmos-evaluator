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

"""
Module for setting up the multistorage client configuration.
"""

import json
import logging
import os
import tempfile

import multistorageclient as msc

logger = logging.getLogger(__name__)
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def setup_msc_config(config: str | None = None) -> None:
    """
    Write MSC config to a temp file and set MSC_CONFIG env var.

    This allows msc.open(), msc.is_file(), etc. to work directly with
    any URL (s3://, msc://, local paths) using path_mapping from config.
    """
    if config is None or config == "":
        logger.warning("Multistorage client configuration is not set")
        return

    # If the config string is wrapped in single quotes, remove them
    if isinstance(config, str) and len(config) >= 2 and config.startswith("'") and config.endswith("'"):
        config = config[1:-1]

    try:
        config = json.loads(config)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in multistorage config: {e}")
        raise ValueError(f"Invalid multistorage configuration: {e}") from e

    # Write to a persistent temp file (not auto-deleted)
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="msc_config_")
    os.close(fd)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    os.environ["MSC_CONFIG"] = config_path
    logger.debug(f"MSC config written to: {config_path}")


def is_remote_path(path: str) -> bool:
    """
    Check if the path is a remote path.
    """
    return path.startswith(("s3://", "gs://", "azure://", "msc://", "https://"))


def download_if_remote(path: str) -> str:
    """
    Download video to temporary file if it's a remote path.

    Args:
        path: Video path (local or remote)

    Returns:
        Local path to video file
    """
    # Check if path is remote (starts with s3://, gs://, etc.)
    if is_remote_path(path):
        logger.debug(f"Downloading remote video: {path}")
        # Create temp file
        suffix = os.path.splitext(path)[1] or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()

        # Download using multistorageclient
        with msc.open(path, "rb") as src:
            with open(temp_path, "wb") as dst:
                while True:
                    chunk = src.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)

        logger.debug(f"Downloaded to: {temp_path}")
        return temp_path
    else:
        logger.debug(f"Path is local: {path}")
        return path


def validate_uri(path: str, is_file: bool = True) -> bool:
    """
    Validate if the path is a valid URI.
    """
    try:
        if is_file:
            return msc.is_file(path)
        else:
            return not msc.is_empty(path)
    except Exception:
        logger.exception(f"Error validating URI {path}")
        return False
