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
Common CLI utilities for cosmos_evaluator checks.

This module provides reusable functions for command-line interface operations
that are shared across multiple check modules (obstacle, vlm, etc.).
"""

from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


def load_default_envs(additional_env_paths: Optional[List[str]] = None) -> None:
    """
    Loads the default environment variables for AWS credentials and optionally
    additional environment files.

    This function loads AWS credentials from ~/.aws/.env and any additional
    environment files specified.

    Args:
        additional_env_paths: Optional list of additional environment file paths
            to load (e.g., ["~/.cosmos_evaluator/.env"] for VLM API keys)

    Raises:
        FileNotFoundError: If any required environment file is not found
    """
    # Load AWS credentials (always required)
    aws_env = Path("~/.aws/.env").expanduser()
    if not aws_env.exists():
        raise FileNotFoundError(f"AWS credentials file not found at: {aws_env}")
    load_dotenv(aws_env, override=False)

    # Load additional environment files if specified
    if additional_env_paths:
        for env_path_str in additional_env_paths:
            env_path = Path(env_path_str).expanduser()
            if not env_path.exists():
                raise FileNotFoundError(f"Environment file not found at: {env_path}")
            load_dotenv(env_path, override=False)


def validate_paths(
    input_data: str,
    video_path: str,
    output_dir: str,
    create_output_dir: bool = True,
) -> bool:
    """
    Validate that input paths exist and optionally create output directory.

    This function works for both local and cloud modes - it validates the
    actual file paths regardless of how they were obtained (CLI args or S3 download).

    Args:
        input_data: Path to input data directory (e.g., RDS HQ dataset)
        video_path: Path to video file
        output_dir: Path to output directory
        create_output_dir: Whether to create output directory if it doesn't exist

    Returns:
        True if validation passes, False otherwise
    """
    # Check if input data directory exists
    if not Path(input_data).exists():
        print(f"Error: Input data directory does not exist: {input_data}")
        return False

    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Error: Video file does not exist: {video_path}")
        return False

    # Create output directory if requested
    if create_output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return True
