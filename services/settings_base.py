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
Configuration module for the services.

This module provides environment-specific configuration management.
"""

from enum import Enum
import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Environment enumeration."""

    LOCAL = "local"
    PROD = "production"
    STAGING = "staging"


class LogLevel(str, Enum):
    """Log level enumeration (uppercase to match logging module)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SettingsBase(BaseSettings):
    """Base settings for the services."""

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="COSMOS_EVALUATOR_", case_sensitive=False, frozen=True
    )

    # Use validation_alias to ensure all services read from COSMOS_EVALUATOR_ENV regardless of the child class's env_prefix
    env: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_ENV", "env"),
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_LOG_LEVEL", "log_level"),
    )

    # Storage configuration (cloud-agnostic)
    storage_type: str = Field(
        default="s3",
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_TYPE", "storage_type"),
    )

    storage_bucket: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_BUCKET", "storage_bucket"),
    )

    storage_region: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_REGION", "AWS_DEFAULT_REGION", "storage_region"),
    )

    storage_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "storage_access_key"),
    )

    storage_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY", "AWS_SECRET_ACCESS_KEY", "storage_secret_key"
        ),
    )

    @staticmethod
    def get_env_files() -> list[Path]:
        """Gets the environment files."""
        env_str = os.environ.get("COSMOS_EVALUATOR_ENV", Environment.LOCAL.value)
        if Environment(env_str) == Environment.LOCAL:
            return [Path("~/.cosmos_evaluator/.env").expanduser(), Path("~/.aws/.env").expanduser()]

        return []
