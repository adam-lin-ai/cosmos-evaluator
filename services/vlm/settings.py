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
Configuration module for the VLM service.

This module provides environment-specific configuration management.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from services.settings_base import SettingsBase


class Settings(SettingsBase):
    """Settings for the VLM service."""

    lepton_api_key: SecretStr | None = None
    nvcf_ndss_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(strict=True, extra="ignore", env_prefix="VLM_", case_sensitive=False, frozen=True)


@lru_cache()
def get_settings() -> Settings:
    """Get the settings instance."""
    return Settings(_env_file=SettingsBase.get_env_files())
