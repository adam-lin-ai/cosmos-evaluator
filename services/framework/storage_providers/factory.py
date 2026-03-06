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

"""Factory for creating StorageProvider instances from application settings."""

from __future__ import annotations

from services.framework.protocols.storage_provider import StorageProvider
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider
from services.settings_base import SettingsBase


def build_storage_provider(settings: SettingsBase, key_prefix: str = "") -> StorageProvider:
    """Creates a StorageProvider from application settings.

    Dispatches on ``settings.storage_type`` to build the appropriate provider.

    Args:
        settings: Application settings containing storage configuration
        key_prefix: Key prefix prepended to all storage keys (for cloud
            providers) or used as the base directory (for local storage)

    Returns:
        A configured StorageProvider instance

    Raises:
        ValueError: If the configured storage type is not supported
    """
    if settings.storage_type == "local":
        return LocalStorageProvider(base_path=key_prefix)

    if settings.storage_type == "s3":
        if not settings.storage_bucket:
            raise ValueError("COSMOS_EVALUATOR_STORAGE_BUCKET must be set for S3 storage")
        if not settings.storage_region:
            raise ValueError("COSMOS_EVALUATOR_STORAGE_REGION must be set for S3 storage")
        config = S3Config(
            bucket_name=settings.storage_bucket,
            region_name=settings.storage_region,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            key_prefix=key_prefix,
        )
        return S3StorageProvider(config=config)

    raise ValueError(f"Unsupported storage type: {settings.storage_type}")
