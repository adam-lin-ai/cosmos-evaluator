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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Optional, Protocol, Self, runtime_checkable


@dataclass
class StorageUrls:
    """Cloud-agnostic container for storage URLs."""

    raw: str | None = None
    presigned: str | None = None


@runtime_checkable
class StorageProvider(Protocol):
    """Protocol for storage operations."""

    async def store(self, data: Any, key: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Stores data.

        Args:
            data: Data to store (bytes, string, or serializable object)
            key: Storage key/path
            metadata: Optional metadata to attach

        Returns:
            Storage reference/URL for retrieval
        """
        ...

    async def retrieve(self, key: str) -> Any:
        """Retrieves data.

        Args:
            key: Storage key/path

        Returns:
            Retrieved data
        """
        ...

    async def delete(self, key: str) -> bool:
        """Deletes data.

        Args:
            key: Storage key/path

        Returns:
            True if deletion successful
        """
        ...

    async def exists(self, key: str) -> bool:
        """Checks if data exists.

        Args:
            key: Storage key/path

        Returns:
            True if data exists
        """
        ...

    def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        """Lists storage keys with optional prefix.

        Args:
            prefix: Optional prefix filter

        Yields:
            Storage keys matching the prefix
        """
        ...

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> StorageUrls:
        """Stores a file from disk and returns both raw and temporary access URLs.

        Args:
            file_path: Local path to the file to upload
            key: Storage key/path
            content_type: MIME content type (auto-detected if not provided)
            metadata: Optional metadata to attach

        Returns:
            StorageUrls with raw and presigned/temporary access URLs
        """
        ...

    async def generate_presigned_url(self, key: str, operation: str = "get_object", expiration: int = 3600) -> str:
        """Generates a temporary access URL for a stored object.

        Args:
            key: Storage key/path
            operation: Storage operation (e.g. get_object, put_object)
            expiration: URL expiration time in seconds

        Returns:
            Temporary access URL
        """
        ...

    async def download_from_url(self, url: str, destination: Path) -> Path:
        """Downloads a file from a URL (presigned or raw) to a local path.

        HTTP(S) URLs are downloaded directly, making this compatible with
        presigned URLs from any cloud provider.

        Args:
            url: HTTP(S) URL or provider-specific URI
            destination: Local file path to write to

        Returns:
            Path to the downloaded file
        """
        ...

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        ...
