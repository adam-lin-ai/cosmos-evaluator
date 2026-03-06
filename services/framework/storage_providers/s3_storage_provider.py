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

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
import ssl
from types import TracebackType
from typing import Any, AsyncIterator, NoReturn
from urllib.parse import urljoin

import aioboto3
import aiohttp
from botocore.config import Config as BotoCoreConfig
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import certifi

from services.framework.protocols.storage_provider import StorageUrls
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.exceptions import (
    S3StorageError,
    StorageAuthenticationError,
    StorageConnectionError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)
from services.framework.storage_providers.s3_url_utils import parse_s3_url
from services.framework.url_security import check_url_security
from services.settings_base import SettingsBase

logger = logging.getLogger(__name__)


class S3StorageProvider:
    """Amazon S3 storage provider implementation.

    Implements the StorageProvider protocol for Amazon S3 and S3-compatible storage services.
    Supports async operations, multipart uploads, server-side encryption, and comprehensive
    error handling.
    """

    DEFAULT_PRESIGNED_URL_EXPIRATION = 43200  # 12 hours

    def __init__(self, config: S3Config) -> None:
        """Initializes S3 storage provider.

        Args:
            config: S3 configuration object
        """
        self.config = config
        self._session: aioboto3.Session | None = None
        self._client: Any = None
        self._client_ctx: Any = None
        self._client_lock = asyncio.Lock()

        # Setup boto3 configuration
        self._boto_config = BotoCoreConfig(
            retries={"max_attempts": config.max_attempts, "mode": config.retry_mode},
            max_pool_connections=config.max_concurrency,
        )

    @classmethod
    def from_settings(cls, bucket_name: str) -> S3StorageProvider:
        """Creates an S3StorageProvider from SettingsBase environment configuration.

        Args:
            bucket_name: S3 bucket name

        Returns:
            Configured S3StorageProvider instance
        """
        settings = SettingsBase(_env_file=SettingsBase.get_env_files())
        config = S3Config(
            bucket_name=bucket_name,
            aws_access_key_id=settings.storage_access_key,
            aws_secret_access_key=settings.storage_secret_key,
            region_name=settings.storage_region or "us-east-1",
        )
        return cls(config)

    async def _get_client(self) -> Any:
        """Gets or creates S3 client."""
        async with self._client_lock:
            if self._client is None:
                if self._session is None:
                    self._session = aioboto3.Session()

                boto3_config = self.config.get_boto3_config()
                self._client_ctx = self._session.client("s3", config=self._boto_config, **boto3_config)
                self._client = await self._client_ctx.__aenter__()
        return self._client

    def _full_key(self, key: str) -> str:
        """Prepends the configured key_prefix to a storage key.

        Args:
            key: Storage key/path

        Returns:
            Key with prefix prepended
        """
        return f"{self.config.key_prefix}{key}"

    def _handle_s3_error(self, error: Exception, operation: str, key: str) -> NoReturn:
        """Handles and converts S3 errors to storage exceptions.

        Args:
            error: Original exception
            operation: Operation being performed
            key: Storage key involved

        Raises:
            Appropriate StorageError subclass
        """
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]

            if error_code == "NoSuchBucket":
                raise S3StorageError(
                    f"Bucket '{self.config.bucket_name}' does not exist",
                    key=key,
                    operation=operation,
                    bucket=self.config.bucket_name,
                    aws_error_code=error_code,
                )
            elif error_code == "NoSuchKey":
                raise StorageNotFoundError(
                    f"Object '{key}' not found in bucket '{self.config.bucket_name}'", key=key, operation=operation
                )
            elif error_code in ["AccessDenied", "Forbidden"]:
                raise StoragePermissionError(
                    f"Access denied for operation '{operation}' on key '{key}': {error_message}",
                    key=key,
                    operation=operation,
                )
            elif error_code in ["QuotaExceeded", "ServiceQuotaExceeded"]:
                raise StorageQuotaExceededError(
                    f"Storage quota exceeded: {error_message}", key=key, operation=operation
                )
            else:
                raise S3StorageError(
                    f"S3 error during {operation}: {error_message}",
                    key=key,
                    operation=operation,
                    bucket=self.config.bucket_name,
                    aws_error_code=error_code,
                    details=error.response,
                )
        elif isinstance(error, NoCredentialsError):
            raise StorageAuthenticationError("AWS credentials not found or invalid", key=key, operation=operation)
        elif isinstance(error, BotoCoreError):
            raise StorageConnectionError(
                f"Connection error during {operation}: {str(error)}", key=key, operation=operation
            )
        else:
            raise S3StorageError(
                f"Unexpected error during {operation}: {str(error)}",
                key=key,
                operation=operation,
                bucket=self.config.bucket_name,
            )

    def _prepare_data_for_storage(self, data: Any) -> bytes:
        """Prepares data for storage by converting to bytes.

        Args:
            data: Data to prepare

        Returns:
            Data as bytes

        Raises:
            S3StorageError: If data cannot be serialized
        """
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode("utf-8")
        elif isinstance(data, (dict, list)):
            try:
                return json.dumps(data, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise S3StorageError(f"Failed to serialize data to JSON: {e}") from e
        else:
            try:
                return str(data).encode("utf-8")
            except Exception as e:
                raise S3StorageError(f"Failed to convert data to bytes: {e}") from e

    def _prepare_metadata(self, metadata: dict[str, Any] | None) -> dict[str, str]:
        """Prepares metadata for S3 storage.

        Args:
            metadata: Original metadata

        Returns:
            S3-compatible metadata (string values only)
        """
        if not metadata:
            return {}

        s3_metadata = {}
        for key, value in metadata.items():
            # S3 metadata keys must be lowercase and contain only letters, numbers, and hyphens
            clean_key = key.lower().replace("_", "-")
            if isinstance(value, str):
                s3_metadata[clean_key] = value
            else:
                try:
                    s3_metadata[clean_key] = json.dumps(value)
                except (TypeError, ValueError):
                    s3_metadata[clean_key] = str(value)

        return s3_metadata

    def _determine_content_type(self, data: Any, key: str) -> str:
        """Determines content type based on data and key.

        Args:
            data: Data being stored
            key: Storage key

        Returns:
            Content type string
        """
        if isinstance(data, str):
            return "text/plain; charset=utf-8"
        elif isinstance(data, (dict, list)):
            return "application/json; charset=utf-8"
        else:
            # Try to guess content type from file extension
            content_type, _ = mimetypes.guess_type(key)
            return content_type or self.config.default_content_type

    async def store(self, data: Any, key: str, metadata: dict[str, Any] | None = None) -> str:
        """Stores data in S3.

        Args:
            data: Data to store (bytes, string, or serializable object)
            key: Storage key/path
            metadata: Optional metadata to attach

        Returns:
            S3 URL for the stored object

        Raises:
            S3StorageError: If storage operation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            # Prepare data
            data_bytes = self._prepare_data_for_storage(data)
            content_type = self._determine_content_type(data, key)
            s3_metadata = self._prepare_metadata(metadata)

            # Prepare upload parameters
            upload_params = {
                "Bucket": self.config.bucket_name,
                "Key": full_key,
                "Body": data_bytes,
                "ContentType": content_type,
                "Metadata": s3_metadata,
                "StorageClass": self.config.storage_class,
            }

            # Add server-side encryption if configured
            if self.config.server_side_encryption:
                upload_params["ServerSideEncryption"] = self.config.server_side_encryption
                if self.config.kms_key_id and self.config.server_side_encryption == "aws:kms":
                    upload_params["SSEKMSKeyId"] = self.config.kms_key_id

            # Upload the object
            await client.put_object(**upload_params)

            logger.info(f"Successfully stored object: {full_key} ({len(data_bytes)} bytes)")

            # Return S3 URL
            if self.config.endpoint_url:
                base_url = self.config.endpoint_url
                if not base_url.endswith("/"):
                    base_url += "/"
                return urljoin(base_url, f"{self.config.bucket_name}/{full_key}")
            else:
                return f"https://s3.{self.config.region_name}.amazonaws.com/{self.config.bucket_name}/{full_key}"

        except Exception as e:
            self._handle_s3_error(e, "store", full_key)

    async def retrieve(self, key: str) -> Any:
        """Retrieves data from S3.

        Args:
            key: Storage key/path

        Returns:
            Retrieved data (attempts to deserialize JSON, otherwise returns bytes)

        Raises:
            StorageNotFoundError: If object doesn't exist
            S3StorageError: If retrieval fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            response = await client.get_object(Bucket=self.config.bucket_name, Key=full_key)

            # Read the data
            data_bytes = await response["Body"].read()

            # Try to determine the original data type and deserialize appropriately
            content_type = response.get("ContentType", "")

            if content_type.startswith("application/json") or full_key.endswith(".json"):
                try:
                    return json.loads(data_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If JSON parsing fails, return as bytes
                    pass
            elif content_type.startswith("text/"):
                try:
                    return data_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # If text decoding fails, return as bytes
                    pass

            # Default: return as bytes
            return data_bytes

        except Exception as e:
            self._handle_s3_error(e, "retrieve", full_key)

    async def delete(self, key: str) -> bool:
        """Deletes data from S3.

        Args:
            key: Storage key/path

        Returns:
            True if deletion was successful

        Raises:
            S3StorageError: If deletion fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            await client.delete_object(Bucket=self.config.bucket_name, Key=full_key)

            logger.info(f"Successfully deleted object: {full_key}")
            return True

        except Exception as e:
            self._handle_s3_error(e, "delete", full_key)

    async def exists(self, key: str) -> bool:
        """Checks if data exists in S3.

        Args:
            key: Storage key/path

        Returns:
            True if object exists
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            await client.head_object(Bucket=self.config.bucket_name, Key=full_key)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "Not Found"):
                return False
            else:
                self._handle_s3_error(e, "exists", full_key)
        except Exception as e:
            self._handle_s3_error(e, "exists", full_key)

    async def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        """Lists storage keys with optional prefix.

        Args:
            prefix: Optional prefix filter

        Yields:
            Storage keys matching the prefix

        Raises:
            S3StorageError: If listing fails
        """
        full_prefix = self._full_key(prefix)
        try:
            client = await self._get_client()

            paginator = client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.config.bucket_name, Prefix=full_prefix)

            async for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj["Key"]

        except Exception as e:
            self._handle_s3_error(e, "list_keys", full_prefix)

    async def get_metadata(self, key: str) -> dict[str, Any]:
        """Gets object metadata from S3.

        Args:
            key: Storage key/path

        Returns:
            Object metadata including S3-specific information

        Raises:
            StorageNotFoundError: If object doesn't exist
            S3StorageError: If operation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            response = await client.head_object(Bucket=self.config.bucket_name, Key=full_key)

            # Extract relevant metadata
            metadata = {
                "content_length": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
                "storage_class": response.get("StorageClass", "STANDARD"),
                "server_side_encryption": response.get("ServerSideEncryption"),
                "user_metadata": response.get("Metadata", {}),
            }

            # Add version ID if versioning is enabled
            if "VersionId" in response:
                metadata["version_id"] = response["VersionId"]

            return metadata

        except Exception as e:
            self._handle_s3_error(e, "get_metadata", full_key)

    async def copy(self, source_key: str, destination_key: str, metadata: dict[str, Any] | None = None) -> str:
        """Copies an object within S3.

        Args:
            source_key: Source storage key
            destination_key: Destination storage key
            metadata: Optional new metadata for the copy

        Returns:
            S3 URL for the copied object

        Raises:
            StorageNotFoundError: If source object doesn't exist
            S3StorageError: If copy operation fails
        """
        full_source = self._full_key(source_key)
        full_dest = self._full_key(destination_key)
        try:
            client = await self._get_client()

            copy_source = {"Bucket": self.config.bucket_name, "Key": full_source}

            copy_params = {
                "CopySource": copy_source,
                "Bucket": self.config.bucket_name,
                "Key": full_dest,
            }

            # Add metadata if provided
            if metadata:
                s3_metadata = self._prepare_metadata(metadata)
                copy_params["Metadata"] = s3_metadata
                copy_params["MetadataDirective"] = "REPLACE"

            await client.copy_object(**copy_params)

            logger.info(f"Successfully copied object from {full_source} to {full_dest}")

            # Return destination URL
            if self.config.endpoint_url:
                base_url = self.config.endpoint_url
                if not base_url.endswith("/"):
                    base_url += "/"
                return urljoin(base_url, f"{self.config.bucket_name}/{full_dest}")
            else:
                return f"https://s3.{self.config.region_name}.amazonaws.com/{self.config.bucket_name}/{full_dest}"

        except Exception as e:
            self._handle_s3_error(e, "copy", f"{full_source} -> {full_dest}")

    async def generate_presigned_url(self, key: str, operation: str = "get_object", expiration: int = 3600) -> str:
        """Generates a presigned URL for S3 operations.

        Args:
            key: Storage key/path
            operation: S3 operation (get_object, put_object, etc.)
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL

        Raises:
            S3StorageError: If URL generation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            url = await client.generate_presigned_url(
                operation, Params={"Bucket": self.config.bucket_name, "Key": full_key}, ExpiresIn=expiration
            )

            return url

        except Exception as e:
            self._handle_s3_error(e, "generate_presigned_url", full_key)

    async def upload_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str:
        """Uploads a file from disk to S3.

        Args:
            file_path: Local file path to upload
            key: S3 key/path for the uploaded object
            content_type: MIME content type (auto-detected from extension if not provided)
            content_disposition: Content-Disposition header value

        Returns:
            S3 URI (s3://bucket/key) for the uploaded object

        Raises:
            S3StorageError: If upload fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            extra_args: dict[str, str] = {}
            if content_type:
                extra_args["ContentType"] = content_type
            else:
                guessed, _ = mimetypes.guess_type(str(file_path))
                if guessed:
                    extra_args["ContentType"] = guessed
            if content_disposition:
                extra_args["ContentDisposition"] = content_disposition

            with open(file_path, "rb") as f:
                await client.upload_fileobj(f, self.config.bucket_name, full_key, ExtraArgs=extra_args or None)

            s3_url = f"s3://{self.config.bucket_name}/{full_key}"
            logger.info(f"Successfully uploaded file to {s3_url}")
            return s3_url

        except Exception as e:
            self._handle_s3_error(e, "upload_file", full_key)

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageUrls:
        """Uploads a file from disk to S3 and returns both raw and presigned URLs.

        Combines upload_file() and generate_presigned_url() into a single operation
        that satisfies the StorageProvider protocol.

        Args:
            file_path: Local path to the file to upload
            key: Storage key/path (key_prefix is applied automatically)
            content_type: MIME content type (auto-detected if not provided)
            metadata: Optional metadata to attach (not yet used for file uploads)

        Returns:
            StorageUrls with raw S3 URI and presigned GET URL
        """
        raw_url = await self.upload_file(file_path, key, content_type=content_type)
        presigned_url = await self.generate_presigned_url(key, expiration=self.DEFAULT_PRESIGNED_URL_EXPIRATION)
        return StorageUrls(raw=raw_url, presigned=presigned_url)

    async def download_file(self, key: str, destination: Path) -> Path:
        """Downloads an S3 object to a local file.

        Args:
            key: S3 key/path of the object to download
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            S3StorageError: If download fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as f:
                await client.download_fileobj(self.config.bucket_name, full_key, f)

            logger.info(f"Successfully downloaded {full_key} to {destination}")
            return destination

        except Exception as e:
            self._handle_s3_error(e, "download_file", full_key)

    async def download_from_url(self, url: str, destination: Path) -> Path:
        """Downloads a file from a URL (presigned or raw) to a local path.

        ``s3://`` URIs are converted to a presigned HTTPS URL before
        downloading.  All other URLs (``https://``, ``http://``) are
        downloaded directly via HTTP GET, making this compatible with
        presigned URLs from any cloud provider.

        Note: ``http://`` URLs are accepted for flexibility but
        ``checks/utils/multistorage.py:is_remote_path`` intentionally
        recognises only ``https://`` -- plain HTTP should not appear in
        normal workflows.

        Args:
            url: ``s3://bucket/key`` URI, or an HTTP(S) URL (e.g. presigned)
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            S3StorageError: If download fails
        """
        download_url = url
        s3_key = None

        if url.startswith("s3://"):
            s3_key = parse_s3_url(url).s3_key
            download_url = await self.generate_presigned_url_from_s3_url(url, expiration=1800)
            logger.info("Generated presigned URL for download")

        logger.info(f"Downloading file to {destination}")

        # Re-validate DNS right before fetching to mitigate TOCTOU / DNS-rebinding
        check_url_security(download_url)

        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            async with aiohttp.ClientSession() as session, session.get(download_url, ssl=ssl_ctx) as response:
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with open(destination, "wb") as file:
                    async for chunk in response.content.iter_chunked(8192):
                        file.write(chunk)
        except Exception as e:
            if destination.exists():
                destination.unlink()
            raise S3StorageError(
                f"Failed to download from '{url}' to '{destination}': {e}",
                operation="download_from_url",
                key=s3_key,
                details=str(e),
            ) from e

        logger.info(f"Downloaded file successfully to {destination}")
        return destination

    async def generate_presigned_url_from_s3_url(
        self,
        s3_url: str,
        operation: str = "get_object",
        expiration: int = DEFAULT_PRESIGNED_URL_EXPIRATION,
    ) -> str:
        """Generates a presigned URL from a raw S3 URL.

        Parses the S3 URL to extract the absolute key and calls the boto3
        client directly, bypassing ``_full_key`` so the configured
        ``key_prefix`` is not applied a second time.

        Args:
            s3_url: Raw S3 URL (virtual-hosted, path-style, or s3://)
            operation: S3 operation (get_object, put_object, etc.)
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL string

        Raises:
            ValueError: If URL cannot be parsed
            S3StorageError: If presigned URL generation fails
        """
        components = parse_s3_url(s3_url)
        absolute_key = components.s3_key
        try:
            client = await self._get_client()
            return await client.generate_presigned_url(
                operation,
                Params={"Bucket": self.config.bucket_name, "Key": absolute_key},
                ExpiresIn=expiration,
            )
        except Exception as e:
            self._handle_s3_error(e, "generate_presigned_url_from_s3_url", absolute_key)

    async def close(self) -> None:
        """Closes the S3 client and releases resources."""
        try:
            if self._client_ctx:
                await self._client_ctx.__aexit__(None, None, None)
        finally:
            self._client = None
            self._client_ctx = None
            self._session = None

    async def __aenter__(self) -> S3StorageProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
