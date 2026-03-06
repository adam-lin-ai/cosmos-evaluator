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

import asyncio
from datetime import datetime
import json
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from botocore.exceptions import ClientError, NoCredentialsError
import pytest

from services.framework.protocols.storage_provider import StorageProvider
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.exceptions import (
    S3StorageError,
    StorageAuthenticationError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider


class TestS3Config:
    """Tests suite for S3Config."""

    def test_config_initialization_minimal(self):
        """Tests S3Config with minimal required parameters."""
        config = S3Config(bucket_name="test-bucket")

        assert config.bucket_name == "test-bucket"
        assert config.region_name == "us-east-1"
        assert config.use_ssl is True
        assert config.verify_ssl is True
        assert config.storage_class == "STANDARD"
        assert config.key_prefix == ""

    def test_config_key_prefix(self):
        """Tests S3Config key_prefix field."""
        config = S3Config(bucket_name="test-bucket", key_prefix="my/prefix/")
        assert config.key_prefix == "my/prefix/"

    def test_config_initialization_full(self):
        """Tests S3Config with all parameters."""
        config = S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
            region_name="us-west-2",
            endpoint_url="https://custom-s3.example.com",
            use_ssl=False,
            verify_ssl=False,
            multipart_threshold=128 * 1024 * 1024,
            multipart_chunksize=32 * 1024 * 1024,
            max_concurrency=20,
            default_content_type="text/plain",
            server_side_encryption="aws:kms",
            kms_key_id="test-key-id",
            max_attempts=5,
            retry_mode="standard",
            storage_class="REDUCED_REDUNDANCY",
            enable_versioning=True,
        )

        assert config.bucket_name == "test-bucket"
        assert config.aws_access_key_id.get_secret_value() == "test-key"
        assert config.aws_secret_access_key.get_secret_value() == "test-secret"
        assert config.aws_session_token.get_secret_value() == "test-token"
        assert config.region_name == "us-west-2"
        assert config.endpoint_url == "https://custom-s3.example.com"
        assert config.use_ssl is False
        assert config.verify_ssl is False
        assert config.multipart_threshold == 128 * 1024 * 1024
        assert config.multipart_chunksize == 32 * 1024 * 1024
        assert config.max_concurrency == 20
        assert config.default_content_type == "text/plain"
        assert config.server_side_encryption == "aws:kms"
        assert config.kms_key_id == "test-key-id"
        assert config.max_attempts == 5
        assert config.retry_mode == "standard"
        assert config.storage_class == "REDUCED_REDUNDANCY"
        assert config.enable_versioning is True

    def test_get_boto3_config(self):
        """Tests boto3 config generation."""
        config = S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-west-2",
            endpoint_url="https://custom-s3.example.com",
            use_ssl=False,
            verify_ssl=False,
        )

        boto3_config = config.get_boto3_config()

        assert boto3_config["region_name"] == "us-west-2"
        assert boto3_config["endpoint_url"] == "https://custom-s3.example.com"
        assert boto3_config["aws_access_key_id"] == "test-key"
        assert boto3_config["aws_secret_access_key"] == "test-secret"
        assert boto3_config["use_ssl"] is False
        assert boto3_config["verify"] is False

    def test_get_transfer_config(self):
        """Tests transfer config generation."""
        MULTIPART_THRESHOLD = 128 * 1024 * 1024
        MULTIPART_CHUNKSIZE = 32 * 1024 * 1024
        MAX_CONCURRENCY = 20

        config = S3Config(
            bucket_name="test-bucket",
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
            max_concurrency=MAX_CONCURRENCY,
        )

        transfer_config = config.get_transfer_config()

        assert transfer_config["multipart_threshold"] == MULTIPART_THRESHOLD
        assert transfer_config["multipart_chunksize"] == MULTIPART_CHUNKSIZE
        assert transfer_config["max_concurrency"] == MAX_CONCURRENCY
        assert transfer_config["use_threads"] is True


class TestS3StorageProvider:
    """Tests suite for S3StorageProvider."""

    @pytest.fixture
    def config(self):
        """Provides a standard S3 config."""
        return S3Config(bucket_name="test-bucket", region_name="us-east-1")

    @pytest.fixture
    def provider(self, config):
        """Provides an S3 storage provider."""
        return S3StorageProvider(config)

    @pytest.fixture
    def mock_client(self):
        """Provides a mock S3 client."""
        return AsyncMock()

    def test_storage_provider_implements_protocol(self, provider):
        assert isinstance(provider, StorageProvider)

    def test_provider_initialization(self, provider, config):
        """Tests that provider initializes correctly."""
        assert provider.config == config
        assert provider._session is None
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_get_client_creation(self, provider):
        """Tests S3 client creation."""
        with patch("services.framework.storage_providers.s3_storage_provider.aioboto3") as mock_aioboto3:
            mock_session = AsyncMock()
            mock_client_ctx = AsyncMock()
            mock_client = AsyncMock()
            mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
            mock_aioboto3.Session.return_value = mock_session
            mock_session.client = Mock(return_value=mock_client_ctx)

            client = await provider._get_client()

            assert client == mock_client
            assert provider._session == mock_session
            assert provider._client == mock_client
            assert provider._client_ctx == mock_client_ctx
            mock_session.client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_reuse(self, provider):
        """Tests that S3 client is reused."""
        mock_client = AsyncMock()
        provider._client = mock_client

        client = await provider._get_client()

        assert client == mock_client

    @pytest.mark.asyncio
    async def test_concurrent_client_access(self, provider):
        """Tests that concurrent client access is properly synchronized."""
        with patch("services.framework.storage_providers.s3_storage_provider.aioboto3") as mock_aioboto3:
            mock_session = AsyncMock()
            mock_client_ctx = AsyncMock()
            mock_client = AsyncMock()
            mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
            mock_aioboto3.Session.return_value = mock_session
            mock_session.client = Mock(return_value=mock_client_ctx)

            # Simulate concurrent access
            tasks = [provider._get_client() for _ in range(10)]
            clients = await asyncio.gather(*tasks)

            # Should only create one session and client
            assert mock_aioboto3.Session.call_count == 1
            assert mock_session.client.call_count == 1
            assert all(c == mock_client for c in clients)

    def test_full_key_no_prefix(self, provider):
        """Tests _full_key with no key_prefix configured."""
        assert provider._full_key("some/key.txt") == "some/key.txt"

    def test_full_key_with_prefix(self):
        """Tests _full_key with key_prefix configured."""
        config = S3Config(bucket_name="test-bucket", key_prefix="my/prefix/")
        prov = S3StorageProvider(config)
        assert prov._full_key("file.json") == "my/prefix/file.json"

    def test_full_key_empty_key(self):
        """Tests _full_key with empty key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        assert prov._full_key("") == "prefix/"

    @pytest.mark.asyncio
    async def test_store_file_combines_upload_and_presign(self):
        """Tests store_file returns StorageUrls with raw and presigned URLs."""
        from pathlib import Path

        from services.framework.protocols.storage_provider import StorageUrls

        config = S3Config(bucket_name="test-bucket", key_prefix="output/")
        prov = S3StorageProvider(config)

        mock_client = AsyncMock()
        mock_client.upload_fileobj = AsyncMock()
        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url")
        prov._client = mock_client

        tmp = Path("/tmp/test_store_file.json")
        try:
            tmp.write_text('{"test": true}')
            result = await prov.store_file(tmp, "data.json", "application/json")
        finally:
            if tmp.exists():
                tmp.unlink()

        assert isinstance(result, StorageUrls)
        assert result.raw == "s3://test-bucket/output/data.json"
        assert result.presigned == "https://presigned.url"
        mock_client.upload_fileobj.assert_called_once()
        mock_client.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_applies_key_prefix(self, mock_client):
        """Tests that store() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.put_object = AsyncMock()

        await prov.store("data", "file.txt")

        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_exists_applies_key_prefix(self, mock_client):
        """Tests that exists() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.head_object = AsyncMock()

        await prov.exists("file.txt")

        call_args = mock_client.head_object.call_args
        assert call_args.kwargs["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_generate_presigned_url_applies_key_prefix(self, mock_client):
        """Tests that generate_presigned_url() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url")

        await prov.generate_presigned_url("file.txt")

        call_args = mock_client.generate_presigned_url.call_args
        assert call_args.kwargs["Params"]["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self, provider, mock_client):
        """Tests that concurrent store operations work correctly."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        # Simulate multiple concurrent stores
        tasks = [provider.store(f"data-{i}", f"key-{i}.txt") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert mock_client.put_object.call_count == 10

    def test_prepare_data_for_storage_bytes(self, provider):
        """Tests data preparation with bytes input."""
        data = b"test data"
        result = provider._prepare_data_for_storage(data)
        assert result == data

    def test_prepare_data_for_storage_string(self, provider):
        """Tests data preparation with string input."""
        data = "test string"
        result = provider._prepare_data_for_storage(data)
        assert result == data.encode("utf-8")

    def test_prepare_data_for_storage_dict(self, provider):
        """Tests data preparation with dict input."""
        data = {"key": "value", "number": 42}
        result = provider._prepare_data_for_storage(data)
        expected = json.dumps(data, ensure_ascii=False).encode("utf-8")
        assert result == expected

    def test_prepare_data_for_storage_list(self, provider):
        """Tests data preparation with list input."""
        data = [1, 2, 3, "test"]
        result = provider._prepare_data_for_storage(data)
        expected = json.dumps(data, ensure_ascii=False).encode("utf-8")
        assert result == expected

    def test_prepare_data_for_storage_other(self, provider):
        """Tests data preparation with other types."""
        data = 42
        result = provider._prepare_data_for_storage(data)
        assert result == b"42"

    def test_prepare_metadata(self, provider):
        """Tests metadata preparation."""
        metadata = {
            "user_id": "123",
            "upload_time": datetime.now(),
            "tags": ["tag1", "tag2"],
            "complex_data": {"nested": "value"},
        }

        result = provider._prepare_metadata(metadata)

        assert result["user-id"] == "123"
        assert "upload-time" in result
        assert "tags" in result
        assert "complex-data" in result

    def test_prepare_metadata_empty(self, provider):
        """Tests metadata preparation with empty input."""
        result = provider._prepare_metadata(None)
        assert result == {}

        result = provider._prepare_metadata({})
        assert result == {}

    def test_determine_content_type_string(self, provider):
        """Tests content type determination for strings."""
        assert provider._determine_content_type("test", "key") == "text/plain; charset=utf-8"

    def test_determine_content_type_dict(self, provider):
        """Tests content type determination for dicts."""
        assert provider._determine_content_type({"key": "value"}, "key") == "application/json; charset=utf-8"

    def test_determine_content_type_by_extension(self, provider):
        """Tests content type determination by file extension."""
        assert provider._determine_content_type(b"data", "file.json") == "application/json"
        assert provider._determine_content_type(b"data", "file.txt") == "text/plain"
        assert provider._determine_content_type(b"data", "file.html") == "text/html"
        assert provider._determine_content_type(b"data", "file.jpg") == "image/jpeg"
        assert provider._determine_content_type(b"data", "file.png") == "image/png"
        assert provider._determine_content_type(b"data", "file.pdf") == "application/pdf"

    def test_determine_content_type_default(self, provider):
        """Tests default content type determination."""
        assert provider._determine_content_type(b"data", "unknown.unknownextension123") == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_store_success(self, provider, mock_client):
        """Tests successful data storage."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        test_data = {"message": "test"}
        result = await provider.store(test_data, "test/key.json")

        assert result == "https://s3.us-east-1.amazonaws.com/test-bucket/test/key.json"
        mock_client.put_object.assert_called_once()

        call_args = mock_client.put_object.call_args[1]
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == "test/key.json"
        assert call_args["ContentType"] == "application/json; charset=utf-8"
        assert call_args["StorageClass"] == "STANDARD"

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, provider, mock_client):
        """Tests data storage with metadata."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        test_data = "test string"
        test_metadata = {"user_id": "123", "category": "test"}

        await provider.store(test_data, "test/key.txt", test_metadata)

        call_args = mock_client.put_object.call_args[1]
        assert "user-id" in call_args["Metadata"]
        assert "category" in call_args["Metadata"]
        assert call_args["Metadata"]["user-id"] == "123"
        assert call_args["Metadata"]["category"] == "test"

    @pytest.mark.asyncio
    async def test_store_with_encryption(self, provider, mock_client):
        """Tests data storage with server-side encryption."""
        provider.config.server_side_encryption = "AES256"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        await provider.store("test data", "test/key.txt")

        call_args = mock_client.put_object.call_args[1]
        assert call_args["ServerSideEncryption"] == "AES256"

    @pytest.mark.asyncio
    async def test_store_with_kms_encryption(self, provider, mock_client):
        """Tests data storage with KMS encryption."""
        provider.config.server_side_encryption = "aws:kms"
        provider.config.kms_key_id = "test-key-id"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        await provider.store("test data", "test/key.txt")

        call_args = mock_client.put_object.call_args[1]
        assert call_args["ServerSideEncryption"] == "aws:kms"
        assert call_args["SSEKMSKeyId"] == "test-key-id"

    @pytest.mark.asyncio
    async def test_store_with_custom_endpoint(self, provider, mock_client):
        """Tests data storage with custom endpoint URL."""
        provider.config.endpoint_url = "https://custom-s3.example.com"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        result = await provider.store("test data", "test/key.txt")

        assert result == "https://custom-s3.example.com/test-bucket/test/key.txt"

    @pytest.mark.asyncio
    async def test_retrieve_json_data(self, provider, mock_client):
        """Tests retrieving JSON data."""
        provider._client = mock_client

        test_data = {"message": "test", "number": 42}
        json_data = json.dumps(test_data).encode("utf-8")

        mock_response = {"Body": AsyncMock(), "ContentType": "application/json"}
        mock_response["Body"].read = AsyncMock(return_value=json_data)
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.json")

        assert result == test_data
        mock_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.json")

    @pytest.mark.asyncio
    async def test_retrieve_text_data(self, provider, mock_client):
        """Tests retrieving text data."""
        provider._client = mock_client

        test_data = "test string content"

        mock_response = {"Body": AsyncMock(), "ContentType": "text/plain"}
        mock_response["Body"].read = AsyncMock(return_value=test_data.encode("utf-8"))
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.txt")

        assert result == test_data

    @pytest.mark.asyncio
    async def test_retrieve_binary_data(self, provider, mock_client):
        """Tests retrieving binary data."""
        provider._client = mock_client

        test_data = b"binary data content"

        mock_response = {"Body": AsyncMock(), "ContentType": "application/octet-stream"}
        mock_response["Body"].read = AsyncMock(return_value=test_data)
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.bin")

        assert result == test_data

    @pytest.mark.asyncio
    async def test_delete_success(self, provider, mock_client):
        """Tests successful data deletion."""
        provider._client = mock_client
        mock_client.delete_object = AsyncMock()

        result = await provider.delete("test/key.txt")

        assert result is True
        mock_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    @pytest.mark.asyncio
    async def test_exists_true(self, provider, mock_client):
        """Tests exists check when object exists."""
        provider._client = mock_client
        mock_client.head_object = AsyncMock()

        result = await provider.exists("test/key.txt")

        assert result is True
        mock_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_code", ["404", "NoSuchKey", "Not Found"])
    async def test_exists_false(self, provider, mock_client, error_code):
        """Tests exists check when object doesn't exist."""
        provider._client = mock_client

        error_response = {"Error": {"Code": error_code}}
        mock_client.head_object = AsyncMock(side_effect=ClientError(error_response, "HeadObject"))

        result = await provider.exists("test/nonexistent.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_keys(self, provider, mock_client):
        """Tests listing storage keys."""
        provider._client = mock_client

        mock_paginator = AsyncMock()
        mock_client.get_paginator = MagicMock(return_value=mock_paginator)

        # Mock paginated results
        mock_pages = [
            {"Contents": [{"Key": "test/file1.txt"}, {"Key": "test/file2.txt"}]},
            {"Contents": [{"Key": "test/file3.txt"}]},
            {},  # Empty page to end iteration
        ]

        async def mock_paginate(*args, **kwargs):
            for page in mock_pages:
                yield page

        mock_paginator.paginate = MagicMock(return_value=mock_paginate())

        keys = []
        async for key in provider.list_keys("test/"):
            keys.append(key)

        assert keys == ["test/file1.txt", "test/file2.txt", "test/file3.txt"]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")

    @pytest.mark.asyncio
    async def test_get_metadata(self, provider, mock_client):
        """Tests getting object metadata."""
        provider._client = mock_client

        mock_response = {
            "ContentLength": 1024,
            "ContentType": "application/json",
            "LastModified": datetime(2025, 1, 1, 12, 0, 0),
            "ETag": '"abc123"',
            "StorageClass": "STANDARD",
            "ServerSideEncryption": "AES256",
            "Metadata": {"user-id": "123", "category": "test"},
            "VersionId": "v1.0",
        }
        mock_client.head_object = AsyncMock(return_value=mock_response)

        metadata = await provider.get_metadata("test/key.json")

        assert metadata["content_length"] == 1024
        assert metadata["content_type"] == "application/json"
        assert metadata["etag"] == "abc123"
        assert metadata["storage_class"] == "STANDARD"
        assert metadata["server_side_encryption"] == "AES256"
        assert metadata["user_metadata"] == {"user-id": "123", "category": "test"}
        assert metadata["version_id"] == "v1.0"

    @pytest.mark.asyncio
    async def test_copy_success(self, provider, mock_client):
        """Tests successful object copy."""
        provider._client = mock_client
        mock_client.copy_object = AsyncMock()

        result = await provider.copy("source/key.txt", "dest/key.txt")

        assert result == "https://s3.us-east-1.amazonaws.com/test-bucket/dest/key.txt"
        mock_client.copy_object.assert_called_once()

        call_args = mock_client.copy_object.call_args[1]
        assert call_args["CopySource"]["Bucket"] == "test-bucket"
        assert call_args["CopySource"]["Key"] == "source/key.txt"
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == "dest/key.txt"

    @pytest.mark.asyncio
    async def test_copy_with_metadata(self, provider, mock_client):
        """Tests object copy with new metadata."""
        provider._client = mock_client
        mock_client.copy_object = AsyncMock()

        metadata = {"new_field": "value"}
        await provider.copy("source/key.txt", "dest/key.txt", metadata)

        call_args = mock_client.copy_object.call_args[1]
        assert call_args["MetadataDirective"] == "REPLACE"
        assert "new-field" in call_args["Metadata"]

    @pytest.mark.asyncio
    async def test_generate_presigned_url(self, provider, mock_client):
        """Tests presigned URL generation."""
        provider._client = mock_client
        expected_url = "https://test-bucket.s3.amazonaws.com/test/key.txt?signature=..."
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        url = await provider.generate_presigned_url("test/key.txt", "get_object", 3600)

        assert url == expected_url
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "test-bucket", "Key": "test/key.txt"}, ExpiresIn=3600
        )

    def test_handle_s3_error_no_such_bucket(self, provider):
        """Tests S3 error handling for NoSuchBucket."""
        error_response = {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(S3StorageError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Bucket 'test-bucket' does not exist" in str(exc_info.value)
        assert exc_info.value.aws_error_code == "NoSuchBucket"

    def test_handle_s3_error_no_such_key(self, provider):
        """Tests S3 error handling for NoSuchKey."""
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Key does not exist"}}
        error = ClientError(error_response, "GetObject")

        with pytest.raises(StorageNotFoundError) as exc_info:
            provider._handle_s3_error(error, "retrieve", "test/key.txt")

        assert "Object 'test/key.txt' not found" in str(exc_info.value)

    def test_handle_s3_error_access_denied(self, provider):
        """Tests S3 error handling for AccessDenied."""
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(StoragePermissionError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Access denied" in str(exc_info.value)

    def test_handle_s3_error_quota_exceeded(self, provider):
        """Tests S3 error handling for QuotaExceeded."""
        error_response = {"Error": {"Code": "QuotaExceeded", "Message": "Quota exceeded"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(StorageQuotaExceededError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Storage quota exceeded" in str(exc_info.value)

    def test_handle_s3_error_no_credentials(self, provider):
        """Tests S3 error handling for NoCredentialsError."""
        error = NoCredentialsError()

        with pytest.raises(StorageAuthenticationError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "AWS credentials not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Tests closing the provider."""
        mock_client = AsyncMock()
        mock_client_ctx = AsyncMock()
        mock_session = MagicMock()
        provider._client = mock_client
        provider._client_ctx = mock_client_ctx
        provider._session = mock_session

        await provider.close()

        mock_client_ctx.__aexit__.assert_called_once_with(None, None, None)
        assert provider._client is None
        assert provider._client_ctx is None
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_context_manager(self, provider):
        """Tests async context manager functionality."""
        with patch.object(provider, "close") as mock_close:
            async with provider as h:
                assert h == provider

            mock_close.assert_called_once()

    def test_multiple_provider_instances(self):
        """Tests that multiple provider instances work independently."""
        config1 = S3Config(bucket_name="bucket1", region_name="us-east-1")
        config2 = S3Config(bucket_name="bucket2", region_name="us-west-2")

        provider1 = S3StorageProvider(config1)
        provider2 = S3StorageProvider(config2)

        assert provider1.config.bucket_name == "bucket1"
        assert provider1.config.region_name == "us-east-1"
        assert provider2.config.bucket_name == "bucket2"
        assert provider2.config.region_name == "us-west-2"


class TestS3StorageProviderNewMethods:
    """Tests for newly added S3StorageProvider methods."""

    @pytest.fixture
    def config(self):
        """Provides a standard S3 config."""
        return S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
        )

    @pytest.fixture
    def provider(self, config):
        """Provides an S3 storage provider."""
        return S3StorageProvider(config)

    @pytest.fixture
    def mock_client(self):
        """Provides a mock S3 client."""
        return AsyncMock()

    def test_default_constants(self):
        """Tests default class constants."""
        assert S3StorageProvider.DEFAULT_PRESIGNED_URL_EXPIRATION == 43200

    @pytest.mark.asyncio
    async def test_upload_file(self, provider, mock_client, tmp_path):
        """Tests uploading a file from disk."""
        provider._client = mock_client
        mock_client.upload_fileobj = AsyncMock()

        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        result = await provider.upload_file(test_file, "results/test.json", content_type="application/json")

        assert "test-bucket" in result
        assert "results/test.json" in result
        mock_client.upload_fileobj.assert_called_once()
        call_args = mock_client.upload_fileobj.call_args
        assert call_args[0][1] == "test-bucket"
        assert call_args[0][2] == "results/test.json"
        assert call_args[1]["ExtraArgs"]["ContentType"] == "application/json"

    @pytest.mark.asyncio
    async def test_upload_file_with_content_disposition(self, provider, mock_client, tmp_path):
        """Tests uploading a file with content disposition."""
        provider._client = mock_client
        mock_client.upload_fileobj = AsyncMock()

        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"fake video")

        await provider.upload_file(
            test_file, "videos/video.mp4", content_type="video/mp4", content_disposition="attachment"
        )

        call_args = mock_client.upload_fileobj.call_args
        assert call_args[1]["ExtraArgs"]["ContentDisposition"] == "attachment"

    @pytest.mark.asyncio
    async def test_download_file(self, provider, mock_client, tmp_path):
        """Tests downloading a file to disk."""
        provider._client = mock_client
        mock_client.download_fileobj = AsyncMock()

        destination = tmp_path / "downloaded.json"
        result = await provider.download_file("results/test.json", destination)

        assert result == destination
        mock_client.download_fileobj.assert_called_once()
        call_args = mock_client.download_fileobj.call_args
        assert call_args[0][0] == "test-bucket"
        assert call_args[0][1] == "results/test.json"

    @pytest.mark.asyncio
    async def test_download_file_creates_parent_dirs(self, provider, mock_client, tmp_path):
        """Tests that download_file creates parent directories."""
        provider._client = mock_client
        mock_client.download_fileobj = AsyncMock()

        destination = tmp_path / "a" / "b" / "c" / "file.txt"
        await provider.download_file("key.txt", destination)

        assert destination.parent.exists()

    @pytest.mark.asyncio
    async def test_generate_presigned_url_from_s3_url(self, provider, mock_client):
        """Tests generating presigned URL from an S3 URL."""
        provider._client = mock_client
        expected_url = "https://test-bucket.s3.amazonaws.com/path/file.mp4?signature=..."
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        result = await provider.generate_presigned_url_from_s3_url("s3://test-bucket/path/file.mp4", expiration=7200)

        assert result == expected_url
        mock_client.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_presigned_url_from_virtual_hosted_url(self, provider, mock_client):
        """Tests generating presigned URL from a virtual-hosted S3 URL."""
        provider._client = mock_client
        expected_url = "https://presigned.url"
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        result = await provider.generate_presigned_url_from_s3_url(
            "https://my-bucket.s3.us-west-2.amazonaws.com/path/file.mp4"
        )

        assert result == expected_url

    @pytest.mark.asyncio
    async def test_download_from_url_presigned(self, provider, tmp_path):
        """Tests downloading from a presigned URL."""
        destination = tmp_path / "downloaded.mp4"
        presigned_url = (
            "https://bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=KEY%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=abc123"
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_iter_chunked(size):
            yield b"video data chunk 1"
            yield b"video data chunk 2"

        mock_response.content.iter_chunked = mock_iter_chunked

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        mock_session_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None),
        )

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            result = await provider.download_from_url(presigned_url, destination)

        assert result == destination
        assert destination.read_bytes() == b"video data chunk 1video data chunk 2"

    @pytest.mark.asyncio
    async def test_download_from_url_raw_s3(self, provider, mock_client, tmp_path):
        """Tests downloading from a raw S3 URL (generates presigned first)."""
        provider._client = mock_client
        destination = tmp_path / "downloaded.mp4"
        raw_url = "s3://test-bucket/path/file.mp4"

        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url/path/file.mp4?sig=abc")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_iter_chunked(size):
            yield b"data"

        mock_response.content.iter_chunked = mock_iter_chunked

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        mock_session_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None),
        )

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            result = await provider.download_from_url(raw_url, destination)

        assert result == destination
        mock_client.generate_presigned_url.assert_called_once()

    def test_from_settings(self):
        """Tests factory method from_settings."""
        mock_settings = MagicMock()
        mock_settings.storage_access_key = "test-access-key"
        mock_settings.storage_secret_key = "test-secret-key"
        mock_settings.storage_region = "us-west-2"

        with patch("services.framework.storage_providers.s3_storage_provider.SettingsBase") as mock_cls:
            mock_cls.return_value = mock_settings
            mock_cls.get_env_files.return_value = []
            provider = S3StorageProvider.from_settings(bucket_name="my-bucket")

        assert provider.config.bucket_name == "my-bucket"
        assert provider.config.region_name == "us-west-2"

    def test_from_settings_default_region(self):
        """Tests from_settings with no region falls back to us-east-1."""
        mock_settings = MagicMock()
        mock_settings.storage_access_key = "test-access-key"
        mock_settings.storage_secret_key = "test-secret-key"
        mock_settings.storage_region = None

        with patch("services.framework.storage_providers.s3_storage_provider.SettingsBase") as mock_cls:
            mock_cls.return_value = mock_settings
            mock_cls.get_env_files.return_value = []
            provider = S3StorageProvider.from_settings(bucket_name="my-bucket")

        assert provider.config.bucket_name == "my-bucket"
        assert provider.config.region_name == "us-east-1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
