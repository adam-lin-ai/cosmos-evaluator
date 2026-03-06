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

from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr


class S3Config(BaseModel):
    """Configuration for S3 storage handler."""

    # Required settings
    bucket_name: str = Field(..., description="S3 bucket name")

    # AWS credentials (optional if using IAM roles or env vars)
    aws_access_key_id: Optional[SecretStr] = Field(None, description="AWS access key ID")
    aws_secret_access_key: Optional[SecretStr] = Field(None, description="AWS secret access key")
    aws_session_token: Optional[SecretStr] = Field(None, description="AWS session token for temporary credentials")

    # Connection settings
    region_name: str = Field("us-east-1", description="AWS region")
    endpoint_url: Optional[str] = Field(None, description="Custom S3 endpoint URL (for S3-compatible services)")

    # Key prefix (prepended to all keys in storage operations)
    key_prefix: str = Field("", description="Prefix prepended to all storage keys")

    # Behavior settings
    use_ssl: bool = Field(True, description="Use SSL for connections")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")

    # Upload settings
    multipart_threshold: int = Field(64 * 1024 * 1024, description="Threshold for multipart uploads (64MB)")
    multipart_chunksize: int = Field(16 * 1024 * 1024, description="Chunk size for multipart uploads (16MB)")
    max_concurrency: int = Field(10, description="Maximum concurrent uploads/downloads")

    # Metadata settings
    default_content_type: str = Field("application/octet-stream", description="Default content type")
    server_side_encryption: Optional[str] = Field(None, description="Server-side encryption method (AES256, aws:kms)")
    kms_key_id: Optional[str] = Field(None, description="KMS key ID for encryption")

    # Retry settings
    max_attempts: int = Field(3, description="Maximum retry attempts")
    retry_mode: Literal["legacy", "standard", "adaptive"] = Field(
        "adaptive", description="Retry mode (legacy, standard, adaptive)"
    )

    # Storage class
    storage_class: str = Field("STANDARD", description="S3 storage class")

    # Lifecycle settings
    enable_versioning: bool = Field(False, description="Enable S3 versioning")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Don't allow extra fields

    def get_boto3_config(self) -> dict:
        """Gets configuration for boto3 client."""
        config = {
            "region_name": self.region_name,
            "use_ssl": self.use_ssl,
            "verify": self.verify_ssl,
        }

        if self.endpoint_url:
            config["endpoint_url"] = self.endpoint_url

        # Include credentials only if we have both key and secret
        if self.aws_access_key_id and self.aws_secret_access_key:
            config["aws_access_key_id"] = self.aws_access_key_id.get_secret_value()
            config["aws_secret_access_key"] = self.aws_secret_access_key.get_secret_value()

            if self.aws_session_token:
                config["aws_session_token"] = self.aws_session_token.get_secret_value()

        return config

    def get_transfer_config(self) -> dict:
        """Gets configuration for S3 transfer manager."""
        return {
            "multipart_threshold": self.multipart_threshold,
            "multipart_chunksize": self.multipart_chunksize,
            "max_concurrency": self.max_concurrency,
            "use_threads": True,
        }
