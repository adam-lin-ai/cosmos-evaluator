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

"""Tests for the storage provider factory."""

import os
import unittest
from unittest import mock

from services.framework.storage_providers.factory import build_storage_provider
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider
from services.settings_base import SettingsBase


class TestBuildStorageProvider(unittest.TestCase):
    """Tests for build_storage_provider factory function."""

    def test_creates_s3_provider(self) -> None:
        """Test that factory creates an S3StorageProvider when bucket and region are set."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_BUCKET": "my-bucket",
            "COSMOS_EVALUATOR_STORAGE_REGION": "us-east-1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings)

        self.assertIsInstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "my-bucket")
        self.assertEqual(provider.config.region_name, "us-east-1")
        self.assertEqual(provider.config.key_prefix, "")

    def test_raises_when_bucket_missing(self) -> None:
        """Test that factory raises ValueError when bucket is not configured."""
        env = {"COSMOS_EVALUATOR_STORAGE_REGION": "us-east-1"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_storage_provider(settings)
        self.assertIn("BUCKET", str(ctx.exception))

    def test_raises_when_region_missing(self) -> None:
        """Test that factory raises ValueError when region is not configured."""
        env = {"COSMOS_EVALUATOR_STORAGE_BUCKET": "my-bucket"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_storage_provider(settings)
        self.assertIn("REGION", str(ctx.exception))

    def test_applies_key_prefix(self) -> None:
        """Test that factory passes key_prefix to the provider config."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_BUCKET": "my-bucket",
            "COSMOS_EVALUATOR_STORAGE_REGION": "us-east-1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings, key_prefix="my/prefix/")

        self.assertIsInstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.key_prefix, "my/prefix/")

    def test_custom_bucket_and_region(self) -> None:
        """Test factory with custom bucket and region from env vars."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_BUCKET": "custom-bucket",
            "COSMOS_EVALUATOR_STORAGE_REGION": "eu-west-1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings)

        self.assertIsInstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "custom-bucket")
        self.assertEqual(provider.config.region_name, "eu-west-1")

    def test_unsupported_storage_type_raises(self) -> None:
        """Test that factory raises ValueError for unsupported storage types."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "gcs"}, clear=True):
            settings = SettingsBase()

        with self.assertRaises(ValueError) as ctx:
            build_storage_provider(settings)
        self.assertIn("gcs", str(ctx.exception))

    def test_creates_local_provider(self) -> None:
        """Test that factory creates a LocalStorageProvider when storage_type is local."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings, key_prefix="/tmp/results")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_local_provider_ignores_bucket_and_region(self) -> None:
        """Test that local provider doesn't require bucket or region."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings, key_prefix="/tmp/output")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_passes_aws_credentials(self) -> None:
        """Test that factory passes AWS credentials from settings to the config."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_BUCKET": "my-bucket",
            "COSMOS_EVALUATOR_STORAGE_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "secret123",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_storage_provider(settings)

        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "AKIATEST")
        self.assertEqual(provider.config.aws_secret_access_key.get_secret_value(), "secret123")


if __name__ == "__main__":
    unittest.main()
