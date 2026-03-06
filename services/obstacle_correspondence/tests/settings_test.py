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

"""Unit tests for obstacle_correspondence/settings.py."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pydantic import ValidationError

from services.obstacle_correspondence.settings import Settings, get_settings
from services.settings_base import Environment, LogLevel


class TestSettings(unittest.TestCase):
    """Tests for the Obstacle Correspondence Settings class."""

    def test_defaults(self) -> None:
        """Test default settings values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = Settings()
            self.assertEqual(s.env, Environment.LOCAL)
            self.assertEqual(s.log_level, LogLevel.INFO)
            self.assertEqual(s.process_concurrency_limit, 3)

    def test_env_override_production(self) -> None:
        """Test environment variables override defaults to production."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "production"}):
            s = Settings()
            self.assertEqual(s.env, Environment.PROD)

    def test_env_override_staging(self) -> None:
        """Test environment variables override defaults to staging."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "staging"}):
            s = Settings()
            self.assertEqual(s.env, Environment.STAGING)

    def test_log_level_override(self) -> None:
        """Test log_level can be overridden via environment variable."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_LOG_LEVEL": "DEBUG"}):
            s = Settings()
            self.assertEqual(s.log_level, LogLevel.DEBUG)

    def test_init_kwargs_override_env(self) -> None:
        """Test constructor arguments override environment variables."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "production"}):
            s = Settings(env=Environment.STAGING)
            self.assertEqual(s.env, Environment.STAGING)

    def test_invalid_env_value(self) -> None:
        """Test validation error for invalid environment value."""
        with self.assertRaises(ValidationError):
            Settings(env="invalid_env")

    def test_invalid_log_level_value(self) -> None:
        """Test validation error for invalid log level value."""
        with self.assertRaises(ValidationError):
            Settings(log_level="INVALID")

    def test_env_prefix_case_insensitive(self) -> None:
        """Test environment variables are case-insensitive."""
        with mock.patch.dict(os.environ, {"cosmos_evaluator_env": "production"}):
            s = Settings()
            self.assertEqual(s.env, Environment.PROD)

    def test_multiple_env_vars(self) -> None:
        """Test multiple environment variables at once."""
        with mock.patch.dict(
            os.environ,
            {
                "COSMOS_EVALUATOR_ENV": "staging",
                "COSMOS_EVALUATOR_LOG_LEVEL": "ERROR",
                "OBJECTS_PROCESS_CONCURRENCY_LIMIT": "5",
            },
            clear=True,
        ):
            s = Settings()
            self.assertEqual(s.env, Environment.STAGING)
            self.assertEqual(s.log_level, LogLevel.ERROR)
            self.assertEqual(s.process_concurrency_limit, 5)

    def test_invalid_process_concurrency_limit(self) -> None:
        """Test validation error for invalid process concurrency limit."""
        with self.assertRaises(ValidationError):
            Settings(process_concurrency_limit=0)


class TestStorageCredentialsFromEnv(unittest.TestCase):
    """Tests for storage credentials loaded from environment variables."""

    def test_backward_compat_aws_access_key_id(self) -> None:
        """Test AWS_ACCESS_KEY_ID env var (backward compat) sets storage_access_key field."""
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_access_key_id"}, clear=True):
            s = Settings()
            self.assertEqual(s.storage_access_key, "test_access_key_id")

    def test_backward_compat_aws_secret_access_key(self) -> None:
        """Test AWS_SECRET_ACCESS_KEY env var (backward compat) sets storage_secret_key field."""
        with mock.patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "test_secret_access_key"}, clear=True):
            s = Settings()
            self.assertEqual(s.storage_secret_key, "test_secret_access_key")

    def test_backward_compat_aws_default_region(self) -> None:
        """Test AWS_DEFAULT_REGION env var (backward compat) sets storage_region field."""
        with mock.patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-west-2"}, clear=True):
            s = Settings()
            self.assertEqual(s.storage_region, "us-west-2")

    def test_backward_compat_all_aws_credentials_from_env(self) -> None:
        """Test all AWS env vars (backward compat) populate storage_* fields."""
        env_vars = {
            "AWS_ACCESS_KEY_ID": "test_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "test_secret_access_key",
            "AWS_DEFAULT_REGION": "eu-central-1",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = Settings()
            self.assertEqual(s.storage_access_key, "test_access_key_id")
            self.assertEqual(s.storage_secret_key, "test_secret_access_key")
            self.assertEqual(s.storage_region, "eu-central-1")

    def test_storage_credentials_default_to_none(self) -> None:
        """Test storage credentials default to None when env vars not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = Settings()
            self.assertIsNone(s.storage_access_key)
            self.assertIsNone(s.storage_secret_key)
            self.assertIsNone(s.storage_region)


class TestStorageCredentialsFromEnvFile(unittest.TestCase):
    """Tests for storage credentials loaded from .env files (AWS_* vars for backward compat)."""

    def test_storage_credentials_from_env_file(self) -> None:
        """Test storage credentials are loaded from .env file (AWS_* vars)."""
        env_file_content = """
AWS_ACCESS_KEY_ID=test_access_key_from_file
AWS_SECRET_ACCESS_KEY=test_secret_from_file
AWS_DEFAULT_REGION=ap-southeast-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_file_content)
            env_file_path = Path(f.name)

        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                s = Settings(_env_file=env_file_path)
                self.assertEqual(s.storage_access_key, "test_access_key_from_file")
                self.assertEqual(s.storage_secret_key, "test_secret_from_file")
                self.assertEqual(s.storage_region, "ap-southeast-1")
        finally:
            env_file_path.unlink()

    def test_env_vars_override_env_file(self) -> None:
        """Test environment variables take precedence over .env file values."""
        env_file_content = """
AWS_ACCESS_KEY_ID=from_file
AWS_SECRET_ACCESS_KEY=from_file
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_file_content)
            env_file_path = Path(f.name)

        try:
            with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "from_env"}, clear=True):
                s = Settings(_env_file=env_file_path)
                # Env var should override file
                self.assertEqual(s.storage_access_key, "from_env")
                # File value should be used when no env var
                self.assertEqual(s.storage_secret_key, "from_file")
        finally:
            env_file_path.unlink()

    def test_multiple_env_files(self) -> None:
        """Test loading from multiple .env files (later files override earlier)."""
        env_file1_content = """
AWS_ACCESS_KEY_ID=from_file1
AWS_SECRET_ACCESS_KEY=from_file1
"""
        env_file2_content = """
AWS_SECRET_ACCESS_KEY=from_file2
AWS_DEFAULT_REGION=us-east-1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f1:
            f1.write(env_file1_content)
            env_file1_path = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f2:
            f2.write(env_file2_content)
            env_file2_path = Path(f2.name)

        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                s = Settings(_env_file=[env_file1_path, env_file2_path])
                # From file1 only
                self.assertEqual(s.storage_access_key, "from_file1")
                # Overridden by file2
                self.assertEqual(s.storage_secret_key, "from_file2")
                # From file2 only
                self.assertEqual(s.storage_region, "us-east-1")
        finally:
            env_file1_path.unlink()
            env_file2_path.unlink()


class TestGetSettings(unittest.TestCase):
    """Tests for the get_settings function."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """Test get_settings returns a Settings instance."""
        get_settings.cache_clear()
        s = get_settings()
        self.assertIsInstance(s, Settings)

    def test_get_settings_singleton(self) -> None:
        """Test get_settings returns a cached singleton instance."""
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        self.assertIs(s1, s2)


if __name__ == "__main__":
    unittest.main()
