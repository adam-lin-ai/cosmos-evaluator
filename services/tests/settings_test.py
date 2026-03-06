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

"""Unit tests for services/settings.py."""

import os
import unittest
from unittest import mock

from pydantic import ValidationError

from services.settings_base import Environment, LogLevel, SettingsBase


class TestEnvironmentEnum(unittest.TestCase):
    """Tests for the Environment enum."""

    def test_local_value(self) -> None:
        """Test Environment.LOCAL has correct value."""
        self.assertEqual(Environment.LOCAL.value, "local")

    def test_prod_value(self) -> None:
        """Test Environment.PROD has correct value."""
        self.assertEqual(Environment.PROD.value, "production")

    def test_staging_value(self) -> None:
        """Test Environment.STAGING has correct value."""
        self.assertEqual(Environment.STAGING.value, "staging")

    def test_environment_is_string_enum(self) -> None:
        """Test Environment is both str and Enum."""
        self.assertIsInstance(Environment.PROD, str)
        self.assertEqual(Environment.PROD, "production")
        self.assertIsInstance(Environment.LOCAL, str)
        self.assertEqual(Environment.LOCAL, "local")
        self.assertIsInstance(Environment.STAGING, str)
        self.assertEqual(Environment.STAGING, "staging")

    def test_environment_from_string(self) -> None:
        """Test Environment can be created from string."""
        self.assertEqual(Environment("local"), Environment.LOCAL)
        self.assertEqual(Environment("production"), Environment.PROD)
        self.assertEqual(Environment("staging"), Environment.STAGING)


class TestLogLevelEnum(unittest.TestCase):
    """Tests for the LogLevel enum."""

    def test_debug_value(self) -> None:
        """Test LogLevel.DEBUG has correct value."""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")

    def test_info_value(self) -> None:
        """Test LogLevel.INFO has correct value."""
        self.assertEqual(LogLevel.INFO.value, "INFO")

    def test_warning_value(self) -> None:
        """Test LogLevel.WARNING has correct value."""
        self.assertEqual(LogLevel.WARNING.value, "WARNING")

    def test_error_value(self) -> None:
        """Test LogLevel.ERROR has correct value."""
        self.assertEqual(LogLevel.ERROR.value, "ERROR")


class TestSettingsDefaults(unittest.TestCase):
    """Tests for SettingsBase default values."""

    def test_default_env_is_local(self) -> None:
        """Test default environment is LOCAL."""
        s = SettingsBase()
        self.assertEqual(s.env, Environment.LOCAL)

    def test_default_log_level_is_info(self) -> None:
        """Test default log level is INFO."""
        s = SettingsBase()
        self.assertEqual(s.log_level, LogLevel.INFO)

    def test_default_storage_fields_are_none(self) -> None:
        """Test default storage credential fields are None."""
        s = SettingsBase()
        self.assertIsNone(s.storage_access_key)
        self.assertIsNone(s.storage_secret_key)
        self.assertIsNone(s.storage_region)


class TestSettingsConstructor(unittest.TestCase):
    """Tests for SettingsBase constructor arguments."""

    def test_env_set_to_production(self) -> None:
        """Test setting env to production via constructor."""
        s = SettingsBase(env=Environment.PROD)
        self.assertEqual(s.env, Environment.PROD)

    def test_env_set_to_staging(self) -> None:
        """Test setting env to staging via constructor."""
        s = SettingsBase(env=Environment.STAGING)
        self.assertEqual(s.env, Environment.STAGING)

    def test_log_level_set_to_debug(self) -> None:
        """Test setting log_level to DEBUG via constructor."""
        s = SettingsBase(log_level=LogLevel.DEBUG)
        self.assertEqual(s.log_level, LogLevel.DEBUG)

    def test_log_level_set_to_error(self) -> None:
        """Test setting log_level to ERROR via constructor."""
        s = SettingsBase(log_level=LogLevel.ERROR)
        self.assertEqual(s.log_level, LogLevel.ERROR)

    def test_invalid_env_value_raises_error(self) -> None:
        """Test that invalid environment value raises ValidationError."""
        with self.assertRaises(ValidationError):
            SettingsBase(env="invalid_env_value")

    def test_invalid_log_level_raises_error(self) -> None:
        """Test that invalid log level raises ValidationError."""
        with self.assertRaises(ValidationError):
            SettingsBase(log_level="INVALID")

    def test_storage_fields_can_be_set(self) -> None:
        """Test storage fields can be set via constructor."""
        s = SettingsBase(
            storage_access_key="dummy_key",
            storage_secret_key="dummy_secret",
            storage_region="us-west-2",
        )
        self.assertEqual(s.storage_access_key, "dummy_key")
        self.assertEqual(s.storage_secret_key, "dummy_secret")
        self.assertEqual(s.storage_region, "us-west-2")


class TestStorageCredentialsFromEnv(unittest.TestCase):
    """Tests for storage credentials loaded from environment variables."""

    def test_storage_access_key_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_ACCESS_KEY env var sets storage_access_key field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "test_key"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_access_key, "test_key")

    def test_storage_secret_key_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_SECRET_KEY env var sets storage_secret_key field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "test_secret"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_secret_key, "test_secret")

    def test_storage_region_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_REGION env var sets storage_region field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_REGION": "us-west-2"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_region, "us-west-2")

    def test_backward_compat_aws_access_key_id(self) -> None:
        """Test AWS_ACCESS_KEY_ID env var still sets storage_access_key (backward compat)."""
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_access_key_id"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_access_key, "test_access_key_id")

    def test_backward_compat_aws_secret_access_key(self) -> None:
        """Test AWS_SECRET_ACCESS_KEY env var still sets storage_secret_key (backward compat)."""
        with mock.patch.dict(os.environ, {"AWS_SECRET_ACCESS_KEY": "test_secret"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_secret_key, "test_secret")

    def test_backward_compat_aws_default_region(self) -> None:
        """Test AWS_DEFAULT_REGION env var still sets storage_region (backward compat)."""
        with mock.patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-west-2"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_region, "us-west-2")

    def test_all_storage_credentials_from_env(self) -> None:
        """Test all storage credentials can be loaded from environment variables."""
        env_vars = {
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "test_access_key_id",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "test_secret_access_key",
            "COSMOS_EVALUATOR_STORAGE_REGION": "eu-central-1",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_access_key, "test_access_key_id")
            self.assertEqual(s.storage_secret_key, "test_secret_access_key")
            self.assertEqual(s.storage_region, "eu-central-1")

    def test_storage_credentials_default_to_none_when_not_set(self) -> None:
        """Test storage credentials default to None when env vars not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = SettingsBase()
            self.assertIsNone(s.storage_access_key)
            self.assertIsNone(s.storage_secret_key)
            self.assertIsNone(s.storage_region)


class TestSettingsGetEnvFiles(unittest.TestCase):
    """Tests for SettingsBase.get_env_files method."""

    def test_get_env_files_returns_paths_for_local(self) -> None:
        """Test get_env_files returns paths in local mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "local"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(len(paths), 2)
            self.assertTrue(any(".cosmos_evaluator" in str(p) for p in paths))
            self.assertTrue(any(".aws" in str(p) for p in paths))

    def test_get_env_files_returns_empty_for_production(self) -> None:
        """Test get_env_files returns empty list in production mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "production"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(paths, [])

    def test_get_env_files_returns_empty_for_staging(self) -> None:
        """Test get_env_files returns empty list in staging mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "staging"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(paths, [])

    def test_get_env_files_defaults_to_local(self) -> None:
        """Test get_env_files defaults to local when COSMOS_EVALUATOR_ENV not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            paths = SettingsBase.get_env_files()
            self.assertEqual(len(paths), 2)


if __name__ == "__main__":
    unittest.main()
