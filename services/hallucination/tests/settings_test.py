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

"""Unit tests for hallucination/settings.py."""

import os
import unittest
from unittest import mock

from pydantic import ValidationError

from services.hallucination.settings import Settings, get_settings
from services.settings_base import Environment, LogLevel


class TestSettings(unittest.TestCase):
    """Tests for the Hallucination Settings class."""

    def test_defaults(self) -> None:
        """Test default settings values."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = Settings()
        self.assertEqual(s.env, Environment.LOCAL)
        self.assertEqual(s.log_level, LogLevel.INFO)
        self.assertEqual(s.process_concurrency_limit, 4)

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
            {"COSMOS_EVALUATOR_ENV": "staging", "COSMOS_EVALUATOR_LOG_LEVEL": "ERROR"},
        ):
            s = Settings()
            self.assertEqual(s.env, Environment.STAGING)
            self.assertEqual(s.log_level, LogLevel.ERROR)

    def test_hallucination_specific_env_override(self) -> None:
        """Test Hallucination-specific settings can be overridden."""
        with mock.patch.dict(os.environ, {"HALLUCINATION_PROCESS_CONCURRENCY_LIMIT": "7"}):
            s = Settings()
            self.assertEqual(s.process_concurrency_limit, 7)


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
