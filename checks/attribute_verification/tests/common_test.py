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

"""Unit tests for the attribute verification common utilities."""

import logging
from typing import Any, Dict
import unittest

from checks.attribute_verification.common import validate_and_cast_config_params


def _param_types(**kwargs: type) -> Dict[str, type]:
    """Helper so mypy accepts param_types as Dict[str, type]."""
    return kwargs


class TestAttributeVerificationCommon(unittest.TestCase):
    def setUp(self) -> None:
        """Set up for common utility tests."""
        self.logger = logging.getLogger(__name__)

    def test_skips_params_not_in_params(self) -> None:
        """Branch: param_name not in params → continue; missing keys omitted from result."""
        params: Dict[str, Any] = {"a": "1"}
        param_types = _param_types(a=int, b=int)
        result = validate_and_cast_config_params(params, param_types, self.logger)
        self.assertEqual(result, {"a": 1})
        self.assertNotIn("b", result)

    def test_returns_value_unchanged_when_already_correct_type(self) -> None:
        """Branch: isinstance(value, expected_type) → use as-is, no cast."""
        params = {"x": 42}
        param_types = _param_types(x=int)
        result = validate_and_cast_config_params(params, param_types, self.logger)
        self.assertEqual(result["x"], 42)

    def test_casts_int_float_bool_strings_and_str(self) -> None:
        """Branches: int/float cast, bool str 'true'/'false', str cast, and logger.debug."""
        params = {
            "retry": "3",
            "temperature": "0.5",
            "enabled": "true",
            "disabled": "false",
            "name": 42,
        }
        param_types = _param_types(retry=int, temperature=float, enabled=bool, disabled=bool, name=str)
        result = validate_and_cast_config_params(params, param_types, self.logger)
        self.assertEqual(result["retry"], 3)
        self.assertEqual(result["temperature"], 0.5)
        self.assertEqual(result["enabled"], True)
        self.assertEqual(result["disabled"], False)
        self.assertEqual(result["name"], "42")

    def test_raises_on_invalid_bool_string(self) -> None:
        """Branch: bool from str not in true/false set → ValueError, then except + logger.error + TypeError."""
        params = {"x": "maybe"}
        param_types = _param_types(x=bool)
        with self.assertRaises(TypeError) as ctx:
            validate_and_cast_config_params(params, param_types, self.logger)
        self.assertIn("Cannot cast parameter 'x'", str(ctx.exception))

    def test_casts_non_string_to_bool(self) -> None:
        """Branch: expected_type is bool and value is not str → bool(value)."""
        params = {"x": 1}
        param_types = _param_types(x=bool)
        result = validate_and_cast_config_params(params, param_types, self.logger)
        self.assertIs(result["x"], True)

    def test_casts_to_other_type(self) -> None:
        """Branch: expected_type not bool/int/float/str → direct expected_type(value)."""
        params = {"x": "ab"}
        param_types = _param_types(x=list)
        result = validate_and_cast_config_params(params, param_types, self.logger)
        self.assertEqual(result["x"], ["a", "b"])

    def test_raises_when_logger_none(self) -> None:
        """Branch: except block with logger None → no logger.error, still raises TypeError."""
        params = {"x": "not_a_number"}
        param_types = _param_types(x=int)
        with self.assertRaises(TypeError) as ctx:
            validate_and_cast_config_params(params, param_types, None)
        self.assertIn("Cannot cast parameter 'x'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
