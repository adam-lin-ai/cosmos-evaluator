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

import logging
from typing import Any, Dict, Optional


def validate_and_cast_config_params(
    params: Dict[str, Any], param_types: Dict[str, type], logger: Optional[logging.Logger]
) -> Dict[str, Any]:
    """
    Validate and cast configuration parameters to their expected types.

    Args:
        params (Dict[str, Any]): Dictionary of parameters to validate and cast
        param_types (Dict[str, type]): Dictionary mapping parameter names to expected types
        logger (logging.Logger, optional): Logger for error reporting

    Returns:
        Dict[str, Any]: Dictionary with parameters cast to correct types

    Raises:
        TypeError: If a parameter cannot be cast to the expected type
        ValueError: If a parameter value is invalid for the expected type
    """
    validated_params = {}

    for param_name, expected_type in param_types.items():
        if param_name not in params:
            continue  # Skip missing parameters - let the calling code handle defaults

        value = params[param_name]

        # If already the correct type, use as-is
        if isinstance(value, expected_type):
            validated_params[param_name] = value
            continue

        # Attempt type casting
        try:
            if expected_type is bool:
                # Handle boolean conversion specially
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes", "on"):
                        validated_params[param_name] = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        validated_params[param_name] = False
                    else:
                        raise ValueError(f"Cannot convert '{value}' to boolean")
                else:
                    validated_params[param_name] = bool(value)
            elif expected_type in (int, float):
                # Handle numeric conversion
                validated_params[param_name] = expected_type(value)
            elif expected_type is str:
                # Handle string conversion
                validated_params[param_name] = str(value)
            else:
                # For other types, try direct casting
                validated_params[param_name] = expected_type(value)

            if logger:
                logger.debug(
                    f"Cast parameter '{param_name}' from {type(value).__name__} to {expected_type.__name__}: {value} -> {validated_params[param_name]}"
                )

        except (ValueError, TypeError) as e:
            error_msg = f"Cannot cast parameter '{param_name}' with value '{value}' (type: {type(value).__name__}) to {expected_type.__name__}: {e}"
            if logger:
                logger.error(error_msg)
            raise TypeError(error_msg)

    return validated_params
