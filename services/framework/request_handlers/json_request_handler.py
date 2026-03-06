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

import json
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ValidationError


class RequestValidationError(Exception):
    """Exception raised when request validation fails."""

    def __init__(self, message: str, field: str = None, value: Any = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.details = details or {}


class JsonRequestHandler:
    """Request handler for JSON payloads."""

    def __init__(self, model_class: type[BaseModel]) -> None:
        """Initializes with a Pydantic model class for validation.

        Args:
            model_class: Pydantic model class for request validation
        """
        self.model_class = model_class

    async def parse(self, raw_request: Request) -> BaseModel:
        """Parses JSON request into Pydantic model.

        Args:
            raw_request: FastAPI Request object

        Returns:
            Parsed and validated Pydantic model instance

        Raises:
            RequestValidationError: If JSON is invalid or validation fails
        """
        try:
            # Check content type
            content_type = raw_request.headers.get("content-type", "")
            if not content_type.lower().startswith("application/json"):
                raise RequestValidationError(
                    f"Expected application/json, got {content_type}", field="content-type", value=content_type
                )

            # Parse JSON body
            body = await raw_request.body()
            if not body:
                raise RequestValidationError("Empty request body")

            try:
                json_data = json.loads(body)
            except json.JSONDecodeError as e:
                raise RequestValidationError(f"Invalid JSON: {e}") from e

            # Validate with Pydantic model
            return self.model_class(**json_data)

        except ValidationError as e:
            raise RequestValidationError(f"Validation failed: {e}", details={"validation_errors": e.errors()}) from e
        except RequestValidationError:
            raise
        except Exception as e:
            raise RequestValidationError(f"Request parsing failed: {e}") from e

    async def validate(self, request_data: BaseModel) -> bool:
        """Validates parsed request data (already validated by Pydantic).

        Args:
            request_data: Parsed request data

        Returns:
            True if valid
        """
        # Pydantic already validated during parsing
        return True

    def get_content_type(self) -> str:
        """Gets expected content type."""
        return "application/json"
