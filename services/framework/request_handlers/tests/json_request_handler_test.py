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
import sys
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel, Field
import pytest

from services.framework.protocols.request_handler import RequestHandler
from services.framework.request_handlers.json_request_handler import JsonRequestHandler, RequestValidationError


# Test models for validation
class SimpleModel(BaseModel):
    """Simple test model."""

    name: str
    age: int


class ComplexModel(BaseModel):
    """Complex test model with various field types."""

    id: int
    name: str
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    active: bool = True
    tags: List[str] = []
    metadata: Optional[dict] = None


class OptionalFieldsModel(BaseModel):
    """Model with optional fields."""

    required_field: str
    optional_field: Optional[str] = None
    default_value: int = 42


class MockRequest:
    """Mock FastAPI Request object for testing."""

    def __init__(self, body_content: bytes = b"", headers: dict = None):
        self._body = body_content
        self.headers = headers if headers is not None else {"content-type": "application/json"}

    async def body(self) -> bytes:
        """Return the request body."""
        return self._body


class TestJsonRequestHandler:
    """Tests suite for JsonRequestHandler."""

    @pytest.fixture
    def simple_handler(self):
        """Provides a handler for simple model."""
        return JsonRequestHandler(SimpleModel)

    @pytest.fixture
    def complex_handler(self):
        """Provides a handler for complex model."""
        return JsonRequestHandler(ComplexModel)

    @pytest.fixture
    def optional_handler(self):
        """Provides a handler for model with optional fields."""
        return JsonRequestHandler(OptionalFieldsModel)

    def test_json_handler_implements_protocol(self, simple_handler, complex_handler):
        assert isinstance(simple_handler, RequestHandler)
        assert isinstance(complex_handler, RequestHandler)

    def test_handler_initialization(self, simple_handler):
        """Tests that handler initializes correctly."""
        assert simple_handler.model_class == SimpleModel
        assert simple_handler.get_content_type() == "application/json"

    @pytest.mark.asyncio
    async def test_parse_valid_simple_json(self, simple_handler):
        """Tests parsing valid simple JSON request."""
        test_data = {"name": "Alice", "age": 30}
        request = MockRequest(body_content=json.dumps(test_data).encode(), headers={"content-type": "application/json"})

        result = await simple_handler.parse(request)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_parse_valid_complex_json(self, complex_handler):
        """Tests parsing valid complex JSON request."""
        test_data = {
            "id": 123,
            "name": "Bob",
            "email": "bob@example.com",
            "active": True,
            "tags": ["user", "premium"],
            "metadata": {"source": "api", "version": 1},
        }
        request = MockRequest(body_content=json.dumps(test_data).encode(), headers={"content-type": "application/json"})

        result = await complex_handler.parse(request)

        assert isinstance(result, ComplexModel)
        assert result.id == 123
        assert result.name == "Bob"
        assert result.email == "bob@example.com"
        assert result.active is True
        assert result.tags == ["user", "premium"]
        assert result.metadata == {"source": "api", "version": 1}

    @pytest.mark.asyncio
    async def test_parse_with_optional_fields(self, optional_handler):
        """Tests parsing JSON with optional fields."""
        # Test with all fields
        test_data = {"required_field": "value", "optional_field": "optional_value", "default_value": 100}
        request = MockRequest(body_content=json.dumps(test_data).encode())

        result = await optional_handler.parse(request)
        assert result.required_field == "value"
        assert result.optional_field == "optional_value"
        assert result.default_value == 100

        # Test with only required field
        test_data_minimal = {"required_field": "value"}
        request_minimal = MockRequest(body_content=json.dumps(test_data_minimal).encode())

        result_minimal = await optional_handler.parse(request_minimal)
        assert result_minimal.required_field == "value"
        assert result_minimal.optional_field is None
        assert result_minimal.default_value == 42  # Default value

    @pytest.mark.asyncio
    async def test_parse_empty_body_error(self, simple_handler):
        """Tests that empty body raises error."""
        request = MockRequest(body_content=b"")

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        assert "Empty request body" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_invalid_json_error(self, simple_handler):
        """Tests that invalid JSON raises error."""
        request = MockRequest(body_content=b'{"invalid": json}')

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_wrong_content_type_error(self, simple_handler):
        """Tests that wrong content type raises error."""
        test_data = {"name": "Alice", "age": 30}
        request = MockRequest(body_content=json.dumps(test_data).encode(), headers={"content-type": "application/xml"})

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        error = exc_info.value
        assert "Expected application/json" in error.message
        assert error.field == "content-type"
        assert error.value == "application/xml"

    @pytest.mark.asyncio
    async def test_parse_missing_content_type(self, simple_handler):
        """Tests handling of missing content-type header."""
        test_data = {"name": "Alice", "age": 30}
        request = MockRequest(
            body_content=json.dumps(test_data).encode(),
            headers={},  # No content-type header
        )

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        error = exc_info.value
        assert "Expected application/json" in error.message
        assert error.field == "content-type"
        assert error.value == ""

    @pytest.mark.asyncio
    async def test_parse_validation_error_missing_field(self, simple_handler):
        """Tests validation error for missing required field."""
        test_data = {"name": "Alice"}  # Missing age field
        request = MockRequest(body_content=json.dumps(test_data).encode())

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        error = exc_info.value
        assert "Validation failed" in error.message
        assert "validation_errors" in error.details

    @pytest.mark.asyncio
    async def test_parse_validation_error_wrong_type(self, simple_handler):
        """Tests validation error for wrong field type."""
        test_data = {"name": "Alice", "age": "thirty"}  # age should be int
        request = MockRequest(body_content=json.dumps(test_data).encode())

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        error = exc_info.value
        assert "Validation failed" in error.message
        assert "validation_errors" in error.details

    @pytest.mark.asyncio
    async def test_parse_validation_error_invalid_email(self, complex_handler):
        """Tests validation error for invalid email format."""
        test_data = {
            "id": 123,
            "name": "Bob",
            "email": "invalid-email",  # Invalid email format
        }
        request = MockRequest(body_content=json.dumps(test_data).encode())

        with pytest.raises(RequestValidationError) as exc_info:
            await complex_handler.parse(request)

        error = exc_info.value
        assert "Validation failed" in error.message
        assert "validation_errors" in error.details

    @pytest.mark.asyncio
    async def test_parse_extra_fields_allowed(self, simple_handler):
        """Tests that extra fields are ignored by default."""
        test_data = {
            "name": "Alice",
            "age": 30,
            "extra_field": "ignored",  # Extra field
        }
        request = MockRequest(body_content=json.dumps(test_data).encode())

        result = await simple_handler.parse(request)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30
        # Extra field should be ignored
        assert not hasattr(result, "extra_field")

    @pytest.mark.asyncio
    async def test_validate_method(self, simple_handler):
        """Tests the validate method."""
        # Create a valid model instance
        model = SimpleModel(name="Alice", age=30)

        # Validate should return True for valid model
        result = await simple_handler.validate(model)
        assert result is True

    def test_get_content_type(self, simple_handler):
        """Tests get_content_type method."""
        assert simple_handler.get_content_type() == "application/json"

    @pytest.mark.asyncio
    async def test_parse_with_different_json_content_types(self, simple_handler):
        """Tests parsing with different JSON content type variations."""
        test_data = {"name": "Alice", "age": 30}

        # Test various JSON content types
        content_types = [
            "application/json",
            "application/json; charset=utf-8",
            "application/json;charset=utf-8",
            "APPLICATION/JSON",  # Case insensitive
        ]

        for content_type in content_types:
            request = MockRequest(body_content=json.dumps(test_data).encode(), headers={"content-type": content_type})

            result = await simple_handler.parse(request)
            assert isinstance(result, SimpleModel)
            assert result.name == "Alice"
            assert result.age == 30

    @pytest.mark.asyncio
    async def test_parse_unicode_content(self, simple_handler):
        """Tests parsing JSON with unicode characters."""
        test_data = {"name": "José María", "age": 25}
        request = MockRequest(body_content=json.dumps(test_data, ensure_ascii=False).encode("utf-8"))

        result = await simple_handler.parse(request)

        assert isinstance(result, SimpleModel)
        assert result.name == "José María"
        assert result.age == 25

    @pytest.mark.asyncio
    async def test_parse_nested_objects(self, complex_handler):
        """Tests parsing JSON with nested objects."""
        test_data = {
            "id": 123,
            "name": "Alice",
            "email": "alice@example.com",
            "metadata": {"nested": {"value": 42, "description": "deeply nested"}, "array": [1, 2, 3]},
        }
        request = MockRequest(body_content=json.dumps(test_data).encode())

        result = await complex_handler.parse(request)

        assert isinstance(result, ComplexModel)
        assert result.metadata["nested"]["value"] == 42
        assert result.metadata["array"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_parse_large_json(self, simple_handler):
        """Tests parsing large JSON payload."""
        # Create a large JSON object
        large_data = {
            "name": "Alice",
            "age": 30,
            "large_field": "x" * 10000,  # 10KB string
        }
        request = MockRequest(body_content=json.dumps(large_data).encode())

        result = await simple_handler.parse(request)

        assert isinstance(result, SimpleModel)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_parse_error_exception_details(self, simple_handler):
        """Tests that RequestValidationError contains proper details."""
        test_data = {"name": 123, "age": "invalid"}  # Both fields wrong type
        request = MockRequest(body_content=json.dumps(test_data).encode())

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(request)

        error = exc_info.value
        assert error.message is not None
        assert error.details is not None
        assert "validation_errors" in error.details
        assert isinstance(error.details["validation_errors"], list)
        assert len(error.details["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_multiple_handlers_independence(self):
        """Tests that multiple handler instances work independently."""
        handler1 = JsonRequestHandler(SimpleModel)
        handler2 = JsonRequestHandler(ComplexModel)

        # Test data for each handler
        simple_data = {"name": "Alice", "age": 30}
        complex_data = {"id": 123, "name": "Bob", "email": "bob@example.com"}

        request1 = MockRequest(body_content=json.dumps(simple_data).encode())
        request2 = MockRequest(body_content=json.dumps(complex_data).encode())

        # Parse with both handlers
        result1 = await handler1.parse(request1)
        result2 = await handler2.parse(request2)

        # Verify results
        assert isinstance(result1, SimpleModel)
        assert result1.name == "Alice"

        assert isinstance(result2, ComplexModel)
        assert result2.name == "Bob"
        assert result2.email == "bob@example.com"

    @pytest.mark.asyncio
    async def test_handler_with_generic_exception(self, simple_handler):
        """Tests handler behavior with unexpected exceptions."""
        # Create a mock request that will cause an unexpected error
        mock_request = Mock()
        mock_request.headers.get.return_value = "application/json"
        mock_request.body = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with pytest.raises(RequestValidationError) as exc_info:
            await simple_handler.parse(mock_request)

        error = exc_info.value
        assert "Request parsing failed" in error.message
        assert "Unexpected error" in error.message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
