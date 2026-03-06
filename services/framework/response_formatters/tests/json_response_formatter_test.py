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

from datetime import datetime
import json
import sys
from typing import Any, Dict
from unittest.mock import patch

from fastapi import status
from fastapi.responses import JSONResponse
import pytest

from services.framework.protocols.response_formatter import ResponseFormatter
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter


class MockException(Exception):
    """Mock exception for testing."""

    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None, field: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details
        self.field = field


class TestJsonResponseFormatter:
    """Tests suite for JsonResponseFormatter."""

    @pytest.fixture
    def formatter(self):
        """Provides a standard JSON formatter."""
        return JsonResponseFormatter()

    @pytest.fixture
    def no_timestamp_formatter(self):
        """Provides a formatter without timestamps."""
        return JsonResponseFormatter(include_timestamp=False)

    def test_formatter_initialization(self, formatter):
        """Tests that formatter initializes correctly."""
        assert formatter.include_timestamp is True
        assert formatter.get_content_type() == "application/json"

    def test_json_formatter_implements_protocol(self, formatter):
        assert isinstance(formatter, ResponseFormatter)

    def test_formatter_initialization_with_options(self):
        """Tests formatter initialization with custom options."""
        formatter = JsonResponseFormatter(include_timestamp=False)
        assert formatter.include_timestamp is False

    @pytest.mark.asyncio
    async def test_format_success_basic(self, formatter):
        """Tests basic success response formatting."""
        test_data = {"message": "Hello, World!"}

        response = await formatter.format_success(test_data)

        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"

        # Parse response content
        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"] == test_data
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_format_success_with_metadata(self, formatter):
        """Tests success response with metadata."""
        test_data = {"user_id": 123}
        test_metadata = {"request_id": "abc-123", "version": "1.0"}

        response = await formatter.format_success(test_data, test_metadata)

        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"] == test_data
        assert content["metadata"] == test_metadata
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_format_success_without_timestamp(self, no_timestamp_formatter):
        """Tests success response without timestamp."""
        test_data = {"message": "test"}

        response = await no_timestamp_formatter.format_success(test_data)

        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"] == test_data
        assert "timestamp" not in content

    @pytest.mark.asyncio
    async def test_format_success_with_various_data_types(self, formatter):
        """Tests success response with different data types."""
        test_cases = [
            "string_data",
            42,
            3.14,
            True,
            None,
            [1, 2, 3],
            {"nested": {"value": 123}},
            {"list": [{"item": 1}, {"item": 2}]},
        ]

        for test_data in test_cases:
            response = await formatter.format_success(test_data)
            content = json.loads(response.body)
            assert content["success"] is True
            assert content["data"] == test_data

    @pytest.mark.asyncio
    async def test_format_error_basic(self, formatter):
        """Tests basic error response formatting."""
        test_error = ValueError("Something went wrong")

        response = await formatter.format_error(test_error)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500
        assert response.headers["content-type"] == "application/json"

        content = json.loads(response.body)
        assert content["success"] is False
        assert content["error"]["type"] == "ValueError"
        assert content["error"]["message"] == "Something went wrong"
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_format_error_with_status_code(self, formatter):
        """Tests error response with custom status code."""
        test_error = KeyError("Resource not found")

        response = await formatter.format_error(test_error, 404)

        assert response.status_code == 404
        content = json.loads(response.body)
        assert content["error"]["type"] == "KeyError"
        assert content["error"]["message"] == "'Resource not found'"

    @pytest.mark.asyncio
    async def test_format_error_with_additional_attributes(self, formatter):
        """Tests error response with custom exception attributes."""
        test_error = MockException(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            details={"field": "email", "value": "invalid"},
            field="email",
        )

        response = await formatter.format_error(test_error, 422)

        content = json.loads(response.body)
        assert content["error"]["type"] == "MockException"
        assert content["error"]["message"] == "Validation failed"
        assert content["error"]["code"] == "VALIDATION_ERROR"
        assert content["error"]["details"] == {"field": "email", "value": "invalid"}
        assert content["error"]["field"] == "email"

    @pytest.mark.asyncio
    async def test_format_error_with_metadata(self, formatter):
        """Tests error response with metadata."""
        test_error = ValueError("Something went wrong")
        test_metadata = {"duration_ms": 123.45, "request_id": "abc-123"}

        response = await formatter.format_error(test_error, 500, metadata=test_metadata)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        content = json.loads(response.body)
        assert content["success"] is False
        assert content["error"]["type"] == "ValueError"
        assert content["error"]["message"] == "Something went wrong"
        assert content["metadata"] == test_metadata
        assert content["metadata"]["duration_ms"] == 123.45
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_format_error_without_metadata(self, formatter):
        """Tests error response without metadata (default behavior)."""
        test_error = ValueError("Test error")

        response = await formatter.format_error(test_error, 400)

        content = json.loads(response.body)
        assert content["success"] is False
        assert "metadata" not in content

    @pytest.mark.asyncio
    async def test_format_progress_basic(self, formatter):
        """Tests basic progress response formatting."""
        response = await formatter.format_progress(50.0, "Processing data...")

        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_200_OK

        content = json.loads(response.body)
        assert content["success"] is True
        assert content["progress"]["percentage"] == 50.0
        assert content["progress"]["message"] == "Processing data..."
        assert content["progress"]["complete"] is False
        assert "timestamp" in content

    @pytest.mark.asyncio
    async def test_format_progress_edge_values(self, formatter):
        """Tests progress response with edge values."""
        test_cases = [
            (0.0, 0.0, False),  # Start
            (100.0, 100.0, True),  # Complete
            (150.0, 100.0, True),  # Over 100% (clamped)
            (-10.0, 0.0, False),  # Negative (clamped)
            (33.333, 33.333, False),  # Decimal
        ]

        for input_progress, expected_progress, expected_complete in test_cases:
            response = await formatter.format_progress(input_progress, "Test message")
            content = json.loads(response.body)

            assert content["progress"]["percentage"] == expected_progress
            assert content["progress"]["complete"] == expected_complete

    @pytest.mark.asyncio
    async def test_format_progress_without_message(self, formatter):
        """Tests progress response without message."""
        response = await formatter.format_progress(75.0)

        content = json.loads(response.body)
        assert content["progress"]["percentage"] == 75.0
        assert content["progress"]["message"] == ""
        assert content["progress"]["complete"] is False

    def test_format_validation_error(self, formatter):
        """Tests validation error formatting."""
        validation_errors = [
            {"field": "email", "message": "Invalid email format"},
            {"field": "age", "message": "Must be a positive integer"},
        ]

        response = formatter.format_validation_error(validation_errors, 422)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 422

        content = json.loads(response.body)
        assert content["success"] is False
        assert content["error"]["type"] == "ValidationError"
        assert content["error"]["message"] == "Request validation failed"
        assert content["error"]["validation_errors"] == validation_errors

    def test_format_paginated_response(self, formatter):
        """Tests paginated response formatting."""
        items = [{"id": i, "name": f"Item {i}"} for i in range(1, 11)]

        response = formatter.format_paginated_response(
            items=items, total=100, page=2, page_size=10, metadata={"query": "test"}
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == status.HTTP_200_OK

        content = json.loads(response.body)
        assert content["success"] is True
        assert content["data"] == items
        assert content["metadata"] == {"query": "test"}

        pagination = content["pagination"]
        assert pagination["total"] == 100
        assert pagination["page"] == 2
        assert pagination["page_size"] == 10
        assert pagination["total_pages"] == 10
        assert pagination["has_next"] is True
        assert pagination["has_prev"] is True

    def test_format_paginated_response_edge_cases(self, formatter):
        """Tests paginated response with edge cases."""
        # First page
        response = formatter.format_paginated_response([], 50, 1, 10)
        content = json.loads(response.body)
        assert content["pagination"]["has_prev"] is False
        assert content["pagination"]["has_next"] is True

        # Last page
        response = formatter.format_paginated_response([], 50, 5, 10)
        content = json.loads(response.body)
        assert content["pagination"]["has_prev"] is True
        assert content["pagination"]["has_next"] is False

        # Single page
        response = formatter.format_paginated_response([], 5, 1, 10)
        content = json.loads(response.body)
        assert content["pagination"]["total_pages"] == 1
        assert content["pagination"]["has_prev"] is False
        assert content["pagination"]["has_next"] is False

    def test_get_content_type(self, formatter):
        """Tests get_content_type method."""
        assert formatter.get_content_type() == "application/json"

    @pytest.mark.asyncio
    async def test_timestamp_format(self, formatter):
        """Tests that timestamps are in ISO format."""
        with patch("services.framework.response_formatters.json_response_formatter.datetime") as mock_datetime:
            mock_now = datetime(2025, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.isoformat = datetime.isoformat

            response = await formatter.format_success({"test": "data"})
            content = json.loads(response.body)

            assert content["timestamp"] == "2025-01-01T12:00:00"

    @pytest.mark.asyncio
    async def test_large_data_handling(self, formatter):
        """Tests formatting with large data sets."""
        # Create large data structure
        large_data = {"items": [{"id": i, "data": "x" * 100} for i in range(1000)], "metadata": {"size": "large"}}

        response = await formatter.format_success(large_data)

        assert isinstance(response, JSONResponse)
        content = json.loads(response.body)
        assert content["success"] is True
        assert len(content["data"]["items"]) == 1000

    @pytest.mark.asyncio
    async def test_unicode_handling(self, formatter):
        """Tests formatting with unicode data."""
        unicode_data = {
            "name": "José María",
            "description": "测试数据",
            "emoji": "🚀",
            "special": "Special chars: àáâãäåæçèéêë",
        }

        response = await formatter.format_success(unicode_data)
        content = json.loads(response.body)

        assert content["data"] == unicode_data

    @pytest.mark.asyncio
    async def test_nested_data_structures(self, formatter):
        """Tests formatting with deeply nested data."""
        nested_data = {
            "level1": {
                "level2": {"level3": {"level4": {"value": "deep_value", "array": [1, 2, {"nested_in_array": True}]}}}
            }
        }

        response = await formatter.format_success(nested_data)
        content = json.loads(response.body)

        assert content["data"] == nested_data

    def test_multiple_formatter_instances(self):
        """Tests that multiple formatter instances work independently."""
        formatter1 = JsonResponseFormatter(include_timestamp=True)
        formatter2 = JsonResponseFormatter(include_timestamp=False)

        assert formatter1.include_timestamp is True
        assert formatter2.include_timestamp is False

    @pytest.mark.asyncio
    async def test_error_without_timestamp(self, no_timestamp_formatter):
        """Tests error response without timestamp."""
        test_error = ValueError("Test error")

        response = await no_timestamp_formatter.format_error(test_error)
        content = json.loads(response.body)

        assert content["success"] is False
        assert "timestamp" not in content

    @pytest.mark.asyncio
    async def test_progress_without_timestamp(self, no_timestamp_formatter):
        """Tests progress response without timestamp."""
        response = await no_timestamp_formatter.format_progress(50.0, "Test")
        content = json.loads(response.body)

        assert content["success"] is True
        assert "timestamp" not in content

    @pytest.mark.asyncio
    async def test_none_value_converts_to_json_null(self, formatter):
        """Tests that None Python value is converted to null JSON value."""
        test_data = {
            "nullable_field": None,
            "non_null_field": "value",
            "nested": {"also_null": None, "not_null": 42},
            "list_with_nulls": [1, None, "text", None],
        }

        response = await formatter.format_success(test_data)

        # Get the raw JSON string to verify null representation
        json_string = response.body.decode("utf-8")

        # Parse back to verify structure
        content = json.loads(json_string)

        # Verify the data structure is preserved
        assert content["success"] is True
        assert content["data"] == test_data

        # Verify that None values appear as 'null' in the JSON string
        assert '"nullable_field":null' in json_string
        assert '"also_null":null' in json_string
        assert '[1,null,"text",null]' in json_string

        # Verify non-null values are not affected
        assert '"non_null_field":"value"' in json_string
        assert '"not_null":42' in json_string


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
