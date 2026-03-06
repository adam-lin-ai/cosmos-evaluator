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

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import status
from fastapi.responses import JSONResponse


class JsonResponseFormatter:
    """JSON response formatter implementation."""

    def __init__(self, include_timestamp: bool = True) -> None:
        """Initializes JSON response formatter.

        Args:
            include_timestamp: Whether to include timestamp in responses
        """
        self.include_timestamp = include_timestamp

    async def format_success(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> JSONResponse:
        """Formats successful JSON response.

        Args:
            data: Response data
            metadata: Optional response metadata

        Returns:
            JSONResponse with formatted data
        """
        response_data = {"success": True, "data": data}

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if metadata:
            response_data["metadata"] = metadata

        return JSONResponse(
            content=response_data, status_code=status.HTTP_200_OK, headers={"Content-Type": "application/json"}
        )

    async def format_error(
        self, error: Exception, status_code: int = 500, metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Formats error JSON response.

        Args:
            error: Exception that occurred
            status_code: HTTP status code
            metadata: Optional response metadata

        Returns:
            JSONResponse with error information
        """
        error_data = {"success": False, "error": {"type": type(error).__name__, "message": str(error)}}

        if self.include_timestamp:
            error_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add additional error details if available
        if hasattr(error, "error_code"):
            error_data["error"]["code"] = error.error_code
        if hasattr(error, "details"):
            error_data["error"]["details"] = error.details
        if hasattr(error, "field"):
            error_data["error"]["field"] = error.field

        if metadata:
            error_data["metadata"] = metadata

        return JSONResponse(content=error_data, status_code=status_code, headers={"Content-Type": "application/json"})

    async def format_progress(self, progress: float, message: str = "") -> JSONResponse:
        """Formats progress JSON response.

        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message

        Returns:
            JSONResponse with progress information
        """
        response_data = {
            "success": True,
            "progress": {
                "percentage": min(100.0, max(0.0, progress)),
                "message": message,
                "complete": progress >= 100.0,
            },
        }

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return JSONResponse(
            content=response_data, status_code=status.HTTP_200_OK, headers={"Content-Type": "application/json"}
        )

    def get_content_type(self) -> str:
        """Gets content type for JSON responses."""
        return "application/json"

    def format_validation_error(self, validation_errors: list, status_code: int = 422) -> JSONResponse:
        """Formats validation error response.

        Args:
            validation_errors: List of validation errors
            status_code: HTTP status code

        Returns:
            JSONResponse with validation error details
        """
        error_data = {
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "validation_errors": validation_errors,
            },
        }

        if self.include_timestamp:
            error_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return JSONResponse(content=error_data, status_code=status_code, headers={"Content-Type": "application/json"})

    def format_paginated_response(
        self, items: list, total: int, page: int, page_size: int, metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Formats paginated response.

        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number (1-based)
            page_size: Number of items per page
            metadata: Optional additional metadata

        Returns:
            JSONResponse with paginated data
        """
        # Validate parameters
        if page_size <= 0:
            raise ValueError("Number of items per page (page_size) must be positive")
        if page < 1:
            raise ValueError("Current page number (page) must be at least 1")
        if total < 0:
            raise ValueError("Total number of items (total) cannot be negative")

        total_pages = (total + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1

        response_data = {
            "success": True,
            "data": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        }

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if metadata:
            response_data["metadata"] = metadata

        return JSONResponse(
            content=response_data, status_code=status.HTTP_200_OK, headers={"Content-Type": "application/json"}
        )
