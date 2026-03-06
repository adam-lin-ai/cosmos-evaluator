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

"""
Protocol for handling different types of HTTP requests.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

RequestType = TypeVar("RequestType")


@runtime_checkable
class RequestHandler(Protocol[RequestType]):
    """Protocol for handling different types of HTTP requests."""

    async def parse(self, raw_request: Any) -> RequestType:
        """Parses raw HTTP request into structured data.

        Args:
            raw_request: Raw HTTP request object (FastAPI Request, etc.)

        Returns:
            Parsed request data of type RequestType

        Raises:
            RequestValidationError: If request is invalid
        """
        pass

    async def validate(self, request_data: RequestType) -> bool:
        """Validates parsed request data.

        Args:
            request_data: Parsed request data

        Returns:
            True if valid, False otherwise

        Raises:
            RequestValidationError: If validation fails
        """
        pass

    def get_content_type(self) -> str:
        """Gets the expected content type for this handler.

        Returns:
            MIME type string (e.g., 'application/json')
        """
        pass
