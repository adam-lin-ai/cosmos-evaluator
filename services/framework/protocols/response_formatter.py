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
Protocol for formatting different types of HTTP responses.
"""

from typing import Any, Dict, Optional, Protocol, TypeVar, runtime_checkable

ResponseType = TypeVar("ResponseType")


@runtime_checkable
class ResponseFormatter(Protocol[ResponseType]):
    """Protocol for formatting different types of HTTP responses."""

    async def format_success(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ResponseType:
        """Formats successful response.

        Args:
            data: Response data
            metadata: Optional response metadata

        Returns:
            Formatted response of type ResponseType
        """
        pass

    async def format_error(self, error: Exception, status_code: int = 500) -> ResponseType:
        """Formats error response.

        Args:
            error: Exception that occurred
            status_code: HTTP status code

        Returns:
            Formatted error response
        """
        pass

    async def format_progress(self, progress: float, message: str = "") -> ResponseType:
        """Formats progress response for long-running operations.

        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message

        Returns:
            Formatted progress response
        """
        pass

    def get_content_type(self) -> str:
        """Gets the content type for responses.

        Returns:
            MIME type string
        """
        pass
