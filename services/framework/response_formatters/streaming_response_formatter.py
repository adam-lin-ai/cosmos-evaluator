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
import json
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union

import aiofiles
from fastapi.responses import JSONResponse, StreamingResponse

from .json_response_formatter import JsonResponseFormatter


class StreamingResponseFormatter:
    """Streaming response formatter for large or real-time data."""

    def __init__(self, chunk_size: int = 1024, include_metadata: bool = True, newline_delimited: bool = True) -> None:
        """Initializes streaming response formatter.

        Args:
            chunk_size: Size of each chunk in bytes
            include_metadata: Whether to include metadata as first chunk
            newline_delimited: Whether to add newlines between chunks
        """
        self.chunk_size = chunk_size
        self.include_metadata = include_metadata
        self.newline_delimited = newline_delimited
        self._json_formatter = JsonResponseFormatter()

    async def format_success(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> StreamingResponse:
        """Formats successful streaming response.

        Args:
            data: Response data (should be iterable or generator)
            metadata: Optional response metadata

        Returns:
            StreamingResponse with data chunks
        """

        async def generate_chunks():
            # Send metadata first if enabled and provided
            if self.include_metadata and metadata:
                metadata_chunk = {
                    "type": "metadata",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata,
                }
                yield self._format_chunk(metadata_chunk)

            # Send start marker
            start_chunk = {"type": "start", "timestamp": datetime.now(timezone.utc).isoformat()}
            yield self._format_chunk(start_chunk)

            # Stream data
            chunk_count = 0
            if hasattr(data, "__aiter__"):
                # Async iterable
                async for item in data:
                    data_chunk = {"type": "data", "sequence": chunk_count, "data": item}
                    yield self._format_chunk(data_chunk)
                    chunk_count += 1
            elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes, dict)):
                # Regular iterable
                for item in data:
                    data_chunk = {"type": "data", "sequence": chunk_count, "data": item}
                    yield self._format_chunk(data_chunk)
                    chunk_count += 1
            else:
                # Single value
                data_chunk = {"type": "data", "sequence": 0, "data": data}
                yield self._format_chunk(data_chunk)
                chunk_count = 1

            # Send end marker
            end_chunk = {
                "type": "end",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_chunks": chunk_count,
            }
            yield self._format_chunk(end_chunk)

        return StreamingResponse(
            generate_chunks(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    async def format_error(self, error: Exception, status_code: int = 500) -> JSONResponse:
        """Formats error response (falls back to JSON for errors).

        Args:
            error: Exception that occurred
            status_code: HTTP status code

        Returns:
            JSONResponse with error information
        """
        # For errors, fall back to JSON response
        return await self._json_formatter.format_error(error, status_code)

    async def format_progress(self, progress: float, message: str = "") -> StreamingResponse:
        """Formats progress streaming response.

        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message

        Returns:
            StreamingResponse with progress updates
        """

        async def generate_progress():
            progress_chunk = {
                "type": "progress",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "progress": {
                    "percentage": min(100.0, max(0.0, progress)),
                    "message": message,
                    "complete": progress >= 100.0,
                },
            }
            yield self._format_chunk(progress_chunk)

        return StreamingResponse(
            generate_progress(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    def get_content_type(self) -> str:
        """Gets content type for streaming responses."""
        return "application/x-ndjson"

    def _format_chunk(self, chunk: Dict[str, Any]) -> bytes:
        """Formats a single chunk for streaming.

        Args:
            chunk: Data chunk to format

        Returns:
            Formatted chunk as bytes
        """
        json_str = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        if self.newline_delimited:
            json_str += "\n"
        return json_str.encode("utf-8")

    async def format_server_sent_events(
        self, data: Union[AsyncIterator[Any], Iterator[Any]], event_type: str = "data", retry_ms: Optional[int] = None
    ) -> StreamingResponse:
        """Formats Server-Sent Events (SSE) response.

        Args:
            data: Data iterator
            event_type: SSE event type
            retry_ms: Retry interval in milliseconds

        Returns:
            StreamingResponse formatted for SSE
        """

        async def generate_sse():
            # Send retry directive if specified
            if retry_ms is not None:
                yield f"retry: {retry_ms}\n\n".encode("utf-8")

            # Stream events
            event_id = 0
            if hasattr(data, "__aiter__"):
                async for item in data:
                    yield self._format_sse_event(item, event_type, event_id).encode("utf-8")
                    event_id += 1
            else:
                for item in data:
                    yield self._format_sse_event(item, event_type, event_id).encode("utf-8")
                    event_id += 1

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    def _format_sse_event(self, data: Any, event_type: str, event_id: int) -> str:
        """Formats a single SSE event.

        Args:
            data: Event data
            event_type: Event type
            event_id: Event ID

        Returns:
            Formatted SSE event string
        """
        # Convert data to JSON string
        data_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        # Format SSE event
        event_str = f"id: {event_id}\n"
        event_str += f"event: {event_type}\n"
        event_str += f"data: {data_str}\n\n"

        return event_str

    async def format_chunked_file(
        self, file_path: str, content_type: str = "application/octet-stream"
    ) -> StreamingResponse:
        """Formats chunked file download response.

        Args:
            file_path: Path to file
            content_type: MIME content type

        Returns:
            StreamingResponse for file download
        """

        async def generate_file_chunks():
            try:
                async with aiofiles.open(file_path, "rb") as file:
                    while True:
                        chunk = await file.read(self.chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except FileNotFoundError:
                # Handle file not found
                error_chunk = json.dumps({"error": "File not found", "path": file_path}).encode("utf-8")
                yield error_chunk

        # Get file size for Content-Length header
        try:
            import os

            file_size = os.path.getsize(file_path)
            headers = {"Content-Length": str(file_size), "Content-Type": content_type}
        except (OSError, FileNotFoundError):
            headers = {"Content-Type": content_type}

        return StreamingResponse(generate_file_chunks(), media_type=content_type, headers=headers)

    async def format_log_stream(
        self, log_generator: AsyncIterator[str], filter_level: Optional[str] = None
    ) -> StreamingResponse:
        """Formats log streaming response.

        Args:
            log_generator: Async generator yielding log lines
            filter_level: Optional log level filter (DEBUG, INFO, WARN, ERROR)

        Returns:
            StreamingResponse with log stream
        """

        async def generate_log_stream():
            async for log_line in log_generator:
                # Apply level filtering if specified
                if filter_level and filter_level.upper() not in log_line.upper():
                    continue

                log_chunk = {
                    "type": "log",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "line": log_line.strip(),
                }
                yield self._format_chunk(log_chunk)

        return StreamingResponse(
            generate_log_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
