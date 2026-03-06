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
import tempfile

from fastapi.responses import JSONResponse, StreamingResponse
import pytest

from services.framework.protocols.response_formatter import ResponseFormatter
from services.framework.response_formatters.streaming_response_formatter import StreamingResponseFormatter


class AsyncDataGenerator:
    """Mock async data generator for testing."""

    def __init__(self, items: list):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestStreamingResponseFormatter:
    """Tests suite for StreamingResponseFormatter."""

    @pytest.fixture
    def formatter(self):
        """Provides a standard streaming formatter."""
        return StreamingResponseFormatter()

    @pytest.fixture
    def custom_formatter(self):
        """Provides a custom configured formatter."""
        return StreamingResponseFormatter(chunk_size=2048, include_metadata=False, newline_delimited=False)

    def test_streaming_formatter_implements_protocol(self, formatter):
        """Test that the formatter has the required protocol methods"""
        assert isinstance(formatter, ResponseFormatter)

    def test_formatter_initialization(self, formatter):
        """Tests that formatter initializes correctly."""
        assert formatter.chunk_size == 1024
        assert formatter.include_metadata is True
        assert formatter.newline_delimited is True
        assert formatter.get_content_type() == "application/x-ndjson"

    def test_formatter_initialization_with_options(self):
        """Tests formatter initialization with custom options."""
        formatter = StreamingResponseFormatter(chunk_size=4096, include_metadata=False, newline_delimited=False)
        assert formatter.chunk_size == 4096
        assert formatter.include_metadata is False
        assert formatter.newline_delimited is False

    @pytest.mark.asyncio
    async def test_format_success_with_list(self, formatter):
        """Tests streaming success response with list data."""
        test_data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

        response = await formatter.format_success(test_data)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/x-ndjson"
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    @pytest.mark.asyncio
    async def test_format_success_with_async_generator(self, formatter):
        """Tests streaming success response with async generator."""
        test_data = AsyncDataGenerator([{"item": 1}, {"item": 2}, {"item": 3}])

        response = await formatter.format_success(test_data)

        assert isinstance(response, StreamingResponse)

        # Collect chunks
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode("utf-8"))

        # Should have metadata, start, data chunks, and end
        assert len(chunks) >= 5  # metadata + start + 3 data + end

    @pytest.mark.asyncio
    async def test_format_success_with_metadata(self, formatter):
        """Tests streaming success response with metadata."""
        test_data = [{"value": 1}, {"value": 2}]
        test_metadata = {"source": "test", "version": "1.0"}

        response = await formatter.format_success(test_data, test_metadata)

        # Collect and parse first chunk (should be metadata)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(json.loads(chunk.decode("utf-8")))

        # First chunk should be metadata
        metadata_chunk = chunks[0]
        assert metadata_chunk["type"] == "metadata"
        assert metadata_chunk["metadata"] == test_metadata

    @pytest.mark.asyncio
    async def test_format_success_without_metadata(self, custom_formatter):
        """Tests streaming success response without metadata."""
        test_data = [{"value": 1}]

        response = await custom_formatter.format_success(test_data)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        # Should not have metadata chunk
        assert chunks[0]["type"] == "start"

    @pytest.mark.asyncio
    async def test_format_success_single_item(self, formatter):
        """Tests streaming success response with single item."""
        test_data = {"single": "item"}

        response = await formatter.format_success(test_data)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        # Find data chunk
        data_chunks = [c for c in chunks if c["type"] == "data"]
        assert len(data_chunks) == 1
        assert data_chunks[0]["data"] == test_data
        assert data_chunks[0]["sequence"] == 0

    @pytest.mark.asyncio
    async def test_chunk_sequence_numbering(self, formatter):
        """Tests that data chunks are properly sequenced."""
        test_data = [{"item": i} for i in range(5)]

        response = await formatter.format_success(test_data)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        data_chunks = [c for c in chunks if c["type"] == "data"]
        assert len(data_chunks) == 5

        for i, chunk in enumerate(data_chunks):
            assert chunk["sequence"] == i
            assert chunk["data"] == {"item": i}

    @pytest.mark.asyncio
    async def test_stream_structure(self, formatter):
        """Tests complete stream structure."""
        test_data = [{"item": 1}, {"item": 2}]
        test_metadata = {"test": "metadata"}

        response = await formatter.format_success(test_data, test_metadata)

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        # Verify stream structure
        assert chunks[0]["type"] == "metadata"
        assert chunks[1]["type"] == "start"
        assert chunks[2]["type"] == "data"
        assert chunks[3]["type"] == "data"
        assert chunks[4]["type"] == "end"

        # Verify end chunk
        end_chunk = chunks[4]
        assert end_chunk["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_format_error_fallback(self, formatter):
        """Tests that error formatting falls back to JSON."""
        test_error = ValueError("Stream error")

        response = await formatter.format_error(test_error, 400)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_format_progress(self, formatter):
        """Tests progress response formatting."""
        response = await formatter.format_progress(75.0, "Processing...")

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/x-ndjson"

        # Get the single chunk
        chunk = None
        async for chunk_bytes in response.body_iterator:
            chunk = json.loads(chunk_bytes.decode("utf-8"))
            break

        assert chunk["type"] == "progress"
        assert chunk["progress"]["percentage"] == 75.0
        assert chunk["progress"]["message"] == "Processing..."
        assert chunk["progress"]["complete"] is False

    @pytest.mark.asyncio
    async def test_format_server_sent_events(self, formatter):
        """Tests Server-Sent Events formatting."""
        test_data = [{"event": 1}, {"event": 2}]

        response = await formatter.format_server_sent_events(iter(test_data), event_type="notification", retry_ms=5000)

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert "Access-Control-Allow-Origin" in response.headers

        # Collect SSE data
        sse_content = b""
        async for chunk in response.body_iterator:
            sse_content += chunk

        sse_text = sse_content.decode("utf-8")
        assert "retry: 5000" in sse_text
        assert "event: notification" in sse_text
        assert "id: 0" in sse_text
        assert "id: 1" in sse_text

    @pytest.mark.asyncio
    async def test_format_server_sent_events_async(self, formatter):
        """Tests SSE formatting with async iterator."""
        test_data = AsyncDataGenerator([{"async": 1}, {"async": 2}])

        response = await formatter.format_server_sent_events(test_data, "async_event")

        sse_content = b""
        async for chunk in response.body_iterator:
            sse_content += chunk

        sse_text = sse_content.decode("utf-8")
        assert "event: async_event" in sse_text
        assert '"async":1' in sse_text
        assert '"async":2' in sse_text

    def test_format_sse_event(self, formatter):
        """Tests individual SSE event formatting."""
        event_data = {"message": "Hello"}

        sse_event = formatter._format_sse_event(event_data, "test_event", 42)

        expected = 'id: 42\nevent: test_event\ndata: {"message":"Hello"}\n\n'
        assert sse_event == expected

    @pytest.mark.asyncio
    async def test_format_chunked_file(self, formatter):
        """Tests chunked file download formatting."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("This is test file content for streaming.")
            temp_path = f.name

        try:
            response = await formatter.format_chunked_file(temp_path, content_type="text/plain")

            assert isinstance(response, StreamingResponse)
            assert response.media_type == "text/plain"
            assert "Content-Length" in response.headers

            # Read file content
            file_content = b""
            async for chunk in response.body_iterator:
                file_content += chunk

            assert b"This is test file content for streaming." in file_content

        finally:
            import os

            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_format_chunked_file_not_found(self, formatter):
        """Tests chunked file download with non-existent file."""
        response = await formatter.format_chunked_file("/nonexistent/file.txt")

        # Should still return StreamingResponse but with error content
        assert isinstance(response, StreamingResponse)

        content = b""
        async for chunk in response.body_iterator:
            content += chunk

        error_data = json.loads(content.decode("utf-8"))
        assert error_data["error"] == "File not found"

    @pytest.mark.asyncio
    async def test_format_log_stream(self, formatter):
        """Tests log streaming formatting."""

        async def log_generator():
            logs = [
                "INFO: Application started",
                "DEBUG: Loading configuration",
                "ERROR: Database connection failed",
                "INFO: Retrying connection",
            ]
            for log in logs:
                yield log

        response = await formatter.format_log_stream(log_generator())

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/x-ndjson"

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk["type"] == "log"
            assert "line" in chunk
            assert "timestamp" in chunk

    @pytest.mark.asyncio
    async def test_format_log_stream_with_filter(self, formatter):
        """Tests log streaming with level filtering."""

        async def log_generator():
            logs = [
                "INFO: Application started",
                "DEBUG: Loading configuration",
                "ERROR: Database connection failed",
                "INFO: Retrying connection",
            ]
            for log in logs:
                yield log

        response = await formatter.format_log_stream(log_generator(), filter_level="ERROR")

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        # Should only have ERROR log
        assert len(chunks) == 1
        assert "ERROR" in chunks[0]["line"]

    def test_format_chunk_with_newlines(self, formatter):
        """Tests chunk formatting with newlines enabled."""
        test_chunk = {"type": "test", "data": "value"}

        formatted = formatter._format_chunk(test_chunk)

        assert formatted.endswith(b"\n")
        # Remove newline and parse
        json_data = json.loads(formatted[:-1].decode("utf-8"))
        assert json_data == test_chunk

    def test_format_chunk_without_newlines(self, custom_formatter):
        """Tests chunk formatting without newlines."""
        test_chunk = {"type": "test", "data": "value"}

        formatted = custom_formatter._format_chunk(test_chunk)

        assert not formatted.endswith(b"\n")
        json_data = json.loads(formatted.decode("utf-8"))
        assert json_data == test_chunk

    def test_format_chunk_unicode(self, formatter):
        """Tests chunk formatting with unicode data."""
        test_chunk = {"message": "Hello 世界 🌍", "emoji": "🚀"}

        formatted = formatter._format_chunk(test_chunk)

        json_data = json.loads(formatted.decode("utf-8"))
        assert json_data == test_chunk

    def test_get_content_type(self, formatter):
        """Tests get_content_type method."""
        assert formatter.get_content_type() == "application/x-ndjson"

    @pytest.mark.asyncio
    async def test_large_data_streaming(self, formatter):
        """Tests streaming with large dataset."""
        # Generate large dataset
        large_data = [{"id": i, "data": "x" * 100} for i in range(1000)]

        response = await formatter.format_success(large_data)

        chunk_count = 0
        data_chunks = 0
        async for chunk in response.body_iterator:
            chunk_count += 1
            chunk_data = json.loads(chunk.decode("utf-8"))
            if chunk_data["type"] == "data":
                data_chunks += 1

        assert data_chunks == 1000  # Should have 1000 data chunks
        assert chunk_count > 1000  # Plus metadata, start, end chunks

    @pytest.mark.asyncio
    async def test_empty_data_streaming(self, formatter):
        """Tests streaming with empty data."""
        response = await formatter.format_success([])

        chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            chunks.append(chunk_data)

        # Should have metadata, start, and end (no data chunks)
        data_chunks = [c for c in chunks if c["type"] == "data"]
        assert len(data_chunks) == 0

        end_chunk = [c for c in chunks if c["type"] == "end"][0]
        assert end_chunk["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_mixed_data_types_streaming(self, formatter):
        """Tests streaming with mixed data types."""
        mixed_data = ["string", 42, {"object": True}, [1, 2, 3], None]

        response = await formatter.format_success(mixed_data)

        data_chunks = []
        async for chunk in response.body_iterator:
            chunk_data = json.loads(chunk.decode("utf-8"))
            if chunk_data["type"] == "data":
                data_chunks.append(chunk_data["data"])

        assert data_chunks == mixed_data

    def test_multiple_formatter_instances(self):
        """Tests that multiple formatter instances work independently."""
        formatter1 = StreamingResponseFormatter(chunk_size=1024, include_metadata=True)
        formatter2 = StreamingResponseFormatter(chunk_size=2048, include_metadata=False)

        assert formatter1.chunk_size == 1024
        assert formatter1.include_metadata is True
        assert formatter2.chunk_size == 2048
        assert formatter2.include_metadata is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
