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

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest

from services.framework.service_base import ServiceBase


class TestService(ServiceBase[dict, str]):
    """Minimal test implementation of ServiceBase for testing purposes."""

    def __init__(self):
        """Initialize the test service."""
        super().__init__()

    async def process(self, request: dict) -> str:
        """Returns a fixed 'processed' string for testing."""
        return "processed"

    async def validate_input(self, request: dict) -> bool:
        """Always validates input as True for testing."""
        return True


@pytest.mark.asyncio
async def test_service_process():
    service = TestService()
    result = await service.process({"test": "data"})
    assert result == "processed"


@pytest.mark.asyncio
async def test_service_validate_input():
    service = TestService()
    is_valid = await service.validate_input({"test": "data"})
    assert is_valid is True


@pytest.mark.asyncio
async def test_service_optional_methods():
    service = TestService()

    # Test cleanup doesn't raise
    with TemporaryDirectory() as temp_dir:
        await service.cleanup(Path(temp_dir))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
