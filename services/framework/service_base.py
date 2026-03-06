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

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import shutil
from typing import Generic, TypeVar

RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class ServiceBase(ABC, Generic[RequestType, ResponseType]):
    """Abstract base class for microservices."""

    def __init__(self):
        """Initialize the service."""
        self.logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)

    @abstractmethod
    async def process(self, request: RequestType) -> ResponseType:
        """Process the business logic.

        Args:
            request: Parsed and validated request

        Returns:
            Processing result
        """
        pass

    @abstractmethod
    async def validate_input(self, request: RequestType) -> bool:
        """Validate input data for processing.

        Args:
            request: Request data to validate

        Returns:
            True if valid for processing
        """
        pass

    async def cleanup(self, output_dir: Path) -> None:
        """Clean up the specified output directory."""
        try:
            if output_dir and output_dir.exists() and output_dir.is_dir():
                shutil.rmtree(output_dir)
                self.logger.info(f"Cleaned up output directory: {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup output directory: {e}")
