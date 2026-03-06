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
HallucinationService implementation for processing hallucination detection.

This service wraps the HallucinationProcessor and provides a standardized
service interface for deployment as a microservice.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from checks.hallucination.processor import HallucinationProcessor, HallucinationResult
from checks.utils.multistorage import is_remote_path, validate_uri
from services.framework.service_base import ServiceBase


class HallucinationRequest(BaseModel):
    clip_id: str
    original_video_path: str
    augmented_video_path: str
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration dictionary for processing")
    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")


class HallucinationService(ServiceBase[HallucinationRequest, HallucinationResult]):
    def __init__(self, verbose: str = "INFO") -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, verbose.upper()))

    async def validate_input(self, request: HallucinationRequest) -> bool:
        """
        Validate that both video paths exist on S3 and can be accessed.

        Args:
            request: The hallucination request containing video paths

        Returns:
            True if both paths are valid and accessible

        Raises:
            ValueError: If either video path does not exist
        """
        if "://" in request.original_video_path and not is_remote_path(request.original_video_path):
            raise ValueError(
                'The provided original video path (field "original_video_path") has an unsupported URI scheme'
            )

        if "://" in request.augmented_video_path and not is_remote_path(request.augmented_video_path):
            raise ValueError(
                'The provided augmented video path (field "augmented_video_path") has an unsupported URI scheme'
            )

        # Skip expensive remote existence checks; they are validated during download/open in processor.
        original_exists = True
        if not is_remote_path(request.original_video_path):
            original_exists = await asyncio.to_thread(validate_uri, request.original_video_path, True)
        if not original_exists:
            raise ValueError(
                'The provided original video path (field "original_video_path") does not exist, or cannot be accessed'
            )

        augmented_exists = True
        if not is_remote_path(request.augmented_video_path):
            augmented_exists = await asyncio.to_thread(validate_uri, request.augmented_video_path, True)
        if not augmented_exists:
            raise ValueError(
                'The provided augmented video path (field "augmented_video_path") does not exist, or cannot be accessed'
            )

        return True

    @staticmethod
    async def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for hallucination processing.

        Returns:
            Default configuration for hallucination processing
        """
        config_dir = os.getenv("CONFIG_DIR", None)
        config = (await asyncio.to_thread(HallucinationProcessor.get_default_config, config_dir)).get(
            "metropolis.hallucination", {}
        )
        return config

    async def process(self, request: HallucinationRequest) -> HallucinationResult:
        config = await HallucinationService.get_default_config()
        if request.config is not None:
            config.update(request.config)
        config_dir = os.getenv("CONFIG_DIR", None)
        processor = HallucinationProcessor(params=config, config_dir=config_dir, verbose=request.verbose)
        return await asyncio.to_thread(
            processor.process, request.clip_id, request.original_video_path, request.augmented_video_path
        )
