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

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from checks.vlm.preset_processor import PresetProcessor, process_preset
from services.framework.service_base import ServiceBase
from services.framework.url_security import (
    InsecureUrlError,
    hostname_resolves_to_private_ip,
    is_private_or_reserved_ip,
)


class PresetRequest(BaseModel):
    augmented_video_url: str = Field(..., description="URL or path to the augmented video file", min_length=1)
    preset_conditions: dict[str, Any] = Field(..., description="Environment preset conditions")
    preset_check_config: dict[str, Any] | None = Field(
        default=None, description="Environment preset check configuration"
    )

    @field_validator("augmented_video_url")
    @classmethod
    def validate_url_security(cls, v: str) -> str:
        """Reject unencrypted HTTP and URLs targeting private/reserved IP addresses."""
        parsed = urlparse(v)

        if parsed.scheme == "http":
            raise InsecureUrlError("Unencrypted HTTP URLs are not allowed; use HTTPS instead")

        if parsed.scheme == "https":
            hostname = parsed.hostname
            if not hostname:
                raise InsecureUrlError("HTTPS URL must include a hostname")

            if is_private_or_reserved_ip(hostname):
                raise InsecureUrlError(f"URLs targeting private/reserved IP addresses are not allowed: {hostname}")

            if hostname_resolves_to_private_ip(hostname):
                raise InsecureUrlError(f"URL hostname '{hostname}' resolves to a private/reserved IP address")

        return v


class PresetResponse(BaseModel):
    result: dict[str, Any] = Field(..., description="Environment preset result")


# Combine the different request and response models into a single overall VLM request and response
class Request(BaseModel):
    preset_request: Optional[PresetRequest] = Field(None, description="Preset request")


class Response(BaseModel):
    preset_response: PresetResponse = Field(..., description="Preset response")


class Service(ServiceBase[Request, Response]):
    def __init__(self):
        """Initialize the VLM service, which will consist of multiple processors."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    async def validate_input(self, request: Request) -> bool:
        """
        Validate input data for processing.

        Args:
            request: Request data to validate

        Returns:
            True if valid for processing
        """
        if request.preset_request and not self.validate_input_preset(request.preset_request):
            return False
        return True

    @staticmethod
    def validate_input_preset(request: PresetRequest) -> bool:
        """
        Validate input data for preset processing.
        """
        if not request.augmented_video_url or not request.augmented_video_url.strip():
            return False
        if not request.preset_conditions:
            return False
        return True

    async def process_preset(self, request: PresetRequest, video_path: str) -> PresetResponse:
        """
        Process the environment preset.

        Args:
            request: The request containing the video URL, preset conditions, and preset check configuration
            video_path: Local filesystem path to the video file.

        Returns:
            The environment preset result
        """
        self.logger.info("Processing environment preset")
        self.logger.info("Video path: {}".format(video_path))
        self.logger.info("Preset conditions: {}".format(request.preset_conditions))
        self.logger.info("Preset check configuration overrides: {}".format(request.preset_check_config))

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            process_preset,  # from checks.vlm.preset_processor
            video_path,
            request.preset_conditions,
            request.preset_check_config,
        )
        return PresetResponse(result=result)

    async def process(self, request: Request) -> Response:
        """Not used — the REST API layer calls :meth:`process_preset` directly."""
        raise NotImplementedError("Use process_preset() via the REST layer instead")

    @staticmethod
    async def get_default_config() -> dict[str, Any]:
        """
        Get the default configuration for VLM processing that includes all processors.

        Returns:
            Default configuration for VLM processing that includes all processors
        """
        return {"av.vlm": PresetProcessor.get_default_config()}
