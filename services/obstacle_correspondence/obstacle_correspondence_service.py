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
ObstacleCorrespondenceService implementation for processing obstacle correspondence analysis.

This service wraps the ObstacleCorrespondenceProcessor and provides a standardized
service interface for deployment as a microservice.
"""

import asyncio
import logging
from pathlib import Path
import tempfile
from typing import Any

import checks.obstacle.api as api
from services.framework.service_base import ServiceBase
from services.obstacle_correspondence.models import (
    ObstacleCorrespondenceRequest,
    ObstacleCorrespondenceResult,
)


class ObstacleCorrespondenceService(ServiceBase[ObstacleCorrespondenceRequest, ObstacleCorrespondenceResult]):
    """
    Service for processing obstacle correspondence analysis.

    This service processes video clips to analyze correspondence between
    segmented objects and tracked obstacles in the world model.
    """

    def __init__(self) -> None:
        """Initialize the service."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    async def validate_input(self, request_data: ObstacleCorrespondenceRequest) -> bool:
        """
        Validate input data for processing.

        Args:
            request_data: Request data to validate

        Returns:
            True if valid for processing

        Raises:
            ValueError: If validation fails
        """
        # Validate required paths exist
        input_path = Path(request_data.input_data_path)
        if not input_path.is_dir():
            raise ValueError(f"Input data directory does not exist: {request_data.input_data_path}")

        video_path = Path(request_data.video_path)
        if not video_path.is_file():
            raise ValueError(f"Video file does not exist: {request_data.video_path}")

        # NOTE: Config is already validated as dict by Pydantic model validation

        # Validate trial_frames if provided
        if request_data.trial_frames is not None and request_data.trial_frames <= 0:
            raise ValueError("trial_frames must be positive if specified")

        return True

    @staticmethod
    async def get_default_config() -> dict[str, Any]:
        """
        Get the default configuration for obstacle correspondence processing.

        Returns:
            Default configuration for obstacle correspondence processing
        """
        return api.get_default_config()

    async def process(self, request_data: ObstacleCorrespondenceRequest) -> ObstacleCorrespondenceResult:
        """
        Process the obstacle correspondence analysis.

        Args:
            request_data: Parsed and validated request data

        Returns:
            Processing result
        """
        output_dir = self._setup_output_dir()
        self.logger.info(f"Starting obstacle correspondence processing for clip {request_data.clip_id}")
        self.logger.info(f"RDS HQ Data: {request_data.input_data_path}")
        self.logger.info(f"Augmented Video: {request_data.video_path}")
        self.logger.info(f"World Model Video: {request_data.world_video_path}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Run obstacle correspondence processing using the unified entry point
            loop = asyncio.get_running_loop()

            # Combine default config with request_data.config, giving preference to request_data.config
            config = request_data.config
            if not config:
                config = api.get_default_config()

            processing_results = await loop.run_in_executor(
                None,
                lambda: api.run_object_processors(
                    config=config,
                    input_data=request_data.input_data_path,
                    clip_id=request_data.clip_id,
                    camera_name=request_data.camera_name,
                    video_path=request_data.video_path,
                    output_dir=str(output_dir),
                    world_video_path=request_data.world_video_path,
                    model_device=request_data.model_device,
                    verbose=request_data.verbose.upper(),  # Convert to uppercase for consistency
                    trial_frames=request_data.trial_frames,
                ),
            )

            # Create result object
            result = ObstacleCorrespondenceResult(
                processed_frames=processing_results.get("dynamic", {}).get("processed_frames", 0),
                total_video_frames=processing_results.get("dynamic", {}).get("total_video_frames", 0),
                mean_score=processing_results.get("dynamic", {}).get("mean_score", 0.0),
                std_score=processing_results.get("dynamic", {}).get("std_score", 0.0),
                min_score=processing_results.get("dynamic", {}).get("min_score", 0.0),
                max_score=processing_results.get("dynamic", {}).get("max_score", 0.0),
                unique_track_ids=sorted(list(processing_results.get("dynamic", {}).get("track_ids", set()))),
                processed_frame_ids=processing_results.get("dynamic", {}).get("processed_frame_ids", []),
                config_summary=request_data.config or {},
                clip_id=request_data.clip_id,
                output_dir=str(output_dir),
            )

            self.logger.info(f"Processing completed successfully for clip {request_data.clip_id}")
            self.logger.info(f"Processed {result.processed_frames} frames with mean score {result.mean_score:.3f}")

            return result

        except Exception as e:
            self.logger.error(f"Error processing clip {request_data.clip_id}: {e}")
            raise

    def _setup_output_dir(self) -> Path:
        """Create a temporary working directory for processing outputs.

        Returns:
            Path to the temporary output directory
        """
        parent_path = Path("/tmp/obstacle_correspondence")
        try:
            parent_path.mkdir(parents=True, exist_ok=True)
            output_dir_path = Path(tempfile.mkdtemp(dir=str(parent_path)))
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

        self.logger.info(f"Using output directory: {output_dir_path}")
        return output_dir_path
