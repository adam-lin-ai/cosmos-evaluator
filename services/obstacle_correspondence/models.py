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
Obstacle Correspondence Request and Response Models.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer


# Base request model
class ObstacleCorrespondenceRequestBase(BaseModel):
    """Base request model for obstacle correspondence processing."""

    # Required fields
    camera_name: str = Field(..., description="Camera name for processing")

    config: dict[str, Any] | None = Field(default=None, description="Configuration dictionary for processing")
    model_device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device to run models on")
    trial_frames: int | None = Field(default=None, description="Limit number of frames for testing")
    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")

    # Output storage
    output_storage_prefix: str | None = Field(
        default=None,
        description="Storage prefix for outputs (e.g., 'my-team/run-42/obstacles/'). "
        "For cloud storage this is a key prefix; for local storage this is a directory path. "
        "If omitted, defaults to '{service_name}/{clip_id}/'.",
    )


# Request model that accepts local paths
class ObstacleCorrespondenceRequest(ObstacleCorrespondenceRequestBase):
    """Request model for obstacle correspondence processing."""

    # Required fields
    clip_id: str = Field(..., description="Clip ID to process")
    input_data_path: str = Field(..., description="Path to input data directory")
    video_path: str = Field(..., description="Path to video file")
    world_video_path: str | None = Field(
        default=None, description="Path to world video file. If omitted, static processing is skipped."
    )


# Request model that accepts cloud storage URLs or local file paths
class ObstacleCorrespondenceCloudRequest(ObstacleCorrespondenceRequestBase):
    """Request model for obstacle correspondence processing with cloud storage URLs or local file paths."""

    # Required fields
    rds_hq_url: str = Field(..., description="Presigned URL of RDS HQ zip file")
    augmented_video_url: str = Field(..., description="URL or path to the augmented video MP4 file")
    world_model_video_url: str | None = Field(
        default=None,
        description="URL or path to the world model video MP4 file. If omitted, static processing is skipped.",
    )


class ObstacleCorrespondenceResult(BaseModel):
    """Result model for local obstacle correspondence processing."""

    # Processing statistics
    processed_frames: int = Field(..., description="Number of frames processed")
    total_video_frames: int = Field(..., description="Total frames in video")

    # Score statistics
    mean_score: float = Field(..., description="Mean correspondence score")
    std_score: float = Field(..., description="Standard deviation of scores")
    min_score: float = Field(..., description="Minimum correspondence score")
    max_score: float = Field(..., description="Maximum correspondence score")

    # Track information
    unique_track_ids: list[int] = Field(..., description="List of unique track IDs found")
    processed_frame_ids: list[int] = Field(..., description="List of frame IDs that were processed")

    # Configuration summary
    config_summary: dict[str, Any] = Field(..., description="Summary of processing configuration")

    # Paths and metadata
    clip_id: str = Field(..., description="Processed clip ID")
    output_dir: str = Field(..., description="Output directory used")

    # Configure JSON serialization to replace NaN and infinite values with null
    # Required due to a bug in Pydantic 2.11.7
    # https://github.com/pydantic/pydantic/issues/10037
    @field_serializer("mean_score", "std_score", "min_score", "max_score", when_used="json")
    def serialize_float_values(self, value: float) -> float | None:
        """Convert NaN and infinity values to null when serializing to JSON.

        Args:
            value: The float value to serialize.

        Returns:
            The serialized float value, or None if the value is NaN or infinity.
        """
        if math.isnan(value) or math.isinf(value):
            return None
        return value


class ObstacleCorrespondenceCloudResult(ObstacleCorrespondenceResult):
    """Result model for obstacle correspondence processing with storage URLs."""

    results_json_presigned_urls: dict[str, str] = Field(
        default={}, description="Presigned URLs of generated results JSON files"
    )
    results_json_urls: dict[str, str] = Field(default={}, description="Storage URLs of generated results JSON files")
    visualizations_presigned_urls: dict[str, str] = Field(
        default={}, description="Presigned URLs of generated visualizations"
    )
    visualizations_urls: dict[str, str] = Field(default={}, description="Storage URLs of generated visualizations")
