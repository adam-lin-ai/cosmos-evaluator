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
FastAPI application for the ObstacleCorrespondenceService.

This application provides a REST API for the ObstacleCorrespondenceService,
providing both synchronous and streaming endpoints for obstacle correspondence
processing.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, AsyncGenerator, Final
import zipfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from services.framework.protocols.storage_provider import StorageProvider, StorageUrls
from services.framework.request_handlers.json_request_handler import RequestValidationError
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter
from services.framework.storage_providers.factory import build_storage_provider
from services.obstacle_correspondence.models import (
    ObstacleCorrespondenceCloudRequest,
    ObstacleCorrespondenceCloudResult,
    ObstacleCorrespondenceRequest,
    ObstacleCorrespondenceResult,
)
from services.obstacle_correspondence.obstacle_correspondence_service import ObstacleCorrespondenceService
from services.obstacle_correspondence.settings import get_settings
from services.settings_base import Environment
import services.utils as utils

# Get settings instance
settings = get_settings()

# Create logger
logger = logging.getLogger(__name__)

# Global service instance
service: ObstacleCorrespondenceService | None = None

# Storage provider for input downloads (initialized during lifespan)
storage: StorageProvider | None = None

# Dedicated /process capacity limit. Requests over this limit are rejected immediately.
PROCESS_CONCURRENCY_LIMIT: Final[int] = settings.process_concurrency_limit
_process_slots_lock: asyncio.Lock | None = None
_active_process_requests: int = 0


async def _try_acquire_process_slot() -> bool:
    """Try to acquire a /process slot without waiting."""
    global _active_process_requests
    if _process_slots_lock is None:
        raise RuntimeError("Process slot lock is not initialized. Service startup may have failed.")
    async with _process_slots_lock:
        if _active_process_requests >= PROCESS_CONCURRENCY_LIMIT:
            return False
        _active_process_requests += 1
        return True


async def _release_process_slot() -> None:
    """Release a previously acquired /process slot."""
    global _active_process_requests
    if _process_slots_lock is None:
        raise RuntimeError("Process slot lock is not initialized. Service startup may have failed.")
    async with _process_slots_lock:
        _active_process_requests = max(0, _active_process_requests - 1)


async def _process_request(
    api_request: ObstacleCorrespondenceCloudRequest,
) -> tuple[ObstacleCorrespondenceRequest, Path]:
    """Downloads/resolves input files and prepares the request for processing.

    Args:
        api_request: Request object containing cloud storage URLs or local file paths

    Returns:
        Tuple of (internal_request, temp_dir) where temp_dir should be cleaned up after processing
    """
    # Create a temporary directory for this request
    temp_dir = Path(tempfile.mkdtemp(prefix="obstacle_correspondence_"))

    logger.info("Processing request using temp directory: {}".format(temp_dir))
    try:
        if not storage:
            logger.error("Storage provider not initialized")
            raise ValueError("Storage provider not initialized")

        # Download RDS HQ zip file
        rds_hq_zip_path = temp_dir / "rds_hq.zip"
        await storage.download_from_url(api_request.rds_hq_url, rds_hq_zip_path)

        # Extract RDS HQ zip file
        rds_hq_extracted_path = temp_dir / "rds_hq_extracted"
        logger.info("Extracting RDS HQ zip to {}".format(rds_hq_extracted_path))
        with zipfile.ZipFile(rds_hq_zip_path, "r") as zip_ref:
            for info in zip_ref.infolist():
                if ".." in info.filename or info.filename.startswith("/"):
                    raise ValueError(f"Zip file contains unsafe path: {info.filename}")
            zip_ref.extractall(rds_hq_extracted_path)

        # Extract clip ID from the tar files
        clip_id = utils.extract_clip_id(rds_hq_extracted_path)

        # Download video file
        video_path = temp_dir / "generated_video.mp4"
        await storage.download_from_url(api_request.augmented_video_url, video_path)

        world_video_path: str | None = None
        if api_request.world_model_video_url:
            world_video_download_path = temp_dir / "world_video.mp4"
            await storage.download_from_url(api_request.world_model_video_url, world_video_download_path)
            world_video_path = str(world_video_download_path)

        # Create the internal request object
        internal_request = ObstacleCorrespondenceRequest(
            input_data_path=str(rds_hq_extracted_path),
            clip_id=clip_id,
            camera_name=api_request.camera_name,
            video_path=str(video_path),
            world_video_path=world_video_path,
            config=api_request.config,
            model_device=api_request.model_device,
            verbose=api_request.verbose,
            trial_frames=api_request.trial_frames,
            output_storage_prefix=api_request.output_storage_prefix,
        )

        return internal_request, temp_dir

    except Exception as e:
        logger.error("Error processing request: {}".format(e))
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up temp directory: {}".format(temp_dir))
        raise e


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for service initialization and cleanup.

    Args:
        _: FastAPI application instance
    """
    global service, storage, _process_slots_lock, _active_process_requests

    # Startup
    logger.info(f"Initializing ObstacleCorrespondenceService for {settings.env.value} environment...")

    service = ObstacleCorrespondenceService()
    storage = build_storage_provider(settings)
    _process_slots_lock = asyncio.Lock()
    _active_process_requests = 0

    logger.info("Service initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down service...")
    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Obstacle Correspondence Service API",
    description="REST API for obstacle correspondence analysis using computer vision and world model data",
    version=utils.get_contents_from_runfile("services/obstacle_correspondence/version.txt"),
    lifespan=lifespan,
    debug=settings.env == Environment.LOCAL,  # Enable debug in local
)

# Initialize response formatters
json_formatter = JsonResponseFormatter()


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return await json_formatter.format_success(
        {
            "service": app.title,
            "status": "healthy",
            "version": app.version,
            "git_sha": utils.get_git_sha(),
            "environment": settings.env.value,
        }
    )


@app.post("/process", response_model=dict[str, Any])
async def process(request: ObstacleCorrespondenceCloudRequest) -> JSONResponse:
    """
    Process obstacle correspondence analysis (synchronous).

    This endpoint processes a complete clip and returns the final results.
    Use this for smaller clips or when you don't need progress updates.

    Args:
        request: Request object containing the request body

    Returns:
        JSONResponse containing the result of the obstacle correspondence analysis

    Raises:
        RequestValidationError: If the request is invalid
        ValueError: If the input is invalid
        Exception: If there is an error processing the request
    """
    global service

    if not service:
        logger.error("Service not initialized")
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    has_process_slot = await _try_acquire_process_slot()
    if not has_process_slot:
        logger.warning(
            f"Rejecting /process request because capacity is full ({PROCESS_CONCURRENCY_LIMIT} concurrent requests)"
        )
        return await json_formatter.format_error(
            Exception("Obstacle Correspondence /process is at capacity. Please retry."),
            status_code=503,
        )

    start_time = time.perf_counter()
    temp_dir: Path | None = None

    local_request: ObstacleCorrespondenceRequest | None = None
    result: ObstacleCorrespondenceResult | None = None

    logger.info("Processing ObstacleCorrespondenceCloudRequest")
    try:
        local_request, temp_dir = await _process_request(request)

        # Validate obstacle correspondence service input
        logger.info("Validating obstacle correspondence service input: {}".format(local_request))
        await service.validate_input(local_request)

        logger.info("Processing obstacle correspondence...")
        result = await service.process(local_request)

        # Upload generated outputs via StorageProvider
        prefix = request.output_storage_prefix or f"obstacles/{result.clip_id}/"
        output_dir_path = Path(result.output_dir)

        results_json_outputs: dict[str, tuple[Path, str]] = {
            "objects.dynamic": (
                output_dir_path / f"{result.clip_id}.dynamic.object.results.json",
                f"{result.clip_id}.dynamic.object.results.json",
            ),
            "objects.static": (
                output_dir_path / f"{result.clip_id}.static.object.results.json",
                f"{result.clip_id}.static.object.results.json",
            ),
        }

        viz_video_outputs: dict[str, tuple[Path, str]] = {
            "objects.dynamic": (
                output_dir_path / f"{result.clip_id}.dynamic.object.mp4",
                f"{result.clip_id}.dynamic.object.mp4",
            ),
            "objects.static": (
                output_dir_path / f"{result.clip_id}.static.object.mp4",
                f"{result.clip_id}.static.object.mp4",
            ),
        }

        results_json_storage_urls: dict[str, StorageUrls] = {}
        viz_video_storage_urls: dict[str, StorageUrls] = {}

        output_provider = build_storage_provider(settings, key_prefix=prefix)
        async with output_provider as provider:
            for output_key, (local_path, storage_key) in results_json_outputs.items():
                if local_path.exists():
                    results_json_storage_urls[output_key] = await provider.store_file(
                        local_path, storage_key, "application/json"
                    )
                else:
                    logger.warning("Results JSON not found at: {}\nSkipping upload.".format(local_path))

            for output_key, (local_path, storage_key) in viz_video_outputs.items():
                if local_path.exists():
                    viz_video_storage_urls[output_key] = await provider.store_file(local_path, storage_key, "video/mp4")
                else:
                    logger.warning("Visualization video not found at: {}\nSkipping upload.".format(local_path))

        cloud_result = ObstacleCorrespondenceCloudResult(
            **result.model_dump(),
            results_json_presigned_urls={
                key: urls.presigned for key, urls in results_json_storage_urls.items() if urls.presigned
            },
            results_json_urls={key: urls.raw for key, urls in results_json_storage_urls.items() if urls.raw},
            visualizations_presigned_urls={
                key: urls.presigned for key, urls in viz_video_storage_urls.items() if urls.presigned
            },
            visualizations_urls={key: urls.raw for key, urls in viz_video_storage_urls.items() if urls.raw},
        )

        response_data = cloud_result.model_dump(mode="json")

        logger.debug(
            "Processed frames: {}, Total video frames: {}".format(result.processed_frames, result.total_video_frames)
        )
        processing_rate = (
            f"{(100.0 * result.processed_frames / result.total_video_frames):.1f}%"
            if result.total_video_frames > 0
            else "null"
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Sending response for clip ID: {}".format(result.clip_id))
        return await json_formatter.format_success(
            data=response_data, metadata={"processing_rate": processing_rate, "duration_ms": duration_ms}
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        metadata = {"duration_ms": duration_ms}
        if isinstance(e, RequestValidationError):
            logger.error("Request validation error: {}".format(e))
            return await json_formatter.format_error(e, status_code=400, metadata=metadata)
        elif isinstance(e, ValueError):
            logger.error("Value error: {}".format(e))
            return await json_formatter.format_error(e, status_code=422, metadata=metadata)
        else:
            logger.error("Processing error: {}".format(e))
            return await json_formatter.format_error(e, status_code=500, metadata=metadata)
    finally:
        await _release_process_slot()
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info("Cleaned up temp directory: {}".format(temp_dir))
            except Exception as cleanup_error:
                logger.warning("Error cleaning up temp directory: {}".format(cleanup_error))

        if result:
            await service.cleanup(Path(result.output_dir))


@app.get("/config")
async def get_default_config() -> JSONResponse:
    """
    Get the default configuration for obstacle correspondence processing.

    This endpoint provides the default configuration for obstacle correspondence processing.
    """
    if not service:
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    default_config = await service.get_default_config()

    return await json_formatter.format_success(
        {
            "default_config": default_config,
            "description": "Default configuration for obstacle correspondence processing",
        }
    )
