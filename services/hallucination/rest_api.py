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
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from checks.utils.multistorage import setup_msc_config
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter
from services.hallucination.hallucination_service import (
    HallucinationRequest,
    HallucinationService,
)
from services.hallucination.settings import get_settings
from services.settings_base import Environment, LogLevel
import services.utils as utils

settings = get_settings()

# Create logger
logger = logging.getLogger(__name__)

# Global service instance
service: Optional[HallucinationService] = None
process_concurrency_limit = settings.process_concurrency_limit
_inflight_count = 0
_inflight_lock = asyncio.Lock()

log_level: LogLevel = LogLevel.INFO


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for service initialization and cleanup.

    Args:
        _: FastAPI application instance
    """
    global service

    # Startup
    logger.info(f"Initializing HallucinationService for {settings.env.value} environment...")

    service = HallucinationService(verbose=log_level.value)
    setup_msc_config(os.environ.get("MULTISTORAGECLIENT_CONFIGURATION"))

    logger.info("Service initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down service...")
    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Hallucination Service API",
    description="REST API for hallucination detection",
    version=utils.get_contents_from_runfile("services/hallucination/version.txt"),
    lifespan=lifespan,
    debug=settings.env == Environment.LOCAL,  # Enable debug in local
)

json_formatter = JsonResponseFormatter()


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    if not service:
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    return await json_formatter.format_success(
        {
            "service": app.title,
            "status": "healthy",
            "version": app.version,
            "git_sha": utils.get_git_sha(),
            "environment": settings.env.value,
        }
    )


@app.post("/process", response_model=Dict[str, Any])
async def process(request: HallucinationRequest) -> JSONResponse:
    """
    Process hallucination detection (synchronous).

    Args:
        request: Request object containing the request body
    """
    if not service:
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    global _inflight_count
    admitted = False
    async with _inflight_lock:
        if _inflight_count >= process_concurrency_limit:
            response = await json_formatter.format_error(
                Exception("Service is at capacity. Please retry."),
                status_code=503,
            )
            response.headers["Retry-After"] = "1"
            return response
        _inflight_count += 1
        admitted = True

    start_time = time.perf_counter()
    try:
        try:
            # Validate hallucination service input
            logger.info("Validating hallucination service input: {}".format(request))
            await service.validate_input(request)
        except Exception as e:
            return await json_formatter.format_error(e, status_code=400)

        try:
            result = await service.process(request)
        except Exception as e:
            return await json_formatter.format_error(e, status_code=500)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return await json_formatter.format_success(
            data=result.model_dump(mode="json"), metadata={"duration_ms": duration_ms}
        )
    finally:
        if admitted:
            async with _inflight_lock:
                _inflight_count -= 1


@app.get("/config")
async def get_default_config() -> JSONResponse:
    """
    Get the default configuration for hallucination detection.

    This endpoint provides the default configuration for hallucination detection.

    Returns:
        JSONResponse containing the default configuration for hallucination processing
    """
    if not service:
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    default_config = await service.get_default_config()

    return await json_formatter.format_success(
        {"default_config": default_config, "description": "Default configuration for hallucination processing"}
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Hallucination API")
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UVICORN_PORT", "8080")),
        choices=range(1, 65536),
        help="Port to bind to",
        metavar="[1-65535]",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.value,
        choices=[log_level.value for log_level in LogLevel],
        help="Log level",
    )

    args = parser.parse_args()

    log_level = LogLevel(args.log_level)
    logger.setLevel(log_level.value)
    logger.propagate = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.value)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    logger.info(f"Starting Hallucination API on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
