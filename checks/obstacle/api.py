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
Unified entry point for obstacle correspondence processing.

This module provides a single `run()` function that executes both dynamic (SegFormer-based)
and static (CWIP-based) obstacle processors sequentially, combining their results.
"""

import logging
from typing import Any, Dict, Optional

from checks.obstacle.object_processor_base import ObjectProcessorBase
from checks.obstacle.object_processor_dynamic import ObjectProcessorDynamic
from checks.utils.onnx import clear_gpu_memory

logger = logging.getLogger(__name__)


def run_object_processors(
    config: Dict[str, Any],
    input_data: str,
    clip_id: str,
    camera_name: str,
    video_path: str,
    output_dir: str,
    world_video_path: Optional[str] = None,
    model_device: str = "cuda",
    verbose: str = "INFO",
    trial_frames: Optional[int] = None,
    target_fps: float = 30.0,
) -> Dict[str, Any]:
    """Run both dynamic and static obstacle correspondence processors sequentially.

    This function acts as the single entry point for obstacle correspondence processing.
    It conditionally runs the dynamic processor (if any dynamic objects are configured
    in overlap_check) and the static processor (if any static objects are configured
    and world_video_path is provided), returning combined results.

    Args:
        config: Configuration dictionary (expects key 'av.obstacle').
        input_data: Path to the dataset root or run directory.
        clip_id: Clip identifier.
        camera_name: Camera stream name for poses/intrinsics.
        video_path: Path to the camera video file.
        output_dir: Output directory for results, logs, and visualizations.
        world_video_path: Path to the world video file (required for static processing).
            If None, static processing is skipped.
        model_device: Device to run models on ('cuda' or 'cpu'). Defaults to 'cuda'.
        verbose: Logging verbosity level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to 'INFO'.
        trial_frames: If provided, process at most this many frames (useful for debugging).
        target_fps: Target FPS of object/pose timeline. Defaults to 30.0.

    Returns:
        Dict[str, Any]: Combined results dictionary with keys:
            - 'dynamic': Results from the dynamic (SegFormer) processor (if run).
            - 'static': Results from the static (CWIP) processor (if run).
    """
    results: Dict[str, Any] = {}

    # Extract overlap_check config to determine which processors to run
    oc_config = config.get("av.obstacle", {})
    overlap_check = oc_config.get("overlap_check", {})
    configured_objects = set(overlap_check.keys())

    # Check if any dynamic objects are configured
    has_dynamic_objects = bool(configured_objects & set(ObjectProcessorBase.DYNAMIC_OBJECTS))

    # Check if any static objects are configured
    has_static_objects = bool(configured_objects & set(ObjectProcessorBase.STATIC_OBJECTS))

    # Run dynamic processor if any dynamic objects are configured
    if has_dynamic_objects:
        dynamic_processor = ObjectProcessorDynamic(
            clip_id=clip_id,
            output_dir=output_dir,
            config=config,
            model_device=model_device,
            verbose=verbose,
        )
        results["dynamic"] = dynamic_processor.process_clip(
            input_data=input_data,
            clip_id=clip_id,
            camera_name=camera_name,
            video_path=video_path,
            output_dir=output_dir,
            trial_frames=trial_frames,
            target_fps=target_fps,
        )

        # Delete the dynamic processor to release ONNX session and GPU memory
        # This is critical before running CWIP to avoid GPU OOM
        del dynamic_processor

    # Clear GPU memory between processors to reduce fragmentation and prevent OOM errors
    # This is especially important when switching between different models (SegFormer -> CWIP)
    if has_dynamic_objects and has_static_objects and world_video_path is not None:
        clear_gpu_memory()

    # Run static processor if any static objects are configured and world_video_path is provided
    if has_static_objects and world_video_path is not None:
        logger.warning("Static object processing is not yet supported.")

    return results


def get_default_config() -> Dict[str, Any]:
    """Load the default configuration from config.yaml.

    Returns:
        Dict[str, Any]: Dict with key 'av.obstacle' containing obstacle config.
    """
    return ObjectProcessorBase.get_default_config()
