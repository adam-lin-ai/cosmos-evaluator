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
Render World Model video from RDS HQ data.

Example:
    output_paths = render_world_model_video(
        rds_hq_path="/path/to/rds_hq_data",
        output_dir="/path/to/output",
        clip_id="my_clip_001",
        camera_names=["camera_front_wide_120fov", "camera_front_tele_30fov"],
    )
    # Returns: [
    #     "/path/to/output/world_scenario/ftheta_camera_front_wide_120fov/my_clip_001_0.mp4",
    #     "/path/to/output/world_scenario/ftheta_camera_front_wide_120fov/my_clip_001_1.mp4",
    #     "/path/to/output/world_scenario/ftheta_camera_front_tele_30fov/my_clip_001_0.mp4",
    #     "/path/to/output/world_scenario/ftheta_camera_front_tele_30fov/my_clip_001_1.mp4",
    # ]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CameraSpec:
    """Camera specification for rendering."""

    camera_list: List[str] = field(default_factory=lambda: ["camera_0"])
    camera_to_rds_hq: Optional[dict] = None


@dataclass
class InputFpsSpec:
    """Input FPS specification."""

    input_pose_fps: int = 30
    input_lidar_fps: int = 10
    gt_video_fps: int = 30


@dataclass
class ResolutionSpec:
    """Resolution specification for rendering."""

    camera_model_resolution: List[int] = field(default_factory=lambda: [1280, 720])
    cosmos_resolution: List[int] = field(default_factory=lambda: [1280, 720])
    to_cosmos_resolution: str = "resize"


@dataclass
class OutputFrameSpec:
    """Output frame specification."""

    target_render_fps: int = 10
    target_chunk_frame: int = 100
    overlap_frame: int = 45
    max_chunk: int = 3


@dataclass
class S3Spec:
    """S3 upload specification."""

    enable_upload: bool = False
    s3_path: Optional[str] = None
    s3_credentials_path: Optional[str] = None


@dataclass
class MinimapSpec:
    """Minimap types specification."""

    minimap_types: List[str] = field(
        default_factory=lambda: [
            "lanelines",
            "road_boundaries",
            "crosswalks",
            "road_markings",
            "wait_lines",
            "poles",
            "traffic_signs",
            "traffic_lights",
        ]
    )


@dataclass
class RenderConfig:
    """Configuration for RDS-HQ rendering.

    This is a simplified version of CosmosAVConfig used internally.
    """

    input_root: str
    output_root: str
    camera_spec: CameraSpec = field(default_factory=CameraSpec)
    input_fps_spec: InputFpsSpec = field(default_factory=InputFpsSpec)
    resolution_spec: ResolutionSpec = field(default_factory=ResolutionSpec)
    output_frame_spec: OutputFrameSpec = field(default_factory=OutputFrameSpec)
    s3_spec: S3Spec = field(default_factory=S3Spec)
    minimap_spec: MinimapSpec = field(default_factory=MinimapSpec)
    camera_type: str = "ftheta"
    novel_pose_folder: Optional[str] = None


def render_world_model_video(
    rds_hq_path: str,
    output_dir: str,
    clip_id: str,
    camera_names: List[str],
) -> List[str]:
    """Render world model videos from RDS-HQ data using render_sample_hdmap_v3.

    Args:
        rds_hq_path: Path to the extracted RDS-HQ dataset directory.
        output_dir: Directory where the rendered videos will be saved.
        clip_id: The clip identifier (e.g., "clip_001").
        camera_names: List of camera names to render (e.g., ["camera_front_wide_120fov"]).

    Returns:
        List of paths to the rendered video files, or empty list if rendering fails.
    """
    # Lazy import to avoid loading heavy dependencies at module import time
    # This allows tests to mock this function without triggering the actual import
    try:
        from third_party.cosmos_drive_dreams_toolkits.render_from_rds_hq import render_sample_hdmap_v3
    except ImportError as e:
        import logging

        logging.warning(
            f"Cannot import render_sample_hdmap_v3: {e}. "
            "WM video rendering is unavailable. Ensure all dependencies are installed."
        )
        return []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build configuration with hardcoded defaults matching the original repo
    cfg = RenderConfig(
        input_root=rds_hq_path,
        output_root=output_dir,
        camera_spec=CameraSpec(camera_list=camera_names),
    )

    # Call the render function using Ray remote
    # render_sample_hdmap_v3 is decorated with @ray.remote, so we need to call .remote()
    import os

    import ray

    # Suppress Ray metrics agent warnings (set before ray.init)
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    os.environ.setdefault("RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING", "0")

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,  # Disable dashboard to reduce overhead
            logging_level="ERROR",  # Only show errors, suppress warnings
            log_to_driver=False,  # Disable worker logs to driver
        )

    try:
        result_ref = render_sample_hdmap_v3.remote(
            clip_id=clip_id,
            cfg=cfg,
        )
        ray.get(result_ref)  # Wait for completion
    except Exception as e:
        # Old clip data format may not be compatible with render_sample_hdmap_v3
        print(f"WM camera render error: {e}")
        print(f"Warning: Failed to render world model video for {clip_id}")
        return None

    # The output is saved to: output_dir/world_scenario/{camera_type}_{camera_name}/{clip_id}_{chunk_idx}.mp4
    # Collect all rendered video files
    wm_video_paths = []
    for camera_name in camera_names:
        rendered_dir = output_path / "world_scenario" / f"{cfg.camera_type}_{camera_name}"
        if rendered_dir.exists():
            # Find all mp4 files in the directory
            for video_file in sorted(rendered_dir.glob("*.mp4")):
                wm_video_paths.append(str(video_file))

    return wm_video_paths
