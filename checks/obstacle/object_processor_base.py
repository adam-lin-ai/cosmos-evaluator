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
Base class for obstacle processors.

This class provides shared functionality for obstacle processors, including logging, data loading, and visualization.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from checks.obstacle.importance_filter import ImportanceFilter
from checks.obstacle.results import save_results_to_json
from checks.obstacle.visualization_helper import VisualizationHelper
from checks.utils.config_manager import ConfigManager
from checks.utils.profiler import Profiler
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer
from checks.utils.segformer import get_dataloader


class ObjectProcessorBase:
    """
    Base processor providing shared functionality for obstacle processing.

    Holds common configuration, logging, helpers, profiling, and utilities
    used by both dynamic (SegFormer-based) and static (CWIP-based) processors.
    """

    DYNAMIC_OBJECTS: ClassVar[list[str]] = [
        "bicycle",
        "motorcycle",
        "pedestrian",
        "vehicle",
    ]

    STATIC_OBJECTS: ClassVar[list[str]] = [
        "crosswalk",
        "lane_line",
        "road_boundary",
        "traffic_light",
        "traffic_sign",
        "wait_line",
    ]

    EGO_REFERENCE_CAMERA: str = "front_center"

    def __init__(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any] | None = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
    ):
        """Initialize the base obstacle processor.

        Args:
            clip_id: Clip identifier used for logging/visualization file names.
            output_dir: Directory where outputs (logs, videos) are written.
            config: Processor configuration dictionary (expects key 'av.obstacle').
            model_device: Torch/ONNX device to run models on, e.g. 'cuda' or 'cpu'.
            verbose: Logging verbosity level (e.g. 'DEBUG', 'INFO').
        """
        self.model_device = model_device
        self.verbose = verbose
        self.clip_id = clip_id
        self.output_dir = output_dir
        self.naming_suffix = "base"

        # Extract configuration
        updated_config = ObjectProcessorBase.get_default_config()
        if config is not None:
            updated_config["av.obstacle"].update(config.get("av.obstacle", {}))
        self.oc_config = updated_config["av.obstacle"]

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize shared components
        self.seg_model = None

        # Initialize filters for tests (and potential external use)
        try:
            importance_cfg = self.oc_config.get("importance_filter", {})
            self.importance_filter = ImportanceFilter(importance_cfg)
        except Exception as e:
            self.logger.error(f"Failed to initialize ImportanceFilter: {e}")
            self.importance_filter = None

        # Initialize visualization helper
        self.vis_helper = None
        if self.oc_config.get("visualization", {}).get("enabled", True):
            vis_verbose = verbose == "DEBUG"
            self.vis_helper = VisualizationHelper(clip_id, output_dir, vis_verbose)
        else:
            self.logger.info("Visualization generation disabled.")

        # DEBUG flag retained for logging verbosity only
        self.debug_enabled = verbose == "DEBUG"

        # Hallucination classes relevant to this processor; overridden by subclasses
        self._hallucination_classes: dict[str, Any] = {}

        # Initialize profiler
        self.profiler = Profiler()

    def setup_logging(self):
        """Setup logging handlers and levels.

        Notes:
        - Uses module logger with both console and optional file handlers.
        - File logging is enabled when clip_id and output_dir are provided.
        - In DEBUG mode we also enable verbose ONNX logging.
        - Expected to be called manually after initialization to allow for overriding the log naming suffix.
        """

        # Set logging level based on verbose mode
        log_level = getattr(logging, self.verbose.upper())
        self.logger.setLevel(log_level)

        # Ensure propagation is enabled
        self.logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if clip_id and output_dir are provided
        if self.clip_id and self.output_dir:
            log_file_path = Path(self.output_dir) / f"{self.clip_id}.{self.naming_suffix}.object.log"

            # Clear the log file by opening in write mode (truncates the file)
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file_path}")

            if self.verbose == "DEBUG":
                self.logger.debug("DEBUG logging is enabled")
                from checks.utils.onnx import configure_onnx_logging

                configure_onnx_logging(verbose=True)

    def _init_data_loader(self, input_data: str, clip_id: str) -> RdsDataLoader:
        """Create and time the `RdsDataLoader` initialization.

        Args:
            input_data (str): Path to the dataset root or run directory.
            clip_id (str): Identifier of the clip to load from the dataset.

        Returns:
            RdsDataLoader: Initialized data loader for the clip.
        """
        self.profiler.start("data_loader_init")
        self.logger.info(f"Loading data for clip {clip_id}")
        loader = RdsDataLoader(input_data, clip_id)
        self.profiler.end()
        return loader

    def _load_camera_data(self, data_loader: RdsDataLoader, camera_name: str):
        """Load camera poses and intrinsics used for projection.

        Args:
            data_loader (RdsDataLoader): Loader providing pose and intrinsic streams.
            camera_name (str): Name of the camera stream to load (e.g. 'front').

        Returns:
            tuple: `(camera_poses, camera_model)` where `camera_poses` is an array/dict of poses
            and `camera_model` is the intrinsic model object used for projection.
        """
        self.profiler.start("camera_data_load")
        camera_poses = data_loader.get_camera_poses(camera_name)
        camera_model = data_loader.get_camera_intrinsics(camera_name, "ftheta")
        self.profiler.end()
        return camera_poses, camera_model

    def _load_ego_reference_poses(self, data_loader: RdsDataLoader, camera_name: str):
        """Load ego-aligned reference poses for importance filtering.

        For non-forward-facing cameras (e.g. front_left, rear_right), the camera
        coordinate frame is rotated relative to the ego vehicle.  Lane assignment
        and oncoming detection require an ego-aligned frame, so we load the
        front_center camera poses as a proxy when the evaluation camera differs.

        Args:
            data_loader: Loader providing pose streams.
            camera_name: The camera currently being evaluated.

        Returns:
            dict | None: Mapping from frame index to 4x4 ego-reference poses,
            or None when the evaluation camera is already the reference camera.
        """
        if camera_name == self.EGO_REFERENCE_CAMERA:
            return None
        try:
            ego_poses = data_loader.get_camera_poses(self.EGO_REFERENCE_CAMERA)
            self.logger.info(
                f"Loaded {len(ego_poses)} ego-reference poses from '{self.EGO_REFERENCE_CAMERA}' "
                f"for importance filtering (evaluation camera: '{camera_name}')"
            )
            return ego_poses
        except Exception as e:
            self.logger.warning(
                f"Could not load ego-reference poses from '{self.EGO_REFERENCE_CAMERA}': {e}. "
                f"Falling back to camera poses for importance filtering."
            )
            return None

    def _prepare_video(self, video_path: str, data_loader: RdsDataLoader, target_fps: Optional[float]):
        """Prepare the video dataloader and derive effective FPS used for frame mapping.

        Args:
            video_path (str): Path to the input video file.
            data_loader (RdsDataLoader): Loader providing camera resolution metadata.
            target_fps (float | None): Desired processing FPS; if None, defaults to 30.0.

        Returns:
            tuple: `(video_dataloader, video_fps, target_fps)` where `video_fps` is detected or
            inferred FPS and `target_fps` is the effective processing FPS.
        """
        self.profiler.start("video_loader_init")
        # Prefer seg_helper-provided dataloader in tests/mocks; fallback to default
        try:
            if hasattr(self.seg_model, "get_video_dataloader") and callable(self.seg_model.get_video_dataloader):
                video_dataloader, video_dataset = self.seg_model.get_video_dataloader(video_path)
            else:
                video_dataloader, video_dataset = get_dataloader(video_path)
        except Exception:
            video_dataloader, video_dataset = get_dataloader(video_path)

        # Log video dataset information
        target_fps = 30.0 if target_fps is None else float(target_fps)
        video_fps = None
        if hasattr(video_dataset, "number_frames"):
            self.logger.info(f"Original video frames: {video_dataset.number_frames}")
            video_fps = getattr(video_dataset, "frames_per_second", None)
            if video_fps is not None:
                try:
                    video_fps = float(video_fps)
                except Exception:
                    video_fps = None
            self.logger.info(f"Video FPS: {video_fps if video_fps is not None else 'unknown'}")
            if hasattr(video_dataset, "_skip"):
                self.logger.info(f"Frame skip: {video_dataset._skip}")
        if video_fps is None or video_fps <= 0:
            video_fps = target_fps
        self.logger.info(f"Target (object/pose) FPS: {target_fps}")

        self.profiler.end()
        return video_dataloader, video_fps, target_fps

    def _maybe_init_visualization(self, video_path: str, data_loader: RdsDataLoader) -> None:
        """Initialize the visualization writer when verbose mode is enabled.

        Args:
            video_path (str): Path to the source video used for visualization metadata.
            data_loader (RdsDataLoader): Provides target width/height for visualization frames.

        This is best-effort; failures to initialize the writer don't block processing.
        """
        # Setup output directory (ensure exists for viz)
        # Note: file writing is managed in VisualizationHelper
        if self.verbose and self.vis_helper:
            self.profiler.start("visualization_init")
            if not self.vis_helper.initialize_video_writer(
                video_path,
                naming_suffix=self.naming_suffix + ".object",
                target_width=data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                target_height=data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                fps=10.0,
            ):
                self.logger.warning("Failed to initialize video writer, continuing without visualization")
            self.profiler.end()

    def _init_results_container(self, total_video_frames: int) -> Dict[str, Any]:
        """Initialize the results structure accumulated over the entire clip.

        Args:
            total_video_frames (int): Total number of frames in the input video.

        Returns:
            Dict[str, Any]: Results dict with keys such as `processed_frames`, `total_video_frames`,
            `score_matrix`, `track_ids`, `processed_frame_ids`, `hallucination_detections`, and
            `hallucination_tracks`.
        """
        hallucination_detections = {}
        for class_name in self._hallucination_classes:
            hallucination_detections[class_name] = []
        return {
            "processed_frames": 0,
            "total_video_frames": int(total_video_frames),
            "score_matrix": None,
            "track_ids": set(),
            "processed_frame_ids": [],
            "hallucination_detections": hallucination_detections,
            "hallucination_tracks": [],
        }

    # Dynamic frame processing lives in ObjectProcessorDynamic

    def _finalize_and_summarize(
        self,
        results: Dict[str, Any],
        score_matrix: np.ndarray,
        track_output_agg: Dict[int, Dict[str, Any]],
        skipped_frames: int,
        total_video_frames: int,
    ) -> None:
        """Finalize results, compute statistics, and release visualization resources.

        Args:
            results (Dict[str, Any]): Results container to be finalized/augmented in-place.
            score_matrix (np.ndarray): Frame-by-track score matrix (may include NaNs).
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregated metadata.
            skipped_frames (int): Number of frames without object data.
            total_video_frames (int): Total frames in the video timeline.
        """
        processed_frames = len(results["processed_frame_ids"])
        results["processed_frames"] = processed_frames

        self.logger.info("Frame processing complete:")
        self.logger.info(f"  - Total video frames: {total_video_frames}")
        self.logger.info(f"  - Frames with object data: {processed_frames}")
        self.logger.info(f"  - Frames skipped (no objects): {skipped_frames}")
        self.logger.info(
            f"  - Processing rate: {processed_frames}/{total_video_frames} ({processed_frames / max(total_video_frames, 1) * 100:.1f}%)"
        )

        # Log filter configuration for reference
        importance_config = self.oc_config.get("importance_filter", {})
        self.logger.info("Filter configuration:")
        self.logger.info(
            f"  - Importance filter: distance_threshold={importance_config.get('distance_threshold_m', 'N/A')}m, oncoming_obstacles={importance_config.get('oncoming_obstacles', False)}, relevant_lanes={importance_config.get('relevant_lanes', [])}"
        )

        # Hallucination results already aggregated directly into results["hallucination_detections"]

        # Release visualization helper resources
        if self.vis_helper:
            self.vis_helper.release()

        # Trim score matrix to processed frames and tracks
        if results["processed_frame_ids"] and results["track_ids"]:
            processed_frame_ids = np.array(results["processed_frame_ids"])
            results["track_ids"] = sorted(results["track_ids"])
            score_matrix = score_matrix[processed_frame_ids][:, results["track_ids"]]

        results["score_matrix"] = score_matrix

        # Compute overall statistics from score matrix
        if results["score_matrix"] is not None:
            results["mean_score"] = np.nanmean(results["score_matrix"])
            results["std_score"] = np.nanstd(results["score_matrix"])
            results["min_score"] = np.nanmin(results["score_matrix"])
            results["max_score"] = np.nanmax(results["score_matrix"])
        else:
            results["mean_score"] = 0.0
            results["std_score"] = 0.0
            results["min_score"] = 0.0
            results["max_score"] = 0.0

        # Build unified tracks list
        try:
            all_labels_set = set()
            for entry in track_output_agg.values():
                all_labels_set.update((entry.get("output_counts") or {}).keys())
            labels = []
            if "scored" in all_labels_set:
                labels.append("scored")
                all_labels_set.discard("scored")
            if "occluded" in all_labels_set:
                labels.append("occluded")
                all_labels_set.discard("occluded")
            labels.extend(sorted(all_labels_set))
            results["processor_output_labels"] = labels

            tracks_list = []
            for t_id in sorted(track_output_agg.keys()):
                entry = track_output_agg[t_id]
                types_set = entry.get("object_types") or set()
                counts = entry.get("output_counts") or {}
                object_type = None
                if types_set:
                    try:
                        object_type = sorted(list(types_set))[0]
                    except Exception:
                        object_type = next(iter(types_set))
                counts_array = [int(counts.get(label, 0)) for label in labels]
                track_entry = {
                    "track_id": int(t_id),
                    "object_type": object_type,
                    "processor_output": counts_array,
                }
                if "object_type_index" in entry:
                    track_entry["object_type_index"] = entry["object_type_index"]
                tracks_list.append(track_entry)
            results["tracks"] = tracks_list
        except Exception as e:
            self.logger.warning("Failed to build tracks list: {}".format(e))
            self.logger.warning("Setting tracks list to empty list")
            results["tracks"] = []

        # Print score matrix in compact format if verbose
        if self.verbose and results["score_matrix"] is not None:
            self._print_score_matrix(results["score_matrix"], results["track_ids"])

        # Save results to JSON file
        try:
            # Use sparse format by default for efficiency
            output_file = save_results_to_json(
                results, self.clip_id, self.output_dir, matrix_format="sparse", output_file_prefix=self.naming_suffix
            )
            self.logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save results to JSON: {e}")

    def _create_frame_visualization(
        self,
        frame_idx: int,
        segmentation_masks: Any,
        frame_scores: Dict[int, float],
        camera_pose: np.ndarray,
        camera_model: Any,
        frame_objects: Dict[str, Any],
        filtered_objects: Dict[str, Any],
        hallucinations: Optional[Dict[str, Any]] = None,
        scene_rasterizer: Optional[SceneRasterizer] = None,
    ):
        """Create visualization for a frame using the visualization helper.

        Args:
            frame_idx (int): Video frame index to annotate.
            segmentation_masks (Any): Segmentation masks or mask RGB image for the frame.
            frame_scores (Dict[int, float]): Per-track correspondence scores for the frame.
            camera_pose (np.ndarray): 4x4 pose matrix of the camera in world coordinates.
            camera_model (Any): Camera intrinsics/projection model instance.
            frame_objects (Dict[str, Any]): Raw object data for the frame.
            filtered_objects (Dict[str, Any]): Filtered/relevant objects for visualization.
            hallucinations (Dict[str, Any] | None): Optional hallucination detections for overlays.
            scene_rasterizer (SceneRasterizer | None): Optional pre-computed SceneRasterizer
                with visibility masks for efficient O(1) lookup.
        """
        # Create visualization using the helper
        vis_image = self.vis_helper.create_frame_visualization(
            frame_idx,
            segmentation_masks,
            frame_scores,
            camera_pose,
            camera_model,
            frame_objects,
            filtered_objects,
            hallucinations,
            scene_rasterizer=scene_rasterizer,
        )

        if vis_image is not None:
            # Write frame to video
            self.vis_helper.write_frame(vis_image, frame_idx)

    def _print_score_matrix(self, score_matrix: np.ndarray, track_ids: set) -> None:
        """Print score matrix in a compact, human-readable format for debugging.

        Args:
            score_matrix (np.ndarray): Matrix of per-frame, per-track scores.
            track_ids (set): Set of track IDs that were populated.
        """
        if not track_ids:
            self.logger.info("No track IDs found in score matrix")
            return

        self.logger.info("=" * 60)
        self.logger.info("SCORE MATRIX SUMMARY")
        self.logger.info("=" * 60)

        # Print matrix dimensions
        frames, max_track_id = score_matrix.shape
        self.logger.info(f"Matrix shape: {frames} frames × {max_track_id} track IDs")
        self.logger.info(f"Unique track IDs: {len(track_ids)}")

        # Calculate tracks and frames with valid scores
        tracks_with_valid_scores = np.sum(~np.all(np.isnan(score_matrix), axis=0))
        frames_with_valid_scores = np.sum(~np.all(np.isnan(score_matrix), axis=1))

        self.logger.info(f"Tracks with valid scores: {tracks_with_valid_scores}")
        self.logger.info(f"Frames with valid scores: {frames_with_valid_scores}")

        # Print statistics
        valid_scores = score_matrix[~np.isnan(score_matrix)]
        nan_count = np.isnan(score_matrix).sum()
        total_elements = score_matrix.size

        self.logger.info(f"Valid scores: {len(valid_scores)}")
        self.logger.info(f"NaN entries: {nan_count} ({nan_count / total_elements * 100:.1f}%)")

        if len(valid_scores) > 0:
            self.logger.info(f"Score range: {valid_scores.min():.3f} - {valid_scores.max():.3f}")
            self.logger.info(f"Mean score: {valid_scores.mean():.3f}")
            self.logger.info(f"Std score: {valid_scores.std():.3f}")

        # Print compact matrix representation (first 10 frames, first 10 track IDs)
        if frames > 0 and max_track_id > 0:
            self.logger.info("\nCompact Matrix (first 10 frames × first 10 track IDs):")

            # Build header row
            header = "Frame\\TrackID"
            for track_id in range(min(10, max_track_id)):
                header += f"  {track_id:3d}"
            self.logger.info(header)

            # Build matrix rows
            for frame_idx in range(min(10, frames)):
                row = f"{frame_idx:4d}"
                for track_id in range(min(10, max_track_id)):
                    score = score_matrix[frame_idx, track_id]
                    if np.isnan(score):
                        row += "  ---"
                    else:
                        row += f" {score:4.2f}"
                self.logger.info(row)

        self.logger.info("=" * 60)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration for logging/inspection.

        Returns:
            Dict[str, Any]: Selected configuration subsections and runtime settings.
        """
        summary = {
            "overlap_check": self.oc_config.get("overlap_check", {}),
            "importance_filter": self.oc_config.get("importance_filter", {}),
            "model_device": self.model_device,
        }
        try:
            hall_cfg = self.oc_config.get("hallucination_detector", {}) or {}
            if hall_cfg:
                summary["hallucination_detector"] = {
                    "enabled": bool(hall_cfg.get("enabled", False)),
                    "classes": list(hall_cfg.get("classes", [])),
                    "min_cluster_area": int(hall_cfg.get("min_cluster_area", 100)),
                    "max_cluster_per_frame": int(hall_cfg.get("max_cluster_per_frame", 100)),
                }
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to summarize hallucination_detector config: {e}")
        return summary

    # Static frame processing lives in ObjectProcessorStatic

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Load the default configuration from config.yaml.

        Returns:
            Dict[str, Any]: Dict with key 'av.obstacle' containing obstacle config.
        """
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config("config")
            return {"av.obstacle": config["av.obstacle"]}
        except Exception as e:
            logging.error(f"Error loading default configuration: {e}")
            raise e
