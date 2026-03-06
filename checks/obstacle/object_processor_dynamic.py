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
Dynamic obstacle processor for SegFormer-based video processing.

This module implements the dynamic (SegFormer-based) obstacle processor,
which processes video frames and scores overlap between dynamic world model objects
and segmentation masks to check for type correctness.
"""

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from checks.obstacle.hallucination_detector import HallucinationDetector
from checks.obstacle.hallucination_tracking import track_hallucinations
from checks.obstacle.object_processor_base import ObjectProcessorBase
from checks.obstacle.overlap_detector import OverlapDetector
from checks.obstacle.seg_helper import SegHelper
from checks.utils.coord_transforms import extract_rpy_in_flu
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer


class ObjectProcessorDynamic(ObjectProcessorBase):
    """Implements dynamic (SegFormer-based) frame processing and scoring."""

    SEGFORMER_STATIC_LABEL_ORDER: ClassVar[list[str]] = [
        "road",
        "sidewalk",
        "traffic light",
        "traffic sign",
        "pole",
    ]

    SEGFORMER_MOVABLE_CLASS_ORDER: ClassVar[list[str]] = [
        "vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
    ]

    def __init__(
        self,
        clip_id: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
    ):
        """Initialize dynamic (SegFormer-based) processing components.

        Args:
            clip_id: Clip identifier for logging/visualization file names.
            output_dir: Directory where outputs (logs, videos) are written.
            config: Processor configuration dictionary.
            model_device: Device for models, e.g. 'cuda' or 'cpu'.
            verbose: Logging verbosity level.
        """
        super().__init__(clip_id, output_dir, config, model_device, verbose)
        self._init_dynamic_components()

    def _init_dynamic_components(self) -> None:
        """Initialize dynamic segmentation, overlap, and hallucination components."""
        self.naming_suffix = "dynamic"
        self.setup_logging()
        self.seg_model = SegHelper(self.model_device)
        # Overlap detector owns its internal filters
        self.overlap_detector = OverlapDetector(
            self.oc_config, self.seg_model, self.logger, debug_enabled=(self.verbose == "DEBUG")
        )

        # Initialize hallucination detector if enabled, filtered to dynamic object types only
        self.hallucination_detector = None
        halluc_cfg = self.oc_config.get("hallucination_detector", {})
        if isinstance(halluc_cfg, dict) and bool(halluc_cfg.get("enabled", False)):
            try:
                filtered_classes = {k: v for k, v in halluc_cfg.get("classes", {}).items() if k in self.DYNAMIC_OBJECTS}
                filtered_cfg = {**halluc_cfg, "classes": filtered_classes}
                self._hallucination_classes = filtered_classes
                self.hallucination_detector = HallucinationDetector(filtered_cfg, self.seg_model, self.logger)
                self.logger.info("Hallucination detector enabled")
                self._hallucination_frames = []
                self._hallucination_summary = {}
            except Exception as e:
                self.logger.warning(f"Failed to initialize HallucinationDetector: {e}")
                self.hallucination_detector = None

    def _process_frames_dynamic(
        self,
        video_dataloader,
        camera_poses,
        camera_model,
        data_loader: RdsDataLoader,
        target_fps: float,
        video_fps: float,
        trial_frames: Optional[int],
        results: Dict[str, Any],
        track_output_agg: Dict[int, Dict[str, Any]],
    ):
        """Iterate video frames and process each one.

        Args:
            video_dataloader (Any): Iterable of video frames.
            camera_poses (Any): Sequence or dict of camera poses.
            camera_model (Any): Camera intrinsics/projection model.
            data_loader (RdsDataLoader): Loader for per-frame object data and sizes.
            target_fps (float): Effective processing FPS.
            video_fps (float): Source video FPS used for frame mapping.
            trial_frames (int | None): If set, process at most this many frames.
            results (Dict[str, Any]): Results container to populate.
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregated metadata.

        Returns:
            tuple: `(score_matrix, skipped_frames)` where `score_matrix` is np.ndarray and
            `skipped_frames` is the number of frames without object data.
        """
        self.logger.info("Starting frame processing")
        if trial_frames is not None:
            self.logger.info(f"Trial mode enabled: processing up to {trial_frames} frames")

        max_track_id = max(data_loader.object_count, 1)
        score_matrix = np.full((len(video_dataloader), max_track_id + 1), np.nan)

        self.logger.info(f"Video dataloader length: {len(video_dataloader)}")
        self.logger.info(f"Camera poses available: {len(camera_poses)}")

        skipped_frames = 0
        for frame_idx, frame in enumerate(video_dataloader):
            mapped_idx = round((frame_idx * target_fps) / max(video_fps, 1e-6))
            # Guard against index overflow if pose stream is shorter than video
            if trial_frames is not None and frame_idx >= trial_frames:
                self.logger.info(f"Trial mode: stopping at frame {frame_idx} (limit: {trial_frames})")
                break
            if mapped_idx >= len(camera_poses):
                self.logger.warning(
                    f"Mapped frame {mapped_idx} (from video idx {frame_idx}) exceeds available camera poses ({len(camera_poses)})"
                )
                break

            camera_pose = camera_poses[mapped_idx]
            num_objects = len(data_loader.get_object_data_for_frame(mapped_idx, include_static=False))
            if num_objects == 0:
                self.logger.debug(f"Skipping frame {frame_idx} - no object data available")
                skipped_frames += 1
                continue

            self.profiler.start("frame_processing")
            self.logger.info(
                f"Processing frame video_idx={frame_idx} -> mapped_idx={mapped_idx} / {len(camera_poses) - 1} (objects: {num_objects})"
            )
            frame_scores = self._process_frame(
                mapped_idx,
                frame,
                data_loader,
                camera_pose,
                camera_model,
                track_output_agg,
                results,
                video_frame_idx=frame_idx,
            )
            self.profiler.end()

            if frame_scores:
                for track_id, score in frame_scores.items():
                    track_id = int(track_id)
                    results["track_ids"].add(track_id)
                    score_matrix[frame_idx, track_id] = score
                results["processed_frame_ids"].append(frame_idx)

        return score_matrix, skipped_frames

    def _process_frame(
        self,
        frame_idx: int,
        frame: Any,
        data_loader: RdsDataLoader,
        camera_pose: np.ndarray,
        camera_model: Any,
        track_output_agg: Dict[int, Dict[str, Any]],
        results: Dict[str, Any],
        video_frame_idx: Optional[int] = None,
    ) -> Optional[Dict[int, float]]:
        """Process a single frame: segmentation → scoring → hallucinations → visualization.

        Args:
            frame_idx (int): Index in the pose/object streams.
            frame (Any): Raw image/frame data for segmentation.
            data_loader (RdsDataLoader): Loader for object data and dimensions.
            camera_pose (np.ndarray): 4x4 camera pose matrix.
            camera_model (Any): Camera intrinsics/projection model.
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregation store (updated in-place).
            results (Dict[str, Any]): Results container for IDs and detections.
            video_frame_idx (int | None): Optional original video index for visualization.

        Returns:
            Optional[Dict[int, float]]: Map of `track_id -> score` for the frame, or None if empty.
        """
        if video_frame_idx is None:
            video_frame_idx = frame_idx
        self.logger.info(
            f"Frame video_idx={video_frame_idx} mapped_idx={frame_idx}: ego position in world =({camera_pose[0, 3]:.1f}, {camera_pose[1, 3]:.1f}, {camera_pose[2, 3]:.1f})"
        )

        roll, pitch, yaw = extract_rpy_in_flu(camera_pose[:3, :3])
        self.logger.info(
            f"Frame video_idx={video_frame_idx} mapped_idx={frame_idx}: camera RPY=({np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f})"
        )

        # Segmentation
        self.profiler.start("segmentation")
        segmentation_masks = self.seg_model.process_frame(frame)
        resized_masks = self.seg_model.resize_masks(
            segmentation_masks,
            data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
            data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
        )
        self.profiler.end()

        # Create SceneRasterizer for this frame - computes visibility masks in constructor
        # This computes occlusion-aware visibility masks that are shared across all components
        self.profiler.start("scene_rasterization")
        scene_rasterizer = SceneRasterizer(
            data_loader.get_object_data_for_frame(frame_idx, include_static=True),
            camera_pose,
            camera_model,
            data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
            data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
            logger=self.logger,
        )
        self.profiler.end()

        # Per-class processing
        scores = {}
        filtered_objects = {}

        self.profiler.start("processing")
        frame_objects = data_loader.get_object_data_for_frame(frame_idx, include_static=False)
        for class_name, _class_cfg_unused in self.oc_config.get("overlap_check", {}).items():
            if class_name not in self.DYNAMIC_OBJECTS:
                continue
            cls_scores, cls_filtered, _debug_info = self.overlap_detector.process_class(
                class_name,
                resized_masks,
                frame_objects,
                camera_pose,
                camera_model,
                data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                track_output_agg,
                frame_idx,
                scene_rasterizer=scene_rasterizer,
            )

            scores.update(cls_scores)
            filtered_objects.update(cls_filtered)

        self.profiler.end()

        # Debug mask video
        if self.debug_enabled and self.vis_helper:
            try:
                h = int(data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT)
                w = int(data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH)
                mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
                seg_rgb = resized_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                for seg_label in self.SEGFORMER_STATIC_LABEL_ORDER:
                    try:
                        target_color = np.array(self.seg_model.get_class_color(seg_label), dtype=np.uint8)
                        mask = np.all(seg_rgb == target_color, axis=-1)
                        rgb = self.vis_helper.mask_colors.get(seg_label)
                        if rgb is not None:
                            mask_frame[mask] = np.array(rgb, dtype=np.uint8)
                    except Exception as e:
                        self.logger.debug(f"Failed to process static label {seg_label}: {e}")
                for cls_name in self.SEGFORMER_MOVABLE_CLASS_ORDER:
                    try:
                        cls_mask = self.seg_model.get_class_mask(resized_masks, cls_name)
                        if cls_mask is not None:
                            rgb = self.vis_helper.mask_colors.get(cls_name)
                            if rgb is not None:
                                mask_frame[cls_mask.astype(bool)] = np.array(rgb, dtype=np.uint8)
                    except Exception as e:
                        self.logger.debug(f"Failed to process movable class {cls_name}: {e}")
                self.vis_helper.write_mask_frame(mask_frame, video_frame_idx, self.naming_suffix)
            except Exception as _e:
                self.logger.warning(f"Failed to write debug mask frame for frame {frame_idx}: {_e}")

        # Hallucinations
        halluc_det = None
        if getattr(self, "hallucination_detector", None) is not None:
            self.profiler.start("hallucination_detection")
            try:
                halluc_det = self.hallucination_detector.detect(
                    frame_idx=video_frame_idx,
                    resized_masks=resized_masks,
                    frame_objects=frame_objects,
                    camera_pose=camera_pose,
                    camera_model=camera_model,
                    image_width=data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                    image_height=data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                    scene_rasterizer=scene_rasterizer,
                )
                if halluc_det:
                    start_indices = {}
                    for cls_name, recs in halluc_det.items():
                        bucket = results.get("hallucination_detections", {}).get(cls_name, None)
                        if bucket is not None and isinstance(recs, list):
                            start_indices[cls_name] = len(bucket)
                            bucket.extend(recs)
                    if any(len(v or []) > 0 for v in halluc_det.values()):
                        try:
                            # Compute road mask from segmentation output
                            road_mask = None
                            try:
                                seg_rgb = resized_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                                road_color_np = SegHelper.get_class_color("road")
                                road_mask = np.all(seg_rgb == road_color_np, axis=-1)
                            except (AttributeError, ValueError):
                                road_mask = None
                            track_hallucinations(results=results, start_indices=start_indices, road_mask=road_mask)
                        except Exception as _te:
                            self.logger.warning(f"Hallucination tracking failed for frame {frame_idx}: {_te}")
                    counts = {k: len(v or []) for k, v in (halluc_det or {}).items()}
                    self.logger.info(f"Frame {frame_idx}: hallucinations {counts}")
            except Exception as _e:
                self.logger.warning(f"Hallucination detection failed for frame {frame_idx}: {_e}")
            finally:
                self.profiler.end()

        # Visualization frame
        if self.verbose and self.vis_helper:
            self.profiler.start("visualization")
            self._create_frame_visualization(
                video_frame_idx,
                resized_masks,
                scores,
                camera_pose,
                camera_model,
                frame_objects,
                filtered_objects,
                halluc_det,
                scene_rasterizer=scene_rasterizer,
            )
            self.profiler.end()

        return scores if scores else None

    def process_clip(
        self,
        input_data: str,
        clip_id: str,
        camera_name: str,
        video_path: str,
        output_dir: str,
        trial_frames: Optional[int] = None,
        target_fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Run the dynamic obstacle correspondence pipeline for a single clip.

        Args:
            input_data (str): Path to the dataset root or run directory.
            clip_id (str): Clip identifier.
            camera_name (str): Camera stream name for poses/intrinsics.
            video_path (str): Path to the camera video file.
            output_dir (str): Output directory for logs/visualizations.
            trial_frames (int | None): If provided, process at most this many frames.
            target_fps (float): Desired processing FPS (defaults to 30.0).

        Returns:
            Dict[str, Any]: Aggregated results including score matrix and statistics.
        """
        self.profiler.start("total_processing")

        # Initialization
        data_loader = self._init_data_loader(input_data, clip_id)
        camera_poses, camera_model = self._load_camera_data(data_loader, camera_name)
        video_dataloader, video_fps, target_fps = self._prepare_video(video_path, data_loader, target_fps)
        self._maybe_init_visualization(video_path, data_loader)

        # Dynamic frame processing
        results = self._init_results_container(len(video_dataloader))
        track_output_agg = {}
        score_matrix, skipped_frames = self._process_frames_dynamic(
            video_dataloader,
            camera_poses,
            camera_model,
            data_loader,
            target_fps,
            video_fps,
            trial_frames,
            results,
            track_output_agg,
        )

        # Finalization
        self._finalize_and_summarize(
            results,
            score_matrix,
            track_output_agg,
            skipped_frames,
            len(video_dataloader),
        )

        self.profiler.end()
        self.profiler.print_summary(self.logger)
        self.logger.info(f"Processing complete. Mean correspondence score: {results['mean_score']:.3f}")
        return results
