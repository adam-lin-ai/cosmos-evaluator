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
Overlap detection for obstacle correspondence.

This module owns the per-class processing pipeline that determines how well
tracked 3D objects correspond to segmentation results in each frame.

Responsibilities
----------------
1) Importance filtering: decide if an object should be evaluated (distance, lanes, etc.).
2) Occlusion detection: optionally skip objects considered occluded (currently not implemented)
3) Scoring: compute a per-object correspondence score via one of two methods:
   - "ratio"  : overlap_pixels / projected_cuboid_pixels (identical to legacy implementation).
   - "cluster": find all segmentation connected components intersecting the projected cuboid
                and compute inside / total_cluster_area. This reduces over-scoring caused by
                large components that spill far outside the cuboid.

Notes
-----
- The class-to-SegFormer label mapping and object-type mapping are centralized in SegHelper.
- For the cluster method, connected components are computed once per class per frame and
  reused for all objects to avoid redundant work.
"""

from contextlib import suppress
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from checks.obstacle.importance_filter import ImportanceFilter
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer


class OverlapDetector:
    def __init__(self, oc_config: Dict[str, Any], seg_model, logger, debug_enabled: bool = False):
        """Initialize the overlap detector and supporting components.

        Args:
            oc_config: Obstacle configuration subtree (includes overlap_check, filters, etc.)
            seg_model: SegHelper or CWIPInference instance for mask extraction
            logger: Logger for diagnostics
            debug_enabled: If True, collect extra per-frame debug metadata
        """
        self.seg_model = seg_model
        self.logger = logger
        self.debug_enabled = debug_enabled
        # Store obstacle config
        self.oc_config = oc_config or {}
        # Build filters from obstacle config
        self.importance_filter = ImportanceFilter(self.oc_config.get("importance_filter", {}), logger)

    def process_class(
        self,
        class_name: str,
        resized_masks: Any,
        frame_objects: Dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: Any,
        image_width: int,
        image_height: int,
        track_output_agg: Dict[int, Dict[str, Any]],
        frame_idx: int,
        scene_rasterizer: Optional[SceneRasterizer] = None,
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process a single class for one frame: filter, score, and (optionally) debug.

        Workflow:
            1) Read scoring method for class from config (ratio|cluster)
            2) Extract class mask from segmentation
            3) If cluster method, compute connected components once on class mask
            4) Iterate class objects → importance filter → occlusion filter → score
            5) Aggregate scores, filtered reasons, and optional debug bboxes

        Args:
            class_name: High-level class (vehicle, pedestrian, ...)
            resized_masks: Segmentation masks (resized to target)
            frame_objects: Mapping of track_id → object dict for current frame
            camera_pose: 4x4 camera-to-world pose
            camera_model: Camera model used for projection
            image_width: Target image width
            image_height: Target image height
            track_output_agg: Accumulator for per-track output labels/stats
            frame_idx: Current frame index (object timeline)
            scene_rasterizer: Optional pre-computed SceneRasterizer with visibility masks.
                              If provided, uses cached masks for efficient O(1) lookup.
                              If None, computes masks directly.

        Returns:
            scores: track_id → score
            filtered_objects: track_id → {reason, type}
            debug_info: Optional visualization info if debug_enabled
        """
        scores: Dict[int, float] = {}
        filtered_objects: Dict[int, Dict[str, Any]] = {}
        debug_info = {"frame_boxes": [], "object_categories": {}} if self.debug_enabled else None
        importance_filtered_count = 0
        occlusion_filtered_count = 0
        importance_filter_reasons: Dict[str, int] = {}

        # Determine method from config
        method = "ratio"
        try:
            class_cfg = None
            try:
                overlap_cfg = self.oc_config.get("overlap_check", {}) or {}
                class_cfg = overlap_cfg.get(class_name, {})
            except Exception:
                class_cfg = {}
            if isinstance(class_cfg, dict):
                method = str(class_cfg.get("method", method)).lower()
        except Exception:
            method = "ratio"

        # Extract binary segmentation mask for the high-level class from SegHelper.
        class_mask = self.seg_model.get_class_mask(resized_masks, class_name)

        # If scoring method is cluster, precompute connected components once on the full class mask
        # to obtain per-pixel labels and per-component stats (bbox, area, centroid). This is reused
        # for every object of this class in the current frame.
        labels = None
        stats = None
        if method == "cluster" and class_mask is not None:
            try:
                _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)
            except Exception:
                labels, stats = None, None

        # Objects for this class
        class_objects = RdsDataLoader.get_objects_by_class(frame_objects, class_name)
        total_objects = len(class_objects)
        if total_objects == 0:
            return scores, filtered_objects, debug_info

        for track_id, tracked_object in class_objects.items():
            scores[track_id] = np.nan

            # Importance filter
            importance_result, filter_reason = self.importance_filter.should_process_object(
                tracked_object, camera_pose, track_id
            )
            if not importance_result:
                filtered_objects[track_id] = {"reason": filter_reason, "type": "importance"}
                importance_filtered_count += 1
                with suppress(Exception):
                    importance_filter_reasons[str(filter_reason)] = (
                        int(importance_filter_reasons.get(str(filter_reason), 0)) + 1
                    )
                self._aggregate_output(track_output_agg, track_id, tracked_object, str(filter_reason))
                continue

            # Score
            # Score each object using the selected method. For cluster, pass the precomputed CC.
            score = self._score_object(
                track_id,
                tracked_object,
                class_mask,
                camera_pose,
                camera_model,
                image_width,
                image_height,
                method,
                labels,
                stats,
                scene_rasterizer,
            )
            if score is not None:
                scores[track_id] = score
                self._aggregate_output(track_output_agg, track_id, tracked_object, "scored")

                # Debug bbox
                if self.debug_enabled and debug_info is not None:
                    try:
                        # Get projected mask from SceneRasterizer if available
                        track_id_str = str(track_id)
                        if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                            projected_mask = scene_rasterizer.get_visibility_mask(track_id_str)
                        else:
                            geometry = tracked_object["geometry"]
                            projected_mask, _ = geometry.get_projected_mask(
                                camera_pose, camera_model, image_width, image_height
                            )
                        ys, xs = np.where(projected_mask)
                        if xs.size > 0 and ys.size > 0:
                            x_min = int(xs.min())
                            y_min = int(ys.min())
                            x_max = int(xs.max()) + 1
                            y_max = int(ys.max()) + 1
                            x0 = max(0, min(x_min, image_width))
                            y0 = max(0, min(y_min, image_height))
                            x1 = max(0, min(x_max, image_width))
                            y1 = max(0, min(y_max, image_height))
                            w = x1 - x0
                            h = y1 - y0
                            if w > 0 and h > 0:
                                debug_info["object_categories"].setdefault(int(track_id), str(class_name))
                                debug_info["frame_boxes"].append(
                                    {"object_id": int(track_id), "bbox_xywh": [int(x0), int(y0), int(w), int(h)]}
                                )
                    except Exception:
                        pass

        # Logging summary for this class
        try:
            processed_count = sum(1 for s in scores.values() if not np.isnan(s))
            self.logger.info(f"Frame {frame_idx} class={class_name} filtering summary:")
            self.logger.info(f"  - Total objects: {total_objects}")
            self.logger.info(f"  - Importance filtered: {importance_filtered_count}")
            if importance_filter_reasons:
                for reason, count in importance_filter_reasons.items():
                    self.logger.info(f"    * {reason}: {count}")
            self.logger.info(f"  - Occlusion filtered: {occlusion_filtered_count}")
            self.logger.info(f"  - Processed: {processed_count}")
            if total_objects > 0:
                self.logger.info(
                    f"  - Filtering rates: {importance_filtered_count / total_objects * 100:.1f}% importance, {occlusion_filtered_count / total_objects * 100:.1f}% occlusion"
                )
        except Exception:
            pass

        return scores, filtered_objects, debug_info

    def _score_object(
        self,
        track_id: Any,
        tracked_object: Dict[str, Any],
        class_mask: np.ndarray,
        camera_pose: np.ndarray,
        camera_model: Any,
        image_width: int,
        image_height: int,
        scoring_method: str,
        labels: Optional[np.ndarray] = None,
        stats: Optional[np.ndarray] = None,
        scene_rasterizer: Optional[SceneRasterizer] = None,
    ) -> Optional[float]:
        """Score one object using the configured method.

        Args:
            track_id: Track ID of the object
            tracked_object: Object dict with at least object_to_world and object_lwh
            class_mask: Binary mask for the object's class
            camera_pose: 4x4 camera-to-world pose
            camera_model: Camera model used for projection
            image_width: Image width for projection
            image_height: Image height for projection
            scoring_method: "ratio" or "cluster"
            labels: Connected component labels for class_mask (if cluster)
            stats: Connected component stats for class_mask (if cluster)
            scene_rasterizer: Optional pre-computed SceneRasterizer for efficient mask lookup

        Returns:
            Score in [0,1] or None if not scorable (e.g., empty projection)
        """
        try:
            # Get projected mask from SceneRasterizer if available, otherwise compute directly
            track_id_str = str(track_id)
            if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                projected_mask = scene_rasterizer.get_visibility_mask(track_id_str)
            else:
                geometry = tracked_object["geometry"]
                projected_mask, _ = geometry.get_projected_mask(camera_pose, camera_model, image_width, image_height)
            if projected_mask.sum() == 0:
                return None
            method = scoring_method.lower() if isinstance(scoring_method, str) else "ratio"
            if method == "ratio":
                # Legacy ratio metric: overlap / cuboid_area.
                return float(self._score_ratio(class_mask, projected_mask))
            elif method == "cluster":
                # New cluster metric: inside / total_area_of_intersecting_components.
                return float(self._score_cluster(class_mask, projected_mask, labels, stats))
            else:
                self.logger.warning(f"Invalid scoring method: {scoring_method}")
                return None
        except Exception as e:
            self.logger.warning(f"Error processing object: {e}")
            return None

    @staticmethod
    def _score_ratio(class_mask: np.ndarray, projected_cuboid_mask: np.ndarray) -> float:
        """
        Ratio-based score: overlap pixels divided by projected cuboid area.

        This implementation is kept identical to the previous behavior to preserve
        historical results.
        """
        if projected_cuboid_mask is None or projected_cuboid_mask.sum() == 0:
            return 0.0
        masked = (class_mask.astype(bool) if class_mask is not None else False) & projected_cuboid_mask.astype(bool)
        num_overlap = int(np.count_nonzero(masked))
        denom = int(np.count_nonzero(projected_cuboid_mask))
        if denom <= 0:
            return 0.0
        return float(num_overlap / denom)

    @staticmethod
    def _score_cluster(
        class_mask: np.ndarray,
        projected_cuboid_mask: np.ndarray,
        labels: Optional[np.ndarray],
        stats: Optional[np.ndarray],
    ) -> float:
        """
        Cluster-based score: For components intersecting the cuboid, score = inside / total_cluster_area.

        Rationale
        ---------
        The previous cluster metric was biased by large components that extend outside the cuboid.
        By normalizing the overlap pixels by the true area of the intersecting components (from CC),
        the score better reflects the fraction of relevant segment(s) actually covered by the cuboid.

        Algorithm
        ---------
        - CC is computed once per class mask upstream; this function reuses labels and stats.
        - Identify all labels (>0) that intersect the projected cuboid (via labels[proj]).
        - total_cluster_area = sum(stats[label, AREA]) across these labels (not double-counting overlaps).
        - inside = count of pixels where (labels in those labels) AND projected cuboid.
        - Score = inside / total_cluster_area (0.0 if no intersecting labels or zero area).
        """
        if (
            class_mask is None
            or projected_cuboid_mask is None
            or not projected_cuboid_mask.any()
            or labels is None
            or stats is None
        ):
            return 0.0

        proj_bool = projected_cuboid_mask.astype(bool)
        # Intersecting labels (>0): labels that appear within the projected cuboid region
        try:
            seed_labels = np.unique(labels[proj_bool])
            seed_labels = seed_labels[seed_labels > 0]
        except Exception:
            return 0.0
        if seed_labels.size == 0:
            return 0.0

        # Build mask for union of intersecting components and count pixels inside the cuboid
        union_mask = np.isin(labels, seed_labels)
        inside = int(np.count_nonzero(union_mask & proj_bool))
        # Sum areas from stats to avoid recounting due to overlaps between components
        try:
            total_area = int(sum(int(stats[int(lab), cv2.CC_STAT_AREA]) for lab in seed_labels))
        except Exception:
            # Fallback to pixel count of union if stats are malformed
            total_area = int(np.count_nonzero(union_mask))

        if total_area <= 0:
            return 0.0
        return float(inside / total_area)

    def _aggregate_output(
        self, track_output_agg: Dict[int, Dict[str, Any]], track_id: int, tracked_object: Dict[str, Any], label: str
    ) -> None:
        """Accumulate per-track metadata for final reporting.

        Increments the count for an output label (e.g., "scored", "occluded", filter reason)
        and tracks encountered object types per track id.
        """
        try:
            t_id = int(track_id)
            entry = track_output_agg.get(t_id)
            if entry is None:
                entry = {"object_types": set(), "output_counts": {}}
                track_output_agg[t_id] = entry
            obj_type = tracked_object.get("object_type")
            if obj_type is not None:
                entry["object_types"].add(str(obj_type))
            obj_type_index = tracked_object.get("object_type_index")
            if obj_type_index is not None:
                entry["object_type_index"] = int(obj_type_index)
            entry["output_counts"][str(label)] = int(entry["output_counts"].get(str(label), 0)) + 1
        except Exception as e:
            self.logger.warning("Failed to aggregate output for track {}: {}".format(track_id, e))
