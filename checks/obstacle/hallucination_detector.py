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
Hallucination detection for obstacles using segmentation masks.

Detects per-frame hallucinated instances: clusters of class-labeled pixels
that do not align with any projected GT object of the same class.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer


class HallucinationDetector:
    """
    Detect hallucinated obstacles by subtracting refined GT support from class masks
    and clustering the remaining unmatched pixels.

    Overview
    --------
    1) Build a per-class GT support mask by intersecting projected 3D cuboids with
       connected components in the corresponding segmentation class mask.
    2) Combine per-class GT masks (OR) into a global support mask.
    3) For each class, compute unmatched regions as (class_mask AND NOT global_GT).
    4) Cluster unmatched regions into hallucination detections.

    Class-specific GT policy:
    - vehicle: For each projected cuboid, select the single largest connected component
      (by pixel area) that intersects the cuboid, and add that full component to the GT.
    - non-vehicle: For each projected cuboid, collect all intersecting components and
      fill the tightest axis-aligned rectangle that encloses them into the GT.
    """

    def __init__(self, config: Dict[str, Any], seg_helper, logger):
        """
        Args:
            config: Dict with keys {enabled, classes (map), max_cluster_per_frame}
            seg_helper: SegHelper instance for class mask extraction
            logger: Logger for diagnostics
        """
        self.config = config or {}
        self.seg_helper = seg_helper
        self.logger = logger

        # External config (strict) — all fields must be provided by caller
        if "enabled" not in self.config:
            raise KeyError("hallucination_detector.enabled is required in config")
        if "classes" not in self.config:
            raise KeyError("hallucination_detector.classes is required in config")
        if "max_cluster_per_frame" not in self.config:
            raise KeyError("hallucination_detector.max_cluster_per_frame is required in config")

        self.enabled: bool = bool(self.config["enabled"])
        # classes: mapping class_name -> {min_cluster_area: int}
        classes_val = self.config["classes"]
        if not isinstance(classes_val, dict):
            raise ValueError("hallucination_detector.classes must be a dict mapping class -> config")
        self.classes_cfg = dict(classes_val)
        self.max_clusters_per_frame: int = int(self.config["max_cluster_per_frame"])

    def detect(
        self,
        frame_idx: int,
        resized_masks: Any,
        frame_objects: Dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: Any,
        image_width: int,
        image_height: int,
        scene_rasterizer: Optional[SceneRasterizer] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run hallucination detection for a single frame.

        Steps per frame:
        - Build a global GT support mask by OR-ing per-class GT masks
        - For each configured class:
          * Extract the class mask from segmentation
          * Compute unmatched = class_mask AND NOT global_GT
          * Cluster unmatched into components with a per-class min-area threshold
          * Emit detections (bbox and a ratio metric)

        Args:
            frame_idx: Frame index
            resized_masks: Segmentation masks resized to target resolution
            frame_objects: Objects in the current frame
            camera_pose: Camera pose for this frame
            camera_model: Camera model for projection
            image_width: Image width
            image_height: Image height
            scene_rasterizer: Optional pre-computed SceneRasterizer with visibility masks.
                              If provided, uses cached masks for efficient O(1) lookup.
                              If None, computes masks directly.

        Returns:
            Dict[str, List[Dict[str, Any]]]: class_name -> list of detections with keys:
                - frame_idx: Frame index of the detection
                - bbox_xywh: Bounding box [x, y, w, h] of the unmatched cluster
                - mask_ratio: Ratio metric for the cluster (implementation-specific)
        """
        detections: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in self.classes_cfg}

        if not self.enabled:
            return detections

        # Build a global refined GT support across all classes once
        gt_mask = np.zeros((image_height, image_width), dtype=bool)
        for class_name in self.classes_cfg:
            # Build refined GT mask for this class
            class_gt_mask = self._build_gt_mask(
                resized_masks=resized_masks,
                frame_objects=frame_objects,
                camera_pose=camera_pose,
                camera_model=camera_model,
                image_width=image_width,
                image_height=image_height,
                class_name=class_name,
                scene_rasterizer=scene_rasterizer,
            )
            if class_gt_mask is not None:
                gt_mask |= class_gt_mask

        for class_name in self.classes_cfg:
            try:
                class_mask = self.seg_helper.get_class_mask(resized_masks, class_name)
            except Exception:
                # Skip unknown classes silently
                continue

            if class_mask is None:
                continue
            if class_mask.dtype != bool:
                class_mask = class_mask.astype(bool)

            # Compute unmatched regions: pixels labeled as this class but not supported by GT
            unmatched_mask = class_mask & (~gt_mask)

            if not unmatched_mask.any():
                continue

            # unmatched_mask is already boolean; avoid redundant conversion and checks
            unmatched = unmatched_mask

            # Cluster unmatched regions with a per-class minimum area threshold
            min_area = int(self._get_min_cluster_area_for_class(class_name))
            comps = self._cluster_components(unmatched, min_area=min_area)
            if not comps:
                continue

            # Cap clusters per frame per class
            comps = comps[: self.max_clusters_per_frame]

            for comp in comps:
                x, y, w, h, area, _, _ = comp
                # Ego-body filter: skip clusters intersecting bottom band
                if self._is_in_ego_body_region(x, y, w, h, image_width, image_height):
                    continue
                detections[class_name].append(
                    {
                        "frame_idx": int(frame_idx),
                        "bbox_xywh": [int(x), int(y), int(w), int(h)],
                        "mask_ratio": float(area / max(w * h, 1)),
                    }
                )

        return detections

    def _build_gt_mask(
        self,
        resized_masks: Any,
        frame_objects: Dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: Any,
        image_width: int,
        image_height: int,
        class_name: str,
        scene_rasterizer: Optional[SceneRasterizer] = None,
    ) -> np.ndarray:
        """
        Build a refined GT mask for a single class using segmentation components and 3D projections.

        Algorithm:
        - Obtain the binary class mask from the resized segmentation output.
        - Compute connected components on the class mask once (labels, stats).
        - For each ground-truth object of this class:
            * Project its cuboid to an image-space polygon mask (using SceneRasterizer if available).
            * Intersect the projection with the component labels to find intersecting labels.
            * If none intersect, the cuboid contributes nothing (no fallback).
            * If class == "vehicle": pick the single largest component by area and OR it.
            * Else (non-vehicle): compute a tight AABB enclosing all intersecting components and fill it.
        - Return the per-class union of supports.

        Args:
            resized_masks: Segmentation masks
            frame_objects: Objects in the current frame
            camera_pose: Camera pose
            camera_model: Camera model
            image_width: Image width
            image_height: Image height
            class_name: Class name to build GT mask for
            scene_rasterizer: Optional pre-computed SceneRasterizer for efficient mask lookup
        """
        if resized_masks is None:
            return np.zeros((image_height, image_width), dtype=bool)

        # Get full-image class mask for this high-level class
        try:
            class_mask = self.seg_helper.get_class_mask(resized_masks, class_name)
        except Exception:
            return np.zeros((image_height, image_width), dtype=bool)
        if class_mask is None:
            return np.zeros((image_height, image_width), dtype=bool)
        if class_mask.dtype != bool:
            class_mask = class_mask.astype(bool)

        # Connected components on the class mask (label 0 is background)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)

        # Objects of this class
        obj_map = RdsDataLoader.get_objects_by_class(frame_objects, class_name)
        if not obj_map:
            return np.zeros((image_height, image_width), dtype=bool)

        # Accumulate GT support for this class here
        class_union = np.zeros((image_height, image_width), dtype=bool)

        for track_id, tracked_object in obj_map.items():
            # Project cuboid to image and rasterize visible faces to a binary mask
            # Use SceneRasterizer if available for occlusion-aware visibility
            try:
                track_id_str = str(track_id)
                if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                    proj = scene_rasterizer.get_visibility_mask(track_id_str)
                else:
                    geometry = tracked_object["geometry"]
                    proj, _ = geometry.get_projected_mask(camera_pose, camera_model, image_width, image_height)
            except Exception:
                continue
            if proj is None or proj.sum() == 0:
                continue

            # Find labels (>0) whose pixels intersect the projected cuboid mask
            seed_labels = np.unique(labels[proj.astype(bool)])
            seed_labels = seed_labels[seed_labels > 0]
            if seed_labels.size == 0:
                # No intersecting clusters; skip (no fallback)
                continue

            if class_name == "vehicle":
                # Include all components that intersect with the projected cuboid
                comp_full = np.isin(labels, seed_labels)  # boolean mask of all selected components
                class_union |= comp_full
            else:
                # Enclose all intersecting components with a tight axis-aligned rectangle (optionally expanded)
                xs = []
                ys = []
                xes = []
                yes = []
                margin = 16
                for lab in seed_labels:
                    lab = int(lab)
                    x = int(stats[lab, cv2.CC_STAT_LEFT])
                    y = int(stats[lab, cv2.CC_STAT_TOP])
                    w = int(stats[lab, cv2.CC_STAT_WIDTH])
                    h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                    xs.append(x)
                    ys.append(y)
                    xes.append(x + max(w, 1) - 1)
                    yes.append(y + max(h, 1) - 1)
                x_min = max(0, min(xs) - margin)
                y_min = max(0, min(ys) - margin)
                x_max = min(image_width - 1, max(xes) + margin)
                y_max = min(image_height - 1, max(yes) + margin)
                if x_min <= x_max and y_min <= y_max:
                    class_union[y_min : y_max + 1, x_min : x_max + 1] = True

        return class_union

    def _cluster_components(self, mask: np.ndarray, min_area: int) -> List[List[int]]:
        """
        Cluster a binary mask into connected components and return bounding boxes.

        Args:
            mask: Binary mask to cluster
            min_area: Minimum area (in pixels) to keep a component

        Returns:
            List of components as [x, y, w, h, area, cx, cy], sorted by area desc.
        """
        comps: List[List[int]] = []
        if mask is None or not mask.any():
            return comps
        num, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for lab in range(1, num):
            x = int(stats[lab, cv2.CC_STAT_LEFT])
            y = int(stats[lab, cv2.CC_STAT_TOP])
            w = int(stats[lab, cv2.CC_STAT_WIDTH])
            h = int(stats[lab, cv2.CC_STAT_HEIGHT])
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = centroids[lab]
            comps.append([x, y, w, h, area, float(cx), float(cy)])
        # Sort largest-first to keep strongest clusters when capping
        comps.sort(key=lambda c: c[4], reverse=True)
        return comps

    def _get_min_cluster_area_for_class(self, class_name: str) -> int:
        """
        Fetch per-class minimum cluster area threshold from configuration.

        Args:
            class_name: High-level class name

        Returns:
            Pixel area threshold (int); defaults to 100 if unspecified/invalid.
        """
        cfg = self.classes_cfg.get(class_name, {}) or {}
        try:
            return int(cfg.get("min_cluster_area", 100))
        except (TypeError, ValueError):
            return 100

    def _is_in_ego_body_region(self, x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> bool:
        """
        Returns True if any part of the bbox intersects the bottom N pixels of the frame.

        Args:
            x, y, w, h: Bounding box in XYWH (image pixel coordinates)
            image_height: Image height in pixels
        """
        bottom_band = int(0.25 * image_height)
        y0 = y
        y1 = y + max(h, 1) - 1
        band_y0 = max(0, image_height - bottom_band)
        # Intersection test between [y0, y1] and [band_y0, image_height - 1]
        return (y1 >= band_y0) and (y0 <= image_height - 1)
