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
Scene rasterizer for computing occlusion-aware visibility masks.

This module provides a SceneRasterizer class that projects multiple RDSGeometry
objects into image space and computes visibility masks accounting for occlusions
using depth buffer compositing.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rds_geometry import RDSGeometry


class SceneRasterizer:
    """
    Rasterizes a collection of RDSGeometry objects into image space with occlusion handling.

    The SceneRasterizer takes a collection of geometry objects and projects them
    all into image space, computing per-object visibility masks that account for
    occlusions based on depth buffer compositing.

    The visibility masks are computed in the constructor, so a new SceneRasterizer
    should be created for each frame. This ensures that the camera pose and model
    are always synced to the object data.

    Usage:
        # Get objects from RdsDataLoader
        frame_objects = data_loader.get_object_data_for_frame(frame_id, include_static=True)
        vehicles = RdsDataLoader.get_objects_by_class(frame_objects, "vehicle")

        # Create rasterizer with objects and camera parameters - masks are computed immediately
        rasterizer = SceneRasterizer(
            objects=vehicles,
            camera_to_world_pose=camera_pose,
            camera_model=camera_model,
            image_width=width,
            image_height=height,
        )

        # Get visibility mask for specific object (O(1) lookup)
        mask = rasterizer.get_visibility_mask("track_123")
    """

    def __init__(
        self,
        objects: Dict[str, Dict[str, Any]],
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        logger: logging.Logger,
        depth_tolerance: float = 0.05,
        min_projected_size: int = 16,
    ):
        """
        Initialize the SceneRasterizer and compute visibility masks.

        The visibility masks are computed immediately in the constructor.
        This ensures that the camera pose and model are always synced to the object data.

        Args:
            objects: Dictionary mapping object IDs (e.g., track_id) to object info dicts.
                     Each object dict must have a "geometry" key containing an RDSGeometry
                     instance. This format matches the output of RdsDataLoader.get_objects_by_class().
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np() method.
            image_width: Output image width in pixels.
            image_height: Output image height in pixels.
            logger: Logger instance for debug output.
            depth_tolerance: Tolerance for depth comparison when determining visibility.
                            A pixel is considered visible if its depth is within this
                            tolerance of the minimum scene depth at that pixel.
                            Default is 0.05 meters.
            min_projected_size: Minimum width and height in pixels for an object's
                               post-occlusion visibility bounding box. Objects whose visible
                               region is smaller than this in either dimension get an empty
                               visibility mask (no scores computed). Default is 16 pixels.
        """
        self._logger = logger
        self._objects = objects
        self._depth_tolerance = depth_tolerance
        self._min_projected_size = min_projected_size
        self._image_width = image_width
        self._image_height = image_height

        # Cached projection results (populated by _project())
        self._visibility_masks: Dict[str, np.ndarray] = {}
        self._projected_masks: Dict[str, np.ndarray] = {}
        self._depth_masks: Dict[str, np.ndarray] = {}
        self._bboxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
        self._scene_depth_buffer: np.ndarray = np.array([])

        # Compute visibility masks immediately
        self._project(camera_to_world_pose, camera_model, image_width, image_height)

    @property
    def objects(self) -> Dict[str, Dict[str, Any]]:
        """Return the collection of objects."""
        return self._objects

    @property
    def image_width(self) -> int:
        """Return the image width."""
        return self._image_width

    @property
    def image_height(self) -> int:
        """Return the image height."""
        return self._image_height

    def has_object(self, object_id: str) -> bool:
        """Return whether the object is in the collection.

        Args:
            object_id: The ID of the object.

        Returns:
            True if the object is in the collection, False otherwise.
        """
        return object_id in self._objects

    @staticmethod
    def _compute_bbox(
        mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the bounding box of True pixels in a boolean mask.

        Args:
            mask: 2D boolean array.

        Returns:
            Tuple (min_x, min_y, max_x, max_y) or None if the mask is empty.
        """
        row_any = np.any(mask, axis=1)
        if not np.any(row_any):
            return None
        col_any = np.any(mask, axis=0)
        rows = np.where(row_any)[0]
        cols = np.where(col_any)[0]
        return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))

    def _project(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Project all objects into image space and compute occlusion-aware visibility masks.

        This method:
        1. Projects each object to get its binary mask and depth mask
        2. Composites all depth masks to create a global scene depth buffer using
           per-object bounding boxes so only relevant pixels are processed
        3. Computes visibility masks by comparing each object's depth to the scene depth,
           again restricted to each object's bounding box

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np() method.
            image_width: Output image width in pixels.
            image_height: Output image height in pixels.
        """
        # Initialize scene depth buffer with infinity
        self._scene_depth_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Step 1: Project all objects, collect masks and bounding boxes
        for obj_id, obj_info in self._objects.items():
            geometry = obj_info.get("geometry")
            if geometry is None or not isinstance(geometry, RDSGeometry):
                # Object has no valid geometry, create empty masks
                self._projected_masks[obj_id] = np.zeros((image_height, image_width), dtype=bool)
                self._depth_masks[obj_id] = np.full((image_height, image_width), np.inf, dtype=np.float32)
                self._bboxes[obj_id] = None
                continue

            # Get projected mask and depth mask from geometry
            proj_mask, depth_mask = geometry.get_projected_mask(
                camera_to_world_pose, camera_model, image_width, image_height
            )

            self._projected_masks[obj_id] = proj_mask
            self._depth_masks[obj_id] = depth_mask

            # Compute bounding box from projected mask for efficient compositing
            bbox = self._compute_bbox(proj_mask)
            self._bboxes[obj_id] = bbox

            # Update scene depth buffer (minimum depth wins) — only within the bbox
            if bbox is not None:
                min_x, min_y, max_x, max_y = bbox
                scene_roi = self._scene_depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                depth_roi = depth_mask[min_y : max_y + 1, min_x : max_x + 1]
                np.minimum(scene_roi, depth_roi, out=scene_roi)

        # Step 2: Compute visibility masks using depth comparison — only within each bbox
        for obj_id in self._objects.keys():
            bbox = self._bboxes[obj_id]

            if bbox is None:
                # No visible pixels — empty visibility mask
                self._visibility_masks[obj_id] = np.zeros((image_height, image_width), dtype=bool)
                continue

            min_x, min_y, max_x, max_y = bbox

            proj_roi = self._projected_masks[obj_id][min_y : max_y + 1, min_x : max_x + 1]
            depth_roi = self._depth_masks[obj_id][min_y : max_y + 1, min_x : max_x + 1]
            scene_roi = self._scene_depth_buffer[min_y : max_y + 1, min_x : max_x + 1]

            # A pixel is visible if:
            # 1. It's in the object's projected mask, AND
            # 2. Its depth is within tolerance of the scene minimum depth
            is_front_roi = (depth_roi - scene_roi) <= self._depth_tolerance
            vis_roi = proj_roi & is_front_roi

            visibility_mask = np.zeros((image_height, image_width), dtype=bool)
            visibility_mask[min_y : max_y + 1, min_x : max_x + 1] = vis_roi

            vis_bbox = self._compute_bbox(visibility_mask)
            if vis_bbox is not None:
                vmin_x, vmin_y, vmax_x, vmax_y = vis_bbox
                vis_w = vmax_x - vmin_x + 1
                vis_h = vmax_y - vmin_y + 1
                if vis_w < self._min_projected_size or vis_h < self._min_projected_size:
                    self._logger.debug(
                        "Projected size too small, filtering object %s: bbox dim after occlusion:%dx%d",
                        obj_id,
                        vis_w,
                        vis_h,
                    )
                    visibility_mask[:] = False

            self._visibility_masks[obj_id] = visibility_mask

    def get_visibility_mask(self, object_id: str) -> np.ndarray:
        """
        Get the occlusion-aware visibility mask for a specific object.

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the visibility mask for.

        Returns:
            Boolean numpy array of shape (image_height, image_width) where True
            indicates visible pixels of the object.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._visibility_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._visibility_masks[object_id]

    def get_projected_mask(self, object_id: str) -> np.ndarray:
        """
        Get the raw projected mask for a specific object (without occlusion handling).

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the projected mask for.

        Returns:
            Boolean numpy array of shape (image_height, image_width) where True
            indicates pixels covered by the object's projection (ignoring occlusions).

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._projected_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._projected_masks[object_id]

    def get_depth_mask(self, object_id: str) -> np.ndarray:
        """
        Get the depth mask for a specific object.

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the depth mask for.

        Returns:
            Float32 numpy array of shape (image_height, image_width) containing
            per-pixel depth values in camera space. Pixels not covered by the
            object have value inf.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._depth_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._depth_masks[object_id]

    def get_scene_depth_buffer(self) -> np.ndarray:
        """
        Get the composited scene depth buffer.

        This is the minimum depth at each pixel across all objects.

        Returns:
            Float32 numpy array of shape (image_height, image_width) containing
            the minimum depth at each pixel. Pixels not covered by any object
            have value inf.
        """
        return self._scene_depth_buffer

    def get_all_visibility_masks(self) -> Dict[str, np.ndarray]:
        """
        Get visibility masks for all objects.

        Returns:
            Dictionary mapping object IDs to their visibility masks.
        """
        return self._visibility_masks.copy()

    def get_visibility_ratio(self, object_id: str) -> float:
        """
        Get the visibility ratio for a specific object.

        The visibility ratio is the fraction of the object's projected area
        that is visible (not occluded by other objects).

        Args:
            object_id: The ID of the object.

        Returns:
            Float in range [0.0, 1.0] representing the fraction of visible pixels.
            Returns 0.0 if the object has no projected pixels.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        proj_mask = self.get_projected_mask(object_id)
        vis_mask = self.get_visibility_mask(object_id)

        total_pixels = np.sum(proj_mask)
        if total_pixels == 0:
            return 0.0

        visible_pixels = np.sum(vis_mask)
        return float(visible_pixels) / float(total_pixels)

    def get_visible_pixel_count(self, object_id: str) -> int:
        """
        Get the number of visible pixels for a specific object.

        Args:
            object_id: The ID of the object.

        Returns:
            Integer count of visible pixels.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        vis_mask = self.get_visibility_mask(object_id)
        return int(np.sum(vis_mask))
