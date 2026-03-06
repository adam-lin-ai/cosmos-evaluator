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
Surface geometry module for projection-based masking.

This module defines a Surface that takes an Nx3 array of world-space points
that parameterize the boundary of the surface and provides utilities to project
the surface polygon into an image given a camera pose and model.
"""

from contextlib import suppress
from typing import Tuple

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rasterization import clip_and_project_polygon, rasterize_polygon_with_depth
from checks.utils.rds_geometry import RDSGeometry


class Surface(RDSGeometry):
    """
    Represents a 3D surface with projection capabilities.

    A surface is defined by an ordered set of Nx3 vertices in world coordinates
    that form the boundary of the surface polygon. The polygon is projected to
    image space and filled to create a binary mask and depth mask.
    """

    def __init__(
        self,
        vertices_world: np.ndarray,
        min_cutoff_distance: float = 3.0,
        max_cutoff_distance: float = 50.0,
    ):
        """
        Initialize a surface.

        Args:
            vertices_world: Nx3 array of surface boundary vertices in world coordinates.
            min_cutoff_distance: Minimum distance from camera (near plane clipping).
            max_cutoff_distance: Maximum distance from camera (far plane clipping).
        """
        verts = np.asarray(vertices_world, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
            raise ValueError(f"vertices_world must be of shape (N, 3) with N>=3, got {verts.shape}")
        self.vertices_world = np.array(verts)
        self.min_cutoff_distance = min_cutoff_distance
        self.max_cutoff_distance = max_cutoff_distance
        if self.min_cutoff_distance >= self.max_cutoff_distance:
            raise ValueError("min_cutoff_distance must be less than max_cutoff_distance")

    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected surface polygon in image space.
        The boundary polygon is transformed to camera coordinates, clipped to the
        near and far planes, then projected to pixels and filled.

        Rasterization is performed into bounding-box-sized local arrays for better
        cache locality and reduced initialization cost, then embedded into full-size
        output arrays.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool.
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        if camera_to_world_pose.shape != (4, 4):
            raise ValueError("camera_to_world_pose must be 4x4")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        empty_mask = np.zeros((image_height, image_width), dtype=bool)
        empty_depth = np.full((image_height, image_width), np.inf, dtype=np.float32)

        world_to_camera_pose = np.linalg.inv(camera_to_world_pose)

        # Transform boundary polygon to camera coordinates, clip behind-camera
        # geometry, and project to pixels.
        poly_cam = CameraBase.transform_points_np(self.vertices_world, world_to_camera_pose)

        with suppress(Exception):
            result = clip_and_project_polygon(
                poly_cam,
                camera_model.ray2pixel_np,
                near_z=self.min_cutoff_distance,
                far_z=self.max_cutoff_distance,
            )
            if result is None:
                return empty_mask, empty_depth
            pts, pts_z = result

            # Compute bounding box from projected vertices, clamped to image
            min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
            max_x = min(image_width - 1, int(np.ceil(np.max(pts[:, 0]))))
            min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
            max_y = min(image_height - 1, int(np.ceil(np.max(pts[:, 1]))))

            if min_x > max_x or min_y > max_y:
                return empty_mask, empty_depth

            bbox_w = max_x - min_x + 1
            bbox_h = max_y - min_y + 1

            # Allocate local bbox-sized arrays for rasterization
            local_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
            local_depth = np.full((bbox_h, bbox_w), np.inf, dtype=np.float32)

            # Offset coordinates to local bbox space and rasterize
            offset_pts = pts.copy()
            offset_pts[:, 0] -= min_x
            offset_pts[:, 1] -= min_y
            rasterize_polygon_with_depth(offset_pts, pts_z, local_mask, local_depth, bbox_w, bbox_h)

            # Embed local results into full-size output arrays
            empty_mask[min_y : max_y + 1, min_x : max_x + 1] = local_mask.astype(bool)
            empty_depth[min_y : max_y + 1, min_x : max_x + 1] = local_depth

        return empty_mask, empty_depth
