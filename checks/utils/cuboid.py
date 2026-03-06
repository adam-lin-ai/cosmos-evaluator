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
Cuboid geometry module for obstacle correspondence processing.

This module handles 3D cuboid geometry and projection operations.
"""

from typing import Tuple

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rasterization import clip_and_project_polygon, rasterize_polygon_with_depth
from checks.utils.rds_geometry import RDSGeometry


class Cuboid(RDSGeometry):
    """
    Represents a 3D cuboid with projection capabilities.

    A cuboid is defined by its center pose (4x4 transformation matrix)
    and its dimensions (length, width, height).
    """

    def __init__(self, object_to_world_pose: np.ndarray, lwh: np.ndarray):
        """
        Initialize a cuboid.

        Args:
            object_to_world_pose: 4x4 transformation matrix from object to world coordinates
            lwh: Length, width, height dimensions as a 3-element array
        """
        self.object_to_world_pose = np.array(object_to_world_pose)
        self.lwh = np.array(lwh)
        self.corners = self._compute_corners()

    def _compute_corners(self) -> np.ndarray:
        """
        Compute the 8 corners of the cuboid in world coordinates.

        Returns:
            Array of shape (8, 3) containing the corner coordinates
        """
        length, width, height = self.lwh.flatten()

        # Define corners in local coordinates (centered at origin)
        # Each corner is (x, y, z) with x in {-l/2, l/2}, y in {-w/2, w/2}, z in {-h/2, h/2}
        offsets = np.array(
            [
                [-length / 2, -width / 2, -height / 2],  # back-bottom-left
                [-length / 2, -width / 2, height / 2],  # back-bottom-right
                [-length / 2, width / 2, -height / 2],  # back-top-left
                [-length / 2, width / 2, height / 2],  # back-top-right
                [length / 2, -width / 2, -height / 2],  # front-bottom-left
                [length / 2, -width / 2, height / 2],  # front-bottom-right
                [length / 2, width / 2, -height / 2],  # front-top-left
                [length / 2, width / 2, height / 2],  # front-top-right
            ]
        )

        # Convert to homogeneous coordinates
        offsets_hom = np.hstack([offsets, np.ones((8, 1))])  # shape (8, 4)

        # Transform to world coordinates
        corners_world = (self.object_to_world_pose @ offsets_hom.T).T[:, :3]  # shape (8, 3)
        return corners_world

    @staticmethod
    def compute_projected_mask(
        corners: np.ndarray,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected cuboid in image space.

        Rasterization is performed into bounding-box-sized local arrays for better
        cache locality and reduced initialization cost, then embedded into full-size
        output arrays.

        Args:
            corners: 8x3 array of corner coordinates in world coordinates
            camera_to_world_pose: 4x4 camera-to-world pose matrix
            camera_model: Camera model with .ray2pixel_np() method
            image_width: Width of the image
            image_height: Height of the image

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        empty_mask = np.zeros((image_height, image_width), dtype=bool)
        empty_depth = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Project all 8 corners to camera frame
        world_to_camera_pose = np.linalg.inv(camera_to_world_pose)
        camera_frame_cuboid_corners = CameraBase.transform_points_np(corners, world_to_camera_pose)

        # Near-plane threshold for clipping behind-camera geometry.
        eps = 1e-6

        # Early out: if ALL corners are behind the camera, skip entirely.
        if np.all(camera_frame_cuboid_corners[:, 2] < eps):
            return empty_mask, empty_depth

        # Define cuboid faces as lists of corner indices
        faces = [
            [0, 1, 3, 2],  # back face
            [4, 5, 7, 6],  # front face
            [0, 1, 5, 4],  # right face
            [2, 3, 7, 6],  # left face
            [1, 3, 7, 5],  # top face
            [0, 2, 6, 4],  # bottom face
        ]

        # First pass: clip and project each face, collecting results and 2D vertices
        # for bounding box computation.  clip_and_project_polygon handles
        # early-out for fully-behind faces and skips clipping when unnecessary.
        face_data = []
        all_2d_points = []
        for face in faces:
            result = clip_and_project_polygon(camera_frame_cuboid_corners[face], camera_model.ray2pixel_np, near_z=eps)
            if result is None:
                continue
            pts, pts_z = result
            face_data.append((pts, pts_z))
            all_2d_points.append(pts)

        if not face_data:
            return empty_mask, empty_depth

        # Compute overall bounding box from all projected vertices, clamped to image
        all_pts = np.vstack(all_2d_points)
        min_x = max(0, int(np.floor(np.min(all_pts[:, 0]))))
        max_x = min(image_width - 1, int(np.ceil(np.max(all_pts[:, 0]))))
        min_y = max(0, int(np.floor(np.min(all_pts[:, 1]))))
        max_y = min(image_height - 1, int(np.ceil(np.max(all_pts[:, 1]))))

        if min_x > max_x or min_y > max_y:
            return empty_mask, empty_depth

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1

        # Allocate local bbox-sized arrays for rasterization
        local_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        local_depth = np.full((bbox_h, bbox_w), np.inf, dtype=np.float32)

        # Second pass: rasterize each face into local arrays with offset coordinates
        for pts, pts_z in face_data:
            offset_pts = pts.copy()
            offset_pts[:, 0] -= min_x
            offset_pts[:, 1] -= min_y
            rasterize_polygon_with_depth(offset_pts, pts_z, local_mask, local_depth, bbox_w, bbox_h)

        # Embed local results into full-size output arrays
        mask = empty_mask
        depth_mask = empty_depth
        mask[min_y : max_y + 1, min_x : max_x + 1] = local_mask.astype(bool)
        depth_mask[min_y : max_y + 1, min_x : max_x + 1] = local_depth

        return mask, depth_mask

    def get_projected_mask(
        self, camera_to_world_pose: np.ndarray, camera_model: CameraBase, image_width: int, image_height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected cuboid in image space.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix
            camera_model: Camera model with .ray2pixel_np() method
            image_width: Width of the image
            image_height: Height of the image

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        return self.compute_projected_mask(self.corners, camera_to_world_pose, camera_model, image_width, image_height)

    def get_center_point(self) -> np.ndarray:
        """
        Get the center point of the cuboid in world coordinates.

        Returns:
            3D center point as numpy array
        """
        return self.object_to_world_pose[:3, 3]

    def get_dimensions(self) -> np.ndarray:
        """
        Get the dimensions of the cuboid.

        Returns:
            Length, width, height as numpy array
        """
        return self.lwh.copy()

    @staticmethod
    def compute_pose_and_lwh_from_corners(corners_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute an object-to-world pose matrix and LWH dimensions from 8 cuboid corners.

        The method estimates the cuboid center and orientation using PCA of corner points,
        then measures the extents along the principal axes to obtain length/width/height.

        Args:
            corners_world: Array of shape (8, 3) containing cuboid corner coordinates in world frame.

        Returns:
            A tuple (object_to_world_pose, lwh) where:
            - object_to_world_pose: 4x4 homogeneous transform with rotation (columns = principal axes)
              and translation set to the cuboid center in world coordinates.
            - lwh: 3-element array [length, width, height] corresponding to extents along the
              first, second, and third principal axes respectively.

        Raises:
            ValueError: If corners are not of shape (8, 3).
        """
        if corners_world is None:
            raise ValueError("corners_world must not be None")
        corners_world = np.asarray(corners_world, dtype=float)
        if corners_world.ndim != 2 or corners_world.shape != (8, 3):
            raise ValueError(f"corners_world must have shape (8, 3), got {corners_world.shape}")

        # Center of the cuboid (mean of corners)
        center = np.mean(corners_world, axis=0)

        # PCA to estimate orientation
        demeaned = corners_world - center[None, :]
        # Use SVD which is numerically stable: demeaned = U S V^T, rows are samples
        # Principal axes are columns of V (right-singular vectors)
        _, _, Vt = np.linalg.svd(demeaned, full_matrices=False)
        R = Vt.T  # shape (3,3), columns are principal axes

        # Ensure right-handed rotation (determinant +1)
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        # Project corners onto principal axes and measure extents
        projections = demeaned @ R  # shape (8,3)
        mins = projections.min(axis=0)
        maxs = projections.max(axis=0)
        lwh = maxs - mins

        # Build homogeneous transform
        object_to_world = np.eye(4, dtype=float)
        object_to_world[:3, :3] = R
        object_to_world[:3, 3] = center

        return object_to_world, lwh
