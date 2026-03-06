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
Polyline geometry module for projection-based masking.

This module defines a Polyline that takes an Nx3 array of world-space points
and provides utilities to project a dilated ribbon of the polyline into an
image given a camera pose and model.
"""

from contextlib import suppress
from typing import Tuple

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rasterization import clip_and_project_polygon, rasterize_polygon_with_depth
from checks.utils.rds_geometry import RDSGeometry


class Polyline(RDSGeometry):
    """
    Represents a 3D polyline with projection capabilities.

    A polyline is defined by an ordered set of Nx3 vertices in world coordinates.
    For image masking, each adjacent pair of points [Pi, Pi+1] forms a segment.
    The segment is dilated in the local ground plane (XY-plane, preserving each
    endpoint's z) by a configurable half-width to form a 3D quad which is then
    clipped to the camera near plane and projected to pixels.
    """

    def __init__(
        self,
        vertices_world: np.ndarray,
        half_width_meters: float = 0.20,
        min_cutoff_distance: float = 3.0,
        max_cutoff_distance: float = 50.0,
    ):
        """
        Initialize a polyline.

        Args:
            vertices_world: Nx3 array of polyline vertices in world coordinates.
            half_width_meters: Half of the ribbon thickness (meters) used to dilate
                               each segment normal in the ground plane.
        """
        verts = np.asarray(vertices_world, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 2:
            raise ValueError(f"vertices_world must be of shape (N, 3) with N>=2, got {verts.shape}")
        self.vertices_world = np.array(verts)
        self.half_width_m = float(max(0.0, half_width_meters))
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
        Compute a binary mask and depth mask of the projected, dilated polyline in image space.
        For each segment, a 3D quad is built by offsetting the endpoints along
        the segment's ground-plane normal by ±half_width_m, then clipped to near
        plane and projected to pixels. The per-segment masks are unioned.

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

        # First pass: build, clip, and project each segment quad to collect
        # 2D vertices for overall bounding box computation.
        verts = self.vertices_world
        segment_data = []
        all_2d_points = []

        for i in range(len(verts) - 1):
            p0 = verts[i]
            p1 = verts[i + 1]
            # Ground-plane direction in XY
            d = p1[:2] - p0[:2]
            norm = float(np.hypot(d[0], d[1]))
            if norm < 1e-6:
                continue
            t = d / norm
            # Left-hand normal in XY (rotate t by +90 deg)
            n = np.array([-t[1], t[0]], dtype=float)
            hw = self.half_width_m

            # Build 4 corners in world coordinates; preserve each endpoint's z
            left0 = np.array([p0[0] + n[0] * hw, p0[1] + n[1] * hw, p0[2]], dtype=float)
            left1 = np.array([p1[0] + n[0] * hw, p1[1] + n[1] * hw, p1[2]], dtype=float)
            right1 = np.array([p1[0] - n[0] * hw, p1[1] - n[1] * hw, p1[2]], dtype=float)
            right0 = np.array([p0[0] - n[0] * hw, p0[1] - n[1] * hw, p0[2]], dtype=float)
            quad_world = np.stack([left0, left1, right1, right0], axis=0)  # (4,3)

            # Transform to camera coordinates, clip behind-camera geometry,
            # and project to pixels in one step.
            quad_cam = CameraBase.transform_points_np(quad_world, world_to_camera_pose)
            with suppress(Exception):
                result = clip_and_project_polygon(
                    quad_cam,
                    camera_model.ray2pixel_np,
                    near_z=self.min_cutoff_distance,
                    far_z=self.max_cutoff_distance,
                )
                if result is None:
                    continue
                pts, pts_z = result
                segment_data.append((pts, pts_z))
                all_2d_points.append(pts)

        if not segment_data:
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

        # Second pass: rasterize each segment into local arrays with offset coordinates
        for pts, pts_z in segment_data:
            offset_pts = pts.copy()
            offset_pts[:, 0] -= min_x
            offset_pts[:, 1] -= min_y
            rasterize_polygon_with_depth(offset_pts, pts_z, local_mask, local_depth, bbox_w, bbox_h)

        # Embed local results into full-size output arrays
        empty_mask[min_y : max_y + 1, min_x : max_x + 1] = local_mask.astype(bool)
        empty_depth[min_y : max_y + 1, min_x : max_x + 1] = local_depth

        return empty_mask, empty_depth
