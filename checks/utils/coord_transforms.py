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
Coordinate transformation utilities for obstacle correspondence processing.

This module provides shared coordinate transformation functions used by
multiple components in the obstacle correspondence system.
"""

from typing import Any, Optional

import numpy as np


def get_object_to_camera_pose(tracked_object: Any, camera_pose: np.ndarray) -> np.ndarray:
    """
    Get the transformation matrix from object to camera coordinates.

    Args:
        tracked_object: Object data from world model (must contain 'object_to_world' key)
        camera_pose: Camera pose for this frame (4x4 transformation matrix)


    Returns:
        Object pose in camera coordinates (4x4 transformation matrix)

    Raises:
        KeyError: If tracked_object doesn't contain 'object_to_world'
        ValueError: If matrices have incorrect shapes
    """
    if not hasattr(tracked_object, "object_to_world_pose"):
        raise KeyError("tracked_object must contain 'object_to_world_pose' attribute")

    if camera_pose.shape != (4, 4):
        raise ValueError("camera_pose must be a 4x4 transformation matrix")

    # Get object pose in world coordinates
    object_to_world = tracked_object.object_to_world_pose

    # Transform object pose to ego coordinates
    if object_to_world.shape != (4, 4):
        raise ValueError(f"object_to_world must be 4x4 matrix, got shape {object_to_world.shape}")

    camera_to_world = camera_pose
    world_to_camera = np.linalg.inv(camera_to_world)
    object_to_camera = world_to_camera @ object_to_world

    return object_to_camera


def extract_rpy_in_flu(R_flu_to_rdf: np.ndarray) -> tuple[float, float, float]:
    """
    Extract roll, pitch, yaw defined in forward-left-up coordinates from rotation matrix.

    Args:
        R_flu_to_rdf: Rotation matrix from forward-left-up to right-down-forward coordinates

    Returns:
        Tuple of (roll, pitch, yaw) in radians

    Raises:
        ValueError: If input matrix is not a 3x3 rotation matrix
    """
    if R_flu_to_rdf.shape != (3, 3):
        raise ValueError(f"R_flu_to_rdf must be 3x3 matrix, got shape {R_flu_to_rdf.shape}")

    # Convert RDF→FLU
    R = R_flu_to_rdf.T  # inverse of rotation matrix

    # Extract angles from R using ZYX convention (yaw-pitch-roll)
    if abs(R[2, 0]) != 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock case
        yaw = 0
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])

    return roll, pitch, yaw


def clip_polygon_to_z_planes(
    poly_cam: np.ndarray, near_plane_z: Optional[float] = 1.0e-6, far_plane_z: Optional[float] = None
) -> np.ndarray:
    """
    Clip a 3D polygon in camera coordinates against z-planes.

    The polygon is clipped against the half-space z >= near_plane_z and/or
    the half-space z <= far_plane_z. If both planes are provided, clipping
    is applied in sequence (near then far). If neither plane is provided,
    the original polygon is returned unchanged.

    Args:
        poly_cam: Nx3 array of polygon vertices in camera coordinates.
        near_plane_z: Optional near plane z value; keep z >= near_plane_z.
        far_plane_z: Optional far plane z value; keep z <= far_plane_z.

    Returns:
        Mx3 array of clipped polygon vertices (M can be 0).
    """
    pts = np.asarray(poly_cam, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"poly_cam must have shape (N, 3); got {pts.shape}")
    if pts.shape[0] == 0:
        return pts

    if near_plane_z is None and far_plane_z is None:
        return pts.copy()

    def _clip_against_plane(vertices: np.ndarray, z_plane: float, keep_greater_equal: bool) -> np.ndarray:
        if vertices.shape[0] == 0:
            return vertices

        current = vertices
        prev = np.roll(vertices, 1, axis=0)

        zc = current[:, 2]
        zp = prev[:, 2]
        if keep_greater_equal:
            inside_cur = zc >= z_plane
            inside_prev = zp >= z_plane
        else:
            inside_cur = zc <= z_plane
            inside_prev = zp <= z_plane

        crosses = inside_cur ^ inside_prev

        # Prepare arrays for intersections (placed before the "current" vertex in ordering)
        intersections = np.full_like(current, np.nan, dtype=float)
        if np.any(crosses):
            dz = (zc - zp)[crosses]
            # Avoid division by zero: for true crossings dz must be non-zero
            dz = np.where(np.abs(dz) < 1e-10, 1e-10, dz)
            t = (z_plane - zp[crosses]) / dz
            pc = current[crosses]
            pp = prev[crosses]
            inter = pp + t[:, None] * (pc - pp)
            inter[:, 2] = z_plane  # snap z to the clip plane (avoid FP drift)
            intersections[crosses] = inter

        # Points to include (current vertices that are inside)
        kept_current = np.full_like(current, np.nan, dtype=float)
        kept_current[inside_cur] = current[inside_cur]

        # Interleave [intersection_before_i, current_i_if_kept] preserving order without loops
        interleaved = np.stack([intersections, kept_current], axis=1).reshape(-1, 3)
        out = interleaved[~np.any(np.isnan(interleaved), axis=1)]

        # Remove consecutive duplicates that can arise when a vertex lies exactly on the plane
        if out.shape[0] >= 2:
            dup = np.all(np.isclose(out[1:], out[:-1]), axis=1)
            mask = np.ones(out.shape[0], dtype=bool)
            mask[1:] = ~dup
            out = out[mask]
        return out

    out = pts
    if near_plane_z is not None:
        out = _clip_against_plane(out, float(near_plane_z), keep_greater_equal=True)
    if far_plane_z is not None and out.shape[0] > 0:
        out = _clip_against_plane(out, float(far_plane_z), keep_greater_equal=False)
    return out
