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
Rasterization utilities for polygon depth interpolation.

This module provides functions to rasterize polygons with per-pixel depth
interpolation using barycentric coordinates. The inner triangle rasterizer
is JIT-compiled with Numba for performance.

The :func:`clip_and_project_polygon` helper consolidates near/far-plane
clipping and projection into a single call with early-out optimisations
for geometry that is entirely behind the camera.
"""

from contextlib import suppress
import math
from typing import Callable, Optional, Tuple

import cv2
import numba
import numpy as np

from checks.utils.coord_transforms import clip_polygon_to_z_planes


def clip_and_project_polygon(
    pts_cam: np.ndarray,
    project_fn: Callable[[np.ndarray], np.ndarray],
    near_z: float = 1e-6,
    far_z: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip a 3D polygon in camera space against z-planes and project to 2D.

    This function consolidates near/far-plane clipping and 2D projection into
    a single call with fast-path optimisations that avoid unnecessary work:

    1. **Early-out** – if *every* vertex is behind the near plane the polygon
       is invisible; return ``None`` without touching the clipper or projector.
    2. **Skip clipping** – if *every* vertex already satisfies the near (and
       optional far) plane constraint, the expensive Sutherland-Hodgman clip
       is bypassed entirely and the vertices are projected directly.
    3. **Clip then project** – only when the polygon actually straddles a
       clipping plane is the full polygon clip performed before projection.

    Args:
        pts_cam: Nx3 array of polygon vertices in camera coordinates.
        project_fn: Callable that maps an Mx3 array of camera-space points
            to an Mx2 array of pixel coordinates (e.g.
            ``camera_model.ray2pixel_np``).
        near_z: Near-plane z value; vertices with z < near_z are behind the
            camera.  Defaults to ``1e-6``.
        far_z: Optional far-plane z value.  When provided, vertices with
            z > far_z are clipped.

    Returns:
        ``(pts_2d, pts_z)`` where *pts_2d* is an Mx2 array of projected
        pixel coordinates and *pts_z* is a length-M array of z-values for
        the (possibly clipped) polygon.  Returns ``None`` when the polygon
        is entirely clipped away or degenerates to fewer than 3 vertices.
    """
    pts_cam = np.asarray(pts_cam, dtype=float)
    if pts_cam.ndim != 2 or pts_cam.shape[1] != 3:
        raise ValueError(f"pts_cam must have shape (N, 3), got {pts_cam.shape}")
    if pts_cam.shape[0] < 3:
        return None

    z_vals = pts_cam[:, 2]

    # --- Optimisation 1: entire polygon behind the near plane ---------------
    if np.all(z_vals < near_z):
        return None

    # --- Optimisation 2: entire polygon within valid range ------------------
    all_in_front = bool(np.all(z_vals >= near_z))
    all_within_far = far_z is None or bool(np.all(z_vals <= far_z))

    if all_in_front and all_within_far:
        clipped = pts_cam
    else:
        # --- Full Sutherland-Hodgman clip against near (and far) plane ------
        clipped = clip_polygon_to_z_planes(pts_cam, near_z, far_z)
        if clipped.shape[0] < 3:
            return None

    # --- Project clipped polygon to 2D pixels --------------------------------
    pts_2d = np.asarray(project_fn(clipped))
    if pts_2d.ndim != 2 or pts_2d.shape[0] < 3:
        return None

    pts_z = clipped[:, 2].copy()
    return pts_2d, pts_z


@numba.njit(cache=True)
def _rasterize_triangle_jit(
    v0x: float,
    v0y: float,
    v1x: float,
    v1y: float,
    v2x: float,
    v2y: float,
    z0: float,
    z1: float,
    z2: float,
    depth_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    """
    JIT-compiled triangle rasterizer with depth interpolation.

    Iterates over pixels in the bounding box, computes barycentric coordinates,
    and updates the depth buffer with minimum depth (z-buffer compositing).
    """
    # Compute bounding box clamped to image
    min_x = max(0, int(math.floor(min(v0x, v1x, v2x))))
    max_x = min(image_width - 1, int(math.ceil(max(v0x, v1x, v2x))))
    min_y = max(0, int(math.floor(min(v0y, v1y, v2y))))
    max_y = min(image_height - 1, int(math.ceil(max(v0y, v1y, v2y))))

    if min_x > max_x or min_y > max_y:
        return

    # Edge vectors
    e0x = v1x - v0x
    e0y = v1y - v0y
    e1x = v2x - v0x
    e1y = v2y - v0y

    # Signed area of triangle (2x)
    area = e0x * e1y - e0y * e1x
    if abs(area) < 1e-10:
        return  # degenerate triangle

    inv_area = 1.0 / area

    for y in range(min_y, max_y + 1):
        py = y + 0.5
        v0py = py - v0y

        # Pre-compute terms that only depend on y
        s_row = -v0py * e1x * inv_area
        t_row = e0x * v0py * inv_area

        for x in range(min_x, max_x + 1):
            px = x + 0.5
            v0px = px - v0x

            # Barycentric coordinates
            s = v0px * e1y * inv_area + s_row
            t = t_row - e0y * v0px * inv_area

            if s >= 0.0 and t >= 0.0 and (s + t) <= 1.0:
                # Inside triangle — interpolate depth
                z_interp = (1.0 - s - t) * z0 + s * z1 + t * z2

                # Z-buffer: minimum depth wins
                if z_interp < depth_mask[y, x]:
                    depth_mask[y, x] = z_interp


def rasterize_polygon_with_depth(
    pts_2d: np.ndarray,
    pts_z: np.ndarray,
    mask: np.ndarray,
    depth_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    """
    Rasterize a polygon into the mask and depth_mask using barycentric interpolation.

    Args:
        pts_2d: Nx2 array of 2D pixel coordinates.
        pts_z: N array of z values in camera space.
        mask: H x W uint8 mask to fill (modified in place).
        depth_mask: H x W float32 depth mask (modified in place, min depth wins).
        image_width: Image width.
        image_height: Image height.
    """
    if len(pts_2d) < 3:
        return

    # Fill the binary mask using OpenCV
    pts_clipped = pts_2d.copy()
    pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, image_width - 1)
    pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, image_height - 1)
    pts_int = np.round(pts_clipped).astype(np.int32)

    with suppress(Exception):
        cv2.fillPoly(mask, [pts_int], color=1)

    # Triangulate using fan from first vertex and rasterize with depth
    n_verts = len(pts_2d)
    if n_verts != len(pts_z):
        raise ValueError("pts_2d and pts_z must have the same length")
    for i in range(1, n_verts - 1):
        _rasterize_triangle_jit(
            float(pts_2d[0, 0]),
            float(pts_2d[0, 1]),
            float(pts_2d[i, 0]),
            float(pts_2d[i, 1]),
            float(pts_2d[i + 1, 0]),
            float(pts_2d[i + 1, 1]),
            float(pts_z[0]),
            float(pts_z[i]),
            float(pts_z[i + 1]),
            depth_mask,
            image_width,
            image_height,
        )


def rasterize_triangle_with_depth(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    z0: float,
    z1: float,
    z2: float,
    depth_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    """
    Rasterize a single triangle with depth interpolation using barycentric coordinates.

    Args:
        v0, v1, v2: 2D vertices (x, y).
        z0, z1, z2: Corresponding z values in camera space.
        depth_mask: H x W float32 depth mask (modified in place, min depth wins).
        image_width: Image width.
        image_height: Image height.
    """
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    _rasterize_triangle_jit(
        float(v0[0]),
        float(v0[1]),
        float(v1[0]),
        float(v1[1]),
        float(v2[0]),
        float(v2[1]),
        float(z0),
        float(z1),
        float(z2),
        depth_mask,
        image_width,
        image_height,
    )
