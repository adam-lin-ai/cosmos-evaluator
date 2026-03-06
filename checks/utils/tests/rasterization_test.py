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

"""Unit tests for rasterization module."""

import unittest

import numpy as np

from checks.utils.rasterization import (
    clip_and_project_polygon,
    rasterize_polygon_with_depth,
    rasterize_triangle_with_depth,
)


class TestRasterizeTriangleWithDepth(unittest.TestCase):
    """Test cases for rasterize_triangle_with_depth function."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_width = 100
        self.image_height = 100

    def test_basic_triangle(self):
        """Test rasterizing a basic triangle."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Define a simple triangle
        v0 = np.array([10.0, 10.0])
        v1 = np.array([30.0, 10.0])
        v2 = np.array([20.0, 30.0])
        z0, z1, z2 = 5.0, 5.0, 5.0  # Constant depth

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Check that some pixels were filled
        filled_pixels = np.sum(~np.isinf(depth_mask))
        self.assertGreater(filled_pixels, 0)

        # Check that filled pixels have expected depth
        filled_mask = ~np.isinf(depth_mask)
        self.assertTrue(np.allclose(depth_mask[filled_mask], 5.0))

    def test_triangle_with_varying_depth(self):
        """Test rasterizing a triangle with varying depth values."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Define a triangle with varying depths
        v0 = np.array([10.0, 10.0])
        v1 = np.array([50.0, 10.0])
        v2 = np.array([30.0, 50.0])
        z0, z1, z2 = 5.0, 10.0, 15.0  # Varying depth

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Check that depth varies across the triangle
        filled_mask = ~np.isinf(depth_mask)
        depths = depth_mask[filled_mask]
        self.assertGreater(len(depths), 0)

        # Depth should be interpolated between z0, z1, z2
        self.assertGreaterEqual(np.min(depths), z0 - 0.1)
        self.assertLessEqual(np.max(depths), z2 + 0.1)

    def test_degenerate_triangle(self):
        """Test that degenerate triangles (zero area) are handled."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Collinear points (zero area triangle)
        v0 = np.array([10.0, 10.0])
        v1 = np.array([20.0, 10.0])
        v2 = np.array([30.0, 10.0])  # On the same line as v0-v1
        z0, z1, z2 = 5.0, 5.0, 5.0

        # Should not raise an exception
        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Depth mask should remain all inf (no pixels filled)
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_triangle_outside_image(self):
        """Test triangle completely outside image bounds."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Triangle completely outside image
        v0 = np.array([200.0, 200.0])
        v1 = np.array([250.0, 200.0])
        v2 = np.array([225.0, 250.0])
        z0, z1, z2 = 5.0, 5.0, 5.0

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Depth mask should remain all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_triangle_partially_outside_image(self):
        """Test triangle partially outside image bounds."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Triangle partially outside image (negative coordinates)
        v0 = np.array([-10.0, 10.0])
        v1 = np.array([30.0, 10.0])
        v2 = np.array([10.0, 30.0])
        z0, z1, z2 = 5.0, 5.0, 5.0

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Some pixels should be filled (the part inside the image)
        filled_pixels = np.sum(~np.isinf(depth_mask))
        self.assertGreater(filled_pixels, 0)

    def test_zbuffer_minimum_depth(self):
        """Test that z-buffer keeps minimum depth for overlapping triangles."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # First triangle at depth 10
        v0 = np.array([10.0, 10.0])
        v1 = np.array([50.0, 10.0])
        v2 = np.array([30.0, 50.0])
        rasterize_triangle_with_depth(v0, v1, v2, 10.0, 10.0, 10.0, depth_mask, self.image_width, self.image_height)

        # Second overlapping triangle at depth 5 (closer)
        v0 = np.array([20.0, 15.0])
        v1 = np.array([40.0, 15.0])
        v2 = np.array([30.0, 35.0])
        rasterize_triangle_with_depth(v0, v1, v2, 5.0, 5.0, 5.0, depth_mask, self.image_width, self.image_height)

        # Check that some pixels have depth 5 (from closer triangle)
        self.assertTrue(np.any(np.isclose(depth_mask, 5.0)))

    def test_zbuffer_does_not_overwrite_closer(self):
        """Test that z-buffer does not overwrite closer pixels with farther ones."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # First triangle at depth 5 (closer)
        v0 = np.array([10.0, 10.0])
        v1 = np.array([50.0, 10.0])
        v2 = np.array([30.0, 50.0])
        rasterize_triangle_with_depth(v0, v1, v2, 5.0, 5.0, 5.0, depth_mask, self.image_width, self.image_height)

        # Get the depths before adding the far triangle
        depths_before = depth_mask.copy()

        # Second overlapping triangle at depth 10 (farther)
        v0 = np.array([20.0, 15.0])
        v1 = np.array([40.0, 15.0])
        v2 = np.array([30.0, 35.0])
        rasterize_triangle_with_depth(v0, v1, v2, 10.0, 10.0, 10.0, depth_mask, self.image_width, self.image_height)

        # Depths that were 5 should still be 5 (closer triangle was not overwritten)
        close_mask = np.isclose(depths_before, 5.0)
        self.assertTrue(np.all(np.isclose(depth_mask[close_mask], 5.0)))

    def test_single_pixel_triangle(self):
        """Test very small triangle that might only cover one pixel."""
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Very small triangle
        v0 = np.array([50.0, 50.0])
        v1 = np.array([51.0, 50.0])
        v2 = np.array([50.5, 51.0])
        z0, z1, z2 = 7.0, 7.0, 7.0

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, self.image_width, self.image_height)

        # Should fill at least one pixel or none (very small triangle)
        filled_pixels = np.sum(~np.isinf(depth_mask))
        self.assertGreaterEqual(filled_pixels, 1)


class TestRasterizePolygonWithDepth(unittest.TestCase):
    """Test cases for rasterize_polygon_with_depth function."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_width = 100
        self.image_height = 100

    def test_triangle_polygon(self):
        """Test rasterizing a triangular polygon."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        pts_2d = np.array([[10.0, 10.0], [50.0, 10.0], [30.0, 50.0]])
        pts_z = np.array([5.0, 5.0, 5.0])

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Check that mask has some filled pixels
        self.assertGreater(np.sum(mask), 0)

        # Check that depth mask has some filled pixels
        filled_pixels = np.sum(~np.isinf(depth_mask))
        self.assertGreater(filled_pixels, 0)

    def test_quad_polygon(self):
        """Test rasterizing a quadrilateral polygon."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Square
        pts_2d = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0]])
        pts_z = np.array([10.0, 10.0, 10.0, 10.0])

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Check that mask has filled pixels
        self.assertGreater(np.sum(mask), 0)

        # Check that depth mask has filled pixels with correct depth
        filled_mask = ~np.isinf(depth_mask)
        self.assertTrue(np.allclose(depth_mask[filled_mask], 10.0))

    def test_pentagon_polygon(self):
        """Test rasterizing a pentagon polygon."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Pentagon
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        center = np.array([50.0, 50.0])
        radius = 20.0
        pts_2d = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pts_z = np.full(5, 8.0)

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Check that mask has filled pixels
        self.assertGreater(np.sum(mask), 0)

        # Check depth values
        filled_mask = ~np.isinf(depth_mask)
        self.assertTrue(np.allclose(depth_mask[filled_mask], 8.0))

    def test_too_few_points(self):
        """Test that polygons with fewer than 3 points are handled."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Only 2 points
        pts_2d = np.array([[10.0, 10.0], [50.0, 50.0]])
        pts_z = np.array([5.0, 5.0])

        # Should not raise an exception
        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Mask and depth_mask should remain empty
        self.assertEqual(np.sum(mask), 0)
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_empty_polygon(self):
        """Test that empty polygons are handled."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        pts_2d = np.array([]).reshape(0, 2)
        pts_z = np.array([])

        # Should not raise an exception
        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Mask and depth_mask should remain empty
        self.assertEqual(np.sum(mask), 0)
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_polygon_with_varying_depth(self):
        """Test polygon with varying depth at vertices."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # Triangle with varying depth
        pts_2d = np.array([[10.0, 10.0], [80.0, 10.0], [45.0, 80.0]])
        pts_z = np.array([5.0, 15.0, 10.0])

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Depth values should be interpolated
        filled_mask = ~np.isinf(depth_mask)
        depths = depth_mask[filled_mask]
        self.assertGreater(len(depths), 0)

        # Depth should be within the range of vertex depths
        self.assertGreaterEqual(np.min(depths), 5.0 - 0.1)
        self.assertLessEqual(np.max(depths), 15.0 + 0.1)

    def test_polygon_outside_image(self):
        """Test polygon completely outside image."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        pts_2d = np.array([[200.0, 200.0], [250.0, 200.0], [225.0, 250.0]])
        pts_z = np.array([5.0, 5.0, 5.0])

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Depth mask should remain all inf (triangle rasterization handles this)
        # Mask might have some pixels from fillPoly clipping
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_concave_polygon(self):
        """Test rasterizing a concave polygon (fan triangulation may not work perfectly)."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        # L-shaped polygon (concave)
        pts_2d = np.array(
            [
                [20.0, 20.0],
                [60.0, 20.0],
                [60.0, 40.0],
                [40.0, 40.0],
                [40.0, 60.0],
                [20.0, 60.0],
            ]
        )
        pts_z = np.full(6, 7.0)

        rasterize_polygon_with_depth(pts_2d, pts_z, mask, depth_mask, self.image_width, self.image_height)

        # Should fill some pixels (fan triangulation covers the polygon)
        self.assertGreater(np.sum(mask), 0)


class TestDepthInterpolationAccuracy(unittest.TestCase):
    """Test cases for depth interpolation accuracy."""

    def test_centroid_depth(self):
        """Test that the centroid of a triangle has correctly interpolated depth."""
        image_width, image_height = 200, 200
        depth_mask = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Define a triangle with known vertex positions and depths
        v0 = np.array([50.0, 50.0])
        v1 = np.array([150.0, 50.0])
        v2 = np.array([100.0, 150.0])
        z0, z1, z2 = 6.0, 12.0, 9.0

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, image_width, image_height)

        # Centroid of the triangle
        centroid_x = int((v0[0] + v1[0] + v2[0]) / 3)
        centroid_y = int((v0[1] + v1[1] + v2[1]) / 3)

        # Expected depth at centroid (average of vertex depths)
        expected_depth = (z0 + z1 + z2) / 3

        # Check the depth at centroid
        actual_depth = depth_mask[centroid_y, centroid_x]
        self.assertFalse(np.isinf(actual_depth))
        self.assertAlmostEqual(actual_depth, expected_depth, places=1)

    def test_vertex_depths(self):
        """Test that depths near vertices are close to vertex depths."""
        image_width, image_height = 200, 200
        depth_mask = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Define a large triangle
        v0 = np.array([20.0, 20.0])
        v1 = np.array([180.0, 20.0])
        v2 = np.array([100.0, 180.0])
        z0, z1, z2 = 5.0, 10.0, 15.0

        rasterize_triangle_with_depth(v0, v1, v2, z0, z1, z2, depth_mask, image_width, image_height)

        # Check depth near vertex v0
        depth_near_v0 = depth_mask[int(v0[1]) + 2, int(v0[0]) + 2]
        if not np.isinf(depth_near_v0):
            self.assertAlmostEqual(depth_near_v0, z0, delta=1.0)

        # Check depth near vertex v1
        depth_near_v1 = depth_mask[int(v1[1]) + 2, int(v1[0]) - 2]
        if not np.isinf(depth_near_v1):
            self.assertAlmostEqual(depth_near_v1, z1, delta=1.0)


class TestClipAndProjectPolygon(unittest.TestCase):
    """Test cases for clip_and_project_polygon function."""

    @staticmethod
    def _simple_project(pts_cam: np.ndarray) -> np.ndarray:
        """Simple pinhole projection: (x/z, y/z) with focal length 1."""
        return pts_cam[:, :2] / pts_cam[:, 2:3]

    @staticmethod
    def _identity_project(pts_cam: np.ndarray) -> np.ndarray:
        """Identity-like projection that just returns x, y (for testing)."""
        return pts_cam[:, :2]

    def test_all_vertices_in_front_no_clipping(self):
        """When all vertices are in front, skip clipping and project directly."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 6.0],
                [-1.0, 0.0, 7.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._simple_project, near_z=1e-6)
        self.assertIsNotNone(result)
        pts_2d, pts_z = result
        self.assertEqual(pts_2d.shape, (3, 2))
        self.assertEqual(pts_z.shape, (3,))
        # Check z values are preserved
        np.testing.assert_array_almost_equal(pts_z, [5.0, 6.0, 7.0])
        # Check projection is correct (x/z, y/z)
        np.testing.assert_array_almost_equal(pts_2d[0], [1.0 / 5.0, 0.0 / 5.0])

    def test_all_vertices_behind_camera_returns_none(self):
        """When all vertices are behind the camera, return None immediately."""
        pts_cam = np.array(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -2.0],
                [-1.0, 0.0, -3.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._simple_project, near_z=1e-6)
        self.assertIsNone(result)

    def test_all_vertices_at_near_plane_returns_none(self):
        """When all vertices are exactly below near_z, return None."""
        near_z = 1.0
        pts_cam = np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.3],
                [-1.0, 0.0, 0.9],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._simple_project, near_z=near_z)
        self.assertIsNone(result)

    def test_mixed_in_front_and_behind_clips(self):
        """When polygon straddles the near plane, clipping is performed."""
        # Two vertices in front, one behind
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],  # in front
                [0.0, 1.0, 5.0],  # in front
                [0.0, 0.0, -1.0],  # behind
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._identity_project, near_z=1e-6)
        self.assertIsNotNone(result)
        pts_2d, pts_z = result
        # After clipping, all z-values should be >= near_z
        self.assertTrue(np.all(pts_z >= 1e-6))
        # Should have at least 3 vertices (clipping a triangle with 1 vertex behind
        # produces a quad = 4 vertices)
        self.assertGreaterEqual(pts_2d.shape[0], 3)

    def test_far_plane_clipping(self):
        """Vertices beyond far_z are clipped."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],  # within range
                [0.0, 1.0, 5.0],  # within range
                [0.0, 0.0, 100.0],  # beyond far plane
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._identity_project, near_z=1e-6, far_z=50.0)
        self.assertIsNotNone(result)
        pts_2d, pts_z = result
        # After clipping, all z-values should be <= 50.0
        self.assertTrue(np.all(pts_z <= 50.0 + 1e-9))

    def test_all_within_near_and_far_no_clipping(self):
        """When all vertices are within [near_z, far_z], skip clipping."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 10.0],
                [-1.0, 0.0, 15.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._identity_project, near_z=1.0, far_z=20.0)
        self.assertIsNotNone(result)
        pts_2d, pts_z = result
        # Should have exactly 3 vertices (no clipping)
        self.assertEqual(pts_2d.shape[0], 3)
        np.testing.assert_array_almost_equal(pts_z, [5.0, 10.0, 15.0])

    def test_all_beyond_far_plane_returns_none(self):
        """When all vertices are beyond far_z, return None."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 100.0],
                [0.0, 1.0, 200.0],
                [-1.0, 0.0, 150.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._identity_project, near_z=1.0, far_z=50.0)
        self.assertIsNone(result)

    def test_too_few_vertices_returns_none(self):
        """Polygons with fewer than 3 vertices return None."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 5.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._simple_project)
        self.assertIsNone(result)

    def test_empty_polygon_returns_none(self):
        """Empty polygon returns None."""
        pts_cam = np.zeros((0, 3))
        result = clip_and_project_polygon(pts_cam, self._simple_project)
        self.assertIsNone(result)

    def test_invalid_shape_raises(self):
        """Non-Nx3 input raises ValueError."""
        pts_cam = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with self.assertRaises(ValueError):
            clip_and_project_polygon(pts_cam, self._simple_project)

    def test_quad_polygon_all_in_front(self):
        """Test with a 4-vertex polygon fully in front of the camera."""
        pts_cam = np.array(
            [
                [-1.0, -1.0, 10.0],
                [1.0, -1.0, 10.0],
                [1.0, 1.0, 10.0],
                [-1.0, 1.0, 10.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._simple_project)
        self.assertIsNotNone(result)
        pts_2d, pts_z = result
        self.assertEqual(pts_2d.shape, (4, 2))
        np.testing.assert_array_almost_equal(pts_z, [10.0, 10.0, 10.0, 10.0])

    def test_project_fn_returning_bad_shape_returns_none(self):
        """If project_fn returns unexpected shape, return None."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 5.0],
                [-1.0, 0.0, 5.0],
            ]
        )

        def bad_project(pts):
            return np.array([1.0, 2.0])  # Wrong shape

        result = clip_and_project_polygon(pts_cam, bad_project)
        self.assertIsNone(result)

    def test_z_values_are_copied(self):
        """Returned z-values should be a copy, not a view into the input."""
        pts_cam = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 6.0],
                [-1.0, 0.0, 7.0],
            ]
        )
        result = clip_and_project_polygon(pts_cam, self._identity_project)
        self.assertIsNotNone(result)
        _, pts_z = result
        # Mutating pts_z should not affect pts_cam
        pts_z[0] = 999.0
        self.assertAlmostEqual(pts_cam[0, 2], 5.0)


if __name__ == "__main__":
    unittest.main()
