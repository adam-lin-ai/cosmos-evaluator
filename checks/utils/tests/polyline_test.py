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

"""Unit tests for polyline module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from checks.utils.polyline import Polyline


class TestPolyline(unittest.TestCase):
    """Test cases for Polyline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample polyline vertices
        self.sample_vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 3.0]])
        self.polyline = Polyline(self.sample_vertices, half_width_meters=0.5)

    def test_init(self):
        """Test Polyline initialization with valid inputs."""
        self.assertTrue(np.allclose(self.polyline.vertices_world, self.sample_vertices))
        self.assertEqual(self.polyline.half_width_m, 0.5)
        self.assertEqual(self.polyline.min_cutoff_distance, 3.0)
        self.assertEqual(self.polyline.max_cutoff_distance, 50.0)

    def test_init_with_custom_parameters(self):
        """Test Polyline initialization with custom parameters."""
        polyline = Polyline(
            self.sample_vertices, half_width_meters=1.0, min_cutoff_distance=5.0, max_cutoff_distance=100.0
        )
        self.assertEqual(polyline.half_width_m, 1.0)
        self.assertEqual(polyline.min_cutoff_distance, 5.0)
        self.assertEqual(polyline.max_cutoff_distance, 100.0)

    def test_init_with_list_input(self):
        """Test Polyline initialization with list input."""
        vertices_list = self.sample_vertices.tolist()
        polyline = Polyline(vertices_list)
        assert_array_almost_equal(polyline.vertices_world, self.sample_vertices)

    def test_init_invalid_shape_1d(self):
        """Test Polyline initialization with 1D array."""
        with self.assertRaises(ValueError):
            Polyline(np.array([1.0, 2.0, 3.0]))

    def test_init_invalid_shape_wrong_columns(self):
        """Test Polyline initialization with wrong number of columns."""
        with self.assertRaises(ValueError):
            Polyline(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_init_invalid_shape_single_point(self):
        """Test Polyline initialization with single point."""
        with self.assertRaises(ValueError):
            Polyline(np.array([[1.0, 2.0, 3.0]]))

    def test_init_negative_half_width(self):
        """Test Polyline initialization with negative half_width."""
        polyline = Polyline(self.sample_vertices, half_width_meters=-1.0)
        # Should clamp to 0.0
        self.assertEqual(polyline.half_width_m, 0.0)

    def test_init_invalid_cutoff_distances(self):
        """Test Polyline initialization with min >= max cutoff distance."""
        with self.assertRaises(ValueError):
            Polyline(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=5.0)

    def test_init_equal_cutoff_distances(self):
        """Test Polyline initialization with equal cutoff distances."""
        with self.assertRaises(ValueError):
            Polyline(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=10.0)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_basic(self, mock_transform):
        """Test get_projected_mask with basic polyline."""

        # Mock camera transformation: transform points to camera coordinates
        # For a simple case, return points with positive z (in front of camera)
        def transform_side_effect(points, pose):
            # Simple identity-like transform: just return points with z=10
            result = np.asarray(points).copy()
            result[:, 2] = 10.0  # All points at z=10 in camera frame
            return result

        mock_transform.side_effect = transform_side_effect

        # Mock camera model: return x,y from 3D points
        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = self.polyline.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)

        # Check that depth_mask has correct shape and type
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

        # Check that transform_points_np was called (at least once per segment)
        self.assertGreaterEqual(mock_transform.call_count, 1)

        # Check that ray2pixel_np was called
        self.assertGreaterEqual(mock_camera_model.ray2pixel_np.call_count, 1)

        # Check that fillPoly was called
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_behind_camera(self, mock_transform):
        """Test get_projected_mask when polyline is behind camera."""

        # Mock camera transformation with negative z (behind camera)
        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            result[:, 2] = -10.0  # All points behind camera
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return all-zeros mask (clipped away)
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf (no valid pixels)
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_zero_length_segment(self, mock_transform):
        """Test get_projected_mask with zero-length segment (duplicate points)."""
        # Polyline with duplicate consecutive points
        vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        polyline = Polyline(vertices)

        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Zero-length segment should be skipped, but other segments should still project
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    def test_get_projected_mask_invalid_pose_shape(self):
        """Test get_projected_mask with invalid pose shape."""
        mock_camera_model = MagicMock()
        invalid_pose = np.eye(3)  # 3x3 instead of 4x4

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(invalid_pose, mock_camera_model, 640, 480)

    def test_get_projected_mask_invalid_image_dimensions(self):
        """Test get_projected_mask with invalid image dimensions."""
        mock_camera_model = MagicMock()
        camera_to_world_pose = np.eye(4)

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(camera_to_world_pose, mock_camera_model, 0, 480)

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(camera_to_world_pose, mock_camera_model, 640, -1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_with_rotation(self, mock_transform):
        """Test get_projected_mask with rotated camera pose."""
        # Create rotation around Z-axis by 90 degrees
        angle = np.pi / 2
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly"):
            mask, depth_mask = self.polyline.get_projected_mask(
                rotation_matrix, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_multiple_segments(self, mock_transform):
        """Test get_projected_mask with polyline having multiple segments."""
        # Polyline with 4 points = 3 segments
        vertices = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        polyline = Polyline(vertices, half_width_meters=0.5)

        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = polyline.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)
        # Should have called fillPoly for each segment (3 segments)
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_cutoff_distances(self, mock_transform):
        """Test get_projected_mask respects cutoff distances."""
        # Create polyline with custom cutoff distances
        polyline = Polyline(self.sample_vertices, min_cutoff_distance=5.0, max_cutoff_distance=20.0)

        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            # Set z values that are outside the cutoff range
            result[:, 2] = 2.0  # Less than min_cutoff_distance
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Points outside cutoff range should be clipped away
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_camera_model_exception(self, mock_transform):
        """Test get_projected_mask handles camera model exceptions gracefully."""

        def transform_side_effect(points, pose):
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        # Mock camera model that raises exception
        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = Exception("Camera projection failed")

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        # Should not raise exception, but return empty mask
        mask, depth_mask = self.polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    def test_vertices_world_immutability(self):
        """Test that modifying input array doesn't affect polyline."""
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        polyline = Polyline(vertices)

        # Modify original array
        vertices[0, 0] = 999.0

        # Polyline should have its own copy
        self.assertNotEqual(polyline.vertices_world[0, 0], 999.0)
        self.assertEqual(polyline.vertices_world[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
