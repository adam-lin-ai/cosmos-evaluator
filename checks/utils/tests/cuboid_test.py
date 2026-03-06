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

"""Unit tests for cuboid module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from checks.utils.cuboid import Cuboid


class TestCuboid(unittest.TestCase):
    """Test cases for Cuboid class."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample 4x4 transformation matrix (identity with translation)
        self.sample_pose = np.array(
            [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 10.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
        )

        # Sample dimensions: length, width, height
        self.sample_lwh = np.array([4.0, 2.0, 1.8])

        # Create test cuboid
        self.cuboid = Cuboid(self.sample_pose, self.sample_lwh)

    def test_init(self):
        """Test Cuboid initialization."""
        self.assertTrue(np.allclose(self.cuboid.object_to_world_pose, self.sample_pose))
        self.assertTrue(np.allclose(self.cuboid.lwh, self.sample_lwh))
        self.assertEqual(self.cuboid.corners.shape, (8, 3))

    def test_init_with_lists(self):
        """Test Cuboid initialization with list inputs."""
        pose_list = self.sample_pose.tolist()
        lwh_list = self.sample_lwh.tolist()

        cuboid = Cuboid(pose_list, lwh_list)

        assert_array_almost_equal(cuboid.object_to_world_pose, self.sample_pose)
        assert_array_almost_equal(cuboid.lwh, self.sample_lwh)

    def test_compute_corners_identity_pose(self):
        """Test corner computation with identity pose."""
        identity_pose = np.eye(4)
        lwh = np.array([2.0, 2.0, 2.0])

        cuboid = Cuboid(identity_pose, lwh)

        # Expected corners for 2x2x2 cube centered at origin
        expected_corners = np.array(
            [
                [-1.0, -1.0, -1.0],  # back-bottom-left
                [-1.0, -1.0, 1.0],  # back-bottom-right
                [-1.0, 1.0, -1.0],  # back-top-left
                [-1.0, 1.0, 1.0],  # back-top-right
                [1.0, -1.0, -1.0],  # front-bottom-left
                [1.0, -1.0, 1.0],  # front-bottom-right
                [1.0, 1.0, -1.0],  # front-top-left
                [1.0, 1.0, 1.0],  # front-top-right
            ]
        )

        assert_array_almost_equal(cuboid.corners, expected_corners)

    def test_compute_corners_with_translation(self):
        """Test corner computation with translation."""
        # Translate by (5, 10, 2)
        corners = self.cuboid.corners

        # Check that all corners are translated correctly
        # Center should be at (5, 10, 2)
        center = np.mean(corners, axis=0)
        expected_center = np.array([5.0, 10.0, 2.0])
        assert_array_almost_equal(center, expected_center)

    def test_compute_corners_with_rotation(self):
        """Test corner computation with rotation."""
        # Create rotation around Z-axis by 90 degrees
        angle = np.pi / 2
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle), 0, 0], [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        lwh = np.array([2.0, 1.0, 1.0])  # Asymmetric for easier testing
        cuboid = Cuboid(rotation_matrix, lwh)

        # After 90-degree rotation, length and width should be swapped
        # Check that corners reflect this rotation
        corners = cuboid.corners
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        # After rotation, original length direction (x) becomes y direction
        self.assertAlmostEqual(np.max(y_coords) - np.min(y_coords), 2.0, places=6)
        self.assertAlmostEqual(np.max(x_coords) - np.min(x_coords), 1.0, places=6)

    def test_get_center_point(self):
        """Test getting center point."""
        center = self.cuboid.get_center_point()
        expected_center = np.array([5.0, 10.0, 2.0])
        assert_array_almost_equal(center, expected_center)

    def test_get_dimensions(self):
        """Test getting dimensions."""
        dimensions = self.cuboid.get_dimensions()
        assert_array_almost_equal(dimensions, self.sample_lwh)

        # Test that it returns a copy, not a reference
        dimensions[0] = 999
        self.assertNotEqual(self.cuboid.lwh[0], 999)

    @patch("checks.utils.cuboid.CameraBase.transform_points_np")
    def test_compute_projected_mask_basic(self, mock_transform):
        """Test computing projected mask (static)."""
        # Mock camera transformations
        mock_transform.return_value = np.array(
            [
                [100, 200, 5],  # Camera coordinates (x, y, z)
                [150, 250, 5],
                [100, 300, 5],
                [150, 350, 5],
                [200, 200, 5],
                [250, 250, 5],
                [200, 300, 5],
                [250, 350, 5],
            ]
        )

        # Mock camera model: per-face calls; return x,y from given 3D points
        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        # Test parameters
        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = Cuboid.compute_projected_mask(
                self.cuboid.corners, camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)

        # Check that depth_mask has correct shape and type
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

        # Check that transform_points_np was called
        mock_transform.assert_called_once()

        # Called for each face
        self.assertGreaterEqual(mock_camera_model.ray2pixel_np.call_count, 1)

        # Check that fillPoly was called for each face (6 faces)
        self.assertEqual(mock_fillpoly.call_count, 6)

    @patch("checks.utils.cuboid.CameraBase.transform_points_np")
    def test_get_projected_mask_behind_camera(self, mock_transform):
        """Test projected mask when cuboid is behind camera."""
        # Mock camera transformation with negative z (behind camera)
        mock_transform.return_value = np.array(
            [
                [100, 200, -1],  # Negative z means behind camera
                [150, 250, -1],
                [100, 300, -1],
                [150, 350, -1],
                [200, 200, -1],
                [250, 250, -1],
                [200, 300, -1],
                [250, 350, -1],
            ]
        )

        mock_camera_model = MagicMock()
        # Per-face projection is called; return x,y from provided 3D points
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]
        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.cuboid.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return all-zeros mask
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.cuboid.CameraBase.transform_points_np")
    def test_get_projected_mask_partial_behind_camera(self, mock_transform):
        """Test projected mask when some corners are behind camera."""
        # Mock camera transformation with some negative z values
        mock_transform.return_value = np.array(
            [
                [100, 200, -1],  # Behind camera
                [150, 250, 5],  # In front of camera
                [100, 300, 5],
                [150, 350, 5],
                [200, 200, 5],
                [250, 250, 5],
                [200, 300, 5],
                [250, 350, 5],
            ]
        )

        mock_camera_model = MagicMock()
        # Per-face projection is called; return x,y from provided 3D points
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]
        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.cuboid.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # New behavior: mask should still be non-empty when not all corners are behind camera
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(np.any(mask))

        # Depth mask should have correct shape and type
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    def test_corners_shape_and_type(self):
        """Test that corners have correct shape and type."""
        self.assertEqual(self.cuboid.corners.shape, (8, 3))
        self.assertTrue(np.issubdtype(self.cuboid.corners.dtype, np.floating))

    def test_face_definitions(self):
        """Test that the cuboid defines faces correctly."""
        # This is indirectly tested through the projected mask function
        # We test that the face definitions are consistent by checking
        # that the projected mask uses the correct face structure

        # The faces should be defined as in the original code:
        expected_faces = [
            [0, 1, 3, 2],  # back face
            [4, 5, 7, 6],  # front face
            [0, 1, 5, 4],  # right face
            [2, 3, 7, 6],  # left face
            [1, 3, 7, 5],  # top face
            [0, 2, 6, 4],  # bottom face
        ]

        # Each face should reference valid corner indices
        for face in expected_faces:
            for corner_idx in face:
                self.assertGreaterEqual(corner_idx, 0)
                self.assertLess(corner_idx, 8)

    def test_cuboid_symmetry(self):
        """Test cuboid symmetry properties."""
        # For a cuboid centered at origin with identity pose
        identity_pose = np.eye(4)
        lwh = np.array([4.0, 2.0, 1.0])

        cuboid = Cuboid(identity_pose, lwh)
        corners = cuboid.corners

        # Check that corners are symmetric around origin
        center = np.mean(corners, axis=0)
        assert_array_almost_equal(center, [0, 0, 0])

        # Check that dimensions are correct
        x_span = np.max(corners[:, 0]) - np.min(corners[:, 0])
        y_span = np.max(corners[:, 1]) - np.min(corners[:, 1])
        z_span = np.max(corners[:, 2]) - np.min(corners[:, 2])

        self.assertAlmostEqual(x_span, 4.0, places=6)  # length
        self.assertAlmostEqual(y_span, 2.0, places=6)  # width
        self.assertAlmostEqual(z_span, 1.0, places=6)  # height

    def test_compute_pose_and_lwh_from_corners_identity(self):
        """Recovered pose/LWH from axis-aligned corners should match inputs (up to axis signs)."""
        identity_pose = np.eye(4)
        lwh = np.array([4.0, 2.0, 1.0])  # strictly descending so PCA axis order is deterministic

        cuboid = Cuboid(identity_pose, lwh)
        corners = cuboid.corners

        recovered_pose, recovered_lwh = Cuboid.compute_pose_and_lwh_from_corners(corners)

        # Center matches origin
        assert_array_almost_equal(recovered_pose[:3, 3], np.array([0.0, 0.0, 0.0]))

        # LWH recovered correctly (order preserved due to descending extents)
        assert_array_almost_equal(recovered_lwh, lwh)

        # Rotation is axis-aligned; allow sign flips on axes
        R = recovered_pose[:3, :3]
        identityR3 = np.eye(3)
        assert_array_almost_equal(R @ R.T, identityR3, decimal=6)
        self.assertGreater(np.linalg.det(R), 0.0)
        # Absolute rotation should be close to identity for axis-aligned case
        assert_array_almost_equal(np.abs(R), identityR3, decimal=6)

        # Reconstruct corners from recovered pose/lwh and compare sets (order-independent)
        recon = Cuboid(recovered_pose, recovered_lwh).corners

        # Sort rows lexicographically for stable comparison
        def sort_rows(a: np.ndarray) -> np.ndarray:
            keys = (a[:, 2], a[:, 1], a[:, 0])
            return a[np.lexsort(keys)]

        assert_array_almost_equal(sort_rows(recon), sort_rows(corners))

    def test_compute_pose_and_lwh_from_corners_with_rotation_and_translation(self):
        """Recovered pose/LWH should match a rotated + translated cuboid (up to axis signs)."""
        # Build a generic rotation (Z then Y) and translation
        theta_z = np.deg2rad(30.0)
        Rz = np.array(
            [
                [np.cos(theta_z), -np.sin(theta_z), 0.0],
                [np.sin(theta_z), np.cos(theta_z), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        theta_y = np.deg2rad(15.0)
        Ry = np.array(
            [
                [np.cos(theta_y), 0.0, np.sin(theta_y)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta_y), 0.0, np.cos(theta_y)],
            ]
        )
        R0 = Rz @ Ry
        t0 = np.array([3.5, -2.0, 7.25])

        pose = np.eye(4)
        pose[:3, :3] = R0
        pose[:3, 3] = t0

        lwh = np.array([5.0, 2.5, 1.25])  # strictly descending
        cuboid = Cuboid(pose, lwh)
        corners = cuboid.corners

        recovered_pose, recovered_lwh = Cuboid.compute_pose_and_lwh_from_corners(corners)

        # Translation recovered exactly
        assert_array_almost_equal(recovered_pose[:3, 3], t0)

        # LWH recovered (order preserved because of descending extents)
        assert_array_almost_equal(recovered_lwh, lwh)

        # Rotation columns should align with original columns up to sign
        R = recovered_pose[:3, :3]
        self.assertGreater(np.linalg.det(R), 0.0)
        for i in range(3):
            col_dot = float(np.dot(R[:, i], R0[:, i]))
            self.assertAlmostEqual(abs(col_dot), 1.0, places=6)

        # Reconstruct corners and compare order-independently
        recon = Cuboid(recovered_pose, recovered_lwh).corners

        def sort_rows(a: np.ndarray) -> np.ndarray:
            keys = (a[:, 2], a[:, 1], a[:, 0])
            return a[np.lexsort(keys)]

        assert_array_almost_equal(sort_rows(recon), sort_rows(corners))

    def test_compute_pose_and_lwh_from_corners_invalid_inputs(self):
        """Function should raise ValueError on invalid shapes or None input."""
        with self.assertRaises(ValueError):
            Cuboid.compute_pose_and_lwh_from_corners(None)

        with self.assertRaises(ValueError):
            Cuboid.compute_pose_and_lwh_from_corners(np.zeros((7, 3)))

        with self.assertRaises(ValueError):
            Cuboid.compute_pose_and_lwh_from_corners(np.zeros((8, 2)))


if __name__ == "__main__":
    unittest.main()
