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

"""Unit tests for coord_transforms module."""

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from checks.utils.coord_transforms import (
    clip_polygon_to_z_planes,
    extract_rpy_in_flu,
    get_object_to_camera_pose,
)
from checks.utils.cuboid import Cuboid


class TestCoordTransforms(unittest.TestCase):
    """Test cases for coordinate transformation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample 4x4 transformation matrices
        self.sample_object_to_world = np.array(
            [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 10.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
        )

        self.sample_camera_pose = np.array(
            [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        )

    def test_get_object_to_camera_pose_basic(self):
        """Test basic object to camera pose transformation."""
        tracked_object = Cuboid(self.sample_object_to_world, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, self.sample_camera_pose)

        # Verify result is a 4x4 matrix
        self.assertEqual(result.shape, (4, 4))

        # Verify it's a valid transformation matrix (last row should be [0,0,0,1])
        assert_array_almost_equal(result[3, :], [0, 0, 0, 1])

    def test_get_object_to_camera_pose_identity(self):
        """Test object to camera pose with identity matrices."""
        identity = np.eye(4)
        tracked_object = Cuboid(identity, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, identity)

        # With identity matrices, result should be identity
        assert_array_almost_equal(result, identity)

    def test_get_object_to_camera_pose_translation_only(self):
        """Test object to camera pose with translation only."""
        object_translation = np.eye(4)
        object_translation[:3, 3] = [1, 2, 3]

        camera_translation = np.eye(4)
        camera_translation[:3, 3] = [0.5, 1.0, 1.5]

        tracked_object = Cuboid(object_translation, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, camera_translation)

        # Expected translation should be object - camera
        expected_translation = [0.5, 1.0, 1.5]  # [1-0.5, 2-1.0, 3-1.5]
        assert_array_almost_equal(result[:3, 3], expected_translation)

    def test_extract_rpy_in_flu_identity(self):
        """Test RPY extraction from identity matrix."""
        identity = np.eye(3)
        roll, pitch, yaw = extract_rpy_in_flu(identity)

        # Identity matrix should give zero rotations
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_x(self):
        """Test RPY extraction from rotation around X-axis (roll)."""
        angle = np.pi / 4  # 45 degrees
        R_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

        roll, pitch, yaw = extract_rpy_in_flu(R_x.T)  # Input is R_flu_to_rdf, so we transpose

        self.assertAlmostEqual(roll, angle, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_y(self):
        """Test RPY extraction from rotation around Y-axis (pitch)."""
        angle = np.pi / 6  # 30 degrees
        R_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

        roll, pitch, yaw = extract_rpy_in_flu(R_y.T)

        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, angle, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_z(self):
        """Test RPY extraction from rotation around Z-axis (yaw)."""
        angle = np.pi / 3  # 60 degrees
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

        roll, pitch, yaw = extract_rpy_in_flu(R_z.T)

        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, angle, places=6)

    def test_extract_rpy_in_flu_combined_rotation(self):
        """Test RPY extraction from combined rotations."""
        roll_angle = np.pi / 6  # 30 degrees
        pitch_angle = np.pi / 8  # 22.5 degrees
        yaw_angle = np.pi / 4  # 45 degrees

        # Create individual rotation matrices
        R_x = np.array(
            [[1, 0, 0], [0, np.cos(roll_angle), -np.sin(roll_angle)], [0, np.sin(roll_angle), np.cos(roll_angle)]]
        )

        R_y = np.array(
            [[np.cos(pitch_angle), 0, np.sin(pitch_angle)], [0, 1, 0], [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]]
        )

        R_z = np.array(
            [[np.cos(yaw_angle), -np.sin(yaw_angle), 0], [np.sin(yaw_angle), np.cos(yaw_angle), 0], [0, 0, 1]]
        )

        # Combined rotation (ZYX order)
        R_combined = R_z @ R_y @ R_x

        roll, pitch, yaw = extract_rpy_in_flu(R_combined.T)

        # Should approximately match original angles
        self.assertAlmostEqual(roll, roll_angle, places=5)
        self.assertAlmostEqual(pitch, pitch_angle, places=5)
        self.assertAlmostEqual(yaw, yaw_angle, places=5)

    def test_extract_rpy_in_flu_gimbal_lock_positive(self):
        """Test RPY extraction in positive gimbal lock condition."""
        # Create a rotation matrix with pitch = 90 degrees (gimbal lock)
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        roll, pitch, yaw = extract_rpy_in_flu(R.T)

        # In gimbal lock, pitch should be ±π/2, yaw should be 0
        self.assertAlmostEqual(pitch, np.pi / 2, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_gimbal_lock_negative(self):
        """Test RPY extraction in negative gimbal lock condition."""
        # Create a rotation matrix with pitch = -90 degrees (gimbal lock)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

        roll, pitch, yaw = extract_rpy_in_flu(R.T)

        # In gimbal lock, pitch should be -π/2, yaw should be 0
        self.assertAlmostEqual(pitch, -np.pi / 2, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_get_object_to_camera_pose_with_numpy_array(self):
        """Test object to camera pose when object_to_world is already numpy array."""
        tracked_object = Cuboid(self.sample_object_to_world, np.array([1.0, 1.0, 1.0]))  # numpy array provided

        result = get_object_to_camera_pose(tracked_object, self.sample_camera_pose)

        # Should work with both numpy arrays and lists
        self.assertEqual(result.shape, (4, 4))
        assert_array_almost_equal(result[3, :], [0, 0, 0, 1])

    def test_extract_rpy_large_angles(self):
        """Test RPY extraction with large angles."""
        # Test with angles greater than π
        large_angle = 3 * np.pi / 2
        R_z = np.array(
            [[np.cos(large_angle), -np.sin(large_angle), 0], [np.sin(large_angle), np.cos(large_angle), 0], [0, 0, 1]]
        )

        roll, pitch, yaw = extract_rpy_in_flu(R_z.T)

        # The extracted angle should be within [-π, π]
        self.assertGreaterEqual(yaw, -np.pi)
        self.assertLessEqual(yaw, np.pi)

        # The rotation should be equivalent
        expected_angle = large_angle - 2 * np.pi  # Normalize to [-π, π]
        self.assertAlmostEqual(yaw, expected_angle, places=6)

    # -------------------------------
    # clip_polygon_to_z_planes tests
    # -------------------------------
    def test_clip_polygon_no_planes_returns_copy(self):
        """Polygon should be unchanged when both planes are None."""
        poly = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=None, far_plane_z=None)
        # Should return a copy, not the same object
        self.assertIsNot(poly, out)
        assert_array_almost_equal(out, poly)

    def test_clip_polygon_empty_returns_empty(self):
        """Empty input should return empty output."""
        poly = np.zeros((0, 3))
        out = clip_polygon_to_z_planes(poly)
        self.assertEqual(out.shape, (0, 3))

    def test_clip_polygon_invalid_shape_raises(self):
        """Invalid shapes should raise ValueError."""
        with self.assertRaises(ValueError):
            clip_polygon_to_z_planes(np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            clip_polygon_to_z_planes(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_clip_polygon_all_inside_near(self):
        """Polygon fully in front of near plane should remain unchanged."""
        poly = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        assert_array_almost_equal(out, poly)

    def test_clip_polygon_all_outside_near(self):
        """Polygon fully behind near plane should be clipped away."""
        poly = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        self.assertEqual(out.shape, (0, 3))

    def test_clip_polygon_near_plane_intersection(self):
        """Polygon partially behind near plane should be clipped with correct intersections."""
        # Two vertices behind near plane z=1.0 (z=0.5) and two in front (z=2.0)
        p0 = np.array([0.0, 0.0, 0.5])
        p1 = np.array([1.0, 0.0, 0.5])
        p2 = np.array([1.0, 1.0, 2.0])
        p3 = np.array([0.0, 1.0, 2.0])
        poly = np.stack([p0, p1, p2, p3], axis=0)

        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        # Expect 4 vertices after clipping:
        # intersections on edges (p3->p0) and (p1->p2), plus kept p2, p3
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[0], 4)
        # All points must satisfy z >= 1.0
        self.assertTrue(np.all(out[:, 2] >= 1.0 - 1e-9))

        # Expected intersection points at z=1
        inter_a = np.array([0.0, 1.0 / 3.0, 1.0])  # on edge p3->p0
        inter_b = np.array([1.0, 1.0 / 3.0, 1.0])  # on edge p1->p2

        # Check that p2 and p3 are present, and intersections present (order-independent)
        def contains_point(points, pt, tol=1e-6):
            return np.any(np.all(np.isclose(points, pt, atol=tol), axis=1))

        self.assertTrue(contains_point(out, p2))
        self.assertTrue(contains_point(out, p3))
        self.assertTrue(contains_point(out, inter_a))
        self.assertTrue(contains_point(out, inter_b))

    def test_clip_polygon_far_plane_intersection(self):
        """Polygon partially beyond far plane should be clipped with correct intersections."""
        # Two vertices beyond far plane z=1.0 (z=2.0) and two inside (z=0.5)
        p0 = np.array([0.0, 0.0, 2.0])
        p1 = np.array([1.0, 0.0, 2.0])
        p2 = np.array([1.0, 1.0, 0.5])
        p3 = np.array([0.0, 1.0, 0.5])
        poly = np.stack([p0, p1, p2, p3], axis=0)

        out = clip_polygon_to_z_planes(poly, near_plane_z=None, far_plane_z=1.0)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[0], 4)
        # All points must satisfy z <= 1.0
        self.assertTrue(np.all(out[:, 2] <= 1.0 + 1e-9))

        # Expected intersection points at z=1 on edges (p0->p3) and (p1->p2)
        inter_a = np.array([0.0, 2.0 / 3.0, 1.0])
        inter_b = np.array([1.0, 2.0 / 3.0, 1.0])

        def contains_point(points, pt, tol=1e-6):
            return np.any(np.all(np.isclose(points, pt, atol=tol), axis=1))

        self.assertTrue(contains_point(out, p2))
        self.assertTrue(contains_point(out, p3))
        self.assertTrue(contains_point(out, inter_a))
        self.assertTrue(contains_point(out, inter_b))

    def test_clip_polygon_both_planes(self):
        """Clipping against both planes should keep only points within the slab."""
        # Polygon with z values spanning from -1 to 3
        poly = np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 3.0],
            ]
        )
        out = clip_polygon_to_z_planes(poly, near_plane_z=0.0, far_plane_z=2.0)
        # All z must be within [0, 2]
        self.assertTrue(np.all(out[:, 2] >= -1e-9))
        self.assertTrue(np.all(out[:, 2] <= 2.0 + 1e-9))
        # Should have at least 3 vertices to form a polygon
        self.assertGreaterEqual(out.shape[0], 3)

    def test_clip_polygon_vertex_on_plane_no_duplicates(self):
        """Vertices exactly on plane should be handled without duplicate consecutive points."""
        # One vertex exactly on near plane z=1.0, others in front
        poly = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.5], [1.0, 1.0, 1.5], [0.0, 1.0, 1.5]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        # Should retain a valid polygon with no consecutive duplicate rows
        self.assertGreaterEqual(out.shape[0], 3)
        if out.shape[0] >= 2:
            diffs = np.linalg.norm(out[1:] - out[:-1], axis=1)
            self.assertTrue(np.all(diffs > 1e-12))


if __name__ == "__main__":
    unittest.main()
