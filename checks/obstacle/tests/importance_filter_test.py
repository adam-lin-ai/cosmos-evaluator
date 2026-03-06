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

"""Unit tests for importance_filter module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.obstacle.importance_filter import ImportanceFilter
from checks.utils.cuboid import Cuboid


class TestImportanceFilter(unittest.TestCase):
    """Test cases for ImportanceFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            "distance_threshold_m": 50.0,
            "skip_oncoming_obstacles": True,
            "relevant_lanes": ["ego", "left", "right"],
        }

        self.logger = MagicMock()
        self.filter = ImportanceFilter(self.base_config, self.logger)

        # Sample camera pose (identity matrix)
        self.camera_pose = np.eye(4)

        # Sample tracked object
        self.sample_object = {
            "geometry": Cuboid(np.eye(4), np.array([10.0, 0.0, 0.0])),
            "is_static": False,
        }

        self.static_object = {
            "geometry": Cuboid(np.eye(4), np.array([10.0, 0.0, 0.0])),
            "is_static": True,
        }

        self.no_centroid_object = {
            "geometry": None,
            "is_static": False,
        }

    def test_init(self):
        """Test ImportanceFilter initialization."""
        self.assertEqual(self.filter.config, self.base_config)
        self.assertEqual(self.filter.logger, self.logger)

    def test_init_without_logger(self):
        """Test ImportanceFilter initialization without logger."""
        filter_no_logger = ImportanceFilter(self.base_config)
        self.assertIsNone(filter_no_logger.logger)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_should_process_object_passes_all_filters(self, mock_get_pose):
        """Test object that passes all filters."""
        # Mock object pose that's close, not oncoming, in relevant lane
        mock_get_pose.return_value = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0m lateral (ego lane)
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 10.0],  # 10m forward
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        with patch.object(self.filter, "_get_object_yaw_in_ego_frame", return_value=0.0):  # Not oncoming
            should_process, reason = self.filter.should_process_object(
                self.sample_object, self.camera_pose, track_id=123
            )

        self.assertTrue(should_process)
        self.assertEqual(reason, "passed")

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_should_process_object_fails_distance_filter(self, mock_get_pose):
        """Test object that fails distance filter."""
        # Mock object pose that's too far
        mock_get_pose.return_value = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0m lateral
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 100.0],  # 100m forward (beyond threshold)
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        should_process, reason = self.filter.should_process_object(self.sample_object, self.camera_pose, track_id=123)

        self.assertFalse(should_process)
        self.assertIn("distance_threshold_m", reason)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_should_process_object_fails_oncoming_filter(self, mock_get_pose):
        """Test object that fails oncoming filter."""
        # Mock object pose that's close and in relevant lane
        mock_get_pose.return_value = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0m lateral
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 10.0],  # 10m forward
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        with patch.object(self.filter, "_get_object_yaw_in_ego_frame", return_value=180.0):  # Oncoming
            should_process, reason = self.filter.should_process_object(
                self.sample_object, self.camera_pose, track_id=123
            )

        self.assertFalse(should_process)
        self.assertIn("skip_oncoming_obstacles", reason)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_should_process_object_fails_lane_filter(self, mock_get_pose):
        """Test object that fails lane filter."""
        # Mock object pose that's far to the side (not in relevant lanes)
        mock_get_pose.return_value = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],  # 10m lateral (far outside lanes)
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 10.0],  # 10m forward
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        with patch.object(self.filter, "_get_object_yaw_in_ego_frame", return_value=0.0):  # Not oncoming
            should_process, reason = self.filter.should_process_object(
                self.sample_object, self.camera_pose, track_id=123
            )

        self.assertFalse(should_process)
        self.assertIn("relevant_lanes", reason)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_is_oncoming_true(self, mock_get_pose):
        """Test oncoming detection when object is oncoming."""
        mock_get_pose.return_value = np.eye(4)

        with patch("checks.obstacle.importance_filter.extract_rpy_in_flu") as mock_extract_rpy:
            # Mock yaw angle of π (180 degrees)
            mock_extract_rpy.return_value = (0.0, 0.0, np.pi)

            is_oncoming = self.filter._is_oncoming(self.sample_object, self.camera_pose, track_id=123)

        self.assertTrue(is_oncoming)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_is_oncoming_false(self, mock_get_pose):
        """Test oncoming detection when object is not oncoming."""
        mock_get_pose.return_value = np.eye(4)

        with patch("checks.obstacle.importance_filter.extract_rpy_in_flu") as mock_extract_rpy:
            # Mock yaw angle of 0 (0 degrees)
            mock_extract_rpy.return_value = (0.0, 0.0, 0.0)

            is_oncoming = self.filter._is_oncoming(self.sample_object, self.camera_pose, track_id=123)

        self.assertFalse(is_oncoming)

    @patch("checks.obstacle.importance_filter.get_object_to_camera_pose")
    def test_is_oncoming_tolerance(self, mock_get_pose):
        """Test oncoming detection tolerance."""
        mock_get_pose.return_value = np.eye(4)

        # Test near 180 degrees within tolerance (30 degrees)
        test_cases = [
            (np.deg2rad(150), True),  # 150 degrees - within tolerance
            (np.deg2rad(210), True),  # 210 degrees - within tolerance
            (np.deg2rad(120), False),  # 120 degrees - outside tolerance
            (np.deg2rad(240), False),  # 240 degrees - outside tolerance
        ]

        for yaw_angle, expected_oncoming in test_cases:
            with patch("checks.obstacle.importance_filter.extract_rpy_in_flu") as mock_extract_rpy:
                mock_extract_rpy.return_value = (0.0, 0.0, yaw_angle)

                is_oncoming = self.filter._is_oncoming(self.sample_object, self.camera_pose, track_id=123)

                self.assertEqual(
                    is_oncoming, expected_oncoming, f"Failed for yaw angle {np.rad2deg(yaw_angle)} degrees"
                )

    def test_get_object_yaw_in_ego_frame(self):
        """Test yaw angle calculation."""
        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            mock_get_pose.return_value = np.eye(4)

            with patch("checks.obstacle.importance_filter.extract_rpy_in_flu") as mock_extract_rpy:
                # Test various yaw angles
                test_cases = [
                    (0.0, 0.0),  # 0 radians -> 0 degrees
                    (np.pi / 2, 90.0),  # π/2 radians -> 90 degrees
                    (np.pi, 180.0),  # π radians -> 180 degrees
                    (-np.pi / 2, 270.0),  # -π/2 radians -> 270 degrees
                    (-np.pi, 180.0),  # -π radians -> 180 degrees (normalized)
                ]

                for yaw_radians, expected_degrees in test_cases:
                    mock_extract_rpy.return_value = (0.0, 0.0, yaw_radians)

                    yaw_degrees = self.filter._get_object_yaw_in_ego_frame(self.sample_object, self.camera_pose)

                    self.assertAlmostEqual(
                        yaw_degrees, expected_degrees, places=6, msg=f"Failed for {yaw_radians} radians"
                    )

    def test_is_in_relevant_lanes_empty_config(self):
        """Test lane filter when no relevant lanes are configured."""
        config_no_lanes = {"relevant_lanes": []}
        filter_no_lanes = ImportanceFilter(config_no_lanes, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            mock_get_pose.return_value = np.eye(4)

            is_relevant = filter_no_lanes._is_in_relevant_lanes(self.sample_object, self.camera_pose, track_id=123)

        # Should return True when no relevant lanes are specified
        self.assertTrue(is_relevant)

    def test_is_in_relevant_lanes_missing_config(self):
        """Test lane filter when relevant_lanes is missing from config."""
        config_missing = {}
        filter_missing = ImportanceFilter(config_missing, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            mock_get_pose.return_value = np.eye(4)

            is_relevant = filter_missing._is_in_relevant_lanes(self.sample_object, self.camera_pose, track_id=123)

        # Should return True when relevant_lanes is not in config
        self.assertTrue(is_relevant)

    def test_assign_object_to_lane(self):
        """Test lane assignment based on lateral position."""
        test_cases = [
            (0.0, "ego"),  # Center lane
            (1.0, "ego"),  # Still in ego lane
            (2.0, "right"),  # Right lane
            (3.0, "right"),  # Still in right lane
            (-2.0, "left"),  # Left lane
            (-3.0, "left"),  # Still in left lane
            (5.0, "unknown"),  # Too far right
            (-5.0, "unknown"),  # Too far left
        ]

        for lateral_pos, expected_lane in test_cases:
            # Create position matrix with lateral position
            position_matrix = np.array(
                [[1.0, 0.0, 0.0, lateral_pos], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )

            lane = self.filter._assign_object_to_lane(position_matrix)

            self.assertEqual(lane, expected_lane, f"Failed for lateral position {lateral_pos}")

    def test_is_within_distance_no_threshold(self):
        """Test distance filter when no threshold is configured."""
        config_no_threshold = {}
        filter_no_threshold = ImportanceFilter(config_no_threshold, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            # Mock very far object
            mock_get_pose.return_value = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1000.0],  # 1000m away
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            is_within = filter_no_threshold._is_within_distance(self.sample_object, self.camera_pose, track_id=123)

        # Should return True when no threshold is configured
        self.assertTrue(is_within)

    def test_is_within_distance_infinite_threshold(self):
        """Test distance filter with infinite threshold."""
        config_infinite = {"distance_threshold_m": float("inf")}
        filter_infinite = ImportanceFilter(config_infinite, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            mock_get_pose.return_value = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1000.0],  # Very far
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            is_within = filter_infinite._is_within_distance(self.sample_object, self.camera_pose, track_id=123)

        self.assertTrue(is_within)

    def test_is_within_distance_calculations(self):
        """Test distance calculations."""
        test_cases = [
            (0.0, 10.0, 10.0, True),  # 10m forward, within 50m threshold
            (3.0, 4.0, 5.0, True),  # 5m diagonal, within threshold
            (30.0, 40.0, 50.0, True),  # Exactly at threshold
            (30.0, 40.1, 50.0, False),  # Just beyond threshold
            (60.0, 0.0, 60.0, False),  # 60m lateral, beyond threshold
        ]

        for lateral, longitudinal, distance, should_be_within in test_cases:
            with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
                mock_get_pose.return_value = np.array(
                    [
                        [1.0, 0.0, 0.0, lateral],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, longitudinal],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )

                is_within = self.filter._is_within_distance(self.sample_object, self.camera_pose, track_id=123)

                self.assertEqual(
                    is_within,
                    should_be_within,
                    f"Failed for lateral={lateral}, longitudinal={longitudinal}, distance={distance}",
                )

    def test_logging_calls(self):
        """Test that logging is called appropriately."""
        # Test with distance filter failure
        config = {"distance_threshold_m": 5.0}
        filter_with_logging = ImportanceFilter(config, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            # Mock object that's too far
            mock_get_pose.return_value = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 20.0],  # 20m away, beyond 5m threshold
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            should_process, reason = filter_with_logging.should_process_object(
                self.sample_object, self.camera_pose, track_id=123
            )

        # Check that debug logging was called
        self.logger.debug.assert_called()
        self.assertFalse(should_process)

    def test_oncoming_filter_disabled(self):
        """Test behavior when oncoming filter is disabled."""
        config_no_oncoming = {"skip_oncoming_obstacles": False, "distance_threshold_m": 50.0, "relevant_lanes": ["ego"]}
        filter_no_oncoming = ImportanceFilter(config_no_oncoming, self.logger)

        with patch("checks.obstacle.importance_filter.get_object_to_camera_pose") as mock_get_pose:
            mock_get_pose.return_value = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],  # Ego lane
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 10.0],  # Close
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            with patch.object(filter_no_oncoming, "_get_object_yaw_in_ego_frame", return_value=180.0):  # Oncoming
                should_process, reason = filter_no_oncoming.should_process_object(
                    self.sample_object, self.camera_pose, track_id=123
                )

        # Should pass even though object is oncoming
        self.assertTrue(should_process)
        self.assertEqual(reason, "passed")

    def test_static_object_passes_through(self):
        """Test that static objects are appropriately handled and pass through."""
        # Default config allows static objects
        should_process, reason = self.filter.should_process_object(self.static_object, self.camera_pose, track_id=456)

        # Should pass through with allow_all_static_objects reason
        self.assertTrue(should_process)
        self.assertEqual(reason, "allow_all_static_objects")
        # Verify logging was called
        self.logger.debug.assert_called()

    def test_no_centroid_object_filtered(self):
        """Test that objects without a meaningful centroid are appropriately handled."""
        should_process, reason = self.filter.should_process_object(
            self.no_centroid_object, self.camera_pose, track_id=789
        )

        # Should be filtered with no_centroid reason
        self.assertFalse(should_process)
        self.assertEqual(reason, "no_centroid")
        # Verify logging was called
        self.logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
