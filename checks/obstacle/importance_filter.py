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
Importance filter module for obstacle correspondence processing.

This module implements filtering logic to determine which objects are important
enough to process based on configuration rules.
"""

from typing import Any, Dict

import numpy as np

from checks.utils.coord_transforms import extract_rpy_in_flu, get_object_to_camera_pose


class ImportanceFilter:
    """
    Filters objects based on importance criteria defined in configuration.

    This class applies various filters to determine if an object should be
    processed for correspondence analysis.
    """

    def __init__(self, filter_config: Dict[str, Any], logger=None):
        """
        Initialize the importance filter.

        Args:
            filter_config: Configuration dictionary for importance filtering
            logger: Logger instance to use (optional)
        """
        self.config = filter_config
        self.logger = logger

    def should_process_object(
        self, tracked_object: Dict[str, Any], camera_pose: np.ndarray, track_id: int
    ) -> tuple[bool, str]:
        """
        Determine if an object should be processed based on importance filters.

        Args:
            tracked_object: Object data from world model
            camera_pose: Camera pose for this frame
            track_id: Track ID of the object (used for logging)

        Returns:
            Tuple of (should_process, reason). If should_process is False, reason contains the filter reason.
        """

        # pass static objects through if configured to do so
        if tracked_object.get("is_static", True) and self.config.get("allow_all_static_objects", True):
            if self.logger:
                self.logger.debug(f"Importance filter: Track {track_id} passed through since it is a static object")
            return True, "allow_all_static_objects"

        # only process objects with a meaningful centroid
        if not hasattr(tracked_object["geometry"], "object_to_world_pose"):
            if self.logger:
                self.logger.debug(
                    f"Importance filter: Track {track_id} passed through since it has no meaningful centroid"
                )
            return False, "no_centroid"

        # the object is not within distance threshold
        if not self._is_within_distance(tracked_object, camera_pose, track_id):
            distance_threshold = self.config.get("distance_threshold_m", float("inf"))
            reason = f"distance_threshold_m ({distance_threshold}m)"
            if self.logger:
                self.logger.debug(f"Importance filter: Track {track_id} filtered by {reason}")
            return False, reason

        # skip oncoming obstacles if configured to do so AND the object is oncoming
        if self.config.get("skip_oncoming_obstacles", True) and self._is_oncoming(
            tracked_object, camera_pose, track_id
        ):
            reason = "skip_oncoming_obstacles (enabled)"
            if self.logger:
                self.logger.debug(f"Importance filter: Track {track_id} filtered by {reason}")
            return False, reason

        # the object is not in relevant lanes
        if not self._is_in_relevant_lanes(tracked_object, camera_pose, track_id):
            relevant_lanes = self.config.get("relevant_lanes", [])
            reason = f"relevant_lanes ({relevant_lanes})"
            if self.logger:
                self.logger.debug(f"Importance filter: Track {track_id} filtered by {reason}")
            return False, reason

        # object passes the importance filter
        if self.logger:
            self.logger.debug(f"Importance filter: Track {track_id} passed all filters")
        return True, "passed"

    def _is_oncoming(self, tracked_object: Dict[str, Any], camera_pose: np.ndarray, track_id: int) -> bool:
        """
        Check if object is oncoming.

        Args:
            tracked_object: Object data from world model
            camera_pose: Camera pose for this frame
            track_id: Track ID for logging

        Returns:
            True if object is oncoming, False otherwise
        """

        # Get object's yaw angle relative to ego
        yaw_degrees = self._get_object_yaw_in_ego_frame(tracked_object, camera_pose)

        # Check if yaw is close to 180 degrees (oncoming)
        # Allow some tolerance (e.g., ±30 degrees)
        tolerance = 30
        is_oncoming = abs(yaw_degrees - 180) <= tolerance

        if self.logger:
            self.logger.debug(
                f"Oncoming check: Track {track_id}, yaw={yaw_degrees:.1f}°, tolerance=±{tolerance}°, is_oncoming={is_oncoming}"
            )

        return is_oncoming

    def _get_object_yaw_in_ego_frame(self, tracked_object: Dict[str, Any], camera_pose: np.ndarray) -> float:
        """
        Get object's yaw angle relative to ego vehicle.

        Args:
            tracked_object: Object data from world model
            camera_pose: Camera pose for this frame

        Returns:
            Yaw angle in degrees [0, 360)
        """
        object_to_camera = get_object_to_camera_pose(tracked_object["geometry"], camera_pose)

        # Extract euler angles in FLU coordinate
        _, _, yaw = extract_rpy_in_flu(object_to_camera[:3, :3])

        # Normalize to [0, 360) range
        yaw_degrees = (yaw * 180 / np.pi + 360) % 360

        return yaw_degrees

    def _is_in_relevant_lanes(self, tracked_object: Dict[str, Any], camera_pose: np.ndarray, track_id: int) -> bool:
        """
        Check if object is in one of the relevant lanes.

        Args:
            tracked_object: Object data from world model
            camera_pose: Camera pose for this frame
            track_id: Track ID for logging

        Returns:
            True if object is in a relevant lane
        """
        relevant_lanes = self.config.get("relevant_lanes", [])
        if not relevant_lanes:
            return True

        # Get object pose in ego coordinates
        object_to_camera = get_object_to_camera_pose(tracked_object["geometry"], camera_pose)

        # Determine lane based on object position in camera frame
        object_type = str(tracked_object.get("object_type", "unknown"))
        lane = self._assign_object_to_lane(object_to_camera, object_type)

        # Check if lane is in relevant lanes
        is_in_relevant = lane in relevant_lanes

        if self.logger:
            self.logger.debug(
                f"Lane check: Track {track_id}, position={object_to_camera[0, 3]:.1f}m, lane={lane}, relevant_lanes={relevant_lanes}, in_relevant={is_in_relevant}"
            )

        return is_in_relevant

    def _assign_object_to_lane(
        self, position_in_camera_frame: np.ndarray[tuple[4, 4], np.float32], object_type: str = "unknown"
    ) -> str:
        """
        Determine lane assignment based on lateral position (x) in ego coordinates.

        Args:
            position_in_camera_frame: Position in camera coordinates (meters)
            object_type: World-model object type string (e.g., "Car", "Pedestrian")

        Returns:
            Lane assignment: "ego", "left", or "right"
        """
        LANE_WIDTH_M = 3.0
        HALF_LANE_WIDTH_M = LANE_WIDTH_M / 2.0
        MARGIN_BAND_M = 0.0
        lateral_position = float(position_in_camera_frame[0, 3])

        # Use tighter lateral bands for pedestrians/riders to avoid classifying sidewalk as lanes
        obj_type_norm = object_type.lower()
        if obj_type_norm in {"pedestrian", "person"}:
            MARGIN_BAND_M = 0.5

        # Default rule for vehicles and others: lane-width based bands
        if abs(lateral_position) < HALF_LANE_WIDTH_M:
            return "ego"
        elif (
            lateral_position > HALF_LANE_WIDTH_M and lateral_position < LANE_WIDTH_M + HALF_LANE_WIDTH_M + MARGIN_BAND_M
        ):
            return "right"
        elif (
            lateral_position < -HALF_LANE_WIDTH_M
            and lateral_position > -LANE_WIDTH_M - HALF_LANE_WIDTH_M - MARGIN_BAND_M
        ):
            return "left"
        else:
            return "unknown"

    def _is_within_distance(self, tracked_object: Dict[str, Any], camera_pose: np.ndarray, track_id: int) -> bool:
        """
        Check if object is within the distance threshold.

        Args:
            tracked_object: Object data from world model
            camera_pose: Camera pose for this frame
            track_id: Track ID for logging

        Returns:
            True if object is within distance threshold
        """
        # If no distance threshold specified, allow all objects
        distance_threshold = self.config.get("distance_threshold_m", float("inf"))
        if distance_threshold == float("inf"):
            return True

        # Get object pose in ego coordinates
        object_to_camera = get_object_to_camera_pose(tracked_object["geometry"], camera_pose)

        # Extract position (x, y, z) in ego coordinates
        lateral_position = object_to_camera[0, 3]  # x in camera coordinate
        longitudinal_position = object_to_camera[2, 3]  # z in camera coordinate

        # Calculate 2D distance from ego vehicle
        distance = np.sqrt(lateral_position**2 + longitudinal_position**2)

        # Check if within threshold
        is_within = distance <= distance_threshold

        if self.logger:
            self.logger.debug(
                f"Distance check: Track {track_id}, \
            object position in camera=({object_to_camera[0, 3]:.1f}, {object_to_camera[1, 3]:.1f}, {object_to_camera[2, 3]:.1f}), \
            distance={distance:.1f}m, threshold={distance_threshold}m, within={is_within}"
            )

        return is_within
