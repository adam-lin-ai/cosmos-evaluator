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

"""Unit tests for hallucination_tracking module."""

import unittest

import numpy as np

from checks.obstacle.hallucination_tracking import track_hallucinations


class TestHallucinationTracking(unittest.TestCase):
    """Test cases for hallucination tracking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset global counter for consistent testing
        import checks.obstacle.hallucination_tracking

        checks.obstacle.hallucination_tracking._TRACK_ID_COUNTER = 0

        # Sample results structure
        self.base_results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 0, "bbox_xywh": [200, 200, 60, 40], "mask_ratio": 0.9},
                ],
                "pedestrian": [{"frame_idx": 0, "bbox_xywh": [300, 300, 30, 60], "mask_ratio": 0.7}],
            },
            "hallucination_tracks": [],
        }

    def test_track_hallucinations_empty_start_indices(self):
        """Test track_hallucinations with empty start_indices."""
        results = self.base_results.copy()
        track_hallucinations(results, {})

        # Should not modify results when start_indices is empty
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_no_existing_tracks(self):
        """Test track_hallucinations with no existing tracks."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0, "pedestrian": 0}

        track_hallucinations(results, start_indices)

        # Should create new tracks for all detections
        tracks = results["hallucination_tracks"]
        self.assertEqual(len(tracks), 3)  # 2 vehicles + 1 pedestrian

        # Check first vehicle track
        vehicle_tracks = [t for t in tracks if t["class"] == "vehicle"]
        self.assertEqual(len(vehicle_tracks), 2)

        # Check pedestrian track
        pedestrian_tracks = [t for t in tracks if t["class"] == "pedestrian"]
        self.assertEqual(len(pedestrian_tracks), 1)

    def test_track_hallucinations_with_existing_tracks(self):
        """Test track_hallucinations with existing tracks."""
        results = self.base_results.copy()

        # Add existing tracks
        results["hallucination_tracks"] = [
            {
                "id": 1,
                "class": "vehicle",
                "detections": [0],  # First vehicle detection
                "relevancy": 0,
            }
        ]

        # Add new detections for current frame
        results["hallucination_detections"]["vehicle"].extend(
            [
                {"frame_idx": 1, "bbox_xywh": [105, 105, 55, 55], "mask_ratio": 0.8},  # Overlaps with first detection
                {"frame_idx": 1, "bbox_xywh": [400, 400, 50, 50], "mask_ratio": 0.85},  # New detection
            ]
        )

        start_indices = {"vehicle": 2}  # New detections start from index 2

        track_hallucinations(results, start_indices)

        tracks = results["hallucination_tracks"]

        # Should have 2 tracks total: extended existing track + new track
        self.assertEqual(len(tracks), 2)

        # First track should be extended
        track1 = tracks[0]
        self.assertEqual(track1["id"], 1)
        self.assertEqual(len(track1["detections"]), 2)  # Original + new overlapping detection
        self.assertIn(2, track1["detections"])  # Should include new overlapping detection

        # Second track should be new
        track2 = tracks[1]
        self.assertEqual(track2["id"], 2)
        self.assertEqual(len(track2["detections"]), 1)
        self.assertIn(3, track2["detections"])  # Should include new non-overlapping detection

    def test_track_hallucinations_overlap_detection(self):
        """Test overlap detection between bounding boxes."""
        # Start with detection from frame 0
        results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}  # Detection 0
                ]
            },
            "hallucination_tracks": [],
        }

        # Frame 0: Process initial detection
        track_hallucinations(results, {"vehicle": 0})

        # Frame 1: Add new detections and process them
        results["hallucination_detections"]["vehicle"].extend(
            [
                {
                    "frame_idx": 1,
                    "bbox_xywh": [110, 110, 50, 50],
                    "mask_ratio": 0.8,
                },  # Detection 1 - overlaps with 0
                {
                    "frame_idx": 1,
                    "bbox_xywh": [200, 200, 50, 50],
                    "mask_ratio": 0.8,
                },  # Detection 2 - non-overlapping
            ]
        )
        track_hallucinations(results, {"vehicle": 1})

        tracks = results["hallucination_tracks"]

        # Should have 2 tracks: one extended, one new
        self.assertEqual(len(tracks), 2)

        # Find tracks by detection count
        extended_track = next(t for t in tracks if len(t["detections"]) == 2)
        new_track = next(t for t in tracks if len(t["detections"]) == 1 and t["id"] != extended_track["id"])

        # Extended track should contain overlapping detections
        self.assertIn(0, extended_track["detections"])
        self.assertIn(1, extended_track["detections"])

        # New track should contain non-overlapping detection
        self.assertIn(2, new_track["detections"])

    def test_track_hallucinations_no_overlap(self):
        """Test track_hallucinations when no overlap is detected."""
        # Start with detection from frame 0
        results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}  # Detection 0
                ]
            },
            "hallucination_tracks": [],
        }

        # Frame 0: Process initial detection
        track_hallucinations(results, {"vehicle": 0})

        # Frame 1: Add new non-overlapping detection and process it
        results["hallucination_detections"]["vehicle"].append(
            {
                "frame_idx": 1,
                "bbox_xywh": [300, 300, 50, 50],
                "mask_ratio": 0.8,
            }  # Detection 1 - far away, no overlap
        )
        track_hallucinations(results, {"vehicle": 1})

        tracks = results["hallucination_tracks"]

        # Should have 2 separate tracks
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0]["detections"]), 1)
        self.assertEqual(len(tracks[1]["detections"]), 1)

    def test_track_hallucinations_relevancy_update_with_road(self):
        """Test relevancy update when detection overlaps with road pixels."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Create road mask with road pixels at detection bbox location (100-150, 100-150)
        road_mask = np.zeros((200, 200), dtype=bool)
        road_mask[100:150, 100:150] = True

        track_hallucinations(results, start_indices, road_mask=road_mask)

        tracks = results["hallucination_tracks"]
        vehicle_track = next(t for t in tracks if t["class"] == "vehicle")

        # Should have increased relevancy due to road overlap
        self.assertGreater(vehicle_track["relevancy"], 0)

    def test_track_hallucinations_relevancy_update_no_road(self):
        """Test relevancy update when detection doesn't overlap with road pixels."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Create road mask with no road pixels at detection locations
        road_mask = np.zeros((200, 200), dtype=bool)

        track_hallucinations(results, start_indices, road_mask=road_mask)

        tracks = results["hallucination_tracks"]
        vehicle_track = next(t for t in tracks if t["class"] == "vehicle")

        # Should have zero relevancy due to no road overlap
        self.assertEqual(vehicle_track["relevancy"], 0)

    def test_track_hallucinations_missing_bbox(self):
        """Test track_hallucinations with detection missing bbox_xywh."""
        results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "mask_ratio": 0.8}  # Missing bbox_xywh
                ]
            },
            "hallucination_tracks": [],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should not create any tracks due to missing bbox
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_invalid_detection_bucket(self):
        """Test track_hallucinations with invalid detection bucket."""
        results = {
            "hallucination_detections": {
                "vehicle": "not_a_list"  # Invalid type
            },
            "hallucination_tracks": [],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should handle gracefully and not create tracks
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_out_of_bounds_start_index(self):
        """Test track_hallucinations with start_index beyond detection list length."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 10}  # Beyond list length

        track_hallucinations(results, start_indices)

        # Should not process any detections or create tracks
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_missing_tracks_key(self):
        """Test track_hallucinations when results doesn't have hallucination_tracks key."""
        results = {
            "hallucination_detections": {
                "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}]
            }
            # Missing hallucination_tracks key
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should create tracks key and add tracks
        self.assertIn("hallucination_tracks", results)
        self.assertEqual(len(results["hallucination_tracks"]), 1)

    def test_track_hallucinations_track_id_counter_initialization(self):
        """Test that track ID counter is properly initialized from existing tracks."""
        results = {
            "hallucination_detections": {
                "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}]
            },
            "hallucination_tracks": [{"id": 5, "class": "vehicle", "detections": [], "relevancy": 0}],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # New track should have ID > 5
        new_track = next(t for t in results["hallucination_tracks"] if t["id"] > 5)
        self.assertEqual(new_track["id"], 6)

    def test_track_hallucinations_multiple_previous_detections(self):
        """Test tracking when comparing against multiple previous detections."""
        results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 1, "bbox_xywh": [110, 110, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 2, "bbox_xywh": [105, 105, 50, 50], "mask_ratio": 0.8},  # Overlaps with both previous
                ]
            },
            "hallucination_tracks": [
                {
                    "id": 1,
                    "class": "vehicle",
                    "detections": [0, 1],  # Track with 2 previous detections
                    "relevancy": 0,
                }
            ],
        }

        start_indices = {"vehicle": 2}  # Process only the new detection

        track_hallucinations(results, start_indices)

        tracks = results["hallucination_tracks"]

        # Should still have only 1 track, extended with new detection
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]["detections"]), 3)
        self.assertIn(2, tracks[0]["detections"])

    def test_track_hallucinations_with_none_road_mask(self):
        """Test track_hallucinations when road_mask is None."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Pass None for road_mask (should skip relevancy updates)
        track_hallucinations(results, start_indices, road_mask=None)

        # Should still create tracks
        self.assertGreater(len(results["hallucination_tracks"]), 0)

        # Relevancy should remain 0 since no road mask was provided
        vehicle_track = next(t for t in results["hallucination_tracks"] if t["class"] == "vehicle")
        self.assertEqual(vehicle_track["relevancy"], 0)


if __name__ == "__main__":
    unittest.main()
