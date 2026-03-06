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

"""Unit tests for hallucination_detector module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.obstacle.hallucination_detector import HallucinationDetector


class TestHallucinationDetector(unittest.TestCase):
    """Test cases for HallucinationDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.config = {
            "enabled": True,
            "classes": {"vehicle": {"min_cluster_area": 100}, "pedestrian": {"min_cluster_area": 50}},
            "max_cluster_per_frame": 10,
        }

        # Mock seg_helper
        self.mock_seg_helper = MagicMock()

        # Mock logger
        self.mock_logger = MagicMock()

        # Create detector instance
        self.detector = HallucinationDetector(
            config=self.config, seg_helper=self.mock_seg_helper, logger=self.mock_logger
        )

    def test_init_valid_config(self):
        """Test HallucinationDetector initialization with valid config."""
        self.assertTrue(self.detector.enabled)
        self.assertEqual(self.detector.classes_cfg, self.config["classes"])
        self.assertEqual(self.detector.max_clusters_per_frame, 10)

    def test_init_missing_enabled_key(self):
        """Test HallucinationDetector initialization with missing enabled key."""
        config = {"classes": {}, "max_cluster_per_frame": 5}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("enabled", str(context.exception))

    def test_init_missing_classes_key(self):
        """Test HallucinationDetector initialization with missing classes key."""
        config = {"enabled": True, "max_cluster_per_frame": 5}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("classes", str(context.exception))

    def test_init_missing_max_cluster_key(self):
        """Test HallucinationDetector initialization with missing max_cluster_per_frame key."""
        config = {"enabled": True, "classes": {}}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("max_cluster_per_frame", str(context.exception))

    def test_init_invalid_classes_type(self):
        """Test HallucinationDetector initialization with invalid classes type."""
        config = {"enabled": True, "classes": "invalid", "max_cluster_per_frame": 5}
        with self.assertRaises(ValueError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("must be a dict", str(context.exception))

    def test_detect_disabled(self):
        """Test detect method when detector is disabled."""
        self.detector.enabled = False
        result = self.detector.detect(
            frame_idx=0,
            resized_masks=None,
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        expected = {"vehicle": [], "pedestrian": []}
        self.assertEqual(result, expected)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_no_masks(self, mock_cv2_stats):
        """Test detect method with no segmentation masks."""
        self.mock_seg_helper.get_class_mask.return_value = None

        result = self.detector.detect(
            frame_idx=0,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        expected = {"vehicle": [], "pedestrian": []}
        self.assertEqual(result, expected)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_with_hallucinations(self, mock_cv2_stats):
        """Test detect method with hallucination detections."""
        # Setup mock segmentation masks
        vehicle_mask = np.zeros((1080, 1920), dtype=bool)
        vehicle_mask[100:200, 100:300] = True  # Create a vehicle mask region

        self.mock_seg_helper.get_class_mask.side_effect = lambda _masks, class_name: {
            "vehicle": vehicle_mask,
            "pedestrian": None,
        }.get(class_name)

        # Mock connected components for clustering
        # Simulate one connected component for the unmatched region
        mock_cv2_stats.return_value = (
            2,  # num components (including background)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # labels (mock)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # stats
            np.array([[0, 0], [200, 150]]),  # centroids
        )

        result = self.detector.detect(
            frame_idx=5,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        # Should detect one hallucination for vehicle class
        self.assertEqual(len(result["vehicle"]), 1)
        self.assertEqual(len(result["pedestrian"]), 0)

        detection = result["vehicle"][0]
        self.assertEqual(detection["frame_idx"], 5)
        self.assertEqual(detection["bbox_xywh"], [100, 100, 200, 100])
        self.assertAlmostEqual(detection["mask_ratio"], 1.0, places=2)

    def test_cluster_components_empty_mask(self):
        """Test _cluster_components with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        result = self.detector._cluster_components(empty_mask, min_area=50)
        self.assertEqual(result, [])

    def test_cluster_components_none_mask(self):
        """Test _cluster_components with None mask."""
        result = self.detector._cluster_components(None, min_area=50)
        self.assertEqual(result, [])

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_cluster_components_with_components(self, mock_cv2_stats):
        """Test _cluster_components with valid components."""
        mask = np.ones((100, 100), dtype=bool)

        # Mock connected components output
        mock_cv2_stats.return_value = (
            3,  # num components (including background)
            np.zeros((100, 100)),  # labels (not used in this test)
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background component
                    [10, 10, 20, 20, 400],  # first component (area 400)
                    [50, 50, 15, 15, 225],  # second component (area 225)
                ]
            ),
            np.array([[0, 0], [20, 20], [57.5, 57.5]]),  # centroids
        )

        result = self.detector._cluster_components(mask, min_area=200)

        # Should return both components (both above min_area=200)
        self.assertEqual(len(result), 2)

        # Should be sorted by area descending
        self.assertEqual(result[0], [10, 10, 20, 20, 400, 20.0, 20.0])
        self.assertEqual(result[1], [50, 50, 15, 15, 225, 57.5, 57.5])

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_cluster_components_min_area_filter(self, mock_cv2_stats):
        """Test _cluster_components with min_area filtering."""
        mask = np.ones((100, 100), dtype=bool)

        # Mock connected components output
        mock_cv2_stats.return_value = (
            3,  # num components
            np.zeros((100, 100)),  # labels
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [10, 10, 20, 20, 100],  # small component (area 100)
                    [50, 50, 15, 15, 500],  # large component (area 500)
                ]
            ),
            np.array([[0, 0], [20, 20], [57.5, 57.5]]),
        )

        result = self.detector._cluster_components(mask, min_area=200)

        # Should only return large component
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [50, 50, 15, 15, 500, 57.5, 57.5])

    def test_get_min_cluster_area_for_class_configured(self):
        """Test _get_min_cluster_area_for_class with configured class."""
        area = self.detector._get_min_cluster_area_for_class("vehicle")
        self.assertEqual(area, 100)

        area = self.detector._get_min_cluster_area_for_class("pedestrian")
        self.assertEqual(area, 50)

    def test_get_min_cluster_area_for_class_unconfigured(self):
        """Test _get_min_cluster_area_for_class with unconfigured class."""
        area = self.detector._get_min_cluster_area_for_class("unknown_class")
        self.assertEqual(area, 100)  # default value

    def test_get_min_cluster_area_for_class_invalid_config(self):
        """Test _get_min_cluster_area_for_class with invalid config."""
        # Test with invalid min_cluster_area value
        self.detector.classes_cfg["vehicle"] = {"min_cluster_area": "invalid"}
        area = self.detector._get_min_cluster_area_for_class("vehicle")
        self.assertEqual(area, 100)  # should fallback to default

    def test_is_in_ego_body_region_inside(self):
        """Test _is_in_ego_body_region for bbox inside ego body region."""
        # Bbox in bottom 25% of image
        image_width, image_height = 1920, 1080
        band_y0 = image_height - int(0.25 * image_height)  # 810 for 1080p

        # Small bbox fully inside ego body region
        result = self.detector._is_in_ego_body_region(
            x=960, y=band_y0 + 10, w=50, h=50, image_width=image_width, image_height=image_height
        )
        self.assertTrue(result)

    def test_is_in_ego_body_region_outside(self):
        """Test _is_in_ego_body_region for bbox outside ego body region."""
        # Bbox in top part of image, above the bottom 25% band
        result = self.detector._is_in_ego_body_region(x=960, y=100, w=50, h=50, image_width=1920, image_height=1080)
        self.assertFalse(result)

    def test_is_in_ego_body_region_partial_overlap(self):
        """Test _is_in_ego_body_region for bbox that partially overlaps the bottom band."""
        image_width, image_height = 1920, 1080
        band_y0 = image_height - int(0.25 * image_height)  # 810 for 1080p

        # Bbox spans from above band into band (e.g. y=780, h=100 -> y1=879, intersects [810,1079])
        result = self.detector._is_in_ego_body_region(
            x=960, y=band_y0 - 30, w=100, h=100, image_width=image_width, image_height=image_height
        )
        self.assertTrue(result)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_with_scene_rasterizer(self, mock_cv2_stats):
        """Test detect method with pre-computed SceneRasterizer."""
        # Setup mock segmentation masks
        vehicle_mask = np.zeros((1080, 1920), dtype=bool)
        vehicle_mask[100:200, 100:300] = True  # Create a vehicle mask region

        self.mock_seg_helper.get_class_mask.side_effect = lambda masks, class_name: {
            "vehicle": vehicle_mask,
            "pedestrian": None,
        }.get(class_name)

        # Mock connected components for clustering
        mock_cv2_stats.return_value = (
            2,  # num components (including background)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # labels (mock)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # stats
            np.array([[0, 0], [200, 150]]),  # centroids
        )

        # Create a mock SceneRasterizer
        mock_rasterizer = MagicMock()
        mock_rasterizer.has_object.return_value = False  # No objects in rasterizer

        result = self.detector.detect(
            frame_idx=5,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            scene_rasterizer=mock_rasterizer,
        )

        # Should still detect hallucinations (scene_rasterizer has no objects to provide masks for)
        self.assertEqual(len(result["vehicle"]), 1)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_no_masks(self, mock_cv2_stats):
        """Test _build_gt_mask with no segmentation masks."""
        result = self.detector._build_gt_mask(
            resized_masks=None,
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        expected = np.zeros((1080, 1920), dtype=bool)
        np.testing.assert_array_equal(result, expected)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_no_objects(self, mock_cv2_stats):
        """Test _build_gt_mask with no frame objects."""
        # Mock class mask
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components with proper return values
        mock_cv2_stats.return_value = (
            1,  # num components (just background)
            np.zeros((1080, 1920), dtype=np.int32),  # labels array with correct shape
            np.array([[0, 0, 0, 0, 0]]),  # stats for background only
            np.array([[0, 0]]),  # centroids for background only
        )

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        expected = np.zeros((1080, 1920), dtype=bool)
        np.testing.assert_array_equal(result, expected)

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_vehicle_class(self, mock_cv2_stats):
        """Test _build_gt_mask for vehicle class."""
        # Setup mock class mask and objects
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components
        labels = np.zeros((1080, 1920), dtype=np.int32)
        labels[100:300, 100:300] = 1  # component 1 region
        labels[500:700, 500:800] = 2  # component 2 region

        mock_cv2_stats.return_value = (
            3,  # num components
            labels,  # labels array with correct 2D shape
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [100, 100, 200, 100, 20000],  # component 1
                    [500, 500, 300, 200, 60000],  # component 2
                ]
            ),
            np.array([[0, 0], [200, 150], [650, 600]]),
        )

        # Mock cuboid projection
        mock_cuboid = MagicMock()
        mock_projection = np.zeros((1080, 1920), dtype=bool)
        mock_projection[100:200, 100:300] = True  # Intersects with component 1
        mock_depth_mask = np.zeros((1080, 1920), dtype=np.float32)
        mock_depth_mask[100:200, 100:300] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projection, mock_depth_mask

        # Mock frame objects
        frame_objects = {"track_1": {"object_type": "Car", "geometry": mock_cuboid}}

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects=frame_objects,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        # Should have GT mask where component 1 is located
        self.assertTrue(result.any())
        mock_cuboid.get_projected_mask.assert_called_once()

    @patch("checks.obstacle.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_non_vehicle_class(self, mock_cv2_stats):
        """Test _build_gt_mask for non-vehicle class (pedestrian)."""
        # Setup mock class mask and objects
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components
        labels = np.zeros((1080, 1920), dtype=np.int32)
        labels[100:150, 100:150] = 1  # component 1 region
        labels[200:240, 200:230] = 2  # component 2 region

        mock_cv2_stats.return_value = (
            3,  # num components
            labels,  # labels array with correct 2D shape
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [100, 100, 50, 50, 2500],  # component 1
                    [200, 200, 30, 40, 1200],  # component 2
                ]
            ),
            np.array([[0, 0], [125, 125], [215, 220]]),
        )

        # Mock cuboid projection that intersects both components
        mock_cuboid = MagicMock()
        mock_projection = np.zeros((1080, 1920), dtype=bool)
        mock_projection[100:250, 100:250] = True  # Intersects with both components
        mock_depth_mask = np.zeros((1080, 1920), dtype=np.float32)
        mock_depth_mask[100:250, 100:250] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projection, mock_depth_mask

        # Mock frame objects
        frame_objects = {"track_1": {"object_type": "Pedestrian", "geometry": mock_cuboid}}

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects=frame_objects,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="pedestrian",
        )

        # Should have GT mask as expanded bounding box around intersecting components
        self.assertTrue(result.any())
        mock_cuboid.get_projected_mask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
