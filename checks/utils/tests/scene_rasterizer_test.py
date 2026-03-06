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

"""Unit tests for scene_rasterizer module."""

import logging
import unittest
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_array_equal

from checks.utils.rds_geometry import RDSGeometry
from checks.utils.scene_rasterizer import SceneRasterizer


class MockGeometry(RDSGeometry):
    """Mock geometry for testing with configurable mask and depth."""

    def __init__(self, mask: np.ndarray, depth_mask: np.ndarray):
        """Initialize mock geometry with predefined masks."""
        self._mask = mask
        self._depth_mask = depth_mask

    def get_projected_mask(self, _camera_to_world_pose, _camera_model, _image_width, _image_height):
        return self._mask.copy(), self._depth_mask.copy()


class TestSceneRasterizer(unittest.TestCase):
    """Test cases for SceneRasterizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_width = 100
        self.image_height = 80
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _create_rasterizer(self, objects, depth_tolerance=0.05, min_projected_size=16):
        """Helper to create a SceneRasterizer with standard parameters."""
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            depth_tolerance=depth_tolerance,
            min_projected_size=min_projected_size,
            logger=self.logger,
        )

    def test_init_empty_objects(self):
        """Test initialization with empty objects dict."""
        rasterizer = self._create_rasterizer({})
        self.assertEqual(len(rasterizer.objects), 0)

    def test_init_with_objects(self):
        """Test initialization with objects."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {
            "obj1": {"geometry": MockGeometry(mask, depth)},
            "obj2": {"geometry": MockGeometry(mask, depth)},
        }
        rasterizer = self._create_rasterizer(objects)
        self.assertEqual(len(rasterizer.objects), 2)
        self.assertTrue(rasterizer.has_object("obj1"))
        self.assertTrue(rasterizer.has_object("obj2"))

    def test_image_dimensions_properties(self):
        """Test that image dimensions are accessible via properties."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)
        self.assertEqual(rasterizer.image_width, self.image_width)
        self.assertEqual(rasterizer.image_height, self.image_height)

    def test_single_object_no_occlusion(self):
        """Test projection of a single object (no occlusion possible)."""
        # Create a mask with some pixels set (24x24, above min_projected_size)
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:34, 10:34] = True

        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:34, 10:34] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        # Visibility should equal projected mask (no occlusion)
        vis_mask = rasterizer.get_visibility_mask("obj1")
        assert_array_equal(vis_mask, mask)

        # Check visibility ratio
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

    def test_two_objects_no_overlap(self):
        """Test two non-overlapping objects."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[10:34, 10:34] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[10:34, 10:34] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[40:64, 50:80] = True  # Non-overlapping region
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[40:64, 50:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Both should be fully visible
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), mask2)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 1.0)

    def test_two_objects_full_occlusion(self):
        """Test two overlapping objects where one fully occludes the other."""
        # Both objects cover the same region
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0  # Closer

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True  # Same region
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 10.0  # Further away

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # obj1 should be fully visible (closer)
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

        # obj2 should be fully occluded (further away)
        expected_obj2_vis = np.zeros((self.image_height, self.image_width), dtype=bool)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), expected_obj2_vis)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_two_objects_partial_occlusion(self):
        """Test two overlapping objects with partial occlusion."""
        # obj1 covers region [20:44, 20:60]
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        # obj2 covers region [30:54, 40:80] - partial overlap at [30:44, 40:60]
        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[30:54, 40:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[30:54, 40:80] = 10.0  # Further away

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # obj1 should be fully visible
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

        # obj2 should be partially visible (overlap region occluded)
        vis2 = rasterizer.get_visibility_mask("obj2")
        # The overlap region [30:44, 40:60] should be occluded
        overlap_region = mask1 & mask2
        expected_vis2 = mask2 & ~overlap_region
        assert_array_equal(vis2, expected_vis2)

        # Visibility ratio should be < 1.0
        self.assertLess(rasterizer.get_visibility_ratio("obj2"), 1.0)
        self.assertGreater(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_depth_tolerance(self):
        """Test that depth tolerance allows nearly-equal depths to be visible."""
        # Two objects at nearly the same depth
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 5.02  # Within default tolerance of 0.05

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects, depth_tolerance=0.05)

        # Both should be visible due to tolerance
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), mask2)

    def test_depth_tolerance_exceeded(self):
        """Test that objects beyond depth tolerance are occluded."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 5.1  # Beyond tolerance of 0.05

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects, depth_tolerance=0.05)

        # obj1 visible, obj2 occluded
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_object_without_geometry(self):
        """Test handling of objects without geometry key."""
        objects = {
            "obj_no_geom": {"other_key": "value"},  # No geometry key
        }
        rasterizer = self._create_rasterizer(objects)

        # Should create empty masks for objects without geometry
        vis_mask = rasterizer.get_visibility_mask("obj_no_geom")
        self.assertEqual(vis_mask.shape, (self.image_height, self.image_width))
        self.assertFalse(np.any(vis_mask))

    def test_get_all_visibility_masks(self):
        """Test getting all visibility masks at once."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[10:34, 10:34] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[10:34, 10:34] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[40:64, 50:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[40:64, 50:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        all_masks = rasterizer.get_all_visibility_masks()
        self.assertIn("obj1", all_masks)
        self.assertIn("obj2", all_masks)
        assert_array_equal(all_masks["obj1"], mask1)
        assert_array_equal(all_masks["obj2"], mask2)

    def test_get_scene_depth_buffer(self):
        """Test getting the composited scene depth buffer."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:40, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:40, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[30:50, 40:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[30:50, 40:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        depth_buffer = rasterizer.get_scene_depth_buffer()

        # Check that minimum depth is used at each pixel
        # obj1-only region [20:30, 20:40] should have depth 5.0
        self.assertTrue(np.all(depth_buffer[20:30, 20:40] == 5.0))

        # obj2-only region [40:50, 60:80] should have depth 10.0
        self.assertTrue(np.all(depth_buffer[40:50, 60:80] == 10.0))

        # Overlap region [30:40, 40:60] should have depth 5.0 (minimum)
        self.assertTrue(np.all(depth_buffer[30:40, 40:60] == 5.0))

        # Uncovered region should be inf
        self.assertTrue(np.isinf(depth_buffer[0, 0]))

    def test_get_visible_pixel_count(self):
        """Test getting visible pixel count."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:34, 10:34] = True  # 24x24 = 576 pixels
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:34, 10:34] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        self.assertEqual(rasterizer.get_visible_pixel_count("obj1"), 576)

    def test_invalid_object_id(self):
        """Test error handling for invalid object IDs."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        with self.assertRaises(KeyError):
            rasterizer.get_visibility_mask("nonexistent")

        with self.assertRaises(KeyError):
            rasterizer.get_projected_mask("nonexistent")

        with self.assertRaises(KeyError):
            rasterizer.get_depth_mask("nonexistent")

    def test_empty_mask_visibility_ratio(self):
        """Test visibility ratio for object with no projected pixels."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        # Visibility ratio should be 0.0 when no pixels are projected
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 0.0)

    def test_three_objects_layered_occlusion(self):
        """Test three objects at different depths with layered occlusion."""
        # All three cover the same region
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:50, 20:70] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:50, 20:70] = 5.0  # Front

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:50, 20:70] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:50, 20:70] = 10.0  # Middle

        mask3 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask3[20:50, 20:70] = True
        depth3 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth3[20:50, 20:70] = 15.0  # Back

        objects = {
            "front": {"geometry": MockGeometry(mask1, depth1)},
            "middle": {"geometry": MockGeometry(mask2, depth2)},
            "back": {"geometry": MockGeometry(mask3, depth3)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Only front object should be visible
        self.assertEqual(rasterizer.get_visibility_ratio("front"), 1.0)
        self.assertEqual(rasterizer.get_visibility_ratio("middle"), 0.0)
        self.assertEqual(rasterizer.get_visibility_ratio("back"), 0.0)


class TestMinProjectedSize(unittest.TestCase):
    """Test cases for the min_projected_size filtering behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_width = 100
        self.image_height = 80
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _create_rasterizer(self, objects, depth_tolerance=0.05, min_projected_size=16):
        """Helper to create a SceneRasterizer with standard parameters."""
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            depth_tolerance=depth_tolerance,
            min_projected_size=min_projected_size,
            logger=self.logger,
        )

    def test_short_object_filtered(self):
        """Test that objects with height < min_projected_size are filtered even if width passes."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:20, 10:40] = True  # 10h x 30w: height < 16
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:20, 10:40] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        vis_mask = rasterizer.get_visibility_mask("obj1")
        self.assertFalse(np.any(vis_mask))
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 0.0)
        self.assertEqual(rasterizer.get_visible_pixel_count("obj1"), 0)

    def test_narrow_object_filtered(self):
        """Test that objects with width < min_projected_size are filtered even if height passes."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:40, 10:20] = True  # 30h x 10w: width < 16
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:40, 10:20] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        vis_mask = rasterizer.get_visibility_mask("obj1")
        self.assertFalse(np.any(vis_mask))

    def test_at_threshold_passes(self):
        """Test that objects exactly at min_projected_size threshold are visible."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:26, 10:26] = True  # 16h x 16w: both == min_projected_size
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:26, 10:26] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        vis_mask = rasterizer.get_visibility_mask("obj1")
        assert_array_equal(vis_mask, mask)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

    def test_one_pixel_below_threshold_filtered(self):
        """Test that objects one pixel below the threshold are filtered."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:25, 10:25] = True  # 15h x 15w: both one pixel below threshold
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:25, 10:25] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        vis_mask = rasterizer.get_visibility_mask("obj1")
        self.assertFalse(np.any(vis_mask))

    def test_custom_threshold(self):
        """Test that a custom min_projected_size is respected."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:20, 10:30] = True  # 10h x 20w
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:20, 10:30] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}

        # With min_projected_size=8, 10h x 20w should pass
        rasterizer = self._create_rasterizer(objects, min_projected_size=8)
        vis_mask = rasterizer.get_visibility_mask("obj1")
        assert_array_equal(vis_mask, mask)

    def test_zero_threshold_disables_filter(self):
        """Test that min_projected_size=0 disables the small object filter."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10, 10] = True  # 1h x 1w — smallest possible
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10, 10] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects, min_projected_size=0)

        vis_mask = rasterizer.get_visibility_mask("obj1")
        assert_array_equal(vis_mask, mask)

    def test_filtered_object_depth_still_occludes(self):
        """Test that a filtered small object's depth still participates in occlusion."""
        # Small close object — will be filtered but its depth enters the scene buffer
        mask_small = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_small[30:35, 30:35] = True  # 5h x 5w: below threshold
        depth_small = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_small[30:35, 30:35] = 1.0

        # Large far object overlapping the small one
        mask_large = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_large[20:50, 20:50] = True  # 30h x 30w: above threshold
        depth_large = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_large[20:50, 20:50] = 5.0

        objects = {
            "small": {"geometry": MockGeometry(mask_small, depth_small)},
            "large": {"geometry": MockGeometry(mask_large, depth_large)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Small object is filtered
        self.assertFalse(np.any(rasterizer.get_visibility_mask("small")))

        # Large object is partially occluded where the small object's depth wins
        vis_large = rasterizer.get_visibility_mask("large")
        expected = mask_large.copy()
        expected[30:35, 30:35] = False
        assert_array_equal(vis_large, expected)

    def test_large_projected_bbox_but_small_visibility_bbox_filtered(self):
        """Test that an object passing the projected-size check is still filtered if occlusion
        shrinks its visible region below the threshold."""
        # Front object covers most of the back object, leaving only a thin visible strip
        mask_front = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_front[20:50, 20:50] = True  # 30x30
        depth_front = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_front[20:50, 20:50] = 1.0

        # Back object is large (30x30) but only a thin strip escapes the front object
        mask_back = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_back[20:50, 20:50] = True  # Same 30x30 region
        depth_back = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_back[20:50, 20:50] = 10.0
        # Extend the back object by a thin 5px strip to the right (visible, not occluded)
        mask_back[20:50, 50:55] = True  # 30h x 5w strip
        depth_back[20:50, 50:55] = 10.0

        objects = {
            "front": {"geometry": MockGeometry(mask_front, depth_front)},
            "back": {"geometry": MockGeometry(mask_back, depth_back)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Front is fully visible
        assert_array_equal(rasterizer.get_visibility_mask("front"), mask_front)

        # Back object's projected bbox is 30x35 (passes threshold), but after occlusion
        # only the 30x5 strip remains — width=5 < 16, so it should be filtered.
        vis_back = rasterizer.get_visibility_mask("back")
        self.assertFalse(np.any(vis_back))

    def test_large_projected_bbox_but_tiny_visibility_filtered(self):
        """Test that an object is filtered when occlusion leaves a visibility region
        smaller than min_projected_size in either dimension."""
        # Front object covers almost all of back object
        mask_front = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_front[20:50, 20:50] = True  # 30x30
        depth_front = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_front[20:50, 20:50] = 1.0

        # Back object is 30x30 but only a tiny 5x5 corner escapes the front object
        mask_back = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_back[20:50, 20:50] = True
        depth_back = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_back[20:50, 20:50] = 10.0
        # Add a 5x5 extension (both dims < 16)
        mask_back[50:55, 50:55] = True
        depth_back[50:55, 50:55] = 10.0

        objects = {
            "front": {"geometry": MockGeometry(mask_front, depth_front)},
            "back": {"geometry": MockGeometry(mask_back, depth_back)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Back object's projected bbox is large, but post-occlusion only the 5x5 corner
        # is visible — both dims < 16, so it should be filtered
        vis_back = rasterizer.get_visibility_mask("back")
        self.assertFalse(np.any(vis_back))

    def test_mixed_small_and_large_objects(self):
        """Test that only small objects are filtered in a mixed scene."""
        mask_small = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_small[5:10, 5:10] = True  # 5h x 5w: below threshold
        depth_small = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_small[5:10, 5:10] = 5.0

        mask_large = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_large[40:64, 50:80] = True  # 24h x 30w: both dims above threshold
        depth_large = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_large[40:64, 50:80] = 10.0

        objects = {
            "small": {"geometry": MockGeometry(mask_small, depth_small)},
            "large": {"geometry": MockGeometry(mask_large, depth_large)},
        }
        rasterizer = self._create_rasterizer(objects)

        self.assertFalse(np.any(rasterizer.get_visibility_mask("small")))
        assert_array_equal(rasterizer.get_visibility_mask("large"), mask_large)
        self.assertEqual(rasterizer.get_visibility_ratio("large"), 1.0)


if __name__ == "__main__":
    unittest.main()
