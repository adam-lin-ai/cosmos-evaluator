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

"""Unit tests for frame processing utilities."""

import unittest

import numpy as np

from checks.utils.frame_processing import bbox_iou_int, compute_dynamic_mask, ensure_same_size, expand_bbox, to_gray
from checks.utils.types import DynParams


class TestFrameProcessing(unittest.TestCase):
    """Unit tests for frame processing utilities."""

    def setUp(self) -> None:
        """Set up test fixtures - equivalent to pytest fixture."""
        # Use default DynParams values for testing
        # (these match the defaults in the DynParams class)
        self.dyn_params = DynParams()

    def test_to_gray_2d(self) -> None:
        """Test the to_gray function with a 2D image."""
        frame = np.array([[0, 0, 255], [128, 0, 128], [255, 128, 0]])
        expected = np.array([[0, 0, 255], [128, 0, 128], [255, 128, 0]])
        result = to_gray(frame)
        np.testing.assert_array_equal(result, expected)

    def test_to_gray_3d(self) -> None:
        """Test the to_gray function with a 3D image."""
        frame = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],  # Red, Green
                [[255, 0, 0], [128, 128, 128]],  # Blue, Gray
            ],
            dtype=np.uint8,
        )

        expected = np.array([[76, 150], [29, 128]], dtype=np.uint8)
        result = to_gray(frame)
        np.testing.assert_array_equal(result, expected)

    def test_ensure_same_size(self) -> None:
        """Test the ensure_same_size function."""
        frame = np.array([[0, 0, 255], [128, 0, 128], [255, 128, 0]])
        target_hw = (3, 3)
        result = ensure_same_size(frame, target_hw)
        np.testing.assert_array_equal(result, frame)

    def test_compute_dynamic_mask_equal(self) -> None:
        """Test the compute_dynamic_mask function with no movement between frames."""
        prev_gray = np.array([[0, 0, 255], [128, 0, 128], [255, 128, 0]], dtype=np.uint8)
        curr_gray = np.array([[0, 0, 255], [128, 0, 128], [255, 128, 0]], dtype=np.uint8)
        result = compute_dynamic_mask(prev_gray, curr_gray, self.dyn_params)
        np.testing.assert_array_equal(result[0], np.zeros_like(curr_gray, dtype=np.uint8))
        np.testing.assert_array_equal(result[1], curr_gray)

    def test_compute_dynamic_mask_diff(self) -> None:
        """Test the compute_dynamic_mask function with movement between frames."""
        prev_gray = np.zeros((10, 10), dtype=np.uint8)
        prev_gray[0:5, 0:5] = 200  # white square on top left
        curr_gray = np.zeros((10, 10), dtype=np.uint8)
        curr_gray[5:10, 5:10] = 200  # white square moved to bottom right
        computed_mask, curr_gray_echo = compute_dynamic_mask(prev_gray, curr_gray, self.dyn_params)

        expected_mask = np.array(
            [
                [255, 255, 255, 255, 255, 255, 255, 0, 0, 0],
                [255, 255, 255, 255, 255, 255, 255, 0, 0, 0],
                [255, 255, 255, 255, 255, 255, 255, 255, 0, 0],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
                [0, 0, 0, 255, 255, 255, 255, 255, 255, 255],
                [0, 0, 0, 255, 255, 255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(computed_mask, expected_mask)
        np.testing.assert_array_equal(curr_gray_echo, curr_gray)

    def test_bbox_iou_int(self) -> None:
        """Test the bbox_iou_int function."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        result = bbox_iou_int(bbox1, bbox2)
        self.assertAlmostEqual(result, 0.142857142)

    def test_expand_bbox(self) -> None:
        """Test the expand_bbox function."""
        bbox = (0, 0, 10, 10)
        scale = 2.0
        width = 100
        height = 100
        result = expand_bbox(*bbox, scale, width, height)
        self.assertEqual(result, (0, 0, 15, 15))


if __name__ == "__main__":
    unittest.main()
