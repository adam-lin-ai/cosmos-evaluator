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

"""Unit tests for seg_helper module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.obstacle.seg_helper import SegHelper


class TestSegHelper(unittest.TestCase):
    """Test cases for SegHelper class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the model setup to avoid loading actual ONNX model
        with (
            patch("checks.obstacle.seg_helper.setup_model") as mock_setup,
            patch("checks.obstacle.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            # Mock setup_model return values
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            # Mock input/output names
            mock_input = MagicMock()
            mock_input.name = "input_tensor"
            mock_output = MagicMock()
            mock_output.name = "output_tensor"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            # Mock colormap
            mock_colormap.return_value = (
                {0: (128, 64, 128), 13: (0, 0, 142)},  # road and car colors
                {"road": 0, "car": 13},
            )

            self.seg_helper = SegHelper("cuda")

    def test_init(self):
        """Test SegHelper initialization."""
        self.assertEqual(self.seg_helper.model_device, "cuda")
        self.assertIsNotNone(self.seg_helper.ort_session)
        self.assertIsNotNone(self.seg_helper.io_binding)
        self.assertIsNotNone(self.seg_helper.colormap)
        self.assertIsNotNone(self.seg_helper.class_label_to_idx)

    def test_init_cpu_device(self):
        """Test SegHelper initialization with CPU device."""
        with (
            patch("checks.obstacle.seg_helper.setup_model") as mock_setup,
            patch("checks.obstacle.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            mock_colormap.return_value = ({}, {})

            seg_helper_cpu = SegHelper("cpu")

            self.assertEqual(seg_helper_cpu.model_device, "cpu")

    def test_get_class_color_valid_class(self):
        """Test getting color for a valid class."""
        color = self.seg_helper.get_class_color("car")
        expected_color = np.array((0, 0, 142), dtype=np.uint8)
        self.assertTrue(np.array_equal(color, expected_color))

    def test_get_class_color_invalid_class(self):
        """Test getting color for an invalid class."""
        with self.assertRaises(ValueError):
            self.seg_helper.get_class_color("nonexistent_class")

    @patch("checks.obstacle.seg_helper.get_masks")
    def test_process_frame(self, mock_get_masks):
        """Test processing a single frame."""
        # Create mock frame
        frame = torch.zeros(3, 480, 640)  # C, H, W format

        # Mock ONNX inference outputs
        mock_output = [np.random.randint(0, 19, (1, 480, 640))]
        self.seg_helper.io_binding.copy_outputs_to_cpu.return_value = mock_output

        # Mock get_masks return value
        mock_masks = torch.zeros(3, 480, 640)
        mock_get_masks.return_value = mock_masks

        result = self.seg_helper.process_frame(frame)

        # Check that inference was called
        self.seg_helper.io_binding.bind_cpu_input.assert_called_once()
        self.seg_helper.ort_session.run_with_iobinding.assert_called_once()
        self.seg_helper.io_binding.copy_outputs_to_cpu.assert_called_once()

        # Check that get_masks was called with correct parameters
        mock_get_masks.assert_called_once_with(mock_output[0], frame.shape[1:])

        self.assertEqual(result.shape, mock_masks.shape)

    def test_get_class_mask_single_class(self):
        """Test extracting mask for a single high-level class (vehicle contains car)."""
        # Create mock segmentation masks with known colors
        seg_masks = torch.zeros(3, 100, 100)  # RGB format

        # Set some pixels to car color (0, 0, 142)
        seg_masks[2, 10:20, 10:20] = 142  # Blue channel

        # Mock get_class_color to return car color (used for vehicle mapping)
        with patch.object(self.seg_helper, "get_class_color", return_value=(0, 0, 142)):
            mask = self.seg_helper.get_class_mask(seg_masks, "vehicle")

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, bool)

        # Check that the car pixels are True
        self.assertTrue(np.any(mask[10:20, 10:20]))

    def test_get_class_mask_vehicle_group(self):
        """Test extracting mask for vehicle group."""
        # Create mock segmentation masks
        seg_masks = torch.zeros(3, 100, 100)

        # Set pixels for different vehicle types
        seg_masks[2, 10:20, 10:20] = 142  # Car color (blue channel)
        seg_masks[1, 30:40, 30:40] = 128  # Truck color (green channel)

        # Mock get_class_color for different vehicle classes
        def mock_get_color(class_name):
            colors = {"car": (0, 0, 142), "truck": (0, 128, 0), "bus": (0, 60, 100)}
            if class_name in colors:
                return colors[class_name]
            raise ValueError(f"Unknown class: {class_name}")

        with patch.object(self.seg_helper, "get_class_color", side_effect=mock_get_color):
            mask = self.seg_helper.get_class_mask(seg_masks, "vehicle")

        # Check that both car and truck pixels are included
        self.assertTrue(np.any(mask[10:20, 10:20]))  # Car pixels
        self.assertTrue(np.any(mask[30:40, 30:40]))  # Truck pixels

    def test_get_class_mask_unknown_class(self):
        """Test extracting mask for unknown class."""
        seg_masks = torch.zeros(3, 100, 100)

        # Mock get_class_color to raise ValueError for unknown classes
        with patch.object(self.seg_helper, "get_class_color", side_effect=ValueError):
            mask = self.seg_helper.get_class_mask(seg_masks, "unknown_class")

        # Should return all-zeros mask
        self.assertEqual(mask.shape, (100, 100))
        self.assertFalse(np.any(mask))

    def test_resize_masks(self):
        """Test resizing segmentation masks."""
        # Create input masks
        input_masks = torch.zeros(3, 100, 100)
        target_height, target_width = 200, 300

        with patch("torch.nn.functional.interpolate") as mock_interpolate:
            # Mock interpolate to return correct shape
            mock_result = torch.zeros(1, 3, 200, 300)
            mock_interpolate.return_value = mock_result

            result = self.seg_helper.resize_masks(input_masks, target_height, target_width)

            # Check that interpolate was called with correct parameters
            mock_interpolate.assert_called_once()
            call_args = mock_interpolate.call_args
            self.assertEqual(call_args[1]["size"], (target_height, target_width))
            self.assertEqual(call_args[1]["mode"], "nearest")

            # Check result shape after squeeze
            self.assertEqual(result.shape, (3, 200, 300))

    # get_video_dataloader is no longer part of SegHelper; handled via processor

    def test_class_to_segmentation_mapping(self):
        """Test the class to segmentation classes mapping."""
        # Test that the mapping is correctly defined
        expected_mappings = {
            "vehicle": ["car", "truck", "bus"],
            "pedestrian": ["person"],
            "motorcycle": ["motorcycle", "rider"],
            "bicycle": ["bicycle", "rider"],
        }

        # This is tested indirectly through get_class_mask
        seg_masks = torch.zeros(3, 100, 100)

        with patch.object(self.seg_helper, "get_class_color") as mock_get_color:
            mock_get_color.side_effect = ValueError  # Simulate unknown classes

            # Test each mapping
            for class_name, expected_classes in expected_mappings.items():
                self.seg_helper.get_class_mask(seg_masks, class_name)

                # Should call get_class_color for each expected segmentation class
                expected_calls = len(expected_classes)
                actual_calls = mock_get_color.call_count

                # Check that the correct number of calls were made
                self.assertEqual(actual_calls, expected_calls)

                # Reset mock for next iteration
                mock_get_color.reset_mock()

    def test_setup_model_calls(self):
        """Test that setup_model is called with correct parameters."""
        with (
            patch("checks.obstacle.seg_helper.setup_model") as mock_setup,
            patch("checks.obstacle.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            mock_colormap.return_value = ({}, {})

            SegHelper("cuda")

            # Check that setup_model was called with verbose=False
            mock_setup.assert_called_once_with(model_device="cuda", verbose=False)


if __name__ == "__main__":
    unittest.main()
