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

"""Unit tests for segformer module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.utils.segformer import (
    CropTransform,
    bbox_iou,
    create_cityscapes_label_colormap,
    estimate_confidence,
    get_bbox,
    get_closest_contour,
    get_contour_points,
    get_dataloader,
    get_decoded_predictions,
    get_masks,
    get_model_input_shape,
    get_normalization_constants,
    get_transforms_fn,
    reached_end_of_segment,
    resample_contours,
    setup_model,
    transform_to_8MP,
    xywh_to_tlbr,
)


class TestCropTransform(unittest.TestCase):
    """Test cases for CropTransform class."""

    def test_init(self):
        """Test CropTransform initialization."""
        transform = CropTransform(top=10, left=20, height=100, width=200)

        self.assertEqual(transform.top, 10)
        self.assertEqual(transform.left, 20)
        self.assertEqual(transform.height, 100)
        self.assertEqual(transform.width, 200)

    def test_call(self):
        """Test CropTransform __call__ method."""
        transform = CropTransform(top=10, left=20, height=100, width=200)

        # Create a test tensor
        test_tensor = torch.zeros(3, 300, 400)

        with patch("torchvision.transforms.functional.crop") as mock_crop:
            mock_crop.return_value = torch.zeros(3, 100, 200)

            result = transform(test_tensor)

            mock_crop.assert_called_once_with(test_tensor, 10, 20, 100, 200)
            self.assertEqual(result.shape, (3, 100, 200))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_get_model_input_shape(self):
        """Test get_model_input_shape function."""
        shape = get_model_input_shape()
        expected_shape = (1, 3, 1024, 1820)
        self.assertEqual(shape, expected_shape)

    @patch("checks.utils.segformer.get_inference_session")
    def test_setup_model(self, mock_get_inference_session):
        """Test setup_model function."""
        # Mock inference session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        mock_io_binding = MagicMock()
        mock_session.io_binding.return_value = mock_io_binding

        mock_get_inference_session.return_value = mock_session

        with patch("checks.utils.onnx.configure_onnx_logging"):
            sess, io_binding = setup_model(verbose=False)

        self.assertEqual(sess, mock_session)
        self.assertEqual(io_binding, mock_io_binding)

        # Verify warm-up run was performed
        mock_io_binding.bind_cpu_input.assert_called_once()
        mock_io_binding.bind_output.assert_called_once_with("output", device_type="cuda")
        mock_session.run_with_iobinding.assert_called_once_with(mock_io_binding)
        mock_io_binding.copy_outputs_to_cpu.assert_called_once()

    def test_create_cityscapes_label_colormap(self):
        """Test create_cityscapes_label_colormap function."""
        colormap, class_idx2label = create_cityscapes_label_colormap()

        # Check that we have the expected number of classes
        self.assertEqual(len(colormap), 19)
        self.assertEqual(len(class_idx2label), 19)

        # Check some specific mappings
        self.assertIn("road", class_idx2label)
        self.assertIn("car", class_idx2label)
        self.assertIn("person", class_idx2label)

        # Check that all colors are RGB tuples
        for color in colormap.values():
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_get_normalization_constants(self):
        """Test get_normalization_constants function."""
        mean, std = get_normalization_constants()

        expected_mean = [123.675, 116.28, 103.53]
        expected_std = [58.395, 57.12, 57.375]

        self.assertEqual(mean, expected_mean)
        self.assertEqual(std, expected_std)

    def test_get_transforms_fn(self):
        """Test get_transforms_fn function."""
        import torchvision.transforms as T

        transforms = get_transforms_fn()

        self.assertIsInstance(transforms, T.Compose)
        # Should have 2 transforms: Resize and Normalize
        self.assertEqual(len(transforms.transforms), 2)

    @patch("checks.utils.segformer.DataLoader")
    @patch("checks.utils.segformer.VideoDataset")
    def test_get_dataloader(self, mock_video_dataset, mock_dataloader):
        """Test get_dataloader function."""
        video_path = "/path/to/video.mp4"
        start_frame = 10

        mock_dataset = MagicMock()
        mock_video_dataset.return_value = mock_dataset

        mock_loader = MagicMock()
        mock_dataloader.return_value = mock_loader

        dataloader, video_dataset = get_dataloader(video_path, start_frame)

        # Check that VideoDataset was called with correct parameters
        mock_video_dataset.assert_called_once()
        call_args = mock_video_dataset.call_args
        self.assertEqual(call_args[0][0], video_path)
        self.assertEqual(call_args[1]["start_frame"], start_frame)

        # Check that DataLoader was called with correct parameters
        mock_dataloader.assert_called_once_with(mock_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.assertEqual(dataloader, mock_loader)
        self.assertEqual(video_dataset, mock_dataset)

    def test_get_masks(self):
        """Test get_masks function."""
        # Create sample mask prediction
        mask_prediction = np.array([[[0, 1, 2], [3, 4, 5]]])  # 1x2x3
        image_shape = (3, 2, 3)  # C, H, W

        result = get_masks(mask_prediction, image_shape)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, image_shape)

    def test_reached_end_of_segment(self):
        """Test reached_end_of_segment function."""
        # Test cases: (current_frame, end_frame, expected_result)
        test_cases = [
            (5, 10, False),  # Not reached end
            (10, 10, False),  # At end frame
            (15, 10, True),  # Past end frame
            (5, -1, False),  # No end frame specified
            (0, 0, False),  # At frame 0 with end frame 0
        ]

        for current_frame, end_frame, expected in test_cases:
            result = reached_end_of_segment(current_frame, end_frame)
            self.assertEqual(result, expected, f"Failed for current_frame={current_frame}, end_frame={end_frame}")

    def test_reached_end_of_segment_invalid_inputs(self):
        """Test reached_end_of_segment with invalid inputs."""
        with self.assertRaises(AssertionError):
            reached_end_of_segment(-1, 10)  # Negative current frame

        with self.assertRaises(AssertionError):
            reached_end_of_segment(5, -2)  # Invalid end frame

    def test_get_decoded_predictions(self):
        """Test get_decoded_predictions function."""
        # Create sample contours
        contours = [np.array([[10, 20], [30, 40], [50, 60]]), np.array([[70, 80], [90, 100]])]
        confidences = [0.8, 0.6]

        with patch("checks.utils.segformer.transform_to_8MP", side_effect=lambda x: x):
            result = get_decoded_predictions(contours, confidences)

        expected_keys = ["classes", "polyline_vertices", "confidences", "track_ids"]
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertEqual(len(result["classes"]), 2)
        self.assertEqual(len(result["polyline_vertices"]), 2)
        self.assertEqual(len(result["confidences"]), 2)
        self.assertEqual(len(result["track_ids"]), 2)

        # Check specific values
        self.assertEqual(result["classes"], [0, 0])
        self.assertEqual(result["confidences"], [0.8, 0.6])
        self.assertEqual(result["track_ids"], [0, 0])

    def test_get_decoded_predictions_no_confidences(self):
        """Test get_decoded_predictions without confidences."""
        contours = [np.array([[10, 20], [30, 40]])]

        with patch("checks.utils.segformer.transform_to_8MP", side_effect=lambda x: x):
            result = get_decoded_predictions(contours)

        self.assertEqual(result["confidences"], [0.0])

    def test_resample_contours(self):
        """Test resample_contours function."""
        # Create test contours
        contour1 = np.array([[i, i * 2] for i in range(100)])  # 100 points
        contour2 = np.array([[i, i * 3] for i in range(50)])  # 50 points
        contours = [contour1, contour2]

        num_points = 20
        result = resample_contours(contours, num_points)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), num_points)
        self.assertEqual(len(result[1]), num_points)

    def test_bbox_iou(self):
        """Test bbox_iou function."""
        # Test cases: (boxA, boxB, expected_iou)
        test_cases = [
            # Perfect overlap
            ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
            # No overlap
            ([0, 0, 5, 5], [10, 10, 15, 15], 0.0),
            # Partial overlap - intersection is 6x6=36, areas are 11x11=121 each, union=121+121-36=206, IoU=36/206≈0.175
            ([0, 0, 10, 10], [5, 5, 15, 15], 36 / 206),
            # One box inside another - intersection is 7x7=49, areas are 121 and 49, union=121, IoU=49/121≈0.405
            ([0, 0, 10, 10], [2, 2, 8, 8], 49 / 121),
            # None boxes
            (None, [0, 0, 5, 5], 0.0),
            ([0, 0, 5, 5], None, 0.0),
        ]

        for boxA, boxB, expected_iou in test_cases:
            result = bbox_iou(boxA, boxB)
            self.assertAlmostEqual(result, expected_iou, places=2, msg=f"Failed for boxA={boxA}, boxB={boxB}")

    def test_xywh_to_tlbr(self):
        """Test xywh_to_tlbr function."""
        # Test cases: (xywh, expected_tlbr)
        test_cases = [
            ([10, 20, 30, 40], (10, 20, 40, 60)),  # x, y, w, h -> x1, y1, x2, y2
            ([0, 0, 5, 5], (0, 0, 5, 5)),
            ([100, 200, 50, 75], (100, 200, 150, 275)),
        ]

        for xywh, expected_tlbr in test_cases:
            result = xywh_to_tlbr(xywh)
            self.assertEqual(result, expected_tlbr)

    def test_xywh_to_tlbr_invalid_input(self):
        """Test xywh_to_tlbr with invalid input."""
        with self.assertRaises(AssertionError):
            xywh_to_tlbr(None)

        with self.assertRaises(AssertionError):
            xywh_to_tlbr([1, 2, 3])  # Too few elements

    @patch("cv2.boundingRect")
    def test_get_bbox(self, mock_bounding_rect):
        """Test get_bbox function."""
        contour = np.array([[10, 20], [30, 40], [50, 60]])
        mock_bounding_rect.return_value = (5, 10, 20, 15)  # x, y, w, h

        result = get_bbox(contour)

        mock_bounding_rect.assert_called_once_with(contour)
        expected_result = (5, 10, 25, 25)  # x, y, x+w, y+h
        self.assertEqual(result, expected_result)

    @patch("cv2.matchShapes")
    @patch("checks.utils.segformer.get_bbox")
    @patch("checks.utils.segformer.bbox_iou")
    def test_get_closest_contour(self, mock_bbox_iou, mock_get_bbox, mock_match_shapes):
        """Test get_closest_contour function."""
        ref_contour = np.array([[10, 20], [30, 40]])
        contours = [
            np.array([[15, 25], [35, 45]]),  # Similar contour
            np.array([[100, 200], [300, 400]]),  # Different contour
        ]

        # Mock return values
        mock_get_bbox.side_effect = [
            (0, 0, 50, 50),  # ref_contour bbox
            (5, 5, 55, 55),  # contour 0 bbox
            (100, 100, 400, 400),  # contour 1 bbox
        ]
        mock_bbox_iou.side_effect = [0.8, 0.1]  # IoUs with ref contour
        mock_match_shapes.side_effect = [2.0, 8.0]  # Shape similarity scores

        closest, score = get_closest_contour(ref_contour, contours, iou_threshold=0.5)

        # Should return the first contour (higher IoU and better score)
        np.testing.assert_array_equal(closest, contours[0])
        self.assertEqual(score, 2.0)

    @patch("cv2.matchShapes")
    @patch("checks.utils.segformer.get_bbox")
    @patch("checks.utils.segformer.bbox_iou")
    def test_get_closest_contour_no_match(self, mock_bbox_iou, mock_get_bbox, mock_match_shapes):
        """Test get_closest_contour when no contour meets IoU threshold."""
        ref_contour = np.array([[10, 20], [30, 40]])
        contours = [np.array([[100, 200], [300, 400]])]

        mock_get_bbox.side_effect = [(0, 0, 50, 50), (100, 100, 400, 400)]
        mock_bbox_iou.return_value = 0.1  # Below threshold

        closest, score = get_closest_contour(ref_contour, contours, iou_threshold=0.5)

        self.assertIsNone(closest)
        self.assertEqual(score, 100.0)

    def test_estimate_confidence(self):
        """Test estimate_confidence function."""
        # Test cases: (score, expected_range)
        test_cases = [
            (0.0, (0.9, 1.0)),  # Perfect match -> high confidence
            (5.0, (0.4, 0.6)),  # Medium score -> medium confidence
            (10.0, (0.0, 0.1)),  # Poor match -> low confidence
        ]

        for score, (min_conf, max_conf) in test_cases:
            result = estimate_confidence(score)
            self.assertGreaterEqual(result, min_conf)
            self.assertLessEqual(result, max_conf)

    @patch("cv2.findContours")
    @patch("cv2.threshold")
    @patch("cv2.cvtColor")
    @patch("cv2.dilate")
    @patch("cv2.arcLength")
    def test_get_contour_points(self, mock_arc_length, mock_dilate, mock_cvt_color, mock_threshold, mock_find_contours):
        """Test get_contour_points function."""
        # Mock inputs
        im_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock return values
        mock_cvt_color.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        mock_dilate.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Mock contours and hierarchy
        contour1 = np.array([[10, 20], [30, 40], [50, 60]])
        contour2 = np.array([[70, 80], [90, 100]])
        contours = [contour1, contour2]
        hierarchy = np.array([[[1, -1, -1, -1], [-1, 0, -1, -1]]])  # No children

        mock_find_contours.return_value = (contours, hierarchy)
        mock_arc_length.side_effect = [30.0, 20.0]  # Lengths above min_length

        result = get_contour_points(im_rgb, min_length=25)

        # Should return contours with length >= min_length
        self.assertEqual(len(result), 1)  # Only first contour meets length requirement
        np.testing.assert_array_equal(result[0], contour1)

    def test_transform_to_8MP(self):
        """Test transform_to_8MP function."""
        # Test cases: (input_points, expected_output)
        test_cases = [
            (np.array([[0, 0]]), np.array([[105.0, 121.0]])),
            (np.array([[10, 20]]), np.array([[125.0, 161.0]])),  # 10*2+105, 20*2+121
            (np.array([[0, 0], [10, 20]]), np.array([[105.0, 121.0], [125.0, 161.0]])),
        ]

        for input_points, expected_output in test_cases:
            result = transform_to_8MP(input_points)
            np.testing.assert_array_almost_equal(result, expected_output)

    def test_transform_to_8MP_scaling_and_padding(self):
        """Test transform_to_8MP scaling and padding calculations."""
        # Input coordinate (1, 1) should become (1*2 + 105, 1*2 + 121) = (107, 123)
        input_point = np.array([[1, 1]])
        result = transform_to_8MP(input_point)
        expected = np.array([[107.0, 123.0]])

        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
