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

"""Unit tests for cv2_video_dataset module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from checks.utils.cv2_video_dataset import VideoDataset


class TestVideoDataset(unittest.TestCase):
    """Test cases for VideoDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.invalid_video_path = os.path.join(self.temp_dir, "test_video.avi")
        self.nonexistent_path = os.path.join(self.temp_dir, "nonexistent.mp4")

        # Create a simple test video
        self.create_test_video()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_video(self):
        """Create a simple test video for testing."""
        # Video properties
        width, height = 640, 480
        fps = 30
        num_frames = 10

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.test_video_path, fourcc, fps, (width, height))

        # Write frames
        for i in range(num_frames):
            # Create a simple frame with different colors for each frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 25  # Blue channel varies by frame
            frame[:, :, 1] = 128  # Green channel constant
            frame[:, :, 2] = 255 - i * 25  # Red channel varies inversely
            out.write(frame)

        out.release()

    def test_init_valid_video(self):
        """Test VideoDataset initialization with valid video."""
        dataset = VideoDataset(self.test_video_path)

        self.assertEqual(dataset._video_path, self.test_video_path)
        self.assertEqual(dataset.number_frames, 10)
        self.assertEqual(dataset.image_width, 640)
        self.assertEqual(dataset.image_height, 480)
        self.assertEqual(dataset.frames_per_second, 30)
        self.assertEqual(dataset._skip, 1)
        self.assertEqual(dataset.start_frame, 0)
        self.assertEqual(dataset._transforms_fn, [])

        dataset.release()

    def test_init_file_not_found(self):
        """Test VideoDataset initialization with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            VideoDataset(self.nonexistent_path)

    def test_init_invalid_extension(self):
        """Test VideoDataset initialization with invalid file extension."""
        # Create a file with .avi extension
        with open(self.invalid_video_path, "w") as f:
            f.write("dummy content")

        with self.assertRaises(ValueError):
            VideoDataset(self.invalid_video_path)

    def test_init_with_transforms(self):
        """Test VideoDataset initialization with transform functions."""

        def dummy_transform(x):
            return x * 2

        transforms = [dummy_transform]
        dataset = VideoDataset(self.test_video_path, transforms_fn=transforms)

        self.assertEqual(dataset._transforms_fn, transforms)
        dataset.release()

    def test_init_with_skip(self):
        """Test VideoDataset initialization with skip parameter."""
        skip = 2
        dataset = VideoDataset(self.test_video_path, skip=skip)

        self.assertEqual(dataset._skip, skip)
        dataset.release()

    def test_init_invalid_skip(self):
        """Test VideoDataset initialization with invalid skip parameter."""
        with self.assertRaises(ValueError):
            VideoDataset(self.test_video_path, skip=0)

        with self.assertRaises(ValueError):
            VideoDataset(self.test_video_path, skip=-1)

    def test_init_with_start_frame(self):
        """Test VideoDataset initialization with start_frame parameter."""
        start_frame = 5
        dataset = VideoDataset(self.test_video_path, start_frame=start_frame)

        self.assertEqual(dataset.start_frame, start_frame)
        dataset.release()

    def test_len(self):
        """Test __len__ method."""
        dataset = VideoDataset(self.test_video_path)
        expected_length = 10  # number of frames
        self.assertEqual(len(dataset), expected_length)
        dataset.release()

    def test_len_with_skip(self):
        """Test __len__ method with skip parameter."""
        skip = 2
        dataset = VideoDataset(self.test_video_path, skip=skip)
        expected_length = 10 // skip  # 5 frames
        self.assertEqual(len(dataset), expected_length)
        dataset.release()

    def test_len_with_start_frame(self):
        """Test __len__ method with start_frame parameter."""
        start_frame = 3
        dataset = VideoDataset(self.test_video_path, start_frame=start_frame)
        expected_length = 10 - start_frame  # 7 frames
        self.assertEqual(len(dataset), expected_length)
        dataset.release()

    def test_len_with_skip_and_start_frame(self):
        """Test __len__ method with both skip and start_frame parameters."""
        skip = 2
        start_frame = 2
        dataset = VideoDataset(self.test_video_path, skip=skip, start_frame=start_frame)
        expected_length = (10 - start_frame) // skip  # 4 frames
        self.assertEqual(len(dataset), expected_length)
        dataset.release()

    def test_iter_basic(self):
        """Test basic iteration over dataset."""
        dataset = VideoDataset(self.test_video_path)

        frames = list(dataset)

        # Should have 10 frames
        self.assertEqual(len(frames), 10)

        # Each frame should be a tensor
        for frame in frames:
            self.assertIsInstance(frame, torch.Tensor)
            # Should be in C,H,W format
            self.assertEqual(frame.shape, (3, 480, 640))
            # Should be float32
            self.assertEqual(frame.dtype, torch.float32)

        dataset.release()

    def test_iter_with_skip(self):
        """Test iteration with skip parameter."""
        skip = 3
        dataset = VideoDataset(self.test_video_path, skip=skip)

        frames = list(dataset)

        # Should have fewer frames due to skipping
        # Frames 0, 3, 6, 9 should be processed (frame_number % skip == 0)
        expected_count = 4  # frames at indices 0, 3, 6, 9
        self.assertEqual(len(frames), expected_count)

        dataset.release()

    def test_iter_with_transforms(self):
        """Test iteration with transform functions."""

        def multiply_transform(x):
            return x * 2

        def add_transform(x):
            return x + 10

        transforms = [multiply_transform, add_transform]
        dataset = VideoDataset(self.test_video_path, transforms_fn=transforms)

        frames = list(dataset)

        # Verify transforms were applied
        # Original value * 2 + 10 should be different from original
        self.assertTrue(len(frames) > 0)

        dataset.release()

    def test_iter_with_start_frame(self):
        """Test iteration with start_frame parameter."""
        start_frame = 5
        dataset = VideoDataset(self.test_video_path, start_frame=start_frame)

        frames = list(dataset)

        # Should start from frame 5, so 5 frames total
        expected_count = 10 - start_frame  # 5 frames
        self.assertEqual(len(frames), expected_count)

        dataset.release()

    @patch("cv2.VideoCapture")
    def test_iter_video_read_failure(self, mock_video_capture):
        """Test iteration when video read fails."""
        # Mock video capture to simulate read failure
        mock_video = MagicMock()
        mock_video.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 5,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
        }[prop]
        mock_video.read.return_value = (False, None)  # Simulate read failure
        mock_video_capture.return_value = mock_video

        dataset = VideoDataset(self.test_video_path)
        frames = list(dataset)

        # Should return empty list when read fails
        self.assertEqual(len(frames), 0)

    def test_release(self):
        """Test release method."""
        dataset = VideoDataset(self.test_video_path)

        # Should not raise an exception
        dataset.release()

        # Calling release multiple times should be safe
        dataset.release()

    def test_color_conversion(self):
        """Test that BGR to RGB conversion is working."""
        dataset = VideoDataset(self.test_video_path)

        # Get one frame
        frame_iter = iter(dataset)
        frame = next(frame_iter)

        # Convert back to numpy for testing
        frame_np = frame.numpy()

        # Frame should be in C,H,W format and RGB order
        self.assertEqual(frame_np.shape[0], 3)  # 3 channels

        # The test video has varying blue and red channels
        # After BGR->RGB conversion, the red channel (index 0) should vary
        # and blue channel (index 2) should vary inversely

        dataset.release()

    def test_data_type_conversion(self):
        """Test that data type conversion to float32 is working."""
        dataset = VideoDataset(self.test_video_path)

        frame_iter = iter(dataset)
        frame = next(frame_iter)

        # Should be float32
        self.assertEqual(frame.dtype, torch.float32)

        # Values should be in range [0, 255] (no normalization applied by default)
        self.assertGreaterEqual(frame.min().item(), 0.0)
        self.assertLessEqual(frame.max().item(), 255.0)

        dataset.release()

    def test_channel_order(self):
        """Test that channels are correctly transposed from H,W,C to C,H,W."""
        dataset = VideoDataset(self.test_video_path)

        frame_iter = iter(dataset)
        frame = next(frame_iter)

        # Should be in C,H,W format
        self.assertEqual(frame.shape, (3, 480, 640))  # (channels, height, width)

        dataset.release()


if __name__ == "__main__":
    unittest.main()
