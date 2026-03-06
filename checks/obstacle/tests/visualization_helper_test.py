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

"""Unit tests for visualization_helper module."""

import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from checks.obstacle.visualization_helper import VisualizationHelper
from checks.utils.scene_rasterizer import SceneRasterizer


class TestVisualizationHelper(unittest.TestCase):
    """Test cases for VisualizationHelper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "test_clip_123"
        self.output_dir = self.temp_dir

        self.vis_helper = VisualizationHelper(self.clip_id, self.output_dir, verbose=False)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test VisualizationHelper initialization."""
        self.assertEqual(self.vis_helper.clip_id, self.clip_id)
        self.assertEqual(self.vis_helper.output_dir, Path(self.output_dir))
        self.assertFalse(self.vis_helper.verbose)
        self.assertIsNone(self.vis_helper.video_writer)
        self.assertIsNone(self.vis_helper.original_video_cap)

    def test_init_verbose(self):
        """Test VisualizationHelper initialization with verbose mode."""
        vis_helper_verbose = VisualizationHelper(self.clip_id, self.output_dir, verbose=True)
        self.assertTrue(vis_helper_verbose.verbose)

    def test_init_creates_output_dir(self):
        """Test that initialization creates output directory."""
        new_output_dir = Path(self.temp_dir) / "new_subdir"
        self.assertFalse(new_output_dir.exists())

        VisualizationHelper(self.clip_id, str(new_output_dir))

        self.assertTrue(new_output_dir.exists())

    @patch("cv2.VideoCapture")
    @patch("cv2.VideoWriter")
    def test_initialize_video_writer_success(self, mock_video_writer, mock_video_capture):
        """Test successful video writer initialization."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        # Mock video writer
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertTrue(result)
        self.assertEqual(self.vis_helper.frame_width, 1280)
        self.assertEqual(self.vis_helper.frame_height, 720)
        self.assertIsNotNone(self.vis_helper.video_output_path)

    @patch("cv2.VideoCapture")
    def test_initialize_video_writer_video_open_fails(self, mock_video_capture):
        """Test video writer initialization when video opening fails."""
        # Mock video capture failure
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        video_path = "/path/to/nonexistent_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertFalse(result)

    @patch("cv2.VideoCapture")
    @patch("cv2.VideoWriter")
    def test_initialize_video_writer_codec_fallback(self, mock_video_writer, mock_video_capture):
        """Test video writer initialization with codec fallback."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        # Mock video writer - first codec fails, second succeeds
        mock_writer_fail = MagicMock()
        mock_writer_fail.isOpened.return_value = False
        mock_writer_success = MagicMock()
        mock_writer_success.isOpened.return_value = True

        mock_video_writer.side_effect = [mock_writer_fail, mock_writer_success]

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertTrue(result)
        # Should have tried multiple codecs
        self.assertEqual(mock_video_writer.call_count, 2)

    @patch("cv2.VideoCapture")
    @patch("cv2.VideoWriter")
    def test_initialize_video_writer_all_codecs_fail(self, mock_video_writer, mock_video_capture):
        """Test video writer initialization when all codecs fail."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        # Mock video writer - all codecs fail
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        mock_video_writer.return_value = mock_writer

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertFalse(result)

    def test_get_original_frame_no_capture(self):
        """Test getting original frame when no video capture is available."""
        # Set frame dimensions for this test
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        frame = self.vis_helper.get_original_frame(frame_idx=5)

        # Should return gray background
        expected_shape = (720, 1280, 3)  # Default height, width, channels
        self.assertEqual(frame.shape, expected_shape)
        self.assertTrue(np.all(frame == 128))  # Gray background

    @patch("cv2.VideoCapture")
    def test_get_original_frame_with_capture(self, mock_video_capture):
        """Test getting original frame with video capture."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        self.vis_helper.original_video_cap = mock_cap
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with patch("cv2.cvtColor") as mock_cvtcolor, patch("cv2.resize") as mock_resize:
            expected_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_cvtcolor.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_resize.return_value = expected_frame

            frame = self.vis_helper.get_original_frame(frame_idx=5)

            # Verify the returned frame
            np.testing.assert_array_equal(frame, expected_frame)

        # Should set frame position and read frame
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 5)
        mock_cap.read.assert_called_once()
        mock_cvtcolor.assert_called_once()
        mock_resize.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_original_frame_read_fails(self, mock_video_capture):
        """Test getting original frame when frame read fails."""
        # Mock video capture with read failure
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        self.vis_helper.original_video_cap = mock_cap
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        frame = self.vis_helper.get_original_frame(frame_idx=5)

        # Should return gray background when read fails
        expected_shape = (720, 1280, 3)
        self.assertEqual(frame.shape, expected_shape)
        self.assertTrue(np.all(frame == 128))

    def test_overlay_segmentation(self):
        """Test overlaying segmentation masks."""
        base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        seg_masks = MagicMock()
        seg_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper._overlay_segmentation(base_image, seg_masks)

        self.assertEqual(result.shape, base_image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_overlay_segmentation_resize_needed(self):
        """Test overlaying segmentation when resizing is needed."""
        base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        seg_masks = MagicMock()
        # Different size than base image
        seg_masks.numpy.return_value = np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper._overlay_segmentation(base_image, seg_masks)

            # Should call resize to match base image dimensions
            mock_resize.assert_called_once_with(unittest.mock.ANY, (1280, 720))

            # Verify the result
            self.assertEqual(result.shape, base_image.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_add_object_overlays(self):
        """Test adding object overlays."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {123: 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects = {}

        # Mock cuboid
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        # Some area
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {123: {"geometry": mock_cuboid}}
        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image

            result = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        # Should create cuboid and draw overlay
        mock_cuboid.get_projected_mask.assert_called_once()
        mock_draw_overlay.assert_called_once()

        # Verify the result is the expected image
        self.assertIs(result, vis_image)

    def test_add_object_overlays_with_scene_rasterizer(self):
        """Test adding object overlays with pre-computed SceneRasterizer."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {1234: 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects = {}

        # Mock cuboid
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 10.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {1234: {"geometry": mock_cuboid}}

        # Create a SceneRasterizer (computes visibility masks in constructor)
        scene_rasterizer = SceneRasterizer(
            frame_objects, camera_pose, camera_model, 1280, 720, logger=logging.getLogger("test")
        )

        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image

            result = self.vis_helper._add_object_overlays(
                vis_image,
                frame_scores,
                frame_objects,
                camera_pose,
                camera_model,
                filtered_objects,
                scene_rasterizer=scene_rasterizer,
            )

        # When scene_rasterizer is provided, it should use cached masks (no additional projection calls)
        # The cuboid's get_projected_mask was called during SceneRasterizer construction, not during _add_object_overlays
        mock_draw_overlay.assert_called_once()
        self.assertIs(result, vis_image)

    def test_add_object_overlays_traffic_light_color(self):
        """Traffic light objects should render with yellow contour color."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {1: 0.9}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects = {}

        mock_cuboid = MagicMock()
        mask = np.zeros((720, 1280), dtype=bool)
        mask[10:20, 10:20] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[10:20, 10:20] = 1.0
        mock_cuboid.get_projected_mask.return_value = mask, mock_depth_mask
        frame_objects = {
            1: {
                "geometry": mock_cuboid,
                "object_type": "TrafficLight",
            }
        }

        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image
            _ = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        # contour_color is the 5th positional argument
        contour_color = mock_draw_overlay.call_args[0][4]
        self.assertEqual(contour_color, (255, 255, 0))  # Yellow

    def test_add_object_overlays_traffic_sign_color(self):
        """Traffic sign objects should render with orange contour color."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {2: 0.9}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects = {}

        mock_cuboid = MagicMock()
        mask = np.zeros((720, 1280), dtype=bool)
        mask[10:20, 10:20] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[10:20, 10:20] = 1.0
        mock_cuboid.get_projected_mask.return_value = mask, mock_depth_mask
        frame_objects = {
            2: {
                "geometry": mock_cuboid,
                "object_type": "TrafficSign",
            }
        }
        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image
            _ = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        contour_color = mock_draw_overlay.call_args[0][4]
        self.assertEqual(contour_color, (255, 165, 0))  # Orange

    @patch("cv2.findContours")
    @patch("cv2.drawContours")
    @patch("cv2.boundingRect")
    @patch("cv2.getTextSize")
    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_object_overlay(
        self, mock_puttext, mock_rectangle, mock_textsize, mock_boundingrect, mock_drawcontours, mock_findcontours
    ):
        """Test drawing object overlay."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        projected_mask = np.zeros((720, 1280), dtype=bool)
        projected_mask[100:200, 100:200] = True
        score = 0.85
        track_id = 123

        # Mock CV2 functions
        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_boundingrect.return_value = (100, 100, 100, 100)  # x, y, w, h
        mock_textsize.return_value = ((30, 15), 3)  # (width, height), baseline

        contour_color = (255, 0, 255)  # magenta sample color
        result = self.vis_helper._draw_object_overlay(vis_image, projected_mask, score, track_id, contour_color)

        # Should find contours and draw overlay
        mock_findcontours.assert_called_once()
        mock_drawcontours.assert_called_once_with(vis_image, [mock_contour], -1, contour_color, 2)
        mock_boundingrect.assert_called_once()
        mock_textsize.assert_called()
        mock_rectangle.assert_called()
        mock_puttext.assert_called()

        # Verify the result
        self.assertIs(result, vis_image)

    def test_mask_legend_override_items(self):
        """Explicit legend items override should be passed to add_legend as-is."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 3
        override_items = [("drivable", (1, 2, 3)), ("vehicle", (9, 8, 7))]

        self.vis_helper.set_mask_styles(legend_items=override_items)

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image
            result = self.vis_helper._add_mask_legend(vis_image, frame_idx)

        mock_add_legend.assert_called_once()
        called_items = mock_add_legend.call_args[0][2]
        self.assertEqual(called_items, override_items)
        self.assertIs(result, vis_image)

        # reset override for other tests
        self.vis_helper.set_mask_styles(legend_items=None)

    def test_mask_legend_ordering(self):
        """Legend should honor provided legend_order, then append remaining colors."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 4
        colors = {"b": (2, 2, 2), "a": (1, 1, 1), "c": (3, 3, 3)}
        order = ["a", "c"]

        self.vis_helper.set_mask_styles(mask_colors=colors, legend_order=order, legend_items=None)

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image
            result = self.vis_helper._add_mask_legend(vis_image, frame_idx)

        called_items = mock_add_legend.call_args[0][2]
        expected = [("a", (1, 1, 1)), ("c", (3, 3, 3)), ("b", (2, 2, 2))]
        self.assertEqual(called_items, expected)
        self.assertIs(result, vis_image)

    def test_named_mask_legend_independent(self):
        """Named mask legends should be independent of the default legend."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5
        default_items = [("MATCH", (0, 255, 0))]
        type_items = [("BBOX_CAR", (200, 50, 50)), ("BBOX_TRUCK", (100, 100, 200))]

        self.vis_helper.set_mask_styles(legend_items=default_items)
        self.vis_helper.set_mask_styles(legend_items=type_items, mask_name="type")

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image

            self.vis_helper._build_mask_legend(vis_image, frame_idx, "default")
            default_call_items = mock_add_legend.call_args[0][2]
            self.assertEqual(default_call_items, default_items)

            self.vis_helper._build_mask_legend(vis_image, frame_idx, "type")
            type_call_items = mock_add_legend.call_args[0][2]
            self.assertEqual(type_call_items, type_items)

    def test_write_mask_frame_creates_named_files(self):
        """write_mask_frame with different mask_name values should create separate files."""
        self.vis_helper.frame_width = 64
        self.vis_helper.frame_height = 48
        self.vis_helper.fps = 10

        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        with patch("checks.obstacle.visualization_helper.VideoWriter") as mock_writer_cls:
            mock_instance = MagicMock()
            mock_instance.open.return_value = True
            mock_instance.write_frame.return_value = True
            mock_writer_cls.return_value = mock_instance

            self.vis_helper.write_mask_frame(frame, 0, "static")
            self.vis_helper.write_mask_frame(frame, 0, "static", mask_name="type")

        # Two separate writers should have been created
        self.assertEqual(mock_writer_cls.call_count, 2)
        paths = [call.kwargs["output_path"] for call in mock_writer_cls.call_args_list]
        self.assertTrue(str(paths[0]).endswith(".static.mask.mp4"))
        self.assertTrue(str(paths[1]).endswith(".static.type_mask.mp4"))

    def test_release_closes_all_mask_writers(self):
        """release() should close overlay writer and all named mask video writers."""
        mock_writer_a = MagicMock()
        mock_writer_b = MagicMock()
        self.vis_helper._mask_video_writers = {"default": mock_writer_a, "type": mock_writer_b}

        self.vis_helper.release()

        mock_writer_a.release.assert_called_once()
        mock_writer_b.release.assert_called_once()

    @patch("cv2.findContours")
    @patch("cv2.moments")
    def test_draw_filter_marker_distance(self, mock_moments, mock_findcontours):
        """Test drawing filter marker for distance filter."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        track_id = 123
        filter_info = {"reason": "distance_threshold_m (50.0m)", "type": "importance"}

        # Mock cuboid and contours
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        tracked_object = {"geometry": mock_cuboid}

        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_moments.return_value = {"m00": 10000, "m10": 1500000, "m01": 1500000}  # Center at (150, 150)

        with patch("cv2.circle") as mock_circle:
            result = self.vis_helper._draw_filter_marker(
                vis_image, tracked_object, camera_pose, camera_model, track_id, filter_info
            )

        # Should draw red dot for distance filter
        mock_circle.assert_called_once_with(vis_image, (150, 150), 2, (255, 0, 0), -1)

        # Verify the result
        self.assertIs(result, vis_image)

    @patch("cv2.findContours")
    @patch("cv2.moments")
    def test_draw_filter_marker_oncoming(self, mock_moments, mock_findcontours):
        """Test drawing filter marker for oncoming filter."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        track_id = 123
        filter_info = {"reason": "skip_oncoming_obstacles (enabled)", "type": "importance"}

        # Mock cuboid and contours
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        tracked_object = {"geometry": mock_cuboid}

        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_moments.return_value = {"m00": 10000, "m10": 1500000, "m01": 1500000}  # Center at (150, 150)

        with patch("cv2.line") as mock_line:
            result = self.vis_helper._draw_filter_marker(
                vis_image, tracked_object, camera_pose, camera_model, track_id, filter_info
            )

        # Should draw blue cross for oncoming filter
        self.assertEqual(mock_line.call_count, 2)  # Two lines for cross

        # Verify the result
        self.assertIs(result, vis_image)

    def test_write_frame_success(self):
        """Test successful frame writing."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        # Mock video writer
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        self.vis_helper.video_writer = mock_writer
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with patch("cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertTrue(result)
        mock_writer.write.assert_called_once()
        mock_cvtcolor.assert_called_once()

    def test_write_frame_no_writer(self):
        """Test frame writing when no video writer is available."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertFalse(result)

    def test_write_frame_resize_needed(self):
        """Test frame writing when frame needs resizing."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Wrong size
        frame_idx = 5

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        self.vis_helper.video_writer = mock_writer
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with patch("cv2.resize") as mock_resize, patch("cv2.cvtColor") as mock_cvtcolor:
            mock_resize.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_cvtcolor.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertTrue(result)
        mock_resize.assert_called_once_with(frame, (1280, 720))

    def test_save_frame_image(self):
        """Test saving frame as individual image."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        with patch("cv2.imwrite") as mock_imwrite, patch("cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.return_value = frame
            mock_imwrite.return_value = True

            result = self.vis_helper.save_frame_image(frame, frame_idx)

        self.assertTrue(result)
        mock_imwrite.assert_called_once()

        # Check that filename contains frame index
        call_args = mock_imwrite.call_args[0]
        filename = call_args[0]
        self.assertIn("0005", filename)  # Frame index formatted with zero padding

    def test_release(self):
        """Test releasing video resources."""
        # Mock video writer and capture
        mock_writer = MagicMock()
        mock_capture = MagicMock()
        self.vis_helper.video_writer = mock_writer
        self.vis_helper.original_video_cap = mock_capture

        self.vis_helper.release()

        mock_writer.release.assert_called_once()
        mock_capture.release.assert_called_once()

    def test_release_no_resources(self):
        """Test releasing when no resources are allocated."""
        # Should not raise an exception
        self.vis_helper.release()

    def test_create_frame_visualization_integration(self):
        """Test the complete frame visualization pipeline."""
        frame_idx = 10
        segmentation_masks = MagicMock()
        segmentation_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        frame_scores = {123: 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        mock_cuboid = MagicMock()
        mock_cuboid.get_projected_mask.return_value = (
            np.zeros((720, 1280), dtype=bool),
            np.full((720, 1280), np.inf, dtype=np.float32),
        )
        frame_objects = {123: {"geometry": mock_cuboid}}
        filtered_objects = {}

        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with (
            patch.object(self.vis_helper, "get_original_frame") as mock_get_frame,
            patch.object(self.vis_helper, "_overlay_segmentation") as mock_overlay,
            patch.object(self.vis_helper, "_add_object_overlays") as mock_add_overlays,
            patch.object(self.vis_helper, "add_legend") as mock_add_legend,
        ):
            mock_base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_get_frame.return_value = mock_base_image
            mock_overlay.return_value = mock_base_image
            mock_add_overlays.return_value = mock_base_image
            mock_add_legend.return_value = mock_base_image

            result = self.vis_helper.create_frame_visualization(
                frame_idx, segmentation_masks, frame_scores, camera_pose, camera_model, frame_objects, filtered_objects
            )

        # Should call all steps in the pipeline
        mock_get_frame.assert_called_once_with(frame_idx)
        mock_overlay.assert_called_once()
        mock_add_overlays.assert_called_once()
        mock_add_legend.assert_called_once_with(mock_base_image, frame_idx, self.vis_helper.legend_items)

        self.assertIsNotNone(result)

    def test_create_frame_visualization_with_scene_rasterizer(self):
        """Test frame visualization pipeline with pre-computed SceneRasterizer."""
        frame_idx = 10
        segmentation_masks = MagicMock()
        segmentation_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        frame_scores = {"123": 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.full((720, 1280), np.inf, dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 5.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {"123": {"geometry": mock_cuboid}}
        filtered_objects = {}

        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        # Create a SceneRasterizer (computes visibility masks in constructor)
        scene_rasterizer = SceneRasterizer(
            frame_objects, camera_pose, camera_model, 1280, 720, logger=logging.getLogger("test")
        )

        with (
            patch.object(self.vis_helper, "get_original_frame") as mock_get_frame,
            patch.object(self.vis_helper, "_overlay_segmentation") as mock_overlay,
            patch.object(self.vis_helper, "_add_object_overlays") as mock_add_overlays,
            patch.object(self.vis_helper, "add_legend") as mock_add_legend,
        ):
            mock_base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_get_frame.return_value = mock_base_image
            mock_overlay.return_value = mock_base_image
            mock_add_overlays.return_value = mock_base_image
            mock_add_legend.return_value = mock_base_image

            result = self.vis_helper.create_frame_visualization(
                frame_idx,
                segmentation_masks,
                frame_scores,
                camera_pose,
                camera_model,
                frame_objects,
                filtered_objects,
                scene_rasterizer=scene_rasterizer,
            )

        # Should pass scene_rasterizer to _add_object_overlays
        mock_add_overlays.assert_called_once()
        call_kwargs = mock_add_overlays.call_args[1]
        self.assertIn("scene_rasterizer", call_kwargs)
        self.assertIs(call_kwargs["scene_rasterizer"], scene_rasterizer)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
