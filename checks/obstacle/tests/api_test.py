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

"""Unit tests for api module."""

import unittest
from unittest.mock import MagicMock, patch

from checks.obstacle.api import run_object_processors as run


class TestRunObjectProcessors(unittest.TestCase):
    """Test case for the run_object_processors() function."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_args = {
            "input_data": "/path/to/data",
            "clip_id": "test_clip",
            "camera_name": "front_camera",
            "video_path": "/path/to/video.mp4",
            "output_dir": "/path/to/output",
            "model_device": "cpu",
            "verbose": "INFO",
            "trial_frames": 5,
            "target_fps": 30.0,
        }
        self.mock_dynamic_result = {"processed_frames": 10, "mean_score": 0.85}

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_only_dynamic_objects(self, mock_dynamic_cls):
        """Test that only dynamic processor is run when only dynamic objects are configured."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                }
            }
        }

        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()

        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result["dynamic"], self.mock_dynamic_result)

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_only_static_objects(self, mock_dynamic_cls):
        """Test that static objects are recognized but produce no results (not yet supported)."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertNotIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_both_object_types(self, mock_dynamic_cls):
        """Test that dynamic processor runs; static objects are recognized but not yet processed."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()

        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result["dynamic"], self.mock_dynamic_result)

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_static_objects_without_world_video_path(self, mock_dynamic_cls):
        """Test that no processors run when only static objects are configured and world_video_path is None."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        result = run(config=config, world_video_path=None, **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_empty_overlap_check(self, mock_dynamic_cls):
        """Test that no processors are run when overlap_check is empty."""
        config = {"av.obstacle": {"overlap_check": {}}}

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_missing_config_sections(self, mock_dynamic_cls):
        """Test that no processors are run when config sections are missing."""
        config = {}

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_all_dynamic_object_types(self, mock_dynamic_cls):
        """Test with all dynamic object types: bicycle, motorcycle, pedestrian, vehicle."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "bicycle": {"method": "cluster"},
                    "motorcycle": {"method": "cluster"},
                    "pedestrian": {"method": "cluster"},
                    "vehicle": {"method": "ratio"},
                }
            }
        }

        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()

        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_all_static_object_types(self, mock_dynamic_cls):
        """Test with all static object types: no results since static processing is not yet supported."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "crosswalk": {"method": "ratio"},
                    "lane_line": {"method": "ratio"},
                    "road_boundary": {"method": "ratio"},
                    "traffic_light": {"method": "ratio"},
                    "traffic_sign": {"method": "ratio"},
                    "wait_line": {"method": "ratio"},
                }
            }
        }

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertNotIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_with_unknown_objects_only(self, mock_dynamic_cls):
        """Test that no processors are run when only unknown object types are configured."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "unknown_object": {"method": "ratio"},
                    "another_unknown": {"method": "cluster"},
                }
            }
        }

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_dynamic_cls.assert_not_called()
        self.assertEqual(result, {})

    @patch("checks.obstacle.api.ObjectProcessorDynamic")
    def test_run_processor_arguments(self, mock_dynamic_cls):
        """Test that the dynamic processor is initialized and called with correct arguments."""
        config = {
            "av.obstacle": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                }
            }
        }

        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        world_video_path = "/path/to/world.mp4"

        run(config=config, world_video_path=world_video_path, **self.base_args)

        mock_dynamic_cls.assert_called_once_with(
            config=config,
            model_device=self.base_args["model_device"],
            verbose=self.base_args["verbose"],
            clip_id=self.base_args["clip_id"],
            output_dir=self.base_args["output_dir"],
        )

        mock_dynamic_instance.process_clip.assert_called_once_with(
            input_data=self.base_args["input_data"],
            clip_id=self.base_args["clip_id"],
            camera_name=self.base_args["camera_name"],
            video_path=self.base_args["video_path"],
            output_dir=self.base_args["output_dir"],
            trial_frames=self.base_args["trial_frames"],
            target_fps=self.base_args["target_fps"],
        )


if __name__ == "__main__":
    unittest.main()
