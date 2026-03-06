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

"""Unit tests for static obstacle processor: ObjectProcessorStatic."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.obstacle.object_processor_static import ObjectProcessorStatic


class TestObjectProcessorStatic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_static"
        self.config = {
            "av.obstacle": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform", "use_decord": False},
            }
        }

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("checks.obstacle.object_processor_static.get_object_type_colors", return_value={})
    @patch("checks.obstacle.object_processor_static.get_object_type_class_mapping", return_value=None)
    @patch(
        "checks.obstacle.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.obstacle.object_processor_static.CWIPInferenceHelper")
    @patch("checks.obstacle.object_processor_static.OverlapDetector")
    def test_static_init_components(self, mock_overlap, mock_cwip, _mock_runfiles, _mock_mapping, _mock_colors):
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        mock_cwip.from_hf_format.assert_called()
        mock_overlap.assert_called()
        # set_mask_styles should be called with similarity legend
        self.assertTrue(p.vis_helper.mask_colors.get("MATCH") is not None)
        # type mask legend config should also be set
        self.assertIn("type", p.vis_helper._mask_legend_configs)

    @patch("checks.obstacle.object_processor_static.get_runfiles_path", return_value=None)
    def test_static_init_missing_model_path(self, _mock_runfiles):
        with self.assertRaises(RuntimeError):
            ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            )

    @patch("checks.obstacle.object_processor_static.get_object_type_colors", return_value={})
    @patch("checks.obstacle.object_processor_static.get_object_type_class_mapping", return_value=None)
    @patch(
        "checks.obstacle.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.obstacle.object_processor_static.CWIPInferenceHelper")
    @patch("checks.obstacle.object_processor_static.OverlapDetector")
    @patch("checks.obstacle.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.obstacle.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.obstacle.object_processor_base.RdsDataLoader")
    def test_process_frames_static_and_clip(
        self,
        mock_loader_cls,
        mock_run_cwip,
        _mock_fps,
        _mock_overlap,
        _mock_cwip,
        _mock_runfiles,
        _mock_mapping,
        _mock_colors,
    ):
        # Mock CWIP outputs: 3 frames of class indices
        pred_similarity = torch.zeros((3, 480, 640), dtype=torch.long)
        pred_type = torch.zeros((3, 480, 640), dtype=torch.long)
        mock_run_cwip.return_value = ({"num_frames": 3}, pred_similarity, pred_type)

        # Mock loader
        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.get_camera_poses.return_value = [np.eye(4) for _ in range(3)]
        loader.get_camera_intrinsics.return_value = MagicMock()
        loader.traffic_cuboids_all_frames = [{}, {}, {}]
        mock_loader_cls.return_value = loader

        processor = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        processor.vis_helper = MagicMock()
        processor.vis_helper.frame_width = 640
        processor.vis_helper.frame_height = 480
        processor.overlap_detector.process_class.side_effect = [({1: 0.2}, {}, {}), ({1: 0.3}, {}, {})] * 3
        result = processor.process_clip(
            input_data="/data",
            clip_id=self.clip_id,
            camera_name="front",
            camera_video_path="/cam.mp4",
            world_video_path="/world.mp4",
        )
        self.assertIn("processed_frames", result)
        self.assertGreaterEqual(result["processed_frames"], 1)

    @patch("checks.obstacle.object_processor_static.get_video_fps", return_value=None)
    @patch("checks.obstacle.object_processor_base.RdsDataLoader")
    def test_process_clip_static_raises_when_no_fps(self, _mock_loader, _mock_fps):
        with self.assertRaises(RuntimeError):
            ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            ).process_clip(
                input_data="/data",
                clip_id="cS",
                camera_name="front",
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                output_dir=self.temp_dir,
            )


class TestObjectProcessorStaticVisualizationDisabled(unittest.TestCase):
    """Tests for ObjectProcessorStatic when visualization is disabled."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_static_no_vis"
        self.config = {
            "av.obstacle": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform", "use_decord": False},
                "visualization": {"enabled": False},
            }
        }

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("checks.obstacle.object_processor_static.get_object_type_colors", return_value={})
    @patch("checks.obstacle.object_processor_static.get_object_type_class_mapping", return_value=None)
    @patch(
        "checks.obstacle.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.obstacle.object_processor_static.CWIPInferenceHelper")
    @patch("checks.obstacle.object_processor_static.OverlapDetector")
    def test_init_succeeds_when_visualization_disabled(
        self, mock_overlap, mock_cwip, _mock_runfiles, _mock_mapping, _mock_colors
    ):
        """Test that ObjectProcessorStatic initializes without error when visualization is disabled.

        This tests the fix for the 'NoneType' object has no attribute 'set_mask_styles' error
        that occurred when visualization was disabled but set_mask_styles was called unconditionally.
        """
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )

        self.assertIsNone(p.vis_helper)

        mock_cwip.from_hf_format.assert_called()
        mock_overlap.assert_called()


if __name__ == "__main__":
    unittest.main()
