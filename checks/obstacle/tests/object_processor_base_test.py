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

"""Unit tests for base obstacle processor: ObjectProcessorBase."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.obstacle.object_processor_base import ObjectProcessorBase


class TestObjectProcessorBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_001"
        self.basic_config = {
            "av.obstacle": {
                "overlap_check": {"vehicle": {}},
                "importance_filter": {"distance_threshold_m": 30.0},
            }
        }

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_logging_console_only_no_file(self):
        # With required clip_id/output_dir, file handler is created but we can verify console handler exists
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.setup_logging()
        # Verify logger has handlers (console handler should exist)
        self.assertGreater(len(proc.logger.handlers), 0)

    @patch("checks.utils.onnx.configure_onnx_logging")
    def test_logging_with_file_and_debug(self, mock_onnx):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id,
            output_dir=self.temp_dir,
            config=self.basic_config,
            model_device="cpu",
            verbose="DEBUG",
        )
        proc.setup_logging()
        log_path = os.path.join(self.temp_dir, f"{self.clip_id}.base.object.log")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotEqual(content.strip(), "")
        mock_onnx.assert_called_once_with(verbose=True)

    def test_init_results_container_structure(self):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        container = proc._init_results_container(total_video_frames=42)
        self.assertEqual(container["processed_frames"], 0)
        self.assertEqual(container["total_video_frames"], 42)
        self.assertIsNone(container["score_matrix"])
        self.assertIsInstance(container["track_ids"], set)
        self.assertIsInstance(container["processed_frame_ids"], list)
        self.assertEqual(container["hallucination_detections"], {})  # base class has no hallucination classes

    def test_finalize_and_summarize_trims_and_stats(self):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        # Build results as during processing
        results = proc._init_results_container(total_video_frames=5)
        results["processed_frame_ids"] = [0, 2]
        results["track_ids"] = {1, 3}

        # Score matrix before trimming (5 frames X 6 possible IDs)
        score_matrix = np.full((5, 6), np.nan)
        score_matrix[0, 1] = 0.5
        score_matrix[2, 3] = 0.9

        # Aggregate per-track outputs (dynamic objects without object_type_index)
        track_output_agg = {
            1: {"object_types": {"vehicle"}, "output_counts": {"scored": 2, "occluded": 1, "extra": 7}},
            3: {"object_types": {"pedestrian"}, "output_counts": {"extra": 4, "scored": 3}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=1,
            total_video_frames=5,
        )

        # Check trimming
        trimmed = results["score_matrix"]
        self.assertEqual(trimmed.shape, (2, 2))  # frames [0,2] X track IDs 1 and 3
        self.assertTrue(np.isnan(trimmed[0, 1]))  # column 1 corresponds to track_id=3; row 0 has NaN
        self.assertAlmostEqual(trimmed[1, 1], 0.9)

        # Check statistics exist
        self.assertIn("mean_score", results)
        self.assertIn("std_score", results)

        # Labels order: "scored", "occluded", then others sorted
        self.assertEqual(results["processor_output_labels"][:2], ["scored", "occluded"])
        self.assertIn("tracks", results)
        self.assertEqual(len(results["tracks"]), 2)

        # Dynamic objects should not have object_type_index
        for track in results["tracks"]:
            self.assertNotIn("object_type_index", track)

    def test_finalize_and_summarize_with_object_type_index(self):
        """Test that object_type_index is included in tracks for static objects."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        # Build results as during processing
        results = proc._init_results_container(total_video_frames=3)
        results["processed_frame_ids"] = [0, 1, 2]
        results["track_ids"] = {100, 101}

        # Score matrix (3 frames X 102 possible IDs to accommodate track IDs 100, 101)
        score_matrix = np.full((3, 102), np.nan)
        score_matrix[0, 100] = 0.8
        score_matrix[1, 100] = 0.85
        score_matrix[2, 101] = 0.9

        # Aggregate per-track outputs (static objects with object_type_index)
        track_output_agg = {
            100: {
                "object_types": {"LaneLine"},
                "object_type_index": 19,  # GEOMETRY_LANELINE_WHITE_SOLID_GROUP
                "output_counts": {"scored": 2},
            },
            101: {
                "object_types": {"TrafficLight"},
                "object_type_index": 8,  # TRAFFIC_LIGHT_RED
                "output_counts": {"scored": 1},
            },
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=3,
        )

        # Check tracks have object_type_index
        self.assertIn("tracks", results)
        self.assertEqual(len(results["tracks"]), 2)

        # Find tracks by track_id
        track_100 = next(t for t in results["tracks"] if t["track_id"] == 100)
        track_101 = next(t for t in results["tracks"] if t["track_id"] == 101)

        # Verify object_type_index is present and correct
        self.assertIn("object_type_index", track_100)
        self.assertEqual(track_100["object_type_index"], 19)
        self.assertEqual(track_100["object_type"], "LaneLine")

        self.assertIn("object_type_index", track_101)
        self.assertEqual(track_101["object_type_index"], 8)
        self.assertEqual(track_101["object_type"], "TrafficLight")

    def test_maybe_init_visualization_calls_helper(self):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.vis_helper = MagicMock()
        dl = MagicMock()
        dl.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        dl.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        proc._maybe_init_visualization("test_fixtures/video.mp4", dl)
        proc.vis_helper.initialize_video_writer.assert_called()

    @patch("checks.obstacle.object_processor_base.get_dataloader")
    def test_prepare_video_prefers_seg_helper(self, mock_get_dataloader):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        # Mock seg_model with custom get_video_dataloader
        seg = MagicMock()
        mock_dl = MagicMock()
        mock_ds = MagicMock()
        mock_ds.number_frames = 10
        mock_ds.frames_per_second = 20
        mock_dl.__len__ = MagicMock(return_value=10)
        seg.get_video_dataloader.return_value = (mock_dl, mock_ds)
        proc.seg_model = seg

        dataloader, video_fps, target_fps = proc._prepare_video("test_fixtures/video.mp4", MagicMock(), None)
        self.assertIs(dataloader, mock_dl)
        self.assertEqual(video_fps, 20.0)
        self.assertEqual(target_fps, 30.0)
        mock_get_dataloader.assert_not_called()

    @patch("checks.obstacle.object_processor_base.get_dataloader")
    def test_prepare_video_fallback_when_no_helper(self, mock_get_dataloader):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        proc.seg_model = MagicMock()
        proc.seg_model.get_video_dataloader = None
        del proc.seg_model.get_video_dataloader

        mock_dl = MagicMock()
        mock_ds = MagicMock()
        mock_ds.number_frames = 5
        mock_ds.frames_per_second = None
        mock_dl.__len__ = MagicMock(return_value=5)
        mock_get_dataloader.return_value = (mock_dl, mock_ds)

        dataloader, video_fps, target_fps = proc._prepare_video("test_fixtures/video.mp4", MagicMock(), 15)
        self.assertIs(dataloader, mock_dl)
        self.assertEqual(video_fps, 15.0)
        self.assertEqual(target_fps, 15.0)

    def test_create_frame_visualization_delegates(self):
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.vis_helper = MagicMock()
        proc._create_frame_visualization(
            frame_idx=0,
            segmentation_masks=MagicMock(),
            frame_scores={1: 0.7},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            frame_objects={"track1": {}},
            filtered_objects={},
            hallucinations=None,
        )
        proc.vis_helper.create_frame_visualization.assert_called()
        proc.vis_helper.write_frame.assert_called()

    def test_get_config_summary_includes_hallucination(self):
        cfg = {
            "av.obstacle": {
                "overlap_check": {},
                "importance_filter": {},
                "hallucination_detector": {
                    "enabled": True,
                    "classes": ["vehicle"],
                    "min_cluster_area": 50,
                    "max_cluster_per_frame": 5,
                },
            }
        }
        proc = ObjectProcessorBase(clip_id=self.clip_id, output_dir=self.temp_dir, config=cfg, model_device="cpu")
        summary = proc.get_config_summary()
        self.assertIn("hallucination_detector", summary)


class TestGetDefaultConfig(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            "av.obstacle": {
                "enabled": True,
                "overlap_check": {"pedestrian": {}, "vehicle": {}},
                "importance_filter": {
                    "distance_threshold_m": 100,
                    "skip_oncoming_obstacles": True,
                    "relevant_lanes": ["ego", "left", "right"],
                },
            },
            "av.vlm": {"enabled": False},
        }

    @patch("checks.obstacle.object_processor_base.ConfigManager")
    def test_get_default_config_success(self, mock_config_manager_class):
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.return_value = self.sample_config
        mock_config_manager_class.return_value = mock_config_manager

        result = ObjectProcessorBase.get_default_config()
        expected_config = {"av.obstacle": self.sample_config["av.obstacle"]}
        self.assertEqual(result, expected_config)

    @patch("checks.obstacle.object_processor_base.ConfigManager")
    def test_get_default_config_file_not_found(self, mock_config_manager_class):
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.side_effect = FileNotFoundError("Configuration file not found")
        mock_config_manager_class.return_value = mock_config_manager
        with self.assertRaises(FileNotFoundError):
            ObjectProcessorBase.get_default_config()

    @patch("checks.obstacle.object_processor_base.ConfigManager")
    @patch("checks.obstacle.object_processor_base.logging")
    def test_get_default_config_generic_exception(self, mock_logging, mock_config_manager_class):
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.side_effect = Exception("Unexpected error")
        mock_config_manager_class.return_value = mock_config_manager
        with self.assertRaises(Exception) as context:
            ObjectProcessorBase.get_default_config()
        self.assertEqual(str(context.exception), "Unexpected error")
        mock_logging.error.assert_called_once_with("Error loading default configuration: Unexpected error")


if __name__ == "__main__":
    unittest.main()
