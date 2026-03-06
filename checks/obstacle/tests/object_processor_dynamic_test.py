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

"""Unit tests for dynamic obstacle processor: ObjectProcessorDynamic."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.obstacle.object_processor_dynamic import ObjectProcessorDynamic


class TestObjectProcessorDynamic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_dyn"
        self.config = {
            "av.obstacle": {
                "overlap_check": {"vehicle": {}, "pedestrian": {}},
                "importance_filter": {"distance_threshold_m": 50.0},
                "hallucination_detector": {
                    "enabled": True,
                    "classes": {"vehicle": {"min_cluster_area": 5000}, "pedestrian": {"min_cluster_area": 1000}},
                    "max_cluster_per_frame": 100,
                },
            }
        }

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("checks.obstacle.object_processor_dynamic.SegHelper")
    @patch("checks.obstacle.object_processor_dynamic.OverlapDetector")
    @patch("checks.obstacle.object_processor_dynamic.HallucinationDetector")
    def test_init_components(self, mock_hall, mock_overlap, mock_seg):
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        mock_seg.assert_called_once()
        mock_overlap.assert_called_once()
        mock_hall.assert_called_once()
        self.assertIsNotNone(getattr(p, "hallucination_detector", None))

    @patch("checks.obstacle.object_processor_dynamic.SceneRasterizer")
    @patch("checks.obstacle.object_processor_dynamic.extract_rpy_in_flu", return_value=(0.0, 0.0, 0.0))
    def test_process_frame_calls_overlap_and_hallucination(self, _mock_rpy, _mock_rasterizer):
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        # Plug in mocks
        p.seg_model = MagicMock()
        p.seg_model.process_frame.return_value = MagicMock()
        p.seg_model.resize_masks.return_value = MagicMock()
        p.overlap_detector = MagicMock()
        # Each class returns its own scores
        p.overlap_detector.process_class.side_effect = [({1: 0.6}, {}, {}), ({3: 0.9}, {}, {})]

        p.hallucination_detector = MagicMock()
        with patch("checks.obstacle.object_processor_dynamic.track_hallucinations") as mock_track:
            res = {
                "hallucination_detections": {
                    "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}],
                    "pedestrian": [],
                    "motorcycle": [],
                    "bicycle": [],
                }
            }
            p.hallucination_detector.detect.return_value = res["hallucination_detections"]
            data_loader = MagicMock(CAMERA_RESCALED_RESOLUTION_HEIGHT=480, CAMERA_RESCALED_RESOLUTION_WIDTH=640)
            data_loader.get_object_data_for_frame.return_value = {"t1": {}}
            scores = p._process_frame(
                frame_idx=0,
                frame=MagicMock(),
                data_loader=data_loader,
                camera_pose=np.eye(4),
                camera_model=MagicMock(),
                track_output_agg={},
                results=res,
            )
            self.assertEqual(scores, {1: 0.6, 3: 0.9})
            mock_track.assert_called()

    @patch("checks.obstacle.object_processor_dynamic.SceneRasterizer")
    def test_process_frames_dynamic_iteration_and_mapping(self, _mock_rasterizer):
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({2: 0.5}, {}, {})
        p.seg_model = MagicMock()
        p.seg_model.process_frame.return_value = MagicMock()
        p.seg_model.resize_masks.return_value = MagicMock()

        # Build a dataloader of 3 frames
        video_dataloader = [MagicMock(), MagicMock(), MagicMock()]
        camera_poses = [np.eye(4) for _ in range(3)]
        data_loader = MagicMock()
        data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        data_loader.object_count = 100

        # Provide objects only for mapped indices 0 and 2; 1 has none (skip)
        # data_loader.get_object_data_for_frame(idx, include_static=False) is used for skipping
        def get_object_data_side_effect(idx, include_static=False):
            objects_by_frame = {0: {"t": {}}, 2: {"t": {}}}
            return objects_by_frame.get(idx, {})

        data_loader.get_object_data_for_frame.side_effect = get_object_data_side_effect

        results = p._init_results_container(total_video_frames=len(video_dataloader))
        track_output_agg = {}
        score_matrix, skipped = p._process_frames_dynamic(
            video_dataloader=video_dataloader,
            camera_poses=camera_poses,
            camera_model=MagicMock(),
            data_loader=data_loader,
            target_fps=30.0,
            video_fps=30.0,
            trial_frames=None,
            results=results,
            track_output_agg=track_output_agg,
        )
        self.assertEqual(skipped, 1)
        self.assertIn(0, results["processed_frame_ids"])  # first frame processed
        self.assertIn(2, results["processed_frame_ids"])  # third frame processed
        self.assertFalse(np.isnan(score_matrix[0, 2]))
        self.assertTrue(np.isnan(score_matrix[1, 2]))

    @patch("checks.obstacle.object_processor_dynamic.SceneRasterizer")
    @patch("checks.obstacle.object_processor_base.RdsDataLoader")
    def test_process_clip_end_to_end_mocked(self, mock_loader_cls, _mock_rasterizer):
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )

        # Mock loader
        loader = MagicMock()
        loader.get_camera_poses.return_value = [np.eye(4) for _ in range(2)]
        loader.get_camera_intrinsics.return_value = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        loader.object_count = 100
        mock_loader_cls.return_value = loader

        # Mock seg model dataloader
        p.seg_model = MagicMock()
        mock_video_dataloader = [MagicMock(), MagicMock()]
        mock_dataset = MagicMock()
        p.seg_model.get_video_dataloader.return_value = (mock_video_dataloader, mock_dataset)

        # Ensure processing returns a score for first frame only
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({5: 0.8}, {}, {})
        loader.get_object_data_for_frame.side_effect = lambda idx, include_static=False: {"t": {}} if idx == 0 else {}

        p.profiler = MagicMock()
        res = p.process_clip(
            input_data="/data",
            clip_id="c1",
            camera_name="front",
            video_path="/vid.mp4",
            output_dir=self.temp_dir,
        )
        self.assertIn("processed_frames", res)
        self.assertIn("score_matrix", res)


if __name__ == "__main__":
    unittest.main()
