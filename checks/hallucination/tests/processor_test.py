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

"""Unit tests for the hallucination processor."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.hallucination.processor import HallucinationProcessor


class _FakeCapture:
    """Small test double for cv2.VideoCapture."""

    def __init__(self, opened: bool, frames: list[tuple[bool, np.ndarray | None]]):
        self._opened = opened
        self._frames = frames
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - matches cv2 API
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames:
            return self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class TestHallucinationProcessor(unittest.TestCase):
    def test_process_real_videos_smoke(self) -> None:
        """Real-data smoke test to guard against accidental API regressions."""
        processor = HallucinationProcessor({})
        result = processor.process(
            clip_id="test_clip_id",
            augmented_video_path="./checks/sample_data/augmented.mp4",
            original_video_path="./checks/sample_data/original.mp4",
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.score, 0.7826, delta=0.0001)

    @patch("checks.hallucination.processor.HallucinationProcessor.get_default_config")
    def test_init_overrides_and_invalid_threshold(self, mock_default_config: MagicMock) -> None:
        mock_default_config.return_value = {"metropolis.hallucination": {"threshold": 0.2, "max_frames": 5}}

        processor = HallucinationProcessor({"threshold": 0.8, "max_frames": 2})
        self.assertEqual(processor.threshold, 0.8)
        self.assertEqual(processor.max_frames, 2)

        with self.assertRaises(ValueError):
            HallucinationProcessor({"threshold": 1.2})

    @patch("checks.hallucination.processor.os.remove")
    @patch("checks.hallucination.processor.os.path.exists")
    @patch("checks.hallucination.processor.cv2.VideoCapture")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_cleanup_paths_and_re_raise(
        self,
        mock_download_if_remote: MagicMock,
        mock_video_capture: MagicMock,
        mock_exists: MagicMock,
        mock_remove: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})

        orig_cap = _FakeCapture(opened=False, frames=[])
        aug_cap = _FakeCapture(opened=True, frames=[])
        mock_video_capture.side_effect = [orig_cap, aug_cap]
        mock_download_if_remote.side_effect = ["/tmp/orig.mp4", "/tmp/aug.mp4"]
        mock_exists.return_value = True
        mock_remove.side_effect = [None, OSError("remove failed")]

        with self.assertRaises(Exception):
            processor.process("clip", "s3://bucket/orig.mp4", "s3://bucket/aug.mp4")

        self.assertEqual(mock_remove.call_count, 2)

    @patch("checks.hallucination.processor.cv2.VideoCapture")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_raises_when_augmented_video_fails_to_open(
        self,
        mock_download_if_remote: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        mock_download_if_remote.side_effect = ["orig.mp4", "aug.mp4"]

        orig_cap = _FakeCapture(opened=True, frames=[(True, np.zeros((2, 2, 3), dtype=np.uint8))])
        aug_cap = _FakeCapture(opened=False, frames=[])
        mock_video_capture.side_effect = [orig_cap, aug_cap]

        with self.assertRaises(Exception):
            processor.process("clip", "orig.mp4", "aug.mp4")

    @patch("checks.hallucination.processor.cv2.VideoCapture")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_raises_when_first_frame_read_fails(
        self,
        mock_download_if_remote: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        mock_download_if_remote.side_effect = ["orig.mp4", "aug.mp4"]

        orig_cap = _FakeCapture(opened=True, frames=[(False, None)])
        aug_cap = _FakeCapture(opened=True, frames=[(True, np.zeros((2, 2, 3), dtype=np.uint8))])
        mock_video_capture.side_effect = [orig_cap, aug_cap]
        with self.assertRaises(Exception):
            processor.process("clip", "orig.mp4", "aug.mp4")

        orig_cap = _FakeCapture(opened=True, frames=[(True, np.zeros((2, 2, 3), dtype=np.uint8))])
        aug_cap = _FakeCapture(opened=True, frames=[(False, None)])
        mock_video_capture.side_effect = [orig_cap, aug_cap]
        with self.assertRaises(Exception):
            processor.process("clip", "orig.mp4", "aug.mp4")

    @patch("checks.hallucination.processor.hallucination_counts")
    @patch("checks.hallucination.processor.compute_dynamic_mask")
    @patch("checks.hallucination.processor.to_gray")
    @patch("checks.hallucination.processor.ensure_same_size")
    @patch("checks.hallucination.processor.cv2.VideoCapture")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_scores_with_and_without_augmented_dynamic_pixels(
        self,
        mock_download_if_remote: MagicMock,
        mock_video_capture: MagicMock,
        mock_ensure_same_size: MagicMock,
        mock_to_gray: MagicMock,
        mock_compute_dynamic_mask: MagicMock,
        mock_hallucination_counts: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({"threshold": 0.75, "max_frames": 2})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        mock_download_if_remote.side_effect = lambda path: path
        mock_ensure_same_size.side_effect = lambda f, _shape: f
        mock_to_gray.side_effect = lambda f: np.zeros((4, 4), dtype=np.uint8)
        mock_compute_dynamic_mask.side_effect = [
            (np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)),
            (np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)),
        ]
        mock_hallucination_counts.return_value = (2, 10)

        orig_cap = _FakeCapture(opened=True, frames=[(True, frame), (True, frame), (False, None)])
        aug_cap = _FakeCapture(opened=True, frames=[(True, frame), (True, frame), (False, None)])
        mock_video_capture.side_effect = [orig_cap, aug_cap]

        result = processor.process("clip", "orig.mp4", "aug.mp4")
        self.assertAlmostEqual(result.score, 0.8)
        self.assertTrue(result.passed)
        self.assertEqual(result.total_frames, 2)

        # Zero dynamic pixels path: score forced to 1.0.
        mock_hallucination_counts.return_value = (0, 0)
        orig_cap = _FakeCapture(opened=True, frames=[(True, frame), (False, None)])
        aug_cap = _FakeCapture(opened=True, frames=[(True, frame), (False, None)])
        mock_video_capture.side_effect = [orig_cap, aug_cap]
        result = processor.process("clip2", "orig.mp4", "aug.mp4")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)

    @patch("checks.hallucination.processor.ConfigManager")
    def test_get_default_config_success_and_failure(self, mock_config_manager: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.load_config.return_value = {"metropolis.hallucination": {"threshold": 0.9}}
        mock_config_manager.return_value = mock_instance

        config = HallucinationProcessor.get_default_config()
        self.assertEqual(config["metropolis.hallucination"]["threshold"], 0.9)

        mock_instance.load_config.side_effect = RuntimeError("bad config")
        with self.assertRaises(RuntimeError):
            HallucinationProcessor.get_default_config()


if __name__ == "__main__":
    unittest.main()
