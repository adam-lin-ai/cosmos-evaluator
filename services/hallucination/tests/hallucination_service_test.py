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

import unittest
import unittest.mock as mock

from checks.hallucination.processor import HallucinationResult
from services.hallucination.hallucination_service import HallucinationRequest, HallucinationService


class TestHallucinationService(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = HallucinationService()

    async def test_validate_input_fails(self) -> None:
        """Test that validation fails when paths don't exist."""
        request = HallucinationRequest(
            clip_id="test_clip_id",
            original_video_path="test_original_video_path",
            augmented_video_path="test_augmented_video_path",
        )
        with self.assertRaises(ValueError):
            await self.service.validate_input(request)

    async def test_validate_input_passes(self) -> None:
        """Test that validation passes when both paths exist and are accessible."""
        request = HallucinationRequest(
            clip_id="test_clip_id",
            original_video_path="s3://bucket/original_video.mp4",
            augmented_video_path="s3://bucket/augmented_video.mp4",
        )

        with mock.patch("services.hallucination.hallucination_service.validate_uri", return_value=True):
            result = await self.service.validate_input(request)
            self.assertTrue(result)

    async def test_validate_input_fails_on_augmented_path(self) -> None:
        """Test that validation fails when augmented path doesn't exist."""
        request = HallucinationRequest(
            clip_id="test_clip_id",
            original_video_path="/tmp/original_video.mp4",
            augmented_video_path="/tmp/augmented_video.mp4",
        )
        with mock.patch(
            "services.hallucination.hallucination_service.validate_uri",
            side_effect=[True, False],
        ):
            with self.assertRaises(ValueError):
                await self.service.validate_input(request)

    async def test_get_default_config_fallback_empty(self) -> None:
        """Test config extraction fallback when key is missing."""
        with mock.patch(
            "services.hallucination.hallucination_service.HallucinationProcessor.get_default_config",
            return_value={},
        ):
            cfg = await HallucinationService.get_default_config()
            self.assertEqual(cfg, {})

    async def test_process_uses_default_config(self) -> None:
        """Test process uses default config and forwards args."""
        request = HallucinationRequest(
            clip_id="clip-a",
            original_video_path="orig.mp4",
            augmented_video_path="aug.mp4",
            config=None,
            verbose="DEBUG",
        )
        expected = HallucinationResult(
            clip_id="clip-a",
            passed=True,
            threshold=0.5,
            score=0.1,
            total_frames=10,
            total_hallucinated_dynamic_pixels=1,
            total_augmented_dynamic_pixels=100,
        )
        with (
            mock.patch.object(HallucinationService, "get_default_config", return_value={"threshold": 0.5}),
            mock.patch("services.hallucination.hallucination_service.HallucinationProcessor") as mock_processor_cls,
        ):
            mock_processor_cls.return_value.process.return_value = expected
            result = await self.service.process(request)

        self.assertEqual(result, expected)
        mock_processor_cls.assert_called_once_with(params={"threshold": 0.5}, config_dir=None, verbose="DEBUG")
        mock_processor_cls.return_value.process.assert_called_once_with("clip-a", "orig.mp4", "aug.mp4")

    async def test_process_merges_override_config(self) -> None:
        """Test request config overrides default config values."""
        request = HallucinationRequest(
            clip_id="clip-b",
            original_video_path="orig.mp4",
            augmented_video_path="aug.mp4",
            config={"threshold": 0.9},
        )
        expected = HallucinationResult(
            clip_id="clip-b",
            passed=True,
            threshold=0.9,
            score=0.1,
            total_frames=10,
            total_hallucinated_dynamic_pixels=1,
            total_augmented_dynamic_pixels=100,
        )
        with (
            mock.patch.object(
                HallucinationService, "get_default_config", return_value={"enabled": True, "threshold": 0.5}
            ),
            mock.patch("services.hallucination.hallucination_service.HallucinationProcessor") as mock_processor_cls,
        ):
            mock_processor_cls.return_value.process.return_value = expected
            await self.service.process(request)

        mock_processor_cls.assert_called_once_with(
            params={"enabled": True, "threshold": 0.9}, config_dir=None, verbose="INFO"
        )


if __name__ == "__main__":
    unittest.main()
