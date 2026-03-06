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

import checks.attribute_verification.processor as processor_mod
from services.attribute_verification.attribute_verification_service import (
    AttributeVerificationRequest,
    AttributeVerificationService,
)


class TestAttributeVerificationService(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = AttributeVerificationService()

    async def test_validate_input_fails(self) -> None:
        """Test that validation fails when paths don't exist."""
        request = AttributeVerificationRequest(
            clip_id="test_clip_id",
            augmented_video_path="test_augmented_video_path",
        )
        with self.assertRaises(ValueError):
            await self.service.validate_input(request)

    async def test_validate_input_passes(self) -> None:
        """Test that validation passes when both paths exist and are accessible."""
        request = AttributeVerificationRequest(
            clip_id="test_clip_id",
            augmented_video_path="s3://bucket/augmented_video.mp4",
        )

        with mock.patch(
            "services.attribute_verification.attribute_verification_service.validate_uri",
            return_value=True,
        ):
            result = await self.service.validate_input(request)
            self.assertTrue(result)

    async def test_get_default_config_extracts_attribute_verification_config(self) -> None:
        """Default config should return only the attribute_verification section."""
        with mock.patch.object(
            processor_mod.AttributeVerificationProcessor,
            "get_default_config",
            return_value={"metropolis.attribute_verification": {"selected_variables": {"color": "red"}}},
        ):
            config = await self.service.get_default_config()
        self.assertEqual(config, {"selected_variables": {"color": "red"}})

    async def test_process_uses_default_config_when_request_config_missing(self) -> None:
        """Service process should initialize processor from default config."""
        request = AttributeVerificationRequest(
            clip_id="clip-1",
            augmented_video_path="s3://bucket/augmented_video.mp4",
        )
        expected_result = processor_mod.AttributeVerificationResult(
            clip_id="clip-1",
            passed=True,
            summary=processor_mod.AttributeVerificationSummary(total_checks=0, passed_checks=0, failed_checks=0),
            checks=[],
        )
        default_config = {"selected_variables": {"color": "red"}}

        with (
            mock.patch.object(
                AttributeVerificationService, "get_default_config", new=mock.AsyncMock(return_value=default_config)
            ),
            mock.patch(
                "services.attribute_verification.attribute_verification_service.AttributeVerificationProcessor"
            ) as processor_cls,
        ):
            processor_cls.return_value.process = mock.AsyncMock(return_value=expected_result)
            result = await self.service.process(request)

        self.assertIs(result, expected_result)
        processor_cls.assert_called_once_with(params=default_config, config_dir=None, verbose="INFO")
        processor_cls.return_value.process.assert_called_once_with("clip-1", "s3://bucket/augmented_video.mp4")

    async def test_process_merges_request_config_over_default(self) -> None:
        """Request config should override/extend default config before processing."""
        request = AttributeVerificationRequest(
            clip_id="clip-2",
            augmented_video_path="s3://bucket/augmented_video.mp4",
            config={"selected_variables": {"size": "large"}, "new_flag": True},
        )
        expected_result = processor_mod.AttributeVerificationResult(
            clip_id="clip-2",
            passed=False,
            summary=processor_mod.AttributeVerificationSummary(total_checks=1, passed_checks=0, failed_checks=1),
            checks=[],
        )
        default_config = {"selected_variables": {"color": "red"}, "threshold": 0.5}
        merged_config = {
            "selected_variables": {"size": "large"},
            "threshold": 0.5,
            "new_flag": True,
        }

        with (
            mock.patch.object(
                AttributeVerificationService,
                "get_default_config",
                new=mock.AsyncMock(return_value=default_config),
            ),
            mock.patch(
                "services.attribute_verification.attribute_verification_service.AttributeVerificationProcessor"
            ) as processor_cls,
        ):
            processor_cls.return_value.process = mock.AsyncMock(return_value=expected_result)
            result = await self.service.process(request)

        self.assertIs(result, expected_result)
        processor_cls.assert_called_once_with(params=merged_config, config_dir=None, verbose="INFO")
        processor_cls.return_value.process.assert_called_once_with("clip-2", "s3://bucket/augmented_video.mp4")


if __name__ == "__main__":
    unittest.main()
