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

"""Unit tests for the attribute verification processor."""

from typing import Any
import unittest
from unittest.mock import MagicMock

import openai

from checks.attribute_verification.processor import AttributeVerificationProcessor, AttributeVerificationResult


def _make_openai_status_error_response(status_code: int = 404) -> MagicMock:
    """Build a minimal fake httpx.Response for openai.APIStatusError subclasses."""
    response = MagicMock()
    response.request = MagicMock()
    response.status_code = status_code
    response.headers = {"x-request-id": "test-request-id"}
    return response


class TestAttributeVerificationProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Set up the test processor."""
        self.processor = AttributeVerificationProcessor({})
        self.processor.question_generator = _DummyQuestionGenerator()  # type: ignore[assignment]
        self.processor.vlm_verifier = _DummyVLMVerifier()  # type: ignore[assignment]

    def test_deep_merge(self) -> None:
        """Test the deep merge method."""
        base = {
            "a": 1,
            "b": 2,
            "c": {
                "d": 3,
                "e": 4,
            },
        }
        update = {
            "a": 10,
            "b": 20,
            "c": {
                "d": 30,
                "e": 40,
                "f": 50,
                "g": {
                    "h": 60,
                    "i": 70,
                },
            },
        }
        expected = {
            "a": 10,
            "b": 20,
            "c": {
                "d": 30,
                "e": 40,
                "f": 50,
                "g": {
                    "h": 60,
                    "i": 70,
                },
            },
        }
        result = AttributeVerificationProcessor._deep_merge(base, update)
        self.assertDictEqual(result, expected)

    def test_get_default_config(self) -> None:
        """Test the get default config method."""
        config = AttributeVerificationProcessor.get_default_config()
        self.assertIsInstance(config, dict)
        self.assertIn("metropolis.attribute_verification", config)
        self.assertIsInstance(config["metropolis.attribute_verification"], dict)
        self.assertIn("question_generation", config["metropolis.attribute_verification"])
        self.assertIn("vlm_verification", config["metropolis.attribute_verification"])
        self.assertIn("parameters", config["metropolis.attribute_verification"]["question_generation"])
        self.assertIn("parameters", config["metropolis.attribute_verification"]["vlm_verification"])

    async def test_process(self) -> None:
        """Test the process method."""
        clip_id = "test_clip"
        augmented_video_path = "test_video.mp4"

        result = await self.processor.process(clip_id, augmented_video_path)
        self.assertIsInstance(result, AttributeVerificationResult)
        self.assertEqual(result.clip_id, clip_id)
        self.assertEqual(result.passed, True)
        self.assertEqual(result.summary.total_checks, 2)
        self.assertEqual(result.summary.passed_checks, 2)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(len(result.checks), 2)
        self.assertEqual(result.checks[0].variable, "color")
        self.assertEqual(result.checks[0].value, "blue")
        self.assertEqual(result.checks[0].question, "What is the color of the car?")
        self.assertEqual(result.checks[0].options, {"A": "blue", "B": "red", "C": "green", "D": "yellow"})
        self.assertEqual(result.checks[0].expected_answer, "A")
        self.assertEqual(result.checks[0].vlm_answer, "A")
        self.assertEqual(result.checks[0].passed, True)
        self.assertEqual(result.checks[0].error, None)
        self.assertEqual(result.checks[1].variable, "type")
        self.assertEqual(result.checks[1].value, "sedan")
        self.assertEqual(result.checks[1].question, "What type of car is it?")
        self.assertEqual(result.checks[1].options, {"A": "truck", "B": "SUV", "C": "sedan", "D": "van"})
        self.assertEqual(result.checks[1].expected_answer, "C")
        self.assertEqual(result.checks[1].vlm_answer, "C")
        self.assertEqual(result.checks[1].error, None)

    def _make_processor_with_dummy_question_generator(self) -> AttributeVerificationProcessor:
        """Return a processor with dummy VLM verifier; caller sets question_generator."""
        processor = AttributeVerificationProcessor({})
        processor.vlm_verifier = _DummyVLMVerifier()  # type: ignore[assignment]
        return processor

    def _make_processor_with_dummy_vlm_verifier(self) -> AttributeVerificationProcessor:
        """Return a processor with dummy question generator; caller sets vlm_verifier."""
        processor = AttributeVerificationProcessor({})
        processor.question_generator = _DummyQuestionGenerator()  # type: ignore[assignment]
        return processor

    # --- generate_questions error tests ---

    async def test_process_generate_questions_runtime_error(self) -> None:
        """RuntimeError from generate_questions is propagated as-is."""
        processor = self._make_processor_with_dummy_question_generator()
        processor.question_generator = _RaisingQuestionGenerator(RuntimeError("LLM failed"))  # type: ignore[assignment]

        with self.assertRaises(RuntimeError) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("LLM failed", str(ctx.exception))

    async def test_process_generate_questions_not_found_error(self) -> None:
        """openai.NotFoundError from generate_questions is re-raised as Exception with LLM endpoint message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.NotFoundError(
            "not found",
            response=_make_openai_status_error_response(404),
            body=None,
        )
        processor.question_generator = _RaisingQuestionGenerator(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("LLM endpoint was not found", str(ctx.exception))

    async def test_process_generate_questions_authentication_error(self) -> None:
        """openai.AuthenticationError from generate_questions is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.AuthenticationError(
            "unauthorized",
            response=_make_openai_status_error_response(401),
            body=None,
        )
        processor.question_generator = _RaisingQuestionGenerator(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("API key is invalid", str(ctx.exception))

    async def test_process_generate_questions_permission_denied_error(self) -> None:
        """openai.PermissionDeniedError from generate_questions is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.PermissionDeniedError(
            "forbidden",
            response=_make_openai_status_error_response(403),
            body=None,
        )
        processor.question_generator = _RaisingQuestionGenerator(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("does not have permission", str(ctx.exception))

    async def test_process_generate_questions_unexpected_error(self) -> None:
        """Generic Exception from generate_questions is propagated as-is."""
        processor = self._make_processor_with_dummy_question_generator()
        processor.question_generator = _RaisingQuestionGenerator(ValueError("unexpected"))  # type: ignore[assignment]

        with self.assertRaises(ValueError) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("unexpected", str(ctx.exception))

    # --- verify_question error tests ---

    async def test_process_verify_question_runtime_error_recorded_as_failed_check(self) -> None:
        """RuntimeError from verify_question now propagates from _verify_one."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = _RaisingVLMVerifier(RuntimeError("Max retries reached"))  # type: ignore[assignment]

        with self.assertRaises(RuntimeError) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("Max retries reached", str(ctx.exception))

    async def test_process_verify_question_not_found_error(self) -> None:
        """openai.NotFoundError from verify_question is re-raised as Exception with VLM endpoint message."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.NotFoundError(
            "not found",
            response=_make_openai_status_error_response(404),
            body=None,
        )
        processor.vlm_verifier = _RaisingVLMVerifier(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("VLM endpoint was not found", str(ctx.exception))

    async def test_process_verify_question_authentication_error(self) -> None:
        """openai.AuthenticationError from verify_question is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.AuthenticationError(
            "unauthorized",
            response=_make_openai_status_error_response(401),
            body=None,
        )
        processor.vlm_verifier = _RaisingVLMVerifier(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("API key is invalid", str(ctx.exception))

    async def test_process_verify_question_permission_denied_error(self) -> None:
        """openai.PermissionDeniedError from verify_question is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.PermissionDeniedError(
            "forbidden",
            response=_make_openai_status_error_response(403),
            body=None,
        )
        processor.vlm_verifier = _RaisingVLMVerifier(exc)  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("does not have permission", str(ctx.exception))

    async def test_process_verify_question_unexpected_error(self) -> None:
        """Generic Exception from verify_question is propagated as-is."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = _RaisingVLMVerifier(Exception("VLM crashed"))  # type: ignore[assignment]

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("VLM crashed", str(ctx.exception))


class _DummyQuestionGenerator:
    """Returns a fixed list of questions (used by tests that only need verify_question to run)."""

    async def generate_questions(self, selected_variables: dict, variable_options: dict) -> list[dict[str, Any]]:
        return [
            {
                "variable": "color",
                "value": "blue",
                "question": "What is the color of the car?",
                "options": {"A": "blue", "B": "red", "C": "green", "D": "yellow"},
                "correct_answer": "A",
            },
            {
                "variable": "type",
                "value": "sedan",
                "question": "What type of car is it?",
                "options": {"A": "truck", "B": "SUV", "C": "sedan", "D": "van"},
                "correct_answer": "C",
            },
        ]


class _DummyVLMVerifier:
    """Returns (True, correct_answer) for every verify_question call."""

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
    ) -> tuple[bool, str]:
        return True, correct_answer


class _RaisingQuestionGenerator:
    """Question generator that raises a given exception from generate_questions."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def generate_questions(self, selected_variables: dict, variable_options: dict) -> list[dict[str, Any]]:
        raise self._exc


class _RaisingVLMVerifier:
    """VLM verifier that raises a given exception from verify_question."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
    ) -> tuple[bool, str]:
        raise self._exc


if __name__ == "__main__":
    unittest.main()
