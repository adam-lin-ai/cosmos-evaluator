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

"""Unit tests for the attribute verification question generator.

One test per code branch for 100% coverage without redundant tests.
"""

import json
import logging
from typing import AsyncGenerator, Optional
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from checks.attribute_verification.question_generator import LLMQuestionGenerator


def _make_mock_completion(content: Optional[str]) -> MagicMock:
    """Build a mock chat completion with the given message content."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


class TestLLMQuestionGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Set up for question generator tests."""
        self.logger = logging.getLogger(__name__)
        with patch("checks.attribute_verification.question_generator.AsyncOpenAI"):
            self.generator = LLMQuestionGenerator(
                system_prompt="You are a helpful assistant.",
                retry=1,
                temperature=0.0,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=2048,
                stream=False,
                endpoint="http://test.invalid",
                model="test-model",
                logger=self.logger,
            )

    def test_from_config_returns_instance(self) -> None:
        """from_config builds an LLMQuestionGenerator with config params."""
        with patch("checks.attribute_verification.question_generator.AsyncOpenAI"):
            instance = LLMQuestionGenerator.from_config(
                config_params={"retry": 2, "temperature": 0.1},
                system_prompt="System",
                endpoint="http://ep",
                model="model",
                logger=self.logger,
            )
        self.assertIsInstance(instance, LLMQuestionGenerator)
        self.assertEqual(instance.max_retries, 2)
        self.assertEqual(instance.temperature, 0.1)

    def test_sanitize_model_output_strips_fences_and_comments(self) -> None:
        """_sanitize_model_output removes markdown fences and block comments."""
        text = '```json\n{"x": 1}\n```\n/* comment */'
        result = self.generator._sanitize_model_output(text)
        self.assertNotIn("```", result)
        self.assertNotIn("/*", result)
        self.assertNotIn("*/", result)
        self.assertIn('{"x": 1}', result)

    def test_parse_llm_response_returns_question_dict(self) -> None:
        """_parse_llm_response extracts and normalizes a question from JSON text."""
        raw = '{"variable": "color", "value": "red", "question": "What color?", "options": {"A": "red", "B": "blue"}, "correct_answer": "A"}'
        result = self.generator._parse_llm_response(raw)
        self.assertEqual(result["variable"], "color")
        self.assertEqual(result["value"], "red")
        self.assertEqual(result["question"], "What color?")
        self.assertEqual(result["options"], {"A": "red", "B": "blue"})
        self.assertEqual(result["correct_answer"], "A")

    def test_normalize_question_output_returns_dict_with_required_fields(self) -> None:
        """_normalize_question_output returns a dict that already has required fields as-is."""
        obj = {
            "variable": "weather",
            "value": "sunny",
            "question": "Weather?",
            "options": {"A": "sunny", "B": "rainy"},
            "correct_answer": "A",
        }
        result = self.generator._normalize_question_output(obj)
        self.assertEqual(result, obj)

    async def test_generate_question_returns_question_when_mock_returns_json(self) -> None:
        """generate_question returns a question dict when client returns valid JSON."""
        question = {
            "variable": "color",
            "value": "blue",
            "question": "What is the color?",
            "options": {"A": "blue", "B": "red"},
            "correct_answer": "A",
        }
        self.generator.client.chat.completions.create = AsyncMock(
            return_value=_make_mock_completion(json.dumps(question))
        )
        result = await self.generator.generate_question("color", "blue", ["blue", "red", "green"])
        self.assertEqual(result["variable"], "color")
        self.assertEqual(result["correct_answer"], "A")

    async def test_generate_questions_returns_list_when_mock_returns_json(self) -> None:
        """generate_questions returns a list of questions per variable."""
        question = {
            "variable": "color",
            "value": "green",
            "question": "What color?",
            "options": {"A": "green", "B": "red"},
            "correct_answer": "A",
        }
        self.generator.client.chat.completions.create = AsyncMock(
            return_value=_make_mock_completion(json.dumps(question))
        )
        variables = {"color": "green"}
        variable_options = {"color": ["green", "red"]}
        result = await self.generator.generate_questions(variables, variable_options)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["variable"], "color")
        self.assertEqual(result[0]["value"], "green")

    # --- One test per additional branch for 100% coverage ---

    def test_init_uses_placeholder_key_when_env_unset(self) -> None:
        """__init__: BUILD_NVIDIA_API_KEY not set → AsyncOpenAI called with api_key='not-used'."""
        with patch("checks.attribute_verification.question_generator.AsyncOpenAI") as mock_openai:
            with patch(
                "checks.attribute_verification.question_generator.os.environ.get",
                return_value=None,
            ):
                LLMQuestionGenerator(
                    system_prompt="",
                    retry=1,
                    temperature=0.0,
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=2048,
                    stream=False,
                    endpoint="http://test",
                    model="m",
                    logger=self.logger,
                )
                mock_openai.assert_called_once()
                call_kw = mock_openai.call_args[1]
                self.assertEqual(call_kw["api_key"], "not-used")

    def test_parse_llm_response_after_think_block(self) -> None:
        """_parse_llm_response: result contains </think> → base_text is text after </think>."""
        raw = '</think>\n{"variable":"x","value":"y","question":"Q?","options":{"A":"a","B":"b"},"correct_answer":"A"}'
        result = self.generator._parse_llm_response(raw)
        self.assertEqual(result["variable"], "x")
        self.assertEqual(result["correct_answer"], "A")

    def test_parse_llm_response_uses_fenced_content(self) -> None:
        """_parse_llm_response: content inside ```json ... ``` is used as candidate."""
        raw = (
            '```json\n{"variable":"v","value":"x","question":"Q","options":{"A":"a","B":"b"},"correct_answer":"A"}\n```'
        )
        result = self.generator._parse_llm_response(raw)
        self.assertEqual(result["variable"], "v")

    def test_parse_llm_response_raises_when_no_valid_json(self) -> None:
        """_parse_llm_response: no decodable JSON → ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.generator._parse_llm_response("not json at all {")
        self.assertIn("Could not parse JSON", str(ctx.exception))

    def test_normalize_question_output_unwraps_wrapper_dict(self) -> None:
        """_normalize_question_output: dict with 'question' key containing full object → return inner."""
        inner = {
            "variable": "a",
            "value": "b",
            "question": "Q?",
            "options": {"A": "x", "B": "y"},
            "correct_answer": "A",
        }
        result = self.generator._normalize_question_output({"question": inner})
        self.assertEqual(result, inner)

    def test_normalize_question_output_unwraps_single_item_list(self) -> None:
        """_normalize_question_output: list with one dict with required fields → return that dict."""
        q = {
            "variable": "a",
            "value": "b",
            "question": "Q?",
            "options": {"A": "x", "B": "y"},
            "correct_answer": "A",
        }
        result = self.generator._normalize_question_output([q])
        self.assertEqual(result, q)

    def test_normalize_question_output_raises_on_invalid_shape(self) -> None:
        """_normalize_question_output: invalid shape (e.g. empty list) → ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.generator._normalize_question_output([])
        self.assertIn("required fields", str(ctx.exception))

    async def test_generate_question_raises_when_content_none(self) -> None:
        """generate_question: completion.choices[0].message.content is None → Exception."""
        self.generator.client.chat.completions.create = AsyncMock(return_value=_make_mock_completion(None))
        with self.assertRaises(Exception) as ctx:
            await self.generator.generate_question("x", "y", ["y", "z"])
        self.assertIn("No content returned", str(ctx.exception))

    async def test_generate_question_fallback_when_structured_output_fails(self) -> None:
        """generate_question: first create raises BadRequestError → fallback to manual parse (non-streaming)."""
        question = {
            "variable": "c",
            "value": "v",
            "question": "Q?",
            "options": {"A": "a", "B": "b"},
            "correct_answer": "A",
        }
        raw = json.dumps(question)
        self.generator.stream = False
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.BadRequestError("bad", response=MagicMock(), body=None),
                _make_mock_completion(raw),
            ]
        )
        result = await self.generator.generate_question("c", "v", ["a", "b"])
        self.assertEqual(result["variable"], "c")
        self.assertEqual(self.generator.client.chat.completions.create.call_count, 2)

    async def test_generate_question_fallback_streaming(self) -> None:
        """generate_question: fallback path with stream=True → accumulate chunks then parse."""
        question = {
            "variable": "c",
            "value": "v",
            "question": "Q?",
            "options": {"A": "a", "B": "b"},
            "correct_answer": "A",
        }
        raw = json.dumps(question)
        self.generator.stream = True
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = raw[:15]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = raw[15:]

        async def _aiter_chunks() -> AsyncGenerator[MagicMock, None]:
            yield chunk1
            yield chunk2

        stream_completion = _aiter_chunks()
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.BadRequestError("bad", response=MagicMock(), body=None),
                stream_completion,
            ]
        )
        result = await self.generator.generate_question("c", "v", ["a", "b"])
        self.assertEqual(result["variable"], "c")

    async def test_generate_question_retries_on_retryable_error(self) -> None:
        """generate_question: retryable error then success → retry and return."""
        question = {
            "variable": "x",
            "value": "y",
            "question": "Q?",
            "options": {"A": "a", "B": "b"},
            "correct_answer": "A",
        }
        self.generator.max_retries = 1
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                _make_mock_completion(json.dumps(question)),
            ]
        )
        result = await self.generator.generate_question("x", "y", ["a", "b"])
        self.assertEqual(result["variable"], "x")
        self.assertEqual(self.generator.client.chat.completions.create.call_count, 2)

    async def test_generate_question_raises_after_exhausted_retries(self) -> None:
        """generate_question: retry=0 and create raises → RuntimeError after retries exhausted."""
        self.generator.max_retries = 0
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        with self.assertRaises(RuntimeError) as ctx:
            await self.generator.generate_question("x", "y", ["a", "b"])
        self.assertIn("Max retries reached", str(ctx.exception))

    async def test_generate_questions_uses_selected_value_when_option_list_missing(self) -> None:
        """generate_questions: variable_options.get(name, [selected_value]) when key missing."""
        question = {
            "variable": "color",
            "value": "blue",
            "question": "Q?",
            "options": {"A": "blue", "B": "red"},
            "correct_answer": "A",
        }
        self.generator.client.chat.completions.create = AsyncMock(
            return_value=_make_mock_completion(json.dumps(question))
        )
        variables = {"color": "blue"}
        variable_options: dict[str, list[str]] = {}
        result = await self.generator.generate_questions(variables, variable_options)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], "blue")

    async def test_generate_questions_continues_after_retryable_error(self) -> None:
        """generate_questions: a failure on first variable now propagates and stops processing."""
        question = {
            "variable": "b",
            "value": "y",
            "question": "Q?",
            "options": {"A": "y", "B": "n"},
            "correct_answer": "A",
        }
        self.generator.max_retries = 0
        # First variable "a": first create() raises (inner except catches and fallback);
        # fallback create() also raises → outer except, no retry, raise. So 2 calls for "a".
        # Second variable "b": one create() returns. Total 3 calls.
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                openai.APITimeoutError(request=MagicMock()),
                _make_mock_completion(json.dumps(question)),
            ]
        )
        variables = {"a": "x", "b": "y"}
        variable_options = {"a": ["x"], "b": ["y", "n"]}
        with self.assertRaises(RuntimeError) as ctx:
            await self.generator.generate_questions(variables, variable_options)
        self.assertIn("Max retries reached", str(ctx.exception))

    async def test_generate_questions_raises_when_all_fail(self) -> None:
        """generate_questions: failure from generate_question propagates RuntimeError."""
        self.generator.max_retries = 0
        self.generator.client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        variables = {"x": "y"}
        variable_options = {"x": ["y"]}
        with self.assertRaises(RuntimeError) as ctx:
            await self.generator.generate_questions(variables, variable_options)
        self.assertIn("Max retries reached", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
