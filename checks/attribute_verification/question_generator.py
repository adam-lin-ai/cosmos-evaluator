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

import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormatJSONSchema

from checks.attribute_verification.common import validate_and_cast_config_params


class LLMQuestionGenerator:
    """Generate multiple choice verification questions using LLM."""

    # Define expected parameter types for validation
    _PARAM_TYPES = {
        "system_prompt": str,
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "presence_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "endpoint": str,
        "model": str,
    }

    # JSON schema for structured output (guided JSON)
    _QUESTION_SCHEMA: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {
            "name": "verification_question",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Name of the variable being verified"},
                    "value": {"type": "string", "description": "The selected value to verify"},
                    "question": {"type": "string", "description": "The multiple choice question text"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B"],
                        "additionalProperties": False,
                    },
                    "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                },
                "required": ["variable", "value", "question", "options", "correct_answer"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        self.logger = logger
        self.logger.info(f"Initializing QuestionGenerator with params: {params}")
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)

        self.system_prompt = validated_params["system_prompt"]
        self.max_retries = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.presence_penalty = validated_params["presence_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]
        self.endpoint = validated_params["endpoint"]
        self.model = validated_params["model"]

        # Check for API key in environment
        api_key = os.environ.get("BUILD_NVIDIA_API_KEY")
        if not api_key:
            logger.warning("BUILD_NVIDIA_API_KEY is not set, using 'not-used' as placeholder")
            api_key = "not-used"

        self.client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

        self.logger = logger

    @classmethod
    def from_config(
        cls,
        config_params: dict,
        system_prompt: str,
        endpoint: str,
        model: str,
        logger: logging.Logger,
    ) -> "LLMQuestionGenerator":
        """
        Create LLMQuestionGenerator instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing LLM configuration parameters
            system_prompt: System prompt text
            endpoint: LLM endpoint URL
            model: LLM model name
            logger: Logger instance

        Returns:
            LLMQuestionGenerator: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": system_prompt,
            "retry": config_params.get("retry", 1),
            "temperature": config_params.get("temperature", 0.0),
            "top_p": config_params.get("top_p", 0.95),
            "frequency_penalty": config_params.get("frequency_penalty", 0.0),
            "presence_penalty": config_params.get("presence_penalty", 0.0),
            "max_tokens": config_params.get("max_tokens", 2048),
            "stream": config_params.get("stream", True),
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)

        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            presence_penalty=validated_params["presence_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
        )

    def _sanitize_model_output(self, text: str) -> str:
        """Remove artifacts from model output."""
        # Remove Markdown code fences
        sanitized = re.sub(r"```[a-zA-Z]*", "", text)
        sanitized = re.sub(r"```", "", sanitized)

        # Remove block comments
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        return sanitized

    def _parse_llm_response(self, result: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract a single question JSON.

        Args:
            result (str): The raw LLM response

        Returns:
            Dict[str, Any]: Question dictionary

        Raises:
            ValueError: If the response cannot be parsed as JSON
        """
        # Step 1: cut away any leading chain-of-thought like </think> blocks
        think_close_idx = result.rfind("</think>")
        base_text = result[think_close_idx + len("</think>") :].strip() if think_close_idx != -1 else result.strip()
        self.logger.debug(f"Post-think text length={len(base_text)} (think_found={think_close_idx != -1})")

        # Step 2: prefer content inside the first fenced code block, if present
        fence_match = re.search(
            r"```(?:json|javascript|js|python)?\s*([\s\S]*?)\s*```",
            base_text,
            flags=re.IGNORECASE,
        )
        candidate_text = fence_match.group(1).strip() if fence_match else base_text
        self.logger.debug(
            f"Fenced block {'found' if fence_match else 'not found'}; candidate length={len(candidate_text)}"
        )

        # Step 3: sanitize comments and stray fences
        sanitized = self._sanitize_model_output(candidate_text)
        self.logger.debug(f"Sanitized candidate length={len(sanitized)}")

        # Step 4: attempt to decode JSON
        try:
            decoder = json.JSONDecoder()
            # Scan for likely JSON starts
            for match in re.finditer(r"[\[{]", sanitized):
                start_idx = match.start()
                try:
                    obj, end_idx = decoder.raw_decode(sanitized, idx=start_idx)
                    self.logger.debug(f"Decoded JSON segment start={start_idx} end={end_idx}")
                    return self._normalize_question_output(obj)
                except json.JSONDecodeError:
                    continue
            raise json.JSONDecodeError("No decodable JSON value found", sanitized, 0)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode failed: msg={getattr(e, 'msg', '')}, pos={getattr(e, 'pos', None)}")
            error_msg = "Could not parse JSON from model output. Enable debug logs to inspect sanitized text."
            raise ValueError(error_msg) from e

    def _normalize_question_output(self, obj: Any) -> Dict[str, Any]:
        """
        Normalize the LLM output to expected format for a single question.

        Expected format:
        {
            "variable": "weather_condition",
            "value": "cloudy",
            "question": "What are the weather conditions?",
            "options": {"A": "clear", "B": "cloudy", "C": "rainy", "D": "snowy"},
            "correct_answer": "B"
        }
        """
        required_fields = ["variable", "value", "question", "options", "correct_answer"]

        # If it's already a dict with required fields, validate and return
        if isinstance(obj, dict):
            # Check if it has all required fields directly
            if all(field in obj for field in required_fields):
                return obj

            # Check if it's a wrapper dict
            for key in ("question", "item", "data", "result", "output"):
                inner = obj.get(key)
                if isinstance(inner, dict) and all(field in inner for field in required_fields):
                    return inner

        # If it's a list with one item, extract that item
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            if all(field in obj[0] for field in required_fields):
                return obj[0]

        raise ValueError(f"Output must be a dict with required fields: {required_fields}. Got: {type(obj).__name__}")

    def _create_user_prompt(self, variable_name: str, selected_value: str, all_options: List[str]) -> str:
        """
        Create the user prompt for generating a single question.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            str: The formatted user prompt
        """
        return f"""Generate a verification question for the following variable:
            Variable: {variable_name}
            Selected value: {selected_value}
            All possible values: {all_options}

            Create ONE multiple choice question that verifies if the selected value is present in the video frame.

            Requirements:
            1. The question must have 2-4 answer options (A, B, C, D)
            2. Each option must be taken from this list of all possible values: {all_options}. Options not in the list are not allowed.
            3. The selected value '{selected_value}' MUST be one of the options
            4. The question should be simple and direct
            5. Options should include other possible values from the list above
            6. Format the correct answer as a single letter (A, B, C, or D)

            Output MUST be a single JSON object with the following structure:
            {{
                "variable": "{variable_name}",
                "value": "{selected_value}",
                "question": "Question text?",
                "options": {{"A": "option1", "B": "option2", etc.}},
                "correct_answer": "B"
            }}

            Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""

    async def generate_question(
        self, variable_name: str, selected_value: str, all_options: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a verification question for a single variable.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            Dict[str, Any]: Question dictionary

        Raises:
            RuntimeError: If question generation fails
        """
        self.logger.debug(f"Generating question for variable '{variable_name}' with value '{selected_value}'")
        retries_remaining = self.max_retries
        user_prompt = self._create_user_prompt(variable_name, selected_value, all_options)

        while retries_remaining >= 0:
            try:
                self.logger.debug("Making API call to generate question with structured output")
                # Try structured output first (guided JSON), fall back to regular parsing
                try:
                    messages: list[ChatCompletionMessageParam] = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    self.logger.info(f"LLM endpoint call starting (model={self.model}, variable='{variable_name}')")
                    _llm_t0 = time.perf_counter()
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=self.max_tokens,
                        presence_penalty=self.presence_penalty,
                        response_format=self._QUESTION_SCHEMA,
                        stream=False,  # Structured output doesn't support streaming
                    )
                    self.logger.info(
                        f"LLM endpoint call completed in {(time.perf_counter() - _llm_t0) * 1000:.0f}ms"
                        f" (variable='{variable_name}')"
                    )
                    result = completion.choices[0].message.content
                    if result is None:
                        raise Exception("No content returned from LLM")
                    question = json.loads(result)
                    self.logger.debug("Successfully used structured output (guided JSON)")
                except (openai.APIError, openai.BadRequestError, json.JSONDecodeError, ValueError) as e:
                    # Fall back to manual parsing if structured output not supported
                    self.logger.debug(f"Structured output not available ({e}), falling back to manual parsing")
                    self.logger.info(
                        f"LLM endpoint call starting (model={self.model}, variable='{variable_name}', fallback=True)"
                    )
                    _llm_t0 = time.perf_counter()
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stream=self.stream,
                    )
                    self.logger.info(
                        f"LLM endpoint call completed in {(time.perf_counter() - _llm_t0) * 1000:.0f}ms"
                        f" (variable='{variable_name}', fallback=True)"
                    )

                    result = ""
                    if self.stream:
                        # Streaming: iterate over chunks and accumulate content
                        async for chunk in completion:
                            if chunk.choices[0].delta.content is not None:
                                result += chunk.choices[0].delta.content
                    else:
                        # Non-streaming: completion is a single response object
                        result = completion.choices[0].message.content
                        if result is None:
                            raise ValueError("No content returned from LLM response")

                    # Parse the result
                    question = self._parse_llm_response(result)

                self.logger.info(f"Successfully generated question for '{variable_name}'")
                self.logger.debug(f"Generated question: {question}")
                return question
            # Retryable errors only
            except (
                ValueError,
                json.JSONDecodeError,
                openai.APITimeoutError,
                openai.ConflictError,
                openai.InternalServerError,
                openai.UnprocessableEntityError,
            ) as e:
                self.logger.error(f"Failed to generate question for '{variable_name}' with LLM: {e}", exc_info=True)

                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        f"Retrying failed call to LLM. Retries remaining: {retries_remaining}/{self.max_retries}"
                    )
                    # Continue the while loop to retry
                else:
                    self.logger.exception(
                        f"Max retries reached. LLM failed to generate question after {self.max_retries + 1} total attempts",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Max retries reached. LLM failed to generate question after {self.max_retries + 1} total attempts"
                    ) from e
        return {}

    async def generate_questions(
        self, variables: Dict[str, str], variable_options: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate verification questions for all given variables (one at a time).

        Args:
            variables: Dictionary of variable names to selected values
            variable_options: Dictionary of variable names to all possible values

        Returns:
            List[Dict[str, Any]]: List of question dictionaries

        Raises:
            RuntimeError: If question generation fails for any variable
        """
        self.logger.info(f"Generating questions for {len(variables)} variables")
        questions = []

        for variable_name, selected_value in variables.items():
            all_options = variable_options.get(variable_name, [selected_value])

            question = await self.generate_question(variable_name, selected_value, all_options)
            questions.append(question)

        if not questions:
            raise RuntimeError("Failed to generate any questions")

        self.logger.info(f"Successfully generated {len(questions)} questions")
        return questions
