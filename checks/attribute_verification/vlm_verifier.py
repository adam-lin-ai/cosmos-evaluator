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

import asyncio
import base64
import logging
import os
import re
import shutil
import sys
import tempfile
import time

import multistorageclient as msc
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from checks.attribute_verification.common import validate_and_cast_config_params

# OpenCV's loader can recurse under PEX layouts unless it replaces sys.path[0].
sys.OpenCV_REPLACE_SYS_PATH_0 = True  # type: ignore[attr-defined]
import cv2  # noqa: E402


class VLMVerifier:
    """Verify video attributes using VLM on first frame."""

    # Define expected parameter types for validation
    _PARAM_TYPES = {
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "system_prompt": str,
        "endpoint": str,
        "model": str,
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
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
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        self.logger = logger
        self.logger.info(f"Initializing VLMVerifier with params: {params}")
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)

        self.system_prompt = validated_params["system_prompt"]
        self.max_retries = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]

        # Ensure endpoint URL is properly formatted
        self.endpoint = validated_params["endpoint"].rstrip("/")
        self.model = validated_params["model"]

        api_key = os.environ.get("BUILD_NVIDIA_API_KEY")
        if not api_key:
            self.logger.warning("BUILD_NVIDIA_API_KEY is not set, using 'not-used' as placeholder")
            api_key = "not-used"
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=api_key, timeout=7200)

    @classmethod
    def from_config(
        cls, config_params: dict, system_prompt: str, endpoint: str, model: str, logger: logging.Logger
    ) -> "VLMVerifier":
        """
        Create VLMVerifier instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing VLM configuration parameters
            system_prompt: System prompt text
            endpoint: VLM endpoint URL
            model: VLM model name
            logger: Logger instance

        Returns:
            VLMVerifier: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": system_prompt,
            "retry": config_params.get("retry", 0),
            "temperature": config_params.get("temperature", 0.0),
            "top_p": config_params.get("top_p", 1.0),
            "frequency_penalty": config_params.get("frequency_penalty", 0.0),
            "max_tokens": config_params.get("max_tokens", 512),
            "stream": config_params.get("stream", False),
            "endpoint": endpoint,
            "model": model,
        }
        logger.info(f"Config params: {config_params}")
        logger.info(f"Creating VLMVerifier from config with params: {params}")

        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)

        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
        )

    def _extract_first_frame(self, video_path: str, output_image_path: str) -> bool:
        """
        Extract the first frame from a video file.

        Args:
            video_path: Path to the video file
            output_image_path: Path to save the extracted frame

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return False

            # Read the first frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.logger.error(f"Failed to read first frame from: {video_path}")
                return False

            # Save the frame as an image
            cv2.imwrite(output_image_path, frame)
            self.logger.debug(f"Extracted first frame to: {output_image_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error extracting first frame: {e}")
            return False

    def _format_question(self, question: str, options: dict) -> str:
        """
        Format the question with options for the VLM.

        Args:
            question: The question text
            options: Dictionary of option letters to values

        Returns:
            str: Formatted question string
        """
        options_text = "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
        return f"{question}\n{options_text}\n\nAnswer with only a single letter (A, B, C, or D)."

    async def verify_question(
        self, video_path: str, question: str, options: dict, correct_answer: str
    ) -> tuple[bool, str]:
        """
        Verify a single question by asking the VLM about the first frame.

        Args:
            video_path: Path to the video file
            question: The question to ask
            options: Dictionary of answer options
            correct_answer: The expected correct answer letter

        Returns:
            Tuple of (is_correct: bool, vlm_answer: str)

        Raises:
            Exception: If verification fails
        """

        retries_remaining = self.max_retries
        formatted_question = self._format_question(question, options)

        def _prepare_frame() -> str:
            """Download video and extract first frame; return base64-encoded JPEG bytes."""
            if not msc.is_file(video_path):
                raise Exception(f"Video file not found: {video_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video_path = os.path.join(temp_dir, "video.mp4")
                with msc.open(video_path, "rb") as src, open(temp_video_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                self.logger.debug(f"Downloaded video to {temp_video_path}")

                temp_image_path = os.path.join(temp_dir, "first_frame.jpg")
                if not self._extract_first_frame(temp_video_path, temp_image_path):
                    raise Exception("Failed to extract first frame from video")

                with open(temp_image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode()

        while retries_remaining >= 0:
            try:
                # Blocking file I/O runs in a thread; the event loop stays free for other requests.
                image_b64 = await asyncio.to_thread(_prepare_frame)

                conversation: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": formatted_question},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ]

                self.logger.info(f"VLM endpoint call starting (model={self.model})")
                _vlm_t0 = time.perf_counter()
                chat_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=self.max_tokens,
                    stream=self.stream,
                )
                self.logger.info(f"VLM endpoint call completed in {(time.perf_counter() - _vlm_t0) * 1000:.0f}ms")
                assistant_message = chat_response.choices[0].message
                self.logger.debug(f"VLM response: {assistant_message.content}")

                if assistant_message.content is None:
                    raise Exception("Unable to get content from VLM response")
                # Extract answer - look for single letter A, B, C, or D
                answer_match = re.search(r"\b([A-D])\b", assistant_message.content.upper())
                if answer_match:
                    vlm_answer = answer_match.group(1)
                else:
                    # If no clear answer found, take the first letter that appears
                    vlm_answer = assistant_message.content.strip().upper()
                    if vlm_answer not in ["A", "B", "C", "D"]:
                        self.logger.warning(
                            f"Could not parse clear answer from VLM response: {assistant_message.content}"
                        )
                        vlm_answer = "UNKNOWN"

                is_correct = vlm_answer == correct_answer.upper()
                self.logger.debug(
                    f"Question verification: expected={correct_answer}, got={vlm_answer}, correct={is_correct}"
                )

                return is_correct, vlm_answer

            # Retryable errors only
            except (
                openai.APITimeoutError,
                openai.ConflictError,
                openai.InternalServerError,
                openai.UnprocessableEntityError,
            ) as e:
                self.logger.error("Failed to verify question with VLM", exc_info=False)
                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        f"Retrying failed call to VLM. Retries remaining: {retries_remaining}/{self.max_retries}"
                    )
                    # Continue the while loop to retry
                else:
                    self.logger.exception(
                        f"Max retries reached. VLM failed to verify question after {self.max_retries + 1} total attempts",
                        exc_info=False,
                    )
                    raise RuntimeError(
                        f"Max retries reached. VLM failed to verify question after {self.max_retries + 1} total attempts"
                    ) from e
        return False, "UNKNOWN"
