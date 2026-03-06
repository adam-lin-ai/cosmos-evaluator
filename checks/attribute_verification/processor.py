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

"""
Module for the attribute verification checker.

This module handles the verification of attributes in the augmented video using a VLM.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
import openai
from pydantic import BaseModel, Field

from checks.attribute_verification.question_generator import LLMQuestionGenerator
from checks.attribute_verification.vlm_verifier import VLMVerifier
from checks.utils.config_manager import ConfigManager


class AttributeVerificationCheck(BaseModel):
    variable: str = Field(..., description="Name of the variable being verified")
    value: str = Field(..., description="The expected value of the variable")
    question: str = Field(..., description="The question for the VLM to verify")
    options: Dict[str, str] = Field(..., description="The multiple choice answers for the question")
    expected_answer: str = Field(..., description="The correct multiple choice answer to the question")
    vlm_answer: str = Field(..., description="The multiple choice answer from the VLM to the question")
    passed: bool = Field(..., description="Whether the VLM answer matches the expected answer")
    error: Optional[str] = Field(None, description="The error message if the check failed to execute")


class AttributeVerificationSummary(BaseModel):
    total_checks: int = Field(..., description="The total number of checks performed")
    passed_checks: int = Field(..., description="The number of checks that passed")
    failed_checks: int = Field(..., description="The number of checks that failed")


class AttributeVerificationResult(BaseModel):
    clip_id: str
    passed: bool
    summary: AttributeVerificationSummary
    checks: List[AttributeVerificationCheck]


class AttributeVerificationProcessor:
    """
    Processor for the attribute verification checker.
    """

    def __init__(self, params: dict, config_dir: Optional[str] = None, verbose: str = "INFO"):
        """
        Initialize the processor.
        """
        self.logger = self._setup_logging(verbose)
        self.config = AttributeVerificationProcessor.get_default_config(config_dir=config_dir).get(
            "metropolis.attribute_verification", {}
        )
        if params is not None:
            self.config = self._deep_merge(self.config, params)
        self.question_generation_config = self.config.get("question_generation", {})
        self.vlm_verification_config = self.config.get("vlm_verification", {})
        self.question_generator = LLMQuestionGenerator.from_config(
            self.question_generation_config.get("parameters", {}),
            self.question_generation_config.get("system_prompt", ""),
            self.question_generation_config.get("llm", {}).get("endpoint", ""),
            self.question_generation_config.get("llm", {}).get("model", ""),
            self.logger,
        )
        self.vlm_verifier = VLMVerifier.from_config(
            self.vlm_verification_config.get("parameters", {}),
            self.vlm_verification_config.get("system_prompt", ""),
            self.vlm_verification_config.get("vlm", {}).get("endpoint", ""),
            self.vlm_verification_config.get("vlm", {}).get("model", ""),
            self.logger,
        )

    def _setup_logging(self, verbose: str) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)

        # Set logging level based on verbose mode
        log_level = getattr(logging, verbose.upper())
        logger.setLevel(log_level)

        # Ensure propagation is enabled
        logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.info(f"Logging level set to: {verbose}")
        return logger

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, recursively merging nested dictionaries.

        Args:
            base: Base dictionary to merge into
            update: Dictionary with updates to apply

        Returns:
            New dictionary with merged values
        """
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AttributeVerificationProcessor._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def get_default_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the default configuration for attribute verification processing.

        Args:
            config_dir: Path to the configuration directory. If None, defaults to
                       checks/config relative to the current working directory.

        Returns:
            Default configuration for attribute verification processing loaded from config.yaml
        """
        try:
            config_manager = ConfigManager(config_dir)
            config = config_manager.load_config("config")
            return {"metropolis.attribute_verification": config["metropolis.attribute_verification"]}
        except Exception as e:
            logging.error("Error loading default configuration: {}".format(e))
            raise e

    async def _verify_one(self, question_data: Dict[str, Any], augmented_video_path: str) -> AttributeVerificationCheck:
        """Run a single VLM verification. RuntimeError is recorded as a failed check; fatal errors propagate."""
        variable = question_data["variable"]
        value = question_data["value"]
        question = question_data["question"]
        options = question_data["options"]
        correct_answer = question_data["correct_answer"]

        self.logger.info(f"Verifying variable '{variable}' with value '{value}'")
        self.logger.debug(f"Question: {question}")

        try:
            is_correct, vlm_answer = await self.vlm_verifier.verify_question(
                augmented_video_path, question, options, correct_answer
            )
            if is_correct:
                self.logger.info(f"✓ Verification passed for '{variable}': expected {correct_answer}, got {vlm_answer}")
            else:
                self.logger.warning(
                    f"✗ Verification failed for '{variable}': expected {correct_answer}, got {vlm_answer}"
                )
            return AttributeVerificationCheck(
                variable=variable,
                value=value,
                question=question,
                options=options,
                expected_answer=correct_answer,
                vlm_answer=vlm_answer,
                passed=is_correct,
            )
        except openai.APIConnectionError as e:
            message = "The provided VLM endpoint is not reachable. Please check the VLM endpoint."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.NotFoundError as e:
            message = "The provided VLM endpoint was not found. Please check the VLM endpoint and model."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.AuthenticationError as e:
            message = "The API key is invalid or the provided VLM endpoint is not authorized. Please check the VLM endpoint and API key."
            self.logger.exception(message)
            raise HTTPException(status_code=401, detail=message) from e
        except openai.PermissionDeniedError as e:
            message = "The API key does not have permission to access the provided VLM endpoint. Please check the VLM endpoint and API key."
            self.logger.exception(message)
            raise HTTPException(status_code=403, detail=message) from e

    async def process(self, clip_id: str, augmented_video_path: str) -> AttributeVerificationResult:
        """
        Verify that the video contains the selected variable values.

        Args:
            augmented_video_path: Path to the generated video
            selected_variables: Dictionary of variable names to selected values
            variable_options: Dictionary of variable names to all possible values

        Returns:
            Tuple of (passed_attribute_check: bool, verification_details: AttributeVerificationSummary, verification_checks: List[AttributeVerificationCheck])

        The verification_details dictionary contains:
            - "questions": List of generated questions
            - "results": List of verification results for each question
            - "summary": Summary statistics
        """

        selected_variables = self.config.get("selected_variables", {})
        variable_options = self.config.get("variable_options", {})

        self.logger.info(f"Starting attribute verification for video: {augmented_video_path}")
        self.logger.info(f"Selected variables: {selected_variables}")

        summary: AttributeVerificationSummary = AttributeVerificationSummary(
            total_checks=0,
            passed_checks=0,
            failed_checks=0,
        )

        # Step 1: Generate verification questions using LLM
        self.logger.info("Generating verification questions...")
        try:
            questions = await self.question_generator.generate_questions(selected_variables, variable_options)
        except openai.APIConnectionError as e:
            message = "The provided LLM endpoint is not reachable. Please check the LLM endpoint."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.NotFoundError as e:
            message = "The provided LLM endpoint was not found. Please check the LLM endpoint and model."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.AuthenticationError as e:
            message = "The API key is invalid or the provided LLM endpoint is not authorized. Please check the LLM endpoint and API key."
            self.logger.exception(message)
            raise HTTPException(status_code=401, detail=message) from e
        except openai.PermissionDeniedError as e:
            message = "The API key does not have permission to access the provided LLM endpoint. Please check the LLM endpoint and API key."
            self.logger.exception(message)
            raise HTTPException(status_code=403, detail=message) from e
        summary.total_checks = len(questions)

        self.logger.info(f"Generated {len(questions)} verification questions")

        # Step 2: Verify all questions using VLM in parallel (each call is independent I/O)
        checks = list(await asyncio.gather(*[self._verify_one(q, augmented_video_path) for q in questions]))

        all_passed = True
        for check in checks:
            if check.error is not None:
                summary.failed_checks += 1
                all_passed = False
            elif check.passed:
                summary.passed_checks += 1
            else:
                summary.failed_checks += 1
                all_passed = False

        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("Verification Summary:")
        self.logger.info(f"  Total checks: {summary.total_checks}")
        self.logger.info(f"  Passed: {summary.passed_checks}")
        self.logger.info(f"  Failed: {summary.failed_checks}")
        self.logger.info(f"  Overall result: {'PASSED' if all_passed else 'FAILED'}")
        self.logger.info("=" * 80)

        return AttributeVerificationResult(clip_id=clip_id, passed=all_passed, summary=summary, checks=checks)
