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

"""Tests for run.py."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from checks.attribute_verification.processor import (
    AttributeVerificationCheck,
    AttributeVerificationResult,
    AttributeVerificationSummary,
)
from checks.attribute_verification.run import load_config, main, parse_args, save_results_to_file, validate_args


class TestValidateArgs(unittest.TestCase):
    """Tests for validate_args function."""

    @patch("multistorageclient.is_file")
    def test_validate_args_valid_s3_paths(self, mock_is_file: MagicMock) -> None:
        """Test validation with valid S3 paths."""
        mock_is_file.return_value = True
        args = argparse.Namespace(
            augmented_video_path="s3://bucket/key",
        )
        self.assertTrue(validate_args(args))

    def test_validate_args_valid_local_paths(self) -> None:
        """Test validation with valid local paths."""
        args = argparse.Namespace(
            augmented_video_path="./checks/sample_data/augmented.mp4",
        )
        self.assertTrue(validate_args(args))

    def test_validate_args_invalid_augmented_video_path(self) -> None:
        """Test validation with invalid augmented video path."""
        args = argparse.Namespace(
            augmented_video_path="s3://invalid/path",
        )
        self.assertFalse(validate_args(args))


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config function."""

    def test_load_config_success(self) -> None:
        """Test successful config loading."""
        config = load_config("config", "./checks/attribute_verification/tests/data")
        self.assertIsNotNone(config)
        assert config is not None
        self.assertIn("metropolis.attribute_verification", config)
        self.assertEqual(
            config["metropolis.attribute_verification"]["selected_variables"],
            {"weather": "cloudy", "time_of_day": "morning"},
        )

    def test_load_config_default_location(self) -> None:
        """Test loading config from default location."""
        config = load_config("config")
        self.assertIsNotNone(config)

    def test_load_config_nonexistent_file(self) -> None:
        """Test loading nonexistent config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                load_config("nonexistent", tmpdir)

    def test_load_config_invalid_yaml(self) -> None:
        """Test loading invalid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid YAML file
            config_file = Path(tmpdir) / "invalid.yaml"
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content:\n  - bad indentation")

            with self.assertRaises(yaml.YAMLError):
                load_config("invalid", tmpdir)


class TestSaveResultsToFile(unittest.TestCase):
    """Tests for save_results_to_file function."""

    def test_save_results_to_file_creates_directory(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            results = {"test": "data", "scores": [{"key": "overall", "value": 0.95}]}
            output_file = save_results_to_file(results, "test_sample", str(output_dir))

            self.assertTrue(Path(output_file).exists())
            self.assertTrue(output_dir.exists())

    def test_save_results_to_file_content(self) -> None:
        """Test that results are correctly saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "rule_id": "cosmos.attribute_verification",
                "scores": [{"key": "overall", "value": 0.85}],
                "passed": True,
            }
            output_file = save_results_to_file(results, "test_sample", tmpdir)

            with open(output_file) as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["rule_id"], "cosmos.attribute_verification")
            self.assertEqual(saved_data["scores"][0]["value"], 0.85)

    def test_save_results_to_file_filename_format(self) -> None:
        """Test that output filename follows expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {"test": "data"}
            clip_id = "sample_12345"
            output_file = save_results_to_file(results, clip_id, tmpdir)

            expected_filename = f"{clip_id}.attribute_verification.results.json"
            self.assertTrue(output_file.endswith(expected_filename))


class TestParseArgs(unittest.TestCase):
    """Tests for parse_args function."""

    def test_parse_args_required_arguments(self) -> None:
        """Test parsing required arguments."""
        with patch(
            "sys.argv",
            [
                "run.py",
                "--augmented-video-path",
                "/path/to/media.mp4",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.augmented_video_path, "/path/to/media.mp4")

    def test_parse_args_with_optional_arguments(self) -> None:
        """Test parsing with optional arguments."""
        with patch(
            "sys.argv",
            [
                "run.py",
                "--augmented-video-path",
                "/path/to/media.mp4",
                "--config",
                "custom_config",
                "--config-dir",
                "/custom/config/dir",
                "--verbose",
                "DEBUG",
                "--output-dir",
                "/custom/output",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.config, "custom_config")
            self.assertEqual(args.config_dir, "/custom/config/dir")
            self.assertEqual(args.verbose, "DEBUG")
            self.assertEqual(args.output_dir, "/custom/output")

    def test_parse_args_default_values(self) -> None:
        """Test default values for optional arguments."""
        with patch(
            "sys.argv",
            [
                "run.py",
                "--augmented-video-path",
                "/path/to/media.mp4",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.config, "config")
            self.assertIsNone(args.config_dir)
            self.assertEqual(args.verbose, "INFO")
            self.assertEqual(args.output_dir, "outputs")
            self.assertIsNotNone(args.clip_id)  # UUID generated


class TestMain(unittest.TestCase):
    """Tests for main function."""

    def test_main_success(self) -> None:
        """Test successful execution of main."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/attribute_verification/tests/data",
                    "--config",
                    "config",
                ],
            ):
                with (
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.get_default_config",
                        return_value={"metropolis.attribute_verification": {}},
                    ),
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.process",
                        return_value=AttributeVerificationResult(
                            clip_id="test_clip_id",
                            passed=True,
                            summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
                            checks=[
                                AttributeVerificationCheck(
                                    variable="test_variable",
                                    value="test_value",
                                    question="test_question",
                                    options={"test_option": "test_option"},
                                    expected_answer="test_answer",
                                    vlm_answer="test_answer",
                                    passed=True,
                                    error=None,
                                )
                            ],
                        ),
                    ),
                ):
                    main()

            output_dir = Path(tmpdir)
            output_files = list(output_dir.glob("*.json"))
            self.assertGreater(len(output_files), 0, f"No output files generated in {output_dir}")

    def test_main_with_clip_id(self) -> None:
        """Test main with explicit sample ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_id = "test_clip_id"
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--clip-id",
                    clip_id,
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/attribute_verification/tests/data",
                    "--config",
                    "config",
                ],
            ):
                with (
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.get_default_config",
                        return_value={"metropolis.attribute_verification": {}},
                    ),
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.process",
                        return_value=AttributeVerificationResult(
                            clip_id="test_clip_id",
                            passed=True,
                            summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
                            checks=[
                                AttributeVerificationCheck(
                                    variable="test_variable",
                                    value="test_value",
                                    question="test_question",
                                    options={"test_option": "test_option"},
                                    expected_answer="test_answer",
                                    vlm_answer="test_answer",
                                    passed=True,
                                    error=None,
                                )
                            ],
                        ),
                    ),
                ):
                    main()

            output_file = Path(tmpdir) / f"{clip_id}.attribute_verification.results.json"
            self.assertTrue(output_file.exists())

    def test_main_missing_env_file(self) -> None:
        """Test main with missing .env file (should warn but continue)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/attribute_verification/tests/data",
                    "--config",
                    "config",
                    "--env-file",
                    "/nonexistent/.env",
                ],
            ):
                # Should not raise exception
                with (
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.get_default_config",
                        return_value={"metropolis.attribute_verification": {}},
                    ),
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.process",
                        return_value=AttributeVerificationResult(
                            clip_id="test_clip_id",
                            passed=True,
                            summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
                            checks=[
                                AttributeVerificationCheck(
                                    variable="test_variable",
                                    value="test_value",
                                    question="test_question",
                                    options={"test_option": "test_option"},
                                    expected_answer="test_answer",
                                    vlm_answer="test_answer",
                                    passed=True,
                                    error=None,
                                )
                            ],
                        ),
                    ),
                ):
                    main()

    def test_main_missing_config_file(self) -> None:
        """Test main with missing config file (should warn but continue with empty config)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    tmpdir,
                    "--config",
                    "nonexistent",
                ],
            ):
                # Should not raise exception, uses empty config
                with (
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.get_default_config",
                        return_value={"metropolis.attribute_verification": {}},
                    ),
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.process",
                        return_value=AttributeVerificationResult(
                            clip_id="test_clip_id",
                            passed=True,
                            summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
                            checks=[
                                AttributeVerificationCheck(
                                    variable="test_variable",
                                    value="test_value",
                                    question="test_question",
                                    options={"test_option": "test_option"},
                                    expected_answer="test_answer",
                                    vlm_answer="test_answer",
                                    passed=True,
                                    error=None,
                                )
                            ],
                        ),
                    ),
                ):
                    main()

    def test_main_invalid_yaml_config(self) -> None:
        """Test main with invalid YAML config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid YAML file
            config_file = Path(tmpdir) / "invalid.yaml"
            with open(config_file, "w") as f:
                f.write("invalid:\n  yaml:\n    content:\n  - bad")

            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    tmpdir,
                    "--config",
                    "invalid",
                ],
            ):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)

    @patch("checks.attribute_verification.run.AttributeVerificationProcessor")
    def test_main_processor_exception(self, mock_processor_class: MagicMock) -> None:
        """Test main with processor raising an exception."""
        mock_processor = MagicMock()
        mock_processor.process.side_effect = Exception("Processing error")
        mock_processor_class.return_value = mock_processor
        # Mock get_default_config to avoid config file loading issues
        mock_processor_class.get_default_config.return_value = {"metropolis.attribute_verification": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/attribute_verification/tests/data",
                    "--config",
                    "config",
                ],
            ):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)

    def test_main_with_verbose_debug(self) -> None:
        """Test main with DEBUG verbose level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/attribute_verification/tests/data",
                    "--config",
                    "config",
                    "--verbose",
                    "DEBUG",
                ],
            ):
                with (
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.get_default_config",
                        return_value={"metropolis.attribute_verification": {}},
                    ),
                    patch(
                        "checks.attribute_verification.processor.AttributeVerificationProcessor.process",
                        return_value=AttributeVerificationResult(
                            clip_id="test_clip_id",
                            passed=True,
                            summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
                            checks=[
                                AttributeVerificationCheck(
                                    variable="test_variable",
                                    value="test_value",
                                    question="test_question",
                                    options={"test_option": "test_option"},
                                    expected_answer="test_answer",
                                    vlm_answer="test_answer",
                                    passed=True,
                                    error=None,
                                )
                            ],
                        ),
                    ),
                ):
                    main()

            # Should complete successfully
            output_files = list(Path(tmpdir).glob("*.json"))
            self.assertGreater(len(output_files), 0)

    def test_main_config_none_handling(self) -> None:
        """Test main when config loading returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("checks.attribute_verification.run.load_config") as mock_load:
                mock_load.return_value = None

                with patch(
                    "sys.argv",
                    [
                        "run.py",
                        "--augmented-video-path",
                        "./checks/sample_data/augmented.mp4",
                        "--output-dir",
                        tmpdir,
                        "--config-dir",
                        "./checks/attribute_verification/tests/data",
                        "--config",
                        "config",
                    ],
                ):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
