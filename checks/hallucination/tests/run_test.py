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

from checks.hallucination.run import load_config, main, parse_args, save_results_to_file, validate_args


class TestValidateArgs(unittest.TestCase):
    """Tests for validate_args function."""

    @patch("checks.utils.multistorage.msc.is_file")
    def test_validate_args_valid_s3_paths(self, mock_is_file: MagicMock) -> None:
        """Test validation with valid S3 paths."""
        mock_is_file.return_value = True
        args = argparse.Namespace(
            augmented_video_path="s3://bucket/key",
            original_video_path="s3://bucket/key",
        )
        self.assertTrue(validate_args(args))

    def test_validate_args_valid_local_paths(self) -> None:
        """Test validation with valid local paths."""
        args = argparse.Namespace(
            augmented_video_path="./checks/sample_data/augmented.mp4",
            original_video_path="./checks/sample_data/original.mp4",
        )
        self.assertTrue(validate_args(args))

    def test_validate_args_invalid_media_path(self) -> None:
        """Test validation with invalid media path."""
        args = argparse.Namespace(
            augmented_video_path="invalid://path",
            original_video_path="./checks/sample_data/original.mp4",
        )
        self.assertFalse(validate_args(args))

    def test_validate_args_invalid_original_video(self) -> None:
        """Test validation with invalid original video path."""
        args = argparse.Namespace(
            augmented_video_path="./checks/sample_data/augmented.mp4",
            original_video_path="invalid://path",
        )
        self.assertFalse(validate_args(args))


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config function."""

    def test_load_config_success(self) -> None:
        """Test successful config loading."""
        config = load_config("config", "./checks/hallucination/tests/data")
        self.assertIsNotNone(config)
        assert config is not None
        self.assertIn("metropolis.hallucination", config)
        self.assertEqual(config["metropolis.hallucination"]["moving_window_size"], 33)

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
                "rule_id": "cosmos.hallucination",
                "scores": [{"key": "overall", "value": 0.85}],
                "passed": True,
            }
            output_file = save_results_to_file(results, "test_sample", tmpdir)

            with open(output_file) as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["rule_id"], "cosmos.hallucination")
            self.assertEqual(saved_data["scores"][0]["value"], 0.85)

    def test_save_results_to_file_filename_format(self) -> None:
        """Test that output filename follows expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {"test": "data"}
            clip_id = "sample_12345"
            output_file = save_results_to_file(results, clip_id, tmpdir)

            expected_filename = f"{clip_id}.hallucination.results.json"
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
                "/path/to/augmented.mp4",
                "--original-video-path",
                "/path/to/original.mp4",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.augmented_video_path, "/path/to/augmented.mp4")
            self.assertEqual(args.original_video_path, "/path/to/original.mp4")

    def test_parse_args_with_optional_arguments(self) -> None:
        """Test parsing with optional arguments."""
        with patch(
            "sys.argv",
            [
                "run.py",
                "--augmented-video-path",
                "/path/to/augmented.mp4",
                "--original-video-path",
                "/path/to/original.mp4",
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
                "/path/to/augmented.mp4",
                "--original-video-path",
                "/path/to/original.mp4",
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
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
                    "--config",
                    "config",
                ],
            ):
                main()

            output_dir = Path(tmpdir)
            output_files = list(output_dir.glob("*.json"))
            self.assertGreater(len(output_files), 0, f"No output files generated in {output_dir}")

    def test_main_with_clip_id(self) -> None:
        """Test main with explicit sample ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_id = "custom_sample_123"
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--clip-id",
                    clip_id,
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
                    "--config",
                    "config",
                ],
            ):
                main()

            output_file = Path(tmpdir) / f"{clip_id}.hallucination.results.json"
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
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
                    "--config",
                    "config",
                    "--env-file",
                    "/nonexistent/.env",
                ],
            ):
                # Should not raise exception
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
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    tmpdir,
                    "--config",
                    "nonexistent",
                ],
            ):
                # Should not raise exception, uses empty config
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
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
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

    def test_main_validation_failure(self) -> None:
        """Test main with invalid arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "invalid://path",
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
                    "--config",
                    "config",
                ],
            ):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertEqual(cm.exception.code, 1)

    @patch("checks.hallucination.run.HallucinationProcessor")
    def test_main_processor_exception(self, mock_processor_class: MagicMock) -> None:
        """Test main with processor raising an exception."""
        mock_processor = MagicMock()
        mock_processor.process.side_effect = Exception("Processing error")
        mock_processor_class.return_value = mock_processor

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "run.py",
                    "--augmented-video-path",
                    "./checks/sample_data/augmented.mp4",
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
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
                    "--original-video-path",
                    "./checks/sample_data/original.mp4",
                    "--output-dir",
                    tmpdir,
                    "--config-dir",
                    "./checks/hallucination/tests/data",
                    "--config",
                    "config",
                    "--verbose",
                    "DEBUG",
                ],
            ):
                main()

            # Should complete successfully
            output_files = list(Path(tmpdir).glob("*.json"))
            self.assertGreater(len(output_files), 0)

    def test_main_config_none_handling(self) -> None:
        """Test main when config loading returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("checks.hallucination.run.load_config") as mock_load:
                mock_load.return_value = None

                with patch(
                    "sys.argv",
                    [
                        "run.py",
                        "--augmented-video-path",
                        "./checks/hallucination/tests/data/augmented.mp4",
                        "--original-video-path",
                        "./checks/hallucination/tests/data/original.mp4",
                        "--output-dir",
                        tmpdir,
                    ],
                ):
                    with self.assertRaises(SystemExit) as cm:
                        main()
                    self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
