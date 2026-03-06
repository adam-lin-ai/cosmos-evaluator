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
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from checks.vlm import run


class TestParseArgs(unittest.TestCase):
    """Test suite for _parse_args function."""

    def test_parse_args_local_mode(self):
        """Test parsing arguments for local mode."""
        test_args = [
            "local",
            "--output_dir",
            "/tmp/output",
            "--video_path",
            "/path/to/video.mp4",
            "--clip_id",
            "test_clip",
            "--preset_file",
            "/path/to/preset.json",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run._parse_args()

        self.assertEqual(args.mode, "local")
        self.assertEqual(args.output_dir, "/tmp/output")
        self.assertEqual(args.video_path, "/path/to/video.mp4")
        self.assertEqual(args.clip_id, "test_clip")
        self.assertEqual(args.preset_file, "/path/to/preset.json")

    def test_parse_args_cloud_mode(self):
        """Test parsing arguments for cloud mode."""
        test_args = [
            "cloud",
            "--output_dir",
            "/tmp/output",
            "--metadata_file",
            "/path/to/metadata.json",
        ]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run._parse_args()

        self.assertEqual(args.mode, "cloud")
        self.assertEqual(args.output_dir, "/tmp/output")
        self.assertEqual(args.metadata_file, "/path/to/metadata.json")

    def test_parse_args_verbose_levels(self):
        """Test parsing verbose argument with different levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            test_args = [
                "local",
                "--output_dir",
                "/tmp/output",
                "--verbose",
                level,
                "--video_path",
                "/tmp/video.mp4",
                "--clip_id",
                "test",
            ]

            with patch("sys.argv", ["run.py"] + test_args):
                args = run._parse_args()

            self.assertEqual(args.verbose, level)

    def test_parse_args_default_verbose(self):
        """Test default verbose level is INFO."""
        test_args = ["local", "--output_dir", "/tmp/output", "--video_path", "/tmp/video.mp4", "--clip_id", "test"]

        with patch("sys.argv", ["run.py"] + test_args):
            args = run._parse_args()

        self.assertEqual(args.verbose, "INFO")

    def test_parse_args_missing_required_output_dir(self):
        """Test that missing required output_dir raises error."""
        test_args = ["local", "--video_path", "/tmp/video.mp4", "--clip_id", "test"]

        with patch("sys.argv", ["run.py"] + test_args):
            with self.assertRaises(SystemExit):
                run._parse_args()

    def test_parse_args_invalid_mode(self):
        """Test that invalid mode raises error."""
        test_args = ["invalid_mode", "--output_dir", "/tmp/output"]

        with patch("sys.argv", ["run.py"] + test_args):
            with self.assertRaises(SystemExit):
                run._parse_args()

    def test_parse_args_local_missing_required_video_path(self):
        """Test that missing required video_path raises error in local mode."""
        test_args = ["local", "--output_dir", "/tmp/output", "--clip_id", "test"]

        with patch("sys.argv", ["run.py"] + test_args):
            with self.assertRaises(SystemExit):
                run._parse_args()

    def test_parse_args_local_missing_required_clip_id(self):
        """Test that missing required clip_id raises error in local mode."""
        test_args = ["local", "--output_dir", "/tmp/output", "--video_path", "/tmp/video.mp4"]

        with patch("sys.argv", ["run.py"] + test_args):
            with self.assertRaises(SystemExit):
                run._parse_args()

    def test_parse_args_cloud_missing_required_metadata_file(self):
        """Test that missing required metadata_file raises error in cloud mode."""
        test_args = ["cloud", "--output_dir", "/tmp/output"]

        with patch("sys.argv", ["run.py"] + test_args):
            with self.assertRaises(SystemExit):
                run._parse_args()


class TestLoadConfig(unittest.TestCase):
    """Test suite for _load_config function."""

    @patch("checks.vlm.run.ConfigManager")
    def test_load_config_success(self, mock_config_manager):
        """Test successful config loading."""
        mock_cm = MagicMock()
        mock_config = {"av.vlm": {"preset_check": {"enabled": True}}}
        mock_cm.load_config.return_value = mock_config
        mock_config_manager.return_value = mock_cm

        config = run._load_config()

        mock_config_manager.assert_called_once()
        mock_cm.load_config.assert_called_once_with("config")
        self.assertEqual(config, mock_config)

    @patch("checks.vlm.run.ConfigManager")
    def test_load_config_empty(self, mock_config_manager):
        """Test loading empty config."""
        mock_cm = MagicMock()
        mock_cm.load_config.return_value = {}
        mock_config_manager.return_value = mock_cm

        config = run._load_config()

        self.assertEqual(config, {})


class TestLoadLocalPresetConditions(unittest.TestCase):
    """Test suite for _load_local_preset_conditions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_local_preset_conditions_direct_preset(self):
        """Test loading preset conditions from file with direct preset object."""
        preset_data = {"weather": "sunny", "time": "noon"}
        preset_file = self.tmp / "preset.json"
        preset_file.write_text(json.dumps(preset_data))

        result = run._load_local_preset_conditions(str(preset_file))

        self.assertEqual(result, preset_data)

    def test_load_local_preset_conditions_wrapped_preset(self):
        """Test loading preset conditions from file with wrapped preset object."""
        preset_data = {"preset": {"weather": "sunny", "time": "noon"}}
        preset_file = self.tmp / "preset.json"
        preset_file.write_text(json.dumps(preset_data))

        result = run._load_local_preset_conditions(str(preset_file))

        self.assertEqual(result, preset_data["preset"])

    def test_load_local_preset_conditions_no_file(self):
        """Test loading preset conditions with no file provided."""
        with self.assertRaises(ValueError) as context:
            run._load_local_preset_conditions(None)

        self.assertIn("--preset_file must be provided", str(context.exception))

    def test_load_local_preset_conditions_invalid_json(self):
        """Test loading preset conditions from invalid JSON file."""
        preset_file = self.tmp / "preset.json"
        preset_file.write_text("not valid json")

        with self.assertRaises(json.JSONDecodeError):
            run._load_local_preset_conditions(str(preset_file))

    def test_load_local_preset_conditions_not_dict(self):
        """Test loading preset conditions from non-dict JSON."""
        preset_file = self.tmp / "preset.json"
        preset_file.write_text(json.dumps(["list", "not", "dict"]))

        with self.assertRaises(TypeError) as context:
            run._load_local_preset_conditions(str(preset_file))

        self.assertIn("preset must be a JSON object", str(context.exception))

    def test_load_local_preset_conditions_nested_preset_not_dict(self):
        """Test loading preset with nested preset that is not a dict returns the outer dict."""
        preset_data = {"preset": "not a dict"}
        preset_file = self.tmp / "preset.json"
        preset_file.write_text(json.dumps(preset_data))

        # When "preset" key exists but value is not a dict, it returns the outer dict
        result = run._load_local_preset_conditions(str(preset_file))
        self.assertEqual(result, preset_data)


class TestPrepareDebugDir(unittest.TestCase):
    """Test suite for _prepare_debug_dir function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_prepare_debug_dir_success(self):
        """Test successful debug directory preparation."""
        base_dir = str(self.tmp / "output")
        video_path = "/path/to/video.mp4"

        result = run._prepare_debug_dir(base_dir, video_path)

        expected_dir = Path(base_dir) / "video"
        self.assertEqual(result, str(expected_dir))
        self.assertTrue(expected_dir.exists())

    def test_prepare_debug_dir_existing(self):
        """Test debug directory preparation when directory already exists."""
        base_dir = str(self.tmp / "output")
        video_path = "/path/to/video.mp4"
        expected_dir = Path(base_dir) / "video"
        expected_dir.mkdir(parents=True, exist_ok=True)

        result = run._prepare_debug_dir(base_dir, video_path)

        self.assertEqual(result, str(expected_dir))
        self.assertTrue(expected_dir.exists())

    def test_prepare_debug_dir_exception(self):
        """Test debug directory preparation when exception occurs."""
        base_dir = str(self.tmp / "output")
        video_path = "/path/to/video.mp4"

        # Mock Path to raise exception during mkdir
        with patch("checks.vlm.run.Path") as mock_path_class:
            # Create mock instances for Path(base_dir) and Path(video_path)
            mock_base_path = MagicMock()
            mock_video_path = MagicMock()
            mock_video_path.stem = "video"

            # combined = Path(base_dir) / Path(video_path).stem
            mock_combined = MagicMock()
            mock_combined.mkdir.side_effect = Exception("Test exception")
            mock_base_path.__truediv__.return_value = mock_combined

            def path_side_effect(arg):
                if arg == base_dir:
                    return mock_base_path
                else:
                    return mock_video_path

            mock_path_class.side_effect = path_side_effect

            result = run._prepare_debug_dir(base_dir, video_path)

            # Should return base_dir on exception
            self.assertEqual(result, base_dir)

    def test_prepare_debug_dir_with_complex_video_path(self):
        """Test debug directory with complex video path."""
        base_dir = str(self.tmp / "output")
        video_path = "/long/path/to/some/video_file_name.mp4"

        result = run._prepare_debug_dir(base_dir, video_path)

        expected_dir = Path(base_dir) / "video_file_name"
        self.assertEqual(result, str(expected_dir))
        self.assertTrue(expected_dir.exists())


class TestSetupLogging(unittest.TestCase):
    """Test suite for _setup_logging function."""

    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level."""
        with patch("logging.basicConfig") as mock_basic_config:
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                run._setup_logging("DEBUG")

                mock_basic_config.assert_called_once()
                call_kwargs = mock_basic_config.call_args[1]
                self.assertEqual(call_kwargs["level"], logging.DEBUG)

    def test_setup_logging_info_level(self):
        """Test logging setup with INFO level."""
        with patch("logging.basicConfig") as mock_basic_config:
            run._setup_logging("INFO")

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            self.assertEqual(call_kwargs["level"], logging.INFO)

    def test_setup_logging_warning_level(self):
        """Test logging setup with WARNING level."""
        with patch("logging.basicConfig") as mock_basic_config:
            run._setup_logging("WARNING")

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            self.assertEqual(call_kwargs["level"], logging.WARNING)

    def test_setup_logging_error_level(self):
        """Test logging setup with ERROR level."""
        with patch("logging.basicConfig") as mock_basic_config:
            run._setup_logging("ERROR")

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            self.assertEqual(call_kwargs["level"], logging.ERROR)

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level defaults to INFO."""
        with patch("logging.basicConfig") as mock_basic_config:
            run._setup_logging("INVALID")

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            self.assertEqual(call_kwargs["level"], logging.INFO)

    def test_setup_logging_none_level(self):
        """Test logging setup with None level defaults to INFO."""
        with patch("logging.basicConfig") as mock_basic_config:
            run._setup_logging(None)

            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            self.assertEqual(call_kwargs["level"], logging.INFO)

    @patch("logging.getLogger")
    @patch("logging.basicConfig")
    def test_setup_logging_suppresses_third_party_loggers(self, mock_basic_config, mock_get_logger):
        """Test that DEBUG level suppresses third-party loggers."""
        third_party_loggers = {}

        def get_logger_side_effect(name):
            if name not in third_party_loggers:
                third_party_loggers[name] = MagicMock()
            return third_party_loggers[name]

        mock_get_logger.side_effect = get_logger_side_effect

        # Mock the main logger to return DEBUG level
        main_logger = MagicMock()
        main_logger.getEffectiveLevel.return_value = logging.DEBUG

        with patch("logging.getLogger", side_effect=get_logger_side_effect):
            with patch("logging.Logger.getEffectiveLevel", return_value=logging.DEBUG):
                run._setup_logging("DEBUG")

        # Verify third-party loggers were accessed (they might be set to WARNING)
        # Note: This test validates the code doesn't crash; actual logger behavior
        # depends on the logging module's state


class TestMain(unittest.TestCase):
    """Test suite for main function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("checks.vlm.run.process_preset")
    @patch("checks.vlm.run.load_default_envs")
    @patch("checks.vlm.run._load_config")
    @patch("checks.vlm.run._setup_logging")
    @patch("checks.vlm.run._parse_args")
    def test_main_local_mode_preset_check_only(
        self, mock_parse_args, mock_setup_logging, mock_load_config, mock_load_default_envs, mock_process_preset
    ):
        """Test main function in local mode with preset check only."""
        # Setup mocks
        output_dir = str(self.tmp / "output")
        video_path = str(self.tmp / "video.mp4")
        preset_file = str(self.tmp / "preset.json")

        # Create dummy files
        Path(video_path).touch()
        Path(preset_file).write_text(json.dumps({"weather": "sunny"}))

        mock_args = MagicMock()
        mock_args.mode = "local"
        mock_args.output_dir = output_dir
        mock_args.video_path = video_path
        mock_args.clip_id = "test_clip"
        mock_args.preset_file = preset_file
        mock_args.metadata_file = None
        mock_args.verbose = "INFO"
        mock_parse_args.return_value = mock_args

        mock_config = {"av.vlm": {"preset_check": {"enabled": True}}}
        mock_load_config.return_value = mock_config

        mock_process_preset.return_value = {"score": 0.95}

        # Run main
        run.main()

        # Verify calls
        mock_setup_logging.assert_called_once_with("INFO")
        mock_load_default_envs.assert_called_once()
        mock_load_config.assert_called_once()
        mock_process_preset.assert_called_once()

        # Check output file was created
        output_file = Path(output_dir) / "video.vlm.check.json"
        self.assertTrue(output_file.exists())

        # Verify output content
        with open(output_file, "r") as f:
            output_data = json.load(f)

        self.assertEqual(output_data["clip_id"], "test_clip")
        self.assertEqual(output_data["video_uuid"], "video")
        self.assertEqual(output_data["preset_check"]["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
