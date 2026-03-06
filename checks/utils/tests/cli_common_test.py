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

"""Unit tests for checks.utils.cli_common module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from checks.utils.cli_common import load_default_envs, validate_paths


class TestLoadDefaultEnvs(unittest.TestCase):
    """Test suite for load_default_envs function."""

    @patch("checks.utils.cli_common.load_dotenv")
    @patch("checks.utils.cli_common.Path")
    def test_load_default_envs_success(self, mock_path, mock_load_dotenv):
        """Test successful loading of default environment files."""
        # Mock Path instances
        mock_aws_env = MagicMock()
        mock_aws_env.exists.return_value = True
        mock_ag_env = MagicMock()
        mock_ag_env.exists.return_value = True

        # Setup Path mock to return different paths
        def path_side_effect(path_str):
            mock_p = MagicMock()
            if ".aws" in path_str:
                mock_p.expanduser.return_value = mock_aws_env
            else:
                mock_p.expanduser.return_value = mock_ag_env
            return mock_p

        mock_path.side_effect = path_side_effect

        load_default_envs(additional_env_paths=["~/.cosmos_evaluator/.env"])

        # Verify load_dotenv was called twice
        self.assertEqual(mock_load_dotenv.call_count, 2)

    @patch("checks.utils.cli_common.load_dotenv")
    @patch("checks.utils.cli_common.Path")
    def test_load_default_envs_aws_only(self, mock_path, mock_load_dotenv):
        """Test loading only AWS environment file."""
        mock_aws_env = MagicMock()
        mock_aws_env.exists.return_value = True

        def path_side_effect(path_str):
            mock_p = MagicMock()
            mock_p.expanduser.return_value = mock_aws_env
            return mock_p

        mock_path.side_effect = path_side_effect

        load_default_envs()

        # Verify load_dotenv was called once (AWS only)
        self.assertEqual(mock_load_dotenv.call_count, 1)

    @patch("checks.utils.cli_common.Path")
    def test_load_default_envs_aws_file_not_found(self, mock_path):
        """Test loading envs when AWS file is missing."""
        mock_aws_env = MagicMock()
        mock_aws_env.exists.return_value = False

        def path_side_effect(path_str):
            mock_p = MagicMock()
            mock_p.expanduser.return_value = mock_aws_env
            return mock_p

        mock_path.side_effect = path_side_effect

        with self.assertRaises(FileNotFoundError) as context:
            load_default_envs()

        self.assertIn("AWS credentials file not found", str(context.exception))

    @patch("checks.utils.cli_common.load_dotenv")
    @patch("checks.utils.cli_common.Path")
    def test_load_default_envs_additional_file_not_found(self, mock_path, mock_load_dotenv):
        """Test loading envs when additional env file is missing."""
        mock_aws_env = MagicMock()
        mock_aws_env.exists.return_value = True
        mock_ag_env = MagicMock()
        mock_ag_env.exists.return_value = False

        call_count = [0]

        def path_side_effect(path_str):
            mock_p = MagicMock()
            if call_count[0] == 0:
                mock_p.expanduser.return_value = mock_aws_env
            else:
                mock_p.expanduser.return_value = mock_ag_env
            call_count[0] += 1
            return mock_p

        mock_path.side_effect = path_side_effect

        with self.assertRaises(FileNotFoundError) as context:
            load_default_envs(additional_env_paths=["~/.cosmos_evaluator/.env"])

        self.assertIn("Environment file not found", str(context.exception))


class TestValidatePaths(unittest.TestCase):
    """Test suite for validate_paths function."""

    def test_validate_paths_missing_input_data(self):
        """Test validation fails when input_data directory doesn't exist."""
        self.assertFalse(
            validate_paths(
                input_data="/does/not/exist",
                video_path="/also/missing.mp4",
                output_dir="/tmp/output",
            )
        )

    def test_validate_paths_missing_video(self):
        """Test validation fails when video file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "data.txt")
            open(data_file, "a").close()

            self.assertFalse(
                validate_paths(
                    input_data=data_file,
                    video_path="/does/not/exist.mp4",
                    output_dir=tmp_dir,
                )
            )

    def test_validate_paths_success_without_creating_output_dir(self):
        """Test validate_paths with valid paths without creating output dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create temporary files
            data_file = os.path.join(tmp_dir, "data.txt")
            video_file = os.path.join(tmp_dir, "video.mp4")
            output_dir_path = os.path.join(tmp_dir, "output")
            open(data_file, "a").close()
            open(video_file, "a").close()

            self.assertTrue(
                validate_paths(
                    input_data=data_file,
                    video_path=video_file,
                    output_dir=output_dir_path,
                    create_output_dir=False,
                )
            )
            # Verify output dir was NOT created
            self.assertFalse(os.path.exists(output_dir_path))

    def test_validate_paths_success_with_creating_output_dir(self):
        """Test validate_paths with valid paths and creates output dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create temporary files
            data_file = os.path.join(tmp_dir, "data.txt")
            video_file = os.path.join(tmp_dir, "video.mp4")
            output_dir_path = os.path.join(tmp_dir, "output")
            open(data_file, "a").close()
            open(video_file, "a").close()

            self.assertTrue(
                validate_paths(
                    input_data=data_file,
                    video_path=video_file,
                    output_dir=output_dir_path,
                    create_output_dir=True,
                )
            )
            # Verify output dir WAS created
            self.assertTrue(os.path.exists(output_dir_path))


if __name__ == "__main__":
    unittest.main()
