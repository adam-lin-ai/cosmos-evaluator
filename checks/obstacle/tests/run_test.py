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

"""Unit tests for run.py CLI script."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from checks.obstacle.run import main, parse_args
from utils.bazel import get_runfiles_path


class TestObstacleRun(unittest.TestCase):
    """Test case for the obstacle correspondence run.py CLI script."""

    def setUp(self):
        """Set up test fixtures."""
        # Get path to the RDS data zip file
        self.rds_zip_path = get_runfiles_path("checks/sample_data/rds_hq.zip")
        self.assertIsNotNone(self.rds_zip_path, "Could not locate rds_hq.zip file")
        self.assertTrue(os.path.exists(self.rds_zip_path), f"RDS zip file does not exist: {self.rds_zip_path}")

        # Get path to the video file
        self.video_path = get_runfiles_path("checks/sample_data/camera_front_wide_120fov.mp4")
        self.assertIsNotNone(self.video_path, "Could not locate camera_front_wide_120fov.mp4 file")
        self.assertTrue(os.path.exists(self.video_path), f"Video file does not exist: {self.video_path}")

        self.world_video_path = get_runfiles_path(
            "checks/sample_data/0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000_0.mp4"
        )
        self.assertIsNotNone(
            self.world_video_path, "Could not locate 0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000_0.mp4 file"
        )
        self.assertTrue(
            os.path.exists(self.world_video_path), f"World video file does not exist: {self.world_video_path}"
        )

    def test_obstacle_run_direct(self):
        """Test run.py by calling main() function directly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract RDS data
            rds_extracted_dir = os.path.join(temp_dir, "rds_hq")
            os.makedirs(rds_extracted_dir, exist_ok=True)

            with zipfile.ZipFile(self.rds_zip_path, "r") as zip_ref:
                for info in zip_ref.infolist():
                    if ".." in info.filename or info.filename.startswith("/"):
                        raise ValueError(f"Zip file contains unsafe path: {info.filename}")
                zip_ref.extractall(rds_extracted_dir)

            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            # Prepare command arguments (simulate CLI args)
            test_args = [
                "run.py",  # script name (sys.argv[0])
                "local",
                "--input_data",
                rds_extracted_dir,
                "--clip_id",
                "ba3d3d1c-2eaf-4e0d-8b6e-f3f5b0c4cf80_26015042000",
                "--camera_name",
                "camera_front_wide_120fov",
                "--video_path",
                self.video_path,
                "--world_video_path",
                self.world_video_path,
                "--output_dir",
                output_dir,
                "--verbose",
                "INFO",
                "--trial",
                "5",  # Process only 5 frames for faster testing
                "--model_device",
                "cpu",  # Test on cpu in case gpu is not available
            ]

            # Mock sys.argv and call main() directly
            with patch.object(sys, "argv", test_args):
                try:
                    main()

                    # Check that output directory contains some files
                    output_files = list(Path(output_dir).rglob("*"))
                    self.assertGreater(len(output_files), 0, f"No output files generated in {output_dir}")

                    print(f"✅ Test passed! Generated {len(output_files)} output files")

                except SystemExit as e:
                    if e.code != 0:
                        self.fail(f"main() exited with non-zero code: {e.code}")
                    # SystemExit with code 0 is normal for successful CLI tools
                except (RuntimeError, ValueError, ImportError) as e:
                    self.fail(f"main() raised an exception: {e}")

    def test_argument_parsing(self):
        """Test that CLI arguments are parsed correctly."""

        test_args = [
            "run.py",
            "local",
            "--input_data",
            "/path/to/data",
            "--clip_id",
            "test_clip_123",
            "--camera_name",
            "test_camera",
            "--video_path",
            "/path/to/video.mp4",
            "--world_video_path",
            "/path/to/world.mp4",
            "--output_dir",
            "/path/to/output",
            "--verbose",
            "DEBUG",
            "--trial",
            "10",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

            # Verify arguments were parsed correctly
            self.assertEqual(args.mode, "local")
            self.assertEqual(args.input_data, "/path/to/data")
            self.assertEqual(args.clip_id, "test_clip_123")
            self.assertEqual(args.camera_name, "test_camera")
            self.assertEqual(args.video_path, "/path/to/video.mp4")
            self.assertEqual(args.world_video_path, "/path/to/world.mp4")
            self.assertEqual(args.output_dir, "/path/to/output")
            self.assertEqual(args.verbose, "DEBUG")
            self.assertEqual(args.trial, 10)


if __name__ == "__main__":
    unittest.main()
