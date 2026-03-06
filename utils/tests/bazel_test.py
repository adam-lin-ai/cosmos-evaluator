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

import unittest
from unittest.mock import Mock, patch

from utils.bazel import get_runfiles_path


class TestBazel(unittest.TestCase):
    def test_get_runfiles_path_success(self):
        """Test successful runfiles path resolution."""
        # This test assumes the file exists in the actual runfiles
        result = get_runfiles_path("checks/vlm/config/endpoints.json")
        if result:  # Only assert if runfiles are available
            self.assertTrue(result.endswith("_main/checks/vlm/config/endpoints.json"))

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_mocked_success(self, mock_exists, mock_runfiles):
        """Test successful runfiles path resolution with mocked dependencies."""
        # Setup mocks
        mock_r = Mock()
        mock_r.Rlocation.return_value = "/mock/path/_main/checks/vlm/config/endpoints.json"
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        # Test
        result = get_runfiles_path("checks/vlm/config/endpoints.json")

        # Assertions
        self.assertEqual(result, "/mock/path/_main/checks/vlm/config/endpoints.json")
        mock_runfiles.Create.assert_called_once()
        mock_r.Rlocation.assert_called_once_with("_main/checks/vlm/config/endpoints.json")
        mock_exists.assert_called_once_with("/mock/path/_main/checks/vlm/config/endpoints.json")

    @patch("utils.bazel.runfiles")
    def test_get_runfiles_path_runfiles_create_fails(self, mock_runfiles):
        """Test when runfiles.Create() returns None."""
        mock_runfiles.Create.return_value = None

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)
        mock_runfiles.Create.assert_called_once()

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_file_not_exists(self, mock_exists, mock_runfiles):
        """Test when the resolved file path doesn't exist."""
        # Setup mocks
        mock_r = Mock()
        mock_r.Rlocation.return_value = "/mock/path/_main/nonexistent/file.txt"
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = False

        # Test
        result = get_runfiles_path("nonexistent/file.txt")

        # Assertions
        self.assertIsNone(result)
        mock_exists.assert_called_once_with("/mock/path/_main/nonexistent/file.txt")

    @patch("utils.bazel.runfiles")
    def test_get_runfiles_path_ioerror_exception(self, mock_runfiles):
        """Test exception handling for IOError."""
        mock_runfiles.Create.side_effect = IOError("Mock IO error")

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)

    @patch("utils.bazel.runfiles")
    def test_get_runfiles_path_typeerror_exception(self, mock_runfiles):
        """Test exception handling for TypeError."""
        mock_runfiles.Create.side_effect = TypeError("Mock type error")

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)

    @patch("utils.bazel.runfiles")
    def test_get_runfiles_path_valueerror_exception(self, mock_runfiles):
        """Test exception handling for ValueError."""
        mock_runfiles.Create.side_effect = ValueError("Mock value error")

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_empty_string(self, mock_exists, mock_runfiles):
        """Test with empty string input."""
        # Setup mocks
        mock_r = Mock()
        mock_r.Rlocation.return_value = "/mock/path/_main/"
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        # Test
        result = get_runfiles_path("")

        # Assertions
        self.assertEqual(result, "/mock/path/_main/")
        mock_r.Rlocation.assert_called_once_with("_main/")

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_with_special_characters(self, mock_exists, mock_runfiles):
        """Test with paths containing special characters."""
        # Setup mocks
        mock_r = Mock()
        test_path = "path/with spaces/and-dashes/file_name.json"
        expected_runfiles_path = f"_main/{test_path}"
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        # Test
        result = get_runfiles_path(test_path)

        # Assertions
        self.assertEqual(result, expected_runtime_path)
        mock_r.Rlocation.assert_called_once_with(expected_runfiles_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_leading_slash_sanitized(self, mock_exists, mock_runfiles):
        """Test that a leading slash in repo_path is sanitized correctly."""
        # Setup mocks
        mock_r = Mock()
        input_path = "/checks/vlm/config/endpoints.json"
        expected_runfiles_path = "_main/checks/vlm/config/endpoints.json"  # leading slash removed
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        # Test
        result = get_runfiles_path(input_path)

        # Assertions
        self.assertEqual(result, expected_runtime_path)
        mock_r.Rlocation.assert_called_once_with(expected_runfiles_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_nested_directories(self, mock_exists, mock_runfiles):
        """Test with deeply nested directory paths."""
        # Setup mocks
        mock_r = Mock()
        test_path = "very/deep/nested/directory/structure/file.py"
        expected_runfiles_path = f"_main/{test_path}"
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        # Test
        result = get_runfiles_path(test_path)

        # Assertions
        self.assertEqual(result, expected_runtime_path)
        mock_r.Rlocation.assert_called_once_with(expected_runfiles_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_different_file_extensions(self, mock_exists, mock_runfiles):
        """Test with various file extensions."""
        file_extensions = ["config.json", "script.py", "data.txt", "image.png", "document.pdf", "archive.tar.gz"]

        for filename in file_extensions:
            with self.subTest(filename=filename):
                # Setup mocks for each iteration
                mock_r = Mock()
                test_path = f"test/files/{filename}"
                expected_runfiles_path = f"_main/{test_path}"
                expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
                mock_r.Rlocation.return_value = expected_runtime_path
                mock_runfiles.Create.return_value = mock_r
                mock_exists.return_value = True

                # Test
                result = get_runfiles_path(test_path)

                # Assertions
                self.assertEqual(result, expected_runtime_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_with_external_repo_mocked_success(self, mock_exists, mock_runfiles):
        """Test successful resolution when a non-default external_repo is provided."""
        mock_r = Mock()
        external_repo = "third_party_repo"
        test_path = "some/dir/file.txt"
        expected_runfiles_path = f"{external_repo}/{test_path}"
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        result = get_runfiles_path(test_path, external_repo=external_repo)

        self.assertEqual(result, expected_runtime_path)
        mock_runfiles.Create.assert_called_once()
        mock_r.Rlocation.assert_called_once_with(expected_runfiles_path)
        mock_exists.assert_called_once_with(expected_runtime_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.logger")
    def test_get_runfiles_path_with_external_repo_file_not_exists(self, mock_logger, mock_exists, mock_runfiles):
        """Test logging and None return when file not found for custom external_repo."""
        mock_r = Mock()
        external_repo = "another_repo"
        test_path = "missing/file.txt"
        expected_runfiles_path = f"{external_repo}/{test_path}"
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = False

        result = get_runfiles_path(test_path, external_repo=external_repo)

        self.assertIsNone(result)
        mock_logger.error.assert_called_once_with(f"Runfiles path not found: {expected_runtime_path}")

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    def test_get_runfiles_path_empty_external_repo(self, mock_exists, mock_runfiles):
        """Test that empty external_repo results in joining to just repo_path."""
        mock_r = Mock()
        external_repo = ""
        test_path = "relative/path/file.json"
        expected_runfiles_path = test_path  # path.join("", repo_path) == repo_path
        expected_runtime_path = f"/mock/path/{expected_runfiles_path}"
        mock_r.Rlocation.return_value = expected_runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = True

        result = get_runfiles_path(test_path, external_repo=external_repo)

        self.assertEqual(result, expected_runtime_path)
        mock_r.Rlocation.assert_called_once_with(expected_runfiles_path)

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.logger")
    def test_get_runfiles_path_logging_runfiles_create_fail(self, mock_logger, mock_runfiles):
        """Test that appropriate logging occurs when runfiles.Create() fails."""
        mock_runfiles.Create.return_value = None

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)
        mock_logger.debug.assert_called_once_with("Runfiles.Create() failed")

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.logger")
    def test_get_runfiles_path_logging_file_not_found(self, mock_logger, mock_exists, mock_runfiles):
        """Test that appropriate logging occurs when file is not found."""
        # Setup mocks
        mock_r = Mock()
        runtime_path = "/mock/path/_main/missing/file.txt"
        mock_r.Rlocation.return_value = runtime_path
        mock_runfiles.Create.return_value = mock_r
        mock_exists.return_value = False

        # Test
        result = get_runfiles_path("missing/file.txt")

        # Assertions
        self.assertIsNone(result)
        mock_logger.error.assert_called_once_with(f"Runfiles path not found: {runtime_path}")

    @patch("utils.bazel.runfiles")
    @patch("utils.bazel.logger")
    def test_get_runfiles_path_logging_exception(self, mock_logger, mock_runfiles):
        """Test that appropriate logging occurs when an exception is raised."""
        test_exception = IOError("Test exception message")
        mock_runfiles.Create.side_effect = test_exception

        result = get_runfiles_path("some/file.txt")

        self.assertIsNone(result)
        mock_logger.error.assert_called_once_with(f"Exception while getting runfiles path: {test_exception}")


if __name__ == "__main__":
    unittest.main()
