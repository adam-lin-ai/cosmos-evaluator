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

"""Unit tests for utils module."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

from services import utils
from services.framework.storage_providers.s3_url_utils import S3UrlComponents, is_presigned_url_s3, parse_s3_url


class TestExtractClipId(unittest.TestCase):
    """Test cases for extract_clip_id function."""

    def test_extract_clip_id_success(self):
        """Test successful clip ID extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create a test tar file
            tar_file = test_dir / "1408ad50-5d9c-463d-a6c9-9e042c7cbcb3_17280960000.tar"
            tar_file.touch()

            clip_id = utils.extract_clip_id(test_dir)
            self.assertEqual(clip_id, "1408ad50-5d9c-463d-a6c9-9e042c7cbcb3_17280960000")

    def test_extract_clip_id_nested_tar(self):
        """Test clip ID extraction from nested subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create nested directory structure
            nested_dir = test_dir / "subdir" / "nested"
            nested_dir.mkdir(parents=True)

            # Create tar file in nested directory
            tar_file = nested_dir / "test_clip_12345.tar"
            tar_file.touch()

            clip_id = utils.extract_clip_id(test_dir)
            self.assertEqual(clip_id, "test_clip_12345")

    def test_extract_clip_id_multiple_tars(self):
        """Test clip ID extraction when multiple tar files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create multiple tar files
            tar_file1 = test_dir / "first_clip_123.tar"
            tar_file2 = test_dir / "second_clip_456.tar"
            tar_file1.touch()
            tar_file2.touch()

            clip_id = utils.extract_clip_id(test_dir)
            # Should return the first one found (order may vary)
            self.assertIn(clip_id, ["first_clip_123", "second_clip_456"])

    def test_extract_clip_id_no_tar_files(self):
        """Test extract_clip_id when no tar files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create some non-tar files
            (test_dir / "test.txt").touch()
            (test_dir / "data.json").touch()

            with self.assertRaises(ValueError) as context:
                utils.extract_clip_id(test_dir)

            self.assertIn("No .tar files found", str(context.exception))

    def test_extract_clip_id_empty_directory(self):
        """Test extract_clip_id with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            with self.assertRaises(ValueError) as context:
                utils.extract_clip_id(test_dir)

            self.assertIn("No .tar files found", str(context.exception))


class TestS3UrlComponents(unittest.TestCase):
    """Test cases for S3UrlComponents dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test S3UrlComponents has expected fields."""
        components = S3UrlComponents(bucket_name="my-bucket", s3_key="path/to/file.mp4")
        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "path/to/file.mp4")
        self.assertEqual(components.region_name, "us-east-1")  # default

    def test_custom_region(self) -> None:
        """Test S3UrlComponents with custom region."""
        components = S3UrlComponents(bucket_name="my-bucket", s3_key="path/to/file.mp4", region_name="eu-west-1")
        self.assertEqual(components.region_name, "eu-west-1")


class TestParseS3Url(unittest.TestCase):
    """Test cases for parse_s3_url function."""

    def test_virtual_hosted_style_without_region(self) -> None:
        """Test parsing virtual-hosted style URL without region."""
        url = "https://test-bucket.s3.amazonaws.com/generated_videos/video.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "test-bucket")
        self.assertEqual(components.s3_key, "generated_videos/video.mp4")
        self.assertEqual(components.region_name, "us-east-1")  # default

    def test_virtual_hosted_style_with_region(self) -> None:
        """Test parsing virtual-hosted style URL with region."""
        url = "https://my-bucket.s3.us-west-2.amazonaws.com/path/to/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "path/to/file.mp4")
        self.assertEqual(components.region_name, "us-west-2")

    def test_virtual_hosted_style_with_eu_region(self) -> None:
        """Test parsing virtual-hosted style URL with EU region."""
        url = "https://bucket-name.s3.eu-central-1.amazonaws.com/folder/document.pdf"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "bucket-name")
        self.assertEqual(components.s3_key, "folder/document.pdf")
        self.assertEqual(components.region_name, "eu-central-1")

    def test_path_style_with_region(self) -> None:
        """Test parsing path-style URL with region."""
        url = "https://s3.us-west-1.amazonaws.com/my-bucket/path/to/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "path/to/file.mp4")
        self.assertEqual(components.region_name, "us-west-1")

    def test_path_style_without_region(self) -> None:
        """Test parsing path-style URL without explicit region."""
        url = "https://s3.amazonaws.com/my-bucket/path/to/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "path/to/file.mp4")
        self.assertEqual(components.region_name, "us-east-1")  # default

    def test_s3_uri_scheme(self) -> None:
        """Test parsing s3:// URI scheme."""
        url = "s3://my-bucket/path/to/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "path/to/file.mp4")
        self.assertEqual(components.region_name, "us-east-1")  # default

    def test_s3_uri_with_nested_path(self) -> None:
        """Test parsing s3:// URI with deeply nested path."""
        url = "s3://bucket/a/b/c/d/e/file.txt"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "bucket")
        self.assertEqual(components.s3_key, "a/b/c/d/e/file.txt")

    def test_presigned_url_extracts_region_from_credential(self) -> None:
        """Test parsing presigned URL extracts region from X-Amz-Credential."""
        url = (
            "https://test-bucket.s3.amazonaws.com/generated_videos/video.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEYID1234%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Date=20260109T224329Z"
            "&X-Amz-Expires=7200"
            "&X-Amz-SignedHeaders=host"
            "&X-Amz-Signature=d36b4e329f7f22979e40133f2a60cf65367b975f862e486b69d63033f9f23937"
        )
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "test-bucket")
        self.assertEqual(components.s3_key, "generated_videos/video.mp4")
        self.assertEqual(components.region_name, "us-east-1")

    def test_presigned_url_with_different_region_in_credential(self) -> None:
        """Test presigned URL with non-default region in credential."""
        url = (
            "https://my-bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEY123456%2F20260109%2Fap-southeast-1%2Fs3%2Faws4_request"
        )
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my-bucket")
        self.assertEqual(components.s3_key, "file.mp4")
        self.assertEqual(components.region_name, "ap-southeast-1")

    def test_url_with_special_characters_in_key(self) -> None:
        """Test parsing URL with special characters in S3 key."""
        url = "https://bucket.s3.amazonaws.com/path/file%20with%20spaces.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "bucket")
        # URL-encoded characters should be decoded to match actual S3 object keys
        self.assertEqual(components.s3_key, "path/file with spaces.mp4")

    def test_http_scheme_supported(self) -> None:
        """Test that HTTP scheme is also supported."""
        url = "http://bucket.s3.amazonaws.com/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "bucket")
        self.assertEqual(components.s3_key, "file.mp4")

    def test_unsupported_scheme_raises_error(self) -> None:
        """Test that unsupported URL schemes raise ValueError."""
        url = "ftp://bucket.s3.amazonaws.com/file.mp4"

        with self.assertRaises(ValueError) as context:
            parse_s3_url(url)

        self.assertIn("Unsupported URL scheme", str(context.exception))
        self.assertIn("ftp", str(context.exception))

    def test_unrecognized_format_raises_error(self) -> None:
        """Test that unrecognized S3 URL formats raise ValueError."""
        url = "https://example.com/some/path/file.mp4"

        with self.assertRaises(ValueError) as context:
            parse_s3_url(url)

        self.assertIn("Unrecognized S3 URL format", str(context.exception))

    def test_path_style_missing_key_raises_error(self) -> None:
        """Test that path-style URL without key raises ValueError."""
        url = "https://s3.us-west-1.amazonaws.com/bucket-only"

        with self.assertRaises(ValueError) as context:
            parse_s3_url(url)

        self.assertIn("Invalid path-style S3 URL", str(context.exception))
        self.assertIn("missing key", str(context.exception))

    def test_path_style_no_region_missing_key_raises_error(self) -> None:
        """Test that path-style URL without region and key raises ValueError."""
        url = "https://s3.amazonaws.com/bucket-only"

        with self.assertRaises(ValueError) as context:
            parse_s3_url(url)

        self.assertIn("Invalid path-style S3 URL", str(context.exception))

    def test_bucket_name_with_dots(self) -> None:
        """Test parsing bucket name containing dots."""
        url = "https://my.bucket.name.s3.amazonaws.com/file.mp4"
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "my.bucket.name")
        self.assertEqual(components.s3_key, "file.mp4")

    def test_real_world_presigned_url(self) -> None:
        """Test with a real-world presigned URL format."""
        url = (
            "https://test-bucket.s3.amazonaws.com/"
            "generated_videos/video_generated_20260109221406_gtfyl5nr.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEYID1234%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Date=20260109T224329Z"
            "&X-Amz-Expires=7200"
            "&X-Amz-SignedHeaders=host"
            "&X-Amz-Signature=d36b4e329f7f22979e40133f2a60cf65367b975f862e486b69d63033f9f23937"
        )
        components = parse_s3_url(url)

        self.assertEqual(components.bucket_name, "test-bucket")
        self.assertEqual(components.s3_key, "generated_videos/video_generated_20260109221406_gtfyl5nr.mp4")
        self.assertEqual(components.region_name, "us-east-1")


class TestIsPresignedUrlS3(unittest.TestCase):
    """Test cases for is_presigned_url_s3 function."""

    def test_presigned_url_returns_true(self) -> None:
        """Test that a valid presigned URL returns True."""
        url = (
            "https://test-bucket.s3.amazonaws.com/generated_videos/video.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEYID1234%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Date=20260109T224329Z"
            "&X-Amz-Expires=7200"
            "&X-Amz-SignedHeaders=host"
            "&X-Amz-Signature=d36b4e329f7f22979e40133f2a60cf65367b975f862e486b69d63033f9f23937"
        )
        self.assertTrue(is_presigned_url_s3(url))

    def test_regular_s3_url_returns_false(self) -> None:
        """Test that a regular S3 URL without presigning returns False."""
        url = "https://test-bucket.s3.amazonaws.com/generated_videos/video.mp4"
        self.assertFalse(is_presigned_url_s3(url))

    def test_s3_uri_returns_false(self) -> None:
        """Test that s3:// URI returns False."""
        url = "s3://my-bucket/path/to/file.mp4"
        self.assertFalse(is_presigned_url_s3(url))

    def test_url_with_unrelated_query_params_returns_false(self) -> None:
        """Test that URL with unrelated query params returns False."""
        url = "https://bucket.s3.amazonaws.com/file.mp4?version=123&format=mp4"
        self.assertFalse(is_presigned_url_s3(url))

    def test_url_missing_signature_returns_false(self) -> None:
        """Test that URL missing X-Amz-Signature returns False."""
        url = (
            "https://bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEY123456%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
        )
        self.assertFalse(is_presigned_url_s3(url))

    def test_url_missing_credential_returns_false(self) -> None:
        """Test that URL missing X-Amz-Credential returns False."""
        url = "https://bucket.s3.amazonaws.com/file.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc123"
        self.assertFalse(is_presigned_url_s3(url))

    def test_url_missing_algorithm_returns_false(self) -> None:
        """Test that URL missing X-Amz-Algorithm returns False."""
        url = (
            "https://bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Credential=TESTEXAMPLEKEY123456%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=abc123"
        )
        self.assertFalse(is_presigned_url_s3(url))

    def test_minimal_presigned_url_returns_true(self) -> None:
        """Test minimal presigned URL with only required params returns True."""
        url = (
            "https://bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEY123456%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=abc123def456"
        )
        self.assertTrue(is_presigned_url_s3(url))

    def test_non_s3_url_with_aws_params_returns_false(self) -> None:
        """Test that function validates hostname is S3, not just query params."""
        url = (
            "https://example.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=TESTEXAMPLEKEY123456%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=abc123"
        )
        self.assertFalse(is_presigned_url_s3(url))

    def test_empty_query_string_returns_false(self) -> None:
        """Test that URL with empty query string returns False."""
        url = "https://bucket.s3.amazonaws.com/file.mp4?"
        self.assertFalse(is_presigned_url_s3(url))


class TestGetContentsFromRunfile(unittest.TestCase):
    """Test cases for get_contents_from_runfile function."""

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_success(self, mock_runfiles_create, mock_path_exists):
        """Test successful contents retrieval from runfiles.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_contents = "1.2.3"
        test_repo_path = "services/obstacle_correspondence/file.txt"

        with patch("builtins.open", mock_open(read_data=test_contents)) as mock_file:
            result = utils.get_contents_from_runfile(test_repo_path)

            # Verify result
            self.assertEqual(result, test_contents)

            # Verify runfiles calls
            mock_runfiles_create.assert_called_once()
            mock_runfiles.Rlocation.assert_called_once_with("_main/services/obstacle_correspondence/file.txt")

            # Verify file was opened and read
            mock_file.assert_called_once_with("/actual/path/to/file.txt", "r")

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_with_whitespace(self, mock_runfiles_create, mock_path_exists):
        """Test contents retrieval strips whitespace correctly.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        # Test contents with surrounding whitespace
        test_contents_with_whitespace = "  1.2.3-rc1  \n"
        expected_contents = "1.2.3-rc1"
        test_repo_path = "/services/test/file.txt"

        with patch("builtins.open", mock_open(read_data=test_contents_with_whitespace)):
            result = utils.get_contents_from_runfile(test_repo_path)
            self.assertEqual(result, expected_contents)

    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_path_not_found(self, mock_runfiles_create):
        """Test FileNotFoundError when runfiles path is not found.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
        """
        # Setup mocks - Rlocation returns None when path not found
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = None
        mock_runfiles_create.return_value = mock_runfiles

        test_repo_path = "/nonexistent/file.txt"

        with self.assertRaises(FileNotFoundError) as context:
            utils.get_contents_from_runfile(test_repo_path)

        self.assertIn("File not found", str(context.exception))
        self.assertIn(test_repo_path, str(context.exception))

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_empty_file(self, mock_runfiles_create, mock_path_exists):
        """Test ValueError when file is empty.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_repo_path = "services/test/file.txt"

        # Test with empty file
        with patch("builtins.open", mock_open(read_data="")):
            with self.assertRaises(ValueError) as context:
                utils.get_contents_from_runfile(test_repo_path)

            self.assertIn("File is empty", str(context.exception))

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_whitespace_only_file(self, mock_runfiles_create, mock_path_exists):
        """Test ValueError when file contains only whitespace.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_repo_path = "services/test/file.txt"

        # Test with whitespace-only file
        with patch("builtins.open", mock_open(read_data="   \n\t  ")):
            with self.assertRaises(ValueError) as context:
                utils.get_contents_from_runfile(test_repo_path)

            self.assertIn("File is empty", str(context.exception))

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_file_read_error(self, mock_runfiles_create, mock_path_exists):
        """Test IOError when file cannot be read due to IO error.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_repo_path = "services/test/file.txt"

        # Mock file open to raise IOError
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with self.assertRaises(IOError) as context:
                utils.get_contents_from_runfile(test_repo_path)

            self.assertIn("Permission denied", str(context.exception))

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_os_error(self, mock_runfiles_create, mock_path_exists):
        """Test OSError when file access causes OSError.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_repo_path = "services/test/file.txt"

        # Mock file open to raise OSError
        with patch("builtins.open", side_effect=OSError("File not accessible")):
            with self.assertRaises(OSError) as context:
                utils.get_contents_from_runfile(test_repo_path)

            self.assertIn("File not accessible", str(context.exception))

    @patch("utils.bazel.path.exists")
    @patch("utils.bazel.runfiles.Create")
    def test_get_contents_from_runfile_complex_contents(self, mock_runfiles_create, mock_path_exists):
        """Test with complex contents formats.

        Args:
            mock_runfiles_create: Mock object for runfiles.Create
            mock_path_exists: Mock object for path.exists
        """
        # Setup mocks
        mock_runfiles = Mock()
        mock_runfiles.Rlocation.return_value = "/actual/path/to/file.txt"
        mock_runfiles_create.return_value = mock_runfiles
        mock_path_exists.return_value = True

        test_repo_path = "services/test/file.txt"

        # Test with complex contents
        complex_contents = ["1.2.3-alpha.1+build.123", "2.0.0-rc.1", "v3.1.4", "0.1.0-dev", "1.0.0+20240101.abc123"]

        for contents in complex_contents:
            with patch("builtins.open", mock_open(read_data=contents)):
                result = utils.get_contents_from_runfile(test_repo_path)
                self.assertEqual(result, contents)

    def test_get_contents_from_runfile_signature(self):
        """Test function signature and documentation."""
        import inspect

        # Verify function exists and has correct signature
        sig = inspect.signature(utils.get_contents_from_runfile)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["repo_path"])

        # Verify it's properly documented
        self.assertIsNotNone(utils.get_contents_from_runfile.__doc__)
        self.assertIn(
            "Read the contents of a file in the Bazel runfiles directory.", utils.get_contents_from_runfile.__doc__
        )


if __name__ == "__main__":
    unittest.main()
