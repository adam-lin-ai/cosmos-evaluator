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

"""Unit tests for metadata_loader module."""

import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from checks.utils.metadata_loader import MetadataLoader


class TestMetadataLoader(unittest.TestCase):
    """Test cases for MetadataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Sample metadata for testing
        self.sample_metadata = {
            "input_video_info": {
                "metadata": {"clip_id": "test_clip_123"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                },
                "rds_hq_data": {"rdshq_archive_created": True, "rdshq_s3_url": "s3://test-bucket/dataset.zip"},
            },
            "generated_video_info": {"output_s3_url": "s3://test-bucket/output_video.mp4"},
            "metadata": {"preset": {"model": "test_model", "version": "1.0"}},
        }

        # Create temporary metadata file
        self.metadata_file = os.path.join(self.temp_dir, "metadata.json")
        with open(self.metadata_file, "w") as f:
            json.dump(self.sample_metadata, f)

        # Mock AWS credentials
        self.aws_env_vars = {
            "AWS_ACCESS_KEY_ID": "test_access_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret_key",
            "AWS_DEFAULT_REGION": "us-west-2",
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_init_success(self, mock_boto3_client):
        """Test successful MetadataLoader initialization."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        self.assertEqual(loader.metadata_json_path, self.metadata_file)
        self.assertEqual(loader.data, self.sample_metadata)
        self.assertEqual(loader.input_video_info, self.sample_metadata["input_video_info"])
        self.assertEqual(loader.generated_video_info, self.sample_metadata["generated_video_info"])
        self.assertEqual(loader.metadata, self.sample_metadata["metadata"])
        self.assertEqual(loader.cosmos_evaluator_result, {})  # Empty when not in metadata
        # cache_dir uses clip_id from metadata
        expected_cache_dir = Path.home() / ".cache" / "cosmos_evaluator" / "s3" / "test_clip_123"
        self.assertEqual(loader.cache_dir, expected_cache_dir)

        # Verify S3 client creation
        mock_boto3_client.assert_called_once_with(
            "s3", aws_access_key_id="test_key", aws_secret_access_key="test_secret", region_name="us-west-2"
        )

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_init_with_cosmos_evaluator_result(self, mock_boto3_client):
        """Test MetadataLoader initialization with cosmos_evaluator_result section."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with cosmos_evaluator_result
        metadata_with_ag = {
            **self.sample_metadata,
            "cosmos_evaluator_result": {"clip_id_with_timestamp": "clip_123_999999"},
        }
        metadata_file = os.path.join(self.temp_dir, "ag_result.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_ag, f)

        loader = MetadataLoader(metadata_file)

        self.assertEqual(loader.cosmos_evaluator_result, {"clip_id_with_timestamp": "clip_123_999999"})

    @patch.dict(
        os.environ,
        {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_SESSION_TOKEN": "test_token"},
    )
    @patch("boto3.client")
    def test_init_with_session_token(self, mock_boto3_client):
        """Test MetadataLoader initialization with session token."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        MetadataLoader(self.metadata_file)

        # Verify S3 client creation with session token
        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2",
            aws_session_token="test_token",
        )

    def test_init_missing_file(self):
        """Test MetadataLoader initialization with missing file."""
        with self.assertRaises(FileNotFoundError):
            MetadataLoader("/nonexistent/file.json")

    def test_init_invalid_json(self):
        """Test MetadataLoader initialization with invalid JSON."""
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, "w") as f:
            f.write("invalid json content")

        with self.assertRaises(json.JSONDecodeError):
            MetadataLoader(invalid_json_file)

    def test_init_missing_aws_credentials(self):
        """Test MetadataLoader initialization with missing AWS credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                MetadataLoader(self.metadata_file)
            self.assertIn("Missing AWS credentials", str(cm.exception))

    # Note: Testing missing boto3 is complex due to import mocking issues.
    # This edge case is covered by the dependency management in the BUILD file.

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_init_default_cache_dir(self, mock_boto3_client):
        """Test MetadataLoader initialization with default cache directory."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)
        # cache_dir includes clip_id from metadata
        expected_cache_dir = Path.home() / ".cache" / "cosmos_evaluator" / "s3" / "test_clip_123"
        self.assertEqual(loader.cache_dir, expected_cache_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_cache_dir(self, mock_boto3_client):
        """Test get_cache_dir method."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)
        cache_dir = loader.get_cache_dir()

        expected_cache_dir = str(Path.home() / ".cache" / "cosmos_evaluator" / "s3" / "test_clip_123")
        self.assertEqual(cache_dir, expected_cache_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_sections(self, mock_boto3_client):
        """Test get_sections method."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)
        sections = loader.get_sections()

        self.assertEqual(sections, self.sample_metadata)
        # Ensure it returns a copy, not the original
        self.assertIsNot(sections, loader.data)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_clip_id_success(self, mock_boto3_client):
        """Test get_clip_id method with valid clip ID."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)
        clip_id = loader.get_clip_id()

        self.assertEqual(clip_id, "test_clip_123")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_clip_id_missing(self, mock_boto3_client):
        """Test get_clip_id method with missing clip ID."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata without clip_id
        metadata_without_clip = {"input_video_info": {"metadata": {}}}
        metadata_file = os.path.join(self.temp_dir, "no_clip.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_without_clip, f)

        loader = MetadataLoader(metadata_file)
        clip_id = loader.get_clip_id()

        self.assertIsNone(clip_id)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_clip_id_with_timestamp_from_cosmos_evaluator_result(self, mock_boto3_client):
        """Test get_clip_id with timestamp returns cosmos_evaluator_result.clip_id_with_timestamp first."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with cosmos_evaluator_result
        metadata_with_ag = {
            "input_video_info": {
                "metadata": {"clip_id": "test_clip"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/test_clip_111111_0.mp4",
                },
            },
            "cosmos_evaluator_result": {"clip_id_with_timestamp": "preferred_clip_999999"},
        }
        metadata_file = os.path.join(self.temp_dir, "ag_clip.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_ag, f)

        loader = MetadataLoader(metadata_file)
        clip_id = loader.get_clip_id(with_timestamp_suffix=True)

        # Should return cosmos_evaluator_result value (priority 1)
        self.assertEqual(clip_id, "preferred_clip_999999")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_clip_id_with_timestamp_from_control_video_url(self, mock_boto3_client):
        """Test get_clip_id with timestamp parses from control video URL when cosmos_evaluator_result missing."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata without cosmos_evaluator_result but with world_model control video
        # clip_id matches the prefix of the control video filename
        metadata_with_wm = {
            "input_video_info": {
                "metadata": {"clip_id": "0835863a-d6b6-442b-b384-bd129254785c"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/0835863a-d6b6-442b-b384-bd129254785c_8777862000_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "wm_clip.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_wm, f)

        loader = MetadataLoader(metadata_file)
        clip_id = loader.get_clip_id(with_timestamp_suffix=True)

        # Should parse from URL using anchored clip_id regex (priority 2)
        self.assertEqual(clip_id, "0835863a-d6b6-442b-b384-bd129254785c_8777862000")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_success(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp_from_control_video_url with valid URL."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata_with_wm = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://cosmos-sdg/rendered_videos/abc123/clip_001_1234567890_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "parse_clip.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_wm, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertEqual(result, "clip_001_1234567890")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_not_world_model(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp returns None when not world_model."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata_not_wm = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "other_type",
                    "control_video_s3_url": "s3://bucket/clip_001_1234567890_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "not_wm.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_not_wm, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertIsNone(result)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_invalid_url_format(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp returns None for invalid URL format."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata_invalid_url = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/invalid_filename.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "invalid_url.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_invalid_url, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertIsNone(result)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_anchored_by_known_clip_id(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp uses known clip_id to anchor regex."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata = {
            "input_video_info": {
                "metadata": {"clip_id": "0835863a-d6b6-442b-b384-bd129254785c"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/path/0835863a-d6b6-442b-b384-bd129254785c_8777862000_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "anchored.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertEqual(result, "0835863a-d6b6-442b-b384-bd129254785c_8777862000")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_anchored_disambiguates_underscore_digits(self, mock_boto3_client):
        """Test anchored regex correctly handles clip_ids containing underscore-digit sequences.

        Without anchoring, a greedy regex could split the clip_id incorrectly
        when it contains patterns like '_123' that look like a timestamp boundary.
        The anchored regex uses the known clip_id to avoid this ambiguity.
        """
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata = {
            "input_video_info": {
                "metadata": {"clip_id": "scene_42_sensor_7"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/scene_42_sensor_7_1700000000_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "disambig.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertEqual(result, "scene_42_sensor_7_1700000000")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_clip_id_not_in_filename(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp returns None when clip_id doesn't appear in filename."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata = {
            "input_video_info": {
                "metadata": {"clip_id": "totally_different_clip"},
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/abc_123_456_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "mismatch.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertIsNone(result)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_parse_clip_id_with_timestamp_no_clip_id_falls_back_to_greedy_regex(self, mock_boto3_client):
        """Test _parse_clip_id_with_timestamp falls back to greedy regex when clip_id is absent."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        metadata = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://bucket/some_clip_1700000000_0.mp4",
                },
            },
        }
        metadata_file = os.path.join(self.temp_dir, "no_clipid.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loader = MetadataLoader(metadata_file)
        result = loader._parse_clip_id_with_timestamp_from_control_video_url()

        self.assertEqual(result, "some_clip_1700000000")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_video_path_success(self, mock_boto3_client):
        """Test get_video_path method with successful download."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        # Mock the download
        expected_path = os.path.join(self.temp_dir, "test-bucket", "output_video.mp4")
        with patch.object(loader, "_download_s3_once", return_value=expected_path) as mock_download:
            video_path = loader.get_video_path()

            self.assertEqual(video_path, expected_path)
            mock_download.assert_called_once_with("s3://test-bucket/output_video.mp4")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_video_path_missing_url(self, mock_boto3_client):
        """Test get_video_path method with missing S3 URL."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata without output_s3_url
        metadata_without_url = {"generated_video_info": {}}
        metadata_file = os.path.join(self.temp_dir, "no_url.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_without_url, f)

        loader = MetadataLoader(metadata_file)

        with self.assertRaises(ValueError) as cm:
            loader.get_video_path()
        self.assertIn("output_s3_url is missing", str(cm.exception))

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_success(self, mock_boto3_client):
        """Test get_wm_video_path method with world model video."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        expected_path = os.path.join(self.temp_dir, "test-bucket", "control_video.mp4")
        with patch.object(loader, "_download_s3_once", return_value=expected_path) as mock_download:
            wm_video_path = loader.get_wm_video_path()

            self.assertEqual(wm_video_path, expected_path)
            mock_download.assert_called_once_with("s3://test-bucket/control_video.mp4")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_not_world_model(self, mock_boto3_client):
        """Test get_wm_video_path method with non-world model control video."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with different control video type
        metadata_no_wm = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "other_type",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                }
            }
        }
        metadata_file = os.path.join(self.temp_dir, "no_wm.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_no_wm, f)

        loader = MetadataLoader(metadata_file)
        wm_video_path = loader.get_wm_video_path()

        self.assertIsNone(wm_video_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_version_match(self, mock_boto3_client):
        """Test get_wm_video_path method with matching version."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with version
        metadata_with_version = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                    "control_video_render_version": "v3",
                }
            }
        }
        metadata_file = os.path.join(self.temp_dir, "wm_v3.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_version, f)

        loader = MetadataLoader(metadata_file)

        expected_path = os.path.join(self.temp_dir, "test-bucket", "control_video.mp4")
        with patch.object(loader, "_download_s3_once", return_value=expected_path):
            wm_video_path = loader.get_wm_video_path(required_version="v3")

        self.assertEqual(wm_video_path, expected_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_version_mismatch(self, mock_boto3_client):
        """Test get_wm_video_path method with mismatched version."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with version v1
        metadata_with_version = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                    "control_video_render_version": "v1",
                }
            }
        }
        metadata_file = os.path.join(self.temp_dir, "wm_v1.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_version, f)

        loader = MetadataLoader(metadata_file)
        wm_video_path = loader.get_wm_video_path(required_version="v3")

        self.assertIsNone(wm_video_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_version_case_insensitive(self, mock_boto3_client):
        """Test get_wm_video_path version comparison is case insensitive."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with uppercase version
        metadata_with_version = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                    "control_video_render_version": "V3",
                }
            }
        }
        metadata_file = os.path.join(self.temp_dir, "wm_V3.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_with_version, f)

        loader = MetadataLoader(metadata_file)

        expected_path = os.path.join(self.temp_dir, "test-bucket", "control_video.mp4")
        with patch.object(loader, "_download_s3_once", return_value=expected_path):
            wm_video_path = loader.get_wm_video_path(required_version="v3")

        self.assertEqual(wm_video_path, expected_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_wm_video_path_version_default_v1(self, mock_boto3_client):
        """Test get_wm_video_path defaults to v1 when version not specified."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata without version (should default to v1)
        metadata_no_version = {
            "input_video_info": {
                "control_video_info": {
                    "control_video_type": "world_model",
                    "control_video_s3_url": "s3://test-bucket/control_video.mp4",
                }
            }
        }
        metadata_file = os.path.join(self.temp_dir, "wm_no_version.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_no_version, f)

        loader = MetadataLoader(metadata_file)
        # Requesting v3 but metadata defaults to v1, should return None
        wm_video_path = loader.get_wm_video_path(required_version="v3")

        self.assertIsNone(wm_video_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_gt_dataset_path_success(self, mock_boto3_client):
        """Test get_gt_dataset_path method with successful extraction."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        archive_path = os.path.join(self.temp_dir, "test-bucket", "dataset.zip")
        extracted_path = os.path.join(self.temp_dir, "extracted")

        with (
            patch.object(loader, "_download_s3_once", return_value=archive_path) as mock_download,
            patch.object(loader, "_extract_archive_once", return_value=extracted_path) as mock_extract,
        ):
            dataset_path = loader.get_gt_dataset_path()

            self.assertEqual(dataset_path, extracted_path)
            mock_download.assert_called_once_with("s3://test-bucket/dataset.zip")
            mock_extract.assert_called_once_with(archive_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_gt_dataset_path_not_created(self, mock_boto3_client):
        """Test get_gt_dataset_path method when archive not created."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with rdshq_archive_created = False
        metadata_no_archive = {
            "input_video_info": {
                "rds_hq_data": {"rdshq_archive_created": False, "rdshq_s3_url": "s3://test-bucket/dataset.zip"}
            }
        }
        metadata_file = os.path.join(self.temp_dir, "no_archive.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_no_archive, f)

        loader = MetadataLoader(metadata_file)
        dataset_path = loader.get_gt_dataset_path()

        self.assertIsNone(dataset_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_preset_success(self, mock_boto3_client):
        """Test get_preset method with valid preset."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)
        preset = loader.get_preset()

        expected_preset = {"model": "test_model", "version": "1.0"}
        self.assertEqual(preset, expected_preset)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_get_preset_invalid_type(self, mock_boto3_client):
        """Test get_preset method with invalid preset type."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        # Create metadata with non-dict preset
        metadata_invalid_preset = {"metadata": {"preset": "not_a_dict"}}
        metadata_file = os.path.join(self.temp_dir, "invalid_preset.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_invalid_preset, f)

        loader = MetadataLoader(metadata_file)

        with self.assertRaises(ValueError) as cm:
            loader.get_preset()
        self.assertIn("preset must be a JSON object", str(cm.exception))

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_download_s3_once_success(self, mock_boto3_client):
        """Test _download_s3_once method with successful download."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        s3_url = "s3://test-bucket/test-file.mp4"
        result = loader._download_s3_once(s3_url)

        # local_path is cache_dir / key (bucket not included)
        expected_path = str(loader.cache_dir / "test-file.mp4")
        self.assertEqual(result, expected_path)

        # Verify S3 download was called
        mock_s3_client.download_file.assert_called_once_with("test-bucket", "test-file.mp4", expected_path)

        # Verify caching
        self.assertEqual(loader._download_cache[s3_url], expected_path)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_download_s3_once_cached(self, mock_boto3_client):
        """Test _download_s3_once method with cached result."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        s3_url = "s3://test-bucket/test-file.mp4"
        cached_path = "/cached/path/test-file.mp4"
        loader._download_cache[s3_url] = cached_path

        result = loader._download_s3_once(s3_url)

        self.assertEqual(result, cached_path)
        # Verify S3 download was NOT called
        mock_s3_client.download_file.assert_not_called()

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_download_s3_once_invalid_url(self, mock_boto3_client):
        """Test _download_s3_once method with invalid S3 URL."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        with self.assertRaises(ValueError) as cm:
            loader._download_s3_once("http://not-s3-url.com/file.mp4")
        self.assertIn("Invalid S3 URL", str(cm.exception))

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_download_s3_once_missing_key(self, mock_boto3_client):
        """Test _download_s3_once method with S3 URL missing key."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        with self.assertRaises(ValueError) as cm:
            loader._download_s3_once("s3://bucket-only")
        self.assertIn("missing key", str(cm.exception))

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_extract_archive_once_zip(self, mock_boto3_client):
        """Test _extract_archive_once method with ZIP file."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        # Create a test ZIP file
        zip_path = os.path.join(self.temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test_file.txt", "test content")
            zf.writestr("subdir/nested_file.txt", "nested content")

        result = loader._extract_archive_once(zip_path)

        expected_extract_dir = os.path.join(self.temp_dir, "test_extracted")
        self.assertEqual(result, expected_extract_dir)

        # Verify files were extracted
        self.assertTrue(os.path.exists(os.path.join(expected_extract_dir, "test_file.txt")))
        self.assertTrue(os.path.exists(os.path.join(expected_extract_dir, "subdir", "nested_file.txt")))

        # Verify caching
        self.assertEqual(loader._extract_cache[zip_path], expected_extract_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_extract_archive_once_tar(self, mock_boto3_client):
        """Test _extract_archive_once method with TAR file."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        # Create a test TAR file
        tar_path = os.path.join(self.temp_dir, "test.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            # Create a temporary file to add to the tar
            temp_file = os.path.join(self.temp_dir, "temp_for_tar.txt")
            with open(temp_file, "w") as f:
                f.write("test content")
            tf.add(temp_file, arcname="test_file.txt")

        result = loader._extract_archive_once(tar_path)

        expected_extract_dir = os.path.join(self.temp_dir, "test_extracted")
        self.assertEqual(result, expected_extract_dir)

        # Verify file was extracted
        self.assertTrue(os.path.exists(os.path.join(expected_extract_dir, "test_file.txt")))

        # Verify caching
        self.assertEqual(loader._extract_cache[tar_path], expected_extract_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_extract_archive_once_cached(self, mock_boto3_client):
        """Test _extract_archive_once method with cached result."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        archive_path = os.path.join(self.temp_dir, "test.zip")
        cached_extract_dir = "/cached/extract/dir"
        loader._extract_cache[archive_path] = cached_extract_dir

        result = loader._extract_archive_once(archive_path)

        self.assertEqual(result, cached_extract_dir)

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_extract_archive_once_unsupported_format(self, mock_boto3_client):
        """Test _extract_archive_once method with unsupported archive format."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        # Create a file that's not an archive
        not_archive_path = os.path.join(self.temp_dir, "not_archive.txt")
        with open(not_archive_path, "w") as f:
            f.write("This is not an archive")

        with self.assertRaises(ValueError) as cm:
            loader._extract_archive_once(not_archive_path)
        self.assertIn("Unsupported archive type", str(cm.exception))

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @patch("boto3.client")
    def test_extract_archive_once_zip_path_traversal_protection(self, mock_boto3_client):
        """Test _extract_archive_once method protects against ZIP path traversal."""
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        loader = MetadataLoader(self.metadata_file)

        # Create a malicious ZIP file with path traversal
        zip_path = os.path.join(self.temp_dir, "malicious.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Try to write outside the extraction directory
            zf.writestr("../../../etc/passwd", "malicious content")

        with self.assertRaises(ValueError) as cm:
            loader._extract_archive_once(zip_path)
        self.assertIn("Unsafe path in ZIP archive", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
