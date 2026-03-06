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

"""Unit tests for Obstacle Correspondence REST API."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from services.framework.protocols.storage_provider import StorageUrls
from services.obstacle_correspondence import rest_api_common as rest_api
from services.obstacle_correspondence.models import (
    ObstacleCorrespondenceCloudRequest,
    ObstacleCorrespondenceRequest,
    ObstacleCorrespondenceResult,
)
from services.utils import get_contents_from_runfile

# Mock get_git_sha for all tests since //utils:git_sha is not in the test's runfiles
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()


class TestObstacleCorrespondenceAPI(unittest.TestCase):
    """Test cases for Obstacle Correspondence REST API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir()

        self.video_file = Path(self.temp_dir) / "test_video.mp4"
        self.video_file.write_text("mock video content")

        self.world_video_file = Path(self.temp_dir) / "test_world_video.mp4"
        self.world_video_file.write_text("mock world video content")

        self.valid_request = {
            "input_data_path": str(self.input_dir),
            "clip_id": "test_clip_12345",
            "camera_name": "front_camera",
            "video_path": str(self.video_file),
            "world_video_path": str(self.world_video_file),
            "config": {
                "av.obstacle": {
                    "overlap_check": {"vehicle": {"method": "ratio"}},
                    "importance_filter": {
                        "distance_threshold_m": 50.0,
                        "oncoming_obstacles": False,
                        "relevant_lanes": ["ego", "left", "right"],
                    },
                }
            },
            "model_device": "cpu",
            "verbose": "INFO",
            "trial_frames": 10,
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["status"], "healthy")
        self.assertEqual(data["data"]["service"], "Obstacle Correspondence Service API")

        # Read expected version from the same source as the API
        expected_version = get_contents_from_runfile("services/obstacle_correspondence/version.txt")
        self.assertEqual(data["data"]["version"], expected_version)
        self.assertIn("timestamp", data)

    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_config_endpoint(self, mock_service: MagicMock) -> None:
        """Test config endpoint."""
        # Mock the service get_default_config method
        mock_service.get_default_config = AsyncMock(
            return_value={
                "av.obstacle": {
                    "enabled": True,
                    "overlap_check": ["pedestrian", "vehicle"],
                    "importance_filter": {
                        "distance_threshold_m": 100,
                        "skip_oncoming_obstacles": True,
                        "relevant_lanes": ["ego", "left", "right"],
                    },
                }
            }
        )

        response = self.client.get("/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIn("default_config", data["data"])
        self.assertIn("description", data["data"])

        # Verify config structure
        default_config = data["data"]["default_config"]
        self.assertIn("av.obstacle", default_config)
        self.assertIn("overlap_check", default_config["av.obstacle"])
        self.assertIn("importance_filter", default_config["av.obstacle"])

    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_process_endpoint_success(self, mock_service: MagicMock, mock_process_s3: MagicMock) -> None:
        """Test successful synchronous processing endpoint for S3 requests."""
        # Create S3 request
        s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {"vehicle": {"method": "ratio"}}}},
            "model_device": "cpu",
            "trial_frames": 10,
        }

        # Mock S3 processing
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config=s3_request["config"],
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path("/tmp/temp_s3")
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        # Mock service methods
        mock_service.validate_input = AsyncMock(return_value=True)
        mock_service.process = AsyncMock(return_value=self._create_mock_result())
        mock_service.cleanup = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(
            return_value=StorageUrls(raw="s3://bucket/results.json", presigned="https://presigned.url/results.json")
        )
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "services.obstacle_correspondence.rest_api_common.build_storage_provider",
            return_value=mock_provider,
        ):
            response = self.client.post("/process", json=s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIn("data", data)
        self.assertIn("metadata", data)
        self.assertIn("timestamp", data)

        # Verify S3 processing was called
        mock_process_s3.assert_called_once()

        # Verify data structure - now returns dict directly
        result_data = data["data"]
        self.assertEqual(result_data["clip_id"], "test_clip_12345")
        self.assertEqual(result_data["processed_frames"], 10)
        self.assertAlmostEqual(result_data["mean_score"], 0.75)

        # Verify S3 URLs are included
        self.assertIn("visualizations_urls", result_data)
        self.assertIn("visualizations_presigned_urls", result_data)
        self.assertIn("results_json_urls", result_data)
        self.assertIn("results_json_presigned_urls", result_data)

        # Verify metadata
        metadata = data["metadata"]
        self.assertIn("processing_rate", metadata)
        self.assertIn("duration_ms", metadata)
        self.assertIsInstance(metadata["duration_ms"], (int, float))
        self.assertGreaterEqual(metadata["duration_ms"], 0)

    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_process_endpoint_validation_error(self, mock_service: MagicMock, mock_process_s3: MagicMock) -> None:
        """Test process endpoint with validation error."""
        # Create S3 request
        s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {}},
            "model_device": "cpu",
        }

        # Mock S3 processing
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
        )
        mock_temp_dir = Path("/tmp/temp_s3")
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        mock_service.validate_input = AsyncMock(side_effect=ValueError("Input data path does not exist"))
        mock_service.cleanup = AsyncMock()

        response = self.client.post("/process", json=s3_request)

        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "ValueError")
        self.assertIn("Input data path does not exist", data["error"]["message"])

        # Verify duration_ms is in error metadata
        self.assertIn("metadata", data)
        self.assertIn("duration_ms", data["metadata"])
        self.assertIsInstance(data["metadata"]["duration_ms"], (int, float))
        self.assertGreaterEqual(data["metadata"]["duration_ms"], 0)

    def test_process_endpoint_invalid_json(self) -> None:
        """Test process endpoint with invalid JSON for S3 requests."""
        invalid_s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            # Missing required augmented_video_url and other fields
        }

        response = self.client.post("/process", json=invalid_s3_request)

        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertGreater(len(data["detail"]), 0)

    def test_process_endpoint_empty_body(self) -> None:
        """Test process endpoint with empty request body."""
        response = self.client.post("/process", json={})

        self.assertEqual(response.status_code, 422)
        data = response.json()

        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertGreater(len(data["detail"]), 0)

    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_process_endpoint_internal_error(self, mock_service: MagicMock, mock_process_s3: MagicMock) -> None:
        """Test process endpoint with internal server error."""
        # Create S3 request
        s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {}},
            "model_device": "cpu",
        }

        # Mock S3 processing
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
        )
        mock_temp_dir = Path("/tmp/temp_s3")
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        mock_service.validate_input = AsyncMock(return_value=True)
        mock_service.process = AsyncMock(side_effect=RuntimeError("Internal processing error"))
        mock_service.cleanup = AsyncMock()

        response = self.client.post("/process", json=s3_request)

        self.assertEqual(response.status_code, 500)
        data = response.json()

        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "RuntimeError")

        # Verify duration_ms is in error metadata
        self.assertIn("metadata", data)
        self.assertIn("duration_ms", data["metadata"])
        self.assertIsInstance(data["metadata"]["duration_ms"], (int, float))
        self.assertGreaterEqual(data["metadata"]["duration_ms"], 0)

    def test_nonexistent_endpoint(self) -> None:
        """Test request to non-existent endpoint."""
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_wrong_method_on_endpoint(self) -> None:
        """Test wrong HTTP method on endpoint."""
        response = self.client.get("/process")  # Should be POST
        self.assertEqual(response.status_code, 405)

    def _create_mock_result(self) -> ObstacleCorrespondenceResult:
        """Create a mock ObstacleCorrespondenceResult for testing."""
        return ObstacleCorrespondenceResult(
            processed_frames=10,
            total_video_frames=10,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3, 5],
            processed_frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            config_summary={
                "overlap_check": {"vehicle": {"method": "ratio"}},
                "importance_filter": {"distance_threshold_m": 50.0},
                "model_device": "cpu",
            },
            clip_id="test_clip_12345",
            output_dir="/tmp/test_output",
        )


class TestObstacleCorrespondenceCloudRequest(unittest.TestCase):
    """Test cases for ObstacleCorrespondenceCloudRequest model validation."""

    def test_s3_request_validation_success(self) -> None:
        """Test successful creation of S3 request with all valid fields."""
        valid_s3_request = ObstacleCorrespondenceCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            camera_name="front_camera",
            config={
                "av.obstacle": {
                    "overlap_check": {"vehicle": {"method": "ratio"}},
                    "importance_filter": {"distance_threshold_m": 50.0},
                }
            },
        )

        # Verify required fields
        self.assertEqual(valid_s3_request.rds_hq_url, "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned")
        self.assertEqual(valid_s3_request.augmented_video_url, "https://s3.amazonaws.com/bucket/video.mp4?presigned")
        self.assertEqual(valid_s3_request.camera_name, "front_camera")
        self.assertIsInstance(valid_s3_request.config, dict)

        # Verify default values
        self.assertEqual(valid_s3_request.model_device, "cuda")
        self.assertEqual(valid_s3_request.verbose, "INFO")
        self.assertIsNone(valid_s3_request.trial_frames)
        self.assertIsNone(valid_s3_request.output_storage_prefix)

    def test_s3_request_missing_required_fields(self) -> None:
        """Test S3 request validation fails with missing required fields."""
        # Test missing rds_hq_url
        with self.assertRaises(Exception) as context:
            ObstacleCorrespondenceCloudRequest(  # type: ignore[call-arg]
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config={"av.obstacle": {}},
            )
        self.assertIn("rds_hq_url", str(context.exception))

        # Test missing augmented_video_url
        with self.assertRaises(Exception) as context:
            ObstacleCorrespondenceCloudRequest(  # type: ignore[call-arg]
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config={"av.obstacle": {}},
            )
        self.assertIn("augmented_video_url", str(context.exception))

        # Test missing camera_name
        with self.assertRaises(Exception) as context:
            ObstacleCorrespondenceCloudRequest(  # type: ignore[call-arg]
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                config={"av.obstacle": {}},
            )
        self.assertIn("camera_name", str(context.exception))

    def test_s3_request_custom_values(self) -> None:
        """Test S3 request with custom field values."""
        custom_s3_request = ObstacleCorrespondenceCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
            camera_name="rear_camera",
            config={"av.obstacle": {"custom": "config"}},
            model_device="cpu",
            verbose="DEBUG",
            trial_frames=100,
        )

        # Verify custom values
        self.assertEqual(custom_s3_request.model_device, "cpu")
        self.assertEqual(custom_s3_request.verbose, "DEBUG")
        self.assertEqual(custom_s3_request.trial_frames, 100)

    def test_s3_request_invalid_config_type(self) -> None:
        """Test S3 request validation fails with invalid config type."""
        with self.assertRaises(Exception) as context:
            ObstacleCorrespondenceCloudRequest(
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config="not_a_dict",  # String instead of dict
            )
        self.assertIn("Input should be a valid dictionary", str(context.exception))

    def test_s3_request_negative_trial_frames(self) -> None:
        """Test S3 request validation with negative trial_frames."""
        # Pydantic should allow this at model level, validation happens in service
        s3_request = ObstacleCorrespondenceCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
            camera_name="front_camera",
            config={"av.obstacle": {}},
            trial_frames=-5,
        )

        # Should be created successfully (validation happens at service level)
        self.assertEqual(s3_request.trial_frames, -5)


class TestRequestProcessing(unittest.IsolatedAsyncioTestCase):
    """Test cases for _process_request function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_s3_request = ObstacleCorrespondenceCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            camera_name="front_camera",
            config={
                "av.obstacle": {
                    "overlap_check": {"vehicle": {"method": "ratio"}},
                    "importance_filter": {"distance_threshold_m": 50.0},
                }
            },
            model_device="cpu",
            verbose="INFO",
            trial_frames=10,
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("services.obstacle_correspondence.rest_api_common.storage")
    @patch("services.obstacle_correspondence.rest_api_common.utils.extract_clip_id")
    @patch("zipfile.ZipFile")
    @patch("services.obstacle_correspondence.rest_api_common.Path")
    @patch("os.getpid")
    @patch("asyncio.get_event_loop")
    async def test_process_request_success(
        self,
        mock_get_loop: MagicMock,
        mock_getpid: MagicMock,
        mock_path_class: MagicMock,
        mock_zipfile: MagicMock,
        mock_extract_clip_id: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Test successful S3 request processing."""
        # Setup mocks
        mock_getpid.return_value = 12345
        mock_get_loop.return_value.time.return_value = 1234567890.123

        mock_temp_dir = MagicMock()
        mock_temp_dir.__truediv__ = MagicMock(side_effect=lambda x: MagicMock())
        mock_path_class.return_value = mock_temp_dir

        mock_storage.download_from_url = AsyncMock(return_value=None)
        mock_extract_clip_id.return_value = "test_clip_12345"

        # Mock zipfile extraction
        mock_zip_ref = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_ref

        # Call the function
        internal_request, _ = await rest_api._process_request(self.mock_s3_request)

        # Verify downloads were called
        self.assertEqual(mock_storage.download_from_url.call_count, 3)

        # Verify zip extraction
        mock_zipfile.assert_called_once()
        mock_zip_ref.extractall.assert_called_once()

        # Verify clip ID extraction
        mock_extract_clip_id.assert_called_once()

        # Verify internal request creation
        self.assertIsNotNone(internal_request)
        self.assertEqual(internal_request.camera_name, "front_camera")
        self.assertEqual(internal_request.model_device, "cpu")
        self.assertEqual(internal_request.verbose, "INFO")
        self.assertEqual(internal_request.trial_frames, 10)

    @patch("services.obstacle_correspondence.rest_api_common.storage")
    @patch("shutil.rmtree")
    @patch("services.obstacle_correspondence.rest_api_common.Path")
    @patch("tempfile.mkdtemp")
    async def test_process_request_download_failure(
        self,
        mock_mkdtemp: MagicMock,
        mock_path_class: MagicMock,
        mock_rmtree: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Test S3 request processing with download failure."""
        mock_mkdtemp.return_value = "/tmp/obstacle_correspondence_test"

        # Create mock temp dir returned by Path()
        mock_temp_dir = MagicMock()
        mock_temp_dir.exists.return_value = True
        mock_temp_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_path_class.return_value = mock_temp_dir

        # Mock download failure
        mock_storage.download_from_url = AsyncMock(side_effect=Exception("Download failed"))

        # Call should raise exception and clean up
        with self.assertRaises(Exception) as context:
            await rest_api._process_request(self.mock_s3_request)

        self.assertIn("Download failed", str(context.exception))

        # Verify cleanup was attempted
        mock_rmtree.assert_called_once_with(mock_temp_dir, ignore_errors=True)

    @patch("services.obstacle_correspondence.rest_api_common.storage")
    @patch("services.obstacle_correspondence.rest_api_common.utils.extract_clip_id")
    @patch("zipfile.ZipFile")
    @patch("shutil.rmtree")
    @patch("services.obstacle_correspondence.rest_api_common.Path")
    @patch("tempfile.mkdtemp")
    async def test_process_request_extraction_failure(
        self,
        mock_mkdtemp: MagicMock,
        mock_path_class: MagicMock,
        mock_rmtree: MagicMock,
        mock_zipfile: MagicMock,
        mock_extract_clip_id: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Test S3 request processing with clip ID extraction failure."""
        mock_mkdtemp.return_value = "/tmp/obstacle_correspondence_test"

        # Create mock temp dir returned by Path()
        mock_temp_dir = MagicMock()
        mock_temp_dir.exists.return_value = True
        mock_temp_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_path_class.return_value = mock_temp_dir

        mock_storage.download_from_url = AsyncMock(return_value=None)
        mock_extract_clip_id.side_effect = ValueError("No .tar files found")

        # Mock zipfile extraction
        mock_zip_ref = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_ref

        # Call should raise exception and clean up
        with self.assertRaises(ValueError) as context:
            await rest_api._process_request(self.mock_s3_request)

        self.assertIn("No .tar files found", str(context.exception))

        # Verify cleanup was attempted
        mock_rmtree.assert_called_once_with(mock_temp_dir, ignore_errors=True)

    @patch("services.obstacle_correspondence.rest_api_common.storage")
    @patch("zipfile.ZipFile")
    @patch("shutil.rmtree")
    @patch("services.obstacle_correspondence.rest_api_common.Path")
    @patch("tempfile.mkdtemp")
    async def test_process_request_zipfile_failure(
        self,
        mock_mkdtemp: MagicMock,
        mock_path_class: MagicMock,
        mock_rmtree: MagicMock,
        mock_zipfile: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Test S3 request processing with zip file extraction failure."""
        mock_mkdtemp.return_value = "/tmp/obstacle_correspondence_test"

        # Create mock temp dir returned by Path()
        mock_temp_dir = MagicMock()
        mock_temp_dir.exists.return_value = True
        mock_temp_dir.__truediv__ = MagicMock(return_value=MagicMock())
        mock_path_class.return_value = mock_temp_dir

        mock_storage.download_from_url = AsyncMock(return_value=None)
        mock_zipfile.side_effect = Exception("Corrupted zip file")

        # Call should raise exception and clean up
        with self.assertRaises(Exception) as context:
            await rest_api._process_request(self.mock_s3_request)

        self.assertIn("Corrupted zip file", str(context.exception))

        # Verify cleanup was attempted
        mock_rmtree.assert_called_once_with(mock_temp_dir, ignore_errors=True)


class TestCloudAPIIntegration(unittest.TestCase):
    """Test cases for Cloud API integration in REST endpoints."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)
        self.temp_dir = tempfile.mkdtemp()

        self.valid_s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {
                "av.obstacle": {
                    "overlap_check": {"vehicle": {"method": "ratio"}},
                    "importance_filter": {"distance_threshold_m": 50.0},
                }
            },
            "model_device": "cpu",
            "verbose": "INFO",
            "trial_frames": 10,
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_process_endpoint_with_s3_request(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test /process endpoint with S3 request."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider

        # Mock S3 processing - create a realistic internal request
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path(self.temp_dir) / "temp_s3"
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        # Mock service
        mock_service.validate_input = AsyncMock(return_value=True)
        mock_service.process = AsyncMock(return_value=self._create_mock_result())
        mock_service.cleanup = AsyncMock()

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify S3 processing was called
        mock_process_s3.assert_called_once()

        # Verify service was called with internal request
        mock_service.validate_input.assert_called_once_with(mock_internal_request)
        mock_service.process.assert_called_once_with(mock_internal_request)

    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_s3_request_processing_failure(self, mock_service: MagicMock, mock_process_s3: MagicMock) -> None:
        """Test handling of S3 processing failures."""
        # Mock S3 processing failure
        mock_process_s3.side_effect = Exception("S3 download failed")

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_s3_request_detection_logic(self) -> None:
        """Test that S3 and local requests are routed to the correct endpoints."""
        # Test with S3 request on /process - should call _process_request
        with patch("services.obstacle_correspondence.rest_api_common._process_request") as mock_process_s3:
            with patch("services.obstacle_correspondence.rest_api_common.service") as mock_service:
                mock_internal_request = ObstacleCorrespondenceRequest(
                    input_data_path="/tmp/test_input",
                    clip_id="test_clip_12345",
                    camera_name="front_camera",
                    video_path="/tmp/test_video.mp4",
                    world_video_path="/tmp/test_world_video.mp4",
                    config={"av.obstacle": {}},
                    model_device="cpu",
                    trial_frames=10,
                )
                mock_temp_dir = Path(self.temp_dir) / "temp_s3"
                mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)
                mock_service.validate_input = AsyncMock(return_value=True)
                mock_service.process = AsyncMock(return_value=self._create_mock_result())
                mock_service.cleanup = AsyncMock()

                self.client.post("/process", json=self.valid_s3_request)

                # Should call S3 processing
                mock_process_s3.assert_called_once()

    def _create_mock_result(self) -> MagicMock:
        """Create a mock ObstacleCorrespondenceResult for testing."""
        return MagicMock(
            processed_frames=10,
            total_video_frames=10,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            config_summary={"test": "config"},
            clip_id="test_clip_12345",
            output_dir="/tmp/test_output",
            model_dump=lambda **kwargs: {
                "processed_frames": 10,
                "total_video_frames": 10,
                "mean_score": 0.75,
                "std_score": 0.12,
                "min_score": 0.45,
                "max_score": 0.95,
                "unique_track_ids": [1, 2, 3],
                "processed_frame_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "config_summary": {"test": "config"},
                "clip_id": "test_clip_12345",
                "output_dir": "/tmp/test_output",
            },
        )


class TestUploadLogic(unittest.TestCase):
    """Test cases for upload logic in the /process endpoint."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)
        self.temp_dir = tempfile.mkdtemp()

        # Create a mock segmentation video file
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
        self.segmentation_video = self.output_dir / "test_clip_12345.dynamic.object.mp4"
        self.segmentation_video.write_text("mock video content")

        self.valid_s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {"vehicle": {"method": "ratio"}}}},
            "model_device": "cpu",
            "trial_frames": 10,
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @staticmethod
    def _mock_output_provider(store_file_return: StorageUrls | None = None) -> MagicMock:
        """Creates a mock StorageProvider for output uploads."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=store_file_return or StorageUrls())
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        return mock_provider

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_s3_upload_success(
        self,
        mock_service: MagicMock,
        mock_process_s3: MagicMock,
        mock_build_provider: MagicMock,
    ) -> None:
        """Test successful S3 upload after processing."""
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path(self.temp_dir) / "temp_s3"
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = self._create_mock_result_with_output_dir(str(self.output_dir))
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        mock_provider = self._mock_output_provider(
            StorageUrls(raw="s3://bucket/obstacles/results.json", presigned="https://presigned.url/results.json")
        )
        mock_build_provider.return_value = mock_provider

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        response_data = data["data"]
        self.assertIn("visualizations_urls", response_data)
        self.assertIn("visualizations_presigned_urls", response_data)
        self.assertIsInstance(response_data["visualizations_urls"], dict)
        self.assertIsInstance(response_data["visualizations_presigned_urls"], dict)

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_s3_upload_failure_continues_processing(
        self,
        mock_service: MagicMock,
        mock_process_s3: MagicMock,
        mock_build_provider: MagicMock,
    ) -> None:
        """Test that S3 upload failure doesn't break the response."""
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path(self.temp_dir) / "temp_s3"
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = self._create_mock_result_with_output_dir(str(self.output_dir))
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        mock_provider = self._mock_output_provider(StorageUrls())
        mock_build_provider.return_value = mock_provider

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        response_data = data["data"]
        self.assertEqual(response_data.get("visualizations_urls"), {})
        self.assertEqual(response_data.get("visualizations_presigned_urls"), {})
        self.assertEqual(response_data["clip_id"], "test_clip_12345")
        self.assertEqual(response_data["processed_frames"], 10)

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_s3_upload_returns_urls(
        self,
        mock_service: MagicMock,
        mock_process_s3: MagicMock,
        mock_build_provider: MagicMock,
    ) -> None:
        """Test that S3 upload returns URLs correctly."""
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path(self.temp_dir) / "temp_s3"
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = self._create_mock_result_with_output_dir(str(self.output_dir))
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        mock_provider = self._mock_output_provider(
            StorageUrls(raw="s3://bucket/obstacles/results.json", presigned="https://presigned.url/results.json")
        )
        mock_build_provider.return_value = mock_provider

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_no_s3_upload_when_video_missing(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test that S3 upload is skipped when segmentation video doesn't exist."""
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip_12345",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=10,
        )
        mock_temp_dir = Path(self.temp_dir) / "temp_s3"
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        empty_output_dir = Path(self.temp_dir) / "empty_output"
        empty_output_dir.mkdir()
        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = self._create_mock_result_with_output_dir(str(empty_output_dir))
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        mock_provider = self._mock_output_provider()
        mock_build_provider.return_value = mock_provider

        response = self.client.post("/process", json=self.valid_s3_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        response_data = data["data"]
        self.assertEqual(response_data.get("visualizations_urls"), {})
        self.assertEqual(response_data.get("visualizations_presigned_urls"), {})

    def _create_mock_result_with_output_dir(self, output_dir: str) -> ObstacleCorrespondenceResult:
        """Create a mock ObstacleCorrespondenceResult with specific output directory."""
        return ObstacleCorrespondenceResult(
            processed_frames=10,
            total_video_frames=10,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            config_summary={"test": "config"},
            clip_id="test_clip_12345",
            output_dir=output_dir,
        )


class TestProcessingRateCalculation(unittest.TestCase):
    """Test cases for processing rate calculation logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)

    def tearDown(self) -> None:
        """Reset capacity state."""
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_processing_rate_calculation_full_processing(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test processing rate calculation for full processing."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider

        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {"overlap_check": {}}},
            model_device="cpu",
        )
        mock_process_s3.return_value = (mock_internal_request, Path("/tmp/temp_s3"))

        mock_service.validate_input = AsyncMock(return_value=True)

        # Mock result with all frames processed
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=100,
            total_video_frames=100,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=list(range(100)),
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        valid_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
        }

        response = self.client.post("/process", json=valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify processing rate is in metadata only
        response_data = data["data"]
        self.assertNotIn("processing_rate", response_data)  # Not in data anymore

        # Verify metadata contains processing rate
        metadata = data["metadata"]
        self.assertEqual(metadata["processing_rate"], "100.0%")

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_processing_rate_calculation_partial_processing(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test processing rate calculation for partial processing."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider

        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {"overlap_check": {}}},
            model_device="cpu",
        )
        mock_process_s3.return_value = (mock_internal_request, Path("/tmp/temp_s3"))

        mock_service.validate_input = AsyncMock(return_value=True)

        # Mock result with partial processing (50 out of 200 frames)
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=50,
            total_video_frames=200,
            mean_score=0.65,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=list(range(50)),
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        valid_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
        }

        response = self.client.post("/process", json=valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify processing rate is in metadata only
        response_data = data["data"]
        self.assertNotIn("processing_rate", response_data)  # Not in data anymore

        # Verify metadata contains processing rate
        metadata = data["metadata"]
        self.assertEqual(metadata["processing_rate"], "25.0%")

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_processing_rate_calculation_zero_total_frames(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test processing rate calculation when total frames is zero."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider

        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {"overlap_check": {}}},
            model_device="cpu",
        )
        mock_process_s3.return_value = (mock_internal_request, Path("/tmp/temp_s3"))

        mock_service.validate_input = AsyncMock(return_value=True)

        # Mock result with zero total frames
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=0,
            total_video_frames=0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            unique_track_ids=[],
            processed_frame_ids=[],
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        valid_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
        }

        response = self.client.post("/process", json=valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify processing rate is in metadata only
        response_data = data["data"]
        self.assertNotIn("processing_rate", response_data)  # Not in data anymore

        # Verify metadata contains processing rate
        metadata = data["metadata"]
        self.assertEqual(metadata["processing_rate"], "null")

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_processing_rate_calculation_precision(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test processing rate calculation precision (1 decimal place)."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider

        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {"overlap_check": {}}},
            model_device="cpu",
        )
        mock_process_s3.return_value = (mock_internal_request, Path("/tmp/temp_s3"))

        mock_service.validate_input = AsyncMock(return_value=True)

        # Mock result with processing rate that needs rounding (33.333...%)
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=33,
            total_video_frames=99,
            mean_score=0.55,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=list(range(33)),
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        valid_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
        }

        response = self.client.post("/process", json=valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify processing rate is in metadata only
        response_data = data["data"]
        self.assertNotIn("processing_rate", response_data)  # Not in data anymore

        # Verify metadata contains processing rate rounded to 1 decimal place
        metadata = data["metadata"]
        self.assertEqual(metadata["processing_rate"], "33.3%")


class TestEndpointSpecificBehavior(unittest.TestCase):
    """Test cases for endpoint-specific behavior for S3 vs local processing."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)

    def tearDown(self) -> None:
        """Reset capacity state."""
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_process_endpoint_handles_s3_requests(self, mock_service: MagicMock, mock_process_s3: MagicMock) -> None:
        """Test that /process endpoint handles S3 requests properly."""
        # S3 request with both required fields
        s3_request = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
        }

        # Mock S3 processing
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
        )
        mock_temp_dir = Path("/tmp/temp_s3")
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        # Mock service
        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=10,
            total_video_frames=10,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=list(range(10)),
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(
            return_value=StorageUrls(raw="s3://bucket/results.json", presigned="https://presigned.url/results.json")
        )
        mock_provider.close = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "services.obstacle_correspondence.rest_api_common.build_storage_provider",
            return_value=mock_provider,
        ):
            response = self.client.post("/process", json=s3_request)

        self.assertEqual(response.status_code, 200)
        mock_process_s3.assert_called_once()

    def test_s3_endpoint_json_parsing_error(self) -> None:
        """Test S3 endpoint with invalid JSON."""
        # Send invalid JSON
        response = self.client.post("/process", content="invalid json", headers={"content-type": "application/json"})

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertGreater(len(data["detail"]), 0)

    def test_s3_endpoint_empty_body(self) -> None:
        """Test S3 endpoint with empty body."""
        response = self.client.post("/process", content="", headers={"content-type": "application/json"})

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertGreater(len(data["detail"]), 0)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test cases for edge cases and error handling in new logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0
        self.client = TestClient(rest_api.app)

    def tearDown(self) -> None:
        """Reset capacity state."""
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @patch("services.obstacle_correspondence.rest_api_common.build_storage_provider")
    @patch("services.obstacle_correspondence.rest_api_common._process_request")
    @patch("services.obstacle_correspondence.rest_api_common.service")
    def test_s3_request_with_default_values(
        self, mock_service: MagicMock, mock_process_s3: MagicMock, mock_build_provider: MagicMock
    ) -> None:
        """Test S3 request with omitted optional fields (using default values)."""
        mock_provider = MagicMock()
        mock_provider.store_file = AsyncMock(return_value=StorageUrls())
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_build_provider.return_value = mock_provider
        s3_request_with_nulls = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4?presigned",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            "camera_name": "front_camera",
            "config": {"av.obstacle": {"overlap_check": {}}},
            "model_device": "cpu",
            # Note: trial_frames and output_dir are omitted to test default values
        }

        # Mock S3 processing
        mock_internal_request = ObstacleCorrespondenceRequest(
            input_data_path="/tmp/test_input",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/tmp/test_video.mp4",
            world_video_path="/tmp/test_world_video.mp4",
            config={"av.obstacle": {}},
            model_device="cpu",
            trial_frames=None,
        )
        mock_temp_dir = Path("/tmp/temp_s3")
        mock_process_s3.return_value = (mock_internal_request, mock_temp_dir)

        # Mock service
        mock_service.validate_input = AsyncMock(return_value=True)
        mock_result = ObstacleCorrespondenceResult(
            processed_frames=10,
            total_video_frames=10,
            mean_score=0.75,
            std_score=0.12,
            min_score=0.45,
            max_score=0.95,
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=list(range(10)),
            config_summary={"test": "config"},
            clip_id="test_clip",
            output_dir="/tmp/test_output",
        )
        mock_service.process = AsyncMock(return_value=mock_result)
        mock_service.cleanup = AsyncMock()

        response = self.client.post("/process", json=s3_request_with_nulls)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        # Verify S3 processing was called
        mock_process_s3.assert_called_once()

    def test_malformed_json_with_s3_fields(self) -> None:
        """Test malformed JSON that contains S3 field names."""
        # This tests the JSON parsing logic before request type detection
        malformed_json = '{"rds_hq_url": "test", "augmented_video_url": incomplete'

        response = self.client.post("/process", content=malformed_json, headers={"content-type": "application/json"})

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIsInstance(data["detail"], list)
        self.assertGreater(len(data["detail"]), 0)


class TestProcessCapacityGate(unittest.IsolatedAsyncioTestCase):
    """Tests for /process capacity gating helpers."""

    async def asyncSetUp(self) -> None:
        """Reset capacity state before each test."""
        rest_api._process_slots_lock = asyncio.Lock()
        rest_api._active_process_requests = 0

    async def asyncTearDown(self) -> None:
        """Reset capacity state after each test."""
        rest_api._process_slots_lock = None
        rest_api._active_process_requests = 0

    @patch.object(rest_api, "PROCESS_CONCURRENCY_LIMIT", 1)
    async def test_acquire_rejects_when_capacity_full(self) -> None:
        """Second acquire is rejected at capacity."""
        try:
            first_acquired = await rest_api._try_acquire_process_slot()
            second_acquired = await rest_api._try_acquire_process_slot()

            self.assertTrue(first_acquired)
            self.assertFalse(second_acquired)
        finally:
            await rest_api._release_process_slot()
            rest_api._active_process_requests = 0

    @patch.object(rest_api, "PROCESS_CONCURRENCY_LIMIT", 1)
    async def test_release_frees_capacity_for_next_request(self) -> None:
        """Releasing a slot allows another acquire."""
        try:
            acquired = await rest_api._try_acquire_process_slot()
            self.assertTrue(acquired)

            await rest_api._release_process_slot()

            acquired_again = await rest_api._try_acquire_process_slot()
            self.assertTrue(acquired_again)
        finally:
            await rest_api._release_process_slot()
            rest_api._active_process_requests = 0


if __name__ == "__main__":
    unittest.main()
