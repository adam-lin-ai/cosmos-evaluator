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

import runpy
import unittest
import unittest.mock as mock

from fastapi.testclient import TestClient

from checks.hallucination.processor import HallucinationResult
from services.hallucination.hallucination_service import HallucinationService
from services.hallucination.rest_api import app
from services.utils import get_contents_from_runfile


class TestHallucinationAPI(unittest.TestCase):
    """Test cases for Hallucination REST API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self._client_cm.__exit__(None, None, None)

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])  # from JsonResponseFormatter
        self.assertEqual(data["data"]["status"], "healthy")
        self.assertEqual(data["data"]["service"], "Hallucination Service API")
        expected_version = get_contents_from_runfile("services/hallucination/version.txt")
        self.assertEqual(data["data"]["version"], expected_version)
        self.assertIn("timestamp", data)

    def test_health_endpoint_service_uninitialized(self) -> None:
        """Test health endpoint when service is unavailable."""
        with mock.patch("services.hallucination.rest_api.service", None):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_config_endpoint(self) -> None:
        """Test config endpoint."""
        response = self.client.get("/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])  # formatter wrapper
        self.assertIn("default_config", data["data"])  # payload content
        self.assertIn("description", data["data"])  # payload content

        default_config = data["data"]["default_config"]
        self.assertIn("enabled", default_config)

    def test_config_endpoint_service_uninitialized(self) -> None:
        """Test config endpoint when service is unavailable."""
        with mock.patch("services.hallucination.rest_api.service", None):
            response = self.client.get("/config")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])

    def test_process_endpoint(self) -> None:
        """Test process endpoint."""
        dummy_result = HallucinationResult(
            clip_id="test_clip_id",
            passed=True,
            threshold=0.682,
            score=0.1,
            total_frames=5,
            total_hallucinated_dynamic_pixels=1,
            total_augmented_dynamic_pixels=10,
        )

        with (
            mock.patch.object(HallucinationService, "validate_input", return_value=True),
            mock.patch.object(HallucinationService, "process", return_value=dummy_result),
        ):
            response = self.client.post(
                "/process",
                json={
                    "clip_id": "test_clip_id",
                    "original_video_path": "test_original_video_path",
                    "augmented_video_path": "test_augmented_video_path",
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["clip_id"], "test_clip_id")
        self.assertTrue(data["data"]["passed"])

    def test_process_endpoint_service_uninitialized(self) -> None:
        """Test process endpoint when service is unavailable."""
        with mock.patch("services.hallucination.rest_api.service", None):
            response = self.client.post(
                "/process",
                json={"clip_id": "x", "original_video_path": "orig.mp4", "augmented_video_path": "aug.mp4"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])

    def test_process_endpoint_over_capacity(self) -> None:
        """Test process endpoint returns 503 when in-flight capacity is exhausted."""
        with (
            mock.patch("services.hallucination.rest_api.process_concurrency_limit", 1),
            mock.patch("services.hallucination.rest_api._inflight_count", 1),
        ):
            response = self.client.post(
                "/process",
                json={"clip_id": "x", "original_video_path": "orig.mp4", "augmented_video_path": "aug.mp4"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("retry-after"), "1")
        self.assertFalse(response.json()["success"])

    def test_process_endpoint_validation_error(self) -> None:
        """Test process endpoint input validation failure."""
        with mock.patch.object(HallucinationService, "validate_input", side_effect=ValueError("bad input")):
            response = self.client.post(
                "/process",
                json={"clip_id": "x", "original_video_path": "orig.mp4", "augmented_video_path": "aug.mp4"},
            )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("bad input", data["error"]["message"])

    def test_process_endpoint_error(self) -> None:
        """Test process endpoint error."""
        with (
            mock.patch.object(HallucinationService, "validate_input", return_value=True),
            mock.patch.object(HallucinationService, "process", side_effect=RuntimeError("some error occurred")),
        ):
            response = self.client.post(
                "/process",
                json={"clip_id": "error_clip_id", "original_video_path": "orig.mp4", "augmented_video_path": "aug.mp4"},
            )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertIn("some error occurred", data["error"]["message"])

    def test_main_entrypoint_runs_uvicorn(self) -> None:
        """Test __main__ block wiring without starting a server."""
        fake_uvicorn = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"uvicorn": fake_uvicorn}),
            mock.patch("sys.argv", ["rest_api.py"]),
        ):
            runpy.run_module("services.hallucination.rest_api", run_name="__main__")
        fake_uvicorn.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
