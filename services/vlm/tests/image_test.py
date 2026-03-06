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

"""Integration test for the VLM container image.

This test loads and runs the VLM OCI image to verify it starts
successfully and responds to health checks.
"""

import json
import logging
import os
import re
import subprocess
import time
from typing import Any
import unittest
import urllib.error
import urllib.request

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from utils.bazel import get_runfiles_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image() -> str:
    """Load the OCI image into Docker using the image_load.sh script.

    The oci_load rule in rules_oci 2.x creates a shell script that pipes
    a tar stream to docker load. The image tag is parsed directly from the
    script's stdout ("Loaded image: <tag>") rather than reconstructing it
    from version.txt, since version.txt is not available in the test's
    runfiles (the oci_load/oci_image rules don't propagate Python runfiles).

    Returns:
        The name/tag of the loaded image.

    Raises:
        FileNotFoundError: If the loader script cannot be found.
        subprocess.CalledProcessError: If loading fails.
        ValueError: If the image tag cannot be parsed from the script output.
    """
    loader_script = get_runfiles_path("services/vlm/image_load.sh")

    if not loader_script or not os.path.isfile(loader_script):
        raise FileNotFoundError(
            "Could not find image_load.sh script. Make sure the test depends on //services/vlm:image_load."
        )

    logger.info(f"Loading image using script: {loader_script}")

    result = subprocess.run(
        ["bash", loader_script],
        capture_output=True,
        text=True,
        check=True,
    )

    logger.info(f"Image load output: {result.stdout}")
    if result.stderr:
        logger.warning(f"Image load stderr: {result.stderr}")

    match = re.search(r"Loaded image:\s*(\S+)", result.stdout)
    if not match:
        raise ValueError(f"Could not parse image tag from image_load.sh output: {result.stdout}")

    image_tag = match.group(1)
    logger.info(f"Parsed image tag: {image_tag}")
    return image_tag


_raw_port = os.environ.get("UVICORN_PORT")
if _raw_port is None:
    raise RuntimeError(
        'UVICORN_PORT environment variable is not set. Set it in the test BUILD rule via env = {"UVICORN_PORT": ...}.'
    )
try:
    UVICORN_PORT = int(_raw_port)
except ValueError:
    raise RuntimeError(f"UVICORN_PORT environment variable must be an integer, got: {_raw_port!r}")


class VlmImageContainer(DockerContainer):
    """Custom container class for the VLM service."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        """Initialize the VLM container."""
        super().__init__(image, **kwargs)
        self.with_bind_ports(UVICORN_PORT, UVICORN_PORT)
        self.with_env("UVICORN_PORT", str(UVICORN_PORT))
        self.with_env("UVICORN_HOST", "0.0.0.0")
        # Use staging env to avoid local env file loading
        self.with_env("COSMOS_EVALUATOR_ENV", "staging")
        self.with_env("COSMOS_EVALUATOR_STORAGE_TYPE", "local")

    def get_health_url(self) -> str:
        """Get the URL for the health endpoint."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/health"

    def get_config_url(self) -> str:
        """Get the URL for the config endpoint."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/config"


class TestVlmImage(unittest.TestCase):
    """Integration tests for the VLM container image."""

    container: VlmImageContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and start the container."""
        cls.image_name = load_image()
        logger.info(f"Using image: {cls.image_name}")

        # Start a shared container for all tests
        cls.container = VlmImageContainer(cls.image_name)
        cls.container.waiting_for(LogMessageWaitStrategy("Application startup complete").with_startup_timeout(600))
        cls.container.start()
        logger.info("Container is ready for tests")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stops the shared container.

        The loaded Docker image is intentionally kept to preserve layer cache for
        subsequent runs. The CI after_script handles cleanup via docker image prune.
        """
        if hasattr(cls, "container") and cls.container:
            cls.container.stop()
            logger.info("Container stopped")

    def test_container_starts_and_becomes_healthy(self) -> None:
        """Tests that the container starts and the health endpoint responds."""
        health_url = self.container.get_health_url()
        logger.info(f"Checking health endpoint: {health_url}")

        max_retries = 10
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    data = json.loads(response.read().decode())

                    self.assertTrue(data.get("success"), f"Health check failed: {data}")
                    self.assertEqual(
                        data["data"]["status"],
                        "healthy",
                        f"Unexpected health status: {data}",
                    )
                    self.assertIn("version", data["data"])
                    self.assertEqual(data["data"]["service"], "VLM Service API")

                    logger.info(f"Health check passed: {data}")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)

        self.fail(f"Health endpoint did not respond after {max_retries} attempts: {last_error}")

    def test_health_endpoint_returns_valid_git_sha(self) -> None:
        """Tests that the health endpoint returns a valid git SHA from git_sha_layer."""
        health_url = self.container.get_health_url()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    data = json.loads(response.read().decode())

                    git_sha = data["data"].get("git_sha")
                    self.assertIsNotNone(git_sha, "git_sha field missing from health response")
                    self.assertEqual(len(git_sha), 40, f"git_sha should be 40 hex characters, got: {git_sha}")
                    int(git_sha, 16)  # Validates it's a hex string

                    logger.info(f"git_sha from container: {git_sha}")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)

        self.fail("Health endpoint did not respond")

    def test_container_exposes_openapi_docs(self) -> None:
        """Tests that the container serves OpenAPI documentation."""
        host = self.container.get_container_host_ip()
        docs_url = f"http://{host}:{UVICORN_PORT}/openapi.json"

        logger.info(f"Checking OpenAPI docs: {docs_url}")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(docs_url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    self.assertIn("openapi", data)
                    self.assertIn("paths", data)
                    self.assertIn("/health", data["paths"])
                    self.assertIn("/process/preset", data["paths"])
                    self.assertIn("/config", data["paths"])
                    logger.info("OpenAPI documentation is available")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)

        self.fail("OpenAPI endpoint did not respond")

    def test_third_party_notices_exists(self) -> None:
        """Tests that THIRD-PARTY-NOTICES.txt is present in the container image."""
        result = self.container.get_wrapped_container().exec_run("test -f /app/THIRD-PARTY-NOTICES.txt")
        self.assertEqual(
            result.exit_code,
            0,
            "THIRD-PARTY-NOTICES.txt not found at /app/THIRD-PARTY-NOTICES.txt in the container image",
        )

    def test_container_stays_running(self) -> None:
        """Tests that the container stays running without crashing."""
        # Check that container is still running after 10 seconds
        time.sleep(10)

        # Verify container is still running
        container_info = self.container.get_wrapped_container()
        container_info.reload()

        self.assertEqual(
            container_info.status,
            "running",
            f"Container stopped unexpectedly. Logs: {self.container.get_logs()}",
        )
        logger.info("Container stayed running for 10 seconds")


if __name__ == "__main__":
    unittest.main()
