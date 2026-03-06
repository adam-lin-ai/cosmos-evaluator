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

"""Shared test infrastructure for Obstacle Correspondence container image tests.

Provides a reusable mixin and helpers for loading OCI images and running standard
integration checks (health, git SHA, OpenAPI, licensing, stability). Concrete test
modules combine ImageTestMixin with unittest.TestCase and set LOADER_SCRIPT_PATH.
"""

import json
import logging
import os
import re
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from utils.bazel import get_runfiles_path

logger = logging.getLogger(__name__)


def load_image(loader_script_path: str) -> str:
    """Load an OCI image into Docker using the given loader script.

    The oci_load rule in rules_oci 2.x creates a shell script that pipes
    a tar stream to docker load. The image tag is parsed directly from the
    script's stdout ("Loaded image: <tag>") rather than reconstructing it
    from version.txt, since version.txt is not available in the test's
    runfiles (the oci_load/oci_image rules don't propagate Python runfiles).

    Args:
        loader_script_path: Runfiles-relative path to the oci_load shell script
            (e.g. "services/obstacle_correspondence/image_load.sh").

    Returns:
        The name/tag of the loaded image.

    Raises:
        FileNotFoundError: If the loader script cannot be found.
        subprocess.CalledProcessError: If loading fails.
        ValueError: If the image tag cannot be parsed from the script output.
    """
    loader_script = get_runfiles_path(loader_script_path)

    if not loader_script or not os.path.isfile(loader_script):
        raise FileNotFoundError(
            f"Could not find {loader_script_path}. Make sure the test depends on the corresponding oci_load target."
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
        raise ValueError(f"Could not parse image tag from output: {result.stdout}")

    image_tag = match.group(1)
    logger.info(f"Parsed image tag: {image_tag}")
    return image_tag


def get_uvicorn_port() -> int:
    """Read and validate the UVICORN_PORT environment variable."""
    raw_port = os.environ.get("UVICORN_PORT")
    if raw_port is None:
        raise RuntimeError(
            "UVICORN_PORT environment variable is not set. "
            'Set it in the test BUILD rule via env = {"UVICORN_PORT": ...}.'
        )
    try:
        return int(raw_port)
    except ValueError:
        raise RuntimeError(f"UVICORN_PORT environment variable must be an integer, got: {raw_port!r}")


UVICORN_PORT = get_uvicorn_port()


class ObstacleCorrespondenceContainer(DockerContainer):
    """Container wrapper for the Obstacle Correspondence service image variants."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        super().__init__(image, **kwargs)
        self.with_bind_ports(UVICORN_PORT, UVICORN_PORT)
        self.with_env("UVICORN_PORT", str(UVICORN_PORT))
        self.with_env("UVICORN_HOST", "0.0.0.0")
        self.with_env("COSMOS_EVALUATOR_ENV", "staging")
        self.with_env("COSMOS_EVALUATOR_STORAGE_TYPE", "local")

    def get_health_url(self) -> str:
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/health"

    def get_config_url(self) -> str:
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/config"


class ImageTestMixin:
    """Mixin providing integration tests for Obstacle Correspondence container images.

    Not a unittest.TestCase subclass, so unittest will not discover or run it directly.
    Concrete test classes should inherit from both this mixin and unittest.TestCase,
    and set LOADER_SCRIPT_PATH to the runfiles-relative path of the oci_load script.
    """

    LOADER_SCRIPT_PATH: str

    container: ObstacleCorrespondenceContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and starts the container."""
        cls.image_name = load_image(cls.LOADER_SCRIPT_PATH)
        logger.info(f"Using image: {cls.image_name}")

        cls.container = ObstacleCorrespondenceContainer(cls.image_name)
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

                    self.assertTrue(data.get("success"), f"Health check failed: {data}")  # type: ignore[attr-defined]
                    self.assertEqual(  # type: ignore[attr-defined]
                        data["data"]["status"],
                        "healthy",
                        f"Unexpected health status: {data}",
                    )
                    self.assertIn("version", data["data"])  # type: ignore[attr-defined]
                    self.assertEqual(data["data"]["service"], "Obstacle Correspondence Service API")  # type: ignore[attr-defined]

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

        self.fail(f"Health endpoint did not respond after {max_retries} attempts: {last_error}")  # type: ignore[attr-defined]

    def test_health_endpoint_returns_valid_git_sha(self) -> None:
        """Tests that the health endpoint returns a valid git SHA from git_sha_layer."""
        health_url = self.container.get_health_url()

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    data = json.loads(response.read().decode())

                    git_sha = data["data"].get("git_sha")
                    self.assertIsNotNone(git_sha, "git_sha field missing from health response")  # type: ignore[attr-defined]
                    self.assertEqual(len(git_sha), 40, f"git_sha should be 40 hex characters, got: {git_sha}")  # type: ignore[attr-defined]
                    int(git_sha, 16)  # Validates it's a hex string

                    logger.info(f"git_sha from container: {git_sha}")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)

        self.fail("Health endpoint did not respond")  # type: ignore[attr-defined]

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
                    self.assertIn("openapi", data)  # type: ignore[attr-defined]
                    self.assertIn("paths", data)  # type: ignore[attr-defined]
                    self.assertIn("/health", data["paths"])  # type: ignore[attr-defined]
                    self.assertIn("/process", data["paths"])  # type: ignore[attr-defined]
                    self.assertIn("/config", data["paths"])  # type: ignore[attr-defined]
                    logger.info("OpenAPI documentation is available")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)

        self.fail("OpenAPI endpoint did not respond")  # type: ignore[attr-defined]

    def test_third_party_notices_exists(self) -> None:
        """Tests that THIRD-PARTY-NOTICES.txt is present in the container image."""
        result = self.container.get_wrapped_container().exec_run("test -f /app/THIRD-PARTY-NOTICES.txt")
        self.assertEqual(  # type: ignore[attr-defined]
            result.exit_code,
            0,
            "THIRD-PARTY-NOTICES.txt not found at /app/THIRD-PARTY-NOTICES.txt in the container image",
        )

    def test_container_stays_running(self) -> None:
        """Tests that the container stays running without crashing."""
        time.sleep(10)

        container_info = self.container.get_wrapped_container()
        container_info.reload()

        self.assertEqual(  # type: ignore[attr-defined]
            container_info.status,
            "running",
            f"Container stopped unexpectedly. Logs: {self.container.get_logs()}",
        )
        logger.info("Container stayed running for 10 seconds")
