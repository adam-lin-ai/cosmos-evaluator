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

"""Tests for the Obstacle Correspondence REST API entry point (rest_api.py).

This test depends on //services/obstacle_correspondence:rest_api which has //utils:git_sha
in its data attribute, so get_git_sha() reads from Bazel runfiles without mocking.
"""

import unittest

from fastapi.testclient import TestClient

from services.obstacle_correspondence.rest_api import app


class TestGetGitSha(unittest.TestCase):
    """Test that the /health endpoint returns a valid git SHA from Bazel runfiles."""

    def test_health_endpoint_returns_valid_git_sha(self) -> None:
        """The /health endpoint should return a real 40-character hex git SHA."""
        client = TestClient(app)
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        git_sha = data["data"]["git_sha"]

        self.assertIsInstance(git_sha, str)
        self.assertEqual(len(git_sha), 40, f"Expected 40-char hex SHA, got: {git_sha}")
        int(git_sha, 16)  # Raises ValueError if not valid hex


if __name__ == "__main__":
    unittest.main()
