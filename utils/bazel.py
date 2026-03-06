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

import logging
from os import path
from typing import Optional

from python.runfiles import runfiles

logger = logging.getLogger(__name__)

"""Bazel utility functions."""


def get_runfiles_path(repo_path: str, external_repo: str = "_main") -> Optional[str]:
    """Get the path to a file using Bazel runfiles.

    Args:
        repo_path: Path to the file from the root of the repository (e.g., "checks/vlm/config/endpoints.json")
        external_repo: Name of the external repository to use without the leading @ (default: "_main")

    Returns:
        Runtime path of a file in Bazel runfiles if found, None otherwise
    """
    try:
        r = runfiles.Create()

        if not r:
            logger.debug("Runfiles.Create() failed")
            return None

        clean_repo_path = repo_path.lstrip(path.sep)
        runfiles_root_relative_path = path.join(external_repo, clean_repo_path)
        runtime_path = r.Rlocation(runfiles_root_relative_path)

        if not path.exists(runtime_path):
            logger.error("Runfiles path not found: {}".format(runtime_path))
            return None

        return runtime_path
    except (IOError, TypeError, ValueError) as e:
        logger.error("Exception while getting runfiles path: {}".format(e))
        return None
