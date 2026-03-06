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
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils import bazel

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    A configuration manager for Cosmos Evaluator that loads and manages YAML configuration files.

    This class provides easy access to configuration parameters from YAML files.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config manager.

        Args:
            config_dir: Path to the configuration directory. If None, defaults to
                       checks/config relative to the current working directory.
        """
        if config_dir:
            config_dir = Path(config_dir)
        else:
            config_path = bazel.get_runfiles_path("checks/config.yaml")
            if not config_path:
                raise FileNotFoundError("Configuration file not found: checks/config.yaml")
            config_dir = Path(config_path).parent

        self._config_dir = config_dir
        self._configs: Dict[str, Any] = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file by name.

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Dictionary containing the configuration data

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        # Validate config_name to prevent path traversal
        if "/" in config_name or "\\" in config_name or ".." in config_name:
            raise ValueError(f"Invalid config name: {config_name}")

        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self._config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            self._configs[config_name] = config_data
            return config_data

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """
        Force reload a configuration file, even if it was previously loaded.

        Args:
            config_name: Name of the config file to reload

        Returns:
            Dictionary containing the configuration data
        """
        if config_name in self._configs:
            del self._configs[config_name]

        return self.load_config(config_name)

    def list_configs(self) -> List[str]:
        """
        List all available configuration files.

        Returns:
            List of configuration file names (without .yaml extension)
        """
        return [file_path.stem for file_path in self._config_dir.glob("*.yaml")]
