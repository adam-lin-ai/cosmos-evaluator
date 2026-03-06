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

"""Unit tests for config_manager module."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from checks.utils.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)

        # Sample YAML config for testing
        self.sample_config = {
            "av.obstacle": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                    "motorcycle": {"method": "cluster"},
                },
                "importance_filter": {
                    "distance_threshold_m": 100,
                    "skip_oncoming_obstacles": False,
                    "relevant_lanes": ["ego", "left", "right"],
                },
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("checks.utils.config_manager.bazel.get_runfiles_path")
    def test_init_default_config_dir(self, mock_get_runfiles_path):
        """Test ConfigManager initialization with default config directory."""
        # Mock the bazel runfiles path to return a config file path
        mock_config_path = "/fake/path/to/checks/config.yaml"
        mock_get_runfiles_path.return_value = mock_config_path

        cm = ConfigManager()
        expected_path = Path(mock_config_path).parent
        self.assertEqual(cm._config_dir, expected_path)

    @patch("checks.utils.config_manager.bazel.get_runfiles_path")
    def test_init_default_config_dir_not_found(self, mock_get_runfiles_path):
        """Test ConfigManager initialization when config file is not found."""
        # Mock the bazel runfiles path to return None (file not found)
        mock_get_runfiles_path.return_value = None

        with self.assertRaises(FileNotFoundError) as context:
            ConfigManager()

        self.assertIn("Configuration file not found", str(context.exception))
        self.assertIn("checks/config.yaml", str(context.exception))

    def test_init_custom_config_dir(self):
        """Test ConfigManager initialization with custom config directory."""
        self.assertEqual(self.config_manager._config_dir, Path(self.temp_dir))

    def test_load_config_success(self):
        """Test successful config loading."""
        config_path = Path(self.temp_dir) / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.sample_config, f)

        result = self.config_manager.load_config("test")
        self.assertEqual(result, self.sample_config)
        self.assertIn("test", self.config_manager._configs)
        self.assertEqual(self.config_manager._configs["test"], self.sample_config)

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        config_path = Path(self.temp_dir) / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(yaml.YAMLError):
            self.config_manager.load_config("invalid")

    def test_load_config_path_traversal_forward_slash(self):
        """Test that config names with forward slashes are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("../etc/passwd")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_backslash(self):
        """Test that config names with backslashes are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("..\\windows\\system32")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_subdirectory(self):
        """Test that config names with directory separators are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("subdir/config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_parent_directory(self):
        """Test that config names with parent directory references are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("..config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_valid_name_with_hyphens_underscores(self):
        """Test that valid config names with hyphens and underscores work."""
        config_path = Path(self.temp_dir) / "valid-config_name.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"test": "value"}, f)

        result = self.config_manager.load_config("valid-config_name")
        self.assertEqual(result["test"], "value")

    def test_load_config_caching(self):
        """Test that config is cached and not reloaded."""
        config_path = Path(self.temp_dir) / "cached.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"test": "value"}, f)

        # Load first time
        result1 = self.config_manager.load_config("cached")

        # Modify file
        with open(config_path, "w") as f:
            yaml.dump({"test": "modified"}, f)

        # Load second time - should return cached version
        result2 = self.config_manager.load_config("cached")
        self.assertEqual(result1, result2)
        self.assertEqual(result2["test"], "value")

    def test_reload_config(self):
        """Test config reloading."""
        config_path = Path(self.temp_dir) / "reload.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"version": 1}, f)

        # Load initial config
        result1 = self.config_manager.load_config("reload")
        self.assertEqual(result1["version"], 1)

        # Modify file
        with open(config_path, "w") as f:
            yaml.dump({"version": 2}, f)

        # Reload config
        result2 = self.config_manager.reload_config("reload")
        self.assertEqual(result2["version"], 2)

    def test_reload_config_path_traversal(self):
        """Test that reload_config rejects invalid config names."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.reload_config("../../secret/config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_list_configs(self):
        """Test listing available configs."""
        # Create test config files
        for name in ["config1", "config2", "config3"]:
            config_path = Path(self.temp_dir) / f"{name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump({"test": "data"}, f)

        configs = self.config_manager.list_configs()
        expected_configs = {"config1", "config2", "config3"}
        self.assertEqual(set(configs), expected_configs)

    def test_list_configs_empty_dir(self):
        """Test listing configs in empty directory."""
        configs = self.config_manager.list_configs()
        self.assertEqual(configs, [])


if __name__ == "__main__":
    unittest.main()
