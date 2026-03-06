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

"""
Entry point for attribute verification processing.

This script provides a unified interface for running attribute verification checks
with configuration management and command line argument parsing.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import uuid

from dotenv import load_dotenv
import yaml

from checks.attribute_verification.processor import AttributeVerificationProcessor
from checks.utils.config_manager import ConfigManager
from checks.utils.multistorage import setup_msc_config, validate_uri


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Attribute Verification Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clip-id", type=str, required=False, default=str(uuid.uuid4()), help="Clip ID for processing")
    parser.add_argument("--augmented-video-path", type=str, required=True, help="Path to the augmented video file")

    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Configuration file name (without .yaml extension, defaults to 'config')",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default="checks/attribute_verification/.env",
        help="Path to the .env file to load",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to the output directory",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    print(f"Validating augmented video path: {args.augmented_video_path}")
    return validate_uri(args.augmented_video_path)


def load_config(config_name: str = "config", config_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load configuration from the specified config file."""
    config_manager = ConfigManager(config_dir)
    config = config_manager.load_config(config_name)
    return config


def save_results_to_file(results: Dict[str, Any], clip_id: str, output_dir: str) -> str:
    """
    Save processing results to a JSON file.

    Args:
        results: Results dictionary from process_clip
        clip_id: Clip ID for filename
        output_dir: Output directory path

    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to file
    output_file = output_path / f"{clip_id}.attribute_verification.results.json"

    with open(output_file, "w") as f:
        f.write(json.dumps(results, indent=2))

    return str(output_file)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load Environment Variables
    if os.path.exists(args.env_file):
        load_dotenv(dotenv_path=args.env_file)
    else:
        print(f"Warning: .env file not found: {args.env_file}")

    setup_msc_config(os.environ.get("MULTISTORAGECLIENT_CONFIGURATION"))

    # Load Configuration
    try:
        config = load_config(args.config, args.config_dir)
    except FileNotFoundError as e:
        print(f"Warning: Configuration file '{args.config}.yaml' not found: {e}")
        config = {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in configuration file '{args.config}.yaml': {e}")
        sys.exit(1)

    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)
    config = config.get("metropolis.attribute_verification", {})

    # Validate arguments
    if not validate_args(args):
        print("Argument validation failed. Exiting.")
        sys.exit(1)

    # Process the sample
    try:
        processor = AttributeVerificationProcessor(params=config, config_dir=args.config_dir, verbose=args.verbose)
        result = asyncio.run(processor.process(args.clip_id, args.augmented_video_path))
        save_results_to_file(result.model_dump(), args.clip_id, args.output_dir)
        print(
            f"\nCheck complete, results saved to: bazel-out/k8-fastbuild/bin/checks/attribute_verification/run.runfiles/_main/{args.output_dir}/{args.clip_id}.attribute_verification.results.json"
        )
        print(f"\nPassed: {result.passed}")
        print(f"Total checks: {result.summary.total_checks}")
        print(f"Passed checks: {result.summary.passed_checks}")
        print(f"Failed checks: {result.summary.failed_checks}")
        print("Checks:")
        for check in result.checks:
            print(f"\t{check.variable}: {check.passed}")
    except Exception as e:
        import traceback

        print(f"Error processing sample: {e}")
        if args.verbose == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
