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

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

from checks.utils.cli_common import load_default_envs
from checks.utils.config_manager import ConfigManager
from checks.vlm.preset_processor import process_preset

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parses the arguments for the VLM checks CLI.

    Returns:
        The parsed arguments
    """
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write output JSON (file will be named using clip_id).",
    )
    parent_parser.add_argument(
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Main parser
    parser = argparse.ArgumentParser(
        description="VLM Checks CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers for local and cloud modes
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Input mode",
    )

    # Local mode subcommand (inherits common arguments from parent)
    local_parser = subparsers.add_parser(
        "local",
        help="Use local files as input",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    local_parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video",
    )
    local_parser.add_argument(
        "--clip_id",
        type=str,
        required=True,
        help="Clip ID for naming outputs",
    )
    local_parser.add_argument(
        "--preset_file",
        type=str,
        help="Path to a JSON file containing the preset object (required if preset check is enabled)",
    )

    # Cloud mode subcommand (inherits common arguments from parent)
    cloud_parser = subparsers.add_parser(
        "cloud",
        help="Read metadata JSON with S3 URLs",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mutually exclusive input source: metadata file OR video UUID
    cloud_input_group = cloud_parser.add_mutually_exclusive_group(required=True)
    cloud_input_group.add_argument(
        "--metadata_file",
        type=str,
        help="Path to metadata JSON containing S3 URLs for 'video_s3_url' and 'preset_s3_url'",
    )
    cloud_input_group.add_argument(
        "--video_uuid",
        type=str,
        help="Video UUID to fetch metadata from database",
    )

    return parser.parse_args()


def _load_config() -> Dict[str, Any]:
    """Loads the configuration from the config.yaml file.

    Returns:
        The configuration
    """
    cm = ConfigManager()
    return cm.load_config("config")


def _load_local_preset_conditions(preset_file: Optional[str]) -> Dict[str, Any]:
    """Loads the local preset conditions from the preset file.

    Args:
        preset_file: The path to the preset file

    Returns:
        The preset conditions

    Raises:
        ValueError: If the preset file is unavailable
        TypeError: If the preset is not a JSON object with required fields
    """
    obj: Any = None
    if preset_file:
        with open(preset_file, "r") as f:
            obj = json.load(f)
    else:
        raise ValueError("--preset_file must be provided in local mode")

    # Support either direct preset dict or wrapper with top-level 'preset'
    if isinstance(obj, dict) and "preset" in obj and isinstance(obj["preset"], dict):
        return obj["preset"]
    if isinstance(obj, dict):
        return obj
    raise TypeError("preset must be a JSON object with required fields")


def _prepare_debug_dir(base_dir: str, video_path: str) -> Optional[str]:
    """Return a debug directory path ensuring existence."""
    try:
        combined = Path(base_dir) / Path(video_path).stem
        combined.mkdir(parents=True, exist_ok=True)
        return str(combined)
    except Exception:
        return base_dir


def _setup_logging(verbosity: str) -> None:
    level = getattr(logging, (verbosity or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.debug("Logging initialized at level %s", logging.getLevelName(level))
    # Suppress noisy third-party DEBUG logs while keeping our own DEBUG logs
    try:
        if level <= logging.DEBUG:
            for name in ("openai", "httpx", "httpcore", "urllib3", "boto3", "botocore", "s3transfer"):
                logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass


def main():
    """Main entry point for the VLM checks CLI."""
    args = _parse_args()
    _setup_logging(args.verbose)

    # Load env files for AWS/VLM credentials before any S3 or VLM calls
    logger.debug("Loading environment files for AWS and VLM credentials")
    load_default_envs(additional_env_paths=["~/.cosmos_evaluator/.env"])

    vlm_cfg = _load_config().get("av.vlm", {})

    results: Dict[str, Any] = {}
    meta_inputs: Dict[str, Any] = {}

    if args.mode == "local":
        meta_inputs["video_path"] = args.video_path
        meta_inputs["clip_id"] = args.clip_id

    elif args.mode == "cloud":
        logger.error("Cloud mode is not supported for VLM checks")
        sys.exit(1)

    if meta_inputs.get("video_path") is None:
        raise ValueError("Missing input video for VLM check")

    logger.info("Starting VLM checks")
    logger.info("  - Clip ID: %s", meta_inputs.get("clip_id"))
    logger.info("  - Video path: %s", meta_inputs.get("video_path"))
    logger.info("  - Output directory: %s", args.output_dir)
    logger.info("  - Log level: %s", args.verbose)

    # DEBUG mode saves debug artifacts to a separate directory
    debug_output_dir = None
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        # Prepare debug directory once; processors will write under it
        # debug directory = {output_dir}/{video_path.stem}
        debug_output_dir = _prepare_debug_dir(str(args.output_dir), meta_inputs.get("video_path"))
        logger.info("  - Debug output directory: %s", debug_output_dir)

    # Run enabled VLM checks
    if vlm_cfg.get("preset_check", {}).get("enabled", False):
        if args.mode == "local":
            meta_inputs["preset_conditions"] = _load_local_preset_conditions(args.preset_file)
        logger.info("Running preset check")
        logger.info("  - Preset conditions: %s", meta_inputs["preset_conditions"])
        meta_inputs["preset_conditions"]["name"] = "Environment"
        results["preset_check"] = process_preset(
            meta_inputs.get("video_path"), meta_inputs.get("preset_conditions"), vlm_cfg.get("preset_check")
        )

    # Build top-level schema: clip_id, video_uuid, and per-check results
    video_uuid = Path(meta_inputs["video_path"]).stem
    final_out: Dict[str, Any] = {
        "clip_id": meta_inputs.get("clip_id"),
        "video_uuid": video_uuid,
    }
    final_out.update(results)

    # Output always to output_dir using clip_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_uuid}.vlm.check.json"
    with open(out_path, "w") as f:
        json.dump(final_out, f, indent=4)
    logger.info("Results written to %s", out_path)

    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        # Copy input video
        src_video = Path(meta_inputs.get("video_path")) if meta_inputs.get("video_path") else None
        if src_video and src_video.exists():
            shutil.copy2(src_video, Path(debug_output_dir) / "video.mp4")

        # Write a copy of the output JSON
        debug_json_path = Path(debug_output_dir) / "vlm.check.json"
        with open(debug_json_path, "w") as f_dbg:
            json.dump(final_out, f_dbg, indent=4)

        logger.info("Debug artifacts saved to %s", debug_output_dir)


if __name__ == "__main__":
    main()
