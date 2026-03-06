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
Entry point for obstacle correspondence processing.

This script provides a unified interface for running obstacle correspondence checks
with configuration management and command line argument parsing.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from checks.obstacle.api import run_object_processors
from checks.utils.cli_common import load_default_envs, validate_paths
from checks.utils.config_manager import ConfigManager


def parse_args():
    """Parse command line arguments.

    Expected CLI:
      dazel run //checks/obstacle:run [cloud|local] [args...]
    """
    parser = argparse.ArgumentParser(
        description="Obstacle Correspondence Processing Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parent for common args (shared by cloud/local)
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    common_parent.add_argument("--camera_name", type=str, required=True, help="Camera name for processing")
    common_parent.add_argument(
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    common_parent.add_argument(
        "--config",
        type=str,
        default="config",
        help="Configuration file name (without .yaml extension)",
    )
    common_parent.add_argument(
        "--trial", type=int, default=None, help="Trial mode: process only up to N frames (useful for debugging)"
    )
    common_parent.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target FPS of object/pose timeline (defaults to 30). Video frames will be mapped to this FPS.",
    )
    common_parent.add_argument(
        "--model_device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to run the model on"
    )

    # Input mode subparsers (local/cloud)
    mode_parsers = parser.add_subparsers(dest="mode", required=True, help="Input mode")

    # local mode
    local_parser = mode_parsers.add_parser(
        "local",
        help="Use local files as input",
        parents=[common_parent],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    local_parser.add_argument("--input_data", type=str, required=True, help="Path to the input data directory")
    local_parser.add_argument("--clip_id", type=str, required=True, help="Clip ID for processing")
    local_parser.add_argument("--video_path", type=str, required=True, help="Path to the camera video file")
    local_parser.add_argument(
        "--world_video_path",
        type=str,
        default=None,
        help="Path to the world video file for CWIP static processing. If omitted, static processing is skipped.",
    )

    # cloud mode
    cloud_parser = mode_parsers.add_parser(
        "cloud",
        help="Read metadata JSON with S3 URLs",
        parents=[common_parent],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mutually exclusive input source: metadata file OR video UUID
    cloud_input_group = cloud_parser.add_mutually_exclusive_group(required=True)
    cloud_input_group.add_argument(
        "--metadata_file",
        type=str,
        help="Path to metadata JSON containing S3 URLs for video and input data",
    )
    cloud_input_group.add_argument(
        "--video_uuid",
        type=str,
        help="Video UUID to fetch metadata from database",
    )
    cloud_parser.add_argument(
        "--db_name",
        type=str,
        default="sdg_prod",
        help="Database config section name for video UUID lookup",
    )

    return parser.parse_args()


def load_config(config_name: str = "config") -> Optional[Dict[str, Any]]:
    """
    Load configuration from the specified config file.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        Configuration dictionary or None if loading fails
    """
    try:
        # Try default ConfigManager (works for direct Python execution)
        config_manager = ConfigManager()
        config = config_manager.load_config(config_name)
        return config
    except FileNotFoundError as e:
        print(f"Warning: Configuration file '{config_name}.yaml' not found: {e}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
    return None


def validate_args(args):
    """Validate command line arguments."""
    if args.mode == "local":
        if args.world_video_path is not None and not Path(args.world_video_path).exists():
            print(f"Error: World video file does not exist: {args.world_video_path}")
            return False
    return True


def _print_results(results: Dict[str, Any], verbose: bool) -> None:
    """Print processing results summary.

    Args:
        results: Results dictionary from a processor.
        verbose: Whether to print detailed summary.
    """
    print(f"  - Total frames processed: {results['processed_frames']}")
    if "track_ids" in results and results["track_ids"]:
        print(f"  - Unique track IDs: {len(results['track_ids'])}")
    if "score_matrix" in results and results["score_matrix"] is not None:
        print(f"  - Score matrix shape: {results['score_matrix'].shape}")
    print(f"  - Mean correspondence score: {results['mean_score']:.3f}")
    print(f"  - Score range: {results['min_score']:.3f} - {results['max_score']:.3f}")
    print(f"  - Standard deviation: {results['std_score']:.3f}")

    if verbose:
        # Print a more readable summary instead of the raw results dictionary
        print("\nDetailed Summary:")
        print(f"  - Processed frames: {results['processed_frames']}")
        print(f"  - Unique track IDs: {len(results['track_ids'])}")
        print("  - Score statistics:")
        print(f"    * Mean: {results['mean_score']:.3f}")
        print(f"    * Std: {results['std_score']:.3f}")
        print(f"    * Min: {results['min_score']:.3f}")
        print(f"    * Max: {results['max_score']:.3f}")


def main():
    """Main entry point."""
    log = logging.getLogger(__name__)
    args = parse_args()

    if not validate_args(args):
        print("Argument validation failed. Exiting.")
        sys.exit(1)

    # Load AWS credentials if cloud mode
    if args.mode == "cloud":
        load_default_envs()

    # Load configuration
    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded configuration from '{args.config}.yaml'")
        print(f"Configuration: {config}")

    # Load inputs based on local/cloud mode
    if args.mode == "local":
        inputs = {
            "video_path": args.video_path,
            "rds_hq_path": args.input_data,
            "clip_id": args.clip_id,
        }
        world_video_path: Optional[str] = args.world_video_path
    else:
        log.error("Cloud mode is not supported for obstacle correspondence processing")
        sys.exit(1)

    # Validate input paths exist and ensure output dir exists (both modes)
    if not validate_paths(
        input_data=inputs["rds_hq_path"],
        video_path=inputs["video_path"],
        output_dir=args.output_dir,
        create_output_dir=True,
    ):
        print("Input path validation failed. Exiting.")
        sys.exit(1)

    # Import and run the actual processing
    try:
        # Display configuration summary
        print("Configuration loaded successfully:")
        print(f"  - Overlap check: {config.get('overlap_check', {})}")
        print(f"  - Importance filter: {config.get('importance_filter', {})}")
        print(f"  - Model device: {args.model_device}")

        print("\nProcessing with arguments:")
        print(f"  - Input data: {inputs['rds_hq_path']}")
        print(f"  - Clip ID: {inputs['clip_id']}")
        print(f"  - Camera name: {args.camera_name}")
        print(f"  - Video path: {inputs['video_path']}")
        print(f"  - World video path: {world_video_path or 'N/A (static processing skipped)'}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - Log level: {args.verbose}")
        if args.trial is not None:
            print(f"  - Trial mode: {args.trial} frames")

        # Process the clip using the unified run() function
        print("\nStarting obstacle correspondence processing...")
        combined_results = run_object_processors(
            config=config,
            input_data=inputs["rds_hq_path"],
            clip_id=inputs["clip_id"],
            camera_name=args.camera_name,
            video_path=inputs["video_path"],
            output_dir=args.output_dir,
            world_video_path=world_video_path,
            model_device=args.model_device,
            verbose=args.verbose,
            trial_frames=args.trial,
            target_fps=args.target_fps,
        )

        # Display results for both processors
        print("\nProcessing complete!")
        for processor_name, results in combined_results.items():
            print(f"\n=== {processor_name.upper()} Processor Results ===")
            _print_results(results, args.verbose)

    except ImportError as e:
        print(f"Error importing processing module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
