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
Prepare batch generation results for analysis by analyze_results.py.

This script reads metadata JSON files from batch generation runs, downloads
the obstacle results JSON files from S3, and organizes them into the directory
structure expected by analyze_results.py.

Each batch directory should contain metadata files from a single model.
The script validates this assumption and uses provided model aliases for
the output directory naming.

Environment Setup:
    AWS credentials can be loaded from ~/.aws/.env or set via environment variables:
    - AWS_ACCESS_KEY_ID (required)
    - AWS_SECRET_ACCESS_KEY (required)
    - AWS_SESSION_TOKEN (optional)
    - AWS_DEFAULT_REGION or AWS_REGION (optional, defaults to us-west-2)
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple


def load_aws_credentials_from_dotenv(verbose: bool = False) -> None:
    """Load AWS credentials from ~/.aws/.env if available.

    Args:
        verbose: If True, print message when credentials are loaded.
    """
    try:
        from dotenv import load_dotenv

        aws_env_path = Path.home() / ".aws" / ".env"
        if aws_env_path.exists():
            load_dotenv(aws_env_path)
            if verbose:
                print(f"Loaded AWS credentials from {aws_env_path}")
    except ImportError:
        pass  # dotenv not available, rely on existing env vars


class S3Client:
    """Wrapper for boto3 S3 client with credential handling."""

    def __init__(self) -> None:
        """Initialize S3 client with credentials from environment variables.

        Raises:
            RuntimeError: If boto3 is not available or credentials are missing.
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise RuntimeError("boto3 is required to download from S3") from e

        # Store exception types for use in download_file
        self._botocore_errors = (BotoCoreError, ClientError)

        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN")  # optional
        self.region_name = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"

        if not (self.aws_access_key_id and self.aws_secret_access_key):
            raise RuntimeError("Missing AWS credentials. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")

        client_kwargs: Dict[str, Any] = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.region_name,
        }
        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token

        self._client = boto3.client("s3", **client_kwargs)

    def download_file(self, s3_url: str, local_path: str) -> None:
        """Download a file from S3.

        Args:
            s3_url: S3 URL in format s3://bucket/key
            local_path: Local path to save the file

        Raises:
            ValueError: If S3 URL is invalid
            RuntimeError: If download fails (S3 errors or local disk issues)
        """
        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {s3_url}")

        path = s3_url[len("s3://") :]
        if "/" not in path:
            raise ValueError(f"Invalid S3 URL (missing key): {s3_url}")
        bucket, key = path.split("/", 1)

        try:
            self._client.download_file(bucket, key, local_path)
        except OSError as e:
            # Local disk issues (permissions, disk full, etc.)
            raise RuntimeError(f"Failed to write s3://{bucket}/{key} to {local_path}: {e}") from e
        except self._botocore_errors as e:
            # S3 client errors (access denied, not found, etc.)
            raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {e}") from e


class BatchConfig(NamedTuple):
    """Configuration for a single batch."""

    directory: str
    model_alias: str


class MetadataInfo(NamedTuple):
    """Extracted information from a metadata file."""

    metadata_path: str
    model_name: str
    model_alias: str
    clip_id_with_timestamp: str
    prompt_name: str
    results_s3_url: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace with batch configs and output directory.
    """
    parser = argparse.ArgumentParser(
        description="Prepare batch generation results for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python prepare_batch_results.py \\
    --batch /path/to/batch_1215 cosmos-edge \\
    --batch /path/to/batch_1211 cosmos-lora \\
    --output_dir /path/to/analysis_input
        """,
    )
    parser.add_argument(
        "--batch",
        nargs=2,
        action="append",
        metavar=("DIRECTORY", "MODEL_ALIAS"),
        required=True,
        help="Batch directory and model alias pair. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for organized results.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview actions without downloading files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def discover_metadata_files(directory: str) -> List[str]:
    """Discover all metadata JSON files in a directory.

    Args:
        directory: Path to the batch directory.

    Returns:
        List of paths to metadata JSON files.
    """
    pattern = os.path.join(directory, "*_metadata.json")
    return sorted(glob.glob(pattern))


def load_metadata(metadata_path: str) -> Dict:
    """Load and parse a metadata JSON file.

    Args:
        metadata_path: Path to the metadata JSON file.

    Returns:
        Parsed metadata dictionary.
    """
    with open(metadata_path, "r") as f:
        return json.load(f)


def extract_prompt_name(output_uuid: str) -> str:
    """Extract the prompt name (e.g., 'v0', 'v1') from the output UUID.

    Args:
        output_uuid: The output_uuid field from metadata
            (e.g., 'video_generated_20251215_233806_f54c6135_v0').

    Returns:
        The prompt name (e.g., 'v0').
    """
    # The format is: video_generated_YYYYMMDD_HHMMSS_HASH_vN
    parts = output_uuid.split("_")
    # Find the part that starts with 'v' and is followed by digits
    for part in reversed(parts):
        if part.startswith("v") and part[1:].isdigit():
            return part
    raise ValueError(f"Could not extract prompt name from output_uuid: {output_uuid}")


def validate_batch_model_consistency(
    metadata_files: List[str],
    batch_dir: str,
) -> Tuple[bool, str]:
    """Validate that all metadata files in a batch have the same model name.

    Args:
        metadata_files: List of metadata file paths.
        batch_dir: Path to the batch directory (for error messages).

    Returns:
        Tuple of (is_valid, common_model_name).
        If is_valid is False, common_model_name will be empty.
    """
    model_names: Set[str] = set()
    model_to_files: Dict[str, List[str]] = {}

    for metadata_path in metadata_files:
        try:
            metadata = load_metadata(metadata_path)
            model_name = metadata.get("metadata", {}).get("model_name", "")
            if not model_name:
                print(f"Warning: No model_name found in {metadata_path}", file=sys.stderr)
                continue
            model_names.add(model_name)
            if model_name not in model_to_files:
                model_to_files[model_name] = []
            model_to_files[model_name].append(os.path.basename(metadata_path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {metadata_path}: {e}", file=sys.stderr)
            continue

    if len(model_names) == 0:
        return False, ""
    elif len(model_names) == 1:
        (single_model,) = model_names  # Unpack without mutation
        return True, single_model
    else:
        print(f"Error: Multiple models found in batch {batch_dir}:", file=sys.stderr)
        for model_name, files in model_to_files.items():
            print(f"  {model_name}: {len(files)} files", file=sys.stderr)
            if len(files) <= 3:
                for f in files:
                    print(f"    - {f}", file=sys.stderr)
            else:
                for f in files[:2]:
                    print(f"    - {f}", file=sys.stderr)
                print(f"    ... and {len(files) - 2} more", file=sys.stderr)
        return False, ""


def extract_metadata_info(
    metadata_path: str,
    model_alias: str,
) -> Optional[MetadataInfo]:
    """Extract relevant information from a metadata file.

    Args:
        metadata_path: Path to the metadata JSON file.
        model_alias: The alias to use for this model.

    Returns:
        MetadataInfo if successful, None if required fields are missing or malformed.
    """
    try:
        metadata = load_metadata(metadata_path)

        model_name = metadata.get("metadata", {}).get("model_name", "")
        output_uuid = metadata.get("generated_video_info", {}).get("output_uuid", "")
        clip_id_with_timestamp = metadata.get("cosmos_evaluator_result", {}).get("clip_id_with_timestamp", "")
        results_s3_url = (
            metadata.get("cosmos_evaluator_result", {}).get("object_dynamic", {}).get("results_json_s3_url", "")
        )

        if not all([model_name, output_uuid, clip_id_with_timestamp, results_s3_url]):
            print(f"Warning: Missing required fields in {metadata_path}", file=sys.stderr)
            return None

        prompt_name = extract_prompt_name(output_uuid)

        return MetadataInfo(
            metadata_path=metadata_path,
            model_name=model_name,
            model_alias=model_alias,
            clip_id_with_timestamp=clip_id_with_timestamp,
            prompt_name=prompt_name,
            results_s3_url=results_s3_url,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # JSONDecodeError: malformed JSON
        # KeyError: unexpected structure
        # TypeError: None encountered where dict expected
        # ValueError: from extract_prompt_name if format is unexpected
        print(f"Warning: Failed to extract info from {metadata_path}: {e}", file=sys.stderr)
        return None


def process_batch(
    batch_config: BatchConfig,
    output_dir: str,
    s3_client: Optional[S3Client] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Process a single batch directory.

    Args:
        batch_config: Configuration for the batch.
        output_dir: Base output directory.
        s3_client: S3 client for downloading files. Required if not dry_run.
        dry_run: If True, don't download files.
        verbose: If True, print detailed progress.

    Returns:
        Tuple of (success_count, skipped_count, failure_count).
    """
    batch_dir = batch_config.directory
    model_alias = batch_config.model_alias

    if verbose:
        print(f"\nProcessing batch: {batch_dir}")
        print(f"  Model alias: {model_alias}")

    # Discover metadata files
    metadata_files = discover_metadata_files(batch_dir)
    if not metadata_files:
        print(f"Warning: No metadata files found in {batch_dir}", file=sys.stderr)
        return 0, 0, 0

    if verbose:
        print(f"  Found {len(metadata_files)} metadata files")

    # Validate model consistency
    is_valid, model_name = validate_batch_model_consistency(metadata_files, batch_dir)
    if not is_valid:
        print(f"Error: Batch validation failed for {batch_dir}", file=sys.stderr)
        return 0, 0, len(metadata_files)

    if verbose:
        print(f"  Validated: all files use model '{model_name}'")

    success_count = 0
    skipped_count = 0
    failure_count = 0

    for metadata_path in metadata_files:
        info = extract_metadata_info(metadata_path, model_alias)
        if info is None:
            failure_count += 1
            continue

        # Create output directory structure:
        # {output_dir}/{clip_id_with_timestamp}/{model_alias}_{prompt_name}/
        clip_dir = os.path.join(output_dir, info.clip_id_with_timestamp)
        model_prompt_dir = os.path.join(clip_dir, f"{info.model_alias}_{info.prompt_name}")

        if not dry_run:
            os.makedirs(model_prompt_dir, exist_ok=True)

        # Download the results JSON
        # Extract filename from S3 URL
        s3_filename = os.path.basename(info.results_s3_url)
        local_path = os.path.join(model_prompt_dir, s3_filename)

        # Skip if file already exists and has content
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            if verbose:
                print(f"  Skipping (cached): {s3_filename}")
            skipped_count += 1
            continue

        if verbose:
            action = "Would download" if dry_run else "Downloading"
            print(f"  {action}: {s3_filename}")
            print(f"    -> {local_path}")

        if dry_run:
            success_count += 1
            continue

        try:
            s3_client.download_file(info.results_s3_url, local_path)
            success_count += 1
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Error downloading {info.results_s3_url}: {e}", file=sys.stderr)
            # Clean up partial download so reruns don't treat corrupt files as cached
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass  # Best effort cleanup
            failure_count += 1

    return success_count, skipped_count, failure_count


def main() -> None:
    """Entry point for the batch preparation script."""
    args = parse_args()

    # Parse batch configs
    batch_configs = [BatchConfig(directory=b[0], model_alias=b[1]) for b in args.batch]

    # Validate batch directories exist
    for config in batch_configs:
        if not os.path.isdir(config.directory):
            print(f"Error: Batch directory does not exist: {config.directory}", file=sys.stderr)
            sys.exit(1)

    # Check for duplicate aliases
    aliases = [c.model_alias for c in batch_configs]
    if len(aliases) != len(set(aliases)):
        print("Error: Duplicate model aliases provided", file=sys.stderr)
        sys.exit(1)

    print("Preparing batch results for analysis")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Batches: {len(batch_configs)}")
    for config in batch_configs:
        print(f"    - {config.directory} -> {config.model_alias}")

    if args.dry_run:
        print("\n  [DRY RUN - no files will be downloaded]")

    # Load AWS credentials from ~/.aws/.env if available
    load_aws_credentials_from_dotenv(verbose=args.verbose)

    # Initialize S3 client (unless dry run)
    s3_client: Optional[S3Client] = None
    if not args.dry_run:
        try:
            s3_client = S3Client()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Create output directory
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    total_success = 0
    total_skipped = 0
    total_failure = 0

    for config in batch_configs:
        success, skipped, failure = process_batch(
            config,
            args.output_dir,
            s3_client=s3_client,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        total_success += success
        total_skipped += skipped
        total_failure += failure

    print("\nSummary:")
    print(f"  Downloaded: {total_success}")
    print(f"  Skipped (cached): {total_skipped}")
    print(f"  Failed: {total_failure}")

    if total_failure > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
