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

import json
import logging
import os
from pathlib import Path
import re
import tarfile
import threading
from typing import Any, ClassVar, Dict, Optional, Union
import zipfile


class MetadataLoader:
    """Utility class for loading video metadata and downloading associated files from S3.

    This class handles:
    - Loading metadata from a JSON file or pre-loaded dictionary
    - Downloading generated videos, control videos, and RDS-HQ datasets from S3
    - Extracting compressed archives (zip, tar.gz, etc.)
    - Caching downloads and extractions to avoid redundant operations

    The cache directory is organized by clip_id under ~/.cache/cosmos_evaluator/s3/.

    Example:
        >>> loader = MetadataLoader("/path/to/metadata.json")
        >>> video_path = loader.get_video_path()
        >>> rds_hq_path = loader.get_gt_dataset_path()
    """

    # Class-level locks for thread-safe extraction
    # Key: archive_path, Value: Lock for that specific archive
    _extraction_locks: ClassVar[Dict[str, threading.Lock]] = {}
    _extraction_locks_lock: ClassVar[threading.Lock] = threading.Lock()  # Protects _extraction_locks dict

    @classmethod
    def _get_extraction_lock(cls, archive_path: str) -> threading.Lock:
        """Get or create a lock for a specific archive path."""
        with cls._extraction_locks_lock:
            if archive_path not in cls._extraction_locks:
                cls._extraction_locks[archive_path] = threading.Lock()
            return cls._extraction_locks[archive_path]

    @classmethod
    def _remove_extraction_lock(cls, archive_path: str) -> None:
        """Remove a lock from the dictionary after extraction is complete.

        This should be called while still holding the lock to prevent race
        conditions. Threads already waiting on the lock will still complete
        normally since they hold a reference to the lock object.
        """
        with cls._extraction_locks_lock:
            cls._extraction_locks.pop(archive_path, None)

    def __init__(self, metadata_source: Union[str, Dict[str, Any]]) -> None:
        """Initialize the MetadataLoader

        Args:
            metadata_source: Either a path to the metadata JSON file (str) or
                a pre-loaded metadata dictionary (Dict)

        Raises:
            RuntimeError: If the metadata JSON file is invalid
            RuntimeError: If the download fails
        """
        if isinstance(metadata_source, dict):
            # Accept pre-loaded metadata dict (e.g., from DB query)
            self.metadata_json_path: Optional[str] = None
            self.data: Dict[str, Any] = metadata_source
        else:
            # Load from file path (existing behavior)
            self.metadata_json_path = metadata_source
            with open(metadata_source, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        # Expose common top-level sections
        self.input_video_info: Dict[str, Any] = self.data.get("input_video_info", {})
        self.generated_video_info: Dict[str, Any] = self.data.get("generated_video_info", {})
        self.metadata: Dict[str, Any] = self.data.get("metadata", {})
        self.cosmos_evaluator_result: Dict[str, Any] = self.data.get("cosmos_evaluator_result", {})

        # Determine clip_id for cache directory
        clip_id = (
            self.get_clip_id(with_timestamp_suffix=True)
            or self.get_clip_id(with_timestamp_suffix=False)
            or "clip_id_unknown"
        )

        # Cache directories and downloads (hardcoded path with clip_id)
        self.cache_dir = Path.home() / ".cache" / "cosmos_evaluator" / "s3" / clip_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_cache: Dict[str, str] = {}
        self._extract_cache: Dict[str, str] = {}

        # Initialize S3 client using explicit credentials from environment
        # to avoid "Unable to locate credentials" errors in sandboxed runs.
        try:
            import boto3
        except Exception as e:
            raise RuntimeError("boto3 is required to download from S3 in cloud mode") from e

        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN")  # optional
        self.region_name = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-west-2"

        if not (self.aws_access_key_id and self.aws_secret_access_key):
            raise RuntimeError(
                "Missing AWS credentials. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set (e.g., in ~/.aws/.env)."
            )

        client_kwargs = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.region_name,
        }
        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token

        self.s3_client = boto3.client("s3", **client_kwargs)

        # Reduce third-party log verbosity in DEBUG mode to avoid flooding logs
        try:
            if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                for name in ("boto3", "botocore", "s3transfer", "urllib3"):
                    logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass

    # --- Internal helpers -------------------------------------------------
    def _download_s3_once(self, s3_url: str) -> str:
        """Download a file from S3 once and cache the result

        Args:
            s3_url: The S3 URL to download from

        Returns:
            The local path to the downloaded file

        Raises:
            ValueError: If the S3 URL is invalid
            RuntimeError: If the download fails
        """
        if s3_url in self._download_cache:
            return self._download_cache[s3_url]

        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {s3_url}")

        path = s3_url[len("s3://") :]
        if "/" not in path:
            raise ValueError(f"Invalid S3 URL (missing key): {s3_url}")
        bucket, key = path.split("/", 1)

        local_path = self.cache_dir / key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)

        # Check if file already exists from a previous session
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("Reusing cached file: %s", local_path)
            self._download_cache[s3_url] = str(local_path)
            return str(local_path)

        logger.info("Downloading from S3: %s", s3_url)
        try:
            self.s3_client.download_file(bucket, key, str(local_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {e}") from e
        logger.info("Downloaded to: %s", local_path)

        self._download_cache[s3_url] = str(local_path)
        return str(local_path)

    def _extract_archive_once(self, archive_path: str) -> str:
        """Extract an archive once and cache the result

        Args:
            archive_path: The path to the archive file

        Returns:
            The local path to the extracted directory

        Raises:
            ValueError: If the archive type is unsupported
            ValueError: If the archive contains unsafe paths
        """
        p = Path(archive_path)
        if archive_path in self._extract_cache:
            return self._extract_cache[archive_path]

        # Decide extraction directory name
        extract_dir = p.with_suffix("")
        # Handle .tar.gz/.tgz double suffix
        if p.suffixes[-2:] in [[".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"], [".tar", ".zst"]]:
            extract_dir = Path(str(p).rsplit(".tar", 1)[0])
        extract_dir = Path(str(extract_dir) + "_extracted")

        # Use a marker file to indicate extraction is complete
        # This prevents race conditions when multiple threads try to extract the same archive
        extraction_complete_marker = extract_dir / ".extraction_complete"

        # Acquire lock for this specific archive to prevent concurrent extraction
        extraction_lock = self._get_extraction_lock(archive_path)

        with extraction_lock:
            # Double-check: another thread may have completed extraction while we waited
            if extraction_complete_marker.exists():
                # Clean up lock since extraction is complete
                self._remove_extraction_lock(archive_path)
                self._extract_cache[archive_path] = str(extract_dir)
                return str(extract_dir)

            # Backwards compatibility: if directory exists and has content but no marker,
            # create the marker and return (for pre-existing cache directories)
            if extract_dir.exists() and any(extract_dir.iterdir()):
                extraction_complete_marker.touch()
                # Clean up lock since extraction is complete
                self._remove_extraction_lock(archive_path)
                self._extract_cache[archive_path] = str(extract_dir)
                return str(extract_dir)

            # Create extraction directory
            extract_dir.mkdir(parents=True, exist_ok=True)

            def _is_within_directory(base: Path, target: Path) -> bool:
                try:
                    base_resolved = base.resolve()
                    target_resolved = target.resolve()
                except FileNotFoundError:
                    # Parent may not exist yet; resolve parent and join name
                    base_resolved = base.resolve()
                    target_resolved = (base / target.name).resolve()
                return str(target_resolved).startswith(str(base_resolved) + os.sep) or target_resolved == base_resolved

            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p, "r") as zf:
                    for info in zf.infolist():
                        member_path = extract_dir / info.filename
                        if not _is_within_directory(extract_dir, member_path):
                            raise ValueError(f"Unsafe path in ZIP archive: {info.filename}")
                        if (hasattr(info, "is_dir") and info.is_dir()) or info.filename.endswith("/"):
                            (extract_dir / info.filename).mkdir(parents=True, exist_ok=True)
                            continue
                        # Ensure parent directories exist
                        (extract_dir / info.filename).parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info, "r") as src, open(extract_dir / info.filename, "wb") as dst:
                            dst.write(src.read())
            elif tarfile.is_tarfile(p):
                with tarfile.open(p, "r:*") as tf:
                    # Use built-in path traversal protection
                    tf.extractall(path=extract_dir, filter="data")
            else:
                raise ValueError(f"Unsupported archive type for {archive_path}")

            # Mark extraction as complete so other threads/processes know it's safe to use
            extraction_complete_marker.touch()

            # Clean up lock from dictionary to prevent unbounded memory growth
            # Done while still holding lock to prevent race conditions
            self._remove_extraction_lock(archive_path)

            self._extract_cache[archive_path] = str(extract_dir)
            return str(extract_dir)

    def _parse_clip_id_with_timestamp_from_control_video_url(self) -> Optional[str]:
        """Parse clip_id with timestamp from control video URL filename.

        Only applies when control_video_type is "world_model".
        Expected filename: {clip_id}_{timestamp}_{chunk_idx}.mp4
        Returns "{clip_id}_{timestamp}" if found, else None.
        """
        ctrl = self.input_video_info.get("control_video_info", {}) or {}
        if ctrl.get("control_video_type") != "world_model":
            return None
        ctrl_url = ctrl.get("control_video_s3_url") or ""
        if isinstance(ctrl_url, str) and ctrl_url:
            fname = ctrl_url.split("/")[-1]
            clip_id = self.input_video_info.get("metadata", {}).get("clip_id")

            if clip_id is not None:
                # If clip_id is present, extract "<clip_id>_<timestamp>" from the filename by anchoring regex to clip_id
                # Format: "...<clip_id>_<timestamp>_<chunk_idx>.mp4" => want "<clip_id>_<timestamp>"
                pattern = re.compile(rf".*({re.escape(clip_id)}_\d+)_\d+\.mp4$")
                m = pattern.match(fname)
                if m:
                    return m.group(1)
            else:
                m = re.match(r"^(?P<clip_id_with_ts>.+_\d+)_\d+\.mp4$", fname)
                if m:
                    return m.group("clip_id_with_ts")
        return None

    # --- Public API -------------------------------------------------------
    def get_sections(self) -> Dict[str, Any]:
        """Return a dict of all top-level sections.

        Returns:
            A dict of all top-level sections

        Raises:
            ValueError: If the metadata JSON file is invalid
        """
        return dict(self.data)

    def get_cache_dir(self) -> str:
        """Get the local cache directory path for this video's downloads.

        The cache directory is organized as ~/.cache/cosmos_evaluator/s3/{clip_id}/.
        All downloaded files and extracted archives are stored here.

        Returns:
            Path to the cache directory (as a string).
        """
        return str(self.cache_dir)

    def get_clip_id(self, with_timestamp_suffix: bool = False) -> Optional[str]:
        """Get the clip_id from metadata, optionally with timestamp suffix.

        The clip_id uniquely identifies the source driving clip. When
        with_timestamp_suffix is True, returns "{clip_id}_{timestamp}" which
        provides more precise identification for caching purposes.

        Args:
            with_timestamp_suffix: When True, attempts to return
                "{clip_id}_{timestamp}" by checking cosmos_evaluator_result or
                parsing from control video URL. Defaults to False.

        Returns:
            The clip_id string, or None if not available in metadata.
        """
        if with_timestamp_suffix:
            # First try to get from cosmos_evaluator_result if available
            if clip_id_with_ts := self.cosmos_evaluator_result.get("clip_id_with_timestamp"):
                return clip_id_with_ts
            # Second try to extract it from WM control video name
            elif clip_id_with_ts := self._parse_clip_id_with_timestamp_from_control_video_url():
                return clip_id_with_ts
        elif clip_id := self.input_video_info.get("metadata", {}).get("clip_id"):
            return clip_id
        return None

    def get_video_path(self) -> str:
        """Get the local path to the generated video, downloading if necessary.

        Downloads the video from S3 to the local cache directory. Subsequent
        calls return the cached path without re-downloading.

        Returns:
            Local path to the downloaded video file.

        Raises:
            ValueError: If output_s3_url is missing from generated_video_info.
            RuntimeError: If the S3 download fails.
        """
        s3_url = self.generated_video_info.get("output_s3_url")
        if not s3_url:
            raise ValueError("generated_video_info.output_s3_url is missing")
        return self._download_s3_once(s3_url)

    def get_wm_video_path(self, required_version: Optional[str] = None) -> Optional[str]:
        """Get the local path to the world model control video, downloading if necessary.

        Only returns a path if control_video_type is "world_model". Optionally
        filters by render version (e.g., "v1", "v2").

        Args:
            required_version: If specified, only return the video if its
                render version matches (case-insensitive). Defaults to None
                (accept any version).

        Returns:
            Local path to the downloaded control video, or None if:
            - control_video_type is not "world_model"
            - required_version doesn't match actual version
            - control_video_s3_url is missing

        Raises:
            RuntimeError: If the S3 download fails.
        """
        ctrl = self.input_video_info.get("control_video_info", {})
        if ctrl.get("control_video_type") == "world_model":
            # Check render version if required
            if required_version is not None:
                actual_version = ctrl.get("control_video_render_version", "v1")
                if actual_version.lower() != required_version.lower():
                    return None
            s3_url = ctrl.get("control_video_s3_url")
            if s3_url:
                return self._download_s3_once(s3_url)
        return None

    def get_gt_dataset_path(self) -> Optional[str]:
        """Get path to extracted RDS-HQ ground truth dataset.

        Downloads and extracts the RDS-HQ archive from S3 if metadata indicates
        availability (rdshq_archive_created=True).

        Returns:
            Path to extracted RDS-HQ directory, or None if not available.
        """
        logger = logging.getLogger(__name__)

        rds = self.input_video_info.get("rds_hq_data") or {}
        if rds.get("rdshq_archive_created") is True:
            s3_url = rds.get("rdshq_s3_url")
            if not s3_url:
                return None
            archive_path = self._download_s3_once(s3_url)
            return self._extract_archive_once(archive_path)

        # Log why we're returning None for debugging
        archive_created = rds.get("rdshq_archive_created")
        if archive_created is not None and archive_created is not True:
            logger.debug(
                "RDS-HQ not available: rdshq_archive_created=%r (expected True)",
                archive_created,
            )

        return None

    def get_preset(self) -> Any:
        """Get the preset from the metadata

        Returns:
            The preset

        Raises:
            ValueError: If the preset is invalid
        """
        preset_obj = self.metadata.get("preset")
        if isinstance(preset_obj, dict):
            return preset_obj
        raise ValueError("metadata.preset must be a JSON object with required fields")
