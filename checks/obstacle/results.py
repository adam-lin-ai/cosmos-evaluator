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
Library for serializing obstacle correspondence results.

This module provides functions for saving and loading obstacle correspondence results to and from JSON files.
"""

import base64
import gzip
import json
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np


def save_results_to_json(
    results: Dict[str, Any],
    clip_id: str,
    output_dir: str,
    matrix_format: str = "sparse",
    output_file_prefix: str | None = None,
) -> str:
    """
    Save processing results to a JSON file.

    Args:
        results: Results dictionary from process_clip
        clip_id: Clip ID for filename
        output_dir: Output directory path
        matrix_format: How to store the score matrix ("sparse", "dense", "compressed", or "summary")

    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON serialization
    json_results = {}

    # Copy basic results (these are JSON-serializable)
    for key, value in results.items():
        if key == "score_matrix":
            continue  # Handle score matrix separately
        elif key == "track_ids":
            # Convert set to sorted list for consistent ordering
            json_results[key] = sorted(value) if value else []
        elif key == "processed_frame_ids":
            json_results[key] = list(value) if value else []
        else:
            json_results[key] = value

    # Handle score matrix based on format
    if results.get("score_matrix") is not None:
        score_matrix = results["score_matrix"]

        if matrix_format == "sparse":
            # Store only non-NaN values with their indices
            sparse_data = []
            for frame_idx in range(score_matrix.shape[0]):
                for track_idx in range(score_matrix.shape[1]):
                    score = score_matrix[frame_idx, track_idx]
                    if not np.isnan(score):
                        sparse_data.append(
                            {"frame_idx": int(frame_idx), "track_idx": int(track_idx), "score": float(score)}
                        )
            json_results["score_matrix"] = {"format": "sparse", "shape": list(score_matrix.shape), "data": sparse_data}

        elif matrix_format == "dense":
            # Store full matrix as nested lists (larger file size)
            json_results["score_matrix"] = {
                "format": "dense",
                "shape": list(score_matrix.shape),
                "data": score_matrix.tolist(),
            }

        elif matrix_format == "compressed":
            # Store as base64-encoded compressed data
            # Convert NaN to a special value for compression
            matrix_copy = score_matrix.copy()
            matrix_copy[np.isnan(matrix_copy)] = -999.0  # Special value for NaN

            # Compress and encode
            compressed_data = gzip.compress(matrix_copy.tobytes())
            encoded_data = base64.b64encode(compressed_data).decode("utf-8")

            json_results["score_matrix"] = {
                "format": "compressed",
                "shape": list(score_matrix.shape),
                "nan_value": -999.0,
                "data": encoded_data,
            }

        elif matrix_format == "summary":
            # Store only summary statistics (smallest file size)
            valid_scores = score_matrix[~np.isnan(score_matrix)]
            json_results["score_matrix"] = {
                "format": "summary",
                "shape": list(score_matrix.shape),
                "statistics": {
                    "valid_count": len(valid_scores),
                    "nan_count": int(np.isnan(score_matrix).sum()),
                    "total_elements": int(score_matrix.size),
                    "mean": float(np.nanmean(score_matrix)),
                    "std": float(np.nanstd(score_matrix)),
                    "min": float(np.nanmin(score_matrix)),
                    "max": float(np.nanmax(score_matrix)),
                },
            }
        else:
            raise ValueError(f"Unknown matrix_format: {matrix_format}")
    else:
        json_results["score_matrix"] = None

    # Add metadata
    json_results["metadata"] = {"matrix_format": matrix_format, "version": "1.0"}

    # Save to file with custom formatting for compact arrays
    output_file = output_path / f"{clip_id}.object.results.json"
    if output_file_prefix:
        output_file = output_path / f"{clip_id}.{output_file_prefix}.object.results.json"

    def format_json_with_compact_arrays(obj: Any, indent: int = 2) -> str:
        """Format JSON with compact arrays for numeric lists."""
        json_str = json.dumps(obj, indent=indent)
        lines = json_str.split("\n")
        result_lines: List[str] = []
        is_num = re.compile(r"[+-]?\d+(?:\.\d+)?$").match
        i = 0
        while i < len(lines):
            stripped = lines[i].rstrip()
            if stripped.endswith("["):
                numbers: List[str] = []
                j = i + 1
                all_numeric = True
                while j < len(lines):
                    inner = lines[j].strip()
                    if inner in ("]", "],"):
                        break
                    value = inner.rstrip(",")
                    if not is_num(value):
                        all_numeric = False
                        break
                    numbers.append(value)
                    j += 1
                if all_numeric and numbers and j < len(lines):
                    trailing = "," if lines[j].strip().endswith(",") else ""
                    result_lines.append(stripped + ",".join(numbers) + "]" + trailing)
                    i = j + 1
                    continue
            result_lines.append(lines[i])
            i += 1
        return "\n".join(result_lines)

    with open(output_file, "w") as f:
        f.write(format_json_with_compact_arrays(json_results))

    return str(output_file)


def load_results_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Results dictionary with reconstructed score matrix
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Reconstruct score matrix if present
    if data.get("score_matrix") is not None:
        matrix_info = data["score_matrix"]
        format_type = matrix_info.get("format", "dense")

        if format_type == "sparse":
            # Reconstruct from sparse data
            shape = matrix_info["shape"]
            score_matrix = np.full(shape, np.nan)
            for item in matrix_info["data"]:
                score_matrix[item["frame_idx"], item["track_idx"]] = item["score"]
            data["score_matrix"] = score_matrix

        elif format_type == "dense":
            # Convert from nested lists
            data["score_matrix"] = np.array(matrix_info["data"])

        elif format_type == "compressed":
            # Decompress from base64
            encoded_data = matrix_info["data"]
            compressed_data = base64.b64decode(encoded_data.encode("utf-8"))
            matrix_bytes = gzip.decompress(compressed_data)

            shape = matrix_info["shape"]
            score_matrix = np.frombuffer(matrix_bytes, dtype=np.float64).reshape(shape)
            writeable_score_matrix = score_matrix.copy()

            # Restore NaN values
            nan_value = matrix_info["nan_value"]
            writeable_score_matrix[np.abs(writeable_score_matrix - nan_value) < 1e-6] = np.nan

            data["score_matrix"] = writeable_score_matrix

        elif format_type == "summary":
            # Cannot reconstruct full matrix from summary
            data["score_matrix"] = None

    return data


def get_object_type_track_idxs(results: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Get the track IDs for each object type.
    """
    object_type_track_idxs = {}
    if "track_ids" not in results or "tracks" not in results:
        return object_type_track_idxs

    track_ids = results["track_ids"]
    track_id_to_track_idx = {track_id: i for i, track_id in enumerate(track_ids)}

    for obj in results["tracks"]:
        obj_type = obj.get("object_type")
        track_id = obj.get("track_id")
        if obj_type is not None and track_id is not None:
            if obj_type not in object_type_track_idxs:
                object_type_track_idxs[obj_type] = []
            object_type_track_idxs[obj_type].append(track_id_to_track_idx[track_id])

    return object_type_track_idxs
