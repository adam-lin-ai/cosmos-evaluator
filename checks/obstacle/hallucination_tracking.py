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

from typing import Any, Dict, Optional

import numpy as np

# Monotonic counter for hallucination track IDs (module-level)
_TRACK_ID_COUNTER = 0


def track_hallucinations(
    results: Dict[str, Any],
    start_indices: Dict[str, int],
    road_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Associate newly appended hallucination detections from the current frame with
    existing tracks using a simple bounding-box overlap heuristic, or start new tracks
    when no overlap is found.

    What this function updates
    - Mutates `results["hallucination_tracks"]` in-place by either appending a new
      track or extending the `detections` list of an existing track. If the tracks
      array is missing, it will be created.

    Inputs
    - results:
        A dict that must contain:
          - `hallucination_detections`: dict[str -> list[dict]] per high-level class
            (e.g., "vehicle", "pedestrian"). Each detection is expected to have at
            least `bbox_xywh: [x, y, w, h]` and `frame_idx`.
          - `hallucination_tracks`: list[track] (optional). Each track is a dict
            `{ "id": int, "class": str, "detections": list[int], "relevancy": int }`.
            The `detections` list stores integer indices into
            `results["hallucination_detections"][track["class"]]`.
    - start_indices:
        dict[class_name -> int] giving, for each class, the starting index (i.e., the
        length of the class list before appending current-frame detections). All
        detections added in the current frame for a class are assumed to be appended
        contiguously; this function will iterate from `start_indices[class]` to the
        current end of the list.
    - road_mask:
        Optional boolean numpy array (HxW) indicating road pixels. If provided,
        used to update `relevancy` for tracks whose detections touch road regions.
        If None, relevancy updates are skipped.

    Association logic
    - For each new detection of a class, compare its bbox against the last up to two
      bboxes of every existing track of the same class (looked up by the stored
      indices in that track). If there is any strict rectangle overlap (intersection
      area > 0), the detection is assigned to that track; otherwise a new track is
      created with `id = len(results["hallucination_tracks"]) + 1` and its `detections`
      initialized with the new detection index.

    Relevancy update
    - For each assignment (new or existing), increment the track's `relevancy` by 1
      if the detection bbox region contains any pixel labeled as "road" in the
      provided `road_mask`.

    Assumptions & limitations
    - The `hallucination_detections[class]` lists are append-only, and this function
      is called exactly once per processed frame after appending the current frame's
      detections.
    - Overlap threshold is binary (> 0 area). There is no IoU gating, motion model,
      track lifecycle management (age/timeouts), or cross-class linking.
    - Indices stored in tracks remain valid for the lifetime of the results object.

    Returns
    - None. All updates are applied directly to `results`.
    """
    if not start_indices:
        return

    # Prepare tracks container
    tracks = results.get("hallucination_tracks")
    if tracks is None:
        tracks = []
        results["hallucination_tracks"] = tracks

    # Ensure module-level counter starts above any pre-existing track IDs
    global _TRACK_ID_COUNTER
    if _TRACK_ID_COUNTER <= 0 and tracks:
        try:
            _TRACK_ID_COUNTER = max(int(t.get("id", 0)) for t in tracks)
        except Exception:
            _TRACK_ID_COUNTER = 0

    def overlaps(b1, b2) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        r1 = (x1, y1, x1 + max(w1, 1) - 1, y1 + max(h1, 1) - 1)
        r2 = (x2, y2, x2 + max(w2, 1) - 1, y2 + max(h2, 1) - 1)
        ix0 = max(r1[0], r2[0])
        iy0 = max(r1[1], r2[1])
        ix1 = min(r1[2], r2[2])
        iy1 = min(r1[3], r2[3])
        return (ix1 >= ix0) and (iy1 >= iy0)

    def get_detection_bucket(cls: str):
        return (results.get("hallucination_detections", {}) or {}).get(cls, [])

    def get_detection_bbox(cls: str, idx: int):
        try:
            det = get_detection_bucket(cls)[idx]
            return det.get("bbox_xywh", None)
        except Exception:
            return None

    # For each class with new detections in this frame
    for cls_name, base_idx in (start_indices or {}).items():
        bucket = get_detection_bucket(cls_name)
        if not isinstance(bucket, list) or base_idx >= len(bucket):
            continue

        for det_idx in range(int(base_idx), len(bucket)):
            det = bucket[det_idx]
            det_bbox = det.get("bbox_xywh")
            if det_bbox is None:
                continue

            matched_track = None
            for tr in tracks:
                if tr.get("class") != cls_name:
                    continue
                det_indices = tr.get("detections") or []
                for prev_idx in det_indices[-2:][::-1]:
                    prev_bbox = get_detection_bbox(cls_name, int(prev_idx))
                    if prev_bbox is None:
                        continue
                    if overlaps(det_bbox, prev_bbox):
                        matched_track = tr
                        break
                if matched_track is not None:
                    break

            if matched_track is None:
                _TRACK_ID_COUNTER += 1
                new_id = int(_TRACK_ID_COUNTER)
                track = {
                    "id": new_id,
                    "class": str(cls_name),
                    "detections": [int(det_idx)],
                    "relevancy": 0,
                }
                tracks.append(track)
                target_track = track
            else:
                matched_track.setdefault("detections", []).append(int(det_idx))
                target_track = matched_track

            # Update relevancy if bbox touches road
            try:
                if road_mask is not None:
                    x, y, w, h = [int(v) for v in det_bbox]
                    x1 = max(0, min(x + max(w, 1) - 1, road_mask.shape[1] - 1))
                    y1 = max(0, min(y + max(h, 1) - 1, road_mask.shape[0] - 1))
                    x0 = max(0, min(x, road_mask.shape[1] - 1))
                    y0 = max(0, min(y, road_mask.shape[0] - 1))
                    if x0 <= x1 and y0 <= y1:
                        roi = road_mask[y0 : y1 + 1, x0 : x1 + 1]
                        if roi.any():
                            target_track["relevancy"] = int(target_track.get("relevancy", 0)) + 1
            except Exception:
                pass
