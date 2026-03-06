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

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class ProjectionOverlayHelper:
    """
    Lightweight visualization helper for drawing projected objects on video frames.

    - Uses a configurable color map that maps object categories to BGR colors
    - Draws dynamic/static boxes and lane polylines
    """

    def __init__(self, color_map: Dict[str, Tuple[int, int, int]] = None):
        # BGR color defaults
        default_colors = {
            # Dynamic objects
            "vehicle": (0, 255, 0),
            "car": (0, 255, 0),
            "automobile": (0, 255, 0),
            "truck": (0, 200, 0),
            "bus": (0, 180, 0),
            "train": (0, 160, 0),
            "pedestrian": (0, 0, 255),
            "person": (0, 0, 255),
            "bicycle": (255, 0, 255),
            "motorcycle": (255, 0, 255),
            "cyclist": (255, 0, 255),
            "rider": (255, 0, 255),
            "unknown": (0, 255, 0),
            # Static roadside
            "traffic_light": (0, 165, 255),
            "traffic_sign": (0, 255, 255),
            "pole": (200, 200, 200),
            "wait_line": (255, 255, 255),
            "crosswalk": (0, 140, 196),
            "road_marking": (128, 128, 255),
            # Lanes
            "lane_line": (255, 255, 0),
            "road_boundary": (255, 0, 0),
        }
        self.color_map = (color_map or {}).copy()
        for k, v in default_colors.items():
            self.color_map.setdefault(k, v)

        self.box_thickness = 2
        self.polyline_thickness = 2

    def get_color(self, category: str) -> Tuple[int, int, int]:
        if not category:
            return self.color_map["unknown"]
        key = str(category).strip().lower()
        return self.color_map.get(key, self.color_map["unknown"])

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        dynamic_boxes: List[Dict[str, Any]],
        static_boxes: List[Dict[str, Any]],
        lane_polylines: List[Dict[str, Any]],
        static_polylines: List[Dict[str, Any]] = None,
        polygon_overlays: List[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Draw all overlays onto a BGR frame.
        dynamic_boxes: list of {object_id, bbox_xywh, category}
        static_boxes: list of {object_id, bbox_xywh, category}
        lane_polylines: list of {object_id, points, category}
        """
        img = frame_bgr

        # Dynamic boxes
        for b in dynamic_boxes or []:
            x, y, w, h = b.get("bbox_xywh", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            color = self.get_color(b.get("category", "unknown"))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, self.box_thickness)

        # Static boxes
        for b in static_boxes or []:
            x, y, w, h = b.get("bbox_xywh", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            color = self.get_color(b.get("category", "unknown"))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, self.box_thickness)

        # Lane polylines
        for p in lane_polylines or []:
            pts = np.asarray(p.get("points_xy", []), dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) >= 2:
                color = self.get_color(p.get("category", "lane_line"))
                cv2.polylines(img, [pts], isClosed=False, color=color, thickness=self.polyline_thickness)
            elif len(pts) == 1:
                color = self.get_color(p.get("category", "lane_line"))
                cv2.circle(img, tuple(pts[0, 0]), 2, color, -1)

        # Static polylines (e.g., wait lines)
        for p in static_polylines or []:
            pts = np.asarray(p.get("points_xy", []), dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) >= 2:
                color = self.get_color(p.get("category", "wait_line"))
                cv2.polylines(img, [pts], isClosed=False, color=color, thickness=self.polyline_thickness)
            elif len(pts) == 1:
                color = self.get_color(p.get("category", "wait_line"))
                cv2.circle(img, tuple(pts[0, 0]), 2, color, -1)

        # Polygon overlays (e.g., crosswalks, road markings)
        for poly in polygon_overlays or []:
            verts = np.asarray(poly.get("vertices_xy", []), dtype=np.int32).reshape(-1, 1, 2)
            if len(verts) >= 3:
                color = self.get_color(poly.get("category", "road_marking"))
                cv2.polylines(img, [verts], isClosed=True, color=color, thickness=self.polyline_thickness)

        return img
