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

"""Frame processing utilities for the hallucination checkers."""

import cv2
import numpy as np

from checks.utils.types import DynParams


def to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert a frame to grayscale.
    """
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def ensure_same_size(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize a frame to a target height and width.
    """
    th, tw = target_hw
    h, w = frame.shape[:2]
    if (h, w) == (th, tw):
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def compute_dynamic_mask(
    prev_gray: np.ndarray | None, curr_gray: np.ndarray, p: DynParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the dynamic mask for a single frame.
    """
    if prev_gray is None:
        return np.zeros_like(curr_gray, dtype=np.uint8), curr_gray
    diff = cv2.absdiff(curr_gray, prev_gray)
    k = max(1, p.blur_ksize | 1)
    if k >= 3:
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    _, mask = cv2.threshold(diff, p.grad_thresh, 255, cv2.THRESH_BINARY)
    mk = max(1, p.morph_k | 1)
    if mk >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, curr_gray


def bbox_iou_int(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """
    Compute the Intersection Over Union (IoU) of two bounding boxes.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def expand_bbox(
    x_min: float, y_min: float, x_max: float, y_max: float, scale: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by a given scale factor.
    """
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = (x_max - x_min) * max(scale, 1.0)
    h = (y_max - y_min) * max(scale, 1.0)
    x0 = int(round(cx - 0.5 * w))
    y0 = int(round(cy - 0.5 * h))
    x1 = int(round(cx + 0.5 * w))
    y1 = int(round(cy + 0.5 * h))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
    return x0, y0, x1, y1
