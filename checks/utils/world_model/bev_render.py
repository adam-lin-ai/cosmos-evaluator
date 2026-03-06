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

from dataclasses import dataclass

import cv2
import numpy as np

from checks.utils.rds_data_loader import RdsDataLoader

OVERLAY_BG_BGR = (0, 0, 0)
OVERLAY_TEXT_BGR = (255, 255, 255)

# Centralized annotation style/color registry
# Colors are specified in BGR to match OpenCV drawing; styles describe symbol geometry
ANNOTATION_STYLE_REGISTRY = {
    "ego_vehicle": {"color": "green", "style": "rectangle"},
    "vehicle": {"color": "cyan", "style": "rectangle"},
    "two_wheeler": {"color": "magenta", "style": "rectangle"},
    "pedestrian": {"color": "blue", "style": "circle"},
    "road_boundary": {"color": "red", "style": "line"},
    "lane_line": {"color": "white", "style": "line"},
    "wait_line": {"color": "white_bright", "style": "line"},
    "crosswalk": {"color": "white_bright", "style": "polygon_striped"},
    "traffic_sign": {"color": "yellow", "style": "rectangle"},
    "traffic_light": {"color": "gray", "style": "rectangle_with_circles"},
}


def color_name_to_bgr(name: str | None) -> list[int]:
    """Convert color name to BGR list.

    Args:
        name: Color name to convert.

    Returns:
        List of three BGR values [B, G, R].
    """
    n = (name or "").strip().lower()
    # Palette tuned to previous visuals
    if n == "green":
        return [0, 185, 118]
    if n == "cyan":
        return [255, 200, 0]
    if n == "magenta":
        return [255, 0, 255]
    if n == "red":
        return [0, 0, 255]
    if n == "blue":
        return [255, 0, 0]
    if n == "yellow":
        return [0, 255, 255]
    if n == "white_bright":
        return [255, 255, 255]
    if n == "white":
        return [200, 200, 200]
    if n == "gray":
        return [50, 50, 50]
    return [200, 200, 200]


@dataclass
class BEVConfig:
    """Configuration for BEV rendering.

    Args:
        x_range: List of x-axis range in meters [min, max].
        y_range: List of y-axis range in meters [min, max].
        resolution: Resolution in meters per pixel.
        height: Height in pixels.
        width: Width in pixels.
    """

    x_range: list[float]
    y_range: list[float]
    resolution: float
    height: int
    width: int

    @classmethod
    def define_render_range(cls, resolution: float = 0.1) -> "BEVConfig":
        """Define the render range for BEV rendering.

        Args:
            resolution: Resolution in meters per pixel.

        Returns:
            BEV configuration.

        Raises:
            ValueError: If resolution is not positive.
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")

        x_range = [-36, 92]
        y_range = [-12.8, 12.8]
        height = int((x_range[1] - x_range[0]) / resolution)
        width = int((y_range[1] - y_range[0]) / resolution)
        return cls(x_range=x_range, y_range=y_range, resolution=resolution, height=height, width=width)


class ObjectTracker:
    """Object tracker for BEV rendering.

    Args:
        persistence_frames: Number of frames to persist an object after it is last seen.
        fade_frames: Number of frames to fade an object out after it is last seen.
    """

    def __init__(self, persistence_frames: int = 3, fade_frames: int = 1) -> None:
        self.persistence_frames = persistence_frames
        self.fade_frames = fade_frames
        self.object_history: dict[str, dict] = {}
        self.current_frame = 0

    def update(self, frame_idx: int, all_object_info: dict) -> None:
        """Update object tracking for the current frame.

        Args:
            frame_idx: Current frame index.
            all_object_info: Dictionary mapping tracking IDs to object information.
        """
        self.current_frame = frame_idx

        for tracking_id, object_info in all_object_info.items():
            self.object_history[tracking_id] = {
                "info": object_info,
                "last_seen": frame_idx,
                "fade_alpha": 1.0,
            }

        objects_to_remove = []
        for tracking_id, tracked_obj in self.object_history.items():
            if tracking_id not in all_object_info:
                frames_since_seen = frame_idx - tracked_obj["last_seen"]

                if frames_since_seen <= self.persistence_frames:
                    if frames_since_seen > self.persistence_frames - self.fade_frames:
                        fade_progress = (frames_since_seen - (self.persistence_frames - self.fade_frames)) / max(
                            self.fade_frames, 1
                        )
                        tracked_obj["fade_alpha"] = max(0.0, 1.0 - fade_progress)
                    else:
                        tracked_obj["fade_alpha"] = 1.0
                else:
                    objects_to_remove.append(tracking_id)

        for tracking_id in objects_to_remove:
            del self.object_history[tracking_id]

    def get_smoothed_objects(self) -> dict:
        """Get all tracked objects with non-zero alpha values.

        Returns:
            Dictionary mapping tracking IDs to tracked object information with fade alpha.
        """
        return {tid: obj for tid, obj in self.object_history.items() if obj["fade_alpha"] > 0}


def grounding_pose(
    input_pose: np.ndarray, reference_height: float | None = None, convention: str = "flu"
) -> np.ndarray:
    """Grounds a pose to the reference frame.

    Args:
        input_pose: Input pose to ground (4x4 matrix).
        reference_height: Reference height in meters, or None to use the input pose height.
        convention: Convention to use for grounding ("flu" or "opencv").

    Returns:
        Grounded pose (4x4 matrix).
    """
    if input_pose.shape != (4, 4):
        raise ValueError(f"input_pose must be a 4x4 matrix, got shape {input_pose.shape}")

    if convention == "flu":
        forward_dir = input_pose[:3, 0]
    elif convention == "opencv":
        forward_dir = input_pose[:3, 2]
    else:
        raise ValueError(f"Invalid convention: {convention}")

    forward_dir_x, forward_dir_y = forward_dir[0], forward_dir[1]
    forward_dir_grounded = np.array([forward_dir_x, forward_dir_y, 0])
    forward_dir_grounded = forward_dir_grounded / np.linalg.norm(forward_dir_grounded)

    up_dir_grounded = np.array([0, 0, 1])

    left_dir_grounded = np.cross(up_dir_grounded, forward_dir_grounded)
    left_dir_grounded = left_dir_grounded / np.linalg.norm(left_dir_grounded)

    forward_dir_grounded = np.cross(left_dir_grounded, up_dir_grounded)
    forward_dir_grounded = forward_dir_grounded / np.linalg.norm(forward_dir_grounded)

    grounded_pose = np.eye(4)
    if convention == "flu":
        grounded_pose[:3, 0] = forward_dir_grounded
        grounded_pose[:3, 1] = left_dir_grounded
        grounded_pose[:3, 2] = up_dir_grounded
    elif convention == "opencv":
        grounded_pose[:3, 0] = -left_dir_grounded
        grounded_pose[:3, 1] = -up_dir_grounded
        grounded_pose[:3, 2] = forward_dir_grounded
    else:
        raise ValueError(f"Invalid convention: {convention}")

    grounded_pose[:3, 3] = input_pose[:3, 3]
    if reference_height is not None:
        grounded_pose[2, 3] = reference_height

    return grounded_pose


def world_to_reference_coordinates(points_world: np.ndarray, world_to_reference: np.ndarray) -> np.ndarray:
    """Convert world coordinates to reference coordinates.

    Args:
        points_world: Points in world coordinates.
        world_to_reference: World to reference transformation matrix.

    Returns:
        Points in reference coordinates.
    """
    points_homogeneous = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_reference_homogeneous = (world_to_reference @ points_homogeneous.T).T
    return points_reference_homogeneous[:, :3]


def reference_to_bev_pixels(points_reference: np.ndarray, bev_config: BEVConfig) -> np.ndarray:
    """Convert reference coordinates to BEV pixels.

    Args:
        points_reference: Points in reference coordinates.
        bev_config: BEV configuration.

    Returns:
        Points in BEV pixels.
    """
    x_coords = points_reference[:, 0]
    y_coords = points_reference[:, 1]

    pixel_per_meter_w = bev_config.width / (bev_config.y_range[1] - bev_config.y_range[0])
    pixel_per_meter_h = bev_config.height / (bev_config.x_range[1] - bev_config.x_range[0])

    front_pixel = (x_coords - bev_config.x_range[0]) * pixel_per_meter_h
    left_pixel = (y_coords - bev_config.y_range[0]) * pixel_per_meter_w

    u_coords = bev_config.width - left_pixel
    v_coords = bev_config.height - front_pixel

    return np.column_stack([u_coords, v_coords])


def filter_points_in_bev_range(points_reference: np.ndarray, bev_config: BEVConfig) -> np.ndarray:
    """Filter points in BEV range.

    Args:
        points_reference: Points in reference coordinates.
        bev_config: BEV configuration.

    Returns:
        Points in BEV range.
    """
    if points_reference.shape[1] < 2:
        raise ValueError(f"points_reference must have at least 2 columns, got shape {points_reference.shape}")

    x_coords = points_reference[:, 0]
    y_coords = points_reference[:, 1]

    in_x_range = (x_coords >= bev_config.x_range[0]) & (x_coords <= bev_config.x_range[1])
    in_y_range = (y_coords >= bev_config.y_range[0]) & (y_coords <= bev_config.y_range[1])

    return in_x_range & in_y_range


def draw_vehicle(
    bev_image: np.ndarray,
    center_reference: np.ndarray,
    lwh: np.ndarray,
    color: tuple[int, int, int] | list[int],
    bev_config: BEVConfig,
    alpha: float = 1.0,
    draw_center_cross: bool = False,
    yaw_radians: float | None = None,
) -> None:
    """Draw a vehicle in BEV image.

    Args:
        bev_image: BEV image.
        center_reference: Center of the vehicle in reference coordinates.
        lwh: Length, width, height of the vehicle (in meters).
        color: Color of the vehicle (BGR tuple).
        bev_config: BEV configuration.
        alpha: Alpha value for blending.
        draw_center_cross: Whether to draw a center cross.
        yaw_radians: Yaw angle in radians.
    """
    if bev_image.shape[:2] != (bev_config.height, bev_config.width):
        raise ValueError(
            f"bev_image must have shape {bev_config.height}x{bev_config.width}, got shape {bev_image.shape[:2]}"
        )

    half_length_m = float(lwh[0]) / 2.0
    half_width_m = float(lwh[1]) / 2.0

    if yaw_radians is None:
        center_pixels = reference_to_bev_pixels(center_reference.reshape(1, 3), bev_config)[0]
        length_pixels = lwh[0] / bev_config.resolution
        width_pixels = lwh[1] / bev_config.resolution
        half_length_px = length_pixels / 2
        half_width_px = width_pixels / 2
        corners = np.array(
            [
                [center_pixels[0] - half_width_px, center_pixels[1] - half_length_px],
                [center_pixels[0] + half_width_px, center_pixels[1] - half_length_px],
                [center_pixels[0] + half_width_px, center_pixels[1] + half_length_px],
                [center_pixels[0] - half_width_px, center_pixels[1] + half_length_px],
            ],
            dtype=np.int32,
        )
    else:
        c = float(np.cos(yaw_radians))
        s = float(np.sin(yaw_radians))
        R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
        local_offsets = np.array(
            [
                [-half_length_m, -half_width_m],
                [half_length_m, -half_width_m],
                [half_length_m, half_width_m],
                [-half_length_m, half_width_m],
            ],
            dtype=np.float32,
        )
        rotated_offsets = (R2 @ local_offsets.T).T
        center_xy = np.array([center_reference[0], center_reference[1]], dtype=np.float32)
        corners_ref_xy = rotated_offsets + center_xy
        corners_ref = np.column_stack(
            [
                corners_ref_xy,
                np.full((4, 1), center_reference[2] if len(center_reference) > 2 else 0.0, dtype=np.float32),
            ]
        )
        corners_px = reference_to_bev_pixels(corners_ref, bev_config)
        corners = corners_px.astype(np.int32)

    if alpha >= 1.0:
        cv2.fillPoly(bev_image, [corners], color)
    else:
        mask = np.zeros(bev_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners], 255)
        temp_image = np.zeros_like(bev_image)
        cv2.fillPoly(temp_image, [corners], color)
        blended = cv2.addWeighted(bev_image, 1 - alpha, temp_image, alpha, 0)
        bev_image[mask > 0] = blended[mask > 0]

    if draw_center_cross:
        center_px = reference_to_bev_pixels(center_reference.reshape(1, 3), bev_config)[0]
        center_point = tuple(center_px.astype(np.int32))
        cross_size = 5
        cv2.line(
            bev_image,
            (center_point[0] - cross_size, center_point[1]),
            (center_point[0] + cross_size, center_point[1]),
            [255, 255, 255],
            3,
        )
        cv2.line(
            bev_image,
            (center_point[0], center_point[1] - cross_size),
            (center_point[0], center_point[1] + cross_size),
            [255, 255, 255],
            3,
        )


def draw_pedestrian_circle(
    bev_image: np.ndarray,
    object_center_reference: np.ndarray,
    color: tuple[int, int, int] | list[int],
    bev_config: BEVConfig,
    alpha: float = 1.0,
) -> None:
    """Draw a pedestrian as a circle in BEV image.

    Args:
        bev_image: BEV image.
        object_center_reference: Center of the pedestrian in reference coordinates.
        color: Color of the circle (BGR tuple).
        bev_config: BEV configuration.
        alpha: Alpha value for blending.
    """
    center_pixels = reference_to_bev_pixels(object_center_reference.reshape(1, 3), bev_config)[0]
    radius_pixels = int(0.5 / bev_config.resolution)
    center_point = tuple(center_pixels.astype(np.int32))

    if alpha >= 1.0:
        cv2.circle(bev_image, center_point, radius_pixels, color, -1)
    else:
        mask = np.zeros(bev_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center_point, radius_pixels, 255, -1)
        temp_image = np.zeros_like(bev_image)
        cv2.circle(temp_image, center_point, radius_pixels, color, -1)
        blended = cv2.addWeighted(bev_image, 1 - alpha, temp_image, alpha, 0)
        bev_image[mask > 0] = blended[mask > 0]


def draw_line_segments_bev(
    bev_image: np.ndarray,
    line_segments_world: np.ndarray,
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    color: tuple[int, int, int] | list[int],
    thickness: int,
) -> None:
    """Draw line segments in BEV image.

    Args:
        bev_image: BEV image.
        line_segments_world: Line segments in world coordinates.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        color: Color of the line segments (BGR tuple).
        thickness: Thickness of the line segments.
    """
    if line_segments_world is None or len(line_segments_world) == 0:
        raise ValueError(f"line_segments_world must be a non-empty array, got shape {line_segments_world.shape}")

    if world_to_reference.shape != (4, 4):
        raise ValueError(f"world_to_reference must be a 4x4 matrix, got shape {world_to_reference.shape}")

    if bev_config.width <= 0 or bev_config.height <= 0:
        raise ValueError(
            f"bev_config must have positive width and height, got width {bev_config.width} and height {bev_config.height}"
        )

    points_world = line_segments_world.reshape(-1, 3)
    if points_world.shape[1] != 3:
        raise ValueError(f"points_world must have 3 columns, got shape {points_world.shape}")

    points_reference = world_to_reference_coordinates(points_world, world_to_reference)
    pixels = reference_to_bev_pixels(points_reference, bev_config)

    x_coords = pixels[:, 0].reshape(-1, 2)
    y_coords = pixels[:, 1].reshape(-1, 2)

    start_point_valid = (
        (x_coords[:, 0] >= 0)
        & (x_coords[:, 0] < bev_config.width)
        & (y_coords[:, 0] >= 0)
        & (y_coords[:, 0] < bev_config.height)
    )
    end_point_valid = (
        (x_coords[:, 1] >= 0)
        & (x_coords[:, 1] < bev_config.width)
        & (y_coords[:, 1] >= 0)
        & (y_coords[:, 1] < bev_config.height)
    )
    both_valid = start_point_valid & end_point_valid

    both_valid_x = np.rint(x_coords[both_valid]).astype(np.int32)
    both_valid_y = np.rint(y_coords[both_valid]).astype(np.int32)
    for x, y in zip(both_valid_x, both_valid_y):
        cv2.line(bev_image, (x[0], y[0]), (x[1], y[1]), color, thickness, lineType=cv2.LINE_AA)

    either_valid = np.logical_xor(start_point_valid, end_point_valid)
    either_valid_x = np.rint(x_coords[either_valid]).astype(np.int32)
    either_valid_y = np.rint(y_coords[either_valid]).astype(np.int32)
    for x, y in zip(either_valid_x, either_valid_y):
        cv2.line(bev_image, (x[0], y[0]), (x[1], y[1]), color, thickness, lineType=cv2.LINE_AA)


def draw_polylines_bev(
    bev_image: np.ndarray,
    polylines_world: list[list[float]],
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    color: tuple[int, int, int] | list[int],
    thickness: int,
) -> None:
    """Draw polylines in BEV image.

    Args:
        bev_image: BEV image.
        polylines_world: Polylines in world coordinates.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        color: Color of the polylines (BGR tuple).
        thickness: Thickness of the polylines.
    """
    if polylines_world is None or len(polylines_world) == 0:
        return
    for vertices_world in polylines_world:
        if vertices_world is None:
            continue
        vertices_world = np.asarray(vertices_world, dtype=np.float32)
        if vertices_world.ndim != 2 or vertices_world.shape[0] < 2 or vertices_world.shape[1] != 3:
            continue
        pts_ref = world_to_reference_coordinates(vertices_world, world_to_reference)
        pts_px = reference_to_bev_pixels(pts_ref, bev_config)
        pts_px = np.rint(pts_px).astype(np.int32)
        # Clip to image bounds to avoid overflow; cv2 will ignore outside points but we keep it safe
        pts_px[:, 0] = np.clip(pts_px[:, 0], 0, bev_config.width - 1)
        pts_px[:, 1] = np.clip(pts_px[:, 1], 0, bev_config.height - 1)
        cv2.polylines(bev_image, [pts_px], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _map_lane_color_to_bgr(name: str | None) -> list[int]:
    """Map lane color name to BGR list.

    Args:
        name: Lane color name.

    Returns:
        List of three BGR values [B, G, R].
    """
    if not name:
        return color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("lane_line", {}).get("color"))
    return color_name_to_bgr(name)


def _polyline_to_pixels(
    vertices_world: np.ndarray, world_to_reference: np.ndarray, bev_config: BEVConfig
) -> np.ndarray:
    """Convert polyline vertices to BEV pixels.

    Args:
        vertices_world: Polyline vertices in world coordinates.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.

    Returns:
        Polyline vertices in BEV pixels.
    """
    if vertices_world.shape[1] != 3:
        raise ValueError(f"vertices_world must have 3 columns, got shape {vertices_world.shape}")

    if world_to_reference.shape != (4, 4):
        raise ValueError(f"world_to_reference must be a 4x4 matrix, got shape {world_to_reference.shape}")

    if bev_config.width <= 0 or bev_config.height <= 0:
        raise ValueError(
            f"bev_config must have positive width and height, got width {bev_config.width} and height {bev_config.height}"
        )

    pts_ref = world_to_reference_coordinates(vertices_world, world_to_reference)
    pts_px = reference_to_bev_pixels(pts_ref, bev_config)
    return pts_px


def _draw_solid_polyline_px(
    img: np.ndarray, pts_px_i32: np.ndarray, color: tuple[int, int, int] | list[int], thickness: int
) -> None:
    """Draw a solid polyline in BEV image.

    Args:
        img: BEV image.
        pts_px_i32: Polyline vertices in BEV pixels (int32).
        color: Color of the polyline (BGR tuple).
        thickness: Thickness of the polyline.
    """
    cv2.polylines(img, [pts_px_i32], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_dashed_polyline_px(
    img: np.ndarray,
    pts_px: np.ndarray,
    color: tuple[int, int, int] | list[int],
    thickness: int,
    dash_len_px: float,
    gap_len_px: float,
) -> None:
    """Draw a dashed polyline in BEV image.

    Args:
        img: BEV image.
        pts_px: Polyline vertices in BEV pixels.
        color: Color of the polyline (BGR list or tuple).
        thickness: Thickness of the polyline.
        dash_len_px: Length of the dash in pixels.
        gap_len_px: Length of the gap in pixels.

    Raises:
        ValueError: If pts_px has less than 2 vertices or invalid shape.
        ValueError: If color is None.
        ValueError: If thickness is not positive.
    """
    if pts_px.shape[1] != 2:
        raise ValueError(f"pts_px must have 2 columns, got shape {pts_px.shape}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if len(pts_px) < 2:
        raise ValueError(f"pts_px must have at least 2 vertices, got {len(pts_px)}")

    dash = float(max(1.0, dash_len_px))
    gap = float(max(1.0, gap_len_px))
    pattern = dash + gap
    acc = 0.0
    p0 = pts_px[0].astype(np.float32)
    for i in range(1, len(pts_px)):
        p1 = pts_px[i].astype(np.float32)
        seg = p1 - p0
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len <= 1e-3:
            p0 = p1
            continue
        dir_vec = seg / seg_len
        start = 0.0
        while start < seg_len:
            # Determine if in dash or gap
            phase = (acc + start) % pattern
            remain = seg_len - start
            if phase < dash:
                draw_len = min(dash - phase, remain)
                a = p0 + dir_vec * start
                b = p0 + dir_vec * (start + draw_len)
                cv2.line(
                    img,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
                start += draw_len
            else:
                skip_len = min(pattern - phase, remain)
                start += skip_len
        acc += seg_len
        p0 = p1


def _offset_polyline_px(pts_px: np.ndarray, offset_px: float) -> np.ndarray:
    """Offset a polyline in BEV image.

    Args:
        pts_px: Polyline vertices in BEV pixels.
        offset_px: Offset in pixels.

    Returns:
        Offset polyline vertices in BEV pixels.
    """
    if len(pts_px) < 2:
        return pts_px
    out = []
    for i in range(len(pts_px)):
        if i == 0:
            t = (pts_px[1] - pts_px[0]).astype(np.float32)
        elif i == len(pts_px) - 1:
            t = (pts_px[-1] - pts_px[-2]).astype(np.float32)
        else:
            t = (pts_px[i + 1] - pts_px[i - 1]).astype(np.float32)
        norm = float(np.hypot(t[0], t[1]))
        if norm < 1e-6:
            n = np.array([0.0, 0.0], dtype=np.float32)
        else:
            t /= norm
            n = np.array([-t[1], t[0]], dtype=np.float32)
        out.append(pts_px[i].astype(np.float32) + n * float(offset_px))
    return np.asarray(out)


def draw_lane_polylines_bev(
    bev_image: np.ndarray,
    lanes_with_attr: list[dict],
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    base_thickness: int = 2,
) -> None:
    """Draw lane polylines in BEV image.

    Args:
        bev_image: BEV image.
        lanes_with_attr: Lane with attributes (vertices, color, style).
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        base_thickness: Base thickness of the lane polylines.
    """
    if not lanes_with_attr:
        return
    ppm_h = bev_config.height / (bev_config.x_range[1] - bev_config.x_range[0])
    ppm_w = bev_config.width / (bev_config.y_range[1] - bev_config.y_range[0])
    # Use average to convert meters to pixels roughly isotropically
    m2px = float(0.5 * (ppm_h + ppm_w))
    for lane in lanes_with_attr:
        verts = np.asarray(lane.get("vertices"), dtype=np.float32)
        if verts.ndim != 2 or verts.shape[0] < 2:
            continue
        color = _map_lane_color_to_bgr(lane.get("color"))
        style = (lane.get("style") or "solid").strip().lower()
        pts_px = _polyline_to_pixels(verts, world_to_reference, bev_config)
        pts_px = np.rint(pts_px).astype(np.int32)
        if style in ("solid", "continuous"):
            _draw_solid_polyline_px(bev_image, pts_px, color, base_thickness)
        elif style in ("dashed", "dash"):
            dash_len_px = 1.0 * m2px
            gap_len_px = 1.0 * m2px
            _draw_dashed_polyline_px(
                bev_image, pts_px.astype(np.float32), color, base_thickness, dash_len_px, gap_len_px
            )
        elif style in ("double", "double_solid", "double_yellow"):
            offset_m = 0.18
            offset_px = offset_m * m2px
            pts_px_f = pts_px.astype(np.float32)
            left = _offset_polyline_px(pts_px_f, offset_px)
            right = _offset_polyline_px(pts_px_f, -offset_px)
            _draw_solid_polyline_px(bev_image, np.rint(left).astype(np.int32), color, base_thickness)
            _draw_solid_polyline_px(bev_image, np.rint(right).astype(np.int32), color, base_thickness)
        elif style in ("dotted", "dot"):
            dash_len_px = 0.5 * m2px
            gap_len_px = 0.5 * m2px
            _draw_dashed_polyline_px(
                bev_image, pts_px.astype(np.float32), color, base_thickness, dash_len_px, gap_len_px
            )
        else:
            _draw_solid_polyline_px(bev_image, pts_px, color, base_thickness)


def draw_square_marker_bev(
    bev_image: np.ndarray,
    center_world: np.ndarray,
    size_meters: float,
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    color: tuple[int, int, int] | list[int],
    alpha: float = 1.0,
) -> None:
    """Draw a square marker in BEV image.

    Args:
        bev_image: BEV image.
        center_world: Center of the marker in world coordinates.
        size_meters: Size of the marker in meters.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        color: Color of the marker (BGR tuple).
        alpha: Alpha value for blending.
    """
    if bev_image.shape[:2] != (bev_config.height, bev_config.width):
        raise ValueError(
            f"bev_image must have shape {bev_config.height}x{bev_config.width}, got shape {bev_image.shape[:2]}"
        )

    if center_world.shape[1] != 3:
        raise ValueError(f"center_world must have 3 columns, got shape {center_world.shape}")
    center_ref = world_to_reference_coordinates(center_world.reshape(1, 3), world_to_reference)[0]
    half_size_m = size_meters / 2.0
    corners_ref = np.array(
        [
            [center_ref[0] - half_size_m, center_ref[1] - half_size_m, center_ref[2]],
            [center_ref[0] + half_size_m, center_ref[1] - half_size_m, center_ref[2]],
            [center_ref[0] + half_size_m, center_ref[1] + half_size_m, center_ref[2]],
            [center_ref[0] - half_size_m, center_ref[1] + half_size_m, center_ref[2]],
        ],
        dtype=np.float32,
    )
    pixels = reference_to_bev_pixels(corners_ref, bev_config).astype(np.int32)
    if alpha >= 1.0:
        cv2.fillPoly(bev_image, [pixels], color)
    else:
        mask = np.zeros(bev_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pixels], 255)
        temp = np.zeros_like(bev_image)
        cv2.fillPoly(temp, [pixels], color)
        blended = cv2.addWeighted(bev_image, 1 - alpha, temp, alpha, 0)
        bev_image[mask > 0] = blended[mask > 0]


def draw_traffic_light_unit_bev(
    bev_image: np.ndarray,
    center_world: np.ndarray,
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    unit_length_m: float = 1.0,
    rotate_90_degrees: bool = True,
):
    """Draw a traffic light unit in BEV image.

    Args:
        bev_image: BEV image.
        center_world: Center of the traffic light unit in world coordinates.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        unit_length_m: Length of the traffic light unit in meters.
        rotate_90_degrees: Whether to rotate the traffic light unit 90 degrees.

    Raises:
        ValueError: If bev_image has invalid shape.
        ValueError: If center_world has invalid shape.
    """
    if bev_image.shape[:2] != (bev_config.height, bev_config.width):
        raise ValueError(
            f"bev_image must have shape {bev_config.height}x{bev_config.width}, got shape {bev_image.shape[:2]}"
        )

    if center_world.shape[1] != 3:
        raise ValueError(f"center_world must have 3 columns, got shape {center_world.shape}")
    center_ref = world_to_reference_coordinates(center_world.reshape(1, 3), world_to_reference)[0]
    center_px = reference_to_bev_pixels(center_ref.reshape(1, 3), bev_config)[0].astype(np.int32)

    ppm_h = bev_config.height / (bev_config.x_range[1] - bev_config.x_range[0])
    ppm_w = bev_config.width / (bev_config.y_range[1] - bev_config.y_range[0])

    height_px = int(round(unit_length_m * ppm_h))
    width_px = int(round(0.35 * unit_length_m * ppm_w))
    radius_px = max(2, int(round(0.12 * unit_length_m * (ppm_h + ppm_w) * 0.5)))
    gap_px = max(2, int(round(0.08 * unit_length_m * (ppm_h + ppm_w) * 0.5)))

    if rotate_90_degrees:
        x0 = int(center_px[0] - height_px // 2)
        x1 = int(center_px[0] + height_px // 2)
        y0 = int(center_px[1] - width_px // 2)
        y1 = int(center_px[1] + width_px // 2)
        cv2.rectangle(bev_image, (x0, y0), (x1, y1), (50, 50, 50), thickness=-1)

        total_bulb_width = 3 * (2 * radius_px) + 2 * gap_px
        start_x = int(center_px[0] - total_bulb_width // 2)
        bulb_centers = [
            (start_x + radius_px, center_px[1]),
            (start_x + 3 * radius_px + gap_px, center_px[1]),
            (start_x + 5 * radius_px + 2 * gap_px, center_px[1]),
        ]
    else:
        x0 = int(center_px[0] - width_px // 2)
        x1 = int(center_px[0] + width_px // 2)
        y0 = int(center_px[1] - height_px // 2)
        y1 = int(center_px[1] + height_px // 2)
        cv2.rectangle(bev_image, (x0, y0), (x1, y1), (50, 50, 50), thickness=-1)

        total_bulb_height = 3 * (2 * radius_px) + 2 * gap_px
        start_y = int(center_px[1] - total_bulb_height // 2)
        bulb_centers = [
            (center_px[0], start_y + radius_px),
            (center_px[0], start_y + 3 * radius_px + gap_px),
            (center_px[0], start_y + 5 * radius_px + 2 * gap_px),
        ]

    cv2.circle(bev_image, bulb_centers[0], radius_px, (0, 0, 255), thickness=-1)
    cv2.circle(bev_image, bulb_centers[1], radius_px, (0, 255, 255), thickness=-1)
    cv2.circle(bev_image, bulb_centers[2], radius_px, (0, 255, 0), thickness=-1)


def draw_crosswalk_zebra_bev(
    bev_image: np.ndarray,
    polygon_world: np.ndarray,
    world_to_reference: np.ndarray,
    bev_config: BEVConfig,
    stripe_width_meters: float = 0.5,
    gap_width_meters: float = 0.5,
    stripe_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> None:
    """Draw a crosswalk with zebra stripes in BEV image.

    Args:
        bev_image: BEV image.
        polygon_world: Crosswalk polygon vertices in world coordinates.
        world_to_reference: World to reference transformation matrix.
        bev_config: BEV configuration.
        stripe_width_meters: Width of each stripe in meters.
        gap_width_meters: Width of gap between stripes in meters.
        stripe_color: Color of the stripes (BGR tuple).
    """
    if polygon_world is None or len(polygon_world) == 0:
        return

    points_ref = world_to_reference_coordinates(polygon_world, world_to_reference)
    polygon_px = reference_to_bev_pixels(points_ref, bev_config).astype(np.int32)

    ref_xy = points_ref[:, :2].astype(np.float32)
    if len(ref_xy) < 2:
        return
    closed_xy = ref_xy
    if not np.allclose(ref_xy[0], ref_xy[-1]):
        closed_xy = np.vstack([ref_xy, ref_xy[0]])
    max_len2 = 0.0
    best_vec = np.array([1.0, 0.0], dtype=np.float32)
    for i in range(len(closed_xy) - 1):
        vec = closed_xy[i + 1] - closed_xy[i]
        L2 = float(vec[0] * vec[0] + vec[1] * vec[1])
        if L2 > max_len2:
            max_len2 = L2
            best_vec = vec

    bx, by = float(best_vec[0]), float(best_vec[1])
    b_norm = float(np.hypot(bx, by))
    if b_norm < 1e-6:
        bx, by = 1.0, 0.0
        b_norm = 1.0
    bx /= b_norm
    by /= b_norm

    ppm_h = bev_config.height / (bev_config.x_range[1] - bev_config.x_range[0])
    ppm_w = bev_config.width / (bev_config.y_range[1] - bev_config.y_range[0])
    e = np.array([-by * ppm_w, -bx * ppm_h], dtype=np.float32)
    e_norm = float(np.hypot(e[0], e[1]))
    if e_norm < 1e-6:
        return
    e /= e_norm

    scale_per_meter_px = float(np.hypot(-by * ppm_w, -bx * ppm_h))
    stripe_thickness_px = max(1.0, stripe_width_meters * scale_per_meter_px)
    gap_thickness_px = max(1.0, gap_width_meters * scale_per_meter_px)
    period_px = stripe_thickness_px + gap_thickness_px

    poly_mask = np.zeros(bev_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon_px], 255)

    x_min = int(max(np.min(polygon_px[:, 0]), 0))
    x_max = int(min(np.max(polygon_px[:, 0]), bev_config.width - 1))
    y_min = int(max(np.min(polygon_px[:, 1]), 0))
    y_max = int(min(np.max(polygon_px[:, 1]), bev_config.height - 1))
    if x_min > x_max or y_min > y_max:
        return

    proj_vertices = polygon_px[:, 0] * e[0] + polygon_px[:, 1] * e[1]
    base = float(np.min(proj_vertices))

    uu, vv = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float32),
        np.arange(y_min, y_max + 1, dtype=np.float32),
    )
    proj = uu * e[0] + vv * e[1] - base
    phase = np.mod(proj, period_px)
    stripe_local = (phase < stripe_thickness_px).astype(np.uint8) * 255

    local_poly_mask = poly_mask[y_min : y_max + 1, x_min : x_max + 1]
    stripes_inside = cv2.bitwise_and(stripe_local, local_poly_mask)
    region = bev_image[y_min : y_max + 1, x_min : x_max + 1]
    region[stripes_inside > 0] = stripe_color


def cluster_points_xy(points_world: np.ndarray | None, radius_meters: float = 1.0) -> np.ndarray | None:
    """Cluster points in XY plane using distance-based connectivity.

    Args:
        points_world: Points in world coordinates to cluster, or None.
        radius_meters: Maximum distance for two points to be in the same cluster.

    Returns:
        Array of cluster centers (mean position of each cluster), or None if input is None/empty.
    """
    if points_world is None or len(points_world) == 0:
        return points_world

    pts = np.asarray(points_world, dtype=np.float32)
    num = len(pts)
    visited = np.zeros(num, dtype=bool)
    clusters = []
    r2 = radius_meters * radius_meters

    for i in range(num):
        if visited[i]:
            continue
        visited[i] = True
        cluster_idx = [i]
        changed = True
        while changed:
            changed = False
            for j in range(num):
                if visited[j]:
                    continue
                for k in cluster_idx:
                    dx = pts[j, 0] - pts[k, 0]
                    dy = pts[j, 1] - pts[k, 1]
                    if dx * dx + dy * dy <= r2:
                        visited[j] = True
                        cluster_idx.append(j)
                        changed = True
                        break
        cluster_pts = pts[cluster_idx]
        center = np.mean(cluster_pts, axis=0)
        clusters.append(center)

    return np.asarray(clusters, dtype=np.float32)


def filter_wait_lines_by_spacing(
    wait_lines: np.ndarray | None, distance_threshold_m: float = 1.0, parallel_cos_threshold: float = 0.98
) -> np.ndarray | None:
    """Filter closely spaced parallel wait lines to keep only the longest ones.

    Args:
        wait_lines: Array of wait line segments (Nx2x3 array), or None.
        distance_threshold_m: Maximum distance in meters for lines to be considered duplicates.
        parallel_cos_threshold: Minimum cosine similarity for lines to be considered parallel.

    Returns:
        Filtered array of wait line segments, or None if input is None/empty.
    """
    if wait_lines is None or len(wait_lines) == 0:
        return wait_lines

    segments = np.asarray(wait_lines, dtype=np.float32)
    p0 = segments[:, 0, :2]
    p1 = segments[:, 1, :2]
    v = p1 - p0
    lengths = np.linalg.norm(v, axis=1)

    valid = lengths > 1e-4
    segments = segments[valid]
    p0 = p0[valid]
    p1 = p1[valid]
    v = v[valid]
    lengths = lengths[valid]

    if len(segments) == 0:
        return segments

    directions = v / lengths[:, None]
    centers = 0.5 * (p0 + p1)

    order = np.argsort(-lengths)
    keep_mask = np.zeros(len(segments), dtype=bool)

    for idx in order:
        if keep_mask[idx]:
            continue
        keep_mask[idx] = True
        d_ref = directions[idx]
        n_ref = np.array([-d_ref[1], d_ref[0]], dtype=np.float32)
        c_ref = centers[idx]

        for j in order:
            if j == idx or keep_mask[j]:
                continue
            cos_sim = float(abs(np.dot(directions[j], d_ref)))
            if cos_sim < parallel_cos_threshold:
                continue
            normal_dist = abs(float(np.dot(centers[j] - c_ref, n_ref)))
            if normal_dist < distance_threshold_m:
                keep_mask[j] = False

    filtered = segments[keep_mask]
    return filtered


def render_bev_frame(
    bev_image: np.ndarray,
    ego_pose: np.ndarray,
    all_object_info: dict,
    lane_lines: np.ndarray | None,
    lanes_with_attr: list[dict] | None,
    road_boundaries: list,
    bev_config: BEVConfig,
    mean_ego_pose: np.ndarray,
    object_tracker: ObjectTracker | None = None,
    camera_convention: str = "rdf",
    crosswalks: list | None = None,
    wait_lines: np.ndarray | None = None,
    traffic_light_centers: np.ndarray | None = None,
    traffic_sign_centers: np.ndarray | None = None,
) -> None:
    """Render a single BEV frame with all scene elements.

    Args:
        bev_image: Output BEV image to render into (modified in-place).
        ego_pose: Current ego vehicle pose (4x4 transformation matrix).
        all_object_info: Dictionary mapping tracking IDs to object information.
        lane_lines: Lane line segments as array, or None.
        lanes_with_attr: Lane polylines with color and style attributes, or None.
        road_boundaries: Road boundary polylines.
        bev_config: BEV rendering configuration.
        mean_ego_pose: Mean ego pose used as reference frame (4x4 matrix).
        object_tracker: Optional object tracker for temporal smoothing.
        camera_convention: Camera coordinate convention ("rdf" or "flu").
        crosswalks: List of crosswalk polygon vertices, or None.
        wait_lines: Wait line segments as array, or None.
        traffic_light_centers: Traffic light center positions, or None.
        traffic_sign_centers: Traffic sign center positions, or None.
    """
    bev_image[:] = 0

    if camera_convention == "rdf":
        camera_to_world_rdf = ego_pose
        ego_pose_flu = np.concatenate(
            [
                camera_to_world_rdf[:, 2:3],
                -camera_to_world_rdf[:, 0:1],
                -camera_to_world_rdf[:, 1:2],
                camera_to_world_rdf[:, 3:],
            ],
            axis=1,
        )
    elif camera_convention == "flu":
        ego_pose_flu = ego_pose
    else:
        raise ValueError(f"Unsupported camera_convention: {camera_convention}")
    ego_pose_flu_grounded = grounding_pose(ego_pose_flu, convention="flu")

    reference_to_world = mean_ego_pose
    world_to_reference = np.linalg.inv(reference_to_world)
    current_ego_to_world = ego_pose_flu_grounded
    current_ego_in_reference = world_to_reference @ current_ego_to_world

    if object_tracker is not None:
        smoothed_objects = object_tracker.get_smoothed_objects()
        object_entries = [(tracked["info"], tracked["fade_alpha"]) for tracked in smoothed_objects.values()]
    else:
        object_entries = [(info, 1.0) for info in all_object_info.values()]

    if road_boundaries is not None and len(road_boundaries) > 0:
        rb_color = color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("road_boundary", {}).get("color"))
        draw_polylines_bev(
            bev_image=bev_image,
            polylines_world=road_boundaries,
            world_to_reference=world_to_reference,
            bev_config=bev_config,
            color=rb_color,
            thickness=4,
        )

    if hasattr(bev_config, "width"):  # no-op, ensure variables used
        pass
    if hasattr(bev_config, "height"):
        pass

    # Lane lines with attributes (color/style)
    if lanes_with_attr is not None and len(lanes_with_attr) > 0:
        draw_lane_polylines_bev(
            bev_image=bev_image,
            lanes_with_attr=lanes_with_attr,
            world_to_reference=world_to_reference,
            bev_config=bev_config,
            base_thickness=2,
        )
    elif lane_lines is not None and len(lane_lines) > 0:
        ll_color = color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("lane_line", {}).get("color"))
        draw_polylines_bev(
            bev_image=bev_image,
            polylines_world=lane_lines,
            world_to_reference=world_to_reference,
            bev_config=bev_config,
            color=ll_color,
            thickness=2,
        )

    if crosswalks is not None and len(crosswalks) > 0:
        for polygon in crosswalks:
            draw_crosswalk_zebra_bev(
                bev_image,
                polygon,
                world_to_reference,
                bev_config,
                stripe_width_meters=0.5,
                gap_width_meters=0.5,
                stripe_color=(255, 255, 255),
            )

    if wait_lines is not None and len(wait_lines) > 0:
        draw_line_segments_bev(
            bev_image=bev_image,
            line_segments_world=wait_lines,
            world_to_reference=world_to_reference,
            bev_config=bev_config,
            color=color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("wait_line", {}).get("color")),
            thickness=6,
        )

    for object_info, fade_alpha in object_entries:
        object_to_world = np.array(object_info["object_to_world"])
        object_lwh = np.array(object_info["object_lwh"])
        object_type = object_info.get("object_type", "unknown")

        object_center_world = object_to_world[:3, 3]
        object_center_reference = world_to_reference_coordinates(object_center_world.reshape(1, 3), world_to_reference)[
            0
        ]

        if filter_points_in_bev_range(object_center_reference.reshape(1, 3), bev_config)[0]:
            object_type_lower = object_type.lower()
            if object_type_lower in ["pedestrian", "person", "walker"]:
                ped_color = color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("pedestrian", {}).get("color"))
                draw_pedestrian_circle(bev_image, object_center_reference, ped_color, bev_config, fade_alpha)
            else:
                is_two_wheeler = object_type_lower in ["cyclist", "motorcycle", "bicycle"]
                object_to_reference = world_to_reference @ object_to_world
                forward_ref = object_to_reference[:3, 0]
                if forward_ref[0] != 0 or forward_ref[1] != 0:
                    yaw = float(np.arctan2(forward_ref[1], forward_ref[0]))
                else:
                    yaw = None
                veh_color = color_name_to_bgr(
                    ANNOTATION_STYLE_REGISTRY.get("two_wheeler" if is_two_wheeler else "vehicle", {}).get("color")
                )
                draw_vehicle(
                    bev_image,
                    object_center_reference,
                    object_lwh,
                    veh_color,
                    bev_config,
                    fade_alpha,
                    draw_center_cross=False,
                    yaw_radians=yaw,
                )

    if traffic_light_centers is not None and len(traffic_light_centers) > 0:
        clustered_lights = cluster_points_xy(traffic_light_centers, radius_meters=3.0)
        for center in clustered_lights:
            draw_traffic_light_unit_bev(
                bev_image,
                np.asarray(center, dtype=np.float32),
                world_to_reference,
                bev_config,
                unit_length_m=2.0,
            )

    if traffic_sign_centers is not None and len(traffic_sign_centers) > 0:
        clustered_signs = cluster_points_xy(traffic_sign_centers, radius_meters=3.0)
        for center in clustered_signs:
            ts_color = color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("traffic_sign", {}).get("color"))
            draw_square_marker_bev(
                bev_image,
                np.asarray(center, dtype=np.float32),
                1.0,
                world_to_reference,
                bev_config,
                ts_color,
            )

    ego_position_in_reference = current_ego_in_reference[:3, 3]
    ego_forward_ref = current_ego_in_reference[:3, 0]
    if ego_forward_ref[0] != 0 or ego_forward_ref[1] != 0:
        ego_yaw = float(np.arctan2(ego_forward_ref[1], ego_forward_ref[0]))
    else:
        ego_yaw = None
    ego_color = color_name_to_bgr(ANNOTATION_STYLE_REGISTRY.get("ego_vehicle", {}).get("color"))
    draw_vehicle(
        bev_image,
        ego_position_in_reference,
        [4.5, 2.0, 1.8],
        ego_color,
        bev_config,
        alpha=1.0,
        draw_center_cross=False,
        yaw_radians=ego_yaw,
    )


class WorldModelBevRender:
    """BEV renderer for autonomous driving scenes.

    Renders bird's-eye-view visualizations of driving scenes using RDS dataset data,
    including ego vehicle, objects, lane lines, road boundaries, and traffic controls.

    Args:
        dataset_dir: Path to RDS dataset directory.
        clip_id: Clip ID to render.
        master_camera: Camera name to use for ego pose.
        resolution: BEV resolution in meters per pixel.
        object_persistence_frames: Number of frames to persist objects after last seen.
        object_fade_frames: Number of frames to fade out objects.
        rotate_clockwise: Whether to rotate the BEV image 90 degrees clockwise.
    """

    def __init__(
        self,
        dataset_dir: str,
        clip_id: str,
        master_camera: str = "camera_front_wide_120fov",
        resolution: float = 0.1,
        object_persistence_frames: int = 3,
        object_fade_frames: int = 1,
        rotate_clockwise: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.clip_id = clip_id
        self.master_camera = master_camera

        self.data_loader = RdsDataLoader(dataset_dir, clip_id)
        self.rotate_clockwise = rotate_clockwise

        self.bev_config = BEVConfig.define_render_range(resolution)

        self.camera_poses = self.data_loader.get_camera_poses(master_camera)
        if not self.camera_poses:
            raise ValueError(f"No camera poses found for camera {master_camera}")

        object_frames = set(self.data_loader.all_dynamic_objects.keys())
        pose_frames = set(self.camera_poses.keys())
        common_frames = sorted(object_frames & pose_frames)
        if not common_frames:
            raise ValueError("No overlapping frames between object data and camera poses")
        self.frame_indices = common_frames

        self.mean_ego_pose = self._compute_mean_ego_pose()

        # Ensure ego stays within x_range by shifting window if needed
        self._maybe_adjust_x_range_for_ego()

        # Load polylines for smooth AA rendering, and segments for legacy helpers
        self.lane_lines_polylines = self._load_polylines(self.data_loader.get_lanelines())
        self.lane_lines_polylines_with_attr = self._load_lane_polylines_with_attr(self.data_loader.get_lanelines())
        self.road_boundaries_polylines = self._load_polylines(self.data_loader.get_road_boundaries())
        self.lane_lines = self._load_polyline_segments(self.data_loader.get_lanelines())
        self.road_boundaries = self._load_polyline_segments(self.data_loader.get_road_boundaries())
        wait_line_segments = self._load_polyline_segments(self.data_loader.get_wait_lines())
        self.wait_lines = filter_wait_lines_by_spacing(wait_line_segments, distance_threshold_m=3.0)
        self.crosswalks = self._load_polygons(self.data_loader.get_crosswalks())
        self.traffic_light_centers = self._load_centers(self.data_loader.get_traffic_lights())
        self.traffic_sign_centers = self._load_centers(self.data_loader.get_traffic_signs())

        self.object_tracker = ObjectTracker(
            persistence_frames=object_persistence_frames, fade_frames=object_fade_frames
        )

    def _compute_mean_ego_pose(self) -> np.ndarray:
        """Compute mean ego pose across all frames to use as reference frame.

        Returns:
            Mean ego pose as 4x4 transformation matrix.

        Raises:
            ValueError: If no positions are available.
        """
        positions = []
        for frame_idx in self.frame_indices:
            ego_pose = self.camera_poses[frame_idx]
            ego_pose_opencv = ego_pose
            ego_pose_flu = np.concatenate(
                [
                    ego_pose_opencv[:, 2:3],
                    -ego_pose_opencv[:, 0:1],
                    -ego_pose_opencv[:, 1:2],
                    ego_pose_opencv[:, 3:],
                ],
                axis=1,
            )
            ego_pose_flu_grounded = grounding_pose(ego_pose_flu, convention="flu")
            positions.append(ego_pose_flu_grounded[:3, 3])

        if not positions:
            raise ValueError("Unable to compute mean ego pose: no positions available")

        positions = np.array(positions)
        mean_position = np.mean(positions, axis=0)

        mean_pose = np.eye(4)
        mean_pose[:3, 3] = mean_position
        return mean_pose

    @staticmethod
    def _load_polyline_segments(label_entries: list) -> np.ndarray:
        """Convert polyline label entries to line segments.

        Args:
            label_entries: List of label entries with vertices.

        Returns:
            Array of line segments (Nx2x3).
        """
        segments = []
        for entry in label_entries:
            vertices = np.asarray(entry.get("vertices"), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[0] < 2:
                continue
            segments.append(np.stack([vertices[:-1], vertices[1:]], axis=1))
        if not segments:
            return np.zeros((0, 2, 3), dtype=np.float32)
        return np.concatenate(segments, axis=0)

    @staticmethod
    def _load_polygons(label_entries: list) -> list:
        """Load polygon vertices from label entries.

        Args:
            label_entries: List of label entries with vertices.

        Returns:
            List of polygon vertex arrays.
        """
        polygons = []
        for entry in label_entries:
            vertices = np.asarray(entry.get("vertices"), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[0] < 3:
                continue
            polygons.append(vertices)
        return polygons

    @staticmethod
    def _load_centers(label_entries: list) -> np.ndarray:
        """Compute center points from label entry vertices.

        Args:
            label_entries: List of label entries with vertices.

        Returns:
            Array of center points (Nx3).
        """
        centers = []
        for entry in label_entries:
            vertices = np.asarray(entry.get("vertices"), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[0] == 0:
                continue
            centers.append(np.mean(vertices, axis=0))
        if not centers:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(centers, dtype=np.float32)

    @staticmethod
    def _load_polylines(label_entries: list) -> list:
        """Load polyline vertices from label entries.

        Args:
            label_entries: List of label entries with vertices.

        Returns:
            List of polyline vertex arrays.
        """
        polylines = []
        for entry in label_entries:
            vertices = np.asarray(entry.get("vertices"), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[0] < 2:
                continue
            polylines.append(vertices)
        return polylines

    @staticmethod
    def _predominant_attr(attributes: dict, key: str) -> str | None:
        """Extract the most common attribute value from a list or return single value.

        Args:
            attributes: Dictionary of attributes.
            key: Attribute key to extract.

        Returns:
            Most frequent attribute value, or None if not found.
        """
        if not attributes or key not in attributes:
            return None
        vals = attributes.get(key)
        if isinstance(vals, list) and len(vals) > 0:
            # pick most frequent non-empty
            counts = {}
            for v in vals:
                if v in (None, ""):
                    continue
                counts[v] = counts.get(v, 0) + 1
            if counts:
                return max(counts.items(), key=lambda kv: kv[1])[0]
            return None
        if isinstance(vals, str):
            return vals
        return None

    @staticmethod
    def _load_lane_polylines_with_attr(label_entries: list) -> list:
        """Load lane polylines with color and style attributes.

        Args:
            label_entries: List of label entries with vertices and attributes.

        Returns:
            List of dictionaries with vertices, color, and style.
        """
        lanes = []
        for entry in label_entries:
            vertices = np.asarray(entry.get("vertices"), dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[0] < 2:
                continue
            attributes = entry.get("attributes", {}) if isinstance(entry, dict) else {}
            color = WorldModelBevRender._predominant_attr(
                attributes, "colors"
            ) or WorldModelBevRender._predominant_attr(attributes, "color")
            style = WorldModelBevRender._predominant_attr(
                attributes, "styles"
            ) or WorldModelBevRender._predominant_attr(attributes, "style")
            lanes.append(
                {
                    "vertices": vertices,
                    "color": color,
                    "style": style,
                }
            )
        return lanes

    def get_frame_range(self) -> list:
        """Get list of available frame indices.

        Returns:
            List of frame indices that can be rendered.
        """
        return list(self.frame_indices)

    def render(self, frame_index: int) -> np.ndarray:
        """Render BEV frame at the specified frame index.

        Args:
            frame_index: Frame index to render.

        Returns:
            Rendered BEV image as numpy array (HxWx3, BGR).

        Raises:
            ValueError: If frame_index is not available for rendering.
        """
        if frame_index not in self.frame_indices:
            raise ValueError(f"Frame {frame_index} not available for rendering")

        ego_pose = self.camera_poses[frame_index]
        all_object_info = self.data_loader.get_object_data_for_frame(frame_index)
        if all_object_info is None:
            all_object_info = {}

        self.object_tracker.update(frame_index, all_object_info)

        raw_height, raw_width = self.bev_config.height, self.bev_config.width
        working_image = np.zeros((raw_height, raw_width, 3), dtype=np.uint8)

        render_bev_frame(
            working_image,
            ego_pose,
            all_object_info,
            self.lane_lines,
            self.lane_lines_polylines_with_attr,
            self.road_boundaries,
            self.bev_config,
            mean_ego_pose=self.mean_ego_pose,
            object_tracker=self.object_tracker,
            camera_convention="rdf",
            crosswalks=self.crosswalks,
            wait_lines=self.wait_lines,
            traffic_light_centers=self.traffic_light_centers,
            traffic_sign_centers=self.traffic_sign_centers,
        )

        if self.rotate_clockwise:
            bev_image = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)
        else:
            bev_image = working_image

        return bev_image.copy()

    def render_at_time_offset(self, time_offset_s: float) -> tuple[np.ndarray, float]:
        """Render a BEV frame at the given time offset in seconds.

        Uses input data fps to convert time to nearest available frame index,
        then delegates to render(frame_index).

        Args:
            time_offset_s: Time offset in seconds from the first frame.

        Returns:
            Tuple of (bev_image, actual_time_offset_s).

        Raises:
            ValueError: If time_offset_s is negative.
        """
        if time_offset_s < 0:
            raise ValueError("time_offset_s must be non-negative")

        raw_fps = float(self.data_loader.get_fps("pose"))
        # Map seconds to frame position within our available frame_indices
        first_id = int(self.frame_indices[0])
        # Convert offset seconds to frame delta in the object stream
        target_index = first_id + int(round(time_offset_s * raw_fps))

        # Snap to closest available frame in self.frame_indices
        # Since indices may not be contiguous, pick the nearest by absolute difference
        closest = min(self.frame_indices, key=lambda x: abs(int(x) - target_index))
        bev_img = self.render(int(closest))
        # Compute actual time offset based on chosen frame and object fps, consistent with manifest t_s computation
        actual_time_offset_s = (int(closest) - first_id) / raw_fps
        return bev_img, float(actual_time_offset_s)

    def _maybe_adjust_x_range_for_ego(self) -> None:
        """Shift the BEV x_range so ego is within view across frames.

        Keeps total x-range span and resolution fixed. Only shifts the window
        if the ego would otherwise fall outside at the beginning or end.
        """
        try:
            # Prepare reference transform based on mean pose
            reference_to_world = self.mean_ego_pose
            world_to_reference = np.linalg.inv(reference_to_world)

            xs = []
            for frame_idx in self.frame_indices:
                ego_pose = self.camera_poses[frame_idx]
                # Convert to FLU and ground, matching render_bev_frame pipeline
                ego_pose_flu = np.concatenate(
                    [
                        ego_pose[:, 2:3],
                        -ego_pose[:, 0:1],
                        -ego_pose[:, 1:2],
                        ego_pose[:, 3:],
                    ],
                    axis=1,
                )
                ego_pose_flu_grounded = grounding_pose(ego_pose_flu, convention="flu")
                current_ego_in_reference = world_to_reference @ ego_pose_flu_grounded
                xs.append(float(current_ego_in_reference[0, 3]))

            if not xs:
                return
            min_x = float(min(xs))
            max_x = float(max(xs))

            x0, x1 = float(self.bev_config.x_range[0]), float(self.bev_config.x_range[1])
            width = x1 - x0
            if width <= 0:
                return

            # If already fully covered, no change
            if min_x >= x0 and max_x <= x1:
                return

            # If span exceeds width, best effort: align window to cover the tail end
            span = max_x - min_x
            if span > width:
                new_x0 = max_x - width
                new_x1 = max_x
            else:
                # Shift window to include both min and max within fixed width
                new_x0 = min_x
                new_x1 = new_x0 + width
                if new_x1 < max_x:
                    new_x0 = max_x - width
                    new_x1 = max_x

            # Apply shift while keeping width unchanged
            self.bev_config.x_range = [float(new_x0), float(new_x1)]
            # Height remains consistent since width/resolution unchanged
        except Exception:
            # Silent fail: keep default range if adjustment fails for any reason
            pass

    def get_annotation(self) -> str:
        """Return a multi-line legend string: '- {object type}: {color} {style}'.

        Returns:
            Multi-line string with annotation style information for all object types.
        """
        lines = []
        for object_type, cfg in ANNOTATION_STYLE_REGISTRY.items():
            color = (cfg.get("color") or "").strip()
            style = (cfg.get("style") or "").strip()
            lines.append(f"- {object_type}: {color} {style}")
        return "\n".join(lines)
