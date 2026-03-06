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

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import numpy as np
from webdataset import WebDataset, non_empty

from checks.obstacle.tools.export_viz_helper import ProjectionOverlayHelper
from checks.utils.cuboid import Cuboid
from checks.utils.cv2_video_dataset import VideoDataset
from checks.utils.rds_data_loader import RdsDataLoader


def _get_sample(url: str):
    """Get a sample from a URL with basic auto-decoding.

    Args:
        url: The URL of the sample to load.

    Returns:
        The sample from the URL.

    Raises:
        RuntimeError: If the sample cannot be loaded.
    """
    try:
        dataset = WebDataset(url, nodesplitter=non_empty, workersplitter=None, shardshuffle=False).decode()
        return next(iter(dataset))
    except Exception as e:
        raise RuntimeError(f"Failed to load sample from {url}: {e}") from e


def _extract_vertices(minimap_data, vertices_list=None):
    """Extract vertices from minimap data.

    Args:
        minimap_data: The minimap data.
        vertices_list: The list of vertices.

    Returns:
        The list of vertices.
    """
    if vertices_list is None:
        vertices_list = []
    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == "vertices":
                vertices_list.append(value)
            else:
                _extract_vertices(value, vertices_list)
    elif isinstance(minimap_data, list):
        for item in minimap_data:
            _extract_vertices(item, vertices_list)
    return vertices_list


def _load_minimap_vertices(input_data: str, clip_id: str, folder: str) -> List[np.ndarray]:
    """Load minimap vertices from a tar file.

    Args:
        input_data: The input data directory.
        clip_id: The clip ID.
        folder: The folder containing the tar file.

    Returns:
        A list of numpy arrays representing the vertices.
    """
    tar_path = os.path.join(input_data, folder, f"{clip_id}.tar")
    if not os.path.exists(tar_path):
        return []
    sample = _get_sample(tar_path)
    json_key = [k for k in sample if k.endswith(".json")]
    if not json_key:
        return []
    data = sample[json_key[0]]
    vertices = _extract_vertices(data)
    # Convert to np arrays
    arrs = []
    for v in vertices:
        try:
            arrs.append(np.asarray(v, dtype=np.float32))
        except (ValueError, TypeError):
            continue
    return arrs


def _project_points_to_pixels(
    points_world: np.ndarray, camera_to_world: np.ndarray, camera_model, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Project points to pixels.

    Args:
        points_world: The points in world coordinates.
        camera_to_world: The camera to world transformation matrix.
        camera_model: The camera model.
        width: The width of the image.
        height: The height of the image.

    Returns:
        The points in pixels and a valid mask.
    """
    if points_world is None or len(points_world) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    world_to_camera = np.linalg.inv(camera_to_world)
    # Homogeneous transform
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    pts_cam = (world_to_camera @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] > 0
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    pixels = camera_model.ray2pixel_np(pts_cam[valid])
    pixels = np.asarray(pixels, dtype=np.float32)
    # Clamp to image bounds
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return pixels, valid


def _bbox_from_pixels(pixels: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the bounding box from pixels.

    Args:
        pixels: The points in pixels.

    Returns:
        The bounding box.
    """
    if pixels is None or len(pixels) == 0:
        return 0, 0, 0, 0
    x_min = int(np.floor(pixels[:, 0].min()))
    y_min = int(np.floor(pixels[:, 1].min()))
    x_max = int(np.ceil(pixels[:, 0].max())) + 1
    y_max = int(np.ceil(pixels[:, 1].max())) + 1
    return x_min, y_min, x_max, y_max


def _write_json(path: Path, payload: Dict[str, Any]):
    """Write a JSON file.

    Args:
        path: The path to the JSON file.
        payload: The payload to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def _setup_video_output(out_dir: Path, clip_id: str, camera_name: str, img_w: int, img_h: int, output_fps: float):
    """Initialize a video writer with codec fallbacks.

    Args:
        out_dir: The output directory.
        clip_id: The clip ID.
        camera_name: The camera name.
        img_w: The width of the image.
        img_h: The height of the image.
        output_fps: The output FPS.

    Returns:
        The opened writer.

    Raises:
        RuntimeError: If the video writer cannot be initialized.
    """
    writer = None
    out_path = None

    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = [
        ("mp4", ["mp4v", "avc1", "H264"]),
        ("avi", ["XVID", "MJPG"]),  # MJPG is widely available
    ]
    for ext, codecs in attempts:
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_path = out_dir / f"{clip_id}.{camera_name}.projection.{ext}"
            temp_writer = cv2.VideoWriter(out_path.as_posix(), fourcc, output_fps, (img_w, img_h))
            if temp_writer.isOpened():
                writer = temp_writer
                break
            else:
                temp_writer.release()
        if writer:
            break

    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to initialize cv2.VideoWriter with available codecs")

    return writer


def _process_frame_data(
    loader: RdsDataLoader,
    frame_idx: int,
    pose: np.ndarray,
    camera_model,
    img_w: int,
    img_h: int,
    static_shapes: List[Tuple[int, str, np.ndarray]],
    lane_shapes: List[Tuple[int, str, np.ndarray]],
    dynamic_objects_catalog: Dict[int, str],
    dynamic_object_has_projection: Dict[int, bool],
    static_object_has_projection: Dict[int, bool],
    lane_object_has_projection: Dict[int, bool],
):
    """Compute per-frame projections for dynamic/static/lane objects.

    Args:
        loader: The loader.
        frame_idx: The frame index.
        pose: The pose.
        camera_model: The camera model.
        img_w: The width of the image.
        img_h: The height of the image.
        static_shapes: The static shapes.
        lane_shapes: The lane shapes.
        dynamic_objects_catalog: The dynamic objects catalog.
        dynamic_object_has_projection: The dynamic object has projection.
        static_object_has_projection: The static object has projection.
        lane_object_has_projection: The lane object has projection.

    Returns:
        The frame dynamic boxes, frame static boxes, frame lane points, frame static polyline points, frame static polygon vertices.
    """
    frame_dynamic_boxes: List[Dict[str, Any]] = []

    # Dynamic objects from all_object_info
    frame_objects = loader.get_object_data_for_frame(frame_idx)
    for track_id_str, tracked_obj in frame_objects.items():
        # Build projected mask using Cuboid for tight bbox
        try:
            cuboid = Cuboid(tracked_obj["object_to_world"], tracked_obj["object_lwh"])
            mask, _ = cuboid.get_projected_mask(pose, camera_model, img_w, img_h)
        except (KeyError, ValueError, np.linalg.LinAlgError):
            continue
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max()) + 1
        y_max = int(ys.max()) + 1
        x0 = max(0, min(x_min, img_w))
        y0 = max(0, min(y_min, img_h))
        x1 = max(0, min(x_max, img_w))
        y1 = max(0, min(y_max, img_h))
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue
        try:
            track_id = int(track_id_str)
        except Exception:
            # fallback to hash-based id if needed
            track_id = abs(hash(track_id_str)) % (10**9)
        obj_cat = str(tracked_obj.get("object_type", "unknown"))
        dynamic_objects_catalog.setdefault(track_id, obj_cat)
        frame_dynamic_boxes.append(
            {"object_id": int(track_id), "bbox_xywh": [int(x0), int(y0), int(w), int(h)], "category": obj_cat}
        )
        dynamic_object_has_projection[track_id] = True

    frame_static_boxes: List[Dict[str, Any]] = []
    frame_lane_points: List[Dict[str, Any]] = []
    frame_static_polyline_points: List[Dict[str, Any]] = []
    frame_static_polygon_vertices: List[Dict[str, Any]] = []

    # Static shapes
    for oid, category, vertices, _ in static_shapes:
        pixels, valid_mask = _project_points_to_pixels(vertices, pose, camera_model, img_w, img_h)
        if pixels.shape[0] < 1:
            continue
        if category in ("wait_line",):
            pts = pixels.astype(np.int32).tolist()
            if len(pts) == 0:
                continue
            frame_static_polyline_points.append({"object_id": int(oid), "points_xy": pts, "category": category})
            static_object_has_projection[oid] = True
        elif category in ("crosswalk", "road_marking"):
            verts = pixels.astype(np.int32).tolist()
            if len(verts) < 3:
                continue
            frame_static_polygon_vertices.append({"object_id": int(oid), "vertices_xy": verts, "category": category})
            static_object_has_projection[oid] = True
        else:
            x0, y0, x1, y1 = _bbox_from_pixels(pixels)
            w = x1 - x0
            h = y1 - y0
            if w <= 0 or h <= 0:
                continue
            frame_static_boxes.append(
                {"object_id": int(oid), "bbox_xywh": [int(x0), int(y0), int(w), int(h)], "category": category}
            )
            static_object_has_projection[oid] = True

    # Lane shapes
    for oid, category, vertices, _ in lane_shapes:
        pixels, valid_mask = _project_points_to_pixels(vertices, pose, camera_model, img_w, img_h)
        if pixels.shape[0] < 1:
            continue
        pts = pixels.astype(np.int32).tolist()
        if len(pts) == 0:
            continue
        frame_lane_points.append({"object_id": int(oid), "points_xy": pts, "category": category})
        lane_object_has_projection[oid] = True

    return (
        frame_dynamic_boxes,
        frame_static_boxes,
        frame_lane_points,
        frame_static_polyline_points,
        frame_static_polygon_vertices,
    )


def _build_catalogs(
    dynamic_objects_catalog: Dict[int, str],
    dynamic_object_has_projection: Dict[int, bool],
    static_objects_catalog: Dict[int, str],
    static_object_has_projection: Dict[int, bool],
    lanes_objects_catalog: Dict[int, str],
    lane_object_has_projection: Dict[int, bool],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Filter catalogs to objects with projections and sort IDs per category.

    Args:
        dynamic_objects_catalog: The dynamic objects catalog.
        dynamic_object_has_projection: The dynamic object has projection.
        static_objects_catalog: The static objects catalog.
        static_object_has_projection: The static object has projection.
        lanes_objects_catalog: The lane objects catalog.
        lane_object_has_projection: The lane object has projection.

    Returns:
        The dynamic objects dictionary, the static objects dictionary, the lane objects dictionary.
    """
    dynamic_objects_dict: DefaultDict[str, List[int]] = defaultdict(list)
    for oid, has in dynamic_object_has_projection.items():
        if has:
            category = str(dynamic_objects_catalog.get(oid, "unknown"))
            dynamic_objects_dict[category].append(int(oid))
    for k in dynamic_objects_dict:
        dynamic_objects_dict[k] = sorted(dynamic_objects_dict[k])

    static_objects_dict: DefaultDict[str, List[int]] = defaultdict(list)
    for oid, has in static_object_has_projection.items():
        if has:
            label_key = str(static_objects_catalog.get(oid, "unknown"))
            static_objects_dict[label_key].append(int(oid))
    for k in static_objects_dict:
        static_objects_dict[k] = sorted(static_objects_dict[k])

    lane_objects_dict: DefaultDict[str, List[int]] = defaultdict(list)
    for oid, has in lane_object_has_projection.items():
        if has:
            label_key = str(lanes_objects_catalog.get(oid, "unknown"))
            lane_objects_dict[label_key].append(int(oid))
    for k in lane_objects_dict:
        lane_objects_dict[k] = sorted(lane_objects_dict[k])

    return dynamic_objects_dict, static_objects_dict, lane_objects_dict


def _load_shapes_and_catalogs(
    input_data: str, clip_id: str
) -> Tuple[List[Tuple[int, str, np.ndarray]], List[Tuple[int, str, np.ndarray]], Dict[int, str], Dict[int, str]]:
    """Loads static and lane shapes from minimap data and initialize catalogs.

    Args:
        input_data: The input data directory.
        clip_id: The clip ID.

    Returns:
        The static shapes, lane shapes, static objects catalog, lane objects catalog.
    """
    static_sources = [
        ("3d_traffic_signs", "3d_traffic_signs", "traffic_sign"),
        ("3d_traffic_lights", "3d_traffic_lights", "traffic_light"),
        ("3d_poles", "3d_poles", "pole"),
        ("3d_wait_lines", "3d_wait_lines", "wait_line"),
        ("3d_crosswalks", "3d_crosswalks", "crosswalk"),
        ("3d_road_markings", "3d_road_markings", "road_marking"),
    ]
    static_shapes: List[Tuple[int, str, np.ndarray]] = []
    static_objects_catalog: Dict[int, str] = {}
    obj_id_counter = 1
    for label_key, folder, overlay_cat in static_sources:
        arrs = _load_minimap_vertices(input_data, clip_id, folder)
        for arr in arrs:
            if arr is None or len(arr) == 0:
                continue
            static_shapes.append((obj_id_counter, overlay_cat, arr, label_key))
            static_objects_catalog[obj_id_counter] = overlay_cat
            obj_id_counter += 1

    lane_sources = [
        ("3d_lanelines", "3d_lanelines", "lane_line"),
        ("3d_road_boundaries", "3d_road_boundaries", "road_boundary"),
    ]
    lane_shapes: List[Tuple[int, str, np.ndarray]] = []
    lanes_objects_catalog: Dict[int, str] = {}
    lane_id_counter = 1
    for label_key, folder, overlay_cat in lane_sources:
        arrs = _load_minimap_vertices(input_data, clip_id, folder)
        for arr in arrs:
            if arr is None or len(arr) == 0:
                continue
            lane_shapes.append((lane_id_counter, overlay_cat, arr, label_key))
            lanes_objects_catalog[lane_id_counter] = overlay_cat
            lane_id_counter += 1

    return static_shapes, lane_shapes, static_objects_catalog, lanes_objects_catalog


def _init_video_dataset(video_path: str) -> Tuple[VideoDataset, float]:
    """Creates a VideoDataset and infer its FPS.

    Args:
        video_path: The path to the video.

    Returns:
        The VideoDataset and the FPS.

    Raises:
        FileNotFoundError: If the video path does not exist.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    dataset = VideoDataset(video_path, transforms_fn=[])
    try:
        video_fps = float(dataset.frames_per_second)
    except Exception:
        video_fps = 30.0
    if video_fps is None or video_fps <= 0:
        video_fps = 30.0
    return dataset, video_fps


def _process_video_frames(
    loader: RdsDataLoader,
    camera_poses,
    camera_model,
    img_w: int,
    img_h: int,
    static_shapes: List[Tuple[int, str, np.ndarray]],
    lane_shapes: List[Tuple[int, str, np.ndarray]],
    dynamic_objects_catalog: Dict[int, str],
    static_objects_catalog: Dict[int, str],
    lanes_objects_catalog: Dict[int, str],
    video_dataset: VideoDataset,
    video_fps: float,
    writer: cv2.VideoWriter,
    target_fps: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, bool],
    Dict[int, bool],
    Dict[int, bool],
]:
    """Processes video frames and collects outputs.

    Args:
        loader: The loader.
        camera_poses: The camera poses.
        camera_model: The camera model.
        img_w: The width of the image.
        img_h: The height of the image.
        static_shapes: The static shapes.
        lane_shapes: The lane shapes.
        dynamic_objects_catalog: The dynamic objects catalog.
        static_objects_catalog: The static objects catalog.
        lanes_objects_catalog: The lane objects catalog.
        video_dataset: The video dataset.
        video_fps: The video FPS.
        writer: The writer.
        target_fps: The target FPS.

    Returns:
        The frames dynamic, frames static, frames lanes, frames static polylines, frames static polygons, dynamic object has projection, static object has projection, lane object has projection.
    """
    dynamic_object_has_projection: Dict[int, bool] = {}
    static_object_has_projection: Dict[int, bool] = {oid: False for oid in static_objects_catalog}
    lane_object_has_projection: Dict[int, bool] = {oid: False for oid in lanes_objects_catalog}

    frames_dynamic: List[Dict[str, Any]] = []
    frames_static: List[Dict[str, Any]] = []
    frames_lanes: List[Dict[str, Any]] = []
    frames_static_polylines: List[Dict[str, Any]] = []
    frames_static_polygons: List[Dict[str, Any]] = []

    overlay_helper = ProjectionOverlayHelper()
    out_frame_idx = 0

    for video_frame_idx, frame in enumerate(video_dataset):
        mapped_idx = round((video_frame_idx * target_fps) / max(video_fps, 1e-6))
        if mapped_idx >= len(camera_poses) or mapped_idx not in camera_poses:
            continue
        pose = camera_poses[mapped_idx]
        (
            frame_dynamic_boxes,
            frame_static_boxes,
            frame_lane_points,
            frame_static_polyline_points,
            frame_static_polygon_vertices,
        ) = _process_frame_data(
            loader,
            mapped_idx,
            pose,
            camera_model,
            img_w,
            img_h,
            static_shapes,
            lane_shapes,
            dynamic_objects_catalog,
            dynamic_object_has_projection,
            static_object_has_projection,
            lane_object_has_projection,
        )
        if not frame_dynamic_boxes:
            continue

        # Append JSON frame payloads
        frame_dynamic_boxes.sort(key=lambda b: int(b.get("object_id", 0)))
        frames_dynamic.append({"index": int(out_frame_idx), "boxes": frame_dynamic_boxes})
        if frame_static_boxes:
            frame_static_boxes.sort(key=lambda b: int(b.get("object_id", 0)))
            frames_static.append({"index": int(out_frame_idx), "boxes": frame_static_boxes})
        if frame_static_polyline_points:
            frame_static_polyline_points.sort(key=lambda b: int(b.get("object_id", 0)))
            frames_static_polylines.append({"index": int(out_frame_idx), "points": frame_static_polyline_points})
        if frame_static_polygon_vertices:
            frame_static_polygon_vertices.sort(key=lambda b: int(b.get("object_id", 0)))
            frames_static_polygons.append({"index": int(out_frame_idx), "polygons": frame_static_polygon_vertices})
        if frame_lane_points:
            frame_lane_points.sort(key=lambda b: int(b.get("object_id", 0)))
            frames_lanes.append({"index": int(out_frame_idx), "points_xy": frame_lane_points})

        # Overlay and write video frame
        if writer:
            frame_np = frame.numpy() if hasattr(frame, "numpy") else frame
            if frame_np.ndim == 3 and frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            if frame_np.dtype == np.float32:
                frame_np = frame_np.astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            if frame_bgr.shape[1] != img_w or frame_bgr.shape[0] != img_h:
                frame_bgr = cv2.resize(frame_bgr, (img_w, img_h))
            overlay_frame = overlay_helper.draw_overlay(
                frame_bgr,
                frame_dynamic_boxes,
                frame_static_boxes,
                lane_polylines=frame_lane_points,
                static_polylines=frame_static_polyline_points,
                polygon_overlays=frame_static_polygon_vertices,
            )
            writer.write(overlay_frame)

        out_frame_idx += 1

    return (
        frames_dynamic,
        frames_static,
        frames_lanes,
        frames_static_polylines,
        frames_static_polygons,
        dynamic_object_has_projection,
        static_object_has_projection,
        lane_object_has_projection,
    )


def export_dynamic_objects(
    out_dir: Path,
    clip_id: str,
    camera_name: str,
    img_w: int,
    img_h: int,
    frames_dynamic: List[Dict[str, Any]],
    dynamic_objects_dict: Dict[str, List[int]],
    fps: int = 10,
):
    """Write dynamic objects JSON.

    Args:
        out_dir: The output directory.
        clip_id: The clip ID.
        camera_name: The camera name.
        img_w: The width of the image.
        img_h: The height of the image.
        frames_dynamic: The frames dynamic.
        dynamic_objects_dict: The dynamic objects dictionary.
        fps: The FPS.
    """
    common = {"camera_name": camera_name, "frame_size": [int(img_w), int(img_h)], "fps": int(fps)}
    dynamic_json = {"objects": dynamic_objects_dict, **common, "frames": frames_dynamic}
    _write_json(out_dir / f"{clip_id}.dynamic.json", dynamic_json)


def export_static_objects(
    out_dir: Path,
    clip_id: str,
    camera_name: str,
    img_w: int,
    img_h: int,
    frames_static: List[Dict[str, Any]],
    frames_static_polylines: List[Dict[str, Any]],
    frames_static_polygons: List[Dict[str, Any]],
    static_objects_dict: Dict[str, List[int]],
    fps: int = 10,
):
    """Write static objects JSON, including polyline and polygon frames."""
    common = {"camera_name": camera_name, "frame_size": [int(img_w), int(img_h)], "fps": int(fps)}
    static_json = {
        "objects": static_objects_dict,
        **common,
        "frames": frames_static,
        "polyline_frames": frames_static_polylines,
        "polygon_frames": frames_static_polygons,
    }
    _write_json(out_dir / f"{clip_id}.static.json", static_json)


def export_lanes(
    out_dir: Path,
    clip_id: str,
    camera_name: str,
    img_w: int,
    img_h: int,
    frames_lanes: List[Dict[str, Any]],
    lane_objects_dict: Dict[str, List[int]],
    fps: int = 10,
):
    """Write lanes objects JSON.

    Args:
        out_dir: The output directory.
        clip_id: The clip ID.
        camera_name: The camera name.
        img_w: The width of the image.
        img_h: The height of the image.
        frames_lanes: The frames lanes.
        lane_objects_dict: The lane objects dictionary.
        fps: The FPS.
    """
    common = {"camera_name": camera_name, "frame_size": [int(img_w), int(img_h)], "fps": int(fps)}
    lanes_json = {"objects": lane_objects_dict, **common, "frames": frames_lanes}
    _write_json(out_dir / f"{clip_id}.lanes.json", lanes_json)


def export_objects(input_data: str, clip_id: str, camera_name: str, output_dir: str, video_path: str):
    """Exports objects from RDS-HQ to JSON and video overlay.

    Args:
        input_data: The input data directory.
        clip_id: The clip ID.
        camera_name: The camera name.
        output_dir: The output directory.
        video_path: The path to the video.
    """
    # 1) Initialize core inputs
    loader = RdsDataLoader(input_data, clip_id)
    camera_poses = loader.get_camera_poses(camera_name)
    camera_model = loader.get_camera_intrinsics(camera_name, "ftheta")
    img_w = loader.CAMERA_RESCALED_RESOLUTION_WIDTH
    img_h = loader.CAMERA_RESCALED_RESOLUTION_HEIGHT

    # 2) Initialize video I/O
    target_fps = 30.0
    output_fps = 10.0
    video_dataset, video_fps = _init_video_dataset(video_path)
    out_dir = Path(output_dir)
    writer = _setup_video_output(out_dir, clip_id, camera_name, img_w, img_h, output_fps)

    # 3) Load static and lane shapes + catalogs
    static_shapes, lane_shapes, static_objects_catalog, lanes_objects_catalog = _load_shapes_and_catalogs(
        input_data, clip_id
    )
    dynamic_objects_catalog: Dict[int, str] = {}

    # 4) Process frames and collect outputs
    (
        frames_dynamic,
        frames_static,
        frames_lanes,
        frames_static_polylines,
        frames_static_polygons,
        dynamic_object_has_projection,
        static_object_has_projection,
        lane_object_has_projection,
    ) = _process_video_frames(
        loader,
        camera_poses,
        camera_model,
        img_w,
        img_h,
        static_shapes,
        lane_shapes,
        dynamic_objects_catalog,
        static_objects_catalog,
        lanes_objects_catalog,
        video_dataset,
        video_fps,
        writer,
        target_fps,
    )

    # 5) Build catalogs and write outputs
    dynamic_objects_dict, static_objects_dict, lane_objects_dict = _build_catalogs(
        dynamic_objects_catalog,
        dynamic_object_has_projection,
        static_objects_catalog,
        static_object_has_projection,
        lanes_objects_catalog,
        lane_object_has_projection,
    )
    export_dynamic_objects(
        out_dir, clip_id, camera_name, img_w, img_h, frames_dynamic, dynamic_objects_dict, fps=int(output_fps)
    )
    export_static_objects(
        out_dir,
        clip_id,
        camera_name,
        img_w,
        img_h,
        frames_static,
        frames_static_polylines,
        frames_static_polygons,
        static_objects_dict,
        fps=int(output_fps),
    )
    export_lanes(out_dir, clip_id, camera_name, img_w, img_h, frames_lanes, lane_objects_dict, fps=int(output_fps))

    # 6) Cleanup
    if writer:
        writer.release()
    if video_dataset is not None:
        video_dataset.release()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Export object projections from RDS-HQ to JSON and video overlay.")
    parser.add_argument("--input_data", "-i", type=str, required=True, help="Root directory of RDS-HQ webdataset")
    parser.add_argument("--clip_id", "-c", type=str, required=True, help="Clip id to process")
    parser.add_argument("--camera_name", "-a", type=str, required=True, help="Camera name to project into")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Input video path for overlay output")
    args = parser.parse_args()
    export_objects(args.input_data, args.clip_id, args.camera_name, args.output_dir, args.video_path)


if __name__ == "__main__":
    main()
