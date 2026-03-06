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

import copy
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional, Union

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.ftheta import FThetaCamera
from webdataset import WebDataset

from checks.utils.cuboid import Cuboid
from checks.utils.polyline import Polyline
from checks.utils.surface import Surface


class RdsDataLoader:
    # Mapping from our high-level classes to dataset object types (for filtering objects)
    CLASS_TO_OBJECT_TYPES: ClassVar[dict[str, list[str]]] = {
        "vehicle": ["Automobile", "Car", "Truck", "Vehicle", "Bus", "Train"],
        "pedestrian": ["Pedestrian", "Person"],
        "motorcycle": ["Motorcycle", "Rider", "Cyclist"],
        "bicycle": ["Bicycle", "Rider", "Cyclist"],
        "traffic_light": ["TrafficLight"],
        "traffic_sign": ["TrafficSign"],
        "lane_line": ["LaneLine"],
        "road_boundary": ["RoadBoundary"],
        "wait_line": ["WaitLine"],
        "crosswalk": ["Crosswalk"],
    }

    def __init__(self, dataset_dir: Union[str, Path], clip_id: str) -> None:
        """Initialize the RDS data loader.

        Args:
            dataset_dir: Base directory containing the RDS dataset assets.
            clip_id: Clip identifier (e.g., "<session_uuid>_<...>").
        """
        self.OBJECT_FOLDER = "all_object_info"
        self.POSE_FOLDER = "pose"
        self.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        self.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        self.LANE_LINE_FOLDER = "3d_lanelines"
        self.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        self.WAIT_LINE_FOLDER = "3d_wait_lines"
        self.CROSSWALK_FOLDER = "3d_crosswalks"
        self.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        self.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"
        self.dataset_dir = dataset_dir
        self.clip_id = clip_id
        self.session_uuid = clip_id.split("_")[0]
        self.all_dynamic_objects, self.object_count = self._get_dynamic_object_data_all_frames()
        self.all_static_objects = self._get_static_objects()
        self.CAMERA_RESCALED_RESOLUTION_WIDTH = 1280
        self.CAMERA_RESCALED_RESOLUTION_HEIGHT = 720

    @staticmethod
    def get_sample(url: Union[str, Path]) -> dict[str, Any]:
        """Get a single WebDataset sample from a URL with basic auto-decoding.

        Args:
            url: Path or string URL to a WebDataset tar shard.

        Returns:
            A mapping representing the sample's decoded items keyed by filename-like strings.
        """
        if isinstance(url, Path):
            url = url.as_posix()

        # Use shardshuffle=False to suppress warning and since we only read single samples
        dataset = WebDataset(url, shardshuffle=False).decode()
        try:
            return next(iter(dataset))
        finally:
            if hasattr(dataset, "close"):
                dataset.close()

    def _get_camera_intrinsics_array(
        self,
        camera_name: str,
        model_type: Literal["pinhole", "ftheta"],
    ) -> np.ndarray:
        """Load camera intrinsics as a numpy array for the given camera and model type.

        Args:
            camera_name: Name of the camera.
            model_type: Camera model type, one of "pinhole" or "ftheta".

        Returns:
            Numpy array containing the camera intrinsics for the specified camera.

        Raises:
            ValueError: If ``model_type`` is not supported.
            FileNotFoundError: If the intrinsics tar file is not found or cannot be opened.
            KeyError: If the intrinsics entry for ``camera_name`` is missing in the sample.
        """
        if model_type not in ["pinhole", "ftheta"]:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'pinhole' or 'ftheta'")

        # Determine intrinsics folder based on model type
        intrinsics_folder = self.PINHOLE_INTRINSICS_FOLDER if model_type == "pinhole" else self.FTHETA_INTRINSICS_FOLDER
        intrinsics_file = Path(self.dataset_dir) / intrinsics_folder / f"{self.clip_id}.tar"

        # Check if intrinsics file exists
        if not intrinsics_file.exists():
            raise FileNotFoundError(f"Camera intrinsics file not found: {intrinsics_file}")

        # Load intrinsics data
        try:
            intrinsics_data = self.get_sample(intrinsics_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load camera intrinsics file {intrinsics_file}: {e}") from e

        # Check if camera intrinsics data exists
        camera_key = f"{model_type}_intrinsic.{camera_name}.npy"
        if camera_key not in intrinsics_data:
            available_keys = list(intrinsics_data.keys())
            raise KeyError(
                f"Camera intrinsics not found for camera '{camera_name}' with model '{model_type}'. "
                f"Available keys: {available_keys}"
            )

        intrinsic_this_cam = intrinsics_data[camera_key]
        return intrinsic_this_cam

    def get_camera_intrinsics(
        self,
        camera_name: str,
        model_type: Literal["pinhole", "ftheta"],
        rescaled: bool = True,
    ) -> Optional[FThetaCamera]:
        """Build a camera model for the given camera name.

        Args:
            camera_name: Name of the camera.
            model_type: Camera model type, one of "pinhole" or "ftheta".
            rescaled: Whether to rescale camera intrinsics to the rescaled resolution.

        Returns:
            A ``FThetaCamera`` instance when ``model_type`` is "ftheta"; otherwise ``None``.

        Raises:
            ValueError: If ``model_type`` is invalid.
            FileNotFoundError: If the intrinsics file is missing.
            KeyError: If the camera intrinsics data is not found.
            RuntimeError: If building the camera model fails.
        """
        intrinsic_this_cam = self._get_camera_intrinsics_array(camera_name, model_type)

        camera_model = None

        if model_type == "ftheta":
            try:
                camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device="cpu")
                if rescaled:
                    rescale_ratio = self.CAMERA_RESCALED_RESOLUTION_HEIGHT / camera_model.height
                    camera_model.rescale(rescale_ratio)
            except Exception as e:
                raise RuntimeError(f"Failed to create FTheta camera model for '{camera_name}': {e}") from e

        return camera_model

    def get_camera_poses(self, camera_name: str) -> dict[int, np.ndarray]:
        """Get camera poses for a given camera.

        Note that camera poses are sampled at 30 FPS.

        Args:
            camera_name: Name of the camera.

        Returns:
            Mapping from pose frame index to 4x4 camera-to-world pose matrices.
        """
        pose_file = Path(self.dataset_dir) / self.POSE_FOLDER / f"{self.clip_id}.tar"
        camera_key = f"pose.{camera_name}.npy"
        pose_data = self.get_sample(pose_file)
        pose_data_this_cam = {int(k.split(".")[0]): v for k, v in pose_data.items() if camera_key in k}
        return pose_data_this_cam

    def _get_dynamic_object_data_all_frames(self) -> dict[int, dict]:
        """Load object annotations for all frames in the clip.

        Returns:
            Mapping from frame index to per-frame object annotation structures.
        """
        max_object_id = 0
        object_file = Path(self.dataset_dir) / self.OBJECT_FOLDER / f"{self.clip_id}.tar"
        object_data = self.get_sample(object_file)
        object_data = {int(k.split(".")[0]): v for k, v in object_data.items() if "all_object_info" in k}
        for _, frame_objects in object_data.items():
            for object_id, object_info in frame_objects.items():
                max_object_id = max(max_object_id, int(object_id))
                if "object_to_world" not in object_info or "object_lwh" not in object_info:
                    continue
                object_info["geometry"] = Cuboid(object_info["object_to_world"], object_info["object_lwh"])
                object_info["is_static"] = False
        return object_data, max_object_id

    @staticmethod
    def get_objects_by_class(frame_objects: Dict[str, Any], class_name: str) -> Dict[str, Any]:
        """
        Filter objects of a specific high-level class from frame objects.

        Args:
            frame_objects: Mapping of track_id -> object dict
            class_name: High-level class name

        Returns:
            Subset of frame_objects containing only objects of the requested class
        """
        object_types = RdsDataLoader.CLASS_TO_OBJECT_TYPES.get(class_name, [class_name])
        return {track_id: obj for track_id, obj in frame_objects.items() if obj.get("object_type") in object_types}

    def _get_static_objects(self) -> dict[str, dict[str, Any]]:
        """Get polyline geometry for static objects in the clip.

        Returns:
            Dict mapping static object id to a dictionary with keys:
            - "object_type": Object type (e.g., "LaneLine", "RoadBoundary", "WaitLine")
            - "geometry": Geometry object with function for projecting to image space as a mask
            - "attributes": Attributes of the object (optional)
        """
        obj_types = {
            "LaneLine": self.get_lanelines(),
            "RoadBoundary": self.get_road_boundaries(),
            "WaitLine": self.get_wait_lines(),
            "TrafficLight": self.get_traffic_lights(),
            "TrafficSign": self.get_traffic_signs(),
            "Crosswalk": self.get_crosswalks(),
        }
        static_objects: dict[str, dict[str, Any]] = {}
        for obj_type, label_data in obj_types.items():
            for obj_type_idx, label in enumerate(label_data):
                if "vertices" not in label or "geom_type" not in label:
                    continue

                try:
                    if label["geom_type"] == "polyline":
                        geometry = Polyline(label["vertices"])
                    elif label["geom_type"] == "cuboid":
                        pose, lwh = Cuboid.compute_pose_and_lwh_from_corners(label["vertices"])
                        geometry = Cuboid(pose, lwh)
                    elif label["geom_type"] == "surface":
                        geometry = Surface(label["vertices"])
                    else:
                        continue
                except Exception as e:
                    logging.warning(f"Failed to instantiate {label['geom_type']} for {obj_type}: {e}")
                    continue

                obj = {
                    "object_type": obj_type,
                    "geometry": geometry,
                    "is_static": True,
                    "object_type_index": obj_type_idx,
                }
                if "attributes" in label:
                    obj["attributes"] = label["attributes"]
                static_objects[str(self.object_count)] = obj
                self.object_count += 1

        return static_objects

    def get_object_data_for_frame(self, frame_id: int, include_static: bool = False) -> dict:
        """Get raw object annotations for a frame.

        Args:
            frame_id: Frame index.

        Returns:
            A dictionary containing the per-frame object annotations.
        """
        all_objects = copy.deepcopy(self.all_dynamic_objects.get(frame_id, {}))
        if include_static:
            all_objects.update(copy.deepcopy(self.all_static_objects))
        return all_objects

    def get_fps(self, data_type: str) -> float:
        """Get the FPS for a given data type.

        Args:
            data_type: Type of data ('object' or 'pose')

        Returns:
            FPS value (10.0 for 'object', 30.0 for 'pose')

        Raises:
            ValueError: If data_type is not 'object' or 'pose'
        """
        if data_type == "object":
            return 10.0
        elif data_type == "pose":
            return 30.0
        else:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'object' or 'pose'")

    def get_time_range(self) -> tuple[int, int]:
        """Get the start and end timestamps for the clip in microseconds.

        The start timestamp is extracted from the clip_id (part after first '_').
        The end timestamp is calculated based on the number of frames in all_object_info,
        which is sampled at 10 FPS.

        Returns:
            Tuple of (start_timestamp_us, end_timestamp_us)
        """
        # Extract start timestamp from clip_id (format: uuid_timestamp)
        parts = self.clip_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid clip_id format: {self.clip_id}. Expected format: uuid_timestamp")

        try:
            start_timestamp_us = int(parts[1])
        except ValueError as e:
            raise ValueError(f"Failed to parse timestamp from clip_id '{self.clip_id}': {e}") from e

        # Calculate end timestamp using number of object frames and FPS
        num_frames = len(self.all_dynamic_objects)
        object_fps = self.get_fps("object")
        frame_duration_us = int(1.0 / object_fps * 1e6)
        end_timestamp_us = start_timestamp_us + (num_frames * frame_duration_us)

        return start_timestamp_us, end_timestamp_us

    @staticmethod
    def _extract_shape_attributes(shape3d):
        attributes = {}

        for attr in shape3d.get("attributes", []):
            attr_name = attr.get("name")
            if not attr_name:
                continue

            value = None

            enums_list_container = attr.get("enumsList")
            if isinstance(enums_list_container, dict):
                enums_list = enums_list_container.get("enumsList", [])
                if enums_list:
                    value = enums_list

            if value is None and "text" in attr:
                text_value = attr.get("text")
                if text_value not in (None, ""):
                    value = text_value

            if value is not None:
                attributes[attr_name] = value

        return attributes

    @staticmethod
    def _parse_shape3d_entry(shape3d):
        if not isinstance(shape3d, dict):
            return None

        geom_type = None
        if "polyline3d" in shape3d:
            geom_data = shape3d["polyline3d"]
            geom_type = "polyline"
        elif "surface" in shape3d:
            geom_data = shape3d["surface"]
            geom_type = "surface"
        elif "cuboid3d" in shape3d:
            geom_data = shape3d["cuboid3d"]
            geom_type = "cuboid"
        else:
            return None

        vertices = geom_data.get("vertices", []) if isinstance(geom_data, dict) else []
        if not vertices:
            return None

        label_entry = {
            "vertices": np.asarray(vertices, dtype=np.float32),
            "geom_type": geom_type,
        }

        attributes = RdsDataLoader._extract_shape_attributes(shape3d)
        if attributes:
            label_entry["attributes"] = attributes

        return label_entry

    def get_labels(self, folder: str):
        """Load label data with attributes from a specific folder.

        Args:
            folder: Folder name for label data (e.g., "3d_lanelines", "3d_wait_lines")

        Returns:
            List of dictionaries containing:
                - vertices: numpy array (N, 3) of 3D coordinates
                - attributes: dict of attribute_name -> list of per-vertex values.
                  For label types without attributes, this will be an empty dict.
        """
        tar_path = Path(self.dataset_dir) / folder / f"{self.clip_id}.tar"
        if not tar_path.exists():
            return []

        sample = self.get_sample(tar_path)
        json_key = [k for k in sample if k.endswith(".json")]
        if not json_key:
            return []

        data = sample[json_key[0]]

        if not isinstance(data, dict) or "labels" not in data:
            # Fallback to old vertex extraction for backward compatibility
            return []

        label_data_list = data["labels"]
        extracted_labels = []

        for label_entry in label_data_list:
            if "labelData" not in label_entry:
                continue

            label_data = label_entry["labelData"]
            if "shape3d" not in label_data:
                continue

            shape3d = label_data["shape3d"]
            parsed = RdsDataLoader._parse_shape3d_entry(shape3d)
            if parsed is None:
                continue

            extracted_labels.append(parsed)

        return extracted_labels

    def get_lanelines(self):
        """Get 3D laneline data with attributes (colors, styles, etc.).

        Returns:
            List of lane lines, each is a dictionary containing:
                - vertices: numpy array (N, 3) of 3D coordinates
                - attributes: dict with keys like 'colors', 'styles', storing
                  per-vertex attribute values when available

        Example:
            [
                {
                    "vertices": array([[x1, y1, z1], [x2, y2, z2], ...]),
                    "attributes": {
                        "colors": [...],
                        "styles": [...]
                    }
                },
                ...
            ]
        """
        return self.get_labels(self.LANE_LINE_FOLDER)

    def get_road_boundaries(self):
        """Get 3D road boundary polylines without per-lane attributes."""
        return self.get_labels(self.ROAD_BOUNDARY_FOLDER)

    def get_wait_lines(self):
        """Get 3D wait line polylines without per-lane attributes."""
        return self.get_labels(self.WAIT_LINE_FOLDER)

    def get_crosswalks(self):
        """Get 3D crosswalk surfaces."""
        return self.get_labels(self.CROSSWALK_FOLDER)

    def get_traffic_lights(self):
        """Get 3D traffic light cuboids with attributes when available."""
        return self.get_labels(self.TRAFFIC_LIGHT_FOLDER)

    def get_traffic_signs(self):
        """Get 3D traffic sign cuboids with attributes when available."""
        return self.get_labels(self.TRAFFIC_SIGN_FOLDER)

    def dump_camera_intrinsics_to_csv(self, camera_name: str, output_dir: str):
        """Dump the camera intrinsics for all available models to a CSV file for a given camera name.
        Args:
            camera_name: The camera name.
            output_dir: Directory where the CSV files will be written.

        Returns:
            Mapping from camera type ("pinhole"/"ftheta") to a boolean indicating success for that type.
        """
        camera_types = ["pinhole", "ftheta"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        success = {camera_type: True for camera_type in camera_types}
        for camera_type in camera_types:
            try:
                intrinsic_this_cam = self._get_camera_intrinsics_array(camera_name, camera_type)
                output_file = output_dir / f"{camera_type}_intrinsic.{camera_name}.csv"
                # Flatten the array and save as a single row
                np.savetxt(output_file, intrinsic_this_cam.flatten()[np.newaxis, :], delimiter=",")
            except Exception:
                success[camera_type] = False
        return success

    def dump_camera_frame_egomotion_to_csv(
        self,
        camera_name: str,
        video_fps: float,
        output_dir: Union[str, Path],
    ) -> None:
        """Write camera egomotion for a camera to a CSV aligned to a target FPS.

        Args:
            camera_name: The camera name.
            video_fps: Target video FPS. Poses are sampled at 30 FPS; ``video_fps`` must be in ``(0, 30]`` and cleanly divide 30.
            output_dir: Directory where the CSV will be written.

        Raises:
            ValueError: If ``video_fps`` is not in the valid range or does not cleanly divide 30.
        """
        # Validate FPS: must be > 0 and <= 30
        if video_fps <= 0.0 or video_fps > 30.0:
            raise ValueError("Invalid video FPS, must be greater than 0 and at most 30")
        pose_data_this_cam = self.get_camera_poses(camera_name)
        sorted_pose_data_this_cam = sorted(pose_data_this_cam.items())

        # Prepare data for CSV: each row is [frame_number, timestamp_us, pose.flatten()]; rotation matrix is flattened in column-major order
        rows = []
        delta_time_us = 1.0 / video_fps * 1e6
        # Must cleanly divide 30 FPS
        ratio = 30.0 / video_fps
        frame_skip = int(round(ratio))
        if abs(ratio - frame_skip) > 1e-6:
            raise ValueError("Invalid video FPS, must cleanly divide 30")
        for frame_num, pose in sorted_pose_data_this_cam:
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            # Only dump the camera poses for the video frames, and re-number the frames such that they correspond to the video frames.
            if frame_num % frame_skip == 0:
                video_frame_num = int(frame_num / frame_skip)
                row = (
                    [video_frame_num]
                    + [int(video_frame_num * delta_time_us)]
                    + rotation.flatten(order="F").tolist()
                    + translation.flatten(order="F").tolist()
                )
                rows.append(row)

        # Write to CSV
        output_file = Path(output_dir) / f"pose.{camera_name}.csv"
        # Format: first two values as integer, rest as float
        fmt = ["%d"] + ["%d"] + ["%.8f"] * (len(rows[0]) - 2) if rows else ["%d"]
        header = ",".join(
            [
                "frame",
                "timestamp_us",
                "R00",
                "R10",
                "R20",
                "R01",
                "R11",
                "R21",
                "R02",
                "R12",
                "R22",
                "t0",
                "t1",
                "t2",
            ]
        )
        np.savetxt(output_file, rows, delimiter=",", fmt=fmt, header=header, comments="")
