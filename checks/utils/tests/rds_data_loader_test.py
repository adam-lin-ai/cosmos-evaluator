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

"""Unit tests for rds_data_loader module."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from checks.utils.rds_data_loader import RdsDataLoader


class TestRdsDataLoader(unittest.TestCase):
    """Test cases for RdsDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.temp_dir)
        self.clip_id = "uuid_1234567890"
        self.session_uuid = "uuid"

        # Create directory structure
        (self.dataset_dir / "all_object_info").mkdir(parents=True)
        (self.dataset_dir / "pose").mkdir(parents=True)
        (self.dataset_dir / "3d_traffic_lights").mkdir(parents=True)
        (self.dataset_dir / "3d_traffic_signs").mkdir(parents=True)
        (self.dataset_dir / "3d_lanelines").mkdir(parents=True)
        (self.dataset_dir / "3d_road_boundaries").mkdir(parents=True)
        (self.dataset_dir / "3d_wait_lines").mkdir(parents=True)
        (self.dataset_dir / "3d_crosswalks").mkdir(parents=True)
        (self.dataset_dir / "pinhole_intrinsic").mkdir(parents=True)
        (self.dataset_dir / "ftheta_intrinsic").mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_webdataset_sample(self, data: dict) -> dict:
        """Helper to create a mock WebDataset sample."""
        return data

    def _create_mock_object_data(self, num_frames: int = 3) -> dict:
        """Create mock object data for testing."""
        object_data = {}
        for frame_idx in range(num_frames):
            frame_objects = {
                "0": {
                    "object_type": "Car",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([4.0, 2.0, 1.5]),
                },
                "1": {
                    "object_type": "Pedestrian",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([0.5, 0.5, 1.7]),
                },
            }
            object_data[f"{frame_idx}.all_object_info.json"] = frame_objects
        return object_data

    def _create_mock_pose_data(self, num_frames: int = 5) -> dict:
        """Create mock pose data for testing."""
        pose_data = {}
        for frame_idx in range(num_frames):
            pose = np.eye(4)
            pose[:3, 3] = [frame_idx * 0.1, 0.0, 0.0]
            pose_data[f"{frame_idx}.pose.camera_front.npy"] = pose
        return pose_data

    def _create_mock_intrinsics_data(self, camera_name: str, model_type: str) -> dict:
        """Create mock intrinsics data for testing."""
        if model_type == "pinhole":
            intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        else:  # ftheta
            intrinsic = np.random.rand(10, 10)  # Mock ftheta intrinsics
        return {f"{model_type}_intrinsic.{camera_name}.npy": intrinsic}

    def _create_mock_label_data(self) -> dict:
        """Create mock label data for testing."""
        return {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
                                },
                                "attributes": [
                                    {
                                        "name": "colors",
                                        "enumsList": {"enumsList": ["red", "yellow", "white"]},
                                    }
                                ],
                            }
                        }
                    },
                    {
                        "labelData": {
                            "shape3d": {
                                "cuboid3d": {
                                    "vertices": [
                                        [0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [0.0, 1.0, 1.0],
                                    ],
                                },
                                "attributes": [{"name": "type", "text": "solid"}],
                            }
                        }
                    },
                ]
            }
        }

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_static_method(self, mock_webdataset):
        """Test static get_sample method."""
        mock_sample = {"key1": "value1", "key2": np.array([1, 2, 3])}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample("test.tar")

        self.assertEqual(result, mock_sample)
        mock_webdataset.assert_called_once_with("test.tar", shardshuffle=False)

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_with_path_object(self, mock_webdataset):
        """Test get_sample with Path object."""
        mock_sample = {"key": "value"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample(Path("test.tar"))

        self.assertEqual(result, mock_sample)
        mock_webdataset.assert_called_once_with("test.tar", shardshuffle=False)

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_closes_dataset(self, mock_webdataset):
        """Test get_sample closes dataset if it has close method."""
        mock_sample = {"key": "value"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_dataset.close = Mock()
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample("test.tar")

        self.assertEqual(result, mock_sample)
        mock_dataset.close.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    @patch.object(RdsDataLoader, "_get_dynamic_object_data_all_frames")
    @patch.object(RdsDataLoader, "_get_static_objects")
    def test_init(self, mock_get_static, mock_get_dynamic, mock_get_sample):
        """Test RdsDataLoader initialization."""
        mock_get_dynamic.return_value = ({}, 0)
        mock_get_static.return_value = {}

        loader = RdsDataLoader(self.dataset_dir, self.clip_id)

        self.assertEqual(loader.dataset_dir, self.dataset_dir)
        self.assertEqual(loader.clip_id, self.clip_id)
        self.assertEqual(loader.session_uuid, self.session_uuid)
        self.assertEqual(loader.CAMERA_RESCALED_RESOLUTION_WIDTH, 1280)
        self.assertEqual(loader.CAMERA_RESCALED_RESOLUTION_HEIGHT, 720)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_pinhole(self, mock_get_sample):
        """Test _get_camera_intrinsics_array for pinhole model."""
        camera_name = "camera_front"
        intrinsics_data = self._create_mock_intrinsics_data(camera_name, "pinhole")
        mock_get_sample.return_value = intrinsics_data

        # Create tar file
        tar_path = self.dataset_dir / "pinhole_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        result = loader._get_camera_intrinsics_array(camera_name, "pinhole")

        self.assertIsInstance(result, np.ndarray)
        mock_get_sample.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_ftheta(self, mock_get_sample):
        """Test _get_camera_intrinsics_array for ftheta model."""
        camera_name = "camera_front"
        intrinsics_data = self._create_mock_intrinsics_data(camera_name, "ftheta")
        mock_get_sample.return_value = intrinsics_data

        tar_path = self.dataset_dir / "ftheta_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        result = loader._get_camera_intrinsics_array(camera_name, "ftheta")

        self.assertIsInstance(result, np.ndarray)

    def test_get_camera_intrinsics_array_invalid_model_type(self):
        """Test _get_camera_intrinsics_array with invalid model type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        with self.assertRaises(ValueError) as context:
            loader._get_camera_intrinsics_array("camera_front", "invalid")
        self.assertIn("Invalid model_type", str(context.exception))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_file_not_found(self, mock_get_sample):
        """Test _get_camera_intrinsics_array when file doesn't exist."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        with self.assertRaises(FileNotFoundError):
            loader._get_camera_intrinsics_array("camera_front", "pinhole")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_missing_camera(self, mock_get_sample):
        """Test _get_camera_intrinsics_array when camera not found."""
        mock_get_sample.return_value = {"other_camera.npy": np.array([1, 2, 3])}

        tar_path = self.dataset_dir / "pinhole_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        with self.assertRaises(KeyError) as context:
            loader._get_camera_intrinsics_array("camera_front", "pinhole")
        self.assertIn("Camera intrinsics not found", str(context.exception))

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta(self, mock_get_array, mock_ftheta_camera):
        """Test get_camera_intrinsics for ftheta model."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_camera_instance = Mock()
        mock_camera_instance.height = 1440
        mock_ftheta_camera.from_numpy.return_value = mock_camera_instance

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 720

        result = loader.get_camera_intrinsics("camera_front", "ftheta", rescaled=True)

        self.assertEqual(result, mock_camera_instance)
        mock_ftheta_camera.from_numpy.assert_called_once_with(mock_intrinsic, device="cpu")
        mock_camera_instance.rescale.assert_called_once()

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta_not_rescaled(self, mock_get_array, mock_ftheta_camera):
        """Test get_camera_intrinsics for ftheta model without rescaling."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_camera_instance = Mock()
        mock_ftheta_camera.from_numpy.return_value = mock_camera_instance

        loader = RdsDataLoader.__new__(RdsDataLoader)

        result = loader.get_camera_intrinsics("camera_front", "ftheta", rescaled=False)

        self.assertEqual(result, mock_camera_instance)
        mock_camera_instance.rescale.assert_not_called()

    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_pinhole(self, mock_get_array):
        """Test get_camera_intrinsics for pinhole model returns None."""
        mock_intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        mock_get_array.return_value = mock_intrinsic

        loader = RdsDataLoader.__new__(RdsDataLoader)

        result = loader.get_camera_intrinsics("camera_front", "pinhole")

        self.assertIsNone(result)

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta_runtime_error(self, mock_get_array, mock_ftheta_camera):
        """Test get_camera_intrinsics raises RuntimeError when camera creation fails."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_ftheta_camera.from_numpy.side_effect = Exception("Camera creation failed")

        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(RuntimeError) as context:
            loader.get_camera_intrinsics("camera_front", "ftheta")
        self.assertIn("Failed to create FTheta camera model", str(context.exception))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_poses(self, mock_get_sample):
        """Test get_camera_poses method."""
        camera_name = "camera_front"
        pose_data = self._create_mock_pose_data(3)
        mock_get_sample.return_value = pose_data

        tar_path = self.dataset_dir / "pose" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.POSE_FOLDER = "pose"

        result = loader.get_camera_poses(camera_name)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        for frame_idx in range(3):
            self.assertIn(frame_idx, result)
            self.assertEqual(result[frame_idx].shape, (4, 4))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_dynamic_object_data_all_frames(self, mock_get_sample):
        """Test get_dynamic_object_data_all_frames method."""
        object_data = self._create_mock_object_data(2)
        mock_get_sample.return_value = object_data

        tar_path = self.dataset_dir / "all_object_info" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.OBJECT_FOLDER = "all_object_info"

        result, max_object_id = loader._get_dynamic_object_data_all_frames()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertEqual(max_object_id, 1)
        for frame_idx in range(2):
            self.assertIn(frame_idx, result)
            for obj_id, obj_info in result[frame_idx].items():
                self.assertIn("geometry", obj_info)
                self.assertFalse(obj_info["is_static"])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_dynamic_object_data_missing_fields(self, mock_get_sample):
        """Test get_dynamic_object_data_all_frames with missing fields."""
        object_data = {
            "0.all_object_info.json": {
                "0": {"object_type": "Car"},  # Missing object_to_world and object_lwh
                "1": {
                    "object_type": "Pedestrian",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([0.5, 0.5, 1.7]),
                },
            }
        }
        mock_get_sample.return_value = object_data

        tar_path = self.dataset_dir / "all_object_info" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.OBJECT_FOLDER = "all_object_info"

        result, max_object_id = loader._get_dynamic_object_data_all_frames()

        # Object 0 should be skipped, only object 1 should have geometry
        self.assertIn(0, result)
        self.assertIn("1", result[0])
        self.assertIn("geometry", result[0]["1"])

    def test_get_objects_by_class_vehicle(self):
        """Test get_objects_by_class for vehicle class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "Pedestrian"},
            "2": {"object_type": "Truck"},
            "3": {"object_type": "Motorcycle"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "vehicle")

        self.assertEqual(len(result), 2)
        self.assertIn("0", result)
        self.assertIn("2", result)
        self.assertNotIn("1", result)
        self.assertNotIn("3", result)

    def test_get_objects_by_class_pedestrian(self):
        """Test get_objects_by_class for pedestrian class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "Pedestrian"},
            "2": {"object_type": "Person"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "pedestrian")

        self.assertEqual(len(result), 2)
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_get_objects_by_class_unknown_class(self):
        """Test get_objects_by_class for unknown class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "UnknownType"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "unknown_class")

        # Should filter by exact match when class not in mapping
        self.assertEqual(len(result), 0)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_geometry(self, mock_get_labels):
        """Test get_static_geometry method."""
        mock_get_labels.side_effect = [
            [  # LaneLine
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
                    "geom_type": "polyline",
                }
            ],
            [  # RoadBoundary
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                    "geom_type": "polyline",
                }
            ],
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        result = loader._get_static_objects()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two polylines
        for obj_id, obj_info in result.items():
            self.assertTrue(obj_info["is_static"])
            self.assertIn("object_type", obj_info)
            self.assertIn("geometry", obj_info)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_geometry_with_cuboid(self, mock_get_labels):
        """Test get_static_geometry with cuboid geometry."""
        # Create cuboid vertices (8 corners)
        cuboid_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [  # TrafficLight
                {
                    "vertices": cuboid_vertices,
                    "geom_type": "cuboid",
                }
            ],
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        result = loader._get_static_objects()

        self.assertEqual(len(result), 1)
        obj_info = list(result.values())[0]
        self.assertEqual(obj_info["object_type"], "TrafficLight")
        self.assertIsNotNone(obj_info["geometry"])

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_geometry_skips_invalid_labels(self, mock_get_labels):
        """Test get_static_geometry skips labels without vertices or geom_type."""
        mock_get_labels.side_effect = [
            [
                {"vertices": np.array([[0, 0, 0], [1, 1, 0]]), "geom_type": "polyline"},  # Valid
                {"vertices": np.array([[0, 0, 0]])},  # Missing geom_type
                {"geom_type": "polyline"},  # Missing vertices
                {},  # Missing both
            ],
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        result = loader._get_static_objects()

        # Should only include the valid label
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_geometry_with_attributes(self, mock_get_labels):
        """Test get_static_geometry preserves attributes."""
        mock_get_labels.side_effect = [
            [
                {
                    "vertices": np.array([[0, 0, 0], [1, 1, 0]]),
                    "geom_type": "polyline",
                    "attributes": {"color": "yellow"},
                }
            ],
            [],
            [],
            [],
            [],
            [],
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        result = loader._get_static_objects()

        obj_info = list(result.values())[0]
        self.assertIn("attributes", obj_info)
        self.assertEqual(obj_info["attributes"]["color"], "yellow")

    def test_get_object_data_for_frame(self):
        """Test get_object_data_for_frame method."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.all_dynamic_objects = {0: {"0": {"object_type": "Car"}}, 1: {"1": {"object_type": "Pedestrian"}}}
        loader.all_static_objects = {"100": {"object_type": "LaneLine", "is_static": True}}

        # Test without static
        result = loader.get_object_data_for_frame(0, include_static=False)
        self.assertEqual(len(result), 1)
        self.assertIn("0", result)

        # Test with static
        result = loader.get_object_data_for_frame(0, include_static=True)
        self.assertEqual(len(result), 2)
        self.assertIn("0", result)
        self.assertIn("100", result)

        # Test missing frame
        result = loader.get_object_data_for_frame(999, include_static=False)
        self.assertEqual(len(result), 0)

    def test_get_fps_object(self):
        """Test get_fps for object data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        result = loader.get_fps("object")
        self.assertEqual(result, 10.0)

    def test_get_fps_pose(self):
        """Test get_fps for pose data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        result = loader.get_fps("pose")
        self.assertEqual(result, 30.0)

    def test_get_fps_invalid(self):
        """Test get_fps with invalid data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        with self.assertRaises(ValueError) as context:
            loader.get_fps("invalid")
        self.assertIn("Invalid data_type", str(context.exception))

    def test_get_time_range(self):
        """Test get_time_range method."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "uuid_1000000"
        loader.all_dynamic_objects = {0: {}, 1: {}, 2: {}}  # 3 frames

        start_us, end_us = loader.get_time_range()

        self.assertEqual(start_us, 1000000)
        # 3 frames at 10 FPS = 0.3 seconds = 300000 microseconds
        self.assertEqual(end_us, 1000000 + 300000)

    def test_get_time_range_invalid_clip_id(self):
        """Test get_time_range with invalid clip_id format."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "invalid"
        loader.all_dynamic_objects = {}

        with self.assertRaises(ValueError) as context:
            loader.get_time_range()
        self.assertIn("Invalid clip_id format", str(context.exception))

    def test_get_time_range_invalid_timestamp(self):
        """Test get_time_range with non-numeric timestamp."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "test_uuid_not_a_number"
        loader.all_dynamic_objects = {}

        with self.assertRaises(ValueError) as context:
            loader.get_time_range()
        self.assertIn("Failed to parse timestamp", str(context.exception))

    def test_extract_shape_attributes_with_enums(self):
        """Test _extract_shape_attributes with enumsList."""
        shape3d = {
            "attributes": [
                {
                    "name": "colors",
                    "enumsList": {"enumsList": ["red", "yellow", "white"]},
                },
                {"name": "styles", "enumsList": {"enumsList": ["solid", "dashed"]}},
            ]
        }

        result = RdsDataLoader._extract_shape_attributes(shape3d)

        self.assertEqual(result["colors"], ["red", "yellow", "white"])
        self.assertEqual(result["styles"], ["solid", "dashed"])

    def test_extract_shape_attributes_with_text(self):
        """Test _extract_shape_attributes with text values."""
        shape3d = {
            "attributes": [
                {"name": "type", "text": "solid"},
                {"name": "material", "text": "concrete"},
            ]
        }

        result = RdsDataLoader._extract_shape_attributes(shape3d)

        self.assertEqual(result["type"], "solid")
        self.assertEqual(result["material"], "concrete")

    def test_extract_shape_attributes_empty(self):
        """Test _extract_shape_attributes with empty attributes."""
        shape3d = {"attributes": []}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_extract_shape_attributes_missing_name(self):
        """Test _extract_shape_attributes with missing name."""
        shape3d = {"attributes": [{"text": "value"}]}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_extract_shape_attributes_none_text(self):
        """Test _extract_shape_attributes with None text."""
        shape3d = {"attributes": [{"name": "attr", "text": None}]}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_parse_shape3d_entry_polyline(self):
        """Test _parse_shape3d_entry with polyline3d."""
        shape3d = {
            "polyline3d": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            }
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        self.assertEqual(result["geom_type"], "polyline")
        assert_array_almost_equal(result["vertices"], np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]))

    def test_parse_shape3d_entry_cuboid(self):
        """Test _parse_shape3d_entry with cuboid3d."""
        shape3d = {
            "cuboid3d": {
                "vertices": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
            }
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        self.assertEqual(result["geom_type"], "cuboid")
        self.assertEqual(len(result["vertices"]), 8)

    def test_parse_shape3d_entry_surface(self):
        """Test _parse_shape3d_entry with surface."""
        shape3d = {"surface": {"vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]}}

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        self.assertEqual(result["geom_type"], "surface")

    def test_parse_shape3d_entry_invalid(self):
        """Test _parse_shape3d_entry with invalid input."""
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry(None))
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry("not a dict"))
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry({}))

    def test_parse_shape3d_entry_empty_vertices(self):
        """Test _parse_shape3d_entry with empty vertices."""
        shape3d = {"polyline3d": {"vertices": []}}
        result = RdsDataLoader._parse_shape3d_entry(shape3d)
        self.assertIsNone(result)

    def test_parse_shape3d_entry_with_attributes(self):
        """Test _parse_shape3d_entry with attributes."""
        shape3d = {
            "polyline3d": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            },
            "attributes": [{"name": "color", "text": "yellow"}],
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        self.assertIn("attributes", result)
        self.assertEqual(result["attributes"]["color"], "yellow")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels(self, mock_get_sample):
        """Test get_labels method."""
        label_data = self._create_mock_label_data()
        mock_get_sample.return_value = label_data

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["geom_type"], "polyline")
        self.assertEqual(result[1]["geom_type"], "cuboid")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_file_not_found(self, mock_get_sample):
        """Test get_labels when tar file doesn't exist."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("nonexistent_folder")

        self.assertEqual(result, [])
        mock_get_sample.assert_not_called()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_no_json_key(self, mock_get_sample):
        """Test get_labels when no JSON key is found."""
        mock_get_sample.return_value = {"other_file.npy": np.array([1, 2, 3])}

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_invalid_data_structure(self, mock_get_sample):
        """Test get_labels with invalid data structure."""
        mock_get_sample.return_value = {"labels.json": "not a dict"}

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_missing_label_data(self, mock_get_sample):
        """Test get_labels with missing labelData."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {"other_field": "value"},  # Missing labelData
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                                }
                            }
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        # Should only include the label with labelData
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_missing_shape3d(self, mock_get_sample):
        """Test get_labels with missing shape3d."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "other_field": "value"  # Missing shape3d
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_skips_invalid_parsed_entries(self, mock_get_sample):
        """Test get_labels skips entries that fail to parse."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [],  # Empty vertices should return None
                                }
                            }
                        }
                    },
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                                }
                            }
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        # Should only include the valid entry
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_lanelines(self, mock_get_labels):
        """Test get_lanelines method."""
        mock_get_labels.return_value = [{"vertices": np.array([[0, 0, 0]]), "geom_type": "polyline"}]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_lanelines()

        mock_get_labels.assert_called_once_with("3d_lanelines")
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_road_boundaries(self, mock_get_labels):
        """Test get_road_boundaries method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_road_boundaries()

        mock_get_labels.assert_called_once_with("3d_road_boundaries")
        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_wait_lines(self, mock_get_labels):
        """Test get_wait_lines method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_wait_lines()

        mock_get_labels.assert_called_once_with("3d_wait_lines")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_crosswalks(self, mock_get_labels):
        """Test get_crosswalks method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_crosswalks()

        mock_get_labels.assert_called_once_with("3d_crosswalks")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_traffic_lights(self, mock_get_labels):
        """Test get_traffic_lights method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_traffic_lights()

        mock_get_labels.assert_called_once_with("3d_traffic_lights")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_traffic_signs(self, mock_get_labels):
        """Test get_traffic_signs method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_traffic_signs()

        mock_get_labels.assert_called_once_with("3d_traffic_signs")

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_dump_camera_intrinsics_to_csv_success(self, mock_get_array, mock_savetxt):
        """Test dump_camera_intrinsics_to_csv with successful dump."""
        mock_intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        mock_get_array.return_value = mock_intrinsic

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        success = loader.dump_camera_intrinsics_to_csv("camera_front", str(output_dir))

        self.assertTrue(success["pinhole"])
        self.assertTrue(success["ftheta"])
        self.assertEqual(mock_savetxt.call_count, 2)

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_dump_camera_intrinsics_to_csv_failure(self, mock_get_array, mock_savetxt):
        """Test dump_camera_intrinsics_to_csv with failure."""
        mock_get_array.side_effect = [np.array([[1, 2], [3, 4]]), FileNotFoundError("Not found")]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        success = loader.dump_camera_intrinsics_to_csv("camera_front", str(output_dir))

        self.assertTrue(success["pinhole"])
        self.assertFalse(success["ftheta"])

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "get_camera_poses")
    def test_dump_camera_frame_egomotion_to_csv(self, mock_get_poses, mock_savetxt):
        """Test dump_camera_frame_egomotion_to_csv method."""
        pose_data = {}
        for frame_idx in range(6):  # 6 frames at 30 FPS
            pose = np.eye(4)
            pose[:3, 3] = [frame_idx * 0.1, 0.0, 0.0]
            pose_data[frame_idx] = pose
        mock_get_poses.return_value = pose_data

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        loader.dump_camera_frame_egomotion_to_csv("camera_front", 10.0, str(output_dir))

        # Should write CSV with frames sampled at 10 FPS (every 3rd frame from 30 FPS)
        mock_savetxt.assert_called_once()
        call_args = mock_savetxt.call_args
        rows = call_args[0][1]  # Second argument is the data
        self.assertEqual(len(rows), 2)  # 6 frames / 3 = 2 frames at 10 FPS

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_too_high(self):
        """Test dump_camera_frame_egomotion_to_csv with FPS > 30."""
        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(ValueError) as context:
            loader.dump_camera_frame_egomotion_to_csv("camera_front", 31.0, self.temp_dir)
        self.assertIn("Invalid video FPS", str(context.exception))

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_zero(self):
        """Test dump_camera_frame_egomotion_to_csv with FPS <= 0."""
        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(ValueError) as context:
            loader.dump_camera_frame_egomotion_to_csv("camera_front", 0.0, self.temp_dir)
        self.assertIn("Invalid video FPS", str(context.exception))

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_not_divisible(self):
        """Test dump_camera_frame_egomotion_to_csv with FPS that doesn't divide 30."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        with patch.object(RdsDataLoader, "get_camera_poses") as mock_get_poses:
            mock_get_poses.return_value = {0: np.eye(4)}

            with self.assertRaises(ValueError) as context:
                loader.dump_camera_frame_egomotion_to_csv("camera_front", 7.0, self.temp_dir)
            self.assertIn("Invalid video FPS", str(context.exception))

    @patch("checks.utils.rds_data_loader.logging")
    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_objects_handles_geometry_instantiation_exception(self, mock_get_labels, mock_logging):
        """Test get_static_objects catches and logs exceptions during geometry instantiation."""
        # First label has invalid vertices that will cause Polyline to raise
        # Second label is valid and should still be processed
        mock_get_labels.side_effect = [
            [
                {
                    "vertices": np.array([[0.0, 0.0]]),  # Invalid: only 2D points, may cause issue
                    "geom_type": "polyline",
                },
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
                    "geom_type": "polyline",
                },
            ],
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        # Mock Polyline to raise on first call (invalid vertices) but succeed on second
        with patch("checks.utils.rds_data_loader.Polyline") as mock_polyline:
            mock_polyline.side_effect = [ValueError("Invalid vertices"), Mock()]

            result = loader._get_static_objects()

            # Should have logged a warning for the failed instantiation
            mock_logging.warning.assert_called_once()
            warning_call = mock_logging.warning.call_args[0][0]
            self.assertIn("Failed to instantiate", warning_call)
            self.assertIn("polyline", warning_call)
            self.assertIn("LaneLine", warning_call)

            # Should still have processed the second valid label
            self.assertEqual(len(result), 1)
            self.assertEqual(list(result.values())[0]["object_type"], "LaneLine")

    @patch("checks.utils.rds_data_loader.logging")
    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_objects_handles_cuboid_exception(self, mock_get_labels, mock_logging):
        """Test get_static_objects catches and logs exceptions during Cuboid instantiation."""
        # Invalid cuboid vertices that will cause compute_pose_and_lwh_from_corners to fail
        invalid_cuboid_vertices = np.array([[0.0, 0.0, 0.0]])  # Too few vertices for a cuboid

        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [
                {
                    "vertices": invalid_cuboid_vertices,
                    "geom_type": "cuboid",
                },
            ],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        with patch("checks.utils.rds_data_loader.Cuboid") as mock_cuboid:
            mock_cuboid.compute_pose_and_lwh_from_corners.side_effect = ValueError("Invalid cuboid vertices")

            result = loader._get_static_objects()

            # Should have logged a warning
            mock_logging.warning.assert_called_once()
            warning_call = mock_logging.warning.call_args[0][0]
            self.assertIn("Failed to instantiate", warning_call)
            self.assertIn("cuboid", warning_call)
            self.assertIn("TrafficLight", warning_call)

            # Should return empty since the only label failed
            self.assertEqual(len(result), 0)

    @patch("checks.utils.rds_data_loader.logging")
    @patch.object(RdsDataLoader, "get_labels")
    def test_get_static_objects_handles_surface_exception(self, mock_get_labels, mock_logging):
        """Test get_static_objects catches and logs exceptions during Surface instantiation."""
        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [
                {
                    "vertices": np.array([[0.0, 0.0, 0.0]]),  # Single point, invalid for surface
                    "geom_type": "surface",
                },
            ],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        with patch("checks.utils.rds_data_loader.Surface") as mock_surface:
            mock_surface.side_effect = RuntimeError("Cannot create surface from single point")

            result = loader._get_static_objects()

            # Should have logged a warning
            mock_logging.warning.assert_called_once()
            warning_call = mock_logging.warning.call_args[0][0]
            self.assertIn("Failed to instantiate", warning_call)
            self.assertIn("surface", warning_call)
            self.assertIn("Crosswalk", warning_call)

            # Should return empty since the only label failed
            self.assertEqual(len(result), 0)

    def test_get_object_data_for_frame_does_not_modify_class_members(self):
        """Test that get_object_data_for_frame does not expose internal state to mutation."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.all_dynamic_objects = {
            0: {
                "0": {"object_type": "Car", "score": 0.9},
                "1": {"object_type": "Pedestrian", "score": 0.8},
            },
        }
        loader.all_static_objects = {
            "100": {"object_type": "LaneLine", "is_static": True, "tags": ["a", "b"]},
            "101": {"object_type": "RoadBoundary", "is_static": True, "tags": ["c"]},
        }

        # Snapshot the original state for later comparison
        import copy

        original_dynamic = copy.deepcopy(loader.all_dynamic_objects)
        original_static = copy.deepcopy(loader.all_static_objects)

        # --- dynamic-only call ---
        result_dynamic = loader.get_object_data_for_frame(0, include_static=False)

        # Mutate the returned dict: add a key, modify a nested value, delete a key
        result_dynamic["new_key"] = {"object_type": "Truck"}
        result_dynamic["0"]["score"] = 0.0
        result_dynamic["0"]["extra"] = "injected"
        del result_dynamic["1"]

        # The underlying class members must be unchanged
        self.assertEqual(loader.all_dynamic_objects, original_dynamic)
        self.assertEqual(loader.all_static_objects, original_static)

        # --- include_static=True call ---
        result_with_static = loader.get_object_data_for_frame(0, include_static=True)

        # Mutate the returned static entries
        result_with_static["100"]["object_type"] = "MUTATED"
        result_with_static["100"]["tags"].append("MUTATED")
        result_with_static["101"]["new_field"] = 42
        del result_with_static["0"]

        # The underlying class members must still be unchanged
        self.assertEqual(loader.all_dynamic_objects, original_dynamic)
        self.assertEqual(loader.all_static_objects, original_static)

    def test_class_to_object_types_mapping(self):
        """Test CLASS_TO_OBJECT_TYPES class variable."""
        self.assertIn("vehicle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("pedestrian", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("motorcycle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("bicycle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("traffic_light", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("traffic_sign", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("lane_line", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("road_boundary", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("wait_line", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("crosswalk", RdsDataLoader.CLASS_TO_OBJECT_TYPES)

        # Verify vehicle types
        vehicle_types = RdsDataLoader.CLASS_TO_OBJECT_TYPES["vehicle"]
        self.assertIn("Car", vehicle_types)
        self.assertIn("Truck", vehicle_types)


if __name__ == "__main__":
    unittest.main()
