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
Abstract base class for RDS geometry types with projection capabilities.

This module defines the RDSGeometry abstract base class that unifies
Cuboid, Polyline, and Surface geometry classes by requiring a common
get_projected_mask() interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase


class RDSGeometry(ABC):
    """
    Abstract base class for 3D geometry objects with projection capabilities.

    All RDS geometry types (Cuboid, Polyline, Surface) inherit from this class
    and must implement the get_projected_mask() method for projecting the
    geometry into image space.
    """

    @abstractmethod
    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected geometry in image space.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool.
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        raise NotImplementedError("Subclasses must implement get_projected_mask()")
