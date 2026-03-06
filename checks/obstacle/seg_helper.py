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
Segmentation helper module for obstacle correspondence processing.

Owns the segmentation model setup and offers utilities to:
- Process video frames to segmentation outputs
- Extract per-class binary masks, using a centralized mapping from our high-level
  class names to SegFormer label names
- Filter frame objects by our high-level classes, using a centralized mapping from
  high-level classes to dataset object types
"""

import logging
from typing import ClassVar

import numpy as np
import torch

from checks.utils.segformer import (
    create_cityscapes_label_colormap,
    get_masks,
    setup_model,
)

logger = logging.getLogger(__name__)


class SegHelper:
    """
    Handles instance segmentation processing for obstacle correspondence.

    This class manages the segmentation model, processes video frames,
    and generates segmentation masks.
    """

    # Mapping from our high-level classes to SegFormer class labels
    CLASS_TO_SEGFORMER_CLASSES: ClassVar[dict[str, list[str]]] = {
        "vehicle": ["car", "truck", "bus"],
        "pedestrian": ["person"],
        "motorcycle": ["motorcycle", "rider"],
        "bicycle": ["bicycle", "rider"],
    }

    # Cached colormap data (lazy initialization)
    _cached_colormap = None
    _cached_class_label_to_idx = None

    def __init__(self, model_device: str = "cuda"):
        """
        Initialize the segmentation processor.

        Args:
            model_device: Device to run the model on ("cuda" or "cpu")
        """
        self.model_device = model_device
        self.ort_session = None
        self.io_binding = None
        self.input_name = None
        self.output_name = None
        self.colormap = None
        self.class_label_to_idx = None

        # Initialize the model
        self._setup_model()

    def _setup_model(self):
        """Set up the segmentation model and related components."""
        # Setup ONNX model with reduced logging
        self.ort_session, self.io_binding = setup_model(model_device=self.model_device, verbose=False)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        logger.info(f"SegHelper initialized with device={self.model_device}")

        # Setup colormap for visualization
        self.colormap, self.class_label_to_idx = create_cityscapes_label_colormap()

    @staticmethod
    def get_class_color(class_name: str) -> np.ndarray:
        """
        Get the RGB color for a SegFormer class label.

        This static method resolves colors using the canonical Cityscapes colormap
        and label index mapping, without requiring an instance of SegHelper.

        Args:
            class_name: Exact SegFormer class label (e.g., "road", "person", "car").

        Returns:
            RGB color as numpy array
        """
        # Lazy initialization of cached colormap
        if SegHelper._cached_colormap is None:
            SegHelper._cached_colormap, SegHelper._cached_class_label_to_idx = create_cityscapes_label_colormap()

        class_label_to_idx = SegHelper._cached_class_label_to_idx

        if class_name not in class_label_to_idx:
            raise ValueError(f"Unknown class name: {class_name}")
        class_idx = class_label_to_idx[class_name]
        return np.array(SegHelper._cached_colormap[class_idx], dtype=np.uint8)

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame through the segmentation model.

        Args:
            frame: Input frame as torch tensor

        Returns:
            segmentation_masks: Full segmentation result as torch tensor
        """
        # Convert frame to numpy for ONNX inference
        image_np_array = frame.cpu().numpy()

        # Clear previous bindings to release GPU memory from prior inference
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        # Bind inputs and outputs fresh for this frame
        self.io_binding.bind_cpu_input(self.input_name, image_np_array)
        self.io_binding.bind_output(self.output_name, device_type=self.model_device)

        # Run inference
        self.ort_session.run_with_iobinding(self.io_binding)

        # Synchronize to ensure GPU operations complete before copying
        self.io_binding.synchronize_outputs()
        output = self.io_binding.copy_outputs_to_cpu()

        # Get segmentation masks
        return get_masks(output[0], frame.shape[1:])

    def get_class_mask(self, segmentation_masks: torch.Tensor, class_name: str) -> np.ndarray:
        """
        Extract binary mask for a specific high-level class from segmentation result.

        Args:
            segmentation_masks: Segmentation result from process_frame
            class_name: High-level class name (e.g., "vehicle", "pedestrian")

        Returns:
            Binary mask for the specified class or class group
        """
        # Convert segmentation masks to numpy (H, W, C) for color comparisons
        seg_mask_np = segmentation_masks.permute(1, 2, 0).cpu().numpy()

        # Initialize combined mask
        combined_mask = np.zeros(seg_mask_np.shape[:2], dtype=bool)

        # Add each SegFormer class corresponding to the high-level class
        segformer_classes = SegHelper.CLASS_TO_SEGFORMER_CLASSES.get(class_name, [])
        for seg_class in segformer_classes:
            try:
                target_color = np.array(self.get_class_color(seg_class), dtype=np.uint8)
                class_mask = np.all(seg_mask_np == target_color, axis=-1)
                combined_mask = combined_mask | class_mask
            except ValueError:
                # Skip if segmentation class doesn't exist
                continue

        return combined_mask

    def resize_masks(self, masks: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        Resize segmentation masks to target dimensions.

        Args:
            masks: Input masks as torch tensor
            target_height: Target height
            target_width: Target width

        Returns:
            Resized masks
        """
        resized_masks = (
            torch.nn.functional.interpolate(
                masks.unsqueeze(0).float(), size=(target_height, target_width), mode="nearest"
            )
            .squeeze(0)
            .type(masks.dtype)
        )

        return resized_masks

    # selection helpers removed; scoring and hallucination refined union now own their logic
