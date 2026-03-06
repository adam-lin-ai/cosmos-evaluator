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

"""Collection of utility functions specific to SegFormer model."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks

from checks.utils.cv2_video_dataset import VideoDataset
from checks.utils.onnx import get_inference_session
from utils.bazel import get_runfiles_path


class CropTransform:
    """Customized crop."""

    def __init__(self, top: int, left: int, height: int, width: int):
        "Crop Transform wrapper class for compose."
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Returns cropped image."""
        return T.functional.crop(img, self.top, self.left, self.height, self.width)


def get_model_input_shape() -> Tuple[int]:
    """Return input shape required by the model."""
    return (1, 3, 1024, 1820)


def setup_model(model_device: str = "cuda", verbose: bool = False) -> Tuple:
    """Method to setup runtime session and io binding for inference.

    Args:
        model_device: Device to run the model on ("cuda" or "cpu")
        verbose: Enable verbose logging if True
    """
    # Configure ONNX logging
    from checks.utils.onnx import configure_onnx_logging

    configure_onnx_logging(verbose)

    # setup model
    model_path = get_runfiles_path("checks/utils/citysemsegformer.onnx")
    if model_path is None or not os.path.exists(model_path):
        # Get the directory where this module is located
        current_dir = Path(__file__).parent
        model_filename = "citysemsegformer.onnx"
        model_path = current_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        model_path = str(model_path)

    # IMPORTANT: pass model_device explicitly; otherwise it defaults to CPU
    sess = get_inference_session(model_path, model_device=model_device, verbose=verbose)

    # setup io
    io_binding = sess.io_binding()

    # warm up run
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    io_binding.bind_cpu_input(input_name, np.zeros(get_model_input_shape(), dtype=np.float32))
    io_binding.bind_output(output_name, device_type=model_device)
    sess.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    _ = io_binding.copy_outputs_to_cpu()

    # Clear bindings after warm-up to release GPU memory references
    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    return sess, io_binding


def create_cityscapes_label_colormap() -> Tuple[Dict[int, Tuple[int, int, int]], Dict[str, int]]:
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

        https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = {}
    colormap[0] = (128, 64, 128)
    colormap[1] = (244, 35, 232)
    colormap[2] = (70, 70, 70)
    colormap[3] = (102, 102, 156)
    colormap[4] = (190, 153, 153)
    colormap[5] = (153, 153, 153)
    colormap[6] = (250, 170, 30)
    colormap[7] = (220, 220, 0)
    colormap[8] = (107, 142, 35)
    colormap[9] = (152, 251, 152)
    colormap[10] = (70, 130, 180)
    # person = red
    colormap[11] = (255, 0, 0)
    # rider = yellow
    colormap[12] = (255, 255, 0)
    colormap[13] = (0, 0, 142)
    colormap[14] = (0, 0, 70)
    colormap[15] = (0, 60, 100)
    colormap[16] = (0, 80, 100)
    # motorcycle = orange
    colormap[17] = (255, 165, 0)
    # bike = green
    colormap[18] = (0, 255, 0)
    class_labels = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    assert len(class_labels) == len(colormap), "Incorrect mapping for colors!"
    class_label_to_idx = dict(zip(class_labels, range(len(colormap))))
    return colormap, class_label_to_idx


def get_normalization_constants():
    """Per channel constants used to normalize image input."""
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    return (mean, std)


def get_transforms_fn() -> T.Compose:
    """Build preprocessing pipeline to prepare image for inference."""
    return T.Compose(
        [
            T.Resize(get_model_input_shape()[-2:]),
            T.Normalize(*get_normalization_constants()),
        ]
    )


def get_dataloader(video_path: str, start_frame: int = 0) -> DataLoader:
    """Dataloader that produces a minibatch for inference."""
    transforms_fn = get_transforms_fn()

    video_dataset = VideoDataset(video_path, transforms_fn=[transforms_fn], start_frame=start_frame)

    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataloader, video_dataset


def get_masks(mask_prediction: np.ndarray, image_shape: Tuple[int], save_path=None) -> torch.Tensor:
    """Produce a segmentation mask tensor with colors."""
    color_map, _ = create_cityscapes_label_colormap()
    num_classes = len(color_map)
    colors = [color_map[c] for c in range(num_classes)]

    mask_np = np.squeeze(mask_prediction)
    mask_tensor = torch.from_numpy(mask_np)
    masks_list = []
    for i in range(num_classes):
        masks_list.append(mask_tensor.eq(i))
    all_masks_tensor = torch.stack(masks_list)

    # since we have alpha set to 1.0 we don't care about original image for masks
    canvas = torch.zeros(image_shape, dtype=torch.uint8)
    masks_combined = draw_segmentation_masks(canvas, all_masks_tensor, colors=colors, alpha=1.0)
    if save_path is not None:
        # save mask file for postprocessing
        img = T.ToPILImage()(masks_combined)
        img.save(save_path)
    return masks_combined


def reached_end_of_segment(current_frame: int, end_frame: int) -> bool:
    "Indicates whether the end of segment has been reached."
    assert current_frame >= 0
    assert end_frame >= -1, "Invalid end of segment ({}) specified.".format(end_frame)
    return False if end_frame == -1 else current_frame > end_frame


def get_decoded_predictions(contours: List[Tuple[float]], confidences: List[float] = None):
    """Convert cv2 contours to mapnet like data structure."""
    # build prediction structure same as mapnet expect populated only for roadboundary
    # put class labels as 0
    num_predictions = len(contours)
    decoded_predictions = {}
    decoded_predictions["classes"] = [0 for _ in range(num_predictions)]
    # polyline coordinates are (x, y)
    decoded_predictions["polyline_vertices"] = [transform_to_8MP(c).squeeze() for c in contours]
    if confidences is None:
        decoded_predictions["confidences"] = [0.0 for _ in range(num_predictions)]
    else:
        decoded_predictions["confidences"] = [np.round(confidences[i], 4) for i in range(num_predictions)]
    decoded_predictions["track_ids"] = [0 for _ in range(num_predictions)]
    return decoded_predictions


def resample_contours(contours: List[Tuple[float]], num_points: int = 50):
    """Return `num_points` (not necessarily equidistant) along a contour.

    Shorter segments are more closely spaced since they are expected to represent curves.
    """
    resampled = []
    for contour in contours:
        indices = np.round(np.linspace(1, len(contour) - 1, num=num_points)).astype(int)
        resampled.append(contour[indices])
    return resampled


def bbox_iou(boxA, boxB):
    """Computes intersection over union between 2 bboxes in TLBR format."""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union_area = area_A + area_B - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def xywh_to_tlbr(bbox):
    """Convert bbox coordinates from (x, y, width, height) to (top left x, top left y, bottom right x, bottom right y)."""
    assert bbox is not None and len(bbox) == 4
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


def get_bbox(contour: List[Tuple[float]]) -> Tuple[float]:
    """Return TLBR coordinates of bbox bounding a contour."""
    bbox = cv2.boundingRect(contour)
    return xywh_to_tlbr(bbox)


def get_closest_contour(
    ref_contour: List[Tuple[float]],
    contours: List[List[Tuple[float]]],
    iou_threshold: float,
) -> Tuple[List[Tuple[float]], float]:
    """Get contour that is most similar in shape to given contour.

    Contour proximity is also taken into consideration by resttricting search within bounded area.
    """
    ref_bbox = get_bbox(ref_contour)
    best_score = 100.0
    closest = None
    for cnt in contours:
        bbox = get_bbox(cnt)
        iou = bbox_iou(bbox, ref_bbox)
        if iou >= iou_threshold:
            score = cv2.matchShapes(cnt, ref_contour, 1, 0.0)
            # bound the score range - 0.0 is perfect match
            score = min(10.0, score)
            if score < best_score:
                closest = cnt
                best_score = score
    return closest, best_score


def estimate_confidence(score: float) -> float:
    """Heuristic to map contour similarity score from [10, 0] to a value between [0,1]."""
    return 1.0 - 1.0 / (1.0 + np.exp(5.0 - score))


def get_contour_points(im_rgb: np.ndarray, min_length: int = 25, decimate: bool = False) -> Tuple[List]:
    """Extract a sample of contour points."""
    im = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    _, im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY, im)
    # reduce lots of fragmented contours
    dilated_im = cv2.dilate(im.copy(), np.ones((3, 3)), iterations=1)
    contours, hierarchy = cv2.findContours(dilated_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    filtered_contours = []
    # filter out closed contours
    for i, c in enumerate(contours):
        has_child = hierarchy[0][i][2] > 0
        if has_child:
            # has child hence is closed
            continue
        filtered_contours.append(c)

    _CLOSED = False
    filtered_contours = [c for c in filtered_contours if cv2.arcLength(c, _CLOSED) > min_length]
    if decimate:
        approximated_contours = []
        for c in filtered_contours:
            epsilon = 0.1 * np.log(cv2.arcLength(c, _CLOSED))
            approx = cv2.approxPolyDP(c, epsilon, _CLOSED)
            approximated_contours.append(approx)
        return approximated_contours
    return filtered_contours


def transform_to_8MP(points: np.ndarray) -> np.ndarray:
    """Convert 1820x1024 coordinates to map back to 3848x2168.

    1. Scale coordinates by 2
    2. pad top by 120 px (all pixels shift down)
    3. pad left by 104 px (all pixels shift right)
    Args:
        points: 2D points that correspond to 1820x1024 output
    Returns:
        An array with same number of elements as `points` mapped to correspond
        to a 3848x2168 image
    """
    mapped_points = points * 2.0 + np.array([105.0, 121.0])
    return mapped_points
