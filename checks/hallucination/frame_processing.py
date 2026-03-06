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

import sys
from typing import Tuple

import numpy as np

# OpenCV's loader can recurse under PEX layouts unless it replaces sys.path[0].
sys.OpenCV_REPLACE_SYS_PATH_0 = True  # type: ignore[attr-defined]
import cv2  # noqa: E402


def hallucination_counts(
    orig_mask: np.ndarray,
    aug_mask: np.ndarray,
    dist_tol_px: float,
) -> Tuple[int, int]:
    """
    Count hallucinated pixels.

    Args:
        orig_mask: Binary dynamic mask from original video
        aug_mask: Binary dynamic mask from augmented video

    Returns:
        Tuple of (hallucinated_count, total_augmented_dynamic_count)
    """
    orig_dyn = orig_mask > 0
    aug_dyn = aug_mask > 0
    num_aug = int(np.count_nonzero(aug_dyn))

    if num_aug == 0:
        return 0, 0

    # Compute distance transform from original dynamic regions
    src = np.where(orig_dyn, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 3)

    # Identify hallucinated pixels (augmented dynamics far from original dynamics)
    hallucinated = (aug_dyn) & (dist > float(dist_tol_px))

    return int(np.count_nonzero(hallucinated)), num_aug
