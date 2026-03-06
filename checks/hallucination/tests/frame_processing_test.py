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

import unittest

import numpy as np

from checks.hallucination.frame_processing import hallucination_counts


class TestFrameProcessing(unittest.TestCase):
    """Unit tests for frame processing utilities."""

    def test_hallucination_counts(self) -> None:
        """Test the hallucination_counts function."""
        orig_mask = np.array([[255, 0, 0], [255, 0, 0], [0, 0, 0]], dtype=np.uint8)
        aug_mask = np.array([[255, 0, 0], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)
        num_hallucinated, num_augmented = hallucination_counts(orig_mask, aug_mask, 0.0)
        self.assertEqual(num_hallucinated, 1)
        self.assertEqual(num_augmented, 2)


if __name__ == "__main__":
    unittest.main()
