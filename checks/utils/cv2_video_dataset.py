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

"""Single camera mp4 dataset that uses open cv backend."""

import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class VideoDataset(IterableDataset):
    """VideoDataset class."""

    def __init__(
        self,
        video_path,
        transforms_fn=None,
        skip=1,
        start_frame=None,
    ):
        """Constructor.

        Args:
            video_path: Path to the input mp4 video. The filename should be the camera name
                where it was recorded. Example filename: `camera_front_wide_120fov.mp4`
            transforms_fn: a function pipeline that is called to transform the frame before yielding
                 it to downstream task
            skip: Value for skipping frames in an video
            start_frame: start yielding frames from this position (1-based index)
        Raises:
            FileNotFoundError: If the `video_path` does not exist.
            ValueError: If the `video_path` does not have `.mp4` extension.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"{video_path} does not exist!")

        _, filename = os.path.split(video_path)
        _, ext = os.path.splitext(filename)
        if ext != ".mp4":
            raise ValueError(f"{video_path} is not a mp4 file!")
        self._video_path = video_path

        video = cv2.VideoCapture(video_path)
        self.number_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames_per_second = int(video.get(cv2.CAP_PROP_FPS))
        self.video = video

        logger.info("number_frames: {}, frames_per_second: {}".format(self.number_frames, self.frames_per_second))
        self._transforms_fn = [] if transforms_fn is None else transforms_fn
        assert isinstance(self._transforms_fn, list)
        if skip < 1:
            raise ValueError("`skip` must be a positive integer (got %s)" % skip)
        self._skip = skip
        self.start_frame = 0
        if start_frame is not None:
            assert start_frame >= 0
            # Jump to frame (0-based)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.start_frame = start_frame

    def __iter__(self):
        """Load and process a sample.

        Returns:
            Iterator object that yields processed image.
        """
        for frame_number in range(self.start_frame, self.number_frames):
            success, image = self.video.read()
            if not success or image is None:
                logger.error("{} failed in video.read() in frame_number: {}".format(self._video_path, frame_number))
                # return in generator is equivalent to raising StopIteration
                return

            if frame_number % self._skip != 0:
                continue
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # convert from H,W,C to C,H,W
            image = np.transpose(image, (2, 0, 1))
            # convert to float32
            image = image.astype(np.float32)

            image_tensor = torch.from_numpy(image)
            for fn in self._transforms_fn:
                image_tensor = fn(image_tensor)
            yield image_tensor

    def __len__(self):
        """Returns the length of the dataset."""
        return (self.number_frames - self.start_frame) // self._skip

    def release(self):
        """Release video capture object."""
        self.video.release()
