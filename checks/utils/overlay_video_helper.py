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
Base visualization helper for modules that render overlays into videos.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from checks.utils.video import VideoWriter, convert_video_ffmpeg, get_h264_ffmpeg_args


class OverlayVideoHelper:
    """
    Handles common operations for drawing overlays on videos.

    This class manages video reading/writing and frame extraction so that
    downstream modules can implement custom overlays and add them back to
    the video.
    """

    def __init__(self, clip_id: str, output_dir: str, verbose: bool = False):
        self.clip_id = clip_id
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.naming_suffix = "overlay"

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video writer and capture
        self.video_writer: Optional[VideoWriter] = None
        self.original_video_cap = None
        self.video_output_path: Optional[Path] = None

        # Video properties
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.fps: float = 10.0

    def initialize_video_writer(
        self,
        video_path: str,
        naming_suffix: str = "overlay",
        target_width: int = 1280,
        target_height: int = 720,
        fps: float = 10.0,
    ) -> bool:
        try:
            self.naming_suffix = naming_suffix
            self.original_video_cap = cv2.VideoCapture(video_path)
            if not self.original_video_cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return False

            self.frame_width = target_width
            self.frame_height = target_height
            self.fps = fps

            self.video_output_path = self.output_dir / f"{self.clip_id}.{self.naming_suffix}.mp4"
            self.video_writer = VideoWriter(
                output_path=self.video_output_path,
                fps=self.fps,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                ffmpeg_args=get_h264_ffmpeg_args(include_audio=True),
            )
            if not self.video_writer.open():
                self.logger.error(f"Failed to create video writer: {self.video_output_path}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error initializing video writer: {e}")
            return False

    def get_original_frame(self, frame_idx: int) -> np.ndarray:
        if self.original_video_cap is not None:
            self.original_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, original_frame = self.original_video_cap.read()
            if ret:
                base_image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                base_image = cv2.resize(base_image, (self.frame_width, self.frame_height))
                return base_image

        return np.full((self.frame_height, self.frame_width, 3), 128, dtype=np.uint8)

    def add_legend(self, vis_image: np.ndarray, frame_idx: int, legend_items):
        frame_text = f"Frame: {frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (250, 240, 230)

        is_color_legend = False
        if isinstance(legend_items, list) and len(legend_items) > 0:
            first = legend_items[0]
            if (
                isinstance(first, (tuple, list))
                and len(first) == 2
                and isinstance(first[1], (tuple, list))
                and len(first[1]) == 3
            ):
                is_color_legend = True

        (frame_width, frame_height), _ = cv2.getTextSize(frame_text, font, font_scale, thickness)

        if is_color_legend:
            max_text_width = 0
            line_height = 0
            for name, _ in legend_items:
                (tw, th), _ = cv2.getTextSize(str(name), font, 0.5, 1)
                max_text_width = max(max_text_width, tw)
                line_height = max(line_height, th)
            swatch_size = 10
            swatch_gap = 8
            total_height = frame_height + 16 + len(legend_items) * (line_height + 6)
            total_width = max(frame_width, swatch_size + swatch_gap + max_text_width)
        else:
            legend_height = len(legend_items) * (frame_height + 4)
            max_legend_width = 0
            for legend_text in legend_items:
                (text_width, _), _ = cv2.getTextSize(str(legend_text), font, font_scale, thickness)
                max_legend_width = max(max_legend_width, text_width)
            total_height = frame_height + 16 + legend_height
            total_width = max(frame_width, max_legend_width)

        margin_outer = 14
        text_x = margin_outer
        text_y = frame_height + 44

        bg_pad = 12
        tl = (text_x - bg_pad, text_y - frame_height - bg_pad)
        br = (text_x + total_width + bg_pad, text_y + (total_height - frame_height - 16) + bg_pad)
        overlay = vis_image.copy()
        bg_color = (24, 28, 32)
        cv2.rectangle(overlay, tl, br, bg_color, -1)
        bg_alpha = 0.65
        vis_image = cv2.addWeighted(overlay, bg_alpha, vis_image, 1.0 - bg_alpha, 0)

        cv2.putText(vis_image, frame_text, (text_x, text_y), font, font_scale, color, thickness)

        legend_y = text_y + frame_height + 10
        if is_color_legend:
            small_font_scale = 0.5
            small_thickness = 1
            (_, line_h), _ = cv2.getTextSize("A", font, small_font_scale, small_thickness)
            cy = legend_y
            for name, rgb in legend_items:
                cv2.rectangle(
                    vis_image, (text_x, cy - line_h + 2), (text_x + 10, cy + 2), tuple(int(v) for v in rgb), -1
                )
                cv2.putText(vis_image, str(name), (text_x + 10 + 8, cy), font, small_font_scale, color, small_thickness)
                cy += line_h + 6
        else:
            for i, legend_text in enumerate(legend_items):
                legend_y_pos = legend_y + i * (frame_height + 4)
                cv2.putText(vis_image, str(legend_text), (text_x, legend_y_pos), font, font_scale, color, thickness)

        return vis_image

    def write_frame(self, frame: np.ndarray, frame_idx: int) -> bool:
        if self.video_writer is not None:
            try:
                # Delegate only if this is our VideoWriter class
                if isinstance(self.video_writer, VideoWriter):
                    return self.video_writer.write_frame(frame, frame_idx)

                # Fallback path: behave like legacy base helper
                if frame.shape[:2] != (self.frame_height, self.frame_width):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if frame_bgr.dtype != np.uint8:
                    frame_bgr = frame_bgr.astype(np.uint8)
                self.video_writer.write(frame_bgr)
                return True
            except Exception as e:
                self.logger.error(f"Error writing frame {frame_idx}: {e}")
                return False
        else:
            self.logger.warning(f"Video writer not available for frame {frame_idx}")
            return False

    def save_frame_image(self, frame: np.ndarray, frame_idx: int) -> bool:
        try:
            output_file = self.output_dir / f"{self.naming_suffix}_frame_{frame_idx:04d}.png"
            cv2.imwrite(str(output_file), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            self.logger.error(f"Error saving frame {frame_idx}: {e}")
            return False

    def release(self):
        if self.video_writer:
            try:
                self.video_writer.release()
            finally:
                # Only run a fallback conversion if the writer isn't our VideoWriter
                if self.video_output_path:
                    if not isinstance(self.video_writer, VideoWriter):
                        maybe_video_output_path = convert_video_ffmpeg(
                            input_video_path=self.video_output_path,
                            ffmpeg_args=get_h264_ffmpeg_args(include_audio=True),
                        )
                        if maybe_video_output_path:
                            self.video_output_path = maybe_video_output_path

        if self.original_video_cap:
            self.original_video_cap.release()
