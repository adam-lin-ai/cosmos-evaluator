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
Module for the hallucination checker.

This module handles the detection of hallucinations in the augmented video.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel

from checks.hallucination.frame_processing import hallucination_counts
from checks.utils.config_manager import ConfigManager
from checks.utils.frame_processing import compute_dynamic_mask, ensure_same_size, to_gray
from checks.utils.multistorage import download_if_remote
from checks.utils.types import DynParams

# OpenCV's loader can recurse under PEX layouts unless it replaces sys.path[0].
sys.OpenCV_REPLACE_SYS_PATH_0 = True  # type: ignore[attr-defined]
import cv2  # noqa: E402


class HallucinationResult(BaseModel):
    clip_id: str
    passed: bool
    threshold: float
    score: float
    total_frames: int
    total_hallucinated_dynamic_pixels: int
    total_augmented_dynamic_pixels: int


class HallucinationProcessor:
    """
    Processor for the hallucination checker.
    """

    def __init__(self, params: Dict[str, Any] | None, config_dir: Optional[str] = None, verbose: str = "INFO"):
        """
        Initialize the processor.
        """
        self.config = HallucinationProcessor.get_default_config(config_dir=config_dir).get(
            "metropolis.hallucination", {}
        )
        if params is not None:
            self.config.update(params)
        self.grad_thresh = float(self.config.get("grad_thresh", 10.0))
        self.blur_ksize = int(self.config.get("blur_ksize", 7))
        self.morph_k = int(self.config.get("morph_k", 3))
        self.dist_tol_px = float(self.config.get("dist_tol_px", 7.0))
        self.moving_window_size = int(self.config.get("moving_window_size", 0))
        max_frames_val = self.config.get("max_frames")
        self.max_frames = int(max_frames_val) if max_frames_val is not None else None
        self.threshold = float(self.config.get("threshold", 0.682))
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._setup_logging(verbose)

        self.logger.info("Hallucination processor initialized with the following parameters:")
        self.logger.info(f"- grad_thresh: {self.grad_thresh}")
        self.logger.info(f"- blur_ksize: {self.blur_ksize}")
        self.logger.info(f"- morph_k: {self.morph_k}")
        self.logger.info(f"- dist_tol_px: {self.dist_tol_px}")
        self.logger.info(f"- moving_window_size: {self.moving_window_size}")
        self.logger.info(f"- max_frames: {self.max_frames}")
        self.logger.info(f"- threshold: {self.threshold}")

    def _setup_logging(self, verbose: str) -> None:
        """Setup logging."""
        self.logger = logging.getLogger(__name__)

        # Set logging level based on verbose mode
        log_level = getattr(logging, verbose.upper())
        self.logger.setLevel(log_level)

        # Ensure propagation is enabled
        self.logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.info("Logging level set to: {}".format(verbose))

    def process(self, clip_id: str, original_video_path: str, augmented_video_path: str) -> HallucinationResult:
        """
        Check for hallucinated motion in augmented video.

        Args:
            original_video_path: Path to original video
            augmented_video_path: Path to augmented video

        Returns:
            HallucinationResult: Result of the hallucination check

        The result contains:
            - score: Overall hallucination score [0, 1]
            - total_frames: Number of frames processed
            - total_hallucinated_pixels: Total hallucinated pixels
            - total_augmented_dynamic_pixels: Total dynamic pixels in augmented video
        """
        self.logger.info("Starting hallucination check...")
        self.logger.debug(f"Original video: {original_video_path}")
        self.logger.debug(f"Augmented video: {augmented_video_path}")

        params = DynParams(
            grad_thresh=self.grad_thresh,
            blur_ksize=self.blur_ksize,
            morph_k=self.morph_k,
            dist_tol_px=self.dist_tol_px,
            max_frames=self.max_frames,
        )

        temp_files = []

        try:
            # Download videos if remote
            orig_path = download_if_remote(original_video_path)
            aug_path = download_if_remote(augmented_video_path)

            if orig_path != original_video_path:
                temp_files.append(orig_path)
            if aug_path != augmented_video_path:
                temp_files.append(aug_path)

            # Open videos
            cap_o = cv2.VideoCapture(orig_path)
            cap_a = cv2.VideoCapture(aug_path)

            if not cap_o.isOpened():
                error_msg = f"Failed to open original video downloaded from {original_video_path}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            if not cap_a.isOpened():
                error_msg = f"Failed to open augmented video downloaded from {augmented_video_path}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            # Read first frames
            ok_o, fo = cap_o.read()
            ok_a, fa = cap_a.read()

            if not ok_o or fo is None:
                error_msg = f"Failed to read first frames from original video downloaded from {original_video_path}"
                self.logger.error(error_msg)
                cap_o.release()
                cap_a.release()
                raise Exception(error_msg)

            if not ok_a or fa is None:
                error_msg = f"Failed to read first frames from augmented video downloaded from {augmented_video_path}"
                self.logger.error(error_msg)
                cap_o.release()
                cap_a.release()
                raise Exception(error_msg)

            # Get dimensions and ensure augmented matches original
            h, w = fo.shape[:2]
            fa = ensure_same_size(fa, (h, w))
            prev_o = to_gray(fo)
            prev_a = to_gray(fa)

            # Process frames
            total_h = 0
            total_aug = 0
            frame_count = 1

            while True:
                ok_o, fo = cap_o.read()
                ok_a, fa = cap_a.read()

                if not ok_o or not ok_a:
                    break

                fa = ensure_same_size(fa, (h, w))
                go = to_gray(fo)
                ga = to_gray(fa)

                # Compute dynamic masks
                mo, prev_o = compute_dynamic_mask(prev_o, go, params)
                ma, prev_a = compute_dynamic_mask(prev_a, ga, params)

                # Count hallucinated pixels
                nh, nau = hallucination_counts(mo, ma, self.dist_tol_px)
                total_h += nh
                total_aug += nau

                frame_count += 1

                # Check frame limit
                if self.max_frames is not None and frame_count >= self.max_frames:
                    self.logger.debug(f"Reached max_frames limit: {self.max_frames}")
                    break

            # Release videos
            cap_o.release()
            cap_a.release()

            # Calculate score
            if total_aug == 0:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (float(total_h) / float(total_aug)))

            self.logger.info(f"Hallucination check complete: score={score:.4f}")
            self.logger.debug(f"Frames processed: {frame_count}")
            self.logger.debug(f"Hallucinated pixels: {total_h} / {total_aug}")

            return HallucinationResult(
                clip_id=clip_id,
                passed=score >= self.threshold,
                threshold=self.threshold,
                score=float(score),
                total_frames=frame_count,
                total_hallucinated_dynamic_pixels=total_h,
                total_augmented_dynamic_pixels=total_aug,
            )

        except Exception as e:
            self.logger.error(f"Error during hallucination check: {e}", exc_info=True)
            raise e

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")

    @staticmethod
    def get_default_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the default configuration for hallucination processing.

        Args:
            config_dir: Path to the configuration directory. If None, defaults to
                       checks/config relative to the current working directory.

        Returns:
            Default configuration for hallucination processing loaded from config.yaml
        """
        try:
            config_manager = ConfigManager(config_dir)
            config = config_manager.load_config("config")
            return {"metropolis.hallucination": config["metropolis.hallucination"]}
        except Exception as e:
            logging.error("Error loading default configuration: {}".format(e))
            raise e
