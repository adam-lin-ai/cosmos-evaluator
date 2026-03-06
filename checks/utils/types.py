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

"""Types used by the hallucination checkers."""

from typing import Any

from pydantic import BaseModel, Field


class DynParams(BaseModel):
    """Parameters for the hallucination checker frame processors."""

    grad_thresh: float = 10.0
    blur_ksize: int = 7
    morph_k: int = 3
    dist_tol_px: float = 7.0
    max_frames: int | None = None
    expand_scale: float = 2.0
    resize_factor: float = 1.0


class ScoreValue(BaseModel):
    """A score title and value for a rule."""

    key: str
    value: float


class FrameMetadata(BaseModel):
    """Metadata for a single frame."""

    index: int
    timestamp_ms: int
    width: int
    height: int
    camera_id: str | None = None


class Event(BaseModel):
    """An event in the video."""

    type: str
    start_frame: int
    end_frame: int
    attributes: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    """A original video and augmented video pair to be checked."""

    sample_id: str
    # Path to media; in full system this may be a URI
    media_path: str | None = None
    # Optional path to the original/unaltered video corresponding to this sample
    original_video: str | None = None
    # Optional path to the input directory containing the object detection files
    input_dir: str | None = None
    # Optional metadata produced by upstream components
    scenario_description: str | None = None
    frames: list[FrameMetadata] | None = None
    events: list[Event] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RuleFinding(BaseModel):
    """The results of a single check of a single sample."""

    rule_id: str
    stage: str
    passed: bool
    # One or more scores computed by the rule. For a single score, emit a one-element list.
    scores: list[ScoreValue] = Field(default_factory=list)
    severity: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
