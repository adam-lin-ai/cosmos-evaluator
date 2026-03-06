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
Tool to analyze obstacle correspondence results across many clips.
"""

import argparse
import glob
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np

from checks.obstacle.results import get_object_type_track_idxs, load_results_from_json

# Increase global font sizes for all plots
plt.rcParams.update(
    {
        "font.size": 14,  # Base font size
        "axes.titlesize": 16,  # Axes title size
        "axes.labelsize": 14,  # X/Y label size
        "xtick.labelsize": 12,  # X tick label size
        "ytick.labelsize": 12,  # Y tick label size
        "legend.fontsize": 12,  # Legend font size
        "figure.titlesize": 18,  # Figure suptitle size
    }
)

# ---- Type aliases ----
ScoreArray = np.ndarray


class AggregatedClip(TypedDict):
    model_names: List[str]
    scores_arrays: List[ScoreArray]
    scores_by_model: Dict[str, ScoreArray]
    object_types: List[Optional[str]]
    track_ids: List[int]


ModelToScoreDict = Dict[str, Dict[str, List[float]]]


def discover_result_json_files(
    input_dirs: List[str],
    model_names: Optional[List[str]] = None,
    prompt_names: Optional[List[str]] = None,
    static_objects: bool = False,
) -> List[str]:
    """Discover obstacle result JSON files under directory trees, filtered by model and prompt names.

    Args:
        input_dirs: List of root directories to search recursively.
        model_names: Optional list of model names to filter by. If provided with prompt_names,
            only returns files in directories matching {model_name}_{prompt_name} pattern.
        prompt_names: Optional list of prompt names to filter by.
        static_objects: If True, only match ``*.static.object.results.json`` files.
            If False, match ``*.obstacle*.results.json`` and ``*.dynamic.object.results.json``.

    Returns:
        List of file paths matching the appropriate pattern, optionally filtered.
    """
    all_files = []
    for input_dir in input_dirs:
        if static_objects:
            patterns = ["**/*.static.object.results.json"]
        else:
            patterns = ["**/*.obstacle*.results.json", "**/*.dynamic.object.results.json"]

        for pattern in patterns:
            files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
            all_files.extend(files)

    # If no filters provided, return all files
    if model_names is None or prompt_names is None:
        return all_files

    # Filter files by model_name and prompt_name pattern
    filtered_files = []
    for file_path in all_files:
        parent_dir = os.path.basename(os.path.dirname(file_path))

        # Check if directory matches any {model_name}_{prompt_name} combination
        for model_name in model_names:
            for prompt_name in prompt_names:
                expected_dirname = f"{model_name}_{prompt_name}"
                if parent_dir == expected_dirname:
                    filtered_files.append(file_path)
                    break
            else:
                continue
            break

    return filtered_files


def discover_all_result_json_files(
    input_dirs: List[str],
    model_names: Optional[List[str]] = None,
    prompt_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Discover both static and dynamic obstacle result JSON files.

    Args:
        input_dirs: List of root directories to search recursively.
        model_names: Optional list of model names to filter by.
        prompt_names: Optional list of prompt names to filter by.

    Returns:
        Tuple of (dynamic_files, static_files) - lists of file paths for each type.
    """
    dynamic_files = discover_result_json_files(input_dirs, model_names, prompt_names, static_objects=False)
    static_files = discover_result_json_files(input_dirs, model_names, prompt_names, static_objects=True)
    return dynamic_files, static_files


def group_files_by_model(
    json_files: Sequence[str],
    model_names: Optional[List[str]] = None,
    prompt_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Group result JSON files by model name.

    Files are grouped by matching their parent directory name against the pattern
    {model_name}_{prompt_name}. Both model_names and prompt_names must be provided.

    Args:
        json_files: Sequence of obstacle results JSON file paths.
        model_names: List of model names. Required for proper grouping.
        prompt_names: List of prompt names. Required for proper grouping.

    Returns:
        Mapping of model name to list of JSON file paths. Returns empty dict if
        model_names or prompt_names is None.
    """
    if model_names is None or prompt_names is None:
        print("Warning: model_names or prompt_names not provided. Cannot group files by model. Returning empty dict.")
        return {}

    model_to_json = {model_name: [] for model_name in model_names}

    for result_json_file in json_files:
        parent_dir = os.path.basename(os.path.dirname(result_json_file))

        # Check if directory matches pattern: {model_name}_{prompt_name}
        for model_name in model_names:
            for prompt_name in prompt_names:
                expected_dirname = f"{model_name}_{prompt_name}"
                if parent_dir == expected_dirname:
                    model_to_json[model_name].append(result_json_file)
                    break

    return model_to_json


def build_clip_to_model_to_json(
    json_files: Sequence[str],
    model_names: Optional[List[str]] = None,
    prompt_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Build mapping of clip to model to JSON file path for cross-model comparisons.

    The ``clip_id`` is created by concatenating the directory name two levels up
    with the prompt name. Directories must match the pattern {model_name}_{prompt_name}.
    Entries where the model is "real" are skipped.

    Args:
        json_files: Sequence of obstacle results JSON file paths.
        model_names: List of model names. Required for proper grouping.
        prompt_names: List of prompt names. Required for proper grouping.

    Returns:
        A nested mapping ``clip_id -> model_name -> json_file``.
    """
    if model_names is None or prompt_names is None:
        print("Warning: model_names or prompt_names not provided. Cannot process files.")
        return {}

    clip_to_model = {}
    for result_json_file in json_files:
        path_parts = os.path.normpath(result_json_file).split(os.sep)
        parent_dir = path_parts[-2]

        # Match directory against pattern: {model_name}_{prompt_name}
        model_name_for_clip = None
        prompt_name_for_clip = None

        for model_name in model_names:
            for prompt_name in prompt_names:
                expected_dirname = f"{model_name}_{prompt_name}"
                if parent_dir == expected_dirname:
                    model_name_for_clip = model_name
                    prompt_name_for_clip = prompt_name
                    break
            if model_name_for_clip:
                break

        if model_name_for_clip is None:
            continue

        # Build clip_id from base directory + prompt_name with underscore separator
        clip_id = path_parts[-3] + "_" + prompt_name_for_clip

        if model_name_for_clip == "real":
            continue
        if clip_id not in clip_to_model:
            clip_to_model[clip_id] = {}
        clip_to_model[clip_id][model_name_for_clip] = result_json_file
    return clip_to_model


def aggregate_clip_scores(
    clip_to_model_to_json: Mapping[str, Mapping[str, str]],
    frame_count: Optional[int] = None,
) -> Dict[str, AggregatedClip]:
    """Align per-model per-track scores to a reference ordering for each clip.

    Args:
        clip_to_model_to_json: Mapping from clip_id to model name to JSON path.
        frame_count: Optional number of frames to average when computing per-track scores.
            If None, all frames are used.

    Returns:
        Mapping from clip_id to an ``AggregatedClip`` with model names, aligned
        score arrays per model, object types aligned to track indices, and track_ids.
    """
    clip_id_to_aggregated_arrays = {}
    for clip_id, model_to_json in clip_to_model_to_json.items():
        if not model_to_json:
            continue
        ordered_models = sorted(model_to_json.keys())

        ref_model = ordered_models[0]
        ref_results = load_results_from_json(model_to_json[ref_model])
        ref_track_ids = ref_results.get("track_ids", [])

        track_id_to_object_type = {}
        if "tracks" in ref_results and ref_results["tracks"] is not None:
            for obj in ref_results["tracks"]:
                tid = obj.get("track_id")
                otype = obj.get("object_type")
                if tid is not None:
                    track_id_to_object_type[tid] = otype
        object_types_by_track_idx = [track_id_to_object_type.get(tid) for tid in ref_track_ids]

        per_model_score_arrays = []
        scores_by_model_name = {}
        for model_name_key in ordered_models:
            results = load_results_from_json(model_to_json[model_name_key])
            score_matrix = results.get("score_matrix")
            model_track_ids = results.get("track_ids", [])

            if score_matrix is None:
                aligned_scores = [np.nan for _ in ref_track_ids]
            else:
                if frame_count is None:
                    per_track_means = np.nanmean(score_matrix, axis=0)
                else:
                    per_track_means = np.nanmean(score_matrix[0:frame_count, :], axis=0)
                tid_to_mean = {tid: per_track_means[idx] for idx, tid in enumerate(model_track_ids)}
                aligned_scores = [tid_to_mean.get(tid, np.nan) for tid in ref_track_ids]

            aligned_scores_arr = np.asarray(aligned_scores, dtype=float)
            per_model_score_arrays.append(aligned_scores_arr)
            scores_by_model_name[model_name_key] = aligned_scores_arr

        clip_id_to_aggregated_arrays[clip_id] = {
            "model_names": ordered_models,
            "scores_arrays": per_model_score_arrays,
            "scores_by_model": scores_by_model_name,
            "object_types": object_types_by_track_idx,
            "track_ids": ref_track_ids,
        }
    return clip_id_to_aggregated_arrays


def init_model_to_score_dict(
    model_to_json_dict: Mapping[str, Sequence[str]],
    object_types: Sequence[str],
) -> ModelToScoreDict:
    """Initialize mapping of model to object type to list of scores.

    Models are ordered alphabetically, with "real" moved to the end if present.

    Args:
        model_to_json_dict: Mapping of model name to list of JSON file paths.
        object_types: Sequence of object type names.

    Returns:
        Ordered mapping of model name to object type to empty list of scores.
    """
    model_to_score = {}
    for model_name in model_to_json_dict:
        model_to_score[model_name] = {obj_type: [] for obj_type in object_types}

    # Sort and move "real" to the end if present
    model_to_score = {k: model_to_score[k] for k in sorted(model_to_score.keys())}
    if "real" in model_to_score:
        real_value = model_to_score.pop("real")
        model_to_score["real"] = real_value
    return model_to_score


def compute_object_type_score_distributions(
    model_to_json_dict: Mapping[str, Sequence[str]],
    object_types: Sequence[str],
    frame_count: Optional[int] = None,
) -> ModelToScoreDict:
    """Compute time-averaged per-track scores grouped by object type for each model.

    Args:
        model_to_json_dict: Mapping of model name to list of JSON file paths.
        object_types: Sequence of object type names to include.
        frame_count: Optional number of frames to average when computing per-track scores.
            If None, all frames are used.

    Returns:
        Mapping of model name to object type to list of per-track mean scores.
    """
    model_to_score_dict = init_model_to_score_dict(model_to_json_dict, object_types)
    for model_name, json_files in model_to_json_dict.items():
        for json_file in json_files:
            result = load_results_from_json(json_file)
            object_type_track_idxs = get_object_type_track_idxs(result)
            score_matrix = result["score_matrix"]
            if score_matrix is None:
                continue
            for obj_type in object_types:
                if obj_type not in object_type_track_idxs:
                    continue
                track_idx_list = object_type_track_idxs[obj_type]
                for track_idx in track_idx_list:
                    if frame_count is None:
                        per_frame_per_track_scores = score_matrix[:, track_idx]
                    else:
                        per_frame_per_track_scores = score_matrix[0:frame_count, track_idx]
                    if np.isnan(per_frame_per_track_scores).all():
                        continue
                    model_to_score_dict[model_name][obj_type].append(np.nanmean(per_frame_per_track_scores))
    return model_to_score_dict


def ensure_output_dir(input_dirs: List[str], output_dir: str | None) -> str:
    """Create and return the plotting output directory path.

    Args:
        input_dirs: List of input directories. The first one is used to default the output location.
        output_dir: Optional explicit output directory.

    Returns:
        Absolute path to the output directory that was created or already existed.
    """
    out_dir = output_dir if output_dir else os.path.join(input_dirs[0], "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_boxplots_for_object_type(
    obj_type: str,
    model_to_score_dict: ModelToScoreDict,
    output_dir: str,
) -> None:
    """Save a boxplot comparing score distributions across models for an object type.

    Args:
        obj_type: Object type name to plot.
        model_to_score_dict: Mapping of model name to per-object-type score lists.
        output_dir: Directory where the PNG will be written.

    Returns:
        None. The image is saved to ``output_dir``.
    """
    data = []
    labels = []
    for model_name, obj_scores in model_to_score_dict.items():
        scores = obj_scores.get(obj_type, [])
        if len(scores) == 0:
            continue
        data.append(scores)
        labels.append(model_name)

    if len(data) == 0:
        print(f"No scores to plot for {obj_type}")
        return

    counts = [len(scores) for scores in data]
    labels_with_counts = [f"{label} (n={count})" for label, count in zip(labels, counts)]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels_with_counts, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Per-model score distribution for {obj_type}")
    plt.ylabel("Average per-track score")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"boxplot_{obj_type.lower()}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved boxplot: {out_path}")


def plot_histograms_for_object_type(
    obj_type: str,
    model_to_score_dict: ModelToScoreDict,
    output_dir: str,
) -> None:
    """Save overlaid histograms comparing models for an object type.

    Args:
        obj_type: Object type name to plot.
        model_to_score_dict: Mapping of model name to per-object-type score lists.
        output_dir: Directory where the PNG will be written.

    Returns:
        None. The image is saved to ``output_dir``.
    """
    data = []
    labels = []
    for model_name, obj_scores in model_to_score_dict.items():
        scores = obj_scores.get(obj_type, [])
        if len(scores) == 0:
            continue
        data.append(scores)
        labels.append(model_name)

    if len(data) == 0:
        return

    counts = [len(scores) for scores in data]
    all_scores = np.concatenate([np.asarray(scores, dtype=float) for scores in data])
    score_min = np.nanmin(all_scores)
    score_max = np.nanmax(all_scores)
    if score_min == score_max:
        score_min -= 0.5
        score_max += 0.5
    bins = np.linspace(score_min, score_max, 50)

    plt.figure(figsize=(12, 6))
    for scores, label, count in zip(data, labels, counts):
        weights = np.ones_like(scores, dtype=float) / float(count)
        plt.hist(
            scores,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            alpha=0.9,
            label=f"{label} (n={count})",
        )
    plt.title(f"Per-model score histogram for {obj_type}")
    plt.xlabel("Average per-track score")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    hist_out_path = os.path.join(output_dir, f"histogram_{obj_type.lower()}.png")
    plt.savefig(hist_out_path, dpi=200)
    plt.close()
    print(f"Saved histogram: {hist_out_path}")


def plot_scatter_grids(
    clip_id_to_aggregated_arrays: Mapping[str, AggregatedClip],
    model_to_score_dict: ModelToScoreDict,
    object_types: Sequence[str],
    output_dir: str,
) -> None:
    """Save n-by-n pairwise scatter grids per object type (excluding "real").

    Args:
        clip_id_to_aggregated_arrays: Mapping of clip_id to aggregated clip data.
        model_to_score_dict: Mapping of model name to per-object-type score lists.
        object_types: Sequence of object type names to include.
        output_dir: Directory where PNGs will be written.

    Returns:
        None. Images are saved to ``output_dir``.
    """
    global_model_names = list(model_to_score_dict.keys())
    if "real" in global_model_names:
        global_model_names.remove("real")
    num_models = len(global_model_names)

    print(f"Scatter grid - Models to plot: {global_model_names}")

    # Debug: Check which models appear in aggregated data
    models_in_agg_data = set()
    for clip_id, agg in clip_id_to_aggregated_arrays.items():
        scores_by_model = agg.get("scores_by_model", {})
        models_in_agg_data.update(scores_by_model.keys())
    print(f"Scatter grid - Models in aggregated data: {sorted(models_in_agg_data)}")

    # Debug: Count clips per model
    model_clip_counts = {m: 0 for m in global_model_names}
    for clip_id, agg in clip_id_to_aggregated_arrays.items():
        scores_by_model = agg.get("scores_by_model", {})
        for m in global_model_names:
            if m in scores_by_model:
                model_clip_counts[m] += 1
    print(f"Scatter grid - Clips per model: {model_clip_counts}")

    for obj_type in object_types:
        fig, axes = plt.subplots(num_models, num_models, figsize=(3.2 * num_models, 3.2 * num_models))
        if num_models == 1:
            axes = np.array([[axes]])

        for i, x_model in enumerate(global_model_names):
            for j, y_model in enumerate(global_model_names):
                ax = axes[j, i]

                xs = []
                ys = []
                clips_with_both = 0
                clips_missing_x = 0
                clips_missing_y = 0
                for _, agg in clip_id_to_aggregated_arrays.items():
                    scores_by_model = agg.get("scores_by_model", {})
                    if x_model not in scores_by_model or y_model not in scores_by_model:
                        if x_model not in scores_by_model:
                            clips_missing_x += 1
                        if y_model not in scores_by_model:
                            clips_missing_y += 1
                        continue
                    clips_with_both += 1
                    obj_types_seq = agg.get("object_types", [])
                    idxs = [k for k, t in enumerate(obj_types_seq) if t == obj_type]
                    if not idxs:
                        continue
                    x_arr = scores_by_model[x_model][idxs]
                    y_arr = scores_by_model[y_model][idxs]
                    for xv, yv in zip(x_arr.tolist(), y_arr.tolist()):
                        if np.isnan(xv) or np.isnan(yv):
                            continue
                        xs.append(xv)
                        ys.append(yv)

                if len(xs) == 0:
                    print(
                        f"  [{obj_type}] ({x_model} vs {y_model}): NO DATA - clips_with_both={clips_with_both}, missing_{x_model}={clips_missing_x}, missing_{y_model}={clips_missing_y}"
                    )
                    ax.set_visible(False)
                    continue

                ax.scatter(xs, ys, s=10, alpha=1.0, edgecolor="none")
                ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

                if j == num_models - 1:
                    ax.set_xlabel(x_model)
                else:
                    ax.set_xticklabels([])
                if i == 0:
                    ax.set_ylabel(y_model)
                else:
                    ax.set_yticklabels([])

        plt.suptitle(f"Per-object scores: model vs model for {obj_type}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        scatter_out_path = os.path.join(output_dir, f"scattergrid_{obj_type.lower()}.png")
        plt.savefig(scatter_out_path, dpi=200)
        plt.close(fig)
        print(f"Saved scatter grid: {scatter_out_path}")


def compute_json_file_to_hallucination_counts_by_class(
    json_files: Sequence[str],
    object_types: Sequence[str],
    only_count_relevant: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Build mapping from JSON file to per-class hallucinated track counts.

    Expects a top-level key ``hallucination_tracks`` which is a list of dicts,
    each with a ``class`` key indicating the object type.

    Args:
        json_files: Sequence of obstacle results JSON file paths.
        object_types: Object type names to include in the output.
        only_count_relevant: If True, only count entries with positive ``relevancy``.

    Returns:
        Mapping from JSON file path to mapping of object type to float count.
    """
    file_to_counts: Dict[str, Dict[str, float]] = {}
    for jf in json_files:
        try:
            result = load_results_from_json(jf)
        except (KeyError, ValueError, OSError) as exc:
            print(f"Failed to load hallucination counts from {jf}: {exc}")
            continue

        counts = {ot: 0.0 for ot in object_types}
        hallu_list = result.get("hallucination_tracks", [])
        if isinstance(hallu_list, list):
            for entry in hallu_list:
                cls = entry.get("class")
                relevancy = entry.get("relevancy", 0)
                if cls in counts and (not only_count_relevant or float(relevancy) > 0):
                    counts[cls] += 1.0
        file_to_counts[jf] = counts
    return file_to_counts


def aggregate_model_hallucination_counts_by_class(
    model_to_json_dict: Mapping[str, Sequence[str]],
    json_file_to_counts: Mapping[str, Mapping[str, float]],
    object_types: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate hallucination counts per class into lists per model.

    Args:
        model_to_json_dict: Mapping of model name to list of JSON file paths.
        json_file_to_counts: Mapping of JSON path to per-class counts.
        object_types: Object type names to include.

    Returns:
        Mapping of model name to object type to list of counts.
    """
    model_to_counts: Dict[str, Dict[str, List[float]]] = {}
    for model_name, json_files in model_to_json_dict.items():
        class_to_vals = {ot: [] for ot in object_types}
        for jf in json_files:
            if jf not in json_file_to_counts:
                continue
            per_class = json_file_to_counts[jf]
            for ot in object_types:
                val = float(per_class.get(ot, 0.0))
                if not np.isnan(val):
                    class_to_vals[ot].append(val)
        model_to_counts[model_name] = class_to_vals
    # Alphabetical with 'real' moved to end if present
    ordered = {k: model_to_counts[k] for k in sorted(model_to_counts.keys())}
    if "real" in ordered:
        real_val = ordered.pop("real")
        ordered["real"] = real_val
    return ordered


def plot_boxplots_for_hallucinations_per_class(
    obj_type: str,
    model_to_counts_by_class: Mapping[str, Mapping[str, Sequence[float]]],
    output_dir: str,
) -> None:
    """Save a boxplot comparing hallucination counts across models for an object type.

    Args:
        obj_type: Object type name to plot.
        model_to_counts_by_class: Mapping of model name to per-object-type lists of counts.
        output_dir: Directory where the PNG will be written.

    Returns:
        None. The image is saved to ``output_dir``.
    """
    data: List[Sequence[float]] = []
    labels: List[str] = []
    for model_name, per_class in model_to_counts_by_class.items():
        vals = per_class.get(obj_type, [])
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(model_name)

    if len(data) == 0:
        print(f"No hallucination counts to plot for {obj_type}")
        return

    counts = [len(vals) for vals in data]
    labels_with_counts = [f"{label} (n={count})" for label, count in zip(labels, counts)]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels_with_counts, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Per-model hallucination count distribution for {obj_type}")
    plt.ylabel("Hallucinated track count")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"hallu_boxplot_{obj_type.lower()}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved boxplot: {out_path}")


def plot_histograms_for_hallucinations_per_class(
    obj_type: str,
    model_to_counts_by_class: Mapping[str, Mapping[str, Sequence[float]]],
    output_dir: str,
) -> None:
    """Save overlaid histograms comparing models for hallucination counts for an object type.

    Args:
        obj_type: Object type name to plot.
        model_to_counts_by_class: Mapping of model name to per-object-type lists of counts.
        output_dir: Directory where the PNG will be written.

    Returns:
        None. The image is saved to ``output_dir``.
    """
    data: List[Sequence[float]] = []
    labels: List[str] = []
    for model_name, per_class in model_to_counts_by_class.items():
        vals = per_class.get(obj_type, [])
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(model_name)

    if len(data) == 0:
        return

    counts = [len(vals) for vals in data]
    all_vals = np.concatenate([np.asarray(vals, dtype=float) for vals in data])
    vmin = np.nanmin(all_vals) if all_vals.size > 0 else 0.0
    vmax = np.nanmax(all_vals) if all_vals.size > 0 else 1.0
    if np.isnan(vmin) or np.isnan(vmax):
        return
    # Use integer-centered bins
    vmin_i = int(np.floor(vmin))
    vmax_i = int(np.ceil(vmax))
    bins = np.arange(vmin_i, vmax_i + 2) - 0.5

    plt.figure(figsize=(12, 6))
    for vals, label, n in zip(data, labels, counts):
        weights = np.ones_like(vals, dtype=float) / float(n)
        plt.hist(
            vals,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            alpha=0.9,
            label=f"{label} (n={n})",
        )
    plt.title(f"Per-model hallucination count histogram for {obj_type}")
    plt.xlabel("Hallucinated track count")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    hist_out_path = os.path.join(output_dir, f"hallu_histogram_{obj_type.lower()}.png")
    plt.savefig(hist_out_path, dpi=200)
    plt.close()
    print(f"Saved histogram: {hist_out_path}")


def plot_stacked_bar_total_hallucinations(
    model_to_counts_by_class: Mapping[str, Mapping[str, Sequence[float]]],
    object_types: Sequence[str],
    output_dir: str,
    static_objects: bool = False,
) -> None:
    """Save a stacked bar chart of total hallucinations per model, stacked by class.

    The height of each stacked segment is the total count of hallucinated objects
    of that type across all clips for the given model.

    Args:
        model_to_counts_by_class: Mapping of model name to per-object-type lists of counts.
        object_types: Sequence of object type names to include.
        output_dir: Directory where the PNG will be written.
        static_objects: If True, append "_static" to filename; otherwise "_dynamic".

    Returns:
        None. The image is saved to ``output_dir``.
    """
    if not model_to_counts_by_class:
        return

    # Preserve model ordering from the input mapping (already sorted with 'real' last)
    model_names = list(model_to_counts_by_class.keys())
    num_models = len(model_names)
    if num_models == 0:
        return

    # Build totals matrix of shape [len(object_types), num_models]
    totals_by_type = []  # list of lists: per-type across models
    for ot in object_types:
        per_model_totals = []
        for m in model_names:
            per_class_vals = model_to_counts_by_class.get(m, {}).get(ot, [])
            per_model_totals.append(
                float(np.nansum(np.asarray(per_class_vals, dtype=float))) if per_class_vals else 0.0
            )
        totals_by_type.append(per_model_totals)

    x = np.arange(num_models)
    width = 0.6

    # Color map for object types
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(object_types))]

    plt.figure(figsize=(max(10, 1.6 * num_models), 6))
    bottoms = np.zeros(num_models, dtype=float)

    for idx, (ot, per_model_totals) in enumerate(zip(object_types, totals_by_type)):
        plt.bar(
            x,
            per_model_totals,
            width,
            bottom=bottoms,
            label=ot,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
        bottoms = bottoms + np.asarray(per_model_totals, dtype=float)

    plt.xticks(x, model_names, rotation=45, ha="right")
    plt.ylabel("Total hallucinated objects")
    plt.title("Total hallucinations by model (stacked by class)")
    plt.legend(title="Class", loc="upper right")
    plt.tight_layout()

    suffix = "static" if static_objects else "dynamic"
    out_path = os.path.join(output_dir, f"hallu_stacked_bar_totals_{suffix}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved stacked bar chart: {out_path}")


def plot_stacked_bar_total_missing_objects(
    model_to_score_dict: Mapping[str, Mapping[str, Sequence[float]]],
    object_types: Sequence[str],
    output_dir: str,
    static_objects: bool = False,
) -> None:
    """Save a stacked bar chart of total missing objects per model, stacked by class.

    A missing object is defined as a track whose mean score equals 0 (i.e., the
    track was observed in the scoring matrix but had zero correspondence).
    Tracks that were never observed (all NaN) are excluded from this count.
    The height of each stacked segment is the total count of missing tracks
    of that type across all clips for the given model.

    Args:
        model_to_score_dict: Mapping of model name to per-object-type score lists.
        object_types: Sequence of object type names to include.
        output_dir: Directory where the PNG will be written.
        static_objects: If True, append "_static" to filename; otherwise "_dynamic".

    Returns:
        None. The image is saved to ``output_dir``.
    """
    if not model_to_score_dict:
        return

    model_names = list(model_to_score_dict.keys())
    num_models = len(model_names)
    if num_models == 0:
        return

    # Count missing (mean == 0) per object type per model
    # Only count tracks with recorded scores (not NaN), where the score is zero
    counts_by_type = []
    for ot in object_types:
        per_model_counts = []
        for m in model_names:
            scores = model_to_score_dict.get(m, {}).get(ot, [])
            if scores:
                arr = np.asarray(scores, dtype=float)
                # Only count zero scores, not NaN (unobserved tracks are already filtered out)
                valid_scores = arr[~np.isnan(arr)]
                missing_count = int(np.sum(np.isclose(valid_scores, 0.0, atol=1e-8)))
            else:
                missing_count = 0
            per_model_counts.append(missing_count)
        counts_by_type.append(per_model_counts)

    x = np.arange(num_models)
    width = 0.6

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(object_types))]

    plt.figure(figsize=(max(10, 1.6 * num_models), 6))
    bottoms = np.zeros(num_models, dtype=float)

    for idx, (ot, per_model_counts) in enumerate(zip(object_types, counts_by_type)):
        plt.bar(
            x,
            per_model_counts,
            width,
            bottom=bottoms,
            label=ot,
            color=colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
        bottoms = bottoms + np.asarray(per_model_counts, dtype=float)

    plt.xticks(x, model_names, rotation=45, ha="right")
    plt.ylabel("Total missing objects")
    plt.title("Total missing objects by model (stacked by class)")
    plt.legend(title="Class", loc="upper right")
    plt.tight_layout()

    suffix = "static" if static_objects else "dynamic"
    out_path = os.path.join(output_dir, f"missing_stacked_bar_totals_{suffix}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved stacked bar chart: {out_path}")


def plot_scatter_grids_for_hallucinations(
    clip_to_model_to_json: Mapping[str, Mapping[str, str]],
    json_file_to_counts_by_class: Mapping[str, Mapping[str, float]],
    object_types: Sequence[str],
    output_dir: str,
) -> None:
    """Save n-by-n pairwise scatter grids per object type for hallucination counts.

    Excludes the "real" model from the grid axes.

    Args:
        clip_to_model_to_json: Mapping of clip_id to model_name to JSON path.
        json_file_to_counts_by_class: Mapping of JSON path to per-class counts.
        object_types: Sequence of object type names to include.
        output_dir: Directory where PNGs will be written.

    Returns:
        None. Images are saved to ``output_dir``.
    """
    # Determine global set of models across all clips (excluding 'real')
    model_set = set()
    for _, model_map in clip_to_model_to_json.items():
        for model_name in model_map.keys():
            if model_name != "real":
                model_set.add(model_name)
    global_model_names = sorted(model_set)
    num_models = len(global_model_names)
    if num_models == 0:
        return

    for obj_type in object_types:
        fig, axes = plt.subplots(num_models, num_models, figsize=(3.2 * num_models, 3.2 * num_models))
        if num_models == 1:
            axes = np.array([[axes]])

        for i, x_model in enumerate(global_model_names):
            for j, y_model in enumerate(global_model_names):
                ax = axes[j, i]

                xs: List[float] = []
                ys: List[float] = []
                for _, model_map in clip_to_model_to_json.items():
                    if x_model not in model_map or y_model not in model_map:
                        continue
                    x_path = model_map[x_model]
                    y_path = model_map[y_model]
                    if x_path not in json_file_to_counts_by_class or y_path not in json_file_to_counts_by_class:
                        continue
                    xv = float(json_file_to_counts_by_class[x_path].get(obj_type, 0.0))
                    yv = float(json_file_to_counts_by_class[y_path].get(obj_type, 0.0))
                    if not (np.isnan(xv) or np.isnan(yv)):
                        xs.append(xv)
                        ys.append(yv)

                if len(xs) == 0:
                    ax.set_visible(False)
                    continue

                ax.scatter(xs, ys, s=10, alpha=1.0, edgecolor="none")
                # Determine equal axes from data with small margins
                vmin = min(min(xs), min(ys))
                vmax = max(max(xs), max(ys))
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
                span = vmax - vmin
                margin = 0.08 * span
                ax_min = vmin - margin
                ax_max = vmax + margin
                ax.plot([ax_min, ax_max], [ax_min, ax_max], "k--", linewidth=1)
                ax.set_xlim(ax_min, ax_max)
                ax.set_ylim(ax_min, ax_max)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

                if j == num_models - 1:
                    ax.set_xlabel(x_model)
                else:
                    ax.set_xticklabels([])
                if i == 0:
                    ax.set_ylabel(y_model)
                else:
                    ax.set_yticklabels([])

        plt.suptitle(f"Hallucination counts: model vs model for {obj_type}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        scatter_out_path = os.path.join(output_dir, f"hallu_scattergrid_{obj_type.lower()}.png")
        plt.savefig(scatter_out_path, dpi=200)
        plt.close(fig)
        print(f"Saved scatter grid: {scatter_out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the obstacle analyzer.

    Returns:
        Parsed ``argparse.Namespace`` with:
          - ``input_dirs``
          - optional ``output_dir``
          - required ``model_names``
          - ``prompt_names`` (from either explicit list or generated from
            ``num_prompt_versions``)
          - optional ``frame_count``
    """
    parser = argparse.ArgumentParser(description="Analyze obstacle correspondence results across many clips")
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the input data directory/directories",
    )
    parser.add_argument("--output_dir", type=str, required=False, help="Path to the output directory")
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to filter and analyze (required for proper grouping)",
    )
    parser.add_argument(
        "--prompt_names",
        type=str,
        nargs="+",
        required=False,
        help="List of prompt names used in directory naming: {model_name}_{prompt_name}. "
        "Either this or --num_prompt_versions must be provided.",
    )
    parser.add_argument(
        "--num_prompt_versions",
        type=int,
        required=False,
        help="Number of prompt versions. Generates prompt names as v0, v1, ..., v{N-1}. "
        "Either this or --prompt_names must be provided.",
    )
    parser.add_argument(
        "--frame_count",
        type=int,
        required=False,
        default=None,
        help="Number of frames to average when computing scores. If not specified, all frames are used.",
    )
    args = parser.parse_args()

    # Validate: exactly one of prompt_names or num_prompt_versions must be provided
    if args.prompt_names is None and args.num_prompt_versions is None:
        parser.error("Either --prompt_names or --num_prompt_versions must be provided.")
    if args.prompt_names is not None and args.num_prompt_versions is not None:
        parser.error("Cannot specify both --prompt_names and --num_prompt_versions.")

    # Generate prompt_names from num_prompt_versions if provided
    if args.num_prompt_versions is not None:
        args.prompt_names = [f"v{i}" for i in range(args.num_prompt_versions)]

    return args


def run_analysis_for_mode(
    json_files: List[str],
    model_names: List[str],
    prompt_names: List[str],
    output_dir: str,
    frame_count: Optional[int],
    static_objects: bool,
) -> None:
    """Run analysis for either static or dynamic object mode.

    Args:
        json_files: List of JSON file paths to analyze.
        model_names: List of model names to filter and analyze.
        prompt_names: List of prompt names used in directory naming.
        output_dir: Directory where the output will be written.
        frame_count: Optional number of frames to average when computing scores.
        static_objects: If True, analyze static objects; otherwise dynamic objects.

    Returns:
        None. Images are written to disk and a brief summary is printed to stdout.
    """
    mode_name = "static" if static_objects else "dynamic"
    print(f"\n{'=' * 60}")
    print(f"Running {mode_name.upper()} object analysis")
    print(f"{'=' * 60}")

    print(f"\nTotal JSON files found (after filtering): {len(json_files)}")
    print(f"Model names to match: {model_names}")
    print(f"Prompt names to match: {prompt_names}\n")

    model_to_json_dict = group_files_by_model(json_files, model_names, prompt_names)
    print("\nFiles grouped by model:")
    for model_name in model_names:
        count = len(model_to_json_dict.get(model_name, []))
        print(f"  {model_name}: {count} files")

    clip_to_model_to_json_dict = build_clip_to_model_to_json(json_files, model_names, prompt_names)
    print(f"\nTotal unique clips found: {len(clip_to_model_to_json_dict)}")

    # Check how many clips have each model
    clips_per_model = {m: 0 for m in model_names}
    for clip_id, model_map in clip_to_model_to_json_dict.items():
        for model_name in model_names:
            if model_name in model_map:
                clips_per_model[model_name] += 1
    print("Clips per model:")
    for model_name in model_names:
        print(f"  {model_name}: {clips_per_model[model_name]} clips")

    # Check for overlapping clips
    print("\nClip overlap analysis:")
    # Show how many clips have results from multiple models
    clips_by_model_count = {}
    for clip_id, model_map in clip_to_model_to_json_dict.items():
        num_models = len(model_map)
        clips_by_model_count[num_models] = clips_by_model_count.get(num_models, 0) + 1
    for num_models in sorted(clips_by_model_count.keys()):
        print(f"  Clips with {num_models} model(s): {clips_by_model_count[num_models]}")

    # Show some example clip_ids for each model
    print("\nExample clip_ids per model (first 3):")
    for model_name in model_names:
        example_clips = []
        for clip_id, model_map in clip_to_model_to_json_dict.items():
            if model_name in model_map:
                example_clips.append(clip_id)
                if len(example_clips) >= 3:
                    break
        print(f"  {model_name}: {example_clips}")

    # Aggregations
    clip_id_to_aggregated_arrays = aggregate_clip_scores(clip_to_model_to_json_dict, frame_count)
    if static_objects:
        object_types = ["Crosswalk", "LaneLine", "TrafficLight", "TrafficSign", "RoadBoundary", "WaitLine"]
        hallucination_object_types = [
            "crosswalk",
            "lane_line",
            "traffic_light",
            "traffic_sign",
            "road_boundary",
            "wait_line",
        ]
    else:
        object_types = ["Car", "Pedestrian", "Truck", "Cyclist"]
        hallucination_object_types = ["vehicle", "pedestrian", "motorcycle", "bicycle"]
    model_to_score_dict = compute_object_type_score_distributions(model_to_json_dict, object_types, frame_count)
    # Hallucination counts by class
    json_file_to_hallu_counts_by_class = compute_json_file_to_hallucination_counts_by_class(
        json_files, hallucination_object_types, only_count_relevant=True
    )

    model_to_hallu_counts_by_class = aggregate_model_hallucination_counts_by_class(
        model_to_json_dict, json_file_to_hallu_counts_by_class, hallucination_object_types
    )

    # Plotting
    for obj_type in object_types:
        plot_boxplots_for_object_type(obj_type, model_to_score_dict, output_dir)
        plot_histograms_for_object_type(obj_type, model_to_score_dict, output_dir)

    for obj_type in hallucination_object_types:
        plot_boxplots_for_hallucinations_per_class(obj_type, model_to_hallu_counts_by_class, output_dir)
        plot_histograms_for_hallucinations_per_class(obj_type, model_to_hallu_counts_by_class, output_dir)

    # Stacked bar: total hallucinations per model, stacked by class
    plot_stacked_bar_total_hallucinations(
        model_to_hallu_counts_by_class,
        hallucination_object_types,
        output_dir,
        static_objects=static_objects,
    )

    # Stacked bar: total missing objects per model, stacked by class
    plot_stacked_bar_total_missing_objects(
        model_to_score_dict,
        object_types,
        output_dir,
        static_objects=static_objects,
    )

    plot_scatter_grids(clip_id_to_aggregated_arrays, model_to_score_dict, object_types, output_dir)
    # Hallucination per-class scatter grids across models
    plot_scatter_grids_for_hallucinations(
        clip_to_model_to_json_dict,
        json_file_to_hallu_counts_by_class,
        hallucination_object_types,
        output_dir,
    )

    print(f"\n{mode_name.upper()} object analysis complete.")


def main() -> None:
    """Entry point to generate analysis plots and summaries from results JSONs.

    Automatically discovers and analyzes both static and dynamic object result files.

    Returns:
        None. Images are written to disk and a brief summary is printed to stdout.
    """
    args = parse_args()

    # Use frame_count from arguments if provided, otherwise None (process all frames)
    frame_count = args.frame_count

    # Discover both static and dynamic files
    dynamic_files, static_files = discover_all_result_json_files(args.input_dirs, args.model_names, args.prompt_names)

    print("\nDiscovered files:")
    print(f"  Dynamic object files: {len(dynamic_files)}")
    print(f"  Static object files: {len(static_files)}")

    # Determine output directory
    output_dir = ensure_output_dir(args.input_dirs, args.output_dir)

    # Run analysis for dynamic objects if files are found
    if dynamic_files:
        run_analysis_for_mode(
            json_files=dynamic_files,
            model_names=args.model_names,
            prompt_names=args.prompt_names,
            output_dir=output_dir,
            frame_count=frame_count,
            static_objects=False,
        )
    else:
        print("\nNo dynamic object result files found. Skipping dynamic analysis.")

    # Run analysis for static objects if files are found
    if static_files:
        run_analysis_for_mode(
            json_files=static_files,
            model_names=args.model_names,
            prompt_names=args.prompt_names,
            output_dir=output_dir,
            frame_count=frame_count,
            static_objects=True,
        )
    else:
        print("\nNo static object result files found. Skipping static analysis.")

    if not dynamic_files and not static_files:
        print("\nNo result files found for either static or dynamic objects.")


if __name__ == "__main__":
    main()
