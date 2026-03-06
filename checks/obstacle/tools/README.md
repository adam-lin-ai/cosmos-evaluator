# Object Check - Tools

Command line tools for analyzing object correspondence quality between AV World Model ground truth and Cosmos-generated video output.

---

## Overview

`analyze_results.py` computes statistics and generates comparison plots from obstacle correspondence result JSON files. These result files can be obtained in two ways:

1. **From batch generation runs** (cloud): Extract metadata from the SDG database, then download results from S3 using `prepare_batch_results.py`
2. **From local obstacle checks**: Run `checks/obstacle/cli.py` locally to generate result files (see obstacle check documentation)

---

## Workflow A: Analyzing Batch Generation Results

### Step 1: Extract Metadata

Use the `extract_metadata` tool to obtain metadata JSON files for your batch runs:

```bash
# Extract metadata for a single batch
dazel run //utils/tools:extract_metadata -- \
  --batch-id batch_20251215_233719_84070f97 \
  --output-dir /path/to/batch_1215 \
  --verbose

# Extract metadata for multiple batches from a file
dazel run //utils/tools:extract_metadata -- \
  --batch-id-list batch_ids.txt \
  --output-dir /path/to/batches \
  --workers 8 \
  --verbose
```

Environment: Requires `SDG_DB_PASSWORD` (set via `~/.cosmos_evaluator/.env` or environment variable).

### Step 2: Prepare Batch Results

Use `prepare_batch_results.py` to download obstacle result JSON files from S3:

```bash
dazel run //checks/obstacle/tools:prepare_batch_results -- \
  --batch /path/to/batch_1215 cosmos-edge \
  --batch /path/to/batch_1211 cosmos-lora \
  --output_dir /path/to/analysis_input \
  --verbose
```

### Step 3: Analyze Results

```bash
dazel run //checks/obstacle/tools:analyze_results -- \
  --input_dirs /path/to/analysis_input \
  --model_names cosmos-edge cosmos-lora \
  --num_prompt_versions 15 \
  --output_dir /path/to/plots
```

---

## Workflow B: Analyzing Local Check Results

If you have result JSON files from running obstacle checks locally (via `checks/obstacle/cli.py`), organize them into the expected directory structure and run `analyze_results.py` directly.

---

## Tool Reference

### `prepare_batch_results.py`

Downloads obstacle result JSON files from S3 based on metadata files and organizes them for analysis.

```bash
usage: prepare_batch_results.py --batch DIRECTORY MODEL_ALIAS [--batch ...]
                                --output_dir OUTPUT_DIR
                                [--dry_run] [--verbose]
```

**Options:**
- `--batch`: Batch directory and model alias pair. Specify multiple times for multiple batches.
- `--output_dir`: Output directory for organized results.
- `--dry_run`: Preview actions without downloading files.
- `--verbose`: Print detailed progress information.

**Notes:**
- Each batch directory should contain metadata files from a single model. The script validates this and reports an error if multiple models are found.
- Metadata files are expected to match `*_metadata.json` pattern.
- The model alias is used for output directory naming (not the full model name from metadata).
- Prompt names are extracted from the `output_uuid` field (e.g., `v0`, `v1`, etc.).
- Already downloaded files are skipped (cached), making re-runs fast.
- AWS credentials are loaded from `~/.aws/.env` if available, or from environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN` and `AWS_DEFAULT_REGION`.

**Output structure:**
```text
output_dir/
└── {clip_id_with_timestamp}/
    └── {model_alias}_{prompt_name}/
        └── *.obstacles.results.json
```

---

### `analyze_results.py`

Analyzes obstacle correspondence results across many clips and models, generating summary plots. The script automatically discovers and analyzes both static and dynamic object result files.

```bash
usage: analyze_results.py --input_dirs INPUT_DIRS [INPUT_DIRS ...]
                          --model_names MODEL_NAMES [MODEL_NAMES ...]
                          (--prompt_names PROMPT_NAMES [...] | --num_prompt_versions N)
                          [--output_dir OUTPUT_DIR]
                          [--frame_count FRAME_COUNT]
```

**Options:**
- `--input_dirs`: One or more root directories containing result JSON files (searched recursively).
- `--model_names`: List of model names to analyze. Must match directory names `{model_name}_{prompt_name}`.
- `--prompt_names`: Explicit list of prompt names. Either this or `--num_prompt_versions` must be provided.
- `--num_prompt_versions`: Number of prompt versions. Generates prompt names as `v0`, `v1`, ..., `v{N-1}`. Either this or `--prompt_names` must be provided.
- `--output_dir`: Where to write plots. Defaults to `INPUT_DIRS[0]/plots`.
- `--frame_count`: Number of frames to average when computing scores. If not specified, all frames are used.

**Notes:**
- The script automatically discovers and analyzes both static and dynamic result files in a single run.
- If only one type of result files is found, only that analysis is performed.
- Files are filtered at discovery: only JSON files in directories matching `{model_name}_{prompt_name}` pattern are processed.
- The `real` model is excluded from pairwise scatter grids.
- "Missing objects" are tracks whose mean score equals 0 (observed but not matched). Tracks that were never observed (all NaN) are excluded from missing counts.
- Hallucination counts only include entries with positive `relevancy` values.

**Static vs Dynamic Objects:**

The script automatically processes both types of objects when their result files are available:

**Dynamic objects** (vehicles, pedestrians, etc.):
- Searches for `*.obstacle*.results.json` and `*.dynamic.object.results.json` files
- Object types: `Car`, `Pedestrian`, `Truck`, `Cyclist`
- Hallucination classes: `vehicle`, `pedestrian`, `motorcycle`, `bicycle`

**Static objects** (road infrastructure):
- Searches for `*.static.object.results.json` files
- Object types: `Crosswalk`, `LaneLine`, `TrafficLight`, `TrafficSign`, `RoadBoundary`, `WaitLine`
- Hallucination classes: `crosswalk`, `lane_line`, `traffic_light`, `traffic_sign`, `road_boundary`, `wait_line`

---

## Output Plots

The script generates plots for both dynamic and static object types when their respective result files are available.

### Correspondence Score Plots
Generated for each object type (dynamic: Car, Pedestrian, Truck, Cyclist; static: Crosswalk, LaneLine, etc.).

- **`boxplot_<ObjectType>.png`**: Per-model score distributions with sample counts.
- **`histogram_<ObjectType>.png`**: Overlaid normalized histograms comparing models.
- **`scattergrid_<ObjectType>.png`**: Pairwise model-vs-model scatter grids.

### Hallucination Plots
Generated for each hallucination class (dynamic: vehicle, pedestrian, etc.; static: crosswalk, lane_line, etc.).

- **`hallu_boxplot_<class>.png`**: Per-model distributions of hallucinated track counts.
- **`hallu_histogram_<class>.png`**: Overlaid normalized histograms of hallucination counts.
- **`hallu_scattergrid_<class>.png`**: Pairwise model-vs-model scatter grids for hallucination counts.
- **`hallu_stacked_bar_totals_dynamic.png`**: Total hallucinations per model, stacked by class (dynamic objects).
- **`hallu_stacked_bar_totals_static.png`**: Total hallucinations per model, stacked by class (static objects).

### Missing Objects Plot
- **`missing_stacked_bar_totals_dynamic.png`**: Total missing objects per model, stacked by class (dynamic objects).
- **`missing_stacked_bar_totals_static.png`**: Total missing objects per model, stacked by class (static objects).

---

## Understanding the Scatter Grid Plots

The scatter grid (`scattergrid_*.png`) provides **pairwise model comparisons**.

**Structure:** An n×n grid where n = number of models, one grid per object type.

**Each subplot (row=j, col=i):**
- X-axis: Per-track mean scores from model `i`
- Y-axis: Per-track mean scores from model `j`
- Each point: One tracked obstacle from the same clip (clip id + timestamp + prompt)

**Interpretation:**
- Points on diagonal (y=x): Both models scored similarly
- Points above diagonal: Y-axis model scored higher
- Points below diagonal: X-axis model scored higher
- Points at (0,0): Both models failed to match
- Points at (1,1): Both models had perfect correspondence
