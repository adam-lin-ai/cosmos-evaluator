# Obstacle Check

The Obstacle check evaluates whether dynamic objects (vehicles, pedestrians, motorcycles, bicycles) in a Cosmos-generated video appear at the positions specified by the RDS-HQ ground truth annotations. It uses semantic segmentation to measure how well detected objects in the generated video correspond to their expected locations from the driving dataset.

## How It Works

The check uses a **SegFormer** semantic segmentation model to analyze dynamic objects — vehicles, pedestrians, motorcycles, and bicycles — in a driving scene. It requires the generated video and an RDS-HQ dataset as inputs.

For each frame, the check:
1. Projects 3D object cuboids from the RDS-HQ ground truth into the camera's image space
2. Runs the segmentation model on the generated video frame
3. Measures the overlap between the projected ground truth and the detected segments
4. Produces a per-object, per-frame correspondence score (0.0 to 1.0)

The check also includes a **hallucination detector** that identifies false positive objects — segments in the generated video that don't correspond to any ground truth object.

## Model Setup

### SegFormer Model (Dynamic Processor)

The dynamic processor requires the CitySemsegFormer ONNX model for semantic segmentation. Download it from the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer):

1. Navigate to the [CitySemsegFormer model page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer?version=deployable_onnx_v1.0)
2. Download the `deployable_onnx_v1.0` zip file
3. Extract the `.onnx` file from the zip and save it as `checks/utils/citysemsegformer.onnx`

```bash
unzip deployable_onnx_v1.0.zip -d /tmp/citysemsegformer
cp /tmp/citysemsegformer/*.onnx checks/utils/citysemsegformer.onnx
```

Without this file, the dynamic obstacle processor will raise a `FileNotFoundError` at runtime. All other checks and services remain unaffected.

## Data Inputs

| Input | Description |
|-------|-------------|
| RDS-HQ dataset | Directory containing ground truth annotations, camera poses, and intrinsics in [WebDataset](https://github.com/webdataset/webdataset) `.tar` format |
| Generated video | The Cosmos-generated camera video to evaluate (MP4) |

### RDS-HQ Dataset Structure

The RDS-HQ dataset directory should have the following layout:

```
dataset_dir/
├── all_object_info/               # Dynamic object annotations
│   └── {clip_id}.tar
├── pose/                          # Camera poses (30 FPS)
│   └── {clip_id}.tar
├── pinhole_intrinsic/             # Pinhole camera intrinsics
│   └── {clip_id}.tar              #   Contains: pinhole_intrinsic.{camera_name}.npy
├── ftheta_intrinsic/              # Fisheye camera intrinsics (if applicable)
│   └── {clip_id}.tar              #   Contains: ftheta_intrinsic.{camera_name}.npy
├── 3d_traffic_lights/             # Static traffic light cuboids
│   └── {clip_id}.tar
├── 3d_traffic_signs/              # Static traffic sign cuboids
│   └── {clip_id}.tar
├── 3d_lanelines/                  # Lane line polylines
│   └── {clip_id}.tar
├── 3d_road_boundaries/            # Road boundary polylines
│   └── {clip_id}.tar
├── 3d_wait_lines/                 # Wait line polylines
│   └── {clip_id}.tar
└── 3d_crosswalks/                 # Crosswalk surfaces
    └── {clip_id}.tar
```

Each `.tar` file uses the [WebDataset](https://github.com/webdataset/webdataset) format.

#### Clip ID

The `clip_id` is used to locate the correct `.tar` files in each RDS-HQ subdirectory (e.g., `all_object_info/{clip_id}.tar`). It must match the base name of the `.tar` files in the dataset.

In the standard RDS-HQ format, the clip ID follows the pattern `{session_uuid}_{start_timestamp_us}` (e.g., `0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000`). The session UUID is extracted from the portion before the first underscore, and the start timestamp (in microseconds) is parsed from the portion after it. The timestamp is used to derive the clip's time range.

## Usage

### CLI

The CLI operates on local filesystem paths only — all inputs (RDS-HQ dataset, video file) must be available on disk.

First, extract the sample RDS-HQ dataset:

```bash
unzip checks/sample_data/cosmos_public/rds_hq.zip -d checks/sample_data/cosmos_public/
```

Run the obstacle check directly from the command line:

```bash
dazel run //checks/obstacle:run -- local \
  --input_data checks/sample_data/cosmos_public/rds_hq/ \
  --clip_id 01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000 \
  --camera_name camera_front_wide_120fov \
  --video_path checks/sample_data/cosmos_public/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4 \
  --output_dir output_dir
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_data` | Yes | — | Path to the RDS-HQ dataset directory |
| `--clip_id` | Yes | — | Clip identifier in the format `{session_uuid}_{start_timestamp_us}` (e.g., `0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000`). Used to locate the correct `.tar` files in each RDS-HQ subdirectory and to derive the session UUID and clip time range. |
| `--camera_name` | Yes | — | Camera stream name (e.g., `camera_front_wide_120fov`) |
| `--video_path` | Yes | — | Path to the generated video file (MP4) |
| `--output_dir` | Yes | — | Directory to write results, logs, and visualizations |
| `--model_device` | No | `cuda` | Device for model inference: `cuda` or `cpu` |
| `--trial` | No | None | Process only the first N frames (useful for debugging) |
| `--target-fps` | No | `30.0` | Target FPS for processing |
| `--verbose` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--config` | No | None | Path to a custom config YAML file |

### Docker Container

#### Build the Container

```bash
dazel run //services/obstacle_correspondence:image_load
```

This builds the OCI image and loads it into the local Docker daemon as `obstacle-correspondence:<version>`.

#### Setup Environment Variables

The Obstacle check uses the [StorageProvider framework](../services/framework/storage_providers/factory.py) to download input files (RDS HQ data, augmented video) from URLs provided in the `/process` request, and to upload output files (result JSONs, visualization videos) back to storage.

Storage is configured through the following environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COSMOS_EVALUATOR_STORAGE_TYPE` | No | `s3` | Storage backend: `s3` or `local` |
| `COSMOS_EVALUATOR_STORAGE_BUCKET` | Yes (S3) | — | S3 bucket name for storing outputs |
| `COSMOS_EVALUATOR_STORAGE_REGION` | Yes (S3) | — | AWS region (also reads `AWS_DEFAULT_REGION`) |
| `COSMOS_EVALUATOR_STORAGE_ACCESS_KEY` | No | — | AWS access key (also reads `AWS_ACCESS_KEY_ID`) |
| `COSMOS_EVALUATOR_STORAGE_SECRET_KEY` | No | — | AWS secret key (also reads `AWS_SECRET_ACCESS_KEY`) |

For convenience, create a `~/.cosmos_evaluator/.env` file with your storage configuration so it is automatically loaded at startup:

```bash
mkdir -p ~/.cosmos_evaluator
cat > ~/.cosmos_evaluator/.env << 'EOF'
COSMOS_EVALUATOR_STORAGE_TYPE=s3
COSMOS_EVALUATOR_STORAGE_BUCKET=my-bucket
COSMOS_EVALUATOR_STORAGE_REGION=us-west-2
COSMOS_EVALUATOR_STORAGE_ACCESS_KEY=your_access_key
COSMOS_EVALUATOR_STORAGE_SECRET_KEY=your_secret_key
EOF
```

If you already have AWS credentials in a `~/.aws/.env` file, the storage settings will also read from `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` as fallbacks if passed via `--env-file` below.

The storage type also determines how input URLs in the `/process` request are resolved. With `s3`, input URLs can be `s3://` URIs, presigned URLs, or any HTTPS URL. With `local`, input paths are treated as local filesystem paths. Output storage follows the same provider — S3 uploads for `s3`, local file copies for `local`.

#### Start the Service

```bash
# Running in local mode with sample data mounted at /data and output at /tmp/output
docker run --gpus all --env-file ~/.cosmos_evaluator/.env \
  -e COSMOS_EVALUATOR_STORAGE_TYPE=local \
  -v $(pwd)/checks/sample_data/cosmos_public:/data \
  -v $(pwd)/output_dir:/tmp/output \
  -p 8000:8000 obstacle-correspondence:1.15.0
```

Verify that the service is running:

```bash
curl http://localhost:8000/health
```

#### Test the Service

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "camera_name": "camera_front_wide_120fov",
    "augmented_video_url": "/data/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4",
    "rds_hq_url": "/data/rds_hq.zip",
    "output_storage_prefix": "/tmp/output"
  }'
```

**Example output:** [01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000.dynamic.object.mp4](../assets/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000.dynamic.object.mp4)

**Request fields:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `clip_id` | Yes | — | Clip identifier |
| `rds_hq_url` | Yes | — | URL to the RDS-HQ dataset archive |
| `augmented_video_url` | Yes | — | URL to the generated video |
| `camera_name` | Yes | — | Camera stream name (e.g., `camera_front_wide_120fov`) |
| `config` | No | None | Full configuration dictionary (replaces the default config entirely). If omitted, the default config from `checks/config.yaml` is used. Use the `/config` endpoint to retrieve the defaults as a starting point. |
| `model_device` | No | `cuda` | Device for model inference: `cuda` or `cpu` |
| `trial_frames` | No | None | Process only the first N frames (useful for debugging) |
| `verbose` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `output_storage_prefix` | No | `{service_name}/{clip_id}/` | Storage prefix for outputs. For cloud storage this is a key prefix; for local storage this is a directory path. |

**Other endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Get default configuration
curl http://localhost:8000/config
```

## Configuration

Configuration is managed through `checks/config.yaml` under the `av.obstacle` key. You can modify the defaults or pass a custom config dict via the API.

### Importance Filter

Controls which objects are included in the evaluation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `allow_all_static_objects` | `true` | Include all static objects (infrastructure) regardless of other filters |
| `distance_threshold_m` | `100` | Maximum distance (meters) from ego vehicle. Objects beyond this are excluded. |
| `relevant_lanes` | `[ego, left, right]` | Only evaluate objects in these lane positions relative to the ego vehicle |
| `skip_oncoming_obstacles` | `false` | If true, exclude vehicles traveling in the opposite direction |

### Overlap Check Methods

Per-class scoring method configuration. Each object class uses one of two methods:

| Method | Description | Best For |
|--------|-------------|----------|
| `ratio` | `overlap_pixels / projected_area_pixels`. Simple area ratio. | Large, well-defined objects (vehicles, infrastructure) |
| `cluster` | Connected component analysis. Selects the largest connected region overlapping the projected cuboid, avoiding over-scoring from background segments. | Smaller objects that may be near similar-class objects (pedestrians, motorcycles) |

Default assignments:

```yaml
overlap_check:
  vehicle:        { method: ratio }
  pedestrian:     { method: cluster }
  motorcycle:     { method: cluster }
  bicycle:        { method: cluster }
  traffic_light:  { method: ratio }
  traffic_sign:   { method: ratio }
  lane_line:      { method: ratio }
  road_boundary:  { method: ratio }
  wait_line:      { method: ratio }
  crosswalk:      { method: ratio }
```

### Hallucination Detector

Detects false positive objects in the generated video that don't correspond to any ground truth:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable hallucination detection |
| `max_cluster_per_frame` | `100` | Maximum clusters to analyze per frame |
| `classes.<class>.min_cluster_area` | varies | Minimum pixel area for a cluster to be considered a hallucination. Smaller values catch more detections but may increase false positives. |

Default minimum cluster areas:

| Class | Min Area (pixels) |
|-------|-------------------|
| vehicle | 5000 |
| motorcycle | 3000 |
| bicycle | 2000 |
| pedestrian | 1000 |
| traffic_light | 1000 |
| traffic_sign | 1000 |

### Visualization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Generate overlay visualization videos showing projected cuboids and segmentation masks |

## Output Format

The check produces two JSON result files per run (one per processor):

- `{clip_id}.dynamic.object.results.json` — Dynamic processor results
- `{clip_id}.static.object.results.json` — Static processor results

If visualization is enabled, corresponding video files are also generated:

- `{clip_id}.dynamic.object.viz.mp4`
- `{clip_id}.static.object.viz.mp4`

### JSON Structure

```json
{
  "processed_frames": 150,
  "total_video_frames": 300,
  "track_ids": [5, 12, 23, 45, 67],
  "processed_frame_ids": [0, 5, 10, 15, ...],

  "mean_score": 0.723,
  "std_score": 0.234,
  "min_score": 0.001,
  "max_score": 0.998,

  "processor_output_labels": [
    "scored",
    "occluded",
    "distance_threshold_m (100m)",
    "relevant_lanes (['ego', 'left', 'right'])"
  ],

  "tracks": [
    {
      "track_id": 5,
      "object_type": "Car",
      "processor_output": [10, 2, 10]
    },
    {
      "track_id": 1001,
      "object_type": "LaneLine",
      "object_type_index": 19,
      "processor_output": [150, 0, 0]
    }
  ],

  "score_matrix": {
    "format": "sparse",
    "shape": [150, 5],
    "data": [
      {"frame_idx": 0, "track_idx": 0, "score": 0.85},
      {"frame_idx": 0, "track_idx": 1, "score": 0.72}
    ]
  },

  "metadata": {
    "matrix_format": "sparse",
    "version": "1.0"
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `processed_frames` | Number of frames where at least one object was evaluated |
| `total_video_frames` | Total frames in the input video |
| `track_ids` | List of unique object track IDs evaluated |
| `processed_frame_ids` | Frame indices that were processed |
| `mean_score` | Mean correspondence score across all valid (object, frame) pairs |
| `std_score` | Standard deviation of scores |
| `min_score` / `max_score` | Range of scores |
| `processor_output_labels` | Labels for the `processor_output` array in each track |
| `tracks[].track_id` | Unique object identifier |
| `tracks[].object_type` | Object class name |
| `tracks[].processor_output` | Per-label counts (e.g., [scored, occluded, filtered_by_distance]) |
| `score_matrix` | Sparse matrix of per-frame, per-track scores |

### Reconstructing the Score Matrix

The score matrix is stored in sparse format for efficiency. To reconstruct it:

```python
import json
import numpy as np

with open("results.json") as f:
    results = json.load(f)

matrix = results["score_matrix"]
scores = np.full(matrix["shape"], np.nan)

for entry in matrix["data"]:
    scores[entry["frame_idx"], entry["track_idx"]] = entry["score"]

# scores[i, j] = correspondence score for frame i, track j
# NaN means the object was not visible/evaluated in that frame
```

## Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| **0.8 – 1.0** | Strong correspondence. The object appears where expected with good shape alignment. |
| **0.3 – 0.8** | Partial correspondence. The object is present but may be shifted, partially occluded, or have shape differences. |
| **0.0 – 0.3** | Poor correspondence. The object may be missing, severely misplaced, or incorrectly rendered. |

The `mean_score` across all objects and frames provides an overall quality metric for the generated video.

## Troubleshooting

### CUDA Out of Memory

The check runs a GPU model. If you encounter OOM errors:
- Reduce video resolution before processing
- Use `--trial N` to process fewer frames

### CPU Mode

If no GPU is available, use `--model_device cpu`. Processing will be significantly slower but functionally identical.

### Debug Logging

Use `--verbose DEBUG` for detailed frame-by-frame processing logs, including object projections, overlap calculations, and filtering decisions.
