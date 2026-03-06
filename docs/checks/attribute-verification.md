# Attribute Verification Check

This checker verifies whether key scene attributes in an augmented video match the expected values configured in `checks/config.yaml`. It answers the question *"Does this video show attributes consistent with the attributes I requested from Cosmos?"*. By using this checker, we may filter videos for specific attributes, discarding videos that do not have these attributes.

## How it works

The attribute verification check evaluates an augmented video against a set of expected scene attributes, such as weather or time of day.

The processing pipeline is:

1. **Question Generation (LLM)**: Generate one multiple-choice verification question for each selected variable. For example:

    ```
    What is the weather like in the video frame?
    A: cloudy
    B: sunny
    C: rainy
    ```

2. **Frame Extraction**: Download the augmented video and extract its first frame.
3. **Question Answering (VLM)**: Ask each generated question against that first frame and compare the VLM answer to the expected answer. For the example question, the VLM might answer `B` if the first video frame appears to be sunny.
4. **Pass/Fail Decision**: Aggregate per-variable checks and mark the clip as passed only if all VLM answers match the expected answers configured by the user in the question input.

The implementation is split across:

- `processor.py`, which orchestrates question generation, VLM verification, and summary scoring
- `question_generator.py`, which generates multiple-choice questions from configured variables and options
- `vlm_verifier.py`, which extracts the first frame and answers questions with the configured VLM
- `run.py`, which provides CLI config loading, execution, and result serialization

## Prerequisites and Compute Requirements

* **LLM and VLM endpoints**: This checker requires both LLM and VLM endpoints to be configured, See the [Model Setup](#model-setup) for more details.
* **S3 Bucket and Credentials**: S3 is used to store the input videos for the attribute verification checker. Set up the attribute verification `.env` file with this information if you haven't already by following the instructions in the [Getting Started Guide](../getting-started.md#attribute-verification-check).
* **Checker Compute Requirements**: The checker container on its own is lightweight and CPU-only. Any linux machine will be able to run this checker.
* **LLM and VLM Compute Requirements**: Any locally deployed LLM or VLM will require a GPU to run. The compute requirements may vary depending on which LLM or VLM is deployed, with specific compute requirements found on the `Model Card` for the model on the [build.nvidia.com](https://build.nvidia.com/) website. The default LLM and VLM for this checker require the following resources:
  - `qwen/qwen3-next-80b-a3b-instruct` (default LLM):
    * NVIDIA Ampere: A100
    * NVIDIA Blackwell: B200, B100
    * NVIDIA Hopper: H100, H200
    * See the [Model Card](https://build.nvidia.com/qwen/qwen3-next-80b-a3b-instruct/modelcard) for more information
  - `qwen/qwen3.5-397b-a17b` (default VLM):
    * NVIDIA Ampere: A100
    * NVIDIA Blackwell: B100, B200, GB200
    * NVIDIA Hopper: H100, H20
    * See the [Model Card](https://build.nvidia.com/qwen/qwen3.5-397b-a17b/modelcard) for more information

  Compute is not required if connecting to an external LLM or VLM.


## Inputs

The check requires:

* **Augmented video**: the Cosmos-augmented video being evaluated
* **Selected variables**: A dictionary with the expected attribute values to verify, such as `weather: sunny`. Each item in the dictionary corresponds to a separate check for the VLM to execute.
* **Variable options**: the allowed answer choices for each variable.

When providing the augmented video, it is simplest to use remote URIs such as `s3://...` for video inputs when testing locally.

For CLI usage, the augmented video path is provided by the cli arguments, and the expected attributes come from configuration file rather than CLI flags.

### Guidance on Constructing Inputs for Good Results

1. Verification is performed on the **first frame only** of the input video, so the checker is best suited to attributes that are visually evident at a glance in the first frame of the video. Make sure that the first frame of the video gives a clear, unobstructed view of the scene and all its attributes. If any of the attributes cannot be identified by a human in the first frame of the video, they will not be identified by the checker either!
2. Attributes must be detectable by the static scene of the frame. The checker will not watch the entire video, so you cannot ask for attributes that describe what happens in the video. For example, `"scenario": "car accident"` is not a valid attribute. Even if a car accident happens in the video, the VLM wont see it since it only looks at the first frame.
3. It is recommended to limit the selected variables to be a targeted list of 2-4 attributes. Using a large number of attributes may cause the checker to fail more often (since more checks are needed to pass), or may overload the VLM or LLM endpoints.
4. Variable options should be easy to distinguish from each other. For example, asking the checker to distinguish between `morning`, `noon`, `midday`, and `afternoon` for a `time of day` attribute will result in many inaccuracies, since it is hard to tell the difference between these times of day based on a video alone.

## Configuration

Configuration lives in `checks/config.yaml` under `metropolis.attribute_verification`.

This file acts as both the default configuration file for the CLI, as well as provides the default configuration values for the service container.

An explanation of each config value is below:
* **enabled**: Enables or disables the checker in shared configuration. Default: `true`.
* **selected_variables**: Mapping of attribute names to the expected value for the clip. Each entry produces one verification question. Default: `weather: sunny`, `time_of_day: morning`.
* **variable_options**: Mapping of attribute names to the full allowed option list that the LLM must use when building answer choices.
* **question_generation.llm.endpoint**: OpenAI-compatible endpoint used for LLM question generation. Default: `https://integrate.api.nvidia.com/v1` (Use this endpoint when connecting to [build.nvidia.com](https://build.nvidia.com/)).
* **question_generation.llm.model**: LLM model used to generate questions. Default: `qwen/qwen3-next-80b-a3b-instruct`.
* **question_generation.parameters.frequency_penalty**: Frequency penalty passed to the LLM request. Default: `0.0`.
* **question_generation.parameters.max_tokens**: Maximum number of output tokens for question generation. Default: `2048`.
* **question_generation.parameters.presence_penalty**: Presence penalty passed to the LLM request. Default: `0.0`.
* **question_generation.parameters.retry**: Number of retry attempts for retryable LLM failures. Default: `1`.
* **question_generation.parameters.stream**: Whether the fallback non-structured LLM request should stream tokens. Default: `true`.
* **question_generation.parameters.temperature**: Sampling temperature for question generation. Default: `0.2`.
* **question_generation.parameters.top_p**: Top-p sampling value for question generation. Default: `0.95`.
* **question_generation.system_prompt**: System prompt instructing the LLM to generate a simple JSON multiple-choice question answerable from a single frame.
* **vlm_verification.vlm.endpoint**: OpenAI-compatible endpoint used for VLM verification. Default: `https://integrate.api.nvidia.com/v1` (Use this endpoint when connecting to [build.nvidia.com](https://build.nvidia.com/)).
* **vlm_verification.vlm.model**: VLM model used to answer the generated questions. Default: `qwen/qwen3.5-397b-a17b`.
* **vlm_verification.parameters.frequency_penalty**: Frequency penalty passed to the VLM request. Default: `0.0`.
* **vlm_verification.parameters.max_tokens**: Maximum number of output tokens for the VLM answer. Default: `10`.
* **vlm_verification.parameters.retry**: Number of retry attempts for retryable VLM failures. Default: `1`.
* **vlm_verification.parameters.stream**: Whether the VLM response should stream. Default: `false`.
* **vlm_verification.parameters.temperature**: Sampling temperature for VLM answering. Default: `0.0`.
* **vlm_verification.parameters.top_p**: Top-p sampling value for VLM answering. Default: `1.0`.
* **vlm_verification.system_prompt**: System prompt instructing the VLM to answer with only a single option letter.

The service request also accepts an optional `verbose` value of `DEBUG`, `INFO`, `WARNING`, or `ERROR`.

## Model Setup

The checker makes two model calls:

1. **An LLM call** to generate one multiple-choice question per configured variable.
2. **A VLM call** to answer those questions from the video's first frame.

The models and endpoints for both the VLM and the LLM are configured in `checks/config.yaml`, and both read the API key from the `BUILD_NVIDIA_API_KEY` environment variable, set in checks/attribute_verification/.env.

By default, the checked-in configuration uses:

* **LLM endpoint**: `https://integrate.api.nvidia.com/v1` ([build.nvidia.com](https://build.nvidia.com/))
* **LLM model**: `qwen/qwen3-next-80b-a3b-instruct`
* **VLM endpoint**: `https://integrate.api.nvidia.com/v1` ([build.nvidia.com](https://build.nvidia.com/))
* **VLM model**: `qwen/qwen3.5-397b-a17b`

See the [Setting up your LLM and VLM API key](../getting-started.md#setting-up-your-llm-and-vlm-api-key) for details on retrieving an API key for the [build.nvidia.com](https://build.nvidia.com/) endpoints.

You can change these defaults in `checks/config.yaml`, or override them per request through the service `config` payload. Any endpoint that is compatible with the OpenAI API can be used in place of the default [build.nvidia.com](https://build.nvidia.com/), including locally-deployed endpoints. The VLM and LLM models may also be replaced with other models, but please keep in mind that the checker performance may degrade if using models other than the provided defaults. The less powerful the model, the less accurate the checker will be.

## Local CLI Testing

### Run the Checker

> Note: Before running this command, ensure that you have already gone through the steps from the [Getting Started](../getting-started.md) and [Customization](../customization.md) guides. Both the environment setup and the build system setup must be complete before this command will work.

Before running the checker locally, ensure that the selected attributes match the ones you want to check for. Please update `checks/config.yaml` to configure these attributes.

From the `cosmos-evaluator` (repo root) directory, run this command:

```bash
dazel run //checks/attribute_verification:run -- \
  --augmented-video-path s3://<bucket_name>/path/to/augmented.mp4
```

The documented CLI supports these options:

- **--augmented-video-path** (required): URI to the augmented video file being evaluated.
- **--clip-id** (optional): Clip identifier for the run. If omitted, the CLI generates a UUID automatically.
- **--config** (optional): Configuration file name to load, without the `.yaml` extension. Defaults to `config`.
- **--config-dir** (optional): Optional path to the directory containing the configuration file.
- **--verbose** (optional): Logging level for the run. Supported values are `DEBUG`, `INFO`, `WARNING`, and `ERROR`. Defaults to `INFO`.
- **--env-file** (optional): Path to the `.env` file to load before processing. Defaults to `checks/attribute_verification/.env`.
- **--output-dir** (optional): Directory where the JSON results file is written. Defaults to `outputs`.

### Interpret the Output

The checker produces:

* An **overall pass/fail** result, where `passed = true` only if every generated check passes.
* A **summary** with `total_checks`, `passed_checks`, and `failed_checks`.
* A console summary with the saved output path, overall result, and one line per checked variable.
* A JSON results file named `{clip_id}.attribute_verification.results.json` under `--output-dir`.

The JSON results will look like this:

```json
{
  "clip_id": "f260297f-ec80-4673-9f5d-d7786da8f0dc",
  "passed": true,
  "summary": {
    "total_checks": 2,
    "passed_checks": 2,
    "failed_checks": 0
  },
  "checks": [
    {
      "variable": "weather",
      "value": "sunny",
      "question": "What is the weather in this video?",
      "options": {
        "A": "sunny",
        "B": "rainy"
      },
      "expected_answer": "A",
      "vlm_answer": "A",
      "passed": true,
      "error": null
    }
  ]
}
```

Higher-quality runs show agreement between the configured expected attributes and the VLM's answers. Failed checks indicate that the model judged one or more expected attributes to be absent or mismatched.

## Building and Running the Service Container

The attribute verification service wraps `AttributeVerificationProcessor` behind a standardized REST API for containerized deployment.

The service exposes these endpoints:

- `GET /health` for health and version status
- `GET /config` to retrieve the default attribute verification configuration
- `POST /process` to process one clip and return per-check verification results

The `POST /process` request body includes:

* **clip_id** (required): This can be set to any arbitrary string, and is used to identify the video in the results
* **augmented_video_path** (required): Remote path for the augmented video
* **config** (optional): Dictionary with any config values which should be altered from the default value. Defaults to `{}`.
* **verbose** (optional): Logging level for the run. Supported values are `DEBUG`, `INFO`, `WARNING`, and `ERROR`. Defaults to `INFO`.

### Build and Run the Container

First, build the container:

```bash
dazel build //services/attribute_verification:image
```

Next, load the built container into the local docker instance:

```bash
dazel run //services/attribute_verification:image_load
```

Finally, run the loaded container:

```bash
docker run --env-file=checks/attribute_verification/.env -p 8080:8080 attribute-verification-checker:1.0.0
```
