# Customizing the Cosmos Evaluator

This section of the documentation describes how to set up the build system, lint, and unit test. After following these steps, you will have everything you need to begin running the code from source.

## Build Setup

Cosmos Evaluator uses **Dazel** — Bazel running inside Docker — for hermetic, reproducible builds. Follow these steps to set up Dazel on your machine.

### Linux Setup

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-evaluator.git
cd cosmos-evaluator

# Source the build environment
source build/env_setup.sh

# Verify the build works
dazel build //...
```

### macOS Setup

macOS requires Colima with specific settings due to filesystem compatibility requirements:

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-evaluator.git
cd cosmos-evaluator

# Install Colima
brew install colima docker

# Start Colima with QEMU and 9p mount (both required)
colima start --cpu 4 --memory 8 --disk 60 --vm-type qemu --mount-type 9p

```

Create a `user.dazelrc` file at the repository root for macOS compatibility:

```python
# user.dazelrc — macOS overrides
import subprocess

# Fix group detection (getent not available on macOS)
DAZEL_DOCKER_RUN_FLAGS += ["-e", "GROUP_ID=1000"]

# Disable GPU (not available on macOS)
DAZEL_DOCKER_RUN_FLAGS = [f for f in DAZEL_DOCKER_RUN_FLAGS if '--gpus' not in f]
```

Then build:

```bash
source build/env_setup.sh
dazel build //...    # ~5 min first build, ~4 sec cached
```

> **Note:** GPU-dependent features (Obstacle check's SegFormer/CWIP models) require a Linux machine with an NVIDIA GPU. macOS can run the all other checkers and non-GPU code paths.


## Linting and Formatting

If you are making a local change and would like to run the linter:

```bash
# Auto-fix lint and formatting issues
dazel fix

# Check for violations without fixing
dazel lint
```

## Running Unit Tests

```bash
# All tests
dazel test //...

# Specific module tests
dazel test //checks/obstacle:all
dazel test //checks/vlm:all
dazel test //checks/hallucination:all
dazel test //checks/attribute_verification:all
dazel test //services/obstacle_correspondence:all
dazel test //services/vlm:all
dazel test //services/hallucination:all
dazel test //services/attribute_verification:all
```

## Project Structure

```bash
cosmos-evaluator/
├── build/                          # Build infrastructure
├── checks/                         # Evaluation modules
│   ├── attribute_verification/     # Attribute verification check
│   ├── config.yaml                 # Central configuration
│   ├── hallucination/              # Hallucination check
│   ├── obstacle/                   # Object correspondence check
│   ├── utils/                      # Shared utilities (video I/O, data loading, transforms)
│   └── vlm/                        # Vision-Language Model checks
├── docs/                           # This documentation
├── services/                       # REST API microservices
│   ├── attribute_verification/     # Attribute verification service
│   ├── framework/                  # Base classes and protocols
│   ├── hallucination/              # Hallucination service
│   ├── obstacle_correspondence/    # Obstacle check service
│   └── vlm/                        # VLM check service
├── third_party                     # External dependencies
```

## Next Steps

Dive into one of the existing checkers, or create your own!

- [Hallucination Check Guide](checks/hallucination.md) — Details on the hallucination checker
- [Obstacle Check Guide](checks/obstacle.md) — Details on the obstacle checker
- [VLM Preset Check Guide](checks/vlm-preset.md) — Details on the VLM preset checker
- [Attribute Verification Check Guide](checks/attribute-verification.md) — Details on the attribute verification checker
- [Custom Checker Tutorial](tutorials/custom-checker.md) — Build your own check from scratch
- [Service Framework Guide](services/framework.md) — Building and deploying a microservice around your custom check
