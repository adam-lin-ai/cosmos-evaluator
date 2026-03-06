# Deployment

This guide covers how to build, tag, push, and run Cosmos Evaluator service containers for production environments. Each checker runs as an independent container exposing a REST API. The default port is configurable via the `UVICORN_PORT` environment variable.

## Services Overview

| Service | Image Tag | GPU Required | Default Port |
|---------|-----------|:------------:|:------------:|
| Obstacle Correspondence | `obstacle-correspondence:1.15.0` | Yes | 8000 |
| VLM Preset | `vlm:1.15.0` | No | 8000 |
| Hallucination | `hallucination-checker:1.0.0` | No | 8080 |
| Attribute Verification | `attribute-verification-checker:1.0.0` | No | 8080 |

## Building Container Images

Set up the build environment, then use `dazel run` to build and load each image into the local Docker daemon:

```bash
source build/env_setup.sh

dazel run //services/obstacle_correspondence:image_load
dazel run //services/vlm:image_load
dazel run //services/hallucination:image_load
dazel run //services/attribute_verification:image_load
```

For detailed build and run instructions per service, see the individual checker documentation:

| Service | Documentation |
|---------|---------------|
| Obstacle Correspondence | [Obstacle Check — Docker Container](checks/obstacle.md#docker-container) |
| VLM Preset | [VLM Preset Check — Docker Container](checks/vlm-preset.md#docker-container) |
| Hallucination | [Hallucination Check](checks/hallucination.md) |
| Attribute Verification | [Attribute Verification Check](checks/attribute-verification.md) |

## Tagging and Pushing to a Registry

After building and loading the images locally, tag and push them to any OCI-compatible container registry:

```bash
# Tag the images for your registry
docker tag obstacle-correspondence:1.15.0 your-registry.example.com/cosmos-evaluator/obstacle-correspondence:1.15.0
docker tag vlm:1.15.0 your-registry.example.com/cosmos-evaluator/vlm:1.15.0
docker tag hallucination-checker:1.0.0 your-registry.example.com/cosmos-evaluator/hallucination-checker:1.0.0
docker tag attribute-verification-checker:1.0.0 your-registry.example.com/cosmos-evaluator/attribute-verification-checker:1.0.0

# Push to the registry
docker push your-registry.example.com/cosmos-evaluator/obstacle-correspondence:1.15.0
docker push your-registry.example.com/cosmos-evaluator/vlm:1.15.0
docker push your-registry.example.com/cosmos-evaluator/hallucination-checker:1.0.0
docker push your-registry.example.com/cosmos-evaluator/attribute-verification-checker:1.0.0
```

Replace `your-registry.example.com/cosmos-evaluator` with your actual registry path (e.g., an ECR, GCR, ACR, or NVCR endpoint).

## Environment Configuration for Production

### Port Configuration

The `UVICORN_PORT` environment variable controls the port the service listens on inside the container. This is useful when running multiple services on the same host or behind a load balancer:

```bash
docker run -e UVICORN_PORT=9000 -p 9000:9000 vlm:1.15.0
```

If not set, the service uses its default port (8000 or 8080 depending on the service).

### Storage Provider

For staging or production, use S3 as the storage backend. Add the following environment variables to a `~/.cosmos_evaluator/.env` file:

| Variable | Description |
|----------|-------------|
| `COSMOS_EVALUATOR_ENV` | Set to `production` or `staging` |
| `COSMOS_EVALUATOR_STORAGE_TYPE` | Storage backend: `s3` (default) or `local` |
| `COSMOS_EVALUATOR_STORAGE_BUCKET` | S3 bucket name |
| `COSMOS_EVALUATOR_STORAGE_REGION` | AWS region |
| `COSMOS_EVALUATOR_STORAGE_ACCESS_KEY` | AWS access key |
| `COSMOS_EVALUATOR_STORAGE_SECRET_KEY` | AWS secret key |

> **Note:** `COSMOS_EVALUATOR_STORAGE_TYPE=local` is not recommended for cloud deployments.

### API Keys

The VLM Preset and Attribute Verification checks require API keys for their model endpoints:

| Variable | Used By | Description |
|----------|---------|-------------|
| `BUILD_NVIDIA_API_KEY` | VLM Preset, Attribute Verification | API key for LLM/VLM endpoints ([build.nvidia.com](https://build.nvidia.com/)) |

## Running in Production

### Obstacle Correspondence (GPU required)

Set `OBJECTS_PROCESS_CONCURRENCY_LIMIT` to control the maximum number of concurrent `/process` requests (default: 3) based on the load your cloud machine can handle concurrently. Requests beyond this limit are rejected immediately with a 503 status.

```bash
docker run \
  --gpus all \
  --env-file ~/.cosmos_evaluator/.env \
  -e OBJECTS_PROCESS_CONCURRENCY_LIMIT=1 \
  -p 8000:8000 \
  obstacle-correspondence:1.15.0
```

### VLM Preset

```bash
docker run \
  --env-file ~/.cosmos_evaluator/.env \
  -p 8000:8000 \
  vlm:1.15.0
```

### Hallucination

```bash
docker run \
  -e MULTISTORAGE_CONFIG='{"profiles":{ ... }}' \
  -p 8080:8080 \
  hallucination-checker:1.0.0
```

> **Note:** For details on configuring `MULTISTORAGE_CONFIG`, see the [Multistorage Configuration Example](https://github.com/nvidia-cosmos/cosmos-evaluator/blob/main/docs/getting-started.md#multistorage-configuration-example).

### Attribute Verification

```bash
docker run \
  -e BUILD_NVIDIA_API_KEY=your_api_key \
  -e MULTISTORAGE_CONFIG='{"profiles":{ ... }}' \
  -p 8080:8080 \
  attribute-verification-checker:1.0.0
```

> **Note:** For details on configuring `MULTISTORAGE_CONFIG`, see the [Multistorage Configuration Example](https://github.com/nvidia-cosmos/cosmos-evaluator/blob/main/docs/getting-started.md#multistorage-configuration-example).

### Verifying the Service

All services expose a `/health` endpoint for liveness and readiness checks:

```bash
curl http://localhost:8000/health
```

Each service also provides a Swagger UI at `/docs` for browsing the API documentation and testing endpoints from a browser:

```
http://localhost:8000/docs
```
